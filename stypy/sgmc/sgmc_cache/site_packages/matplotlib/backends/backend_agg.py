
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: An agg http://antigrain.com/ backend
3: 
4: Features that are implemented
5: 
6:  * capstyles and join styles
7:  * dashes
8:  * linewidth
9:  * lines, rectangles, ellipses
10:  * clipping to a rectangle
11:  * output to RGBA and PNG, optionally JPEG and TIFF
12:  * alpha blending
13:  * DPI scaling properly - everything scales properly (dashes, linewidths, etc)
14:  * draw polygon
15:  * freetype2 w/ ft2font
16: 
17: TODO:
18: 
19:   * integrate screen dpi w/ ppi and text
20: 
21: '''
22: from __future__ import (absolute_import, division, print_function,
23:                         unicode_literals)
24: 
25: import six
26: 
27: import threading
28: import numpy as np
29: from collections import OrderedDict
30: from math import radians, cos, sin
31: from matplotlib import verbose, rcParams, __version__
32: from matplotlib.backend_bases import (
33:     _Backend, FigureCanvasBase, FigureManagerBase, RendererBase, cursors)
34: from matplotlib.cbook import maxdict
35: from matplotlib.figure import Figure
36: from matplotlib.font_manager import findfont, get_font
37: from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
38:                                 LOAD_DEFAULT, LOAD_NO_AUTOHINT)
39: from matplotlib.mathtext import MathTextParser
40: from matplotlib.path import Path
41: from matplotlib.transforms import Bbox, BboxBase
42: from matplotlib import colors as mcolors
43: 
44: from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
45: from matplotlib import _png
46: 
47: try:
48:     from PIL import Image
49:     _has_pil = True
50: except ImportError:
51:     _has_pil = False
52: 
53: backend_version = 'v2.2'
54: 
55: def get_hinting_flag():
56:     mapping = {
57:         True: LOAD_FORCE_AUTOHINT,
58:         False: LOAD_NO_HINTING,
59:         'either': LOAD_DEFAULT,
60:         'native': LOAD_NO_AUTOHINT,
61:         'auto': LOAD_FORCE_AUTOHINT,
62:         'none': LOAD_NO_HINTING
63:         }
64:     return mapping[rcParams['text.hinting']]
65: 
66: 
67: class RendererAgg(RendererBase):
68:     '''
69:     The renderer handles all the drawing primitives using a graphics
70:     context instance that controls the colors/styles
71:     '''
72:     debug=1
73: 
74:     # we want to cache the fonts at the class level so that when
75:     # multiple figures are created we can reuse them.  This helps with
76:     # a bug on windows where the creation of too many figures leads to
77:     # too many open file handles.  However, storing them at the class
78:     # level is not thread safe.  The solution here is to let the
79:     # FigureCanvas acquire a lock on the fontd at the start of the
80:     # draw, and release it when it is done.  This allows multiple
81:     # renderers to share the cached fonts, but only one figure can
82:     # draw at time and so the font cache is used by only one
83:     # renderer at a time
84: 
85:     lock = threading.RLock()
86:     def __init__(self, width, height, dpi):
87:         RendererBase.__init__(self)
88: 
89:         self.dpi = dpi
90:         self.width = width
91:         self.height = height
92:         self._renderer = _RendererAgg(int(width), int(height), dpi, debug=False)
93:         self._filter_renderers = []
94: 
95:         self._update_methods()
96:         self.mathtext_parser = MathTextParser('Agg')
97: 
98:         self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)
99: 
100:     def __getstate__(self):
101:         # We only want to preserve the init keywords of the Renderer.
102:         # Anything else can be re-created.
103:         return {'width': self.width, 'height': self.height, 'dpi': self.dpi}
104: 
105:     def __setstate__(self, state):
106:         self.__init__(state['width'], state['height'], state['dpi'])
107: 
108:     def _get_hinting_flag(self):
109:         if rcParams['text.hinting']:
110:             return LOAD_FORCE_AUTOHINT
111:         else:
112:             return LOAD_NO_HINTING
113: 
114:     # for filtering to work with rasterization, methods needs to be wrapped.
115:     # maybe there is better way to do it.
116:     def draw_markers(self, *kl, **kw):
117:         return self._renderer.draw_markers(*kl, **kw)
118: 
119:     def draw_path_collection(self, *kl, **kw):
120:         return self._renderer.draw_path_collection(*kl, **kw)
121: 
122:     def _update_methods(self):
123:         self.draw_quad_mesh = self._renderer.draw_quad_mesh
124:         self.draw_gouraud_triangle = self._renderer.draw_gouraud_triangle
125:         self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
126:         self.draw_image = self._renderer.draw_image
127:         self.copy_from_bbox = self._renderer.copy_from_bbox
128:         self.get_content_extents = self._renderer.get_content_extents
129: 
130:     def tostring_rgba_minimized(self):
131:         extents = self.get_content_extents()
132:         bbox = [[extents[0], self.height - (extents[1] + extents[3])],
133:                 [extents[0] + extents[2], self.height - extents[1]]]
134:         region = self.copy_from_bbox(bbox)
135:         return np.array(region), extents
136: 
137:     def draw_path(self, gc, path, transform, rgbFace=None):
138:         '''
139:         Draw the path
140:         '''
141:         nmax = rcParams['agg.path.chunksize'] # here at least for testing
142:         npts = path.vertices.shape[0]
143: 
144:         if (nmax > 100 and npts > nmax and path.should_simplify and
145:                 rgbFace is None and gc.get_hatch() is None):
146:             nch = np.ceil(npts / float(nmax))
147:             chsize = int(np.ceil(npts / nch))
148:             i0 = np.arange(0, npts, chsize)
149:             i1 = np.zeros_like(i0)
150:             i1[:-1] = i0[1:] - 1
151:             i1[-1] = npts
152:             for ii0, ii1 in zip(i0, i1):
153:                 v = path.vertices[ii0:ii1, :]
154:                 c = path.codes
155:                 if c is not None:
156:                     c = c[ii0:ii1]
157:                     c[0] = Path.MOVETO  # move to end of last chunk
158:                 p = Path(v, c)
159:                 try:
160:                     self._renderer.draw_path(gc, p, transform, rgbFace)
161:                 except OverflowError:
162:                     raise OverflowError("Exceeded cell block limit (set 'agg.path.chunksize' rcparam)")
163:         else:
164:             try:
165:                 self._renderer.draw_path(gc, path, transform, rgbFace)
166:             except OverflowError:
167:                 raise OverflowError("Exceeded cell block limit (set 'agg.path.chunksize' rcparam)")
168: 
169: 
170:     def draw_mathtext(self, gc, x, y, s, prop, angle):
171:         '''
172:         Draw the math text using matplotlib.mathtext
173:         '''
174:         ox, oy, width, height, descent, font_image, used_characters = \
175:             self.mathtext_parser.parse(s, self.dpi, prop)
176: 
177:         xd = descent * sin(radians(angle))
178:         yd = descent * cos(radians(angle))
179:         x = np.round(x + ox + xd)
180:         y = np.round(y - oy + yd)
181:         self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)
182: 
183:     def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
184:         '''
185:         Render the text
186:         '''
187:         if ismath:
188:             return self.draw_mathtext(gc, x, y, s, prop, angle)
189: 
190:         flags = get_hinting_flag()
191:         font = self._get_agg_font(prop)
192: 
193:         if font is None:
194:             return None
195:         if len(s) == 1 and ord(s) > 127:
196:             font.load_char(ord(s), flags=flags)
197:         else:
198:             # We pass '0' for angle here, since it will be rotated (in raster
199:             # space) in the following call to draw_text_image).
200:             font.set_text(s, 0, flags=flags)
201:         font.draw_glyphs_to_bitmap(antialiased=rcParams['text.antialiased'])
202:         d = font.get_descent() / 64.0
203:         # The descent needs to be adjusted for the angle
204:         xo, yo = font.get_bitmap_offset()
205:         xo /= 64.0
206:         yo /= 64.0
207:         xd = -d * sin(radians(angle))
208:         yd = d * cos(radians(angle))
209: 
210:         #print x, y, int(x), int(y), s
211:         self._renderer.draw_text_image(
212:             font, np.round(x - xd + xo), np.round(y + yd + yo) + 1, angle, gc)
213: 
214:     def get_text_width_height_descent(self, s, prop, ismath):
215:         '''
216:         Get the width, height, and descent (offset from the bottom
217:         to the baseline), in display coords, of the string *s* with
218:         :class:`~matplotlib.font_manager.FontProperties` *prop*
219:         '''
220:         if rcParams['text.usetex']:
221:             # todo: handle props
222:             size = prop.get_size_in_points()
223:             texmanager = self.get_texmanager()
224:             fontsize = prop.get_size_in_points()
225:             w, h, d = texmanager.get_text_width_height_descent(
226:                 s, fontsize, renderer=self)
227:             return w, h, d
228: 
229:         if ismath:
230:             ox, oy, width, height, descent, fonts, used_characters = \
231:                 self.mathtext_parser.parse(s, self.dpi, prop)
232:             return width, height, descent
233: 
234:         flags = get_hinting_flag()
235:         font = self._get_agg_font(prop)
236:         font.set_text(s, 0.0, flags=flags)  # the width and height of unrotated string
237:         w, h = font.get_width_height()
238:         d = font.get_descent()
239:         w /= 64.0  # convert from subpixels
240:         h /= 64.0
241:         d /= 64.0
242:         return w, h, d
243: 
244:     def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!', mtext=None):
245:         # todo, handle props, angle, origins
246:         size = prop.get_size_in_points()
247: 
248:         texmanager = self.get_texmanager()
249: 
250:         Z = texmanager.get_grey(s, size, self.dpi)
251:         Z = np.array(Z * 255.0, np.uint8)
252: 
253:         w, h, d = self.get_text_width_height_descent(s, prop, ismath)
254:         xd = d * sin(radians(angle))
255:         yd = d * cos(radians(angle))
256:         x = np.round(x + xd)
257:         y = np.round(y + yd)
258: 
259:         self._renderer.draw_text_image(Z, x, y, angle, gc)
260: 
261:     def get_canvas_width_height(self):
262:         'return the canvas width and height in display coords'
263:         return self.width, self.height
264: 
265:     def _get_agg_font(self, prop):
266:         '''
267:         Get the font for text instance t, cacheing for efficiency
268:         '''
269:         fname = findfont(prop)
270:         font = get_font(
271:             fname,
272:             hinting_factor=rcParams['text.hinting_factor'])
273: 
274:         font.clear()
275:         size = prop.get_size_in_points()
276:         font.set_size(size, self.dpi)
277: 
278:         return font
279: 
280:     def points_to_pixels(self, points):
281:         '''
282:         convert point measures to pixes using dpi and the pixels per
283:         inch of the display
284:         '''
285:         return points*self.dpi/72.0
286: 
287:     def tostring_rgb(self):
288:         return self._renderer.tostring_rgb()
289: 
290:     def tostring_argb(self):
291:         return self._renderer.tostring_argb()
292: 
293:     def buffer_rgba(self):
294:         return self._renderer.buffer_rgba()
295: 
296:     def clear(self):
297:         self._renderer.clear()
298: 
299:     def option_image_nocomposite(self):
300:         # It is generally faster to composite each image directly to
301:         # the Figure, and there's no file size benefit to compositing
302:         # with the Agg backend
303:         return True
304: 
305:     def option_scale_image(self):
306:         '''
307:         agg backend doesn't support arbitrary scaling of image.
308:         '''
309:         return False
310: 
311:     def restore_region(self, region, bbox=None, xy=None):
312:         '''
313:         Restore the saved region. If bbox (instance of BboxBase, or
314:         its extents) is given, only the region specified by the bbox
315:         will be restored. *xy* (a tuple of two floasts) optionally
316:         specifies the new position (the LLC of the original region,
317:         not the LLC of the bbox) where the region will be restored.
318: 
319:         >>> region = renderer.copy_from_bbox()
320:         >>> x1, y1, x2, y2 = region.get_extents()
321:         >>> renderer.restore_region(region, bbox=(x1+dx, y1, x2, y2),
322:         ...                         xy=(x1-dx, y1))
323: 
324:         '''
325:         if bbox is not None or xy is not None:
326:             if bbox is None:
327:                 x1, y1, x2, y2 = region.get_extents()
328:             elif isinstance(bbox, BboxBase):
329:                 x1, y1, x2, y2 = bbox.extents
330:             else:
331:                 x1, y1, x2, y2 = bbox
332: 
333:             if xy is None:
334:                 ox, oy = x1, y1
335:             else:
336:                 ox, oy = xy
337: 
338:             # The incoming data is float, but the _renderer type-checking wants
339:             # to see integers.
340:             self._renderer.restore_region(region, int(x1), int(y1),
341:                                           int(x2), int(y2), int(ox), int(oy))
342: 
343:         else:
344:             self._renderer.restore_region(region)
345: 
346:     def start_filter(self):
347:         '''
348:         Start filtering. It simply create a new canvas (the old one is saved).
349:         '''
350:         self._filter_renderers.append(self._renderer)
351:         self._renderer = _RendererAgg(int(self.width), int(self.height),
352:                                       self.dpi)
353:         self._update_methods()
354: 
355:     def stop_filter(self, post_processing):
356:         '''
357:         Save the plot in the current canvas as a image and apply
358:         the *post_processing* function.
359: 
360:            def post_processing(image, dpi):
361:              # ny, nx, depth = image.shape
362:              # image (numpy array) has RGBA channels and has a depth of 4.
363:              ...
364:              # create a new_image (numpy array of 4 channels, size can be
365:              # different). The resulting image may have offsets from
366:              # lower-left corner of the original image
367:              return new_image, offset_x, offset_y
368: 
369:         The saved renderer is restored and the returned image from
370:         post_processing is plotted (using draw_image) on it.
371:         '''
372: 
373:         # WARNING.
374:         # For agg_filter to work, the rendere's method need
375:         # to overridden in the class. See draw_markers, and draw_path_collections
376: 
377:         width, height = int(self.width), int(self.height)
378: 
379:         buffer, bounds = self.tostring_rgba_minimized()
380: 
381:         l, b, w, h = bounds
382: 
383:         self._renderer = self._filter_renderers.pop()
384:         self._update_methods()
385: 
386:         if w > 0 and h > 0:
387:             img = np.fromstring(buffer, np.uint8)
388:             img, ox, oy = post_processing(img.reshape((h, w, 4)) / 255.,
389:                                           self.dpi)
390:             gc = self.new_gc()
391:             if img.dtype.kind == 'f':
392:                 img = np.asarray(img * 255., np.uint8)
393:             img = img[::-1]
394:             self._renderer.draw_image(
395:                 gc, l + ox, height - b - h + oy, img)
396: 
397: 
398: class FigureCanvasAgg(FigureCanvasBase):
399:     '''
400:     The canvas the figure renders into.  Calls the draw and print fig
401:     methods, creates the renderers, etc...
402: 
403:     Attributes
404:     ----------
405:     figure : `matplotlib.figure.Figure`
406:         A high-level Figure instance
407: 
408:     '''
409: 
410:     def copy_from_bbox(self, bbox):
411:         renderer = self.get_renderer()
412:         return renderer.copy_from_bbox(bbox)
413: 
414:     def restore_region(self, region, bbox=None, xy=None):
415:         renderer = self.get_renderer()
416:         return renderer.restore_region(region, bbox, xy)
417: 
418:     def draw(self):
419:         '''
420:         Draw the figure using the renderer
421:         '''
422:         self.renderer = self.get_renderer(cleared=True)
423:         # acquire a lock on the shared font cache
424:         RendererAgg.lock.acquire()
425: 
426:         toolbar = self.toolbar
427:         try:
428:             if toolbar:
429:                 toolbar.set_cursor(cursors.WAIT)
430:             self.figure.draw(self.renderer)
431:         finally:
432:             if toolbar:
433:                 toolbar.set_cursor(toolbar._lastCursor)
434:             RendererAgg.lock.release()
435: 
436:     def get_renderer(self, cleared=False):
437:         l, b, w, h = self.figure.bbox.bounds
438:         key = w, h, self.figure.dpi
439:         try: self._lastKey, self.renderer
440:         except AttributeError: need_new_renderer = True
441:         else:  need_new_renderer = (self._lastKey != key)
442: 
443:         if need_new_renderer:
444:             self.renderer = RendererAgg(w, h, self.figure.dpi)
445:             self._lastKey = key
446:         elif cleared:
447:             self.renderer.clear()
448:         return self.renderer
449: 
450:     def tostring_rgb(self):
451:         '''Get the image as an RGB byte string
452: 
453:         `draw` must be called at least once before this function will work and
454:         to update the renderer for any subsequent changes to the Figure.
455: 
456:         Returns
457:         -------
458:         bytes
459:         '''
460:         return self.renderer.tostring_rgb()
461: 
462:     def tostring_argb(self):
463:         '''Get the image as an ARGB byte string
464: 
465:         `draw` must be called at least once before this function will work and
466:         to update the renderer for any subsequent changes to the Figure.
467: 
468:         Returns
469:         -------
470:         bytes
471: 
472:         '''
473:         return self.renderer.tostring_argb()
474: 
475:     def buffer_rgba(self):
476:         '''Get the image as an RGBA byte string
477: 
478:         `draw` must be called at least once before this function will work and
479:         to update the renderer for any subsequent changes to the Figure.
480: 
481:         Returns
482:         -------
483:         bytes
484:         '''
485:         return self.renderer.buffer_rgba()
486: 
487:     def print_raw(self, filename_or_obj, *args, **kwargs):
488:         FigureCanvasAgg.draw(self)
489:         renderer = self.get_renderer()
490:         original_dpi = renderer.dpi
491:         renderer.dpi = self.figure.dpi
492:         if isinstance(filename_or_obj, six.string_types):
493:             fileobj = open(filename_or_obj, 'wb')
494:             close = True
495:         else:
496:             fileobj = filename_or_obj
497:             close = False
498:         try:
499:             fileobj.write(renderer._renderer.buffer_rgba())
500:         finally:
501:             if close:
502:                 fileobj.close()
503:             renderer.dpi = original_dpi
504:     print_rgba = print_raw
505: 
506:     def print_png(self, filename_or_obj, *args, **kwargs):
507:         FigureCanvasAgg.draw(self)
508:         renderer = self.get_renderer()
509:         original_dpi = renderer.dpi
510:         renderer.dpi = self.figure.dpi
511:         if isinstance(filename_or_obj, six.string_types):
512:             filename_or_obj = open(filename_or_obj, 'wb')
513:             close = True
514:         else:
515:             close = False
516: 
517:         version_str = 'matplotlib version ' + __version__ + \
518:             ', http://matplotlib.org/'
519:         metadata = OrderedDict({'Software': version_str})
520:         user_metadata = kwargs.pop("metadata", None)
521:         if user_metadata is not None:
522:             metadata.update(user_metadata)
523: 
524:         try:
525:             _png.write_png(renderer._renderer, filename_or_obj,
526:                            self.figure.dpi, metadata=metadata)
527:         finally:
528:             if close:
529:                 filename_or_obj.close()
530:             renderer.dpi = original_dpi
531: 
532:     def print_to_buffer(self):
533:         FigureCanvasAgg.draw(self)
534:         renderer = self.get_renderer()
535:         original_dpi = renderer.dpi
536:         renderer.dpi = self.figure.dpi
537:         try:
538:             result = (renderer._renderer.buffer_rgba(),
539:                       (int(renderer.width), int(renderer.height)))
540:         finally:
541:             renderer.dpi = original_dpi
542:         return result
543: 
544:     if _has_pil:
545:         # add JPEG support
546:         def print_jpg(self, filename_or_obj, *args, **kwargs):
547:             '''
548:             Other Parameters
549:             ----------------
550:             quality : int
551:                 The image quality, on a scale from 1 (worst) to
552:                 95 (best). The default is 95, if not given in the
553:                 matplotlibrc file in the savefig.jpeg_quality parameter.
554:                 Values above 95 should be avoided; 100 completely
555:                 disables the JPEG quantization stage.
556: 
557:             optimize : bool
558:                 If present, indicates that the encoder should
559:                 make an extra pass over the image in order to select
560:                 optimal encoder settings.
561: 
562:             progressive : bool
563:                 If present, indicates that this image
564:                 should be stored as a progressive JPEG file.
565:             '''
566:             buf, size = self.print_to_buffer()
567:             if kwargs.pop("dryrun", False):
568:                 return
569:             # The image is "pasted" onto a white background image to safely
570:             # handle any transparency
571:             image = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
572:             rgba = mcolors.to_rgba(rcParams['savefig.facecolor'])
573:             color = tuple([int(x * 255.0) for x in rgba[:3]])
574:             background = Image.new('RGB', size, color)
575:             background.paste(image, image)
576:             options = {k: kwargs[k]
577:                        for k in ['quality', 'optimize', 'progressive', 'dpi']
578:                        if k in kwargs}
579:             options.setdefault('quality', rcParams['savefig.jpeg_quality'])
580:             if 'dpi' in options:
581:                 # Set the same dpi in both x and y directions
582:                 options['dpi'] = (options['dpi'], options['dpi'])
583: 
584:             return background.save(filename_or_obj, format='jpeg', **options)
585:         print_jpeg = print_jpg
586: 
587:         # add TIFF support
588:         def print_tif(self, filename_or_obj, *args, **kwargs):
589:             buf, size = self.print_to_buffer()
590:             if kwargs.pop("dryrun", False):
591:                 return
592:             image = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
593:             dpi = (self.figure.dpi, self.figure.dpi)
594:             return image.save(filename_or_obj, format='tiff',
595:                               dpi=dpi)
596:         print_tiff = print_tif
597: 
598: 
599: @_Backend.export
600: class _BackendAgg(_Backend):
601:     FigureCanvas = FigureCanvasAgg
602:     FigureManager = FigureManagerBase
603: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_217836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'unicode', u'\nAn agg http://antigrain.com/ backend\n\nFeatures that are implemented\n\n * capstyles and join styles\n * dashes\n * linewidth\n * lines, rectangles, ellipses\n * clipping to a rectangle\n * output to RGBA and PNG, optionally JPEG and TIFF\n * alpha blending\n * DPI scaling properly - everything scales properly (dashes, linewidths, etc)\n * draw polygon\n * freetype2 w/ ft2font\n\nTODO:\n\n  * integrate screen dpi w/ ppi and text\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import six' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217837 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six')

if (type(import_217837) is not StypyTypeError):

    if (import_217837 != 'pyd_module'):
        __import__(import_217837)
        sys_modules_217838 = sys.modules[import_217837]
        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', sys_modules_217838.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', import_217837)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import threading' statement (line 27)
import threading

import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'threading', threading, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import numpy' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217839 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy')

if (type(import_217839) is not StypyTypeError):

    if (import_217839 != 'pyd_module'):
        __import__(import_217839)
        sys_modules_217840 = sys.modules[import_217839]
        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'np', sys_modules_217840.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy', import_217839)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from collections import OrderedDict' statement (line 29)
try:
    from collections import OrderedDict

except:
    OrderedDict = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'collections', None, module_type_store, ['OrderedDict'], [OrderedDict])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from math import radians, cos, sin' statement (line 30)
try:
    from math import radians, cos, sin

except:
    radians = UndefinedType
    cos = UndefinedType
    sin = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'math', None, module_type_store, ['radians', 'cos', 'sin'], [radians, cos, sin])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from matplotlib import verbose, rcParams, __version__' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217841 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib')

if (type(import_217841) is not StypyTypeError):

    if (import_217841 != 'pyd_module'):
        __import__(import_217841)
        sys_modules_217842 = sys.modules[import_217841]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib', sys_modules_217842.module_type_store, module_type_store, ['verbose', 'rcParams', '__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_217842, sys_modules_217842.module_type_store, module_type_store)
    else:
        from matplotlib import verbose, rcParams, __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib', None, module_type_store, ['verbose', 'rcParams', '__version__'], [verbose, rcParams, __version__])

else:
    # Assigning a type to the variable 'matplotlib' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib', import_217841)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, RendererBase, cursors' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217843 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.backend_bases')

if (type(import_217843) is not StypyTypeError):

    if (import_217843 != 'pyd_module'):
        __import__(import_217843)
        sys_modules_217844 = sys.modules[import_217843]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.backend_bases', sys_modules_217844.module_type_store, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'RendererBase', 'cursors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_217844, sys_modules_217844.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, RendererBase, cursors

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'RendererBase', 'cursors'], [_Backend, FigureCanvasBase, FigureManagerBase, RendererBase, cursors])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.backend_bases', import_217843)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from matplotlib.cbook import maxdict' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217845 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.cbook')

if (type(import_217845) is not StypyTypeError):

    if (import_217845 != 'pyd_module'):
        __import__(import_217845)
        sys_modules_217846 = sys.modules[import_217845]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.cbook', sys_modules_217846.module_type_store, module_type_store, ['maxdict'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_217846, sys_modules_217846.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import maxdict

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.cbook', None, module_type_store, ['maxdict'], [maxdict])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.cbook', import_217845)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from matplotlib.figure import Figure' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217847 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.figure')

if (type(import_217847) is not StypyTypeError):

    if (import_217847 != 'pyd_module'):
        __import__(import_217847)
        sys_modules_217848 = sys.modules[import_217847]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.figure', sys_modules_217848.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_217848, sys_modules_217848.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.figure', import_217847)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from matplotlib.font_manager import findfont, get_font' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217849 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.font_manager')

if (type(import_217849) is not StypyTypeError):

    if (import_217849 != 'pyd_module'):
        __import__(import_217849)
        sys_modules_217850 = sys.modules[import_217849]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.font_manager', sys_modules_217850.module_type_store, module_type_store, ['findfont', 'get_font'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_217850, sys_modules_217850.module_type_store, module_type_store)
    else:
        from matplotlib.font_manager import findfont, get_font

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.font_manager', None, module_type_store, ['findfont', 'get_font'], [findfont, get_font])

else:
    # Assigning a type to the variable 'matplotlib.font_manager' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.font_manager', import_217849)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from matplotlib.ft2font import LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING, LOAD_DEFAULT, LOAD_NO_AUTOHINT' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217851 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.ft2font')

if (type(import_217851) is not StypyTypeError):

    if (import_217851 != 'pyd_module'):
        __import__(import_217851)
        sys_modules_217852 = sys.modules[import_217851]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.ft2font', sys_modules_217852.module_type_store, module_type_store, ['LOAD_FORCE_AUTOHINT', 'LOAD_NO_HINTING', 'LOAD_DEFAULT', 'LOAD_NO_AUTOHINT'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_217852, sys_modules_217852.module_type_store, module_type_store)
    else:
        from matplotlib.ft2font import LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING, LOAD_DEFAULT, LOAD_NO_AUTOHINT

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.ft2font', None, module_type_store, ['LOAD_FORCE_AUTOHINT', 'LOAD_NO_HINTING', 'LOAD_DEFAULT', 'LOAD_NO_AUTOHINT'], [LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING, LOAD_DEFAULT, LOAD_NO_AUTOHINT])

else:
    # Assigning a type to the variable 'matplotlib.ft2font' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.ft2font', import_217851)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from matplotlib.mathtext import MathTextParser' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.mathtext')

if (type(import_217853) is not StypyTypeError):

    if (import_217853 != 'pyd_module'):
        __import__(import_217853)
        sys_modules_217854 = sys.modules[import_217853]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.mathtext', sys_modules_217854.module_type_store, module_type_store, ['MathTextParser'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_217854, sys_modules_217854.module_type_store, module_type_store)
    else:
        from matplotlib.mathtext import MathTextParser

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.mathtext', None, module_type_store, ['MathTextParser'], [MathTextParser])

else:
    # Assigning a type to the variable 'matplotlib.mathtext' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.mathtext', import_217853)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from matplotlib.path import Path' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217855 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.path')

if (type(import_217855) is not StypyTypeError):

    if (import_217855 != 'pyd_module'):
        __import__(import_217855)
        sys_modules_217856 = sys.modules[import_217855]
        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.path', sys_modules_217856.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 40, 0), __file__, sys_modules_217856, sys_modules_217856.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.path', import_217855)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'from matplotlib.transforms import Bbox, BboxBase' statement (line 41)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217857 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.transforms')

if (type(import_217857) is not StypyTypeError):

    if (import_217857 != 'pyd_module'):
        __import__(import_217857)
        sys_modules_217858 = sys.modules[import_217857]
        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.transforms', sys_modules_217858.module_type_store, module_type_store, ['Bbox', 'BboxBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 41, 0), __file__, sys_modules_217858, sys_modules_217858.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Bbox, BboxBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.transforms', None, module_type_store, ['Bbox', 'BboxBase'], [Bbox, BboxBase])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.transforms', import_217857)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 0))

# 'from matplotlib import mcolors' statement (line 42)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217859 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'matplotlib')

if (type(import_217859) is not StypyTypeError):

    if (import_217859 != 'pyd_module'):
        __import__(import_217859)
        sys_modules_217860 = sys.modules[import_217859]
        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'matplotlib', sys_modules_217860.module_type_store, module_type_store, ['colors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 42, 0), __file__, sys_modules_217860, sys_modules_217860.module_type_store, module_type_store)
    else:
        from matplotlib import colors as mcolors

        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'matplotlib', None, module_type_store, ['colors'], [mcolors])

else:
    # Assigning a type to the variable 'matplotlib' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'matplotlib', import_217859)

# Adding an alias
module_type_store.add_alias('mcolors', 'colors')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from matplotlib.backends._backend_agg import _RendererAgg' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217861 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.backends._backend_agg')

if (type(import_217861) is not StypyTypeError):

    if (import_217861 != 'pyd_module'):
        __import__(import_217861)
        sys_modules_217862 = sys.modules[import_217861]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.backends._backend_agg', sys_modules_217862.module_type_store, module_type_store, ['RendererAgg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_217862, sys_modules_217862.module_type_store, module_type_store)
    else:
        from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.backends._backend_agg', None, module_type_store, ['RendererAgg'], [_RendererAgg])

else:
    # Assigning a type to the variable 'matplotlib.backends._backend_agg' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.backends._backend_agg', import_217861)

# Adding an alias
module_type_store.add_alias('_RendererAgg', 'RendererAgg')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'from matplotlib import _png' statement (line 45)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217863 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib')

if (type(import_217863) is not StypyTypeError):

    if (import_217863 != 'pyd_module'):
        __import__(import_217863)
        sys_modules_217864 = sys.modules[import_217863]
        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib', sys_modules_217864.module_type_store, module_type_store, ['_png'])
        nest_module(stypy.reporting.localization.Localization(__file__, 45, 0), __file__, sys_modules_217864, sys_modules_217864.module_type_store, module_type_store)
    else:
        from matplotlib import _png

        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib', None, module_type_store, ['_png'], [_png])

else:
    # Assigning a type to the variable 'matplotlib' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib', import_217863)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')



# SSA begins for try-except statement (line 47)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 4))

# 'from PIL import Image' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_217865 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 4), 'PIL')

if (type(import_217865) is not StypyTypeError):

    if (import_217865 != 'pyd_module'):
        __import__(import_217865)
        sys_modules_217866 = sys.modules[import_217865]
        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 4), 'PIL', sys_modules_217866.module_type_store, module_type_store, ['Image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 48, 4), __file__, sys_modules_217866, sys_modules_217866.module_type_store, module_type_store)
    else:
        from PIL import Image

        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 4), 'PIL', None, module_type_store, ['Image'], [Image])

else:
    # Assigning a type to the variable 'PIL' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'PIL', import_217865)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Name to a Name (line 49):

# Assigning a Name to a Name (line 49):
# Getting the type of 'True' (line 49)
True_217867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'True')
# Assigning a type to the variable '_has_pil' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), '_has_pil', True_217867)
# SSA branch for the except part of a try statement (line 47)
# SSA branch for the except 'ImportError' branch of a try statement (line 47)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 51):

# Assigning a Name to a Name (line 51):
# Getting the type of 'False' (line 51)
False_217868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'False')
# Assigning a type to the variable '_has_pil' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), '_has_pil', False_217868)
# SSA join for try-except statement (line 47)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Str to a Name (line 53):

# Assigning a Str to a Name (line 53):
unicode_217869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'unicode', u'v2.2')
# Assigning a type to the variable 'backend_version' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'backend_version', unicode_217869)

@norecursion
def get_hinting_flag(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_hinting_flag'
    module_type_store = module_type_store.open_function_context('get_hinting_flag', 55, 0, False)
    
    # Passed parameters checking function
    get_hinting_flag.stypy_localization = localization
    get_hinting_flag.stypy_type_of_self = None
    get_hinting_flag.stypy_type_store = module_type_store
    get_hinting_flag.stypy_function_name = 'get_hinting_flag'
    get_hinting_flag.stypy_param_names_list = []
    get_hinting_flag.stypy_varargs_param_name = None
    get_hinting_flag.stypy_kwargs_param_name = None
    get_hinting_flag.stypy_call_defaults = defaults
    get_hinting_flag.stypy_call_varargs = varargs
    get_hinting_flag.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_hinting_flag', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_hinting_flag', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_hinting_flag(...)' code ##################

    
    # Assigning a Dict to a Name (line 56):
    
    # Assigning a Dict to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'dict' (line 56)
    dict_217870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 56)
    # Adding element type (key, value) (line 56)
    # Getting the type of 'True' (line 57)
    True_217871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'True')
    # Getting the type of 'LOAD_FORCE_AUTOHINT' (line 57)
    LOAD_FORCE_AUTOHINT_217872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'LOAD_FORCE_AUTOHINT')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 14), dict_217870, (True_217871, LOAD_FORCE_AUTOHINT_217872))
    # Adding element type (key, value) (line 56)
    # Getting the type of 'False' (line 58)
    False_217873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'False')
    # Getting the type of 'LOAD_NO_HINTING' (line 58)
    LOAD_NO_HINTING_217874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'LOAD_NO_HINTING')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 14), dict_217870, (False_217873, LOAD_NO_HINTING_217874))
    # Adding element type (key, value) (line 56)
    unicode_217875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'unicode', u'either')
    # Getting the type of 'LOAD_DEFAULT' (line 59)
    LOAD_DEFAULT_217876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'LOAD_DEFAULT')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 14), dict_217870, (unicode_217875, LOAD_DEFAULT_217876))
    # Adding element type (key, value) (line 56)
    unicode_217877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'unicode', u'native')
    # Getting the type of 'LOAD_NO_AUTOHINT' (line 60)
    LOAD_NO_AUTOHINT_217878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'LOAD_NO_AUTOHINT')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 14), dict_217870, (unicode_217877, LOAD_NO_AUTOHINT_217878))
    # Adding element type (key, value) (line 56)
    unicode_217879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'unicode', u'auto')
    # Getting the type of 'LOAD_FORCE_AUTOHINT' (line 61)
    LOAD_FORCE_AUTOHINT_217880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'LOAD_FORCE_AUTOHINT')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 14), dict_217870, (unicode_217879, LOAD_FORCE_AUTOHINT_217880))
    # Adding element type (key, value) (line 56)
    unicode_217881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'unicode', u'none')
    # Getting the type of 'LOAD_NO_HINTING' (line 62)
    LOAD_NO_HINTING_217882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'LOAD_NO_HINTING')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 14), dict_217870, (unicode_217881, LOAD_NO_HINTING_217882))
    
    # Assigning a type to the variable 'mapping' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'mapping', dict_217870)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    unicode_217883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'unicode', u'text.hinting')
    # Getting the type of 'rcParams' (line 64)
    rcParams_217884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___217885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), rcParams_217884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_217886 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), getitem___217885, unicode_217883)
    
    # Getting the type of 'mapping' (line 64)
    mapping_217887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'mapping')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___217888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), mapping_217887, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_217889 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), getitem___217888, subscript_call_result_217886)
    
    # Assigning a type to the variable 'stypy_return_type' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type', subscript_call_result_217889)
    
    # ################# End of 'get_hinting_flag(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_hinting_flag' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_217890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_217890)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_hinting_flag'
    return stypy_return_type_217890

# Assigning a type to the variable 'get_hinting_flag' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'get_hinting_flag', get_hinting_flag)
# Declaration of the 'RendererAgg' class
# Getting the type of 'RendererBase' (line 67)
RendererBase_217891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'RendererBase')

class RendererAgg(RendererBase_217891, ):
    unicode_217892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'unicode', u'\n    The renderer handles all the drawing primitives using a graphics\n    context instance that controls the colors/styles\n    ')
    
    # Assigning a Num to a Name (line 72):
    
    # Assigning a Call to a Name (line 85):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.__init__', ['width', 'height', 'dpi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['width', 'height', 'dpi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_217895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'self', False)
        # Processing the call keyword arguments (line 87)
        kwargs_217896 = {}
        # Getting the type of 'RendererBase' (line 87)
        RendererBase_217893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'RendererBase', False)
        # Obtaining the member '__init__' of a type (line 87)
        init___217894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), RendererBase_217893, '__init__')
        # Calling __init__(args, kwargs) (line 87)
        init___call_result_217897 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), init___217894, *[self_217895], **kwargs_217896)
        
        
        # Assigning a Name to a Attribute (line 89):
        
        # Assigning a Name to a Attribute (line 89):
        # Getting the type of 'dpi' (line 89)
        dpi_217898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'dpi')
        # Getting the type of 'self' (line 89)
        self_217899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member 'dpi' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_217899, 'dpi', dpi_217898)
        
        # Assigning a Name to a Attribute (line 90):
        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'width' (line 90)
        width_217900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'width')
        # Getting the type of 'self' (line 90)
        self_217901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'width' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_217901, 'width', width_217900)
        
        # Assigning a Name to a Attribute (line 91):
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'height' (line 91)
        height_217902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'height')
        # Getting the type of 'self' (line 91)
        self_217903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'height' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_217903, 'height', height_217902)
        
        # Assigning a Call to a Attribute (line 92):
        
        # Assigning a Call to a Attribute (line 92):
        
        # Call to _RendererAgg(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to int(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'width' (line 92)
        width_217906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'width', False)
        # Processing the call keyword arguments (line 92)
        kwargs_217907 = {}
        # Getting the type of 'int' (line 92)
        int_217905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 38), 'int', False)
        # Calling int(args, kwargs) (line 92)
        int_call_result_217908 = invoke(stypy.reporting.localization.Localization(__file__, 92, 38), int_217905, *[width_217906], **kwargs_217907)
        
        
        # Call to int(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'height' (line 92)
        height_217910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 54), 'height', False)
        # Processing the call keyword arguments (line 92)
        kwargs_217911 = {}
        # Getting the type of 'int' (line 92)
        int_217909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 50), 'int', False)
        # Calling int(args, kwargs) (line 92)
        int_call_result_217912 = invoke(stypy.reporting.localization.Localization(__file__, 92, 50), int_217909, *[height_217910], **kwargs_217911)
        
        # Getting the type of 'dpi' (line 92)
        dpi_217913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 63), 'dpi', False)
        # Processing the call keyword arguments (line 92)
        # Getting the type of 'False' (line 92)
        False_217914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 74), 'False', False)
        keyword_217915 = False_217914
        kwargs_217916 = {'debug': keyword_217915}
        # Getting the type of '_RendererAgg' (line 92)
        _RendererAgg_217904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), '_RendererAgg', False)
        # Calling _RendererAgg(args, kwargs) (line 92)
        _RendererAgg_call_result_217917 = invoke(stypy.reporting.localization.Localization(__file__, 92, 25), _RendererAgg_217904, *[int_call_result_217908, int_call_result_217912, dpi_217913], **kwargs_217916)
        
        # Getting the type of 'self' (line 92)
        self_217918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member '_renderer' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_217918, '_renderer', _RendererAgg_call_result_217917)
        
        # Assigning a List to a Attribute (line 93):
        
        # Assigning a List to a Attribute (line 93):
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_217919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        
        # Getting the type of 'self' (line 93)
        self_217920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member '_filter_renderers' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_217920, '_filter_renderers', list_217919)
        
        # Call to _update_methods(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_217923 = {}
        # Getting the type of 'self' (line 95)
        self_217921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self', False)
        # Obtaining the member '_update_methods' of a type (line 95)
        _update_methods_217922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_217921, '_update_methods')
        # Calling _update_methods(args, kwargs) (line 95)
        _update_methods_call_result_217924 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), _update_methods_217922, *[], **kwargs_217923)
        
        
        # Assigning a Call to a Attribute (line 96):
        
        # Assigning a Call to a Attribute (line 96):
        
        # Call to MathTextParser(...): (line 96)
        # Processing the call arguments (line 96)
        unicode_217926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 46), 'unicode', u'Agg')
        # Processing the call keyword arguments (line 96)
        kwargs_217927 = {}
        # Getting the type of 'MathTextParser' (line 96)
        MathTextParser_217925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'MathTextParser', False)
        # Calling MathTextParser(args, kwargs) (line 96)
        MathTextParser_call_result_217928 = invoke(stypy.reporting.localization.Localization(__file__, 96, 31), MathTextParser_217925, *[unicode_217926], **kwargs_217927)
        
        # Getting the type of 'self' (line 96)
        self_217929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'mathtext_parser' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_217929, 'mathtext_parser', MathTextParser_call_result_217928)
        
        # Assigning a Call to a Attribute (line 98):
        
        # Assigning a Call to a Attribute (line 98):
        
        # Call to from_bounds(...): (line 98)
        # Processing the call arguments (line 98)
        int_217932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 37), 'int')
        int_217933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 40), 'int')
        # Getting the type of 'self' (line 98)
        self_217934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 43), 'self', False)
        # Obtaining the member 'width' of a type (line 98)
        width_217935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 43), self_217934, 'width')
        # Getting the type of 'self' (line 98)
        self_217936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 55), 'self', False)
        # Obtaining the member 'height' of a type (line 98)
        height_217937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 55), self_217936, 'height')
        # Processing the call keyword arguments (line 98)
        kwargs_217938 = {}
        # Getting the type of 'Bbox' (line 98)
        Bbox_217930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'Bbox', False)
        # Obtaining the member 'from_bounds' of a type (line 98)
        from_bounds_217931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), Bbox_217930, 'from_bounds')
        # Calling from_bounds(args, kwargs) (line 98)
        from_bounds_call_result_217939 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), from_bounds_217931, *[int_217932, int_217933, width_217935, height_217937], **kwargs_217938)
        
        # Getting the type of 'self' (line 98)
        self_217940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member 'bbox' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_217940, 'bbox', from_bounds_call_result_217939)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getstate__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getstate__'
        module_type_store = module_type_store.open_function_context('__getstate__', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_function_name', 'RendererAgg.__getstate__')
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.__getstate__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.__getstate__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getstate__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getstate__(...)' code ##################

        
        # Obtaining an instance of the builtin type 'dict' (line 103)
        dict_217941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 103)
        # Adding element type (key, value) (line 103)
        unicode_217942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 16), 'unicode', u'width')
        # Getting the type of 'self' (line 103)
        self_217943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'self')
        # Obtaining the member 'width' of a type (line 103)
        width_217944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 25), self_217943, 'width')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), dict_217941, (unicode_217942, width_217944))
        # Adding element type (key, value) (line 103)
        unicode_217945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 37), 'unicode', u'height')
        # Getting the type of 'self' (line 103)
        self_217946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 47), 'self')
        # Obtaining the member 'height' of a type (line 103)
        height_217947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 47), self_217946, 'height')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), dict_217941, (unicode_217945, height_217947))
        # Adding element type (key, value) (line 103)
        unicode_217948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 60), 'unicode', u'dpi')
        # Getting the type of 'self' (line 103)
        self_217949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 67), 'self')
        # Obtaining the member 'dpi' of a type (line 103)
        dpi_217950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 67), self_217949, 'dpi')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), dict_217941, (unicode_217948, dpi_217950))
        
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', dict_217941)
        
        # ################# End of '__getstate__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getstate__' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_217951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217951)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getstate__'
        return stypy_return_type_217951


    @norecursion
    def __setstate__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setstate__'
        module_type_store = module_type_store.open_function_context('__setstate__', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_function_name', 'RendererAgg.__setstate__')
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_param_names_list', ['state'])
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.__setstate__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.__setstate__', ['state'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setstate__', localization, ['state'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setstate__(...)' code ##################

        
        # Call to __init__(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining the type of the subscript
        unicode_217954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'unicode', u'width')
        # Getting the type of 'state' (line 106)
        state_217955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'state', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___217956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 22), state_217955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_217957 = invoke(stypy.reporting.localization.Localization(__file__, 106, 22), getitem___217956, unicode_217954)
        
        
        # Obtaining the type of the subscript
        unicode_217958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 44), 'unicode', u'height')
        # Getting the type of 'state' (line 106)
        state_217959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 38), 'state', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___217960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 38), state_217959, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_217961 = invoke(stypy.reporting.localization.Localization(__file__, 106, 38), getitem___217960, unicode_217958)
        
        
        # Obtaining the type of the subscript
        unicode_217962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 61), 'unicode', u'dpi')
        # Getting the type of 'state' (line 106)
        state_217963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 55), 'state', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___217964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 55), state_217963, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_217965 = invoke(stypy.reporting.localization.Localization(__file__, 106, 55), getitem___217964, unicode_217962)
        
        # Processing the call keyword arguments (line 106)
        kwargs_217966 = {}
        # Getting the type of 'self' (line 106)
        self_217952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self', False)
        # Obtaining the member '__init__' of a type (line 106)
        init___217953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_217952, '__init__')
        # Calling __init__(args, kwargs) (line 106)
        init___call_result_217967 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), init___217953, *[subscript_call_result_217957, subscript_call_result_217961, subscript_call_result_217965], **kwargs_217966)
        
        
        # ################# End of '__setstate__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setstate__' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_217968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217968)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setstate__'
        return stypy_return_type_217968


    @norecursion
    def _get_hinting_flag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_hinting_flag'
        module_type_store = module_type_store.open_function_context('_get_hinting_flag', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_function_name', 'RendererAgg._get_hinting_flag')
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg._get_hinting_flag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg._get_hinting_flag', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Obtaining the type of the subscript
        unicode_217969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'unicode', u'text.hinting')
        # Getting the type of 'rcParams' (line 109)
        rcParams_217970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___217971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), rcParams_217970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_217972 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), getitem___217971, unicode_217969)
        
        # Testing the type of an if condition (line 109)
        if_condition_217973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), subscript_call_result_217972)
        # Assigning a type to the variable 'if_condition_217973' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_217973', if_condition_217973)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'LOAD_FORCE_AUTOHINT' (line 110)
        LOAD_FORCE_AUTOHINT_217974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'LOAD_FORCE_AUTOHINT')
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'stypy_return_type', LOAD_FORCE_AUTOHINT_217974)
        # SSA branch for the else part of an if statement (line 109)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'LOAD_NO_HINTING' (line 112)
        LOAD_NO_HINTING_217975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'LOAD_NO_HINTING')
        # Assigning a type to the variable 'stypy_return_type' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'stypy_return_type', LOAD_NO_HINTING_217975)
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_get_hinting_flag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_hinting_flag' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_217976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217976)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_hinting_flag'
        return stypy_return_type_217976


    @norecursion
    def draw_markers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_markers'
        module_type_store = module_type_store.open_function_context('draw_markers', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_function_name', 'RendererAgg.draw_markers')
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_varargs_param_name', 'kl')
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.draw_markers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.draw_markers', [], 'kl', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_markers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_markers(...)' code ##################

        
        # Call to draw_markers(...): (line 117)
        # Getting the type of 'kl' (line 117)
        kl_217980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 44), 'kl', False)
        # Processing the call keyword arguments (line 117)
        # Getting the type of 'kw' (line 117)
        kw_217981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 50), 'kw', False)
        kwargs_217982 = {'kw_217981': kw_217981}
        # Getting the type of 'self' (line 117)
        self_217977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'self', False)
        # Obtaining the member '_renderer' of a type (line 117)
        _renderer_217978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), self_217977, '_renderer')
        # Obtaining the member 'draw_markers' of a type (line 117)
        draw_markers_217979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), _renderer_217978, 'draw_markers')
        # Calling draw_markers(args, kwargs) (line 117)
        draw_markers_call_result_217983 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), draw_markers_217979, *[kl_217980], **kwargs_217982)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', draw_markers_call_result_217983)
        
        # ################# End of 'draw_markers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_markers' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_217984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217984)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_markers'
        return stypy_return_type_217984


    @norecursion
    def draw_path_collection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_path_collection'
        module_type_store = module_type_store.open_function_context('draw_path_collection', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_function_name', 'RendererAgg.draw_path_collection')
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_varargs_param_name', 'kl')
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.draw_path_collection.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.draw_path_collection', [], 'kl', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path_collection', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path_collection(...)' code ##################

        
        # Call to draw_path_collection(...): (line 120)
        # Getting the type of 'kl' (line 120)
        kl_217988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 52), 'kl', False)
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'kw' (line 120)
        kw_217989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 58), 'kw', False)
        kwargs_217990 = {'kw_217989': kw_217989}
        # Getting the type of 'self' (line 120)
        self_217985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'self', False)
        # Obtaining the member '_renderer' of a type (line 120)
        _renderer_217986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), self_217985, '_renderer')
        # Obtaining the member 'draw_path_collection' of a type (line 120)
        draw_path_collection_217987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), _renderer_217986, 'draw_path_collection')
        # Calling draw_path_collection(args, kwargs) (line 120)
        draw_path_collection_call_result_217991 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), draw_path_collection_217987, *[kl_217988], **kwargs_217990)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', draw_path_collection_call_result_217991)
        
        # ################# End of 'draw_path_collection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path_collection' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_217992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path_collection'
        return stypy_return_type_217992


    @norecursion
    def _update_methods(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_methods'
        module_type_store = module_type_store.open_function_context('_update_methods', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg._update_methods.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg._update_methods.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg._update_methods.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg._update_methods.__dict__.__setitem__('stypy_function_name', 'RendererAgg._update_methods')
        RendererAgg._update_methods.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg._update_methods.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg._update_methods.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg._update_methods.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg._update_methods.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg._update_methods.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg._update_methods.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg._update_methods', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_methods', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_methods(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 123):
        
        # Assigning a Attribute to a Attribute (line 123):
        # Getting the type of 'self' (line 123)
        self_217993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'self')
        # Obtaining the member '_renderer' of a type (line 123)
        _renderer_217994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 30), self_217993, '_renderer')
        # Obtaining the member 'draw_quad_mesh' of a type (line 123)
        draw_quad_mesh_217995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 30), _renderer_217994, 'draw_quad_mesh')
        # Getting the type of 'self' (line 123)
        self_217996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self')
        # Setting the type of the member 'draw_quad_mesh' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_217996, 'draw_quad_mesh', draw_quad_mesh_217995)
        
        # Assigning a Attribute to a Attribute (line 124):
        
        # Assigning a Attribute to a Attribute (line 124):
        # Getting the type of 'self' (line 124)
        self_217997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'self')
        # Obtaining the member '_renderer' of a type (line 124)
        _renderer_217998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 37), self_217997, '_renderer')
        # Obtaining the member 'draw_gouraud_triangle' of a type (line 124)
        draw_gouraud_triangle_217999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 37), _renderer_217998, 'draw_gouraud_triangle')
        # Getting the type of 'self' (line 124)
        self_218000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self')
        # Setting the type of the member 'draw_gouraud_triangle' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_218000, 'draw_gouraud_triangle', draw_gouraud_triangle_217999)
        
        # Assigning a Attribute to a Attribute (line 125):
        
        # Assigning a Attribute to a Attribute (line 125):
        # Getting the type of 'self' (line 125)
        self_218001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'self')
        # Obtaining the member '_renderer' of a type (line 125)
        _renderer_218002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 38), self_218001, '_renderer')
        # Obtaining the member 'draw_gouraud_triangles' of a type (line 125)
        draw_gouraud_triangles_218003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 38), _renderer_218002, 'draw_gouraud_triangles')
        # Getting the type of 'self' (line 125)
        self_218004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self')
        # Setting the type of the member 'draw_gouraud_triangles' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_218004, 'draw_gouraud_triangles', draw_gouraud_triangles_218003)
        
        # Assigning a Attribute to a Attribute (line 126):
        
        # Assigning a Attribute to a Attribute (line 126):
        # Getting the type of 'self' (line 126)
        self_218005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'self')
        # Obtaining the member '_renderer' of a type (line 126)
        _renderer_218006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 26), self_218005, '_renderer')
        # Obtaining the member 'draw_image' of a type (line 126)
        draw_image_218007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 26), _renderer_218006, 'draw_image')
        # Getting the type of 'self' (line 126)
        self_218008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'self')
        # Setting the type of the member 'draw_image' of a type (line 126)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), self_218008, 'draw_image', draw_image_218007)
        
        # Assigning a Attribute to a Attribute (line 127):
        
        # Assigning a Attribute to a Attribute (line 127):
        # Getting the type of 'self' (line 127)
        self_218009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'self')
        # Obtaining the member '_renderer' of a type (line 127)
        _renderer_218010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), self_218009, '_renderer')
        # Obtaining the member 'copy_from_bbox' of a type (line 127)
        copy_from_bbox_218011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), _renderer_218010, 'copy_from_bbox')
        # Getting the type of 'self' (line 127)
        self_218012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self')
        # Setting the type of the member 'copy_from_bbox' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_218012, 'copy_from_bbox', copy_from_bbox_218011)
        
        # Assigning a Attribute to a Attribute (line 128):
        
        # Assigning a Attribute to a Attribute (line 128):
        # Getting the type of 'self' (line 128)
        self_218013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 35), 'self')
        # Obtaining the member '_renderer' of a type (line 128)
        _renderer_218014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 35), self_218013, '_renderer')
        # Obtaining the member 'get_content_extents' of a type (line 128)
        get_content_extents_218015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 35), _renderer_218014, 'get_content_extents')
        # Getting the type of 'self' (line 128)
        self_218016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'get_content_extents' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_218016, 'get_content_extents', get_content_extents_218015)
        
        # ################# End of '_update_methods(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_methods' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_218017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218017)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_methods'
        return stypy_return_type_218017


    @norecursion
    def tostring_rgba_minimized(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tostring_rgba_minimized'
        module_type_store = module_type_store.open_function_context('tostring_rgba_minimized', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_function_name', 'RendererAgg.tostring_rgba_minimized')
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.tostring_rgba_minimized.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.tostring_rgba_minimized', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tostring_rgba_minimized', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tostring_rgba_minimized(...)' code ##################

        
        # Assigning a Call to a Name (line 131):
        
        # Assigning a Call to a Name (line 131):
        
        # Call to get_content_extents(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_218020 = {}
        # Getting the type of 'self' (line 131)
        self_218018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), 'self', False)
        # Obtaining the member 'get_content_extents' of a type (line 131)
        get_content_extents_218019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 18), self_218018, 'get_content_extents')
        # Calling get_content_extents(args, kwargs) (line 131)
        get_content_extents_call_result_218021 = invoke(stypy.reporting.localization.Localization(__file__, 131, 18), get_content_extents_218019, *[], **kwargs_218020)
        
        # Assigning a type to the variable 'extents' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'extents', get_content_extents_call_result_218021)
        
        # Assigning a List to a Name (line 132):
        
        # Assigning a List to a Name (line 132):
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_218022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_218023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        
        # Obtaining the type of the subscript
        int_218024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'int')
        # Getting the type of 'extents' (line 132)
        extents_218025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'extents')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___218026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 17), extents_218025, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_218027 = invoke(stypy.reporting.localization.Localization(__file__, 132, 17), getitem___218026, int_218024)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), list_218023, subscript_call_result_218027)
        # Adding element type (line 132)
        # Getting the type of 'self' (line 132)
        self_218028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 29), 'self')
        # Obtaining the member 'height' of a type (line 132)
        height_218029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 29), self_218028, 'height')
        
        # Obtaining the type of the subscript
        int_218030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 52), 'int')
        # Getting the type of 'extents' (line 132)
        extents_218031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 44), 'extents')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___218032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 44), extents_218031, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_218033 = invoke(stypy.reporting.localization.Localization(__file__, 132, 44), getitem___218032, int_218030)
        
        
        # Obtaining the type of the subscript
        int_218034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 65), 'int')
        # Getting the type of 'extents' (line 132)
        extents_218035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 57), 'extents')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___218036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 57), extents_218035, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_218037 = invoke(stypy.reporting.localization.Localization(__file__, 132, 57), getitem___218036, int_218034)
        
        # Applying the binary operator '+' (line 132)
        result_add_218038 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 44), '+', subscript_call_result_218033, subscript_call_result_218037)
        
        # Applying the binary operator '-' (line 132)
        result_sub_218039 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 29), '-', height_218029, result_add_218038)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), list_218023, result_sub_218039)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 15), list_218022, list_218023)
        # Adding element type (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_218040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        
        # Obtaining the type of the subscript
        int_218041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
        # Getting the type of 'extents' (line 133)
        extents_218042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'extents')
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___218043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 17), extents_218042, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_218044 = invoke(stypy.reporting.localization.Localization(__file__, 133, 17), getitem___218043, int_218041)
        
        
        # Obtaining the type of the subscript
        int_218045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'int')
        # Getting the type of 'extents' (line 133)
        extents_218046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 30), 'extents')
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___218047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 30), extents_218046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_218048 = invoke(stypy.reporting.localization.Localization(__file__, 133, 30), getitem___218047, int_218045)
        
        # Applying the binary operator '+' (line 133)
        result_add_218049 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 17), '+', subscript_call_result_218044, subscript_call_result_218048)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 16), list_218040, result_add_218049)
        # Adding element type (line 133)
        # Getting the type of 'self' (line 133)
        self_218050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'self')
        # Obtaining the member 'height' of a type (line 133)
        height_218051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 42), self_218050, 'height')
        
        # Obtaining the type of the subscript
        int_218052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 64), 'int')
        # Getting the type of 'extents' (line 133)
        extents_218053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 56), 'extents')
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___218054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 56), extents_218053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_218055 = invoke(stypy.reporting.localization.Localization(__file__, 133, 56), getitem___218054, int_218052)
        
        # Applying the binary operator '-' (line 133)
        result_sub_218056 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 42), '-', height_218051, subscript_call_result_218055)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 16), list_218040, result_sub_218056)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 15), list_218022, list_218040)
        
        # Assigning a type to the variable 'bbox' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'bbox', list_218022)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to copy_from_bbox(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'bbox' (line 134)
        bbox_218059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 37), 'bbox', False)
        # Processing the call keyword arguments (line 134)
        kwargs_218060 = {}
        # Getting the type of 'self' (line 134)
        self_218057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'self', False)
        # Obtaining the member 'copy_from_bbox' of a type (line 134)
        copy_from_bbox_218058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 17), self_218057, 'copy_from_bbox')
        # Calling copy_from_bbox(args, kwargs) (line 134)
        copy_from_bbox_call_result_218061 = invoke(stypy.reporting.localization.Localization(__file__, 134, 17), copy_from_bbox_218058, *[bbox_218059], **kwargs_218060)
        
        # Assigning a type to the variable 'region' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'region', copy_from_bbox_call_result_218061)
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_218062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        
        # Call to array(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'region' (line 135)
        region_218065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'region', False)
        # Processing the call keyword arguments (line 135)
        kwargs_218066 = {}
        # Getting the type of 'np' (line 135)
        np_218063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 135)
        array_218064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), np_218063, 'array')
        # Calling array(args, kwargs) (line 135)
        array_call_result_218067 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), array_218064, *[region_218065], **kwargs_218066)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 15), tuple_218062, array_call_result_218067)
        # Adding element type (line 135)
        # Getting the type of 'extents' (line 135)
        extents_218068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'extents')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 15), tuple_218062, extents_218068)
        
        # Assigning a type to the variable 'stypy_return_type' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', tuple_218062)
        
        # ################# End of 'tostring_rgba_minimized(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tostring_rgba_minimized' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_218069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tostring_rgba_minimized'
        return stypy_return_type_218069


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 137)
        None_218070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 53), 'None')
        defaults = [None_218070]
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.draw_path.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.draw_path.__dict__.__setitem__('stypy_function_name', 'RendererAgg.draw_path')
        RendererAgg.draw_path.__dict__.__setitem__('stypy_param_names_list', ['gc', 'path', 'transform', 'rgbFace'])
        RendererAgg.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.draw_path', ['gc', 'path', 'transform', 'rgbFace'], None, None, defaults, varargs, kwargs)

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

        unicode_218071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, (-1)), 'unicode', u'\n        Draw the path\n        ')
        
        # Assigning a Subscript to a Name (line 141):
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        unicode_218072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 24), 'unicode', u'agg.path.chunksize')
        # Getting the type of 'rcParams' (line 141)
        rcParams_218073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___218074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), rcParams_218073, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_218075 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), getitem___218074, unicode_218072)
        
        # Assigning a type to the variable 'nmax' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'nmax', subscript_call_result_218075)
        
        # Assigning a Subscript to a Name (line 142):
        
        # Assigning a Subscript to a Name (line 142):
        
        # Obtaining the type of the subscript
        int_218076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 35), 'int')
        # Getting the type of 'path' (line 142)
        path_218077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'path')
        # Obtaining the member 'vertices' of a type (line 142)
        vertices_218078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), path_218077, 'vertices')
        # Obtaining the member 'shape' of a type (line 142)
        shape_218079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), vertices_218078, 'shape')
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___218080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), shape_218079, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_218081 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), getitem___218080, int_218076)
        
        # Assigning a type to the variable 'npts' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'npts', subscript_call_result_218081)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'nmax' (line 144)
        nmax_218082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'nmax')
        int_218083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 19), 'int')
        # Applying the binary operator '>' (line 144)
        result_gt_218084 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), '>', nmax_218082, int_218083)
        
        
        # Getting the type of 'npts' (line 144)
        npts_218085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'npts')
        # Getting the type of 'nmax' (line 144)
        nmax_218086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'nmax')
        # Applying the binary operator '>' (line 144)
        result_gt_218087 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 27), '>', npts_218085, nmax_218086)
        
        # Applying the binary operator 'and' (line 144)
        result_and_keyword_218088 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), 'and', result_gt_218084, result_gt_218087)
        # Getting the type of 'path' (line 144)
        path_218089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 43), 'path')
        # Obtaining the member 'should_simplify' of a type (line 144)
        should_simplify_218090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 43), path_218089, 'should_simplify')
        # Applying the binary operator 'and' (line 144)
        result_and_keyword_218091 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), 'and', result_and_keyword_218088, should_simplify_218090)
        
        # Getting the type of 'rgbFace' (line 145)
        rgbFace_218092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'rgbFace')
        # Getting the type of 'None' (line 145)
        None_218093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 27), 'None')
        # Applying the binary operator 'is' (line 145)
        result_is__218094 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 16), 'is', rgbFace_218092, None_218093)
        
        # Applying the binary operator 'and' (line 144)
        result_and_keyword_218095 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), 'and', result_and_keyword_218091, result_is__218094)
        
        
        # Call to get_hatch(...): (line 145)
        # Processing the call keyword arguments (line 145)
        kwargs_218098 = {}
        # Getting the type of 'gc' (line 145)
        gc_218096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'gc', False)
        # Obtaining the member 'get_hatch' of a type (line 145)
        get_hatch_218097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 36), gc_218096, 'get_hatch')
        # Calling get_hatch(args, kwargs) (line 145)
        get_hatch_call_result_218099 = invoke(stypy.reporting.localization.Localization(__file__, 145, 36), get_hatch_218097, *[], **kwargs_218098)
        
        # Getting the type of 'None' (line 145)
        None_218100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 54), 'None')
        # Applying the binary operator 'is' (line 145)
        result_is__218101 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 36), 'is', get_hatch_call_result_218099, None_218100)
        
        # Applying the binary operator 'and' (line 144)
        result_and_keyword_218102 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), 'and', result_and_keyword_218095, result_is__218101)
        
        # Testing the type of an if condition (line 144)
        if_condition_218103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), result_and_keyword_218102)
        # Assigning a type to the variable 'if_condition_218103' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_218103', if_condition_218103)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to ceil(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'npts' (line 146)
        npts_218106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'npts', False)
        
        # Call to float(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'nmax' (line 146)
        nmax_218108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'nmax', False)
        # Processing the call keyword arguments (line 146)
        kwargs_218109 = {}
        # Getting the type of 'float' (line 146)
        float_218107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'float', False)
        # Calling float(args, kwargs) (line 146)
        float_call_result_218110 = invoke(stypy.reporting.localization.Localization(__file__, 146, 33), float_218107, *[nmax_218108], **kwargs_218109)
        
        # Applying the binary operator 'div' (line 146)
        result_div_218111 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 26), 'div', npts_218106, float_call_result_218110)
        
        # Processing the call keyword arguments (line 146)
        kwargs_218112 = {}
        # Getting the type of 'np' (line 146)
        np_218104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'np', False)
        # Obtaining the member 'ceil' of a type (line 146)
        ceil_218105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 18), np_218104, 'ceil')
        # Calling ceil(args, kwargs) (line 146)
        ceil_call_result_218113 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), ceil_218105, *[result_div_218111], **kwargs_218112)
        
        # Assigning a type to the variable 'nch' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'nch', ceil_call_result_218113)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to int(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to ceil(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'npts' (line 147)
        npts_218117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'npts', False)
        # Getting the type of 'nch' (line 147)
        nch_218118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 40), 'nch', False)
        # Applying the binary operator 'div' (line 147)
        result_div_218119 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 33), 'div', npts_218117, nch_218118)
        
        # Processing the call keyword arguments (line 147)
        kwargs_218120 = {}
        # Getting the type of 'np' (line 147)
        np_218115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'np', False)
        # Obtaining the member 'ceil' of a type (line 147)
        ceil_218116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), np_218115, 'ceil')
        # Calling ceil(args, kwargs) (line 147)
        ceil_call_result_218121 = invoke(stypy.reporting.localization.Localization(__file__, 147, 25), ceil_218116, *[result_div_218119], **kwargs_218120)
        
        # Processing the call keyword arguments (line 147)
        kwargs_218122 = {}
        # Getting the type of 'int' (line 147)
        int_218114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'int', False)
        # Calling int(args, kwargs) (line 147)
        int_call_result_218123 = invoke(stypy.reporting.localization.Localization(__file__, 147, 21), int_218114, *[ceil_call_result_218121], **kwargs_218122)
        
        # Assigning a type to the variable 'chsize' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'chsize', int_call_result_218123)
        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to arange(...): (line 148)
        # Processing the call arguments (line 148)
        int_218126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'int')
        # Getting the type of 'npts' (line 148)
        npts_218127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'npts', False)
        # Getting the type of 'chsize' (line 148)
        chsize_218128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 36), 'chsize', False)
        # Processing the call keyword arguments (line 148)
        kwargs_218129 = {}
        # Getting the type of 'np' (line 148)
        np_218124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 148)
        arange_218125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 17), np_218124, 'arange')
        # Calling arange(args, kwargs) (line 148)
        arange_call_result_218130 = invoke(stypy.reporting.localization.Localization(__file__, 148, 17), arange_218125, *[int_218126, npts_218127, chsize_218128], **kwargs_218129)
        
        # Assigning a type to the variable 'i0' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'i0', arange_call_result_218130)
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to zeros_like(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'i0' (line 149)
        i0_218133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'i0', False)
        # Processing the call keyword arguments (line 149)
        kwargs_218134 = {}
        # Getting the type of 'np' (line 149)
        np_218131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 149)
        zeros_like_218132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 17), np_218131, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 149)
        zeros_like_call_result_218135 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), zeros_like_218132, *[i0_218133], **kwargs_218134)
        
        # Assigning a type to the variable 'i1' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'i1', zeros_like_call_result_218135)
        
        # Assigning a BinOp to a Subscript (line 150):
        
        # Assigning a BinOp to a Subscript (line 150):
        
        # Obtaining the type of the subscript
        int_218136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 25), 'int')
        slice_218137 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 150, 22), int_218136, None, None)
        # Getting the type of 'i0' (line 150)
        i0_218138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'i0')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___218139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 22), i0_218138, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_218140 = invoke(stypy.reporting.localization.Localization(__file__, 150, 22), getitem___218139, slice_218137)
        
        int_218141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 31), 'int')
        # Applying the binary operator '-' (line 150)
        result_sub_218142 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 22), '-', subscript_call_result_218140, int_218141)
        
        # Getting the type of 'i1' (line 150)
        i1_218143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'i1')
        int_218144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'int')
        slice_218145 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 150, 12), None, int_218144, None)
        # Storing an element on a container (line 150)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 12), i1_218143, (slice_218145, result_sub_218142))
        
        # Assigning a Name to a Subscript (line 151):
        
        # Assigning a Name to a Subscript (line 151):
        # Getting the type of 'npts' (line 151)
        npts_218146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'npts')
        # Getting the type of 'i1' (line 151)
        i1_218147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'i1')
        int_218148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'int')
        # Storing an element on a container (line 151)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 12), i1_218147, (int_218148, npts_218146))
        
        
        # Call to zip(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'i0' (line 152)
        i0_218150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'i0', False)
        # Getting the type of 'i1' (line 152)
        i1_218151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 36), 'i1', False)
        # Processing the call keyword arguments (line 152)
        kwargs_218152 = {}
        # Getting the type of 'zip' (line 152)
        zip_218149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'zip', False)
        # Calling zip(args, kwargs) (line 152)
        zip_call_result_218153 = invoke(stypy.reporting.localization.Localization(__file__, 152, 28), zip_218149, *[i0_218150, i1_218151], **kwargs_218152)
        
        # Testing the type of a for loop iterable (line 152)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 12), zip_call_result_218153)
        # Getting the type of the for loop variable (line 152)
        for_loop_var_218154 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 12), zip_call_result_218153)
        # Assigning a type to the variable 'ii0' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'ii0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 12), for_loop_var_218154))
        # Assigning a type to the variable 'ii1' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'ii1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 12), for_loop_var_218154))
        # SSA begins for a for statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 153):
        
        # Assigning a Subscript to a Name (line 153):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ii0' (line 153)
        ii0_218155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'ii0')
        # Getting the type of 'ii1' (line 153)
        ii1_218156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'ii1')
        slice_218157 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 153, 20), ii0_218155, ii1_218156, None)
        slice_218158 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 153, 20), None, None, None)
        # Getting the type of 'path' (line 153)
        path_218159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'path')
        # Obtaining the member 'vertices' of a type (line 153)
        vertices_218160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 20), path_218159, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___218161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 20), vertices_218160, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_218162 = invoke(stypy.reporting.localization.Localization(__file__, 153, 20), getitem___218161, (slice_218157, slice_218158))
        
        # Assigning a type to the variable 'v' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'v', subscript_call_result_218162)
        
        # Assigning a Attribute to a Name (line 154):
        
        # Assigning a Attribute to a Name (line 154):
        # Getting the type of 'path' (line 154)
        path_218163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'path')
        # Obtaining the member 'codes' of a type (line 154)
        codes_218164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 20), path_218163, 'codes')
        # Assigning a type to the variable 'c' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'c', codes_218164)
        
        # Type idiom detected: calculating its left and rigth part (line 155)
        # Getting the type of 'c' (line 155)
        c_218165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'c')
        # Getting the type of 'None' (line 155)
        None_218166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'None')
        
        (may_be_218167, more_types_in_union_218168) = may_not_be_none(c_218165, None_218166)

        if may_be_218167:

            if more_types_in_union_218168:
                # Runtime conditional SSA (line 155)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 156):
            
            # Assigning a Subscript to a Name (line 156):
            
            # Obtaining the type of the subscript
            # Getting the type of 'ii0' (line 156)
            ii0_218169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'ii0')
            # Getting the type of 'ii1' (line 156)
            ii1_218170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'ii1')
            slice_218171 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 156, 24), ii0_218169, ii1_218170, None)
            # Getting the type of 'c' (line 156)
            c_218172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'c')
            # Obtaining the member '__getitem__' of a type (line 156)
            getitem___218173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 24), c_218172, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 156)
            subscript_call_result_218174 = invoke(stypy.reporting.localization.Localization(__file__, 156, 24), getitem___218173, slice_218171)
            
            # Assigning a type to the variable 'c' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'c', subscript_call_result_218174)
            
            # Assigning a Attribute to a Subscript (line 157):
            
            # Assigning a Attribute to a Subscript (line 157):
            # Getting the type of 'Path' (line 157)
            Path_218175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'Path')
            # Obtaining the member 'MOVETO' of a type (line 157)
            MOVETO_218176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 27), Path_218175, 'MOVETO')
            # Getting the type of 'c' (line 157)
            c_218177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'c')
            int_218178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 22), 'int')
            # Storing an element on a container (line 157)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), c_218177, (int_218178, MOVETO_218176))

            if more_types_in_union_218168:
                # SSA join for if statement (line 155)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to Path(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'v' (line 158)
        v_218180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'v', False)
        # Getting the type of 'c' (line 158)
        c_218181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'c', False)
        # Processing the call keyword arguments (line 158)
        kwargs_218182 = {}
        # Getting the type of 'Path' (line 158)
        Path_218179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'Path', False)
        # Calling Path(args, kwargs) (line 158)
        Path_call_result_218183 = invoke(stypy.reporting.localization.Localization(__file__, 158, 20), Path_218179, *[v_218180, c_218181], **kwargs_218182)
        
        # Assigning a type to the variable 'p' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'p', Path_call_result_218183)
        
        
        # SSA begins for try-except statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to draw_path(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'gc' (line 160)
        gc_218187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 45), 'gc', False)
        # Getting the type of 'p' (line 160)
        p_218188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 49), 'p', False)
        # Getting the type of 'transform' (line 160)
        transform_218189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 52), 'transform', False)
        # Getting the type of 'rgbFace' (line 160)
        rgbFace_218190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 63), 'rgbFace', False)
        # Processing the call keyword arguments (line 160)
        kwargs_218191 = {}
        # Getting the type of 'self' (line 160)
        self_218184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'self', False)
        # Obtaining the member '_renderer' of a type (line 160)
        _renderer_218185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), self_218184, '_renderer')
        # Obtaining the member 'draw_path' of a type (line 160)
        draw_path_218186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), _renderer_218185, 'draw_path')
        # Calling draw_path(args, kwargs) (line 160)
        draw_path_call_result_218192 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), draw_path_218186, *[gc_218187, p_218188, transform_218189, rgbFace_218190], **kwargs_218191)
        
        # SSA branch for the except part of a try statement (line 159)
        # SSA branch for the except 'OverflowError' branch of a try statement (line 159)
        module_type_store.open_ssa_branch('except')
        
        # Call to OverflowError(...): (line 162)
        # Processing the call arguments (line 162)
        unicode_218194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 40), 'unicode', u"Exceeded cell block limit (set 'agg.path.chunksize' rcparam)")
        # Processing the call keyword arguments (line 162)
        kwargs_218195 = {}
        # Getting the type of 'OverflowError' (line 162)
        OverflowError_218193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'OverflowError', False)
        # Calling OverflowError(args, kwargs) (line 162)
        OverflowError_call_result_218196 = invoke(stypy.reporting.localization.Localization(__file__, 162, 26), OverflowError_218193, *[unicode_218194], **kwargs_218195)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 162, 20), OverflowError_call_result_218196, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 144)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to draw_path(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'gc' (line 165)
        gc_218200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 41), 'gc', False)
        # Getting the type of 'path' (line 165)
        path_218201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 45), 'path', False)
        # Getting the type of 'transform' (line 165)
        transform_218202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 51), 'transform', False)
        # Getting the type of 'rgbFace' (line 165)
        rgbFace_218203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 62), 'rgbFace', False)
        # Processing the call keyword arguments (line 165)
        kwargs_218204 = {}
        # Getting the type of 'self' (line 165)
        self_218197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'self', False)
        # Obtaining the member '_renderer' of a type (line 165)
        _renderer_218198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), self_218197, '_renderer')
        # Obtaining the member 'draw_path' of a type (line 165)
        draw_path_218199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), _renderer_218198, 'draw_path')
        # Calling draw_path(args, kwargs) (line 165)
        draw_path_call_result_218205 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), draw_path_218199, *[gc_218200, path_218201, transform_218202, rgbFace_218203], **kwargs_218204)
        
        # SSA branch for the except part of a try statement (line 164)
        # SSA branch for the except 'OverflowError' branch of a try statement (line 164)
        module_type_store.open_ssa_branch('except')
        
        # Call to OverflowError(...): (line 167)
        # Processing the call arguments (line 167)
        unicode_218207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 36), 'unicode', u"Exceeded cell block limit (set 'agg.path.chunksize' rcparam)")
        # Processing the call keyword arguments (line 167)
        kwargs_218208 = {}
        # Getting the type of 'OverflowError' (line 167)
        OverflowError_218206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'OverflowError', False)
        # Calling OverflowError(args, kwargs) (line 167)
        OverflowError_call_result_218209 = invoke(stypy.reporting.localization.Localization(__file__, 167, 22), OverflowError_218206, *[unicode_218207], **kwargs_218208)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 167, 16), OverflowError_call_result_218209, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_218210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218210)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_218210


    @norecursion
    def draw_mathtext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_mathtext'
        module_type_store = module_type_store.open_function_context('draw_mathtext', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_function_name', 'RendererAgg.draw_mathtext')
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle'])
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.draw_mathtext.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.draw_mathtext', ['gc', 'x', 'y', 's', 'prop', 'angle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_mathtext', localization, ['gc', 'x', 'y', 's', 'prop', 'angle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_mathtext(...)' code ##################

        unicode_218211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, (-1)), 'unicode', u'\n        Draw the math text using matplotlib.mathtext\n        ')
        
        # Assigning a Call to a Tuple (line 174):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 's' (line 175)
        s_218215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 39), 's', False)
        # Getting the type of 'self' (line 175)
        self_218216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), 'self', False)
        # Obtaining the member 'dpi' of a type (line 175)
        dpi_218217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 42), self_218216, 'dpi')
        # Getting the type of 'prop' (line 175)
        prop_218218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 52), 'prop', False)
        # Processing the call keyword arguments (line 175)
        kwargs_218219 = {}
        # Getting the type of 'self' (line 175)
        self_218212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'self', False)
        # Obtaining the member 'mathtext_parser' of a type (line 175)
        mathtext_parser_218213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), self_218212, 'mathtext_parser')
        # Obtaining the member 'parse' of a type (line 175)
        parse_218214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), mathtext_parser_218213, 'parse')
        # Calling parse(args, kwargs) (line 175)
        parse_call_result_218220 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), parse_218214, *[s_218215, dpi_218217, prop_218218], **kwargs_218219)
        
        # Assigning a type to the variable 'call_assignment_217766' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217766', parse_call_result_218220)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218224 = {}
        # Getting the type of 'call_assignment_217766' (line 174)
        call_assignment_217766_218221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217766', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___218222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), call_assignment_217766_218221, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218225 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218222, *[int_218223], **kwargs_218224)
        
        # Assigning a type to the variable 'call_assignment_217767' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217767', getitem___call_result_218225)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'call_assignment_217767' (line 174)
        call_assignment_217767_218226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217767')
        # Assigning a type to the variable 'ox' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'ox', call_assignment_217767_218226)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218230 = {}
        # Getting the type of 'call_assignment_217766' (line 174)
        call_assignment_217766_218227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217766', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___218228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), call_assignment_217766_218227, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218231 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218228, *[int_218229], **kwargs_218230)
        
        # Assigning a type to the variable 'call_assignment_217768' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217768', getitem___call_result_218231)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'call_assignment_217768' (line 174)
        call_assignment_217768_218232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217768')
        # Assigning a type to the variable 'oy' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'oy', call_assignment_217768_218232)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218236 = {}
        # Getting the type of 'call_assignment_217766' (line 174)
        call_assignment_217766_218233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217766', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___218234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), call_assignment_217766_218233, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218237 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218234, *[int_218235], **kwargs_218236)
        
        # Assigning a type to the variable 'call_assignment_217769' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217769', getitem___call_result_218237)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'call_assignment_217769' (line 174)
        call_assignment_217769_218238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217769')
        # Assigning a type to the variable 'width' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'width', call_assignment_217769_218238)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218242 = {}
        # Getting the type of 'call_assignment_217766' (line 174)
        call_assignment_217766_218239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217766', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___218240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), call_assignment_217766_218239, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218243 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218240, *[int_218241], **kwargs_218242)
        
        # Assigning a type to the variable 'call_assignment_217770' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217770', getitem___call_result_218243)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'call_assignment_217770' (line 174)
        call_assignment_217770_218244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217770')
        # Assigning a type to the variable 'height' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'height', call_assignment_217770_218244)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218248 = {}
        # Getting the type of 'call_assignment_217766' (line 174)
        call_assignment_217766_218245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217766', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___218246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), call_assignment_217766_218245, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218249 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218246, *[int_218247], **kwargs_218248)
        
        # Assigning a type to the variable 'call_assignment_217771' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217771', getitem___call_result_218249)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'call_assignment_217771' (line 174)
        call_assignment_217771_218250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217771')
        # Assigning a type to the variable 'descent' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'descent', call_assignment_217771_218250)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218254 = {}
        # Getting the type of 'call_assignment_217766' (line 174)
        call_assignment_217766_218251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217766', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___218252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), call_assignment_217766_218251, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218255 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218252, *[int_218253], **kwargs_218254)
        
        # Assigning a type to the variable 'call_assignment_217772' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217772', getitem___call_result_218255)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'call_assignment_217772' (line 174)
        call_assignment_217772_218256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217772')
        # Assigning a type to the variable 'font_image' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 40), 'font_image', call_assignment_217772_218256)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218260 = {}
        # Getting the type of 'call_assignment_217766' (line 174)
        call_assignment_217766_218257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217766', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___218258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), call_assignment_217766_218257, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218261 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218258, *[int_218259], **kwargs_218260)
        
        # Assigning a type to the variable 'call_assignment_217773' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217773', getitem___call_result_218261)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'call_assignment_217773' (line 174)
        call_assignment_217773_218262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'call_assignment_217773')
        # Assigning a type to the variable 'used_characters' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 52), 'used_characters', call_assignment_217773_218262)
        
        # Assigning a BinOp to a Name (line 177):
        
        # Assigning a BinOp to a Name (line 177):
        # Getting the type of 'descent' (line 177)
        descent_218263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'descent')
        
        # Call to sin(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to radians(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'angle' (line 177)
        angle_218266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'angle', False)
        # Processing the call keyword arguments (line 177)
        kwargs_218267 = {}
        # Getting the type of 'radians' (line 177)
        radians_218265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), 'radians', False)
        # Calling radians(args, kwargs) (line 177)
        radians_call_result_218268 = invoke(stypy.reporting.localization.Localization(__file__, 177, 27), radians_218265, *[angle_218266], **kwargs_218267)
        
        # Processing the call keyword arguments (line 177)
        kwargs_218269 = {}
        # Getting the type of 'sin' (line 177)
        sin_218264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'sin', False)
        # Calling sin(args, kwargs) (line 177)
        sin_call_result_218270 = invoke(stypy.reporting.localization.Localization(__file__, 177, 23), sin_218264, *[radians_call_result_218268], **kwargs_218269)
        
        # Applying the binary operator '*' (line 177)
        result_mul_218271 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 13), '*', descent_218263, sin_call_result_218270)
        
        # Assigning a type to the variable 'xd' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'xd', result_mul_218271)
        
        # Assigning a BinOp to a Name (line 178):
        
        # Assigning a BinOp to a Name (line 178):
        # Getting the type of 'descent' (line 178)
        descent_218272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'descent')
        
        # Call to cos(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Call to radians(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'angle' (line 178)
        angle_218275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'angle', False)
        # Processing the call keyword arguments (line 178)
        kwargs_218276 = {}
        # Getting the type of 'radians' (line 178)
        radians_218274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'radians', False)
        # Calling radians(args, kwargs) (line 178)
        radians_call_result_218277 = invoke(stypy.reporting.localization.Localization(__file__, 178, 27), radians_218274, *[angle_218275], **kwargs_218276)
        
        # Processing the call keyword arguments (line 178)
        kwargs_218278 = {}
        # Getting the type of 'cos' (line 178)
        cos_218273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'cos', False)
        # Calling cos(args, kwargs) (line 178)
        cos_call_result_218279 = invoke(stypy.reporting.localization.Localization(__file__, 178, 23), cos_218273, *[radians_call_result_218277], **kwargs_218278)
        
        # Applying the binary operator '*' (line 178)
        result_mul_218280 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 13), '*', descent_218272, cos_call_result_218279)
        
        # Assigning a type to the variable 'yd' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'yd', result_mul_218280)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to round(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'x' (line 179)
        x_218283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'x', False)
        # Getting the type of 'ox' (line 179)
        ox_218284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'ox', False)
        # Applying the binary operator '+' (line 179)
        result_add_218285 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 21), '+', x_218283, ox_218284)
        
        # Getting the type of 'xd' (line 179)
        xd_218286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 30), 'xd', False)
        # Applying the binary operator '+' (line 179)
        result_add_218287 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 28), '+', result_add_218285, xd_218286)
        
        # Processing the call keyword arguments (line 179)
        kwargs_218288 = {}
        # Getting the type of 'np' (line 179)
        np_218281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'np', False)
        # Obtaining the member 'round' of a type (line 179)
        round_218282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), np_218281, 'round')
        # Calling round(args, kwargs) (line 179)
        round_call_result_218289 = invoke(stypy.reporting.localization.Localization(__file__, 179, 12), round_218282, *[result_add_218287], **kwargs_218288)
        
        # Assigning a type to the variable 'x' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'x', round_call_result_218289)
        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to round(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'y' (line 180)
        y_218292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'y', False)
        # Getting the type of 'oy' (line 180)
        oy_218293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'oy', False)
        # Applying the binary operator '-' (line 180)
        result_sub_218294 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 21), '-', y_218292, oy_218293)
        
        # Getting the type of 'yd' (line 180)
        yd_218295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'yd', False)
        # Applying the binary operator '+' (line 180)
        result_add_218296 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 28), '+', result_sub_218294, yd_218295)
        
        # Processing the call keyword arguments (line 180)
        kwargs_218297 = {}
        # Getting the type of 'np' (line 180)
        np_218290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'np', False)
        # Obtaining the member 'round' of a type (line 180)
        round_218291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), np_218290, 'round')
        # Calling round(args, kwargs) (line 180)
        round_call_result_218298 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), round_218291, *[result_add_218296], **kwargs_218297)
        
        # Assigning a type to the variable 'y' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'y', round_call_result_218298)
        
        # Call to draw_text_image(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'font_image' (line 181)
        font_image_218302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 39), 'font_image', False)
        # Getting the type of 'x' (line 181)
        x_218303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 51), 'x', False)
        # Getting the type of 'y' (line 181)
        y_218304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 54), 'y', False)
        int_218305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 58), 'int')
        # Applying the binary operator '+' (line 181)
        result_add_218306 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 54), '+', y_218304, int_218305)
        
        # Getting the type of 'angle' (line 181)
        angle_218307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 61), 'angle', False)
        # Getting the type of 'gc' (line 181)
        gc_218308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 68), 'gc', False)
        # Processing the call keyword arguments (line 181)
        kwargs_218309 = {}
        # Getting the type of 'self' (line 181)
        self_218299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member '_renderer' of a type (line 181)
        _renderer_218300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_218299, '_renderer')
        # Obtaining the member 'draw_text_image' of a type (line 181)
        draw_text_image_218301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), _renderer_218300, 'draw_text_image')
        # Calling draw_text_image(args, kwargs) (line 181)
        draw_text_image_call_result_218310 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), draw_text_image_218301, *[font_image_218302, x_218303, result_add_218306, angle_218307, gc_218308], **kwargs_218309)
        
        
        # ################# End of 'draw_mathtext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_mathtext' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_218311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_mathtext'
        return stypy_return_type_218311


    @norecursion
    def draw_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 183)
        False_218312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 57), 'False')
        # Getting the type of 'None' (line 183)
        None_218313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 70), 'None')
        defaults = [False_218312, None_218313]
        # Create a new context for function 'draw_text'
        module_type_store = module_type_store.open_function_context('draw_text', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.draw_text.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.draw_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.draw_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.draw_text.__dict__.__setitem__('stypy_function_name', 'RendererAgg.draw_text')
        RendererAgg.draw_text.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'])
        RendererAgg.draw_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.draw_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.draw_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.draw_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.draw_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.draw_text.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.draw_text', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'], None, None, defaults, varargs, kwargs)

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

        unicode_218314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, (-1)), 'unicode', u'\n        Render the text\n        ')
        
        # Getting the type of 'ismath' (line 187)
        ismath_218315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'ismath')
        # Testing the type of an if condition (line 187)
        if_condition_218316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), ismath_218315)
        # Assigning a type to the variable 'if_condition_218316' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_218316', if_condition_218316)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to draw_mathtext(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'gc' (line 188)
        gc_218319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 38), 'gc', False)
        # Getting the type of 'x' (line 188)
        x_218320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 42), 'x', False)
        # Getting the type of 'y' (line 188)
        y_218321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 45), 'y', False)
        # Getting the type of 's' (line 188)
        s_218322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 48), 's', False)
        # Getting the type of 'prop' (line 188)
        prop_218323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 51), 'prop', False)
        # Getting the type of 'angle' (line 188)
        angle_218324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 57), 'angle', False)
        # Processing the call keyword arguments (line 188)
        kwargs_218325 = {}
        # Getting the type of 'self' (line 188)
        self_218317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'self', False)
        # Obtaining the member 'draw_mathtext' of a type (line 188)
        draw_mathtext_218318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 19), self_218317, 'draw_mathtext')
        # Calling draw_mathtext(args, kwargs) (line 188)
        draw_mathtext_call_result_218326 = invoke(stypy.reporting.localization.Localization(__file__, 188, 19), draw_mathtext_218318, *[gc_218319, x_218320, y_218321, s_218322, prop_218323, angle_218324], **kwargs_218325)
        
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'stypy_return_type', draw_mathtext_call_result_218326)
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to get_hinting_flag(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_218328 = {}
        # Getting the type of 'get_hinting_flag' (line 190)
        get_hinting_flag_218327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'get_hinting_flag', False)
        # Calling get_hinting_flag(args, kwargs) (line 190)
        get_hinting_flag_call_result_218329 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), get_hinting_flag_218327, *[], **kwargs_218328)
        
        # Assigning a type to the variable 'flags' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'flags', get_hinting_flag_call_result_218329)
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to _get_agg_font(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'prop' (line 191)
        prop_218332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 34), 'prop', False)
        # Processing the call keyword arguments (line 191)
        kwargs_218333 = {}
        # Getting the type of 'self' (line 191)
        self_218330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'self', False)
        # Obtaining the member '_get_agg_font' of a type (line 191)
        _get_agg_font_218331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 15), self_218330, '_get_agg_font')
        # Calling _get_agg_font(args, kwargs) (line 191)
        _get_agg_font_call_result_218334 = invoke(stypy.reporting.localization.Localization(__file__, 191, 15), _get_agg_font_218331, *[prop_218332], **kwargs_218333)
        
        # Assigning a type to the variable 'font' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'font', _get_agg_font_call_result_218334)
        
        # Type idiom detected: calculating its left and rigth part (line 193)
        # Getting the type of 'font' (line 193)
        font_218335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'font')
        # Getting the type of 'None' (line 193)
        None_218336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'None')
        
        (may_be_218337, more_types_in_union_218338) = may_be_none(font_218335, None_218336)

        if may_be_218337:

            if more_types_in_union_218338:
                # Runtime conditional SSA (line 193)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 194)
            None_218339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'stypy_return_type', None_218339)

            if more_types_in_union_218338:
                # SSA join for if statement (line 193)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 's' (line 195)
        s_218341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 's', False)
        # Processing the call keyword arguments (line 195)
        kwargs_218342 = {}
        # Getting the type of 'len' (line 195)
        len_218340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'len', False)
        # Calling len(args, kwargs) (line 195)
        len_call_result_218343 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), len_218340, *[s_218341], **kwargs_218342)
        
        int_218344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 21), 'int')
        # Applying the binary operator '==' (line 195)
        result_eq_218345 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 11), '==', len_call_result_218343, int_218344)
        
        
        
        # Call to ord(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 's' (line 195)
        s_218347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 31), 's', False)
        # Processing the call keyword arguments (line 195)
        kwargs_218348 = {}
        # Getting the type of 'ord' (line 195)
        ord_218346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'ord', False)
        # Calling ord(args, kwargs) (line 195)
        ord_call_result_218349 = invoke(stypy.reporting.localization.Localization(__file__, 195, 27), ord_218346, *[s_218347], **kwargs_218348)
        
        int_218350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 36), 'int')
        # Applying the binary operator '>' (line 195)
        result_gt_218351 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 27), '>', ord_call_result_218349, int_218350)
        
        # Applying the binary operator 'and' (line 195)
        result_and_keyword_218352 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 11), 'and', result_eq_218345, result_gt_218351)
        
        # Testing the type of an if condition (line 195)
        if_condition_218353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), result_and_keyword_218352)
        # Assigning a type to the variable 'if_condition_218353' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'if_condition_218353', if_condition_218353)
        # SSA begins for if statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to load_char(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to ord(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 's' (line 196)
        s_218357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 's', False)
        # Processing the call keyword arguments (line 196)
        kwargs_218358 = {}
        # Getting the type of 'ord' (line 196)
        ord_218356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'ord', False)
        # Calling ord(args, kwargs) (line 196)
        ord_call_result_218359 = invoke(stypy.reporting.localization.Localization(__file__, 196, 27), ord_218356, *[s_218357], **kwargs_218358)
        
        # Processing the call keyword arguments (line 196)
        # Getting the type of 'flags' (line 196)
        flags_218360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 41), 'flags', False)
        keyword_218361 = flags_218360
        kwargs_218362 = {'flags': keyword_218361}
        # Getting the type of 'font' (line 196)
        font_218354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'font', False)
        # Obtaining the member 'load_char' of a type (line 196)
        load_char_218355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), font_218354, 'load_char')
        # Calling load_char(args, kwargs) (line 196)
        load_char_call_result_218363 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), load_char_218355, *[ord_call_result_218359], **kwargs_218362)
        
        # SSA branch for the else part of an if statement (line 195)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_text(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 's' (line 200)
        s_218366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 's', False)
        int_218367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'int')
        # Processing the call keyword arguments (line 200)
        # Getting the type of 'flags' (line 200)
        flags_218368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 38), 'flags', False)
        keyword_218369 = flags_218368
        kwargs_218370 = {'flags': keyword_218369}
        # Getting the type of 'font' (line 200)
        font_218364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'font', False)
        # Obtaining the member 'set_text' of a type (line 200)
        set_text_218365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), font_218364, 'set_text')
        # Calling set_text(args, kwargs) (line 200)
        set_text_call_result_218371 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), set_text_218365, *[s_218366, int_218367], **kwargs_218370)
        
        # SSA join for if statement (line 195)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_glyphs_to_bitmap(...): (line 201)
        # Processing the call keyword arguments (line 201)
        
        # Obtaining the type of the subscript
        unicode_218374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 56), 'unicode', u'text.antialiased')
        # Getting the type of 'rcParams' (line 201)
        rcParams_218375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 47), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___218376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 47), rcParams_218375, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_218377 = invoke(stypy.reporting.localization.Localization(__file__, 201, 47), getitem___218376, unicode_218374)
        
        keyword_218378 = subscript_call_result_218377
        kwargs_218379 = {'antialiased': keyword_218378}
        # Getting the type of 'font' (line 201)
        font_218372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'font', False)
        # Obtaining the member 'draw_glyphs_to_bitmap' of a type (line 201)
        draw_glyphs_to_bitmap_218373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), font_218372, 'draw_glyphs_to_bitmap')
        # Calling draw_glyphs_to_bitmap(args, kwargs) (line 201)
        draw_glyphs_to_bitmap_call_result_218380 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), draw_glyphs_to_bitmap_218373, *[], **kwargs_218379)
        
        
        # Assigning a BinOp to a Name (line 202):
        
        # Assigning a BinOp to a Name (line 202):
        
        # Call to get_descent(...): (line 202)
        # Processing the call keyword arguments (line 202)
        kwargs_218383 = {}
        # Getting the type of 'font' (line 202)
        font_218381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'font', False)
        # Obtaining the member 'get_descent' of a type (line 202)
        get_descent_218382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), font_218381, 'get_descent')
        # Calling get_descent(args, kwargs) (line 202)
        get_descent_call_result_218384 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), get_descent_218382, *[], **kwargs_218383)
        
        float_218385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 33), 'float')
        # Applying the binary operator 'div' (line 202)
        result_div_218386 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 12), 'div', get_descent_call_result_218384, float_218385)
        
        # Assigning a type to the variable 'd' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'd', result_div_218386)
        
        # Assigning a Call to a Tuple (line 204):
        
        # Assigning a Call to a Name:
        
        # Call to get_bitmap_offset(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_218389 = {}
        # Getting the type of 'font' (line 204)
        font_218387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 17), 'font', False)
        # Obtaining the member 'get_bitmap_offset' of a type (line 204)
        get_bitmap_offset_218388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 17), font_218387, 'get_bitmap_offset')
        # Calling get_bitmap_offset(args, kwargs) (line 204)
        get_bitmap_offset_call_result_218390 = invoke(stypy.reporting.localization.Localization(__file__, 204, 17), get_bitmap_offset_218388, *[], **kwargs_218389)
        
        # Assigning a type to the variable 'call_assignment_217774' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'call_assignment_217774', get_bitmap_offset_call_result_218390)
        
        # Assigning a Call to a Name (line 204):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218394 = {}
        # Getting the type of 'call_assignment_217774' (line 204)
        call_assignment_217774_218391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'call_assignment_217774', False)
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___218392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), call_assignment_217774_218391, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218395 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218392, *[int_218393], **kwargs_218394)
        
        # Assigning a type to the variable 'call_assignment_217775' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'call_assignment_217775', getitem___call_result_218395)
        
        # Assigning a Name to a Name (line 204):
        # Getting the type of 'call_assignment_217775' (line 204)
        call_assignment_217775_218396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'call_assignment_217775')
        # Assigning a type to the variable 'xo' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'xo', call_assignment_217775_218396)
        
        # Assigning a Call to a Name (line 204):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218400 = {}
        # Getting the type of 'call_assignment_217774' (line 204)
        call_assignment_217774_218397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'call_assignment_217774', False)
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___218398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), call_assignment_217774_218397, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218401 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218398, *[int_218399], **kwargs_218400)
        
        # Assigning a type to the variable 'call_assignment_217776' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'call_assignment_217776', getitem___call_result_218401)
        
        # Assigning a Name to a Name (line 204):
        # Getting the type of 'call_assignment_217776' (line 204)
        call_assignment_217776_218402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'call_assignment_217776')
        # Assigning a type to the variable 'yo' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'yo', call_assignment_217776_218402)
        
        # Getting the type of 'xo' (line 205)
        xo_218403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'xo')
        float_218404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 14), 'float')
        # Applying the binary operator 'div=' (line 205)
        result_div_218405 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 8), 'div=', xo_218403, float_218404)
        # Assigning a type to the variable 'xo' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'xo', result_div_218405)
        
        
        # Getting the type of 'yo' (line 206)
        yo_218406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'yo')
        float_218407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 14), 'float')
        # Applying the binary operator 'div=' (line 206)
        result_div_218408 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 8), 'div=', yo_218406, float_218407)
        # Assigning a type to the variable 'yo' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'yo', result_div_218408)
        
        
        # Assigning a BinOp to a Name (line 207):
        
        # Assigning a BinOp to a Name (line 207):
        
        # Getting the type of 'd' (line 207)
        d_218409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 14), 'd')
        # Applying the 'usub' unary operator (line 207)
        result___neg___218410 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 13), 'usub', d_218409)
        
        
        # Call to sin(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Call to radians(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'angle' (line 207)
        angle_218413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 30), 'angle', False)
        # Processing the call keyword arguments (line 207)
        kwargs_218414 = {}
        # Getting the type of 'radians' (line 207)
        radians_218412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'radians', False)
        # Calling radians(args, kwargs) (line 207)
        radians_call_result_218415 = invoke(stypy.reporting.localization.Localization(__file__, 207, 22), radians_218412, *[angle_218413], **kwargs_218414)
        
        # Processing the call keyword arguments (line 207)
        kwargs_218416 = {}
        # Getting the type of 'sin' (line 207)
        sin_218411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 18), 'sin', False)
        # Calling sin(args, kwargs) (line 207)
        sin_call_result_218417 = invoke(stypy.reporting.localization.Localization(__file__, 207, 18), sin_218411, *[radians_call_result_218415], **kwargs_218416)
        
        # Applying the binary operator '*' (line 207)
        result_mul_218418 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 13), '*', result___neg___218410, sin_call_result_218417)
        
        # Assigning a type to the variable 'xd' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'xd', result_mul_218418)
        
        # Assigning a BinOp to a Name (line 208):
        
        # Assigning a BinOp to a Name (line 208):
        # Getting the type of 'd' (line 208)
        d_218419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 13), 'd')
        
        # Call to cos(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Call to radians(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'angle' (line 208)
        angle_218422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'angle', False)
        # Processing the call keyword arguments (line 208)
        kwargs_218423 = {}
        # Getting the type of 'radians' (line 208)
        radians_218421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 21), 'radians', False)
        # Calling radians(args, kwargs) (line 208)
        radians_call_result_218424 = invoke(stypy.reporting.localization.Localization(__file__, 208, 21), radians_218421, *[angle_218422], **kwargs_218423)
        
        # Processing the call keyword arguments (line 208)
        kwargs_218425 = {}
        # Getting the type of 'cos' (line 208)
        cos_218420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), 'cos', False)
        # Calling cos(args, kwargs) (line 208)
        cos_call_result_218426 = invoke(stypy.reporting.localization.Localization(__file__, 208, 17), cos_218420, *[radians_call_result_218424], **kwargs_218425)
        
        # Applying the binary operator '*' (line 208)
        result_mul_218427 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 13), '*', d_218419, cos_call_result_218426)
        
        # Assigning a type to the variable 'yd' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'yd', result_mul_218427)
        
        # Call to draw_text_image(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'font' (line 212)
        font_218431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'font', False)
        
        # Call to round(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'x' (line 212)
        x_218434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'x', False)
        # Getting the type of 'xd' (line 212)
        xd_218435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'xd', False)
        # Applying the binary operator '-' (line 212)
        result_sub_218436 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 27), '-', x_218434, xd_218435)
        
        # Getting the type of 'xo' (line 212)
        xo_218437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'xo', False)
        # Applying the binary operator '+' (line 212)
        result_add_218438 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 34), '+', result_sub_218436, xo_218437)
        
        # Processing the call keyword arguments (line 212)
        kwargs_218439 = {}
        # Getting the type of 'np' (line 212)
        np_218432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 18), 'np', False)
        # Obtaining the member 'round' of a type (line 212)
        round_218433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 18), np_218432, 'round')
        # Calling round(args, kwargs) (line 212)
        round_call_result_218440 = invoke(stypy.reporting.localization.Localization(__file__, 212, 18), round_218433, *[result_add_218438], **kwargs_218439)
        
        
        # Call to round(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'y' (line 212)
        y_218443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 50), 'y', False)
        # Getting the type of 'yd' (line 212)
        yd_218444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 54), 'yd', False)
        # Applying the binary operator '+' (line 212)
        result_add_218445 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 50), '+', y_218443, yd_218444)
        
        # Getting the type of 'yo' (line 212)
        yo_218446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 59), 'yo', False)
        # Applying the binary operator '+' (line 212)
        result_add_218447 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 57), '+', result_add_218445, yo_218446)
        
        # Processing the call keyword arguments (line 212)
        kwargs_218448 = {}
        # Getting the type of 'np' (line 212)
        np_218441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'np', False)
        # Obtaining the member 'round' of a type (line 212)
        round_218442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 41), np_218441, 'round')
        # Calling round(args, kwargs) (line 212)
        round_call_result_218449 = invoke(stypy.reporting.localization.Localization(__file__, 212, 41), round_218442, *[result_add_218447], **kwargs_218448)
        
        int_218450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 65), 'int')
        # Applying the binary operator '+' (line 212)
        result_add_218451 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 41), '+', round_call_result_218449, int_218450)
        
        # Getting the type of 'angle' (line 212)
        angle_218452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 68), 'angle', False)
        # Getting the type of 'gc' (line 212)
        gc_218453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 75), 'gc', False)
        # Processing the call keyword arguments (line 211)
        kwargs_218454 = {}
        # Getting the type of 'self' (line 211)
        self_218428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self', False)
        # Obtaining the member '_renderer' of a type (line 211)
        _renderer_218429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_218428, '_renderer')
        # Obtaining the member 'draw_text_image' of a type (line 211)
        draw_text_image_218430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), _renderer_218429, 'draw_text_image')
        # Calling draw_text_image(args, kwargs) (line 211)
        draw_text_image_call_result_218455 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), draw_text_image_218430, *[font_218431, round_call_result_218440, result_add_218451, angle_218452, gc_218453], **kwargs_218454)
        
        
        # ################# End of 'draw_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_text' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_218456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218456)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_text'
        return stypy_return_type_218456


    @norecursion
    def get_text_width_height_descent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_text_width_height_descent'
        module_type_store = module_type_store.open_function_context('get_text_width_height_descent', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_function_name', 'RendererAgg.get_text_width_height_descent')
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_param_names_list', ['s', 'prop', 'ismath'])
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.get_text_width_height_descent.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.get_text_width_height_descent', ['s', 'prop', 'ismath'], None, None, defaults, varargs, kwargs)

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

        unicode_218457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'unicode', u'\n        Get the width, height, and descent (offset from the bottom\n        to the baseline), in display coords, of the string *s* with\n        :class:`~matplotlib.font_manager.FontProperties` *prop*\n        ')
        
        
        # Obtaining the type of the subscript
        unicode_218458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 20), 'unicode', u'text.usetex')
        # Getting the type of 'rcParams' (line 220)
        rcParams_218459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___218460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 11), rcParams_218459, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_218461 = invoke(stypy.reporting.localization.Localization(__file__, 220, 11), getitem___218460, unicode_218458)
        
        # Testing the type of an if condition (line 220)
        if_condition_218462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 8), subscript_call_result_218461)
        # Assigning a type to the variable 'if_condition_218462' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'if_condition_218462', if_condition_218462)
        # SSA begins for if statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to get_size_in_points(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_218465 = {}
        # Getting the type of 'prop' (line 222)
        prop_218463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 222)
        get_size_in_points_218464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 19), prop_218463, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 222)
        get_size_in_points_call_result_218466 = invoke(stypy.reporting.localization.Localization(__file__, 222, 19), get_size_in_points_218464, *[], **kwargs_218465)
        
        # Assigning a type to the variable 'size' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'size', get_size_in_points_call_result_218466)
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to get_texmanager(...): (line 223)
        # Processing the call keyword arguments (line 223)
        kwargs_218469 = {}
        # Getting the type of 'self' (line 223)
        self_218467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'self', False)
        # Obtaining the member 'get_texmanager' of a type (line 223)
        get_texmanager_218468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), self_218467, 'get_texmanager')
        # Calling get_texmanager(args, kwargs) (line 223)
        get_texmanager_call_result_218470 = invoke(stypy.reporting.localization.Localization(__file__, 223, 25), get_texmanager_218468, *[], **kwargs_218469)
        
        # Assigning a type to the variable 'texmanager' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'texmanager', get_texmanager_call_result_218470)
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to get_size_in_points(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_218473 = {}
        # Getting the type of 'prop' (line 224)
        prop_218471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 224)
        get_size_in_points_218472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 23), prop_218471, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 224)
        get_size_in_points_call_result_218474 = invoke(stypy.reporting.localization.Localization(__file__, 224, 23), get_size_in_points_218472, *[], **kwargs_218473)
        
        # Assigning a type to the variable 'fontsize' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'fontsize', get_size_in_points_call_result_218474)
        
        # Assigning a Call to a Tuple (line 225):
        
        # Assigning a Call to a Name:
        
        # Call to get_text_width_height_descent(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 's' (line 226)
        s_218477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 's', False)
        # Getting the type of 'fontsize' (line 226)
        fontsize_218478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'fontsize', False)
        # Processing the call keyword arguments (line 225)
        # Getting the type of 'self' (line 226)
        self_218479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 38), 'self', False)
        keyword_218480 = self_218479
        kwargs_218481 = {'renderer': keyword_218480}
        # Getting the type of 'texmanager' (line 225)
        texmanager_218475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 22), 'texmanager', False)
        # Obtaining the member 'get_text_width_height_descent' of a type (line 225)
        get_text_width_height_descent_218476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 22), texmanager_218475, 'get_text_width_height_descent')
        # Calling get_text_width_height_descent(args, kwargs) (line 225)
        get_text_width_height_descent_call_result_218482 = invoke(stypy.reporting.localization.Localization(__file__, 225, 22), get_text_width_height_descent_218476, *[s_218477, fontsize_218478], **kwargs_218481)
        
        # Assigning a type to the variable 'call_assignment_217777' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217777', get_text_width_height_descent_call_result_218482)
        
        # Assigning a Call to a Name (line 225):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218486 = {}
        # Getting the type of 'call_assignment_217777' (line 225)
        call_assignment_217777_218483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217777', False)
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___218484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), call_assignment_217777_218483, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218487 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218484, *[int_218485], **kwargs_218486)
        
        # Assigning a type to the variable 'call_assignment_217778' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217778', getitem___call_result_218487)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'call_assignment_217778' (line 225)
        call_assignment_217778_218488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217778')
        # Assigning a type to the variable 'w' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'w', call_assignment_217778_218488)
        
        # Assigning a Call to a Name (line 225):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218492 = {}
        # Getting the type of 'call_assignment_217777' (line 225)
        call_assignment_217777_218489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217777', False)
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___218490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), call_assignment_217777_218489, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218493 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218490, *[int_218491], **kwargs_218492)
        
        # Assigning a type to the variable 'call_assignment_217779' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217779', getitem___call_result_218493)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'call_assignment_217779' (line 225)
        call_assignment_217779_218494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217779')
        # Assigning a type to the variable 'h' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'h', call_assignment_217779_218494)
        
        # Assigning a Call to a Name (line 225):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218498 = {}
        # Getting the type of 'call_assignment_217777' (line 225)
        call_assignment_217777_218495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217777', False)
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___218496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), call_assignment_217777_218495, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218499 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218496, *[int_218497], **kwargs_218498)
        
        # Assigning a type to the variable 'call_assignment_217780' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217780', getitem___call_result_218499)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'call_assignment_217780' (line 225)
        call_assignment_217780_218500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_217780')
        # Assigning a type to the variable 'd' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 18), 'd', call_assignment_217780_218500)
        
        # Obtaining an instance of the builtin type 'tuple' (line 227)
        tuple_218501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 227)
        # Adding element type (line 227)
        # Getting the type of 'w' (line 227)
        w_218502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 19), tuple_218501, w_218502)
        # Adding element type (line 227)
        # Getting the type of 'h' (line 227)
        h_218503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 19), tuple_218501, h_218503)
        # Adding element type (line 227)
        # Getting the type of 'd' (line 227)
        d_218504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 19), tuple_218501, d_218504)
        
        # Assigning a type to the variable 'stypy_return_type' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'stypy_return_type', tuple_218501)
        # SSA join for if statement (line 220)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'ismath' (line 229)
        ismath_218505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'ismath')
        # Testing the type of an if condition (line 229)
        if_condition_218506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), ismath_218505)
        # Assigning a type to the variable 'if_condition_218506' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_218506', if_condition_218506)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 230):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 's' (line 231)
        s_218510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 's', False)
        # Getting the type of 'self' (line 231)
        self_218511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 46), 'self', False)
        # Obtaining the member 'dpi' of a type (line 231)
        dpi_218512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 46), self_218511, 'dpi')
        # Getting the type of 'prop' (line 231)
        prop_218513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 56), 'prop', False)
        # Processing the call keyword arguments (line 231)
        kwargs_218514 = {}
        # Getting the type of 'self' (line 231)
        self_218507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'self', False)
        # Obtaining the member 'mathtext_parser' of a type (line 231)
        mathtext_parser_218508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 16), self_218507, 'mathtext_parser')
        # Obtaining the member 'parse' of a type (line 231)
        parse_218509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 16), mathtext_parser_218508, 'parse')
        # Calling parse(args, kwargs) (line 231)
        parse_call_result_218515 = invoke(stypy.reporting.localization.Localization(__file__, 231, 16), parse_218509, *[s_218510, dpi_218512, prop_218513], **kwargs_218514)
        
        # Assigning a type to the variable 'call_assignment_217781' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217781', parse_call_result_218515)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218519 = {}
        # Getting the type of 'call_assignment_217781' (line 230)
        call_assignment_217781_218516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217781', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___218517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_217781_218516, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218520 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218517, *[int_218518], **kwargs_218519)
        
        # Assigning a type to the variable 'call_assignment_217782' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217782', getitem___call_result_218520)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_217782' (line 230)
        call_assignment_217782_218521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217782')
        # Assigning a type to the variable 'ox' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'ox', call_assignment_217782_218521)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218525 = {}
        # Getting the type of 'call_assignment_217781' (line 230)
        call_assignment_217781_218522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217781', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___218523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_217781_218522, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218526 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218523, *[int_218524], **kwargs_218525)
        
        # Assigning a type to the variable 'call_assignment_217783' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217783', getitem___call_result_218526)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_217783' (line 230)
        call_assignment_217783_218527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217783')
        # Assigning a type to the variable 'oy' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'oy', call_assignment_217783_218527)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218531 = {}
        # Getting the type of 'call_assignment_217781' (line 230)
        call_assignment_217781_218528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217781', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___218529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_217781_218528, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218532 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218529, *[int_218530], **kwargs_218531)
        
        # Assigning a type to the variable 'call_assignment_217784' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217784', getitem___call_result_218532)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_217784' (line 230)
        call_assignment_217784_218533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217784')
        # Assigning a type to the variable 'width' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'width', call_assignment_217784_218533)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218537 = {}
        # Getting the type of 'call_assignment_217781' (line 230)
        call_assignment_217781_218534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217781', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___218535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_217781_218534, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218538 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218535, *[int_218536], **kwargs_218537)
        
        # Assigning a type to the variable 'call_assignment_217785' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217785', getitem___call_result_218538)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_217785' (line 230)
        call_assignment_217785_218539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217785')
        # Assigning a type to the variable 'height' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 27), 'height', call_assignment_217785_218539)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218543 = {}
        # Getting the type of 'call_assignment_217781' (line 230)
        call_assignment_217781_218540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217781', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___218541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_217781_218540, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218544 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218541, *[int_218542], **kwargs_218543)
        
        # Assigning a type to the variable 'call_assignment_217786' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217786', getitem___call_result_218544)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_217786' (line 230)
        call_assignment_217786_218545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217786')
        # Assigning a type to the variable 'descent' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'descent', call_assignment_217786_218545)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218549 = {}
        # Getting the type of 'call_assignment_217781' (line 230)
        call_assignment_217781_218546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217781', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___218547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_217781_218546, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218550 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218547, *[int_218548], **kwargs_218549)
        
        # Assigning a type to the variable 'call_assignment_217787' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217787', getitem___call_result_218550)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_217787' (line 230)
        call_assignment_217787_218551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217787')
        # Assigning a type to the variable 'fonts' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 44), 'fonts', call_assignment_217787_218551)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_218555 = {}
        # Getting the type of 'call_assignment_217781' (line 230)
        call_assignment_217781_218552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217781', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___218553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_217781_218552, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218556 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218553, *[int_218554], **kwargs_218555)
        
        # Assigning a type to the variable 'call_assignment_217788' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217788', getitem___call_result_218556)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_217788' (line 230)
        call_assignment_217788_218557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_217788')
        # Assigning a type to the variable 'used_characters' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 51), 'used_characters', call_assignment_217788_218557)
        
        # Obtaining an instance of the builtin type 'tuple' (line 232)
        tuple_218558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 232)
        # Adding element type (line 232)
        # Getting the type of 'width' (line 232)
        width_218559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 19), tuple_218558, width_218559)
        # Adding element type (line 232)
        # Getting the type of 'height' (line 232)
        height_218560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 19), tuple_218558, height_218560)
        # Adding element type (line 232)
        # Getting the type of 'descent' (line 232)
        descent_218561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'descent')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 19), tuple_218558, descent_218561)
        
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'stypy_return_type', tuple_218558)
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to get_hinting_flag(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_218563 = {}
        # Getting the type of 'get_hinting_flag' (line 234)
        get_hinting_flag_218562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'get_hinting_flag', False)
        # Calling get_hinting_flag(args, kwargs) (line 234)
        get_hinting_flag_call_result_218564 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), get_hinting_flag_218562, *[], **kwargs_218563)
        
        # Assigning a type to the variable 'flags' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'flags', get_hinting_flag_call_result_218564)
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to _get_agg_font(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'prop' (line 235)
        prop_218567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'prop', False)
        # Processing the call keyword arguments (line 235)
        kwargs_218568 = {}
        # Getting the type of 'self' (line 235)
        self_218565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'self', False)
        # Obtaining the member '_get_agg_font' of a type (line 235)
        _get_agg_font_218566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 15), self_218565, '_get_agg_font')
        # Calling _get_agg_font(args, kwargs) (line 235)
        _get_agg_font_call_result_218569 = invoke(stypy.reporting.localization.Localization(__file__, 235, 15), _get_agg_font_218566, *[prop_218567], **kwargs_218568)
        
        # Assigning a type to the variable 'font' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'font', _get_agg_font_call_result_218569)
        
        # Call to set_text(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 's' (line 236)
        s_218572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 's', False)
        float_218573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 25), 'float')
        # Processing the call keyword arguments (line 236)
        # Getting the type of 'flags' (line 236)
        flags_218574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 36), 'flags', False)
        keyword_218575 = flags_218574
        kwargs_218576 = {'flags': keyword_218575}
        # Getting the type of 'font' (line 236)
        font_218570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'font', False)
        # Obtaining the member 'set_text' of a type (line 236)
        set_text_218571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), font_218570, 'set_text')
        # Calling set_text(args, kwargs) (line 236)
        set_text_call_result_218577 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), set_text_218571, *[s_218572, float_218573], **kwargs_218576)
        
        
        # Assigning a Call to a Tuple (line 237):
        
        # Assigning a Call to a Name:
        
        # Call to get_width_height(...): (line 237)
        # Processing the call keyword arguments (line 237)
        kwargs_218580 = {}
        # Getting the type of 'font' (line 237)
        font_218578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'font', False)
        # Obtaining the member 'get_width_height' of a type (line 237)
        get_width_height_218579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 15), font_218578, 'get_width_height')
        # Calling get_width_height(args, kwargs) (line 237)
        get_width_height_call_result_218581 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), get_width_height_218579, *[], **kwargs_218580)
        
        # Assigning a type to the variable 'call_assignment_217789' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'call_assignment_217789', get_width_height_call_result_218581)
        
        # Assigning a Call to a Name (line 237):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218585 = {}
        # Getting the type of 'call_assignment_217789' (line 237)
        call_assignment_217789_218582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'call_assignment_217789', False)
        # Obtaining the member '__getitem__' of a type (line 237)
        getitem___218583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), call_assignment_217789_218582, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218586 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218583, *[int_218584], **kwargs_218585)
        
        # Assigning a type to the variable 'call_assignment_217790' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'call_assignment_217790', getitem___call_result_218586)
        
        # Assigning a Name to a Name (line 237):
        # Getting the type of 'call_assignment_217790' (line 237)
        call_assignment_217790_218587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'call_assignment_217790')
        # Assigning a type to the variable 'w' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'w', call_assignment_217790_218587)
        
        # Assigning a Call to a Name (line 237):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218591 = {}
        # Getting the type of 'call_assignment_217789' (line 237)
        call_assignment_217789_218588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'call_assignment_217789', False)
        # Obtaining the member '__getitem__' of a type (line 237)
        getitem___218589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), call_assignment_217789_218588, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218592 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218589, *[int_218590], **kwargs_218591)
        
        # Assigning a type to the variable 'call_assignment_217791' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'call_assignment_217791', getitem___call_result_218592)
        
        # Assigning a Name to a Name (line 237):
        # Getting the type of 'call_assignment_217791' (line 237)
        call_assignment_217791_218593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'call_assignment_217791')
        # Assigning a type to the variable 'h' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'h', call_assignment_217791_218593)
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to get_descent(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_218596 = {}
        # Getting the type of 'font' (line 238)
        font_218594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'font', False)
        # Obtaining the member 'get_descent' of a type (line 238)
        get_descent_218595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), font_218594, 'get_descent')
        # Calling get_descent(args, kwargs) (line 238)
        get_descent_call_result_218597 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), get_descent_218595, *[], **kwargs_218596)
        
        # Assigning a type to the variable 'd' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'd', get_descent_call_result_218597)
        
        # Getting the type of 'w' (line 239)
        w_218598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'w')
        float_218599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 13), 'float')
        # Applying the binary operator 'div=' (line 239)
        result_div_218600 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 8), 'div=', w_218598, float_218599)
        # Assigning a type to the variable 'w' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'w', result_div_218600)
        
        
        # Getting the type of 'h' (line 240)
        h_218601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'h')
        float_218602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 13), 'float')
        # Applying the binary operator 'div=' (line 240)
        result_div_218603 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 8), 'div=', h_218601, float_218602)
        # Assigning a type to the variable 'h' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'h', result_div_218603)
        
        
        # Getting the type of 'd' (line 241)
        d_218604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'd')
        float_218605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 13), 'float')
        # Applying the binary operator 'div=' (line 241)
        result_div_218606 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 8), 'div=', d_218604, float_218605)
        # Assigning a type to the variable 'd' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'd', result_div_218606)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 242)
        tuple_218607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 242)
        # Adding element type (line 242)
        # Getting the type of 'w' (line 242)
        w_218608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 15), tuple_218607, w_218608)
        # Adding element type (line 242)
        # Getting the type of 'h' (line 242)
        h_218609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 18), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 15), tuple_218607, h_218609)
        # Adding element type (line 242)
        # Getting the type of 'd' (line 242)
        d_218610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 15), tuple_218607, d_218610)
        
        # Assigning a type to the variable 'stypy_return_type' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type', tuple_218607)
        
        # ################# End of 'get_text_width_height_descent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text_width_height_descent' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_218611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text_width_height_descent'
        return stypy_return_type_218611


    @norecursion
    def draw_tex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_218612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 56), 'unicode', u'TeX!')
        # Getting the type of 'None' (line 244)
        None_218613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 70), 'None')
        defaults = [unicode_218612, None_218613]
        # Create a new context for function 'draw_tex'
        module_type_store = module_type_store.open_function_context('draw_tex', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_function_name', 'RendererAgg.draw_tex')
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'])
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.draw_tex.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.draw_tex', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_tex', localization, ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_tex(...)' code ##################

        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to get_size_in_points(...): (line 246)
        # Processing the call keyword arguments (line 246)
        kwargs_218616 = {}
        # Getting the type of 'prop' (line 246)
        prop_218614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 246)
        get_size_in_points_218615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), prop_218614, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 246)
        get_size_in_points_call_result_218617 = invoke(stypy.reporting.localization.Localization(__file__, 246, 15), get_size_in_points_218615, *[], **kwargs_218616)
        
        # Assigning a type to the variable 'size' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'size', get_size_in_points_call_result_218617)
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to get_texmanager(...): (line 248)
        # Processing the call keyword arguments (line 248)
        kwargs_218620 = {}
        # Getting the type of 'self' (line 248)
        self_218618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'self', False)
        # Obtaining the member 'get_texmanager' of a type (line 248)
        get_texmanager_218619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 21), self_218618, 'get_texmanager')
        # Calling get_texmanager(args, kwargs) (line 248)
        get_texmanager_call_result_218621 = invoke(stypy.reporting.localization.Localization(__file__, 248, 21), get_texmanager_218619, *[], **kwargs_218620)
        
        # Assigning a type to the variable 'texmanager' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'texmanager', get_texmanager_call_result_218621)
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to get_grey(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 's' (line 250)
        s_218624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 32), 's', False)
        # Getting the type of 'size' (line 250)
        size_218625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 35), 'size', False)
        # Getting the type of 'self' (line 250)
        self_218626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 41), 'self', False)
        # Obtaining the member 'dpi' of a type (line 250)
        dpi_218627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 41), self_218626, 'dpi')
        # Processing the call keyword arguments (line 250)
        kwargs_218628 = {}
        # Getting the type of 'texmanager' (line 250)
        texmanager_218622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'texmanager', False)
        # Obtaining the member 'get_grey' of a type (line 250)
        get_grey_218623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), texmanager_218622, 'get_grey')
        # Calling get_grey(args, kwargs) (line 250)
        get_grey_call_result_218629 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), get_grey_218623, *[s_218624, size_218625, dpi_218627], **kwargs_218628)
        
        # Assigning a type to the variable 'Z' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'Z', get_grey_call_result_218629)
        
        # Assigning a Call to a Name (line 251):
        
        # Assigning a Call to a Name (line 251):
        
        # Call to array(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'Z' (line 251)
        Z_218632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 21), 'Z', False)
        float_218633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'float')
        # Applying the binary operator '*' (line 251)
        result_mul_218634 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 21), '*', Z_218632, float_218633)
        
        # Getting the type of 'np' (line 251)
        np_218635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 32), 'np', False)
        # Obtaining the member 'uint8' of a type (line 251)
        uint8_218636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 32), np_218635, 'uint8')
        # Processing the call keyword arguments (line 251)
        kwargs_218637 = {}
        # Getting the type of 'np' (line 251)
        np_218630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 251)
        array_218631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), np_218630, 'array')
        # Calling array(args, kwargs) (line 251)
        array_call_result_218638 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), array_218631, *[result_mul_218634, uint8_218636], **kwargs_218637)
        
        # Assigning a type to the variable 'Z' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'Z', array_call_result_218638)
        
        # Assigning a Call to a Tuple (line 253):
        
        # Assigning a Call to a Name:
        
        # Call to get_text_width_height_descent(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 's' (line 253)
        s_218641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 53), 's', False)
        # Getting the type of 'prop' (line 253)
        prop_218642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 56), 'prop', False)
        # Getting the type of 'ismath' (line 253)
        ismath_218643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 62), 'ismath', False)
        # Processing the call keyword arguments (line 253)
        kwargs_218644 = {}
        # Getting the type of 'self' (line 253)
        self_218639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 18), 'self', False)
        # Obtaining the member 'get_text_width_height_descent' of a type (line 253)
        get_text_width_height_descent_218640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 18), self_218639, 'get_text_width_height_descent')
        # Calling get_text_width_height_descent(args, kwargs) (line 253)
        get_text_width_height_descent_call_result_218645 = invoke(stypy.reporting.localization.Localization(__file__, 253, 18), get_text_width_height_descent_218640, *[s_218641, prop_218642, ismath_218643], **kwargs_218644)
        
        # Assigning a type to the variable 'call_assignment_217792' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217792', get_text_width_height_descent_call_result_218645)
        
        # Assigning a Call to a Name (line 253):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218649 = {}
        # Getting the type of 'call_assignment_217792' (line 253)
        call_assignment_217792_218646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217792', False)
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___218647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), call_assignment_217792_218646, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218650 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218647, *[int_218648], **kwargs_218649)
        
        # Assigning a type to the variable 'call_assignment_217793' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217793', getitem___call_result_218650)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'call_assignment_217793' (line 253)
        call_assignment_217793_218651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217793')
        # Assigning a type to the variable 'w' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'w', call_assignment_217793_218651)
        
        # Assigning a Call to a Name (line 253):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218655 = {}
        # Getting the type of 'call_assignment_217792' (line 253)
        call_assignment_217792_218652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217792', False)
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___218653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), call_assignment_217792_218652, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218656 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218653, *[int_218654], **kwargs_218655)
        
        # Assigning a type to the variable 'call_assignment_217794' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217794', getitem___call_result_218656)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'call_assignment_217794' (line 253)
        call_assignment_217794_218657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217794')
        # Assigning a type to the variable 'h' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'h', call_assignment_217794_218657)
        
        # Assigning a Call to a Name (line 253):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218661 = {}
        # Getting the type of 'call_assignment_217792' (line 253)
        call_assignment_217792_218658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217792', False)
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___218659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), call_assignment_217792_218658, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218662 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218659, *[int_218660], **kwargs_218661)
        
        # Assigning a type to the variable 'call_assignment_217795' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217795', getitem___call_result_218662)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'call_assignment_217795' (line 253)
        call_assignment_217795_218663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'call_assignment_217795')
        # Assigning a type to the variable 'd' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 14), 'd', call_assignment_217795_218663)
        
        # Assigning a BinOp to a Name (line 254):
        
        # Assigning a BinOp to a Name (line 254):
        # Getting the type of 'd' (line 254)
        d_218664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 13), 'd')
        
        # Call to sin(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Call to radians(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'angle' (line 254)
        angle_218667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 29), 'angle', False)
        # Processing the call keyword arguments (line 254)
        kwargs_218668 = {}
        # Getting the type of 'radians' (line 254)
        radians_218666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'radians', False)
        # Calling radians(args, kwargs) (line 254)
        radians_call_result_218669 = invoke(stypy.reporting.localization.Localization(__file__, 254, 21), radians_218666, *[angle_218667], **kwargs_218668)
        
        # Processing the call keyword arguments (line 254)
        kwargs_218670 = {}
        # Getting the type of 'sin' (line 254)
        sin_218665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'sin', False)
        # Calling sin(args, kwargs) (line 254)
        sin_call_result_218671 = invoke(stypy.reporting.localization.Localization(__file__, 254, 17), sin_218665, *[radians_call_result_218669], **kwargs_218670)
        
        # Applying the binary operator '*' (line 254)
        result_mul_218672 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 13), '*', d_218664, sin_call_result_218671)
        
        # Assigning a type to the variable 'xd' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'xd', result_mul_218672)
        
        # Assigning a BinOp to a Name (line 255):
        
        # Assigning a BinOp to a Name (line 255):
        # Getting the type of 'd' (line 255)
        d_218673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'd')
        
        # Call to cos(...): (line 255)
        # Processing the call arguments (line 255)
        
        # Call to radians(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'angle' (line 255)
        angle_218676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 29), 'angle', False)
        # Processing the call keyword arguments (line 255)
        kwargs_218677 = {}
        # Getting the type of 'radians' (line 255)
        radians_218675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'radians', False)
        # Calling radians(args, kwargs) (line 255)
        radians_call_result_218678 = invoke(stypy.reporting.localization.Localization(__file__, 255, 21), radians_218675, *[angle_218676], **kwargs_218677)
        
        # Processing the call keyword arguments (line 255)
        kwargs_218679 = {}
        # Getting the type of 'cos' (line 255)
        cos_218674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'cos', False)
        # Calling cos(args, kwargs) (line 255)
        cos_call_result_218680 = invoke(stypy.reporting.localization.Localization(__file__, 255, 17), cos_218674, *[radians_call_result_218678], **kwargs_218679)
        
        # Applying the binary operator '*' (line 255)
        result_mul_218681 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 13), '*', d_218673, cos_call_result_218680)
        
        # Assigning a type to the variable 'yd' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'yd', result_mul_218681)
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to round(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'x' (line 256)
        x_218684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 21), 'x', False)
        # Getting the type of 'xd' (line 256)
        xd_218685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 25), 'xd', False)
        # Applying the binary operator '+' (line 256)
        result_add_218686 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 21), '+', x_218684, xd_218685)
        
        # Processing the call keyword arguments (line 256)
        kwargs_218687 = {}
        # Getting the type of 'np' (line 256)
        np_218682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'np', False)
        # Obtaining the member 'round' of a type (line 256)
        round_218683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), np_218682, 'round')
        # Calling round(args, kwargs) (line 256)
        round_call_result_218688 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), round_218683, *[result_add_218686], **kwargs_218687)
        
        # Assigning a type to the variable 'x' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'x', round_call_result_218688)
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to round(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'y' (line 257)
        y_218691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 21), 'y', False)
        # Getting the type of 'yd' (line 257)
        yd_218692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'yd', False)
        # Applying the binary operator '+' (line 257)
        result_add_218693 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 21), '+', y_218691, yd_218692)
        
        # Processing the call keyword arguments (line 257)
        kwargs_218694 = {}
        # Getting the type of 'np' (line 257)
        np_218689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'np', False)
        # Obtaining the member 'round' of a type (line 257)
        round_218690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), np_218689, 'round')
        # Calling round(args, kwargs) (line 257)
        round_call_result_218695 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), round_218690, *[result_add_218693], **kwargs_218694)
        
        # Assigning a type to the variable 'y' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'y', round_call_result_218695)
        
        # Call to draw_text_image(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'Z' (line 259)
        Z_218699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 39), 'Z', False)
        # Getting the type of 'x' (line 259)
        x_218700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 42), 'x', False)
        # Getting the type of 'y' (line 259)
        y_218701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 45), 'y', False)
        # Getting the type of 'angle' (line 259)
        angle_218702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 48), 'angle', False)
        # Getting the type of 'gc' (line 259)
        gc_218703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 55), 'gc', False)
        # Processing the call keyword arguments (line 259)
        kwargs_218704 = {}
        # Getting the type of 'self' (line 259)
        self_218696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member '_renderer' of a type (line 259)
        _renderer_218697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_218696, '_renderer')
        # Obtaining the member 'draw_text_image' of a type (line 259)
        draw_text_image_218698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), _renderer_218697, 'draw_text_image')
        # Calling draw_text_image(args, kwargs) (line 259)
        draw_text_image_call_result_218705 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), draw_text_image_218698, *[Z_218699, x_218700, y_218701, angle_218702, gc_218703], **kwargs_218704)
        
        
        # ################# End of 'draw_tex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_tex' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_218706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218706)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_tex'
        return stypy_return_type_218706


    @norecursion
    def get_canvas_width_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_canvas_width_height'
        module_type_store = module_type_store.open_function_context('get_canvas_width_height', 261, 4, False)
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_function_name', 'RendererAgg.get_canvas_width_height')
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.get_canvas_width_height.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.get_canvas_width_height', [], None, None, defaults, varargs, kwargs)

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

        unicode_218707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 8), 'unicode', u'return the canvas width and height in display coords')
        
        # Obtaining an instance of the builtin type 'tuple' (line 263)
        tuple_218708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 263)
        # Adding element type (line 263)
        # Getting the type of 'self' (line 263)
        self_218709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'self')
        # Obtaining the member 'width' of a type (line 263)
        width_218710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 15), self_218709, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 15), tuple_218708, width_218710)
        # Adding element type (line 263)
        # Getting the type of 'self' (line 263)
        self_218711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'self')
        # Obtaining the member 'height' of a type (line 263)
        height_218712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 27), self_218711, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 15), tuple_218708, height_218712)
        
        # Assigning a type to the variable 'stypy_return_type' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'stypy_return_type', tuple_218708)
        
        # ################# End of 'get_canvas_width_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_canvas_width_height' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_218713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_canvas_width_height'
        return stypy_return_type_218713


    @norecursion
    def _get_agg_font(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_agg_font'
        module_type_store = module_type_store.open_function_context('_get_agg_font', 265, 4, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_function_name', 'RendererAgg._get_agg_font')
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_param_names_list', ['prop'])
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg._get_agg_font.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg._get_agg_font', ['prop'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_agg_font', localization, ['prop'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_agg_font(...)' code ##################

        unicode_218714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, (-1)), 'unicode', u'\n        Get the font for text instance t, cacheing for efficiency\n        ')
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to findfont(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'prop' (line 269)
        prop_218716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'prop', False)
        # Processing the call keyword arguments (line 269)
        kwargs_218717 = {}
        # Getting the type of 'findfont' (line 269)
        findfont_218715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'findfont', False)
        # Calling findfont(args, kwargs) (line 269)
        findfont_call_result_218718 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), findfont_218715, *[prop_218716], **kwargs_218717)
        
        # Assigning a type to the variable 'fname' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'fname', findfont_call_result_218718)
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to get_font(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'fname' (line 271)
        fname_218720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'fname', False)
        # Processing the call keyword arguments (line 270)
        
        # Obtaining the type of the subscript
        unicode_218721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 36), 'unicode', u'text.hinting_factor')
        # Getting the type of 'rcParams' (line 272)
        rcParams_218722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 27), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___218723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 27), rcParams_218722, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_218724 = invoke(stypy.reporting.localization.Localization(__file__, 272, 27), getitem___218723, unicode_218721)
        
        keyword_218725 = subscript_call_result_218724
        kwargs_218726 = {'hinting_factor': keyword_218725}
        # Getting the type of 'get_font' (line 270)
        get_font_218719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'get_font', False)
        # Calling get_font(args, kwargs) (line 270)
        get_font_call_result_218727 = invoke(stypy.reporting.localization.Localization(__file__, 270, 15), get_font_218719, *[fname_218720], **kwargs_218726)
        
        # Assigning a type to the variable 'font' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'font', get_font_call_result_218727)
        
        # Call to clear(...): (line 274)
        # Processing the call keyword arguments (line 274)
        kwargs_218730 = {}
        # Getting the type of 'font' (line 274)
        font_218728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'font', False)
        # Obtaining the member 'clear' of a type (line 274)
        clear_218729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), font_218728, 'clear')
        # Calling clear(args, kwargs) (line 274)
        clear_call_result_218731 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), clear_218729, *[], **kwargs_218730)
        
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to get_size_in_points(...): (line 275)
        # Processing the call keyword arguments (line 275)
        kwargs_218734 = {}
        # Getting the type of 'prop' (line 275)
        prop_218732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 275)
        get_size_in_points_218733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 15), prop_218732, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 275)
        get_size_in_points_call_result_218735 = invoke(stypy.reporting.localization.Localization(__file__, 275, 15), get_size_in_points_218733, *[], **kwargs_218734)
        
        # Assigning a type to the variable 'size' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'size', get_size_in_points_call_result_218735)
        
        # Call to set_size(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'size' (line 276)
        size_218738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'size', False)
        # Getting the type of 'self' (line 276)
        self_218739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'self', False)
        # Obtaining the member 'dpi' of a type (line 276)
        dpi_218740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 28), self_218739, 'dpi')
        # Processing the call keyword arguments (line 276)
        kwargs_218741 = {}
        # Getting the type of 'font' (line 276)
        font_218736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'font', False)
        # Obtaining the member 'set_size' of a type (line 276)
        set_size_218737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), font_218736, 'set_size')
        # Calling set_size(args, kwargs) (line 276)
        set_size_call_result_218742 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), set_size_218737, *[size_218738, dpi_218740], **kwargs_218741)
        
        # Getting the type of 'font' (line 278)
        font_218743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'font')
        # Assigning a type to the variable 'stypy_return_type' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'stypy_return_type', font_218743)
        
        # ################# End of '_get_agg_font(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_agg_font' in the type store
        # Getting the type of 'stypy_return_type' (line 265)
        stypy_return_type_218744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218744)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_agg_font'
        return stypy_return_type_218744


    @norecursion
    def points_to_pixels(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'points_to_pixels'
        module_type_store = module_type_store.open_function_context('points_to_pixels', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_function_name', 'RendererAgg.points_to_pixels')
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_param_names_list', ['points'])
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.points_to_pixels.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.points_to_pixels', ['points'], None, None, defaults, varargs, kwargs)

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

        unicode_218745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, (-1)), 'unicode', u'\n        convert point measures to pixes using dpi and the pixels per\n        inch of the display\n        ')
        # Getting the type of 'points' (line 285)
        points_218746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'points')
        # Getting the type of 'self' (line 285)
        self_218747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), 'self')
        # Obtaining the member 'dpi' of a type (line 285)
        dpi_218748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 22), self_218747, 'dpi')
        # Applying the binary operator '*' (line 285)
        result_mul_218749 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 15), '*', points_218746, dpi_218748)
        
        float_218750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 31), 'float')
        # Applying the binary operator 'div' (line 285)
        result_div_218751 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 30), 'div', result_mul_218749, float_218750)
        
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', result_div_218751)
        
        # ################# End of 'points_to_pixels(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'points_to_pixels' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_218752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218752)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'points_to_pixels'
        return stypy_return_type_218752


    @norecursion
    def tostring_rgb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tostring_rgb'
        module_type_store = module_type_store.open_function_context('tostring_rgb', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_function_name', 'RendererAgg.tostring_rgb')
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.tostring_rgb.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.tostring_rgb', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tostring_rgb', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tostring_rgb(...)' code ##################

        
        # Call to tostring_rgb(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_218756 = {}
        # Getting the type of 'self' (line 288)
        self_218753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'self', False)
        # Obtaining the member '_renderer' of a type (line 288)
        _renderer_218754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), self_218753, '_renderer')
        # Obtaining the member 'tostring_rgb' of a type (line 288)
        tostring_rgb_218755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), _renderer_218754, 'tostring_rgb')
        # Calling tostring_rgb(args, kwargs) (line 288)
        tostring_rgb_call_result_218757 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), tostring_rgb_218755, *[], **kwargs_218756)
        
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', tostring_rgb_call_result_218757)
        
        # ################# End of 'tostring_rgb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tostring_rgb' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_218758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tostring_rgb'
        return stypy_return_type_218758


    @norecursion
    def tostring_argb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tostring_argb'
        module_type_store = module_type_store.open_function_context('tostring_argb', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_function_name', 'RendererAgg.tostring_argb')
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.tostring_argb.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.tostring_argb', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tostring_argb', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tostring_argb(...)' code ##################

        
        # Call to tostring_argb(...): (line 291)
        # Processing the call keyword arguments (line 291)
        kwargs_218762 = {}
        # Getting the type of 'self' (line 291)
        self_218759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'self', False)
        # Obtaining the member '_renderer' of a type (line 291)
        _renderer_218760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 15), self_218759, '_renderer')
        # Obtaining the member 'tostring_argb' of a type (line 291)
        tostring_argb_218761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 15), _renderer_218760, 'tostring_argb')
        # Calling tostring_argb(args, kwargs) (line 291)
        tostring_argb_call_result_218763 = invoke(stypy.reporting.localization.Localization(__file__, 291, 15), tostring_argb_218761, *[], **kwargs_218762)
        
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', tostring_argb_call_result_218763)
        
        # ################# End of 'tostring_argb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tostring_argb' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_218764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tostring_argb'
        return stypy_return_type_218764


    @norecursion
    def buffer_rgba(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'buffer_rgba'
        module_type_store = module_type_store.open_function_context('buffer_rgba', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_function_name', 'RendererAgg.buffer_rgba')
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.buffer_rgba.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.buffer_rgba', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'buffer_rgba', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'buffer_rgba(...)' code ##################

        
        # Call to buffer_rgba(...): (line 294)
        # Processing the call keyword arguments (line 294)
        kwargs_218768 = {}
        # Getting the type of 'self' (line 294)
        self_218765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'self', False)
        # Obtaining the member '_renderer' of a type (line 294)
        _renderer_218766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), self_218765, '_renderer')
        # Obtaining the member 'buffer_rgba' of a type (line 294)
        buffer_rgba_218767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), _renderer_218766, 'buffer_rgba')
        # Calling buffer_rgba(args, kwargs) (line 294)
        buffer_rgba_call_result_218769 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), buffer_rgba_218767, *[], **kwargs_218768)
        
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', buffer_rgba_call_result_218769)
        
        # ################# End of 'buffer_rgba(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'buffer_rgba' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_218770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'buffer_rgba'
        return stypy_return_type_218770


    @norecursion
    def clear(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clear'
        module_type_store = module_type_store.open_function_context('clear', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.clear.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.clear.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.clear.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.clear.__dict__.__setitem__('stypy_function_name', 'RendererAgg.clear')
        RendererAgg.clear.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.clear.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.clear.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.clear.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.clear.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.clear.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.clear.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.clear', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clear', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clear(...)' code ##################

        
        # Call to clear(...): (line 297)
        # Processing the call keyword arguments (line 297)
        kwargs_218774 = {}
        # Getting the type of 'self' (line 297)
        self_218771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'self', False)
        # Obtaining the member '_renderer' of a type (line 297)
        _renderer_218772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), self_218771, '_renderer')
        # Obtaining the member 'clear' of a type (line 297)
        clear_218773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), _renderer_218772, 'clear')
        # Calling clear(args, kwargs) (line 297)
        clear_call_result_218775 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), clear_218773, *[], **kwargs_218774)
        
        
        # ################# End of 'clear(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clear' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_218776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218776)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clear'
        return stypy_return_type_218776


    @norecursion
    def option_image_nocomposite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'option_image_nocomposite'
        module_type_store = module_type_store.open_function_context('option_image_nocomposite', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_function_name', 'RendererAgg.option_image_nocomposite')
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.option_image_nocomposite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.option_image_nocomposite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'option_image_nocomposite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'option_image_nocomposite(...)' code ##################

        # Getting the type of 'True' (line 303)
        True_218777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'stypy_return_type', True_218777)
        
        # ################# End of 'option_image_nocomposite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'option_image_nocomposite' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_218778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218778)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'option_image_nocomposite'
        return stypy_return_type_218778


    @norecursion
    def option_scale_image(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'option_scale_image'
        module_type_store = module_type_store.open_function_context('option_scale_image', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_function_name', 'RendererAgg.option_scale_image')
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.option_scale_image.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.option_scale_image', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'option_scale_image', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'option_scale_image(...)' code ##################

        unicode_218779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'unicode', u"\n        agg backend doesn't support arbitrary scaling of image.\n        ")
        # Getting the type of 'False' (line 309)
        False_218780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'stypy_return_type', False_218780)
        
        # ################# End of 'option_scale_image(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'option_scale_image' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_218781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'option_scale_image'
        return stypy_return_type_218781


    @norecursion
    def restore_region(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 311)
        None_218782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'None')
        # Getting the type of 'None' (line 311)
        None_218783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 51), 'None')
        defaults = [None_218782, None_218783]
        # Create a new context for function 'restore_region'
        module_type_store = module_type_store.open_function_context('restore_region', 311, 4, False)
        # Assigning a type to the variable 'self' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.restore_region.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.restore_region.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.restore_region.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.restore_region.__dict__.__setitem__('stypy_function_name', 'RendererAgg.restore_region')
        RendererAgg.restore_region.__dict__.__setitem__('stypy_param_names_list', ['region', 'bbox', 'xy'])
        RendererAgg.restore_region.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.restore_region.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.restore_region.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.restore_region.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.restore_region.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.restore_region.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.restore_region', ['region', 'bbox', 'xy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'restore_region', localization, ['region', 'bbox', 'xy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'restore_region(...)' code ##################

        unicode_218784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, (-1)), 'unicode', u'\n        Restore the saved region. If bbox (instance of BboxBase, or\n        its extents) is given, only the region specified by the bbox\n        will be restored. *xy* (a tuple of two floasts) optionally\n        specifies the new position (the LLC of the original region,\n        not the LLC of the bbox) where the region will be restored.\n\n        >>> region = renderer.copy_from_bbox()\n        >>> x1, y1, x2, y2 = region.get_extents()\n        >>> renderer.restore_region(region, bbox=(x1+dx, y1, x2, y2),\n        ...                         xy=(x1-dx, y1))\n\n        ')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'bbox' (line 325)
        bbox_218785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'bbox')
        # Getting the type of 'None' (line 325)
        None_218786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 23), 'None')
        # Applying the binary operator 'isnot' (line 325)
        result_is_not_218787 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 11), 'isnot', bbox_218785, None_218786)
        
        
        # Getting the type of 'xy' (line 325)
        xy_218788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 31), 'xy')
        # Getting the type of 'None' (line 325)
        None_218789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 41), 'None')
        # Applying the binary operator 'isnot' (line 325)
        result_is_not_218790 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 31), 'isnot', xy_218788, None_218789)
        
        # Applying the binary operator 'or' (line 325)
        result_or_keyword_218791 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 11), 'or', result_is_not_218787, result_is_not_218790)
        
        # Testing the type of an if condition (line 325)
        if_condition_218792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 8), result_or_keyword_218791)
        # Assigning a type to the variable 'if_condition_218792' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'if_condition_218792', if_condition_218792)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 326)
        # Getting the type of 'bbox' (line 326)
        bbox_218793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'bbox')
        # Getting the type of 'None' (line 326)
        None_218794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 23), 'None')
        
        (may_be_218795, more_types_in_union_218796) = may_be_none(bbox_218793, None_218794)

        if may_be_218795:

            if more_types_in_union_218796:
                # Runtime conditional SSA (line 326)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Tuple (line 327):
            
            # Assigning a Call to a Name:
            
            # Call to get_extents(...): (line 327)
            # Processing the call keyword arguments (line 327)
            kwargs_218799 = {}
            # Getting the type of 'region' (line 327)
            region_218797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 33), 'region', False)
            # Obtaining the member 'get_extents' of a type (line 327)
            get_extents_218798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 33), region_218797, 'get_extents')
            # Calling get_extents(args, kwargs) (line 327)
            get_extents_call_result_218800 = invoke(stypy.reporting.localization.Localization(__file__, 327, 33), get_extents_218798, *[], **kwargs_218799)
            
            # Assigning a type to the variable 'call_assignment_217796' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217796', get_extents_call_result_218800)
            
            # Assigning a Call to a Name (line 327):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_218803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'int')
            # Processing the call keyword arguments
            kwargs_218804 = {}
            # Getting the type of 'call_assignment_217796' (line 327)
            call_assignment_217796_218801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217796', False)
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___218802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), call_assignment_217796_218801, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_218805 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218802, *[int_218803], **kwargs_218804)
            
            # Assigning a type to the variable 'call_assignment_217797' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217797', getitem___call_result_218805)
            
            # Assigning a Name to a Name (line 327):
            # Getting the type of 'call_assignment_217797' (line 327)
            call_assignment_217797_218806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217797')
            # Assigning a type to the variable 'x1' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'x1', call_assignment_217797_218806)
            
            # Assigning a Call to a Name (line 327):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_218809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'int')
            # Processing the call keyword arguments
            kwargs_218810 = {}
            # Getting the type of 'call_assignment_217796' (line 327)
            call_assignment_217796_218807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217796', False)
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___218808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), call_assignment_217796_218807, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_218811 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218808, *[int_218809], **kwargs_218810)
            
            # Assigning a type to the variable 'call_assignment_217798' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217798', getitem___call_result_218811)
            
            # Assigning a Name to a Name (line 327):
            # Getting the type of 'call_assignment_217798' (line 327)
            call_assignment_217798_218812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217798')
            # Assigning a type to the variable 'y1' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 20), 'y1', call_assignment_217798_218812)
            
            # Assigning a Call to a Name (line 327):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_218815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'int')
            # Processing the call keyword arguments
            kwargs_218816 = {}
            # Getting the type of 'call_assignment_217796' (line 327)
            call_assignment_217796_218813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217796', False)
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___218814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), call_assignment_217796_218813, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_218817 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218814, *[int_218815], **kwargs_218816)
            
            # Assigning a type to the variable 'call_assignment_217799' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217799', getitem___call_result_218817)
            
            # Assigning a Name to a Name (line 327):
            # Getting the type of 'call_assignment_217799' (line 327)
            call_assignment_217799_218818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217799')
            # Assigning a type to the variable 'x2' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 24), 'x2', call_assignment_217799_218818)
            
            # Assigning a Call to a Name (line 327):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_218821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'int')
            # Processing the call keyword arguments
            kwargs_218822 = {}
            # Getting the type of 'call_assignment_217796' (line 327)
            call_assignment_217796_218819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217796', False)
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___218820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), call_assignment_217796_218819, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_218823 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218820, *[int_218821], **kwargs_218822)
            
            # Assigning a type to the variable 'call_assignment_217800' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217800', getitem___call_result_218823)
            
            # Assigning a Name to a Name (line 327):
            # Getting the type of 'call_assignment_217800' (line 327)
            call_assignment_217800_218824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'call_assignment_217800')
            # Assigning a type to the variable 'y2' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 28), 'y2', call_assignment_217800_218824)

            if more_types_in_union_218796:
                # Runtime conditional SSA for else branch (line 326)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_218795) or more_types_in_union_218796):
            
            
            # Call to isinstance(...): (line 328)
            # Processing the call arguments (line 328)
            # Getting the type of 'bbox' (line 328)
            bbox_218826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'bbox', False)
            # Getting the type of 'BboxBase' (line 328)
            BboxBase_218827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 34), 'BboxBase', False)
            # Processing the call keyword arguments (line 328)
            kwargs_218828 = {}
            # Getting the type of 'isinstance' (line 328)
            isinstance_218825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 328)
            isinstance_call_result_218829 = invoke(stypy.reporting.localization.Localization(__file__, 328, 17), isinstance_218825, *[bbox_218826, BboxBase_218827], **kwargs_218828)
            
            # Testing the type of an if condition (line 328)
            if_condition_218830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 17), isinstance_call_result_218829)
            # Assigning a type to the variable 'if_condition_218830' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'if_condition_218830', if_condition_218830)
            # SSA begins for if statement (line 328)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Tuple (line 329):
            
            # Assigning a Subscript to a Name (line 329):
            
            # Obtaining the type of the subscript
            int_218831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 16), 'int')
            # Getting the type of 'bbox' (line 329)
            bbox_218832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'bbox')
            # Obtaining the member 'extents' of a type (line 329)
            extents_218833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 33), bbox_218832, 'extents')
            # Obtaining the member '__getitem__' of a type (line 329)
            getitem___218834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), extents_218833, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 329)
            subscript_call_result_218835 = invoke(stypy.reporting.localization.Localization(__file__, 329, 16), getitem___218834, int_218831)
            
            # Assigning a type to the variable 'tuple_var_assignment_217801' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'tuple_var_assignment_217801', subscript_call_result_218835)
            
            # Assigning a Subscript to a Name (line 329):
            
            # Obtaining the type of the subscript
            int_218836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 16), 'int')
            # Getting the type of 'bbox' (line 329)
            bbox_218837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'bbox')
            # Obtaining the member 'extents' of a type (line 329)
            extents_218838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 33), bbox_218837, 'extents')
            # Obtaining the member '__getitem__' of a type (line 329)
            getitem___218839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), extents_218838, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 329)
            subscript_call_result_218840 = invoke(stypy.reporting.localization.Localization(__file__, 329, 16), getitem___218839, int_218836)
            
            # Assigning a type to the variable 'tuple_var_assignment_217802' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'tuple_var_assignment_217802', subscript_call_result_218840)
            
            # Assigning a Subscript to a Name (line 329):
            
            # Obtaining the type of the subscript
            int_218841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 16), 'int')
            # Getting the type of 'bbox' (line 329)
            bbox_218842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'bbox')
            # Obtaining the member 'extents' of a type (line 329)
            extents_218843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 33), bbox_218842, 'extents')
            # Obtaining the member '__getitem__' of a type (line 329)
            getitem___218844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), extents_218843, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 329)
            subscript_call_result_218845 = invoke(stypy.reporting.localization.Localization(__file__, 329, 16), getitem___218844, int_218841)
            
            # Assigning a type to the variable 'tuple_var_assignment_217803' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'tuple_var_assignment_217803', subscript_call_result_218845)
            
            # Assigning a Subscript to a Name (line 329):
            
            # Obtaining the type of the subscript
            int_218846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 16), 'int')
            # Getting the type of 'bbox' (line 329)
            bbox_218847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'bbox')
            # Obtaining the member 'extents' of a type (line 329)
            extents_218848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 33), bbox_218847, 'extents')
            # Obtaining the member '__getitem__' of a type (line 329)
            getitem___218849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), extents_218848, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 329)
            subscript_call_result_218850 = invoke(stypy.reporting.localization.Localization(__file__, 329, 16), getitem___218849, int_218846)
            
            # Assigning a type to the variable 'tuple_var_assignment_217804' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'tuple_var_assignment_217804', subscript_call_result_218850)
            
            # Assigning a Name to a Name (line 329):
            # Getting the type of 'tuple_var_assignment_217801' (line 329)
            tuple_var_assignment_217801_218851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'tuple_var_assignment_217801')
            # Assigning a type to the variable 'x1' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'x1', tuple_var_assignment_217801_218851)
            
            # Assigning a Name to a Name (line 329):
            # Getting the type of 'tuple_var_assignment_217802' (line 329)
            tuple_var_assignment_217802_218852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'tuple_var_assignment_217802')
            # Assigning a type to the variable 'y1' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'y1', tuple_var_assignment_217802_218852)
            
            # Assigning a Name to a Name (line 329):
            # Getting the type of 'tuple_var_assignment_217803' (line 329)
            tuple_var_assignment_217803_218853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'tuple_var_assignment_217803')
            # Assigning a type to the variable 'x2' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 24), 'x2', tuple_var_assignment_217803_218853)
            
            # Assigning a Name to a Name (line 329):
            # Getting the type of 'tuple_var_assignment_217804' (line 329)
            tuple_var_assignment_217804_218854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'tuple_var_assignment_217804')
            # Assigning a type to the variable 'y2' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 28), 'y2', tuple_var_assignment_217804_218854)
            # SSA branch for the else part of an if statement (line 328)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Tuple (line 331):
            
            # Assigning a Subscript to a Name (line 331):
            
            # Obtaining the type of the subscript
            int_218855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 16), 'int')
            # Getting the type of 'bbox' (line 331)
            bbox_218856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'bbox')
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___218857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 16), bbox_218856, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_218858 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), getitem___218857, int_218855)
            
            # Assigning a type to the variable 'tuple_var_assignment_217805' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'tuple_var_assignment_217805', subscript_call_result_218858)
            
            # Assigning a Subscript to a Name (line 331):
            
            # Obtaining the type of the subscript
            int_218859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 16), 'int')
            # Getting the type of 'bbox' (line 331)
            bbox_218860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'bbox')
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___218861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 16), bbox_218860, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_218862 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), getitem___218861, int_218859)
            
            # Assigning a type to the variable 'tuple_var_assignment_217806' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'tuple_var_assignment_217806', subscript_call_result_218862)
            
            # Assigning a Subscript to a Name (line 331):
            
            # Obtaining the type of the subscript
            int_218863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 16), 'int')
            # Getting the type of 'bbox' (line 331)
            bbox_218864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'bbox')
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___218865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 16), bbox_218864, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_218866 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), getitem___218865, int_218863)
            
            # Assigning a type to the variable 'tuple_var_assignment_217807' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'tuple_var_assignment_217807', subscript_call_result_218866)
            
            # Assigning a Subscript to a Name (line 331):
            
            # Obtaining the type of the subscript
            int_218867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 16), 'int')
            # Getting the type of 'bbox' (line 331)
            bbox_218868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'bbox')
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___218869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 16), bbox_218868, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_218870 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), getitem___218869, int_218867)
            
            # Assigning a type to the variable 'tuple_var_assignment_217808' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'tuple_var_assignment_217808', subscript_call_result_218870)
            
            # Assigning a Name to a Name (line 331):
            # Getting the type of 'tuple_var_assignment_217805' (line 331)
            tuple_var_assignment_217805_218871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'tuple_var_assignment_217805')
            # Assigning a type to the variable 'x1' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'x1', tuple_var_assignment_217805_218871)
            
            # Assigning a Name to a Name (line 331):
            # Getting the type of 'tuple_var_assignment_217806' (line 331)
            tuple_var_assignment_217806_218872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'tuple_var_assignment_217806')
            # Assigning a type to the variable 'y1' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'y1', tuple_var_assignment_217806_218872)
            
            # Assigning a Name to a Name (line 331):
            # Getting the type of 'tuple_var_assignment_217807' (line 331)
            tuple_var_assignment_217807_218873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'tuple_var_assignment_217807')
            # Assigning a type to the variable 'x2' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'x2', tuple_var_assignment_217807_218873)
            
            # Assigning a Name to a Name (line 331):
            # Getting the type of 'tuple_var_assignment_217808' (line 331)
            tuple_var_assignment_217808_218874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'tuple_var_assignment_217808')
            # Assigning a type to the variable 'y2' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'y2', tuple_var_assignment_217808_218874)
            # SSA join for if statement (line 328)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_218795 and more_types_in_union_218796):
                # SSA join for if statement (line 326)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 333)
        # Getting the type of 'xy' (line 333)
        xy_218875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'xy')
        # Getting the type of 'None' (line 333)
        None_218876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 21), 'None')
        
        (may_be_218877, more_types_in_union_218878) = may_be_none(xy_218875, None_218876)

        if may_be_218877:

            if more_types_in_union_218878:
                # Runtime conditional SSA (line 333)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Tuple to a Tuple (line 334):
            
            # Assigning a Name to a Name (line 334):
            # Getting the type of 'x1' (line 334)
            x1_218879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 25), 'x1')
            # Assigning a type to the variable 'tuple_assignment_217809' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_assignment_217809', x1_218879)
            
            # Assigning a Name to a Name (line 334):
            # Getting the type of 'y1' (line 334)
            y1_218880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 29), 'y1')
            # Assigning a type to the variable 'tuple_assignment_217810' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_assignment_217810', y1_218880)
            
            # Assigning a Name to a Name (line 334):
            # Getting the type of 'tuple_assignment_217809' (line 334)
            tuple_assignment_217809_218881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_assignment_217809')
            # Assigning a type to the variable 'ox' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'ox', tuple_assignment_217809_218881)
            
            # Assigning a Name to a Name (line 334):
            # Getting the type of 'tuple_assignment_217810' (line 334)
            tuple_assignment_217810_218882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_assignment_217810')
            # Assigning a type to the variable 'oy' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'oy', tuple_assignment_217810_218882)

            if more_types_in_union_218878:
                # Runtime conditional SSA for else branch (line 333)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_218877) or more_types_in_union_218878):
            
            # Assigning a Name to a Tuple (line 336):
            
            # Assigning a Subscript to a Name (line 336):
            
            # Obtaining the type of the subscript
            int_218883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 16), 'int')
            # Getting the type of 'xy' (line 336)
            xy_218884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 25), 'xy')
            # Obtaining the member '__getitem__' of a type (line 336)
            getitem___218885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 16), xy_218884, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 336)
            subscript_call_result_218886 = invoke(stypy.reporting.localization.Localization(__file__, 336, 16), getitem___218885, int_218883)
            
            # Assigning a type to the variable 'tuple_var_assignment_217811' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'tuple_var_assignment_217811', subscript_call_result_218886)
            
            # Assigning a Subscript to a Name (line 336):
            
            # Obtaining the type of the subscript
            int_218887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 16), 'int')
            # Getting the type of 'xy' (line 336)
            xy_218888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 25), 'xy')
            # Obtaining the member '__getitem__' of a type (line 336)
            getitem___218889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 16), xy_218888, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 336)
            subscript_call_result_218890 = invoke(stypy.reporting.localization.Localization(__file__, 336, 16), getitem___218889, int_218887)
            
            # Assigning a type to the variable 'tuple_var_assignment_217812' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'tuple_var_assignment_217812', subscript_call_result_218890)
            
            # Assigning a Name to a Name (line 336):
            # Getting the type of 'tuple_var_assignment_217811' (line 336)
            tuple_var_assignment_217811_218891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'tuple_var_assignment_217811')
            # Assigning a type to the variable 'ox' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'ox', tuple_var_assignment_217811_218891)
            
            # Assigning a Name to a Name (line 336):
            # Getting the type of 'tuple_var_assignment_217812' (line 336)
            tuple_var_assignment_217812_218892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'tuple_var_assignment_217812')
            # Assigning a type to the variable 'oy' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'oy', tuple_var_assignment_217812_218892)

            if (may_be_218877 and more_types_in_union_218878):
                # SSA join for if statement (line 333)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to restore_region(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'region' (line 340)
        region_218896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 42), 'region', False)
        
        # Call to int(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'x1' (line 340)
        x1_218898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 54), 'x1', False)
        # Processing the call keyword arguments (line 340)
        kwargs_218899 = {}
        # Getting the type of 'int' (line 340)
        int_218897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 50), 'int', False)
        # Calling int(args, kwargs) (line 340)
        int_call_result_218900 = invoke(stypy.reporting.localization.Localization(__file__, 340, 50), int_218897, *[x1_218898], **kwargs_218899)
        
        
        # Call to int(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'y1' (line 340)
        y1_218902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 63), 'y1', False)
        # Processing the call keyword arguments (line 340)
        kwargs_218903 = {}
        # Getting the type of 'int' (line 340)
        int_218901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 59), 'int', False)
        # Calling int(args, kwargs) (line 340)
        int_call_result_218904 = invoke(stypy.reporting.localization.Localization(__file__, 340, 59), int_218901, *[y1_218902], **kwargs_218903)
        
        
        # Call to int(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'x2' (line 341)
        x2_218906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 46), 'x2', False)
        # Processing the call keyword arguments (line 341)
        kwargs_218907 = {}
        # Getting the type of 'int' (line 341)
        int_218905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 42), 'int', False)
        # Calling int(args, kwargs) (line 341)
        int_call_result_218908 = invoke(stypy.reporting.localization.Localization(__file__, 341, 42), int_218905, *[x2_218906], **kwargs_218907)
        
        
        # Call to int(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'y2' (line 341)
        y2_218910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 55), 'y2', False)
        # Processing the call keyword arguments (line 341)
        kwargs_218911 = {}
        # Getting the type of 'int' (line 341)
        int_218909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 51), 'int', False)
        # Calling int(args, kwargs) (line 341)
        int_call_result_218912 = invoke(stypy.reporting.localization.Localization(__file__, 341, 51), int_218909, *[y2_218910], **kwargs_218911)
        
        
        # Call to int(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'ox' (line 341)
        ox_218914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 64), 'ox', False)
        # Processing the call keyword arguments (line 341)
        kwargs_218915 = {}
        # Getting the type of 'int' (line 341)
        int_218913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 60), 'int', False)
        # Calling int(args, kwargs) (line 341)
        int_call_result_218916 = invoke(stypy.reporting.localization.Localization(__file__, 341, 60), int_218913, *[ox_218914], **kwargs_218915)
        
        
        # Call to int(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'oy' (line 341)
        oy_218918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 73), 'oy', False)
        # Processing the call keyword arguments (line 341)
        kwargs_218919 = {}
        # Getting the type of 'int' (line 341)
        int_218917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 69), 'int', False)
        # Calling int(args, kwargs) (line 341)
        int_call_result_218920 = invoke(stypy.reporting.localization.Localization(__file__, 341, 69), int_218917, *[oy_218918], **kwargs_218919)
        
        # Processing the call keyword arguments (line 340)
        kwargs_218921 = {}
        # Getting the type of 'self' (line 340)
        self_218893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'self', False)
        # Obtaining the member '_renderer' of a type (line 340)
        _renderer_218894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), self_218893, '_renderer')
        # Obtaining the member 'restore_region' of a type (line 340)
        restore_region_218895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), _renderer_218894, 'restore_region')
        # Calling restore_region(args, kwargs) (line 340)
        restore_region_call_result_218922 = invoke(stypy.reporting.localization.Localization(__file__, 340, 12), restore_region_218895, *[region_218896, int_call_result_218900, int_call_result_218904, int_call_result_218908, int_call_result_218912, int_call_result_218916, int_call_result_218920], **kwargs_218921)
        
        # SSA branch for the else part of an if statement (line 325)
        module_type_store.open_ssa_branch('else')
        
        # Call to restore_region(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'region' (line 344)
        region_218926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 42), 'region', False)
        # Processing the call keyword arguments (line 344)
        kwargs_218927 = {}
        # Getting the type of 'self' (line 344)
        self_218923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'self', False)
        # Obtaining the member '_renderer' of a type (line 344)
        _renderer_218924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), self_218923, '_renderer')
        # Obtaining the member 'restore_region' of a type (line 344)
        restore_region_218925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), _renderer_218924, 'restore_region')
        # Calling restore_region(args, kwargs) (line 344)
        restore_region_call_result_218928 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), restore_region_218925, *[region_218926], **kwargs_218927)
        
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'restore_region(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'restore_region' in the type store
        # Getting the type of 'stypy_return_type' (line 311)
        stypy_return_type_218929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218929)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'restore_region'
        return stypy_return_type_218929


    @norecursion
    def start_filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'start_filter'
        module_type_store = module_type_store.open_function_context('start_filter', 346, 4, False)
        # Assigning a type to the variable 'self' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.start_filter.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.start_filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.start_filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.start_filter.__dict__.__setitem__('stypy_function_name', 'RendererAgg.start_filter')
        RendererAgg.start_filter.__dict__.__setitem__('stypy_param_names_list', [])
        RendererAgg.start_filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.start_filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.start_filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.start_filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.start_filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.start_filter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.start_filter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'start_filter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'start_filter(...)' code ##################

        unicode_218930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, (-1)), 'unicode', u'\n        Start filtering. It simply create a new canvas (the old one is saved).\n        ')
        
        # Call to append(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'self' (line 350)
        self_218934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 38), 'self', False)
        # Obtaining the member '_renderer' of a type (line 350)
        _renderer_218935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 38), self_218934, '_renderer')
        # Processing the call keyword arguments (line 350)
        kwargs_218936 = {}
        # Getting the type of 'self' (line 350)
        self_218931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self', False)
        # Obtaining the member '_filter_renderers' of a type (line 350)
        _filter_renderers_218932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_218931, '_filter_renderers')
        # Obtaining the member 'append' of a type (line 350)
        append_218933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), _filter_renderers_218932, 'append')
        # Calling append(args, kwargs) (line 350)
        append_call_result_218937 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), append_218933, *[_renderer_218935], **kwargs_218936)
        
        
        # Assigning a Call to a Attribute (line 351):
        
        # Assigning a Call to a Attribute (line 351):
        
        # Call to _RendererAgg(...): (line 351)
        # Processing the call arguments (line 351)
        
        # Call to int(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'self' (line 351)
        self_218940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 42), 'self', False)
        # Obtaining the member 'width' of a type (line 351)
        width_218941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 42), self_218940, 'width')
        # Processing the call keyword arguments (line 351)
        kwargs_218942 = {}
        # Getting the type of 'int' (line 351)
        int_218939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 38), 'int', False)
        # Calling int(args, kwargs) (line 351)
        int_call_result_218943 = invoke(stypy.reporting.localization.Localization(__file__, 351, 38), int_218939, *[width_218941], **kwargs_218942)
        
        
        # Call to int(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'self' (line 351)
        self_218945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 59), 'self', False)
        # Obtaining the member 'height' of a type (line 351)
        height_218946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 59), self_218945, 'height')
        # Processing the call keyword arguments (line 351)
        kwargs_218947 = {}
        # Getting the type of 'int' (line 351)
        int_218944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 55), 'int', False)
        # Calling int(args, kwargs) (line 351)
        int_call_result_218948 = invoke(stypy.reporting.localization.Localization(__file__, 351, 55), int_218944, *[height_218946], **kwargs_218947)
        
        # Getting the type of 'self' (line 352)
        self_218949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 38), 'self', False)
        # Obtaining the member 'dpi' of a type (line 352)
        dpi_218950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 38), self_218949, 'dpi')
        # Processing the call keyword arguments (line 351)
        kwargs_218951 = {}
        # Getting the type of '_RendererAgg' (line 351)
        _RendererAgg_218938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 25), '_RendererAgg', False)
        # Calling _RendererAgg(args, kwargs) (line 351)
        _RendererAgg_call_result_218952 = invoke(stypy.reporting.localization.Localization(__file__, 351, 25), _RendererAgg_218938, *[int_call_result_218943, int_call_result_218948, dpi_218950], **kwargs_218951)
        
        # Getting the type of 'self' (line 351)
        self_218953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self')
        # Setting the type of the member '_renderer' of a type (line 351)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), self_218953, '_renderer', _RendererAgg_call_result_218952)
        
        # Call to _update_methods(...): (line 353)
        # Processing the call keyword arguments (line 353)
        kwargs_218956 = {}
        # Getting the type of 'self' (line 353)
        self_218954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'self', False)
        # Obtaining the member '_update_methods' of a type (line 353)
        _update_methods_218955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), self_218954, '_update_methods')
        # Calling _update_methods(args, kwargs) (line 353)
        _update_methods_call_result_218957 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), _update_methods_218955, *[], **kwargs_218956)
        
        
        # ################# End of 'start_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'start_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 346)
        stypy_return_type_218958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218958)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'start_filter'
        return stypy_return_type_218958


    @norecursion
    def stop_filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stop_filter'
        module_type_store = module_type_store.open_function_context('stop_filter', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_localization', localization)
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_function_name', 'RendererAgg.stop_filter')
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_param_names_list', ['post_processing'])
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererAgg.stop_filter.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererAgg.stop_filter', ['post_processing'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stop_filter', localization, ['post_processing'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stop_filter(...)' code ##################

        unicode_218959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, (-1)), 'unicode', u'\n        Save the plot in the current canvas as a image and apply\n        the *post_processing* function.\n\n           def post_processing(image, dpi):\n             # ny, nx, depth = image.shape\n             # image (numpy array) has RGBA channels and has a depth of 4.\n             ...\n             # create a new_image (numpy array of 4 channels, size can be\n             # different). The resulting image may have offsets from\n             # lower-left corner of the original image\n             return new_image, offset_x, offset_y\n\n        The saved renderer is restored and the returned image from\n        post_processing is plotted (using draw_image) on it.\n        ')
        
        # Assigning a Tuple to a Tuple (line 377):
        
        # Assigning a Call to a Name (line 377):
        
        # Call to int(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'self' (line 377)
        self_218961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 28), 'self', False)
        # Obtaining the member 'width' of a type (line 377)
        width_218962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 28), self_218961, 'width')
        # Processing the call keyword arguments (line 377)
        kwargs_218963 = {}
        # Getting the type of 'int' (line 377)
        int_218960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 24), 'int', False)
        # Calling int(args, kwargs) (line 377)
        int_call_result_218964 = invoke(stypy.reporting.localization.Localization(__file__, 377, 24), int_218960, *[width_218962], **kwargs_218963)
        
        # Assigning a type to the variable 'tuple_assignment_217813' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'tuple_assignment_217813', int_call_result_218964)
        
        # Assigning a Call to a Name (line 377):
        
        # Call to int(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'self' (line 377)
        self_218966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 45), 'self', False)
        # Obtaining the member 'height' of a type (line 377)
        height_218967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 45), self_218966, 'height')
        # Processing the call keyword arguments (line 377)
        kwargs_218968 = {}
        # Getting the type of 'int' (line 377)
        int_218965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 41), 'int', False)
        # Calling int(args, kwargs) (line 377)
        int_call_result_218969 = invoke(stypy.reporting.localization.Localization(__file__, 377, 41), int_218965, *[height_218967], **kwargs_218968)
        
        # Assigning a type to the variable 'tuple_assignment_217814' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'tuple_assignment_217814', int_call_result_218969)
        
        # Assigning a Name to a Name (line 377):
        # Getting the type of 'tuple_assignment_217813' (line 377)
        tuple_assignment_217813_218970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'tuple_assignment_217813')
        # Assigning a type to the variable 'width' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'width', tuple_assignment_217813_218970)
        
        # Assigning a Name to a Name (line 377):
        # Getting the type of 'tuple_assignment_217814' (line 377)
        tuple_assignment_217814_218971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'tuple_assignment_217814')
        # Assigning a type to the variable 'height' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'height', tuple_assignment_217814_218971)
        
        # Assigning a Call to a Tuple (line 379):
        
        # Assigning a Call to a Name:
        
        # Call to tostring_rgba_minimized(...): (line 379)
        # Processing the call keyword arguments (line 379)
        kwargs_218974 = {}
        # Getting the type of 'self' (line 379)
        self_218972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 25), 'self', False)
        # Obtaining the member 'tostring_rgba_minimized' of a type (line 379)
        tostring_rgba_minimized_218973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 25), self_218972, 'tostring_rgba_minimized')
        # Calling tostring_rgba_minimized(args, kwargs) (line 379)
        tostring_rgba_minimized_call_result_218975 = invoke(stypy.reporting.localization.Localization(__file__, 379, 25), tostring_rgba_minimized_218973, *[], **kwargs_218974)
        
        # Assigning a type to the variable 'call_assignment_217815' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'call_assignment_217815', tostring_rgba_minimized_call_result_218975)
        
        # Assigning a Call to a Name (line 379):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218979 = {}
        # Getting the type of 'call_assignment_217815' (line 379)
        call_assignment_217815_218976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'call_assignment_217815', False)
        # Obtaining the member '__getitem__' of a type (line 379)
        getitem___218977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), call_assignment_217815_218976, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218980 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218977, *[int_218978], **kwargs_218979)
        
        # Assigning a type to the variable 'call_assignment_217816' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'call_assignment_217816', getitem___call_result_218980)
        
        # Assigning a Name to a Name (line 379):
        # Getting the type of 'call_assignment_217816' (line 379)
        call_assignment_217816_218981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'call_assignment_217816')
        # Assigning a type to the variable 'buffer' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'buffer', call_assignment_217816_218981)
        
        # Assigning a Call to a Name (line 379):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_218984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 8), 'int')
        # Processing the call keyword arguments
        kwargs_218985 = {}
        # Getting the type of 'call_assignment_217815' (line 379)
        call_assignment_217815_218982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'call_assignment_217815', False)
        # Obtaining the member '__getitem__' of a type (line 379)
        getitem___218983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), call_assignment_217815_218982, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_218986 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___218983, *[int_218984], **kwargs_218985)
        
        # Assigning a type to the variable 'call_assignment_217817' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'call_assignment_217817', getitem___call_result_218986)
        
        # Assigning a Name to a Name (line 379):
        # Getting the type of 'call_assignment_217817' (line 379)
        call_assignment_217817_218987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'call_assignment_217817')
        # Assigning a type to the variable 'bounds' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'bounds', call_assignment_217817_218987)
        
        # Assigning a Name to a Tuple (line 381):
        
        # Assigning a Subscript to a Name (line 381):
        
        # Obtaining the type of the subscript
        int_218988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 8), 'int')
        # Getting the type of 'bounds' (line 381)
        bounds_218989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 381)
        getitem___218990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), bounds_218989, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 381)
        subscript_call_result_218991 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), getitem___218990, int_218988)
        
        # Assigning a type to the variable 'tuple_var_assignment_217818' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'tuple_var_assignment_217818', subscript_call_result_218991)
        
        # Assigning a Subscript to a Name (line 381):
        
        # Obtaining the type of the subscript
        int_218992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 8), 'int')
        # Getting the type of 'bounds' (line 381)
        bounds_218993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 381)
        getitem___218994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), bounds_218993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 381)
        subscript_call_result_218995 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), getitem___218994, int_218992)
        
        # Assigning a type to the variable 'tuple_var_assignment_217819' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'tuple_var_assignment_217819', subscript_call_result_218995)
        
        # Assigning a Subscript to a Name (line 381):
        
        # Obtaining the type of the subscript
        int_218996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 8), 'int')
        # Getting the type of 'bounds' (line 381)
        bounds_218997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 381)
        getitem___218998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), bounds_218997, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 381)
        subscript_call_result_218999 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), getitem___218998, int_218996)
        
        # Assigning a type to the variable 'tuple_var_assignment_217820' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'tuple_var_assignment_217820', subscript_call_result_218999)
        
        # Assigning a Subscript to a Name (line 381):
        
        # Obtaining the type of the subscript
        int_219000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 8), 'int')
        # Getting the type of 'bounds' (line 381)
        bounds_219001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 381)
        getitem___219002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), bounds_219001, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 381)
        subscript_call_result_219003 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), getitem___219002, int_219000)
        
        # Assigning a type to the variable 'tuple_var_assignment_217821' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'tuple_var_assignment_217821', subscript_call_result_219003)
        
        # Assigning a Name to a Name (line 381):
        # Getting the type of 'tuple_var_assignment_217818' (line 381)
        tuple_var_assignment_217818_219004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'tuple_var_assignment_217818')
        # Assigning a type to the variable 'l' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'l', tuple_var_assignment_217818_219004)
        
        # Assigning a Name to a Name (line 381):
        # Getting the type of 'tuple_var_assignment_217819' (line 381)
        tuple_var_assignment_217819_219005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'tuple_var_assignment_217819')
        # Assigning a type to the variable 'b' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'b', tuple_var_assignment_217819_219005)
        
        # Assigning a Name to a Name (line 381):
        # Getting the type of 'tuple_var_assignment_217820' (line 381)
        tuple_var_assignment_217820_219006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'tuple_var_assignment_217820')
        # Assigning a type to the variable 'w' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 14), 'w', tuple_var_assignment_217820_219006)
        
        # Assigning a Name to a Name (line 381):
        # Getting the type of 'tuple_var_assignment_217821' (line 381)
        tuple_var_assignment_217821_219007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'tuple_var_assignment_217821')
        # Assigning a type to the variable 'h' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 17), 'h', tuple_var_assignment_217821_219007)
        
        # Assigning a Call to a Attribute (line 383):
        
        # Assigning a Call to a Attribute (line 383):
        
        # Call to pop(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_219011 = {}
        # Getting the type of 'self' (line 383)
        self_219008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 25), 'self', False)
        # Obtaining the member '_filter_renderers' of a type (line 383)
        _filter_renderers_219009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 25), self_219008, '_filter_renderers')
        # Obtaining the member 'pop' of a type (line 383)
        pop_219010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 25), _filter_renderers_219009, 'pop')
        # Calling pop(args, kwargs) (line 383)
        pop_call_result_219012 = invoke(stypy.reporting.localization.Localization(__file__, 383, 25), pop_219010, *[], **kwargs_219011)
        
        # Getting the type of 'self' (line 383)
        self_219013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member '_renderer' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_219013, '_renderer', pop_call_result_219012)
        
        # Call to _update_methods(...): (line 384)
        # Processing the call keyword arguments (line 384)
        kwargs_219016 = {}
        # Getting the type of 'self' (line 384)
        self_219014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self', False)
        # Obtaining the member '_update_methods' of a type (line 384)
        _update_methods_219015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_219014, '_update_methods')
        # Calling _update_methods(args, kwargs) (line 384)
        _update_methods_call_result_219017 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), _update_methods_219015, *[], **kwargs_219016)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'w' (line 386)
        w_219018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 11), 'w')
        int_219019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 15), 'int')
        # Applying the binary operator '>' (line 386)
        result_gt_219020 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 11), '>', w_219018, int_219019)
        
        
        # Getting the type of 'h' (line 386)
        h_219021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 21), 'h')
        int_219022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 25), 'int')
        # Applying the binary operator '>' (line 386)
        result_gt_219023 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 21), '>', h_219021, int_219022)
        
        # Applying the binary operator 'and' (line 386)
        result_and_keyword_219024 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 11), 'and', result_gt_219020, result_gt_219023)
        
        # Testing the type of an if condition (line 386)
        if_condition_219025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 8), result_and_keyword_219024)
        # Assigning a type to the variable 'if_condition_219025' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'if_condition_219025', if_condition_219025)
        # SSA begins for if statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Call to fromstring(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'buffer' (line 387)
        buffer_219028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 32), 'buffer', False)
        # Getting the type of 'np' (line 387)
        np_219029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 40), 'np', False)
        # Obtaining the member 'uint8' of a type (line 387)
        uint8_219030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 40), np_219029, 'uint8')
        # Processing the call keyword arguments (line 387)
        kwargs_219031 = {}
        # Getting the type of 'np' (line 387)
        np_219026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 18), 'np', False)
        # Obtaining the member 'fromstring' of a type (line 387)
        fromstring_219027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 18), np_219026, 'fromstring')
        # Calling fromstring(args, kwargs) (line 387)
        fromstring_call_result_219032 = invoke(stypy.reporting.localization.Localization(__file__, 387, 18), fromstring_219027, *[buffer_219028, uint8_219030], **kwargs_219031)
        
        # Assigning a type to the variable 'img' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'img', fromstring_call_result_219032)
        
        # Assigning a Call to a Tuple (line 388):
        
        # Assigning a Call to a Name:
        
        # Call to post_processing(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Call to reshape(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Obtaining an instance of the builtin type 'tuple' (line 388)
        tuple_219036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 388)
        # Adding element type (line 388)
        # Getting the type of 'h' (line 388)
        h_219037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 55), 'h', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 55), tuple_219036, h_219037)
        # Adding element type (line 388)
        # Getting the type of 'w' (line 388)
        w_219038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 58), 'w', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 55), tuple_219036, w_219038)
        # Adding element type (line 388)
        int_219039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 55), tuple_219036, int_219039)
        
        # Processing the call keyword arguments (line 388)
        kwargs_219040 = {}
        # Getting the type of 'img' (line 388)
        img_219034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 42), 'img', False)
        # Obtaining the member 'reshape' of a type (line 388)
        reshape_219035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 42), img_219034, 'reshape')
        # Calling reshape(args, kwargs) (line 388)
        reshape_call_result_219041 = invoke(stypy.reporting.localization.Localization(__file__, 388, 42), reshape_219035, *[tuple_219036], **kwargs_219040)
        
        float_219042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 67), 'float')
        # Applying the binary operator 'div' (line 388)
        result_div_219043 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 42), 'div', reshape_call_result_219041, float_219042)
        
        # Getting the type of 'self' (line 389)
        self_219044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 42), 'self', False)
        # Obtaining the member 'dpi' of a type (line 389)
        dpi_219045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 42), self_219044, 'dpi')
        # Processing the call keyword arguments (line 388)
        kwargs_219046 = {}
        # Getting the type of 'post_processing' (line 388)
        post_processing_219033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 26), 'post_processing', False)
        # Calling post_processing(args, kwargs) (line 388)
        post_processing_call_result_219047 = invoke(stypy.reporting.localization.Localization(__file__, 388, 26), post_processing_219033, *[result_div_219043, dpi_219045], **kwargs_219046)
        
        # Assigning a type to the variable 'call_assignment_217822' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217822', post_processing_call_result_219047)
        
        # Assigning a Call to a Name (line 388):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_219050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 12), 'int')
        # Processing the call keyword arguments
        kwargs_219051 = {}
        # Getting the type of 'call_assignment_217822' (line 388)
        call_assignment_217822_219048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217822', False)
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___219049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), call_assignment_217822_219048, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_219052 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___219049, *[int_219050], **kwargs_219051)
        
        # Assigning a type to the variable 'call_assignment_217823' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217823', getitem___call_result_219052)
        
        # Assigning a Name to a Name (line 388):
        # Getting the type of 'call_assignment_217823' (line 388)
        call_assignment_217823_219053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217823')
        # Assigning a type to the variable 'img' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'img', call_assignment_217823_219053)
        
        # Assigning a Call to a Name (line 388):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_219056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 12), 'int')
        # Processing the call keyword arguments
        kwargs_219057 = {}
        # Getting the type of 'call_assignment_217822' (line 388)
        call_assignment_217822_219054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217822', False)
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___219055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), call_assignment_217822_219054, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_219058 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___219055, *[int_219056], **kwargs_219057)
        
        # Assigning a type to the variable 'call_assignment_217824' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217824', getitem___call_result_219058)
        
        # Assigning a Name to a Name (line 388):
        # Getting the type of 'call_assignment_217824' (line 388)
        call_assignment_217824_219059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217824')
        # Assigning a type to the variable 'ox' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 17), 'ox', call_assignment_217824_219059)
        
        # Assigning a Call to a Name (line 388):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_219062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 12), 'int')
        # Processing the call keyword arguments
        kwargs_219063 = {}
        # Getting the type of 'call_assignment_217822' (line 388)
        call_assignment_217822_219060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217822', False)
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___219061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), call_assignment_217822_219060, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_219064 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___219061, *[int_219062], **kwargs_219063)
        
        # Assigning a type to the variable 'call_assignment_217825' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217825', getitem___call_result_219064)
        
        # Assigning a Name to a Name (line 388):
        # Getting the type of 'call_assignment_217825' (line 388)
        call_assignment_217825_219065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'call_assignment_217825')
        # Assigning a type to the variable 'oy' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 21), 'oy', call_assignment_217825_219065)
        
        # Assigning a Call to a Name (line 390):
        
        # Assigning a Call to a Name (line 390):
        
        # Call to new_gc(...): (line 390)
        # Processing the call keyword arguments (line 390)
        kwargs_219068 = {}
        # Getting the type of 'self' (line 390)
        self_219066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'self', False)
        # Obtaining the member 'new_gc' of a type (line 390)
        new_gc_219067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 17), self_219066, 'new_gc')
        # Calling new_gc(args, kwargs) (line 390)
        new_gc_call_result_219069 = invoke(stypy.reporting.localization.Localization(__file__, 390, 17), new_gc_219067, *[], **kwargs_219068)
        
        # Assigning a type to the variable 'gc' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'gc', new_gc_call_result_219069)
        
        
        # Getting the type of 'img' (line 391)
        img_219070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'img')
        # Obtaining the member 'dtype' of a type (line 391)
        dtype_219071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 15), img_219070, 'dtype')
        # Obtaining the member 'kind' of a type (line 391)
        kind_219072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 15), dtype_219071, 'kind')
        unicode_219073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 33), 'unicode', u'f')
        # Applying the binary operator '==' (line 391)
        result_eq_219074 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 15), '==', kind_219072, unicode_219073)
        
        # Testing the type of an if condition (line 391)
        if_condition_219075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 12), result_eq_219074)
        # Assigning a type to the variable 'if_condition_219075' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'if_condition_219075', if_condition_219075)
        # SSA begins for if statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to asarray(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'img' (line 392)
        img_219078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 33), 'img', False)
        float_219079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 39), 'float')
        # Applying the binary operator '*' (line 392)
        result_mul_219080 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 33), '*', img_219078, float_219079)
        
        # Getting the type of 'np' (line 392)
        np_219081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 45), 'np', False)
        # Obtaining the member 'uint8' of a type (line 392)
        uint8_219082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 45), np_219081, 'uint8')
        # Processing the call keyword arguments (line 392)
        kwargs_219083 = {}
        # Getting the type of 'np' (line 392)
        np_219076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'np', False)
        # Obtaining the member 'asarray' of a type (line 392)
        asarray_219077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 22), np_219076, 'asarray')
        # Calling asarray(args, kwargs) (line 392)
        asarray_call_result_219084 = invoke(stypy.reporting.localization.Localization(__file__, 392, 22), asarray_219077, *[result_mul_219080, uint8_219082], **kwargs_219083)
        
        # Assigning a type to the variable 'img' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'img', asarray_call_result_219084)
        # SSA join for if statement (line 391)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 393):
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        int_219085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 24), 'int')
        slice_219086 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 393, 18), None, None, int_219085)
        # Getting the type of 'img' (line 393)
        img_219087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 18), 'img')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___219088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 18), img_219087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_219089 = invoke(stypy.reporting.localization.Localization(__file__, 393, 18), getitem___219088, slice_219086)
        
        # Assigning a type to the variable 'img' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'img', subscript_call_result_219089)
        
        # Call to draw_image(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'gc' (line 395)
        gc_219093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'gc', False)
        # Getting the type of 'l' (line 395)
        l_219094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 20), 'l', False)
        # Getting the type of 'ox' (line 395)
        ox_219095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 24), 'ox', False)
        # Applying the binary operator '+' (line 395)
        result_add_219096 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 20), '+', l_219094, ox_219095)
        
        # Getting the type of 'height' (line 395)
        height_219097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), 'height', False)
        # Getting the type of 'b' (line 395)
        b_219098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 37), 'b', False)
        # Applying the binary operator '-' (line 395)
        result_sub_219099 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 28), '-', height_219097, b_219098)
        
        # Getting the type of 'h' (line 395)
        h_219100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 'h', False)
        # Applying the binary operator '-' (line 395)
        result_sub_219101 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 39), '-', result_sub_219099, h_219100)
        
        # Getting the type of 'oy' (line 395)
        oy_219102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 45), 'oy', False)
        # Applying the binary operator '+' (line 395)
        result_add_219103 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 43), '+', result_sub_219101, oy_219102)
        
        # Getting the type of 'img' (line 395)
        img_219104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 49), 'img', False)
        # Processing the call keyword arguments (line 394)
        kwargs_219105 = {}
        # Getting the type of 'self' (line 394)
        self_219090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'self', False)
        # Obtaining the member '_renderer' of a type (line 394)
        _renderer_219091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), self_219090, '_renderer')
        # Obtaining the member 'draw_image' of a type (line 394)
        draw_image_219092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), _renderer_219091, 'draw_image')
        # Calling draw_image(args, kwargs) (line 394)
        draw_image_call_result_219106 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), draw_image_219092, *[gc_219093, result_add_219096, result_add_219103, img_219104], **kwargs_219105)
        
        # SSA join for if statement (line 386)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'stop_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stop_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_219107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stop_filter'
        return stypy_return_type_219107


# Assigning a type to the variable 'RendererAgg' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'RendererAgg', RendererAgg)

# Assigning a Num to a Name (line 72):
int_219108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 10), 'int')
# Getting the type of 'RendererAgg'
RendererAgg_219109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RendererAgg')
# Setting the type of the member 'debug' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RendererAgg_219109, 'debug', int_219108)

# Assigning a Call to a Name (line 85):

# Call to RLock(...): (line 85)
# Processing the call keyword arguments (line 85)
kwargs_219112 = {}
# Getting the type of 'threading' (line 85)
threading_219110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'threading', False)
# Obtaining the member 'RLock' of a type (line 85)
RLock_219111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 11), threading_219110, 'RLock')
# Calling RLock(args, kwargs) (line 85)
RLock_call_result_219113 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), RLock_219111, *[], **kwargs_219112)

# Getting the type of 'RendererAgg'
RendererAgg_219114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RendererAgg')
# Setting the type of the member 'lock' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RendererAgg_219114, 'lock', RLock_call_result_219113)
# Declaration of the 'FigureCanvasAgg' class
# Getting the type of 'FigureCanvasBase' (line 398)
FigureCanvasBase_219115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 22), 'FigureCanvasBase')

class FigureCanvasAgg(FigureCanvasBase_219115, ):
    unicode_219116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, (-1)), 'unicode', u'\n    The canvas the figure renders into.  Calls the draw and print fig\n    methods, creates the renderers, etc...\n\n    Attributes\n    ----------\n    figure : `matplotlib.figure.Figure`\n        A high-level Figure instance\n\n    ')

    @norecursion
    def copy_from_bbox(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy_from_bbox'
        module_type_store = module_type_store.open_function_context('copy_from_bbox', 410, 4, False)
        # Assigning a type to the variable 'self' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.copy_from_bbox')
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_param_names_list', ['bbox'])
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.copy_from_bbox.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.copy_from_bbox', ['bbox'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy_from_bbox', localization, ['bbox'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy_from_bbox(...)' code ##################

        
        # Assigning a Call to a Name (line 411):
        
        # Assigning a Call to a Name (line 411):
        
        # Call to get_renderer(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_219119 = {}
        # Getting the type of 'self' (line 411)
        self_219117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 411)
        get_renderer_219118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 19), self_219117, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 411)
        get_renderer_call_result_219120 = invoke(stypy.reporting.localization.Localization(__file__, 411, 19), get_renderer_219118, *[], **kwargs_219119)
        
        # Assigning a type to the variable 'renderer' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'renderer', get_renderer_call_result_219120)
        
        # Call to copy_from_bbox(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'bbox' (line 412)
        bbox_219123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 39), 'bbox', False)
        # Processing the call keyword arguments (line 412)
        kwargs_219124 = {}
        # Getting the type of 'renderer' (line 412)
        renderer_219121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 15), 'renderer', False)
        # Obtaining the member 'copy_from_bbox' of a type (line 412)
        copy_from_bbox_219122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 15), renderer_219121, 'copy_from_bbox')
        # Calling copy_from_bbox(args, kwargs) (line 412)
        copy_from_bbox_call_result_219125 = invoke(stypy.reporting.localization.Localization(__file__, 412, 15), copy_from_bbox_219122, *[bbox_219123], **kwargs_219124)
        
        # Assigning a type to the variable 'stypy_return_type' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'stypy_return_type', copy_from_bbox_call_result_219125)
        
        # ################# End of 'copy_from_bbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy_from_bbox' in the type store
        # Getting the type of 'stypy_return_type' (line 410)
        stypy_return_type_219126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy_from_bbox'
        return stypy_return_type_219126


    @norecursion
    def restore_region(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 414)
        None_219127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 42), 'None')
        # Getting the type of 'None' (line 414)
        None_219128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 51), 'None')
        defaults = [None_219127, None_219128]
        # Create a new context for function 'restore_region'
        module_type_store = module_type_store.open_function_context('restore_region', 414, 4, False)
        # Assigning a type to the variable 'self' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.restore_region')
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_param_names_list', ['region', 'bbox', 'xy'])
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.restore_region.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.restore_region', ['region', 'bbox', 'xy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'restore_region', localization, ['region', 'bbox', 'xy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'restore_region(...)' code ##################

        
        # Assigning a Call to a Name (line 415):
        
        # Assigning a Call to a Name (line 415):
        
        # Call to get_renderer(...): (line 415)
        # Processing the call keyword arguments (line 415)
        kwargs_219131 = {}
        # Getting the type of 'self' (line 415)
        self_219129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 19), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 415)
        get_renderer_219130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 19), self_219129, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 415)
        get_renderer_call_result_219132 = invoke(stypy.reporting.localization.Localization(__file__, 415, 19), get_renderer_219130, *[], **kwargs_219131)
        
        # Assigning a type to the variable 'renderer' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'renderer', get_renderer_call_result_219132)
        
        # Call to restore_region(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'region' (line 416)
        region_219135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 39), 'region', False)
        # Getting the type of 'bbox' (line 416)
        bbox_219136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 47), 'bbox', False)
        # Getting the type of 'xy' (line 416)
        xy_219137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 53), 'xy', False)
        # Processing the call keyword arguments (line 416)
        kwargs_219138 = {}
        # Getting the type of 'renderer' (line 416)
        renderer_219133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'renderer', False)
        # Obtaining the member 'restore_region' of a type (line 416)
        restore_region_219134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 15), renderer_219133, 'restore_region')
        # Calling restore_region(args, kwargs) (line 416)
        restore_region_call_result_219139 = invoke(stypy.reporting.localization.Localization(__file__, 416, 15), restore_region_219134, *[region_219135, bbox_219136, xy_219137], **kwargs_219138)
        
        # Assigning a type to the variable 'stypy_return_type' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'stypy_return_type', restore_region_call_result_219139)
        
        # ################# End of 'restore_region(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'restore_region' in the type store
        # Getting the type of 'stypy_return_type' (line 414)
        stypy_return_type_219140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'restore_region'
        return stypy_return_type_219140


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 418, 4, False)
        # Assigning a type to the variable 'self' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.draw')
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.draw', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw(...)' code ##################

        unicode_219141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'unicode', u'\n        Draw the figure using the renderer\n        ')
        
        # Assigning a Call to a Attribute (line 422):
        
        # Assigning a Call to a Attribute (line 422):
        
        # Call to get_renderer(...): (line 422)
        # Processing the call keyword arguments (line 422)
        # Getting the type of 'True' (line 422)
        True_219144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 50), 'True', False)
        keyword_219145 = True_219144
        kwargs_219146 = {'cleared': keyword_219145}
        # Getting the type of 'self' (line 422)
        self_219142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 422)
        get_renderer_219143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 24), self_219142, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 422)
        get_renderer_call_result_219147 = invoke(stypy.reporting.localization.Localization(__file__, 422, 24), get_renderer_219143, *[], **kwargs_219146)
        
        # Getting the type of 'self' (line 422)
        self_219148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'self')
        # Setting the type of the member 'renderer' of a type (line 422)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), self_219148, 'renderer', get_renderer_call_result_219147)
        
        # Call to acquire(...): (line 424)
        # Processing the call keyword arguments (line 424)
        kwargs_219152 = {}
        # Getting the type of 'RendererAgg' (line 424)
        RendererAgg_219149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'RendererAgg', False)
        # Obtaining the member 'lock' of a type (line 424)
        lock_219150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), RendererAgg_219149, 'lock')
        # Obtaining the member 'acquire' of a type (line 424)
        acquire_219151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), lock_219150, 'acquire')
        # Calling acquire(args, kwargs) (line 424)
        acquire_call_result_219153 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), acquire_219151, *[], **kwargs_219152)
        
        
        # Assigning a Attribute to a Name (line 426):
        
        # Assigning a Attribute to a Name (line 426):
        # Getting the type of 'self' (line 426)
        self_219154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 18), 'self')
        # Obtaining the member 'toolbar' of a type (line 426)
        toolbar_219155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 18), self_219154, 'toolbar')
        # Assigning a type to the variable 'toolbar' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'toolbar', toolbar_219155)
        
        # Try-finally block (line 427)
        
        # Getting the type of 'toolbar' (line 428)
        toolbar_219156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'toolbar')
        # Testing the type of an if condition (line 428)
        if_condition_219157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 12), toolbar_219156)
        # Assigning a type to the variable 'if_condition_219157' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'if_condition_219157', if_condition_219157)
        # SSA begins for if statement (line 428)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_cursor(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'cursors' (line 429)
        cursors_219160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 35), 'cursors', False)
        # Obtaining the member 'WAIT' of a type (line 429)
        WAIT_219161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 35), cursors_219160, 'WAIT')
        # Processing the call keyword arguments (line 429)
        kwargs_219162 = {}
        # Getting the type of 'toolbar' (line 429)
        toolbar_219158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'toolbar', False)
        # Obtaining the member 'set_cursor' of a type (line 429)
        set_cursor_219159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 16), toolbar_219158, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 429)
        set_cursor_call_result_219163 = invoke(stypy.reporting.localization.Localization(__file__, 429, 16), set_cursor_219159, *[WAIT_219161], **kwargs_219162)
        
        # SSA join for if statement (line 428)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'self' (line 430)
        self_219167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 29), 'self', False)
        # Obtaining the member 'renderer' of a type (line 430)
        renderer_219168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 29), self_219167, 'renderer')
        # Processing the call keyword arguments (line 430)
        kwargs_219169 = {}
        # Getting the type of 'self' (line 430)
        self_219164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'self', False)
        # Obtaining the member 'figure' of a type (line 430)
        figure_219165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), self_219164, 'figure')
        # Obtaining the member 'draw' of a type (line 430)
        draw_219166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), figure_219165, 'draw')
        # Calling draw(args, kwargs) (line 430)
        draw_call_result_219170 = invoke(stypy.reporting.localization.Localization(__file__, 430, 12), draw_219166, *[renderer_219168], **kwargs_219169)
        
        
        # finally branch of the try-finally block (line 427)
        
        # Getting the type of 'toolbar' (line 432)
        toolbar_219171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 15), 'toolbar')
        # Testing the type of an if condition (line 432)
        if_condition_219172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 12), toolbar_219171)
        # Assigning a type to the variable 'if_condition_219172' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'if_condition_219172', if_condition_219172)
        # SSA begins for if statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_cursor(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'toolbar' (line 433)
        toolbar_219175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 35), 'toolbar', False)
        # Obtaining the member '_lastCursor' of a type (line 433)
        _lastCursor_219176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 35), toolbar_219175, '_lastCursor')
        # Processing the call keyword arguments (line 433)
        kwargs_219177 = {}
        # Getting the type of 'toolbar' (line 433)
        toolbar_219173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 16), 'toolbar', False)
        # Obtaining the member 'set_cursor' of a type (line 433)
        set_cursor_219174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 16), toolbar_219173, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 433)
        set_cursor_call_result_219178 = invoke(stypy.reporting.localization.Localization(__file__, 433, 16), set_cursor_219174, *[_lastCursor_219176], **kwargs_219177)
        
        # SSA join for if statement (line 432)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to release(...): (line 434)
        # Processing the call keyword arguments (line 434)
        kwargs_219182 = {}
        # Getting the type of 'RendererAgg' (line 434)
        RendererAgg_219179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'RendererAgg', False)
        # Obtaining the member 'lock' of a type (line 434)
        lock_219180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), RendererAgg_219179, 'lock')
        # Obtaining the member 'release' of a type (line 434)
        release_219181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), lock_219180, 'release')
        # Calling release(args, kwargs) (line 434)
        release_call_result_219183 = invoke(stypy.reporting.localization.Localization(__file__, 434, 12), release_219181, *[], **kwargs_219182)
        
        
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 418)
        stypy_return_type_219184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_219184


    @norecursion
    def get_renderer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 436)
        False_219185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 35), 'False')
        defaults = [False_219185]
        # Create a new context for function 'get_renderer'
        module_type_store = module_type_store.open_function_context('get_renderer', 436, 4, False)
        # Assigning a type to the variable 'self' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.get_renderer')
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_param_names_list', ['cleared'])
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.get_renderer.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.get_renderer', ['cleared'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_renderer', localization, ['cleared'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_renderer(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 437):
        
        # Assigning a Subscript to a Name (line 437):
        
        # Obtaining the type of the subscript
        int_219186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
        # Getting the type of 'self' (line 437)
        self_219187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 21), 'self')
        # Obtaining the member 'figure' of a type (line 437)
        figure_219188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), self_219187, 'figure')
        # Obtaining the member 'bbox' of a type (line 437)
        bbox_219189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), figure_219188, 'bbox')
        # Obtaining the member 'bounds' of a type (line 437)
        bounds_219190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), bbox_219189, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 437)
        getitem___219191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), bounds_219190, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 437)
        subscript_call_result_219192 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___219191, int_219186)
        
        # Assigning a type to the variable 'tuple_var_assignment_217826' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_217826', subscript_call_result_219192)
        
        # Assigning a Subscript to a Name (line 437):
        
        # Obtaining the type of the subscript
        int_219193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
        # Getting the type of 'self' (line 437)
        self_219194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 21), 'self')
        # Obtaining the member 'figure' of a type (line 437)
        figure_219195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), self_219194, 'figure')
        # Obtaining the member 'bbox' of a type (line 437)
        bbox_219196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), figure_219195, 'bbox')
        # Obtaining the member 'bounds' of a type (line 437)
        bounds_219197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), bbox_219196, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 437)
        getitem___219198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), bounds_219197, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 437)
        subscript_call_result_219199 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___219198, int_219193)
        
        # Assigning a type to the variable 'tuple_var_assignment_217827' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_217827', subscript_call_result_219199)
        
        # Assigning a Subscript to a Name (line 437):
        
        # Obtaining the type of the subscript
        int_219200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
        # Getting the type of 'self' (line 437)
        self_219201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 21), 'self')
        # Obtaining the member 'figure' of a type (line 437)
        figure_219202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), self_219201, 'figure')
        # Obtaining the member 'bbox' of a type (line 437)
        bbox_219203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), figure_219202, 'bbox')
        # Obtaining the member 'bounds' of a type (line 437)
        bounds_219204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), bbox_219203, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 437)
        getitem___219205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), bounds_219204, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 437)
        subscript_call_result_219206 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___219205, int_219200)
        
        # Assigning a type to the variable 'tuple_var_assignment_217828' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_217828', subscript_call_result_219206)
        
        # Assigning a Subscript to a Name (line 437):
        
        # Obtaining the type of the subscript
        int_219207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 8), 'int')
        # Getting the type of 'self' (line 437)
        self_219208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 21), 'self')
        # Obtaining the member 'figure' of a type (line 437)
        figure_219209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), self_219208, 'figure')
        # Obtaining the member 'bbox' of a type (line 437)
        bbox_219210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), figure_219209, 'bbox')
        # Obtaining the member 'bounds' of a type (line 437)
        bounds_219211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), bbox_219210, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 437)
        getitem___219212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), bounds_219211, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 437)
        subscript_call_result_219213 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), getitem___219212, int_219207)
        
        # Assigning a type to the variable 'tuple_var_assignment_217829' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_217829', subscript_call_result_219213)
        
        # Assigning a Name to a Name (line 437):
        # Getting the type of 'tuple_var_assignment_217826' (line 437)
        tuple_var_assignment_217826_219214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_217826')
        # Assigning a type to the variable 'l' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'l', tuple_var_assignment_217826_219214)
        
        # Assigning a Name to a Name (line 437):
        # Getting the type of 'tuple_var_assignment_217827' (line 437)
        tuple_var_assignment_217827_219215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_217827')
        # Assigning a type to the variable 'b' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'b', tuple_var_assignment_217827_219215)
        
        # Assigning a Name to a Name (line 437):
        # Getting the type of 'tuple_var_assignment_217828' (line 437)
        tuple_var_assignment_217828_219216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_217828')
        # Assigning a type to the variable 'w' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 14), 'w', tuple_var_assignment_217828_219216)
        
        # Assigning a Name to a Name (line 437):
        # Getting the type of 'tuple_var_assignment_217829' (line 437)
        tuple_var_assignment_217829_219217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'tuple_var_assignment_217829')
        # Assigning a type to the variable 'h' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 17), 'h', tuple_var_assignment_217829_219217)
        
        # Assigning a Tuple to a Name (line 438):
        
        # Assigning a Tuple to a Name (line 438):
        
        # Obtaining an instance of the builtin type 'tuple' (line 438)
        tuple_219218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 438)
        # Adding element type (line 438)
        # Getting the type of 'w' (line 438)
        w_219219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 14), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 14), tuple_219218, w_219219)
        # Adding element type (line 438)
        # Getting the type of 'h' (line 438)
        h_219220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 17), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 14), tuple_219218, h_219220)
        # Adding element type (line 438)
        # Getting the type of 'self' (line 438)
        self_219221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'self')
        # Obtaining the member 'figure' of a type (line 438)
        figure_219222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 20), self_219221, 'figure')
        # Obtaining the member 'dpi' of a type (line 438)
        dpi_219223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 20), figure_219222, 'dpi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 14), tuple_219218, dpi_219223)
        
        # Assigning a type to the variable 'key' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'key', tuple_219218)
        
        
        # SSA begins for try-except statement (line 439)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining an instance of the builtin type 'tuple' (line 439)
        tuple_219224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 439)
        # Adding element type (line 439)
        # Getting the type of 'self' (line 439)
        self_219225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 13), 'self')
        # Obtaining the member '_lastKey' of a type (line 439)
        _lastKey_219226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 13), self_219225, '_lastKey')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 13), tuple_219224, _lastKey_219226)
        # Adding element type (line 439)
        # Getting the type of 'self' (line 439)
        self_219227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 28), 'self')
        # Obtaining the member 'renderer' of a type (line 439)
        renderer_219228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 28), self_219227, 'renderer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 13), tuple_219224, renderer_219228)
        
        # SSA branch for the except part of a try statement (line 439)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 439)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 440):
        
        # Assigning a Name to a Name (line 440):
        # Getting the type of 'True' (line 440)
        True_219229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 51), 'True')
        # Assigning a type to the variable 'need_new_renderer' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 31), 'need_new_renderer', True_219229)
        # SSA branch for the else branch of a try statement (line 439)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Compare to a Name (line 441):
        
        # Assigning a Compare to a Name (line 441):
        
        # Getting the type of 'self' (line 441)
        self_219230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 36), 'self')
        # Obtaining the member '_lastKey' of a type (line 441)
        _lastKey_219231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 36), self_219230, '_lastKey')
        # Getting the type of 'key' (line 441)
        key_219232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 53), 'key')
        # Applying the binary operator '!=' (line 441)
        result_ne_219233 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 36), '!=', _lastKey_219231, key_219232)
        
        # Assigning a type to the variable 'need_new_renderer' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 15), 'need_new_renderer', result_ne_219233)
        # SSA join for try-except statement (line 439)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'need_new_renderer' (line 443)
        need_new_renderer_219234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 11), 'need_new_renderer')
        # Testing the type of an if condition (line 443)
        if_condition_219235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 8), need_new_renderer_219234)
        # Assigning a type to the variable 'if_condition_219235' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'if_condition_219235', if_condition_219235)
        # SSA begins for if statement (line 443)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 444):
        
        # Assigning a Call to a Attribute (line 444):
        
        # Call to RendererAgg(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'w' (line 444)
        w_219237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 40), 'w', False)
        # Getting the type of 'h' (line 444)
        h_219238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 43), 'h', False)
        # Getting the type of 'self' (line 444)
        self_219239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 46), 'self', False)
        # Obtaining the member 'figure' of a type (line 444)
        figure_219240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 46), self_219239, 'figure')
        # Obtaining the member 'dpi' of a type (line 444)
        dpi_219241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 46), figure_219240, 'dpi')
        # Processing the call keyword arguments (line 444)
        kwargs_219242 = {}
        # Getting the type of 'RendererAgg' (line 444)
        RendererAgg_219236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 28), 'RendererAgg', False)
        # Calling RendererAgg(args, kwargs) (line 444)
        RendererAgg_call_result_219243 = invoke(stypy.reporting.localization.Localization(__file__, 444, 28), RendererAgg_219236, *[w_219237, h_219238, dpi_219241], **kwargs_219242)
        
        # Getting the type of 'self' (line 444)
        self_219244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'self')
        # Setting the type of the member 'renderer' of a type (line 444)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), self_219244, 'renderer', RendererAgg_call_result_219243)
        
        # Assigning a Name to a Attribute (line 445):
        
        # Assigning a Name to a Attribute (line 445):
        # Getting the type of 'key' (line 445)
        key_219245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 28), 'key')
        # Getting the type of 'self' (line 445)
        self_219246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'self')
        # Setting the type of the member '_lastKey' of a type (line 445)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 12), self_219246, '_lastKey', key_219245)
        # SSA branch for the else part of an if statement (line 443)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'cleared' (line 446)
        cleared_219247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 13), 'cleared')
        # Testing the type of an if condition (line 446)
        if_condition_219248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 446, 13), cleared_219247)
        # Assigning a type to the variable 'if_condition_219248' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 13), 'if_condition_219248', if_condition_219248)
        # SSA begins for if statement (line 446)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to clear(...): (line 447)
        # Processing the call keyword arguments (line 447)
        kwargs_219252 = {}
        # Getting the type of 'self' (line 447)
        self_219249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'self', False)
        # Obtaining the member 'renderer' of a type (line 447)
        renderer_219250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), self_219249, 'renderer')
        # Obtaining the member 'clear' of a type (line 447)
        clear_219251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), renderer_219250, 'clear')
        # Calling clear(args, kwargs) (line 447)
        clear_call_result_219253 = invoke(stypy.reporting.localization.Localization(__file__, 447, 12), clear_219251, *[], **kwargs_219252)
        
        # SSA join for if statement (line 446)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 443)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 448)
        self_219254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'self')
        # Obtaining the member 'renderer' of a type (line 448)
        renderer_219255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 15), self_219254, 'renderer')
        # Assigning a type to the variable 'stypy_return_type' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'stypy_return_type', renderer_219255)
        
        # ################# End of 'get_renderer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_renderer' in the type store
        # Getting the type of 'stypy_return_type' (line 436)
        stypy_return_type_219256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_renderer'
        return stypy_return_type_219256


    @norecursion
    def tostring_rgb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tostring_rgb'
        module_type_store = module_type_store.open_function_context('tostring_rgb', 450, 4, False)
        # Assigning a type to the variable 'self' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.tostring_rgb')
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.tostring_rgb.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.tostring_rgb', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tostring_rgb', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tostring_rgb(...)' code ##################

        unicode_219257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, (-1)), 'unicode', u'Get the image as an RGB byte string\n\n        `draw` must be called at least once before this function will work and\n        to update the renderer for any subsequent changes to the Figure.\n\n        Returns\n        -------\n        bytes\n        ')
        
        # Call to tostring_rgb(...): (line 460)
        # Processing the call keyword arguments (line 460)
        kwargs_219261 = {}
        # Getting the type of 'self' (line 460)
        self_219258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 15), 'self', False)
        # Obtaining the member 'renderer' of a type (line 460)
        renderer_219259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 15), self_219258, 'renderer')
        # Obtaining the member 'tostring_rgb' of a type (line 460)
        tostring_rgb_219260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 15), renderer_219259, 'tostring_rgb')
        # Calling tostring_rgb(args, kwargs) (line 460)
        tostring_rgb_call_result_219262 = invoke(stypy.reporting.localization.Localization(__file__, 460, 15), tostring_rgb_219260, *[], **kwargs_219261)
        
        # Assigning a type to the variable 'stypy_return_type' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'stypy_return_type', tostring_rgb_call_result_219262)
        
        # ################# End of 'tostring_rgb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tostring_rgb' in the type store
        # Getting the type of 'stypy_return_type' (line 450)
        stypy_return_type_219263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tostring_rgb'
        return stypy_return_type_219263


    @norecursion
    def tostring_argb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tostring_argb'
        module_type_store = module_type_store.open_function_context('tostring_argb', 462, 4, False)
        # Assigning a type to the variable 'self' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.tostring_argb')
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.tostring_argb.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.tostring_argb', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tostring_argb', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tostring_argb(...)' code ##################

        unicode_219264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, (-1)), 'unicode', u'Get the image as an ARGB byte string\n\n        `draw` must be called at least once before this function will work and\n        to update the renderer for any subsequent changes to the Figure.\n\n        Returns\n        -------\n        bytes\n\n        ')
        
        # Call to tostring_argb(...): (line 473)
        # Processing the call keyword arguments (line 473)
        kwargs_219268 = {}
        # Getting the type of 'self' (line 473)
        self_219265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 15), 'self', False)
        # Obtaining the member 'renderer' of a type (line 473)
        renderer_219266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 15), self_219265, 'renderer')
        # Obtaining the member 'tostring_argb' of a type (line 473)
        tostring_argb_219267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 15), renderer_219266, 'tostring_argb')
        # Calling tostring_argb(args, kwargs) (line 473)
        tostring_argb_call_result_219269 = invoke(stypy.reporting.localization.Localization(__file__, 473, 15), tostring_argb_219267, *[], **kwargs_219268)
        
        # Assigning a type to the variable 'stypy_return_type' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'stypy_return_type', tostring_argb_call_result_219269)
        
        # ################# End of 'tostring_argb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tostring_argb' in the type store
        # Getting the type of 'stypy_return_type' (line 462)
        stypy_return_type_219270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219270)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tostring_argb'
        return stypy_return_type_219270


    @norecursion
    def buffer_rgba(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'buffer_rgba'
        module_type_store = module_type_store.open_function_context('buffer_rgba', 475, 4, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.buffer_rgba')
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.buffer_rgba.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.buffer_rgba', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'buffer_rgba', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'buffer_rgba(...)' code ##################

        unicode_219271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, (-1)), 'unicode', u'Get the image as an RGBA byte string\n\n        `draw` must be called at least once before this function will work and\n        to update the renderer for any subsequent changes to the Figure.\n\n        Returns\n        -------\n        bytes\n        ')
        
        # Call to buffer_rgba(...): (line 485)
        # Processing the call keyword arguments (line 485)
        kwargs_219275 = {}
        # Getting the type of 'self' (line 485)
        self_219272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 15), 'self', False)
        # Obtaining the member 'renderer' of a type (line 485)
        renderer_219273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 15), self_219272, 'renderer')
        # Obtaining the member 'buffer_rgba' of a type (line 485)
        buffer_rgba_219274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 15), renderer_219273, 'buffer_rgba')
        # Calling buffer_rgba(args, kwargs) (line 485)
        buffer_rgba_call_result_219276 = invoke(stypy.reporting.localization.Localization(__file__, 485, 15), buffer_rgba_219274, *[], **kwargs_219275)
        
        # Assigning a type to the variable 'stypy_return_type' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'stypy_return_type', buffer_rgba_call_result_219276)
        
        # ################# End of 'buffer_rgba(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'buffer_rgba' in the type store
        # Getting the type of 'stypy_return_type' (line 475)
        stypy_return_type_219277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219277)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'buffer_rgba'
        return stypy_return_type_219277


    @norecursion
    def print_raw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_raw'
        module_type_store = module_type_store.open_function_context('print_raw', 487, 4, False)
        # Assigning a type to the variable 'self' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.print_raw')
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_param_names_list', ['filename_or_obj'])
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.print_raw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.print_raw', ['filename_or_obj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_raw', localization, ['filename_or_obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_raw(...)' code ##################

        
        # Call to draw(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'self' (line 488)
        self_219280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 29), 'self', False)
        # Processing the call keyword arguments (line 488)
        kwargs_219281 = {}
        # Getting the type of 'FigureCanvasAgg' (line 488)
        FigureCanvasAgg_219278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'FigureCanvasAgg', False)
        # Obtaining the member 'draw' of a type (line 488)
        draw_219279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), FigureCanvasAgg_219278, 'draw')
        # Calling draw(args, kwargs) (line 488)
        draw_call_result_219282 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), draw_219279, *[self_219280], **kwargs_219281)
        
        
        # Assigning a Call to a Name (line 489):
        
        # Assigning a Call to a Name (line 489):
        
        # Call to get_renderer(...): (line 489)
        # Processing the call keyword arguments (line 489)
        kwargs_219285 = {}
        # Getting the type of 'self' (line 489)
        self_219283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 19), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 489)
        get_renderer_219284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 19), self_219283, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 489)
        get_renderer_call_result_219286 = invoke(stypy.reporting.localization.Localization(__file__, 489, 19), get_renderer_219284, *[], **kwargs_219285)
        
        # Assigning a type to the variable 'renderer' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'renderer', get_renderer_call_result_219286)
        
        # Assigning a Attribute to a Name (line 490):
        
        # Assigning a Attribute to a Name (line 490):
        # Getting the type of 'renderer' (line 490)
        renderer_219287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'renderer')
        # Obtaining the member 'dpi' of a type (line 490)
        dpi_219288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 23), renderer_219287, 'dpi')
        # Assigning a type to the variable 'original_dpi' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'original_dpi', dpi_219288)
        
        # Assigning a Attribute to a Attribute (line 491):
        
        # Assigning a Attribute to a Attribute (line 491):
        # Getting the type of 'self' (line 491)
        self_219289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 23), 'self')
        # Obtaining the member 'figure' of a type (line 491)
        figure_219290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 23), self_219289, 'figure')
        # Obtaining the member 'dpi' of a type (line 491)
        dpi_219291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 23), figure_219290, 'dpi')
        # Getting the type of 'renderer' (line 491)
        renderer_219292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'renderer')
        # Setting the type of the member 'dpi' of a type (line 491)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), renderer_219292, 'dpi', dpi_219291)
        
        
        # Call to isinstance(...): (line 492)
        # Processing the call arguments (line 492)
        # Getting the type of 'filename_or_obj' (line 492)
        filename_or_obj_219294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 22), 'filename_or_obj', False)
        # Getting the type of 'six' (line 492)
        six_219295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 39), 'six', False)
        # Obtaining the member 'string_types' of a type (line 492)
        string_types_219296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 39), six_219295, 'string_types')
        # Processing the call keyword arguments (line 492)
        kwargs_219297 = {}
        # Getting the type of 'isinstance' (line 492)
        isinstance_219293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 492)
        isinstance_call_result_219298 = invoke(stypy.reporting.localization.Localization(__file__, 492, 11), isinstance_219293, *[filename_or_obj_219294, string_types_219296], **kwargs_219297)
        
        # Testing the type of an if condition (line 492)
        if_condition_219299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 492, 8), isinstance_call_result_219298)
        # Assigning a type to the variable 'if_condition_219299' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'if_condition_219299', if_condition_219299)
        # SSA begins for if statement (line 492)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 493):
        
        # Assigning a Call to a Name (line 493):
        
        # Call to open(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'filename_or_obj' (line 493)
        filename_or_obj_219301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 27), 'filename_or_obj', False)
        unicode_219302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 44), 'unicode', u'wb')
        # Processing the call keyword arguments (line 493)
        kwargs_219303 = {}
        # Getting the type of 'open' (line 493)
        open_219300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 22), 'open', False)
        # Calling open(args, kwargs) (line 493)
        open_call_result_219304 = invoke(stypy.reporting.localization.Localization(__file__, 493, 22), open_219300, *[filename_or_obj_219301, unicode_219302], **kwargs_219303)
        
        # Assigning a type to the variable 'fileobj' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'fileobj', open_call_result_219304)
        
        # Assigning a Name to a Name (line 494):
        
        # Assigning a Name to a Name (line 494):
        # Getting the type of 'True' (line 494)
        True_219305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 20), 'True')
        # Assigning a type to the variable 'close' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'close', True_219305)
        # SSA branch for the else part of an if statement (line 492)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 496):
        
        # Assigning a Name to a Name (line 496):
        # Getting the type of 'filename_or_obj' (line 496)
        filename_or_obj_219306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 22), 'filename_or_obj')
        # Assigning a type to the variable 'fileobj' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'fileobj', filename_or_obj_219306)
        
        # Assigning a Name to a Name (line 497):
        
        # Assigning a Name to a Name (line 497):
        # Getting the type of 'False' (line 497)
        False_219307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 20), 'False')
        # Assigning a type to the variable 'close' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'close', False_219307)
        # SSA join for if statement (line 492)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Try-finally block (line 498)
        
        # Call to write(...): (line 499)
        # Processing the call arguments (line 499)
        
        # Call to buffer_rgba(...): (line 499)
        # Processing the call keyword arguments (line 499)
        kwargs_219313 = {}
        # Getting the type of 'renderer' (line 499)
        renderer_219310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 26), 'renderer', False)
        # Obtaining the member '_renderer' of a type (line 499)
        _renderer_219311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 26), renderer_219310, '_renderer')
        # Obtaining the member 'buffer_rgba' of a type (line 499)
        buffer_rgba_219312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 26), _renderer_219311, 'buffer_rgba')
        # Calling buffer_rgba(args, kwargs) (line 499)
        buffer_rgba_call_result_219314 = invoke(stypy.reporting.localization.Localization(__file__, 499, 26), buffer_rgba_219312, *[], **kwargs_219313)
        
        # Processing the call keyword arguments (line 499)
        kwargs_219315 = {}
        # Getting the type of 'fileobj' (line 499)
        fileobj_219308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'fileobj', False)
        # Obtaining the member 'write' of a type (line 499)
        write_219309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 12), fileobj_219308, 'write')
        # Calling write(args, kwargs) (line 499)
        write_call_result_219316 = invoke(stypy.reporting.localization.Localization(__file__, 499, 12), write_219309, *[buffer_rgba_call_result_219314], **kwargs_219315)
        
        
        # finally branch of the try-finally block (line 498)
        
        # Getting the type of 'close' (line 501)
        close_219317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 15), 'close')
        # Testing the type of an if condition (line 501)
        if_condition_219318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 12), close_219317)
        # Assigning a type to the variable 'if_condition_219318' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'if_condition_219318', if_condition_219318)
        # SSA begins for if statement (line 501)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 502)
        # Processing the call keyword arguments (line 502)
        kwargs_219321 = {}
        # Getting the type of 'fileobj' (line 502)
        fileobj_219319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'fileobj', False)
        # Obtaining the member 'close' of a type (line 502)
        close_219320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 16), fileobj_219319, 'close')
        # Calling close(args, kwargs) (line 502)
        close_call_result_219322 = invoke(stypy.reporting.localization.Localization(__file__, 502, 16), close_219320, *[], **kwargs_219321)
        
        # SSA join for if statement (line 501)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 503):
        
        # Assigning a Name to a Attribute (line 503):
        # Getting the type of 'original_dpi' (line 503)
        original_dpi_219323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 27), 'original_dpi')
        # Getting the type of 'renderer' (line 503)
        renderer_219324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'renderer')
        # Setting the type of the member 'dpi' of a type (line 503)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 12), renderer_219324, 'dpi', original_dpi_219323)
        
        
        # ################# End of 'print_raw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_raw' in the type store
        # Getting the type of 'stypy_return_type' (line 487)
        stypy_return_type_219325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219325)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_raw'
        return stypy_return_type_219325


    @norecursion
    def print_png(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_png'
        module_type_store = module_type_store.open_function_context('print_png', 506, 4, False)
        # Assigning a type to the variable 'self' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.print_png')
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_param_names_list', ['filename_or_obj'])
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.print_png.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.print_png', ['filename_or_obj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_png', localization, ['filename_or_obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_png(...)' code ##################

        
        # Call to draw(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'self' (line 507)
        self_219328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 29), 'self', False)
        # Processing the call keyword arguments (line 507)
        kwargs_219329 = {}
        # Getting the type of 'FigureCanvasAgg' (line 507)
        FigureCanvasAgg_219326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'FigureCanvasAgg', False)
        # Obtaining the member 'draw' of a type (line 507)
        draw_219327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), FigureCanvasAgg_219326, 'draw')
        # Calling draw(args, kwargs) (line 507)
        draw_call_result_219330 = invoke(stypy.reporting.localization.Localization(__file__, 507, 8), draw_219327, *[self_219328], **kwargs_219329)
        
        
        # Assigning a Call to a Name (line 508):
        
        # Assigning a Call to a Name (line 508):
        
        # Call to get_renderer(...): (line 508)
        # Processing the call keyword arguments (line 508)
        kwargs_219333 = {}
        # Getting the type of 'self' (line 508)
        self_219331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 19), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 508)
        get_renderer_219332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 19), self_219331, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 508)
        get_renderer_call_result_219334 = invoke(stypy.reporting.localization.Localization(__file__, 508, 19), get_renderer_219332, *[], **kwargs_219333)
        
        # Assigning a type to the variable 'renderer' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'renderer', get_renderer_call_result_219334)
        
        # Assigning a Attribute to a Name (line 509):
        
        # Assigning a Attribute to a Name (line 509):
        # Getting the type of 'renderer' (line 509)
        renderer_219335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 23), 'renderer')
        # Obtaining the member 'dpi' of a type (line 509)
        dpi_219336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 23), renderer_219335, 'dpi')
        # Assigning a type to the variable 'original_dpi' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'original_dpi', dpi_219336)
        
        # Assigning a Attribute to a Attribute (line 510):
        
        # Assigning a Attribute to a Attribute (line 510):
        # Getting the type of 'self' (line 510)
        self_219337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 23), 'self')
        # Obtaining the member 'figure' of a type (line 510)
        figure_219338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 23), self_219337, 'figure')
        # Obtaining the member 'dpi' of a type (line 510)
        dpi_219339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 23), figure_219338, 'dpi')
        # Getting the type of 'renderer' (line 510)
        renderer_219340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'renderer')
        # Setting the type of the member 'dpi' of a type (line 510)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), renderer_219340, 'dpi', dpi_219339)
        
        
        # Call to isinstance(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'filename_or_obj' (line 511)
        filename_or_obj_219342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 22), 'filename_or_obj', False)
        # Getting the type of 'six' (line 511)
        six_219343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 39), 'six', False)
        # Obtaining the member 'string_types' of a type (line 511)
        string_types_219344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 39), six_219343, 'string_types')
        # Processing the call keyword arguments (line 511)
        kwargs_219345 = {}
        # Getting the type of 'isinstance' (line 511)
        isinstance_219341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 511)
        isinstance_call_result_219346 = invoke(stypy.reporting.localization.Localization(__file__, 511, 11), isinstance_219341, *[filename_or_obj_219342, string_types_219344], **kwargs_219345)
        
        # Testing the type of an if condition (line 511)
        if_condition_219347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 8), isinstance_call_result_219346)
        # Assigning a type to the variable 'if_condition_219347' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'if_condition_219347', if_condition_219347)
        # SSA begins for if statement (line 511)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 512):
        
        # Assigning a Call to a Name (line 512):
        
        # Call to open(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'filename_or_obj' (line 512)
        filename_or_obj_219349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 35), 'filename_or_obj', False)
        unicode_219350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 52), 'unicode', u'wb')
        # Processing the call keyword arguments (line 512)
        kwargs_219351 = {}
        # Getting the type of 'open' (line 512)
        open_219348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 30), 'open', False)
        # Calling open(args, kwargs) (line 512)
        open_call_result_219352 = invoke(stypy.reporting.localization.Localization(__file__, 512, 30), open_219348, *[filename_or_obj_219349, unicode_219350], **kwargs_219351)
        
        # Assigning a type to the variable 'filename_or_obj' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'filename_or_obj', open_call_result_219352)
        
        # Assigning a Name to a Name (line 513):
        
        # Assigning a Name to a Name (line 513):
        # Getting the type of 'True' (line 513)
        True_219353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'True')
        # Assigning a type to the variable 'close' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'close', True_219353)
        # SSA branch for the else part of an if statement (line 511)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 515):
        
        # Assigning a Name to a Name (line 515):
        # Getting the type of 'False' (line 515)
        False_219354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 20), 'False')
        # Assigning a type to the variable 'close' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'close', False_219354)
        # SSA join for if statement (line 511)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 517):
        
        # Assigning a BinOp to a Name (line 517):
        unicode_219355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 22), 'unicode', u'matplotlib version ')
        # Getting the type of '__version__' (line 517)
        version___219356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 46), '__version__')
        # Applying the binary operator '+' (line 517)
        result_add_219357 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 22), '+', unicode_219355, version___219356)
        
        unicode_219358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 12), 'unicode', u', http://matplotlib.org/')
        # Applying the binary operator '+' (line 517)
        result_add_219359 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 58), '+', result_add_219357, unicode_219358)
        
        # Assigning a type to the variable 'version_str' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'version_str', result_add_219359)
        
        # Assigning a Call to a Name (line 519):
        
        # Assigning a Call to a Name (line 519):
        
        # Call to OrderedDict(...): (line 519)
        # Processing the call arguments (line 519)
        
        # Obtaining an instance of the builtin type 'dict' (line 519)
        dict_219361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 31), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 519)
        # Adding element type (key, value) (line 519)
        unicode_219362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 32), 'unicode', u'Software')
        # Getting the type of 'version_str' (line 519)
        version_str_219363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 44), 'version_str', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 31), dict_219361, (unicode_219362, version_str_219363))
        
        # Processing the call keyword arguments (line 519)
        kwargs_219364 = {}
        # Getting the type of 'OrderedDict' (line 519)
        OrderedDict_219360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 19), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 519)
        OrderedDict_call_result_219365 = invoke(stypy.reporting.localization.Localization(__file__, 519, 19), OrderedDict_219360, *[dict_219361], **kwargs_219364)
        
        # Assigning a type to the variable 'metadata' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'metadata', OrderedDict_call_result_219365)
        
        # Assigning a Call to a Name (line 520):
        
        # Assigning a Call to a Name (line 520):
        
        # Call to pop(...): (line 520)
        # Processing the call arguments (line 520)
        unicode_219368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 35), 'unicode', u'metadata')
        # Getting the type of 'None' (line 520)
        None_219369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 47), 'None', False)
        # Processing the call keyword arguments (line 520)
        kwargs_219370 = {}
        # Getting the type of 'kwargs' (line 520)
        kwargs_219366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 24), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 520)
        pop_219367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 24), kwargs_219366, 'pop')
        # Calling pop(args, kwargs) (line 520)
        pop_call_result_219371 = invoke(stypy.reporting.localization.Localization(__file__, 520, 24), pop_219367, *[unicode_219368, None_219369], **kwargs_219370)
        
        # Assigning a type to the variable 'user_metadata' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'user_metadata', pop_call_result_219371)
        
        # Type idiom detected: calculating its left and rigth part (line 521)
        # Getting the type of 'user_metadata' (line 521)
        user_metadata_219372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'user_metadata')
        # Getting the type of 'None' (line 521)
        None_219373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 32), 'None')
        
        (may_be_219374, more_types_in_union_219375) = may_not_be_none(user_metadata_219372, None_219373)

        if may_be_219374:

            if more_types_in_union_219375:
                # Runtime conditional SSA (line 521)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to update(...): (line 522)
            # Processing the call arguments (line 522)
            # Getting the type of 'user_metadata' (line 522)
            user_metadata_219378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 28), 'user_metadata', False)
            # Processing the call keyword arguments (line 522)
            kwargs_219379 = {}
            # Getting the type of 'metadata' (line 522)
            metadata_219376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'metadata', False)
            # Obtaining the member 'update' of a type (line 522)
            update_219377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 12), metadata_219376, 'update')
            # Calling update(args, kwargs) (line 522)
            update_call_result_219380 = invoke(stypy.reporting.localization.Localization(__file__, 522, 12), update_219377, *[user_metadata_219378], **kwargs_219379)
            

            if more_types_in_union_219375:
                # SSA join for if statement (line 521)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Try-finally block (line 524)
        
        # Call to write_png(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'renderer' (line 525)
        renderer_219383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 27), 'renderer', False)
        # Obtaining the member '_renderer' of a type (line 525)
        _renderer_219384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 27), renderer_219383, '_renderer')
        # Getting the type of 'filename_or_obj' (line 525)
        filename_or_obj_219385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 47), 'filename_or_obj', False)
        # Getting the type of 'self' (line 526)
        self_219386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 27), 'self', False)
        # Obtaining the member 'figure' of a type (line 526)
        figure_219387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 27), self_219386, 'figure')
        # Obtaining the member 'dpi' of a type (line 526)
        dpi_219388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 27), figure_219387, 'dpi')
        # Processing the call keyword arguments (line 525)
        # Getting the type of 'metadata' (line 526)
        metadata_219389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 53), 'metadata', False)
        keyword_219390 = metadata_219389
        kwargs_219391 = {'metadata': keyword_219390}
        # Getting the type of '_png' (line 525)
        _png_219381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), '_png', False)
        # Obtaining the member 'write_png' of a type (line 525)
        write_png_219382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 12), _png_219381, 'write_png')
        # Calling write_png(args, kwargs) (line 525)
        write_png_call_result_219392 = invoke(stypy.reporting.localization.Localization(__file__, 525, 12), write_png_219382, *[_renderer_219384, filename_or_obj_219385, dpi_219388], **kwargs_219391)
        
        
        # finally branch of the try-finally block (line 524)
        
        # Getting the type of 'close' (line 528)
        close_219393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 'close')
        # Testing the type of an if condition (line 528)
        if_condition_219394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 12), close_219393)
        # Assigning a type to the variable 'if_condition_219394' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'if_condition_219394', if_condition_219394)
        # SSA begins for if statement (line 528)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 529)
        # Processing the call keyword arguments (line 529)
        kwargs_219397 = {}
        # Getting the type of 'filename_or_obj' (line 529)
        filename_or_obj_219395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 16), 'filename_or_obj', False)
        # Obtaining the member 'close' of a type (line 529)
        close_219396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 16), filename_or_obj_219395, 'close')
        # Calling close(args, kwargs) (line 529)
        close_call_result_219398 = invoke(stypy.reporting.localization.Localization(__file__, 529, 16), close_219396, *[], **kwargs_219397)
        
        # SSA join for if statement (line 528)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 530):
        
        # Assigning a Name to a Attribute (line 530):
        # Getting the type of 'original_dpi' (line 530)
        original_dpi_219399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 27), 'original_dpi')
        # Getting the type of 'renderer' (line 530)
        renderer_219400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'renderer')
        # Setting the type of the member 'dpi' of a type (line 530)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 12), renderer_219400, 'dpi', original_dpi_219399)
        
        
        # ################# End of 'print_png(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_png' in the type store
        # Getting the type of 'stypy_return_type' (line 506)
        stypy_return_type_219401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219401)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_png'
        return stypy_return_type_219401


    @norecursion
    def print_to_buffer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_to_buffer'
        module_type_store = module_type_store.open_function_context('print_to_buffer', 532, 4, False)
        # Assigning a type to the variable 'self' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasAgg.print_to_buffer')
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasAgg.print_to_buffer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.print_to_buffer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_to_buffer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_to_buffer(...)' code ##################

        
        # Call to draw(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'self' (line 533)
        self_219404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 29), 'self', False)
        # Processing the call keyword arguments (line 533)
        kwargs_219405 = {}
        # Getting the type of 'FigureCanvasAgg' (line 533)
        FigureCanvasAgg_219402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'FigureCanvasAgg', False)
        # Obtaining the member 'draw' of a type (line 533)
        draw_219403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), FigureCanvasAgg_219402, 'draw')
        # Calling draw(args, kwargs) (line 533)
        draw_call_result_219406 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), draw_219403, *[self_219404], **kwargs_219405)
        
        
        # Assigning a Call to a Name (line 534):
        
        # Assigning a Call to a Name (line 534):
        
        # Call to get_renderer(...): (line 534)
        # Processing the call keyword arguments (line 534)
        kwargs_219409 = {}
        # Getting the type of 'self' (line 534)
        self_219407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 19), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 534)
        get_renderer_219408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 19), self_219407, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 534)
        get_renderer_call_result_219410 = invoke(stypy.reporting.localization.Localization(__file__, 534, 19), get_renderer_219408, *[], **kwargs_219409)
        
        # Assigning a type to the variable 'renderer' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'renderer', get_renderer_call_result_219410)
        
        # Assigning a Attribute to a Name (line 535):
        
        # Assigning a Attribute to a Name (line 535):
        # Getting the type of 'renderer' (line 535)
        renderer_219411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 23), 'renderer')
        # Obtaining the member 'dpi' of a type (line 535)
        dpi_219412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 23), renderer_219411, 'dpi')
        # Assigning a type to the variable 'original_dpi' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'original_dpi', dpi_219412)
        
        # Assigning a Attribute to a Attribute (line 536):
        
        # Assigning a Attribute to a Attribute (line 536):
        # Getting the type of 'self' (line 536)
        self_219413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 23), 'self')
        # Obtaining the member 'figure' of a type (line 536)
        figure_219414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 23), self_219413, 'figure')
        # Obtaining the member 'dpi' of a type (line 536)
        dpi_219415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 23), figure_219414, 'dpi')
        # Getting the type of 'renderer' (line 536)
        renderer_219416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'renderer')
        # Setting the type of the member 'dpi' of a type (line 536)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), renderer_219416, 'dpi', dpi_219415)
        
        # Try-finally block (line 537)
        
        # Assigning a Tuple to a Name (line 538):
        
        # Assigning a Tuple to a Name (line 538):
        
        # Obtaining an instance of the builtin type 'tuple' (line 538)
        tuple_219417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 538)
        # Adding element type (line 538)
        
        # Call to buffer_rgba(...): (line 538)
        # Processing the call keyword arguments (line 538)
        kwargs_219421 = {}
        # Getting the type of 'renderer' (line 538)
        renderer_219418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 22), 'renderer', False)
        # Obtaining the member '_renderer' of a type (line 538)
        _renderer_219419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 22), renderer_219418, '_renderer')
        # Obtaining the member 'buffer_rgba' of a type (line 538)
        buffer_rgba_219420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 22), _renderer_219419, 'buffer_rgba')
        # Calling buffer_rgba(args, kwargs) (line 538)
        buffer_rgba_call_result_219422 = invoke(stypy.reporting.localization.Localization(__file__, 538, 22), buffer_rgba_219420, *[], **kwargs_219421)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 22), tuple_219417, buffer_rgba_call_result_219422)
        # Adding element type (line 538)
        
        # Obtaining an instance of the builtin type 'tuple' (line 539)
        tuple_219423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 539)
        # Adding element type (line 539)
        
        # Call to int(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'renderer' (line 539)
        renderer_219425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 27), 'renderer', False)
        # Obtaining the member 'width' of a type (line 539)
        width_219426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 27), renderer_219425, 'width')
        # Processing the call keyword arguments (line 539)
        kwargs_219427 = {}
        # Getting the type of 'int' (line 539)
        int_219424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 23), 'int', False)
        # Calling int(args, kwargs) (line 539)
        int_call_result_219428 = invoke(stypy.reporting.localization.Localization(__file__, 539, 23), int_219424, *[width_219426], **kwargs_219427)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 23), tuple_219423, int_call_result_219428)
        # Adding element type (line 539)
        
        # Call to int(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'renderer' (line 539)
        renderer_219430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 48), 'renderer', False)
        # Obtaining the member 'height' of a type (line 539)
        height_219431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 48), renderer_219430, 'height')
        # Processing the call keyword arguments (line 539)
        kwargs_219432 = {}
        # Getting the type of 'int' (line 539)
        int_219429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 44), 'int', False)
        # Calling int(args, kwargs) (line 539)
        int_call_result_219433 = invoke(stypy.reporting.localization.Localization(__file__, 539, 44), int_219429, *[height_219431], **kwargs_219432)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 23), tuple_219423, int_call_result_219433)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 22), tuple_219417, tuple_219423)
        
        # Assigning a type to the variable 'result' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'result', tuple_219417)
        
        # finally branch of the try-finally block (line 537)
        
        # Assigning a Name to a Attribute (line 541):
        
        # Assigning a Name to a Attribute (line 541):
        # Getting the type of 'original_dpi' (line 541)
        original_dpi_219434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 27), 'original_dpi')
        # Getting the type of 'renderer' (line 541)
        renderer_219435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'renderer')
        # Setting the type of the member 'dpi' of a type (line 541)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 12), renderer_219435, 'dpi', original_dpi_219434)
        
        # Getting the type of 'result' (line 542)
        result_219436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'stypy_return_type', result_219436)
        
        # ################# End of 'print_to_buffer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_to_buffer' in the type store
        # Getting the type of 'stypy_return_type' (line 532)
        stypy_return_type_219437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219437)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_to_buffer'
        return stypy_return_type_219437


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 398, 0, False)
        # Assigning a type to the variable 'self' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasAgg' (line 398)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 0), 'FigureCanvasAgg', FigureCanvasAgg)

# Assigning a Name to a Name (line 504):
# Getting the type of 'FigureCanvasAgg'
FigureCanvasAgg_219438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasAgg')
# Obtaining the member 'print_raw' of a type
print_raw_219439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasAgg_219438, 'print_raw')
# Getting the type of 'FigureCanvasAgg'
FigureCanvasAgg_219440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasAgg')
# Setting the type of the member 'print_rgba' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasAgg_219440, 'print_rgba', print_raw_219439)

# Assigning a Name to a Name (line 504):

# Getting the type of '_has_pil' (line 544)
_has_pil_219441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 7), '_has_pil')
# Testing the type of an if condition (line 544)
if_condition_219442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 4), _has_pil_219441)
# Assigning a type to the variable 'if_condition_219442' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'if_condition_219442', if_condition_219442)
# SSA begins for if statement (line 544)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def print_jpg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_jpg'
    module_type_store = module_type_store.open_function_context('print_jpg', 546, 8, False)
    
    # Passed parameters checking function
    print_jpg.stypy_localization = localization
    print_jpg.stypy_type_of_self = None
    print_jpg.stypy_type_store = module_type_store
    print_jpg.stypy_function_name = 'print_jpg'
    print_jpg.stypy_param_names_list = ['self', 'filename_or_obj']
    print_jpg.stypy_varargs_param_name = 'args'
    print_jpg.stypy_kwargs_param_name = 'kwargs'
    print_jpg.stypy_call_defaults = defaults
    print_jpg.stypy_call_varargs = varargs
    print_jpg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_jpg', ['self', 'filename_or_obj'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_jpg', localization, ['self', 'filename_or_obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_jpg(...)' code ##################

    unicode_219443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, (-1)), 'unicode', u'\n            Other Parameters\n            ----------------\n            quality : int\n                The image quality, on a scale from 1 (worst) to\n                95 (best). The default is 95, if not given in the\n                matplotlibrc file in the savefig.jpeg_quality parameter.\n                Values above 95 should be avoided; 100 completely\n                disables the JPEG quantization stage.\n\n            optimize : bool\n                If present, indicates that the encoder should\n                make an extra pass over the image in order to select\n                optimal encoder settings.\n\n            progressive : bool\n                If present, indicates that this image\n                should be stored as a progressive JPEG file.\n            ')
    
    # Assigning a Call to a Tuple (line 566):
    
    # Assigning a Call to a Name:
    
    # Call to print_to_buffer(...): (line 566)
    # Processing the call keyword arguments (line 566)
    kwargs_219446 = {}
    # Getting the type of 'self' (line 566)
    self_219444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 24), 'self', False)
    # Obtaining the member 'print_to_buffer' of a type (line 566)
    print_to_buffer_219445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 24), self_219444, 'print_to_buffer')
    # Calling print_to_buffer(args, kwargs) (line 566)
    print_to_buffer_call_result_219447 = invoke(stypy.reporting.localization.Localization(__file__, 566, 24), print_to_buffer_219445, *[], **kwargs_219446)
    
    # Assigning a type to the variable 'call_assignment_217830' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'call_assignment_217830', print_to_buffer_call_result_219447)
    
    # Assigning a Call to a Name (line 566):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_219450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 12), 'int')
    # Processing the call keyword arguments
    kwargs_219451 = {}
    # Getting the type of 'call_assignment_217830' (line 566)
    call_assignment_217830_219448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'call_assignment_217830', False)
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___219449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 12), call_assignment_217830_219448, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_219452 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___219449, *[int_219450], **kwargs_219451)
    
    # Assigning a type to the variable 'call_assignment_217831' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'call_assignment_217831', getitem___call_result_219452)
    
    # Assigning a Name to a Name (line 566):
    # Getting the type of 'call_assignment_217831' (line 566)
    call_assignment_217831_219453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'call_assignment_217831')
    # Assigning a type to the variable 'buf' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'buf', call_assignment_217831_219453)
    
    # Assigning a Call to a Name (line 566):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_219456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 12), 'int')
    # Processing the call keyword arguments
    kwargs_219457 = {}
    # Getting the type of 'call_assignment_217830' (line 566)
    call_assignment_217830_219454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'call_assignment_217830', False)
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___219455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 12), call_assignment_217830_219454, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_219458 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___219455, *[int_219456], **kwargs_219457)
    
    # Assigning a type to the variable 'call_assignment_217832' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'call_assignment_217832', getitem___call_result_219458)
    
    # Assigning a Name to a Name (line 566):
    # Getting the type of 'call_assignment_217832' (line 566)
    call_assignment_217832_219459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'call_assignment_217832')
    # Assigning a type to the variable 'size' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 17), 'size', call_assignment_217832_219459)
    
    
    # Call to pop(...): (line 567)
    # Processing the call arguments (line 567)
    unicode_219462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 26), 'unicode', u'dryrun')
    # Getting the type of 'False' (line 567)
    False_219463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 36), 'False', False)
    # Processing the call keyword arguments (line 567)
    kwargs_219464 = {}
    # Getting the type of 'kwargs' (line 567)
    kwargs_219460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 567)
    pop_219461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 15), kwargs_219460, 'pop')
    # Calling pop(args, kwargs) (line 567)
    pop_call_result_219465 = invoke(stypy.reporting.localization.Localization(__file__, 567, 15), pop_219461, *[unicode_219462, False_219463], **kwargs_219464)
    
    # Testing the type of an if condition (line 567)
    if_condition_219466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 567, 12), pop_call_result_219465)
    # Assigning a type to the variable 'if_condition_219466' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'if_condition_219466', if_condition_219466)
    # SSA begins for if statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 16), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 567)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 571):
    
    # Assigning a Call to a Name (line 571):
    
    # Call to frombuffer(...): (line 571)
    # Processing the call arguments (line 571)
    unicode_219469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 37), 'unicode', u'RGBA')
    # Getting the type of 'size' (line 571)
    size_219470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 45), 'size', False)
    # Getting the type of 'buf' (line 571)
    buf_219471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 51), 'buf', False)
    unicode_219472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 56), 'unicode', u'raw')
    unicode_219473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 63), 'unicode', u'RGBA')
    int_219474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 71), 'int')
    int_219475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 74), 'int')
    # Processing the call keyword arguments (line 571)
    kwargs_219476 = {}
    # Getting the type of 'Image' (line 571)
    Image_219467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'Image', False)
    # Obtaining the member 'frombuffer' of a type (line 571)
    frombuffer_219468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 20), Image_219467, 'frombuffer')
    # Calling frombuffer(args, kwargs) (line 571)
    frombuffer_call_result_219477 = invoke(stypy.reporting.localization.Localization(__file__, 571, 20), frombuffer_219468, *[unicode_219469, size_219470, buf_219471, unicode_219472, unicode_219473, int_219474, int_219475], **kwargs_219476)
    
    # Assigning a type to the variable 'image' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'image', frombuffer_call_result_219477)
    
    # Assigning a Call to a Name (line 572):
    
    # Assigning a Call to a Name (line 572):
    
    # Call to to_rgba(...): (line 572)
    # Processing the call arguments (line 572)
    
    # Obtaining the type of the subscript
    unicode_219480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 44), 'unicode', u'savefig.facecolor')
    # Getting the type of 'rcParams' (line 572)
    rcParams_219481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 35), 'rcParams', False)
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___219482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 35), rcParams_219481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_219483 = invoke(stypy.reporting.localization.Localization(__file__, 572, 35), getitem___219482, unicode_219480)
    
    # Processing the call keyword arguments (line 572)
    kwargs_219484 = {}
    # Getting the type of 'mcolors' (line 572)
    mcolors_219478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 19), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 572)
    to_rgba_219479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 19), mcolors_219478, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 572)
    to_rgba_call_result_219485 = invoke(stypy.reporting.localization.Localization(__file__, 572, 19), to_rgba_219479, *[subscript_call_result_219483], **kwargs_219484)
    
    # Assigning a type to the variable 'rgba' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'rgba', to_rgba_call_result_219485)
    
    # Assigning a Call to a Name (line 573):
    
    # Assigning a Call to a Name (line 573):
    
    # Call to tuple(...): (line 573)
    # Processing the call arguments (line 573)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_219493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 57), 'int')
    slice_219494 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 573, 51), None, int_219493, None)
    # Getting the type of 'rgba' (line 573)
    rgba_219495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 51), 'rgba', False)
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___219496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 51), rgba_219495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 573)
    subscript_call_result_219497 = invoke(stypy.reporting.localization.Localization(__file__, 573, 51), getitem___219496, slice_219494)
    
    comprehension_219498 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 27), subscript_call_result_219497)
    # Assigning a type to the variable 'x' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 27), 'x', comprehension_219498)
    
    # Call to int(...): (line 573)
    # Processing the call arguments (line 573)
    # Getting the type of 'x' (line 573)
    x_219488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 31), 'x', False)
    float_219489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 35), 'float')
    # Applying the binary operator '*' (line 573)
    result_mul_219490 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 31), '*', x_219488, float_219489)
    
    # Processing the call keyword arguments (line 573)
    kwargs_219491 = {}
    # Getting the type of 'int' (line 573)
    int_219487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 27), 'int', False)
    # Calling int(args, kwargs) (line 573)
    int_call_result_219492 = invoke(stypy.reporting.localization.Localization(__file__, 573, 27), int_219487, *[result_mul_219490], **kwargs_219491)
    
    list_219499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 27), list_219499, int_call_result_219492)
    # Processing the call keyword arguments (line 573)
    kwargs_219500 = {}
    # Getting the type of 'tuple' (line 573)
    tuple_219486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 573)
    tuple_call_result_219501 = invoke(stypy.reporting.localization.Localization(__file__, 573, 20), tuple_219486, *[list_219499], **kwargs_219500)
    
    # Assigning a type to the variable 'color' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'color', tuple_call_result_219501)
    
    # Assigning a Call to a Name (line 574):
    
    # Assigning a Call to a Name (line 574):
    
    # Call to new(...): (line 574)
    # Processing the call arguments (line 574)
    unicode_219504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 35), 'unicode', u'RGB')
    # Getting the type of 'size' (line 574)
    size_219505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'size', False)
    # Getting the type of 'color' (line 574)
    color_219506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 48), 'color', False)
    # Processing the call keyword arguments (line 574)
    kwargs_219507 = {}
    # Getting the type of 'Image' (line 574)
    Image_219502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 25), 'Image', False)
    # Obtaining the member 'new' of a type (line 574)
    new_219503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 25), Image_219502, 'new')
    # Calling new(args, kwargs) (line 574)
    new_call_result_219508 = invoke(stypy.reporting.localization.Localization(__file__, 574, 25), new_219503, *[unicode_219504, size_219505, color_219506], **kwargs_219507)
    
    # Assigning a type to the variable 'background' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'background', new_call_result_219508)
    
    # Call to paste(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'image' (line 575)
    image_219511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 29), 'image', False)
    # Getting the type of 'image' (line 575)
    image_219512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 36), 'image', False)
    # Processing the call keyword arguments (line 575)
    kwargs_219513 = {}
    # Getting the type of 'background' (line 575)
    background_219509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'background', False)
    # Obtaining the member 'paste' of a type (line 575)
    paste_219510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), background_219509, 'paste')
    # Calling paste(args, kwargs) (line 575)
    paste_call_result_219514 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), paste_219510, *[image_219511, image_219512], **kwargs_219513)
    
    
    # Assigning a DictComp to a Name (line 576):
    
    # Assigning a DictComp to a Name (line 576):
    # Calculating dict comprehension
    module_type_store = module_type_store.open_function_context('dict comprehension expression', 576, 23, True)
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 577)
    list_219523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 577)
    # Adding element type (line 577)
    unicode_219524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 33), 'unicode', u'quality')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 32), list_219523, unicode_219524)
    # Adding element type (line 577)
    unicode_219525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 44), 'unicode', u'optimize')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 32), list_219523, unicode_219525)
    # Adding element type (line 577)
    unicode_219526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 56), 'unicode', u'progressive')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 32), list_219523, unicode_219526)
    # Adding element type (line 577)
    unicode_219527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 71), 'unicode', u'dpi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 32), list_219523, unicode_219527)
    
    comprehension_219528 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 23), list_219523)
    # Assigning a type to the variable 'k' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 23), 'k', comprehension_219528)
    
    # Getting the type of 'k' (line 578)
    k_219520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 26), 'k')
    # Getting the type of 'kwargs' (line 578)
    kwargs_219521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 31), 'kwargs')
    # Applying the binary operator 'in' (line 578)
    result_contains_219522 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 26), 'in', k_219520, kwargs_219521)
    
    # Getting the type of 'k' (line 576)
    k_219515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 23), 'k')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 576)
    k_219516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 33), 'k')
    # Getting the type of 'kwargs' (line 576)
    kwargs_219517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 26), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 576)
    getitem___219518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 26), kwargs_219517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 576)
    subscript_call_result_219519 = invoke(stypy.reporting.localization.Localization(__file__, 576, 26), getitem___219518, k_219516)
    
    dict_219529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 23), 'dict')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 23), dict_219529, (k_219515, subscript_call_result_219519))
    # Assigning a type to the variable 'options' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'options', dict_219529)
    
    # Call to setdefault(...): (line 579)
    # Processing the call arguments (line 579)
    unicode_219532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 31), 'unicode', u'quality')
    
    # Obtaining the type of the subscript
    unicode_219533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 51), 'unicode', u'savefig.jpeg_quality')
    # Getting the type of 'rcParams' (line 579)
    rcParams_219534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 42), 'rcParams', False)
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___219535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 42), rcParams_219534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_219536 = invoke(stypy.reporting.localization.Localization(__file__, 579, 42), getitem___219535, unicode_219533)
    
    # Processing the call keyword arguments (line 579)
    kwargs_219537 = {}
    # Getting the type of 'options' (line 579)
    options_219530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'options', False)
    # Obtaining the member 'setdefault' of a type (line 579)
    setdefault_219531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 12), options_219530, 'setdefault')
    # Calling setdefault(args, kwargs) (line 579)
    setdefault_call_result_219538 = invoke(stypy.reporting.localization.Localization(__file__, 579, 12), setdefault_219531, *[unicode_219532, subscript_call_result_219536], **kwargs_219537)
    
    
    
    unicode_219539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 15), 'unicode', u'dpi')
    # Getting the type of 'options' (line 580)
    options_219540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 24), 'options')
    # Applying the binary operator 'in' (line 580)
    result_contains_219541 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 15), 'in', unicode_219539, options_219540)
    
    # Testing the type of an if condition (line 580)
    if_condition_219542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 12), result_contains_219541)
    # Assigning a type to the variable 'if_condition_219542' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'if_condition_219542', if_condition_219542)
    # SSA begins for if statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Subscript (line 582):
    
    # Assigning a Tuple to a Subscript (line 582):
    
    # Obtaining an instance of the builtin type 'tuple' (line 582)
    tuple_219543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 582)
    # Adding element type (line 582)
    
    # Obtaining the type of the subscript
    unicode_219544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 42), 'unicode', u'dpi')
    # Getting the type of 'options' (line 582)
    options_219545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 34), 'options')
    # Obtaining the member '__getitem__' of a type (line 582)
    getitem___219546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 34), options_219545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 582)
    subscript_call_result_219547 = invoke(stypy.reporting.localization.Localization(__file__, 582, 34), getitem___219546, unicode_219544)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 34), tuple_219543, subscript_call_result_219547)
    # Adding element type (line 582)
    
    # Obtaining the type of the subscript
    unicode_219548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 58), 'unicode', u'dpi')
    # Getting the type of 'options' (line 582)
    options_219549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 50), 'options')
    # Obtaining the member '__getitem__' of a type (line 582)
    getitem___219550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 50), options_219549, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 582)
    subscript_call_result_219551 = invoke(stypy.reporting.localization.Localization(__file__, 582, 50), getitem___219550, unicode_219548)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 34), tuple_219543, subscript_call_result_219551)
    
    # Getting the type of 'options' (line 582)
    options_219552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'options')
    unicode_219553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 24), 'unicode', u'dpi')
    # Storing an element on a container (line 582)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 16), options_219552, (unicode_219553, tuple_219543))
    # SSA join for if statement (line 580)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to save(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'filename_or_obj' (line 584)
    filename_or_obj_219556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 35), 'filename_or_obj', False)
    # Processing the call keyword arguments (line 584)
    unicode_219557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 59), 'unicode', u'jpeg')
    keyword_219558 = unicode_219557
    # Getting the type of 'options' (line 584)
    options_219559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 69), 'options', False)
    kwargs_219560 = {'options_219559': options_219559, 'format': keyword_219558}
    # Getting the type of 'background' (line 584)
    background_219554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 19), 'background', False)
    # Obtaining the member 'save' of a type (line 584)
    save_219555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 19), background_219554, 'save')
    # Calling save(args, kwargs) (line 584)
    save_call_result_219561 = invoke(stypy.reporting.localization.Localization(__file__, 584, 19), save_219555, *[filename_or_obj_219556], **kwargs_219560)
    
    # Assigning a type to the variable 'stypy_return_type' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'stypy_return_type', save_call_result_219561)
    
    # ################# End of 'print_jpg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_jpg' in the type store
    # Getting the type of 'stypy_return_type' (line 546)
    stypy_return_type_219562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_219562)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_jpg'
    return stypy_return_type_219562

# Assigning a type to the variable 'print_jpg' (line 546)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'print_jpg', print_jpg)

# Assigning a Name to a Name (line 585):

# Assigning a Name to a Name (line 585):
# Getting the type of 'print_jpg' (line 585)
print_jpg_219563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 21), 'print_jpg')
# Assigning a type to the variable 'print_jpeg' (line 585)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'print_jpeg', print_jpg_219563)

@norecursion
def print_tif(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_tif'
    module_type_store = module_type_store.open_function_context('print_tif', 588, 8, False)
    
    # Passed parameters checking function
    print_tif.stypy_localization = localization
    print_tif.stypy_type_of_self = None
    print_tif.stypy_type_store = module_type_store
    print_tif.stypy_function_name = 'print_tif'
    print_tif.stypy_param_names_list = ['self', 'filename_or_obj']
    print_tif.stypy_varargs_param_name = 'args'
    print_tif.stypy_kwargs_param_name = 'kwargs'
    print_tif.stypy_call_defaults = defaults
    print_tif.stypy_call_varargs = varargs
    print_tif.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_tif', ['self', 'filename_or_obj'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_tif', localization, ['self', 'filename_or_obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_tif(...)' code ##################

    
    # Assigning a Call to a Tuple (line 589):
    
    # Assigning a Call to a Name:
    
    # Call to print_to_buffer(...): (line 589)
    # Processing the call keyword arguments (line 589)
    kwargs_219566 = {}
    # Getting the type of 'self' (line 589)
    self_219564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'self', False)
    # Obtaining the member 'print_to_buffer' of a type (line 589)
    print_to_buffer_219565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 24), self_219564, 'print_to_buffer')
    # Calling print_to_buffer(args, kwargs) (line 589)
    print_to_buffer_call_result_219567 = invoke(stypy.reporting.localization.Localization(__file__, 589, 24), print_to_buffer_219565, *[], **kwargs_219566)
    
    # Assigning a type to the variable 'call_assignment_217833' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'call_assignment_217833', print_to_buffer_call_result_219567)
    
    # Assigning a Call to a Name (line 589):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_219570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 12), 'int')
    # Processing the call keyword arguments
    kwargs_219571 = {}
    # Getting the type of 'call_assignment_217833' (line 589)
    call_assignment_217833_219568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'call_assignment_217833', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___219569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 12), call_assignment_217833_219568, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_219572 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___219569, *[int_219570], **kwargs_219571)
    
    # Assigning a type to the variable 'call_assignment_217834' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'call_assignment_217834', getitem___call_result_219572)
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'call_assignment_217834' (line 589)
    call_assignment_217834_219573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'call_assignment_217834')
    # Assigning a type to the variable 'buf' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'buf', call_assignment_217834_219573)
    
    # Assigning a Call to a Name (line 589):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_219576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 12), 'int')
    # Processing the call keyword arguments
    kwargs_219577 = {}
    # Getting the type of 'call_assignment_217833' (line 589)
    call_assignment_217833_219574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'call_assignment_217833', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___219575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 12), call_assignment_217833_219574, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_219578 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___219575, *[int_219576], **kwargs_219577)
    
    # Assigning a type to the variable 'call_assignment_217835' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'call_assignment_217835', getitem___call_result_219578)
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'call_assignment_217835' (line 589)
    call_assignment_217835_219579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'call_assignment_217835')
    # Assigning a type to the variable 'size' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 17), 'size', call_assignment_217835_219579)
    
    
    # Call to pop(...): (line 590)
    # Processing the call arguments (line 590)
    unicode_219582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 26), 'unicode', u'dryrun')
    # Getting the type of 'False' (line 590)
    False_219583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 36), 'False', False)
    # Processing the call keyword arguments (line 590)
    kwargs_219584 = {}
    # Getting the type of 'kwargs' (line 590)
    kwargs_219580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 15), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 590)
    pop_219581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 15), kwargs_219580, 'pop')
    # Calling pop(args, kwargs) (line 590)
    pop_call_result_219585 = invoke(stypy.reporting.localization.Localization(__file__, 590, 15), pop_219581, *[unicode_219582, False_219583], **kwargs_219584)
    
    # Testing the type of an if condition (line 590)
    if_condition_219586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 590, 12), pop_call_result_219585)
    # Assigning a type to the variable 'if_condition_219586' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'if_condition_219586', if_condition_219586)
    # SSA begins for if statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 590)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 592):
    
    # Assigning a Call to a Name (line 592):
    
    # Call to frombuffer(...): (line 592)
    # Processing the call arguments (line 592)
    unicode_219589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 37), 'unicode', u'RGBA')
    # Getting the type of 'size' (line 592)
    size_219590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 45), 'size', False)
    # Getting the type of 'buf' (line 592)
    buf_219591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 51), 'buf', False)
    unicode_219592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 56), 'unicode', u'raw')
    unicode_219593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 63), 'unicode', u'RGBA')
    int_219594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 71), 'int')
    int_219595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 74), 'int')
    # Processing the call keyword arguments (line 592)
    kwargs_219596 = {}
    # Getting the type of 'Image' (line 592)
    Image_219587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 20), 'Image', False)
    # Obtaining the member 'frombuffer' of a type (line 592)
    frombuffer_219588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 20), Image_219587, 'frombuffer')
    # Calling frombuffer(args, kwargs) (line 592)
    frombuffer_call_result_219597 = invoke(stypy.reporting.localization.Localization(__file__, 592, 20), frombuffer_219588, *[unicode_219589, size_219590, buf_219591, unicode_219592, unicode_219593, int_219594, int_219595], **kwargs_219596)
    
    # Assigning a type to the variable 'image' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'image', frombuffer_call_result_219597)
    
    # Assigning a Tuple to a Name (line 593):
    
    # Assigning a Tuple to a Name (line 593):
    
    # Obtaining an instance of the builtin type 'tuple' (line 593)
    tuple_219598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 593)
    # Adding element type (line 593)
    # Getting the type of 'self' (line 593)
    self_219599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 19), 'self')
    # Obtaining the member 'figure' of a type (line 593)
    figure_219600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 19), self_219599, 'figure')
    # Obtaining the member 'dpi' of a type (line 593)
    dpi_219601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 19), figure_219600, 'dpi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 19), tuple_219598, dpi_219601)
    # Adding element type (line 593)
    # Getting the type of 'self' (line 593)
    self_219602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 36), 'self')
    # Obtaining the member 'figure' of a type (line 593)
    figure_219603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 36), self_219602, 'figure')
    # Obtaining the member 'dpi' of a type (line 593)
    dpi_219604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 36), figure_219603, 'dpi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 19), tuple_219598, dpi_219604)
    
    # Assigning a type to the variable 'dpi' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'dpi', tuple_219598)
    
    # Call to save(...): (line 594)
    # Processing the call arguments (line 594)
    # Getting the type of 'filename_or_obj' (line 594)
    filename_or_obj_219607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 30), 'filename_or_obj', False)
    # Processing the call keyword arguments (line 594)
    unicode_219608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 54), 'unicode', u'tiff')
    keyword_219609 = unicode_219608
    # Getting the type of 'dpi' (line 595)
    dpi_219610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 34), 'dpi', False)
    keyword_219611 = dpi_219610
    kwargs_219612 = {'dpi': keyword_219611, 'format': keyword_219609}
    # Getting the type of 'image' (line 594)
    image_219605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 19), 'image', False)
    # Obtaining the member 'save' of a type (line 594)
    save_219606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 19), image_219605, 'save')
    # Calling save(args, kwargs) (line 594)
    save_call_result_219613 = invoke(stypy.reporting.localization.Localization(__file__, 594, 19), save_219606, *[filename_or_obj_219607], **kwargs_219612)
    
    # Assigning a type to the variable 'stypy_return_type' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'stypy_return_type', save_call_result_219613)
    
    # ################# End of 'print_tif(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_tif' in the type store
    # Getting the type of 'stypy_return_type' (line 588)
    stypy_return_type_219614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_219614)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_tif'
    return stypy_return_type_219614

# Assigning a type to the variable 'print_tif' (line 588)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'print_tif', print_tif)

# Assigning a Name to a Name (line 596):

# Assigning a Name to a Name (line 596):
# Getting the type of 'print_tif' (line 596)
print_tif_219615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 21), 'print_tif')
# Assigning a type to the variable 'print_tiff' (line 596)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'print_tiff', print_tif_219615)
# SSA join for if statement (line 544)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the '_BackendAgg' class
# Getting the type of '_Backend' (line 600)
_Backend_219616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 18), '_Backend')

class _BackendAgg(_Backend_219616, ):
    
    # Assigning a Name to a Name (line 601):
    
    # Assigning a Name to a Name (line 602):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 599, 0, False)
        # Assigning a type to the variable 'self' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendAgg' (line 599)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 0), '_BackendAgg', _BackendAgg)

# Assigning a Name to a Name (line 601):
# Getting the type of 'FigureCanvasAgg' (line 601)
FigureCanvasAgg_219617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 19), 'FigureCanvasAgg')
# Getting the type of '_BackendAgg'
_BackendAgg_219618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendAgg')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendAgg_219618, 'FigureCanvas', FigureCanvasAgg_219617)

# Assigning a Name to a Name (line 602):
# Getting the type of 'FigureManagerBase' (line 602)
FigureManagerBase_219619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 20), 'FigureManagerBase')
# Getting the type of '_BackendAgg'
_BackendAgg_219620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendAgg')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendAgg_219620, 'FigureManager', FigureManagerBase_219619)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
