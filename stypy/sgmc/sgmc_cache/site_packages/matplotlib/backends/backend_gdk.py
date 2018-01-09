
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import math
7: import os
8: import sys
9: import warnings
10: 
11: import gobject
12: import gtk; gdk = gtk.gdk
13: import pango
14: pygtk_version_required = (2,2,0)
15: if gtk.pygtk_version < pygtk_version_required:
16:     raise ImportError ("PyGTK %d.%d.%d is installed\n"
17:                       "PyGTK %d.%d.%d or later is required"
18:                       % (gtk.pygtk_version + pygtk_version_required))
19: del pygtk_version_required
20: 
21: import numpy as np
22: 
23: import matplotlib
24: from matplotlib import rcParams
25: from matplotlib._pylab_helpers import Gcf
26: from matplotlib.backend_bases import (
27:     _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
28:     RendererBase)
29: from matplotlib.cbook import warn_deprecated
30: from matplotlib.figure import Figure
31: from matplotlib.mathtext import MathTextParser
32: from matplotlib.transforms import Affine2D
33: from matplotlib.backends._backend_gdk import pixbuf_get_pixels_array
34: 
35: backend_version = "%d.%d.%d" % gtk.pygtk_version
36: 
37: # Image formats that this backend supports - for FileChooser and print_figure()
38: IMAGE_FORMAT = sorted(['bmp', 'eps', 'jpg', 'png', 'ps', 'svg']) # 'raw', 'rgb'
39: IMAGE_FORMAT_DEFAULT = 'png'
40: 
41: 
42: class RendererGDK(RendererBase):
43:     fontweights = {
44:         100          : pango.WEIGHT_ULTRALIGHT,
45:         200          : pango.WEIGHT_LIGHT,
46:         300          : pango.WEIGHT_LIGHT,
47:         400          : pango.WEIGHT_NORMAL,
48:         500          : pango.WEIGHT_NORMAL,
49:         600          : pango.WEIGHT_BOLD,
50:         700          : pango.WEIGHT_BOLD,
51:         800          : pango.WEIGHT_HEAVY,
52:         900          : pango.WEIGHT_ULTRABOLD,
53:         'ultralight' : pango.WEIGHT_ULTRALIGHT,
54:         'light'      : pango.WEIGHT_LIGHT,
55:         'normal'     : pango.WEIGHT_NORMAL,
56:         'medium'     : pango.WEIGHT_NORMAL,
57:         'semibold'   : pango.WEIGHT_BOLD,
58:         'bold'       : pango.WEIGHT_BOLD,
59:         'heavy'      : pango.WEIGHT_HEAVY,
60:         'ultrabold'  : pango.WEIGHT_ULTRABOLD,
61:         'black'      : pango.WEIGHT_ULTRABOLD,
62:                    }
63: 
64:     # cache for efficiency, these must be at class, not instance level
65:     layoutd = {}  # a map from text prop tups to pango layouts
66:     rotated = {}  # a map from text prop tups to rotated text pixbufs
67: 
68:     def __init__(self, gtkDA, dpi):
69:         # widget gtkDA is used for:
70:         #  '<widget>.create_pango_layout(s)'
71:         #  cmap line below)
72:         self.gtkDA = gtkDA
73:         self.dpi   = dpi
74:         self._cmap = gtkDA.get_colormap()
75:         self.mathtext_parser = MathTextParser("Agg")
76: 
77:     def set_pixmap (self, pixmap):
78:         self.gdkDrawable = pixmap
79: 
80:     def set_width_height (self, width, height):
81:         '''w,h is the figure w,h not the pixmap w,h
82:         '''
83:         self.width, self.height = width, height
84: 
85:     def draw_path(self, gc, path, transform, rgbFace=None):
86:         transform = transform + Affine2D(). \
87:             scale(1.0, -1.0).translate(0, self.height)
88:         polygons = path.to_polygons(transform, self.width, self.height)
89:         for polygon in polygons:
90:             # draw_polygon won't take an arbitrary sequence -- it must be a list
91:             # of tuples
92:             polygon = [(int(np.round(x)), int(np.round(y))) for x, y in polygon]
93:             if rgbFace is not None:
94:                 saveColor = gc.gdkGC.foreground
95:                 gc.gdkGC.foreground = gc.rgb_to_gdk_color(rgbFace)
96:                 self.gdkDrawable.draw_polygon(gc.gdkGC, True, polygon)
97:                 gc.gdkGC.foreground = saveColor
98:             if gc.gdkGC.line_width > 0:
99:                 self.gdkDrawable.draw_lines(gc.gdkGC, polygon)
100: 
101:     def draw_image(self, gc, x, y, im):
102:         bbox = gc.get_clip_rectangle()
103: 
104:         if bbox != None:
105:             l,b,w,h = bbox.bounds
106:             #rectangle = (int(l), self.height-int(b+h),
107:             #             int(w), int(h))
108:             # set clip rect?
109: 
110:         rows, cols = im.shape[:2]
111: 
112:         pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,
113:                                 has_alpha=True, bits_per_sample=8,
114:                                 width=cols, height=rows)
115: 
116:         array = pixbuf_get_pixels_array(pixbuf)
117:         array[:, :, :] = im[::-1]
118: 
119:         gc = self.new_gc()
120: 
121: 
122:         y = self.height-y-rows
123: 
124:         try: # new in 2.2
125:             # can use None instead of gc.gdkGC, if don't need clipping
126:             self.gdkDrawable.draw_pixbuf (gc.gdkGC, pixbuf, 0, 0,
127:                                           int(x), int(y), cols, rows,
128:                                           gdk.RGB_DITHER_NONE, 0, 0)
129:         except AttributeError:
130:             # deprecated in 2.2
131:             pixbuf.render_to_drawable(self.gdkDrawable, gc.gdkGC, 0, 0,
132:                                   int(x), int(y), cols, rows,
133:                                   gdk.RGB_DITHER_NONE, 0, 0)
134: 
135:     def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
136:         x, y = int(x), int(y)
137: 
138:         if x < 0 or y < 0: # window has shrunk and text is off the edge
139:             return
140: 
141:         if angle not in (0,90):
142:             warnings.warn('backend_gdk: unable to draw text at angles ' +
143:                           'other than 0 or 90')
144:         elif ismath:
145:             self._draw_mathtext(gc, x, y, s, prop, angle)
146: 
147:         elif angle==90:
148:             self._draw_rotated_text(gc, x, y, s, prop, angle)
149: 
150:         else:
151:             layout, inkRect, logicalRect = self._get_pango_layout(s, prop)
152:             l, b, w, h = inkRect
153:             if (x + w > self.width or y + h > self.height):
154:                 return
155: 
156:             self.gdkDrawable.draw_layout(gc.gdkGC, x, y-h-b, layout)
157: 
158:     def _draw_mathtext(self, gc, x, y, s, prop, angle):
159:         ox, oy, width, height, descent, font_image, used_characters = \
160:             self.mathtext_parser.parse(s, self.dpi, prop)
161: 
162:         if angle == 90:
163:             width, height = height, width
164:             x -= width
165:         y -= height
166: 
167:         imw = font_image.get_width()
168:         imh = font_image.get_height()
169: 
170:         pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, has_alpha=True,
171:                                 bits_per_sample=8, width=imw, height=imh)
172: 
173:         array = pixbuf_get_pixels_array(pixbuf)
174: 
175:         rgb = gc.get_rgb()
176:         array[:,:,0] = int(rgb[0]*255)
177:         array[:,:,1] = int(rgb[1]*255)
178:         array[:,:,2] = int(rgb[2]*255)
179:         array[:,:,3] = (
180:             np.fromstring(font_image.as_str(), np.uint8).reshape((imh, imw)))
181: 
182:         # can use None instead of gc.gdkGC, if don't need clipping
183:         self.gdkDrawable.draw_pixbuf(gc.gdkGC, pixbuf, 0, 0,
184:                                      int(x), int(y), imw, imh,
185:                                      gdk.RGB_DITHER_NONE, 0, 0)
186: 
187:     def _draw_rotated_text(self, gc, x, y, s, prop, angle):
188:         '''
189:         Draw the text rotated 90 degrees, other angles are not supported
190:         '''
191:         # this function (and its called functions) is a bottleneck
192:         # Pango 1.6 supports rotated text, but pygtk 2.4.0 does not yet have
193:         # wrapper functions
194:         # GTK+ 2.6 pixbufs support rotation
195: 
196:         gdrawable = self.gdkDrawable
197:         ggc = gc.gdkGC
198: 
199:         layout, inkRect, logicalRect = self._get_pango_layout(s, prop)
200:         l, b, w, h = inkRect
201:         x = int(x-h)
202:         y = int(y-w)
203: 
204:         if (x < 0 or y < 0 or # window has shrunk and text is off the edge
205:             x + w > self.width or y + h > self.height):
206:             return
207: 
208:         key = (x,y,s,angle,hash(prop))
209:         imageVert = self.rotated.get(key)
210:         if imageVert != None:
211:             gdrawable.draw_image(ggc, imageVert, 0, 0, x, y, h, w)
212:             return
213: 
214:         imageBack = gdrawable.get_image(x, y, w, h)
215:         imageVert = gdrawable.get_image(x, y, h, w)
216:         imageFlip = gtk.gdk.Image(type=gdk.IMAGE_FASTEST,
217:                                   visual=gdrawable.get_visual(),
218:                                   width=w, height=h)
219:         if imageFlip == None or imageBack == None or imageVert == None:
220:             warnings.warn("Could not renderer vertical text")
221:             return
222:         imageFlip.set_colormap(self._cmap)
223:         for i in range(w):
224:             for j in range(h):
225:                 imageFlip.put_pixel(i, j, imageVert.get_pixel(j,w-i-1) )
226: 
227:         gdrawable.draw_image(ggc, imageFlip, 0, 0, x, y, w, h)
228:         gdrawable.draw_layout(ggc, x, y-b, layout)
229: 
230:         imageIn  = gdrawable.get_image(x, y, w, h)
231:         for i in range(w):
232:             for j in range(h):
233:                 imageVert.put_pixel(j, i, imageIn.get_pixel(w-i-1,j) )
234: 
235:         gdrawable.draw_image(ggc, imageBack, 0, 0, x, y, w, h)
236:         gdrawable.draw_image(ggc, imageVert, 0, 0, x, y, h, w)
237:         self.rotated[key] = imageVert
238: 
239:     def _get_pango_layout(self, s, prop):
240:         '''
241:         Create a pango layout instance for Text 's' with properties 'prop'.
242:         Return - pango layout (from cache if already exists)
243: 
244:         Note that pango assumes a logical DPI of 96
245:         Ref: pango/fonts.c/pango_font_description_set_size() manual page
246:         '''
247:         # problem? - cache gets bigger and bigger, is never cleared out
248:         # two (not one) layouts are created for every text item s (then they
249:         # are cached) - why?
250: 
251:         key = self.dpi, s, hash(prop)
252:         value = self.layoutd.get(key)
253:         if value != None:
254:             return value
255: 
256:         size = prop.get_size_in_points() * self.dpi / 96.0
257:         size = np.round(size)
258: 
259:         font_str = '%s, %s %i' % (prop.get_name(), prop.get_style(), size,)
260:         font = pango.FontDescription(font_str)
261: 
262:         # later - add fontweight to font_str
263:         font.set_weight(self.fontweights[prop.get_weight()])
264: 
265:         layout = self.gtkDA.create_pango_layout(s)
266:         layout.set_font_description(font)
267:         inkRect, logicalRect = layout.get_pixel_extents()
268: 
269:         self.layoutd[key] = layout, inkRect, logicalRect
270:         return layout, inkRect, logicalRect
271: 
272:     def flipy(self):
273:         return True
274: 
275:     def get_canvas_width_height(self):
276:         return self.width, self.height
277: 
278:     def get_text_width_height_descent(self, s, prop, ismath):
279:         if ismath:
280:             ox, oy, width, height, descent, font_image, used_characters = \
281:                 self.mathtext_parser.parse(s, self.dpi, prop)
282:             return width, height, descent
283: 
284:         layout, inkRect, logicalRect = self._get_pango_layout(s, prop)
285:         l, b, w, h = inkRect
286:         ll, lb, lw, lh = logicalRect
287: 
288:         return w, h + 1, h - lh
289: 
290:     def new_gc(self):
291:         return GraphicsContextGDK(renderer=self)
292: 
293:     def points_to_pixels(self, points):
294:         return points/72.0 * self.dpi
295: 
296: 
297: class GraphicsContextGDK(GraphicsContextBase):
298:     # a cache shared by all class instances
299:     _cached = {}  # map: rgb color -> gdk.Color
300: 
301:     _joind = {
302:         'bevel' : gdk.JOIN_BEVEL,
303:         'miter' : gdk.JOIN_MITER,
304:         'round' : gdk.JOIN_ROUND,
305:         }
306: 
307:     _capd = {
308:         'butt'       : gdk.CAP_BUTT,
309:         'projecting' : gdk.CAP_PROJECTING,
310:         'round'      : gdk.CAP_ROUND,
311:         }
312: 
313: 
314:     def __init__(self, renderer):
315:         GraphicsContextBase.__init__(self)
316:         self.renderer = renderer
317:         self.gdkGC    = gtk.gdk.GC(renderer.gdkDrawable)
318:         self._cmap    = renderer._cmap
319: 
320: 
321:     def rgb_to_gdk_color(self, rgb):
322:         '''
323:         rgb - an RGB tuple (three 0.0-1.0 values)
324:         return an allocated gtk.gdk.Color
325:         '''
326:         try:
327:             return self._cached[tuple(rgb)]
328:         except KeyError:
329:             color = self._cached[tuple(rgb)] = \
330:                     self._cmap.alloc_color(
331:                         int(rgb[0]*65535),int(rgb[1]*65535),int(rgb[2]*65535))
332:             return color
333: 
334: 
335:     #def set_antialiased(self, b):
336:         # anti-aliasing is not supported by GDK
337: 
338:     def set_capstyle(self, cs):
339:         GraphicsContextBase.set_capstyle(self, cs)
340:         self.gdkGC.cap_style = self._capd[self._capstyle]
341: 
342: 
343:     def set_clip_rectangle(self, rectangle):
344:         GraphicsContextBase.set_clip_rectangle(self, rectangle)
345:         if rectangle is None:
346:             return
347:         l,b,w,h = rectangle.bounds
348:         rectangle = (int(l), self.renderer.height-int(b+h)+1,
349:                      int(w), int(h))
350:         #rectangle = (int(l), self.renderer.height-int(b+h),
351:         #             int(w+1), int(h+2))
352:         self.gdkGC.set_clip_rectangle(rectangle)
353: 
354:     def set_dashes(self, dash_offset, dash_list):
355:         GraphicsContextBase.set_dashes(self, dash_offset, dash_list)
356: 
357:         if dash_list == None:
358:             self.gdkGC.line_style = gdk.LINE_SOLID
359:         else:
360:             pixels = self.renderer.points_to_pixels(np.asarray(dash_list))
361:             dl = [max(1, int(np.round(val))) for val in pixels]
362:             self.gdkGC.set_dashes(dash_offset, dl)
363:             self.gdkGC.line_style = gdk.LINE_ON_OFF_DASH
364: 
365: 
366:     def set_foreground(self, fg, isRGBA=False):
367:         GraphicsContextBase.set_foreground(self, fg, isRGBA)
368:         self.gdkGC.foreground = self.rgb_to_gdk_color(self.get_rgb())
369: 
370: 
371:     def set_joinstyle(self, js):
372:         GraphicsContextBase.set_joinstyle(self, js)
373:         self.gdkGC.join_style = self._joind[self._joinstyle]
374: 
375: 
376:     def set_linewidth(self, w):
377:         GraphicsContextBase.set_linewidth(self, w)
378:         if w == 0:
379:             self.gdkGC.line_width = 0
380:         else:
381:             pixels = self.renderer.points_to_pixels(w)
382:             self.gdkGC.line_width = max(1, int(np.round(pixels)))
383: 
384: 
385: class FigureCanvasGDK (FigureCanvasBase):
386:     def __init__(self, figure):
387:         FigureCanvasBase.__init__(self, figure)
388:         if self.__class__ == matplotlib.backends.backend_gdk.FigureCanvasGDK:
389:             warn_deprecated('2.0', message="The GDK backend is "
390:                             "deprecated. It is untested, known to be "
391:                             "broken and will be removed in Matplotlib 2.2. "
392:                             "Use the Agg backend instead. "
393:                             "See Matplotlib usage FAQ for"
394:                             " more info on backends.",
395:                             alternative="Agg")
396:         self._renderer_init()
397: 
398:     def _renderer_init(self):
399:         self._renderer = RendererGDK (gtk.DrawingArea(), self.figure.dpi)
400: 
401:     def _render_figure(self, pixmap, width, height):
402:         self._renderer.set_pixmap (pixmap)
403:         self._renderer.set_width_height (width, height)
404:         self.figure.draw (self._renderer)
405: 
406:     filetypes = FigureCanvasBase.filetypes.copy()
407:     filetypes['jpg'] = 'JPEG'
408:     filetypes['jpeg'] = 'JPEG'
409: 
410:     def print_jpeg(self, filename, *args, **kwargs):
411:         return self._print_image(filename, 'jpeg')
412:     print_jpg = print_jpeg
413: 
414:     def print_png(self, filename, *args, **kwargs):
415:         return self._print_image(filename, 'png')
416: 
417:     def _print_image(self, filename, format, *args, **kwargs):
418:         width, height = self.get_width_height()
419:         pixmap = gtk.gdk.Pixmap (None, width, height, depth=24)
420:         self._render_figure(pixmap, width, height)
421: 
422:         # jpg colors don't match the display very well, png colors match
423:         # better
424:         pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, 0, 8,
425:                                 width, height)
426:         pixbuf.get_from_drawable(pixmap, pixmap.get_colormap(),
427:                                  0, 0, 0, 0, width, height)
428: 
429:         # set the default quality, if we are writing a JPEG.
430:         # http://www.pygtk.org/docs/pygtk/class-gdkpixbuf.html#method-gdkpixbuf--save
431:         options = {k: kwargs[k] for k in ['quality'] if k in kwargs}
432:         if format in ['jpg', 'jpeg']:
433:             options.setdefault('quality', rcParams['savefig.jpeg_quality'])
434:             options['quality'] = str(options['quality'])
435: 
436:         pixbuf.save(filename, format, options=options)
437: 
438: 
439: @_Backend.export
440: class _BackendGDK(_Backend):
441:     FigureCanvas = FigureCanvasGDK
442:     FigureManager = FigureManagerBase
443: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221562 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_221562) is not StypyTypeError):

    if (import_221562 != 'pyd_module'):
        __import__(import_221562)
        sys_modules_221563 = sys.modules[import_221562]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_221563.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_221562)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import math' statement (line 6)
import math

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import os' statement (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import warnings' statement (line 9)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import gobject' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'gobject')

if (type(import_221564) is not StypyTypeError):

    if (import_221564 != 'pyd_module'):
        __import__(import_221564)
        sys_modules_221565 = sys.modules[import_221564]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'gobject', sys_modules_221565.module_type_store, module_type_store)
    else:
        import gobject

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'gobject', gobject, module_type_store)

else:
    # Assigning a type to the variable 'gobject' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'gobject', import_221564)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import gtk' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221566 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'gtk')

if (type(import_221566) is not StypyTypeError):

    if (import_221566 != 'pyd_module'):
        __import__(import_221566)
        sys_modules_221567 = sys.modules[import_221566]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'gtk', sys_modules_221567.module_type_store, module_type_store)
    else:
        import gtk

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'gtk', gtk, module_type_store)

else:
    # Assigning a type to the variable 'gtk' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'gtk', import_221566)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Attribute to a Name (line 12):

# Assigning a Attribute to a Name (line 12):
# Getting the type of 'gtk' (line 12)
gtk_221568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'gtk')
# Obtaining the member 'gdk' of a type (line 12)
gdk_221569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 18), gtk_221568, 'gdk')
# Assigning a type to the variable 'gdk' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'gdk', gdk_221569)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import pango' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pango')

if (type(import_221570) is not StypyTypeError):

    if (import_221570 != 'pyd_module'):
        __import__(import_221570)
        sys_modules_221571 = sys.modules[import_221570]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pango', sys_modules_221571.module_type_store, module_type_store)
    else:
        import pango

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pango', pango, module_type_store)

else:
    # Assigning a type to the variable 'pango' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'pango', import_221570)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Tuple to a Name (line 14):

# Assigning a Tuple to a Name (line 14):

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_221572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
int_221573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 26), tuple_221572, int_221573)
# Adding element type (line 14)
int_221574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 26), tuple_221572, int_221574)
# Adding element type (line 14)
int_221575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 26), tuple_221572, int_221575)

# Assigning a type to the variable 'pygtk_version_required' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'pygtk_version_required', tuple_221572)


# Getting the type of 'gtk' (line 15)
gtk_221576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 3), 'gtk')
# Obtaining the member 'pygtk_version' of a type (line 15)
pygtk_version_221577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 3), gtk_221576, 'pygtk_version')
# Getting the type of 'pygtk_version_required' (line 15)
pygtk_version_required_221578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'pygtk_version_required')
# Applying the binary operator '<' (line 15)
result_lt_221579 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 3), '<', pygtk_version_221577, pygtk_version_required_221578)

# Testing the type of an if condition (line 15)
if_condition_221580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 0), result_lt_221579)
# Assigning a type to the variable 'if_condition_221580' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'if_condition_221580', if_condition_221580)
# SSA begins for if statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to ImportError(...): (line 16)
# Processing the call arguments (line 16)
unicode_221582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'unicode', u'PyGTK %d.%d.%d is installed\nPyGTK %d.%d.%d or later is required')
# Getting the type of 'gtk' (line 18)
gtk_221583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'gtk', False)
# Obtaining the member 'pygtk_version' of a type (line 18)
pygtk_version_221584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), gtk_221583, 'pygtk_version')
# Getting the type of 'pygtk_version_required' (line 18)
pygtk_version_required_221585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 45), 'pygtk_version_required', False)
# Applying the binary operator '+' (line 18)
result_add_221586 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 25), '+', pygtk_version_221584, pygtk_version_required_221585)

# Applying the binary operator '%' (line 16)
result_mod_221587 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 23), '%', unicode_221582, result_add_221586)

# Processing the call keyword arguments (line 16)
kwargs_221588 = {}
# Getting the type of 'ImportError' (line 16)
ImportError_221581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 16)
ImportError_call_result_221589 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), ImportError_221581, *[result_mod_221587], **kwargs_221588)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 16, 4), ImportError_call_result_221589, 'raise parameter', BaseException)
# SSA join for if statement (line 15)
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 19, 0), module_type_store, 'pygtk_version_required')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import numpy' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221590 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy')

if (type(import_221590) is not StypyTypeError):

    if (import_221590 != 'pyd_module'):
        __import__(import_221590)
        sys_modules_221591 = sys.modules[import_221590]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'np', sys_modules_221591.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', import_221590)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import matplotlib' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221592 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib')

if (type(import_221592) is not StypyTypeError):

    if (import_221592 != 'pyd_module'):
        __import__(import_221592)
        sys_modules_221593 = sys.modules[import_221592]
        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', sys_modules_221593.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', import_221592)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from matplotlib import rcParams' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221594 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib')

if (type(import_221594) is not StypyTypeError):

    if (import_221594 != 'pyd_module'):
        __import__(import_221594)
        sys_modules_221595 = sys.modules[import_221594]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', sys_modules_221595.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_221595, sys_modules_221595.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', import_221594)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221596 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib._pylab_helpers')

if (type(import_221596) is not StypyTypeError):

    if (import_221596 != 'pyd_module'):
        __import__(import_221596)
        sys_modules_221597 = sys.modules[import_221596]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib._pylab_helpers', sys_modules_221597.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_221597, sys_modules_221597.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib._pylab_helpers', import_221596)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221598 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.backend_bases')

if (type(import_221598) is not StypyTypeError):

    if (import_221598 != 'pyd_module'):
        __import__(import_221598)
        sys_modules_221599 = sys.modules[import_221598]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.backend_bases', sys_modules_221599.module_type_store, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'GraphicsContextBase', 'RendererBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_221599, sys_modules_221599.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'GraphicsContextBase', 'RendererBase'], [_Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.backend_bases', import_221598)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from matplotlib.cbook import warn_deprecated' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221600 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.cbook')

if (type(import_221600) is not StypyTypeError):

    if (import_221600 != 'pyd_module'):
        __import__(import_221600)
        sys_modules_221601 = sys.modules[import_221600]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.cbook', sys_modules_221601.module_type_store, module_type_store, ['warn_deprecated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_221601, sys_modules_221601.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import warn_deprecated

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.cbook', None, module_type_store, ['warn_deprecated'], [warn_deprecated])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.cbook', import_221600)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from matplotlib.figure import Figure' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221602 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.figure')

if (type(import_221602) is not StypyTypeError):

    if (import_221602 != 'pyd_module'):
        __import__(import_221602)
        sys_modules_221603 = sys.modules[import_221602]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.figure', sys_modules_221603.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_221603, sys_modules_221603.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.figure', import_221602)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from matplotlib.mathtext import MathTextParser' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221604 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.mathtext')

if (type(import_221604) is not StypyTypeError):

    if (import_221604 != 'pyd_module'):
        __import__(import_221604)
        sys_modules_221605 = sys.modules[import_221604]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.mathtext', sys_modules_221605.module_type_store, module_type_store, ['MathTextParser'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_221605, sys_modules_221605.module_type_store, module_type_store)
    else:
        from matplotlib.mathtext import MathTextParser

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.mathtext', None, module_type_store, ['MathTextParser'], [MathTextParser])

else:
    # Assigning a type to the variable 'matplotlib.mathtext' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.mathtext', import_221604)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from matplotlib.transforms import Affine2D' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221606 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.transforms')

if (type(import_221606) is not StypyTypeError):

    if (import_221606 != 'pyd_module'):
        __import__(import_221606)
        sys_modules_221607 = sys.modules[import_221606]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.transforms', sys_modules_221607.module_type_store, module_type_store, ['Affine2D'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_221607, sys_modules_221607.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Affine2D

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.transforms', None, module_type_store, ['Affine2D'], [Affine2D])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.transforms', import_221606)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from matplotlib.backends._backend_gdk import pixbuf_get_pixels_array' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_221608 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backends._backend_gdk')

if (type(import_221608) is not StypyTypeError):

    if (import_221608 != 'pyd_module'):
        __import__(import_221608)
        sys_modules_221609 = sys.modules[import_221608]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backends._backend_gdk', sys_modules_221609.module_type_store, module_type_store, ['pixbuf_get_pixels_array'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_221609, sys_modules_221609.module_type_store, module_type_store)
    else:
        from matplotlib.backends._backend_gdk import pixbuf_get_pixels_array

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backends._backend_gdk', None, module_type_store, ['pixbuf_get_pixels_array'], [pixbuf_get_pixels_array])

else:
    # Assigning a type to the variable 'matplotlib.backends._backend_gdk' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backends._backend_gdk', import_221608)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a BinOp to a Name (line 35):

# Assigning a BinOp to a Name (line 35):
unicode_221610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'unicode', u'%d.%d.%d')
# Getting the type of 'gtk' (line 35)
gtk_221611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'gtk')
# Obtaining the member 'pygtk_version' of a type (line 35)
pygtk_version_221612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 31), gtk_221611, 'pygtk_version')
# Applying the binary operator '%' (line 35)
result_mod_221613 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 18), '%', unicode_221610, pygtk_version_221612)

# Assigning a type to the variable 'backend_version' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'backend_version', result_mod_221613)

# Assigning a Call to a Name (line 38):

# Assigning a Call to a Name (line 38):

# Call to sorted(...): (line 38)
# Processing the call arguments (line 38)

# Obtaining an instance of the builtin type 'list' (line 38)
list_221615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
unicode_221616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'unicode', u'bmp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 22), list_221615, unicode_221616)
# Adding element type (line 38)
unicode_221617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'unicode', u'eps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 22), list_221615, unicode_221617)
# Adding element type (line 38)
unicode_221618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 37), 'unicode', u'jpg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 22), list_221615, unicode_221618)
# Adding element type (line 38)
unicode_221619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 44), 'unicode', u'png')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 22), list_221615, unicode_221619)
# Adding element type (line 38)
unicode_221620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 51), 'unicode', u'ps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 22), list_221615, unicode_221620)
# Adding element type (line 38)
unicode_221621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 57), 'unicode', u'svg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 22), list_221615, unicode_221621)

# Processing the call keyword arguments (line 38)
kwargs_221622 = {}
# Getting the type of 'sorted' (line 38)
sorted_221614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'sorted', False)
# Calling sorted(args, kwargs) (line 38)
sorted_call_result_221623 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), sorted_221614, *[list_221615], **kwargs_221622)

# Assigning a type to the variable 'IMAGE_FORMAT' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'IMAGE_FORMAT', sorted_call_result_221623)

# Assigning a Str to a Name (line 39):

# Assigning a Str to a Name (line 39):
unicode_221624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'unicode', u'png')
# Assigning a type to the variable 'IMAGE_FORMAT_DEFAULT' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'IMAGE_FORMAT_DEFAULT', unicode_221624)
# Declaration of the 'RendererGDK' class
# Getting the type of 'RendererBase' (line 42)
RendererBase_221625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'RendererBase')

class RendererGDK(RendererBase_221625, ):
    
    # Assigning a Dict to a Name (line 43):
    
    # Assigning a Dict to a Name (line 65):
    
    # Assigning a Dict to a Name (line 66):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.__init__', ['gtkDA', 'dpi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['gtkDA', 'dpi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 72):
        
        # Assigning a Name to a Attribute (line 72):
        # Getting the type of 'gtkDA' (line 72)
        gtkDA_221626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'gtkDA')
        # Getting the type of 'self' (line 72)
        self_221627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'gtkDA' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_221627, 'gtkDA', gtkDA_221626)
        
        # Assigning a Name to a Attribute (line 73):
        
        # Assigning a Name to a Attribute (line 73):
        # Getting the type of 'dpi' (line 73)
        dpi_221628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 21), 'dpi')
        # Getting the type of 'self' (line 73)
        self_221629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'dpi' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_221629, 'dpi', dpi_221628)
        
        # Assigning a Call to a Attribute (line 74):
        
        # Assigning a Call to a Attribute (line 74):
        
        # Call to get_colormap(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_221632 = {}
        # Getting the type of 'gtkDA' (line 74)
        gtkDA_221630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'gtkDA', False)
        # Obtaining the member 'get_colormap' of a type (line 74)
        get_colormap_221631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 21), gtkDA_221630, 'get_colormap')
        # Calling get_colormap(args, kwargs) (line 74)
        get_colormap_call_result_221633 = invoke(stypy.reporting.localization.Localization(__file__, 74, 21), get_colormap_221631, *[], **kwargs_221632)
        
        # Getting the type of 'self' (line 74)
        self_221634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member '_cmap' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_221634, '_cmap', get_colormap_call_result_221633)
        
        # Assigning a Call to a Attribute (line 75):
        
        # Assigning a Call to a Attribute (line 75):
        
        # Call to MathTextParser(...): (line 75)
        # Processing the call arguments (line 75)
        unicode_221636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 46), 'unicode', u'Agg')
        # Processing the call keyword arguments (line 75)
        kwargs_221637 = {}
        # Getting the type of 'MathTextParser' (line 75)
        MathTextParser_221635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'MathTextParser', False)
        # Calling MathTextParser(args, kwargs) (line 75)
        MathTextParser_call_result_221638 = invoke(stypy.reporting.localization.Localization(__file__, 75, 31), MathTextParser_221635, *[unicode_221636], **kwargs_221637)
        
        # Getting the type of 'self' (line 75)
        self_221639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'mathtext_parser' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_221639, 'mathtext_parser', MathTextParser_call_result_221638)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_pixmap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_pixmap'
        module_type_store = module_type_store.open_function_context('set_pixmap', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_function_name', 'RendererGDK.set_pixmap')
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_param_names_list', ['pixmap'])
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.set_pixmap.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.set_pixmap', ['pixmap'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_pixmap', localization, ['pixmap'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_pixmap(...)' code ##################

        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'pixmap' (line 78)
        pixmap_221640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'pixmap')
        # Getting the type of 'self' (line 78)
        self_221641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'gdkDrawable' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_221641, 'gdkDrawable', pixmap_221640)
        
        # ################# End of 'set_pixmap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_pixmap' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_221642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221642)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_pixmap'
        return stypy_return_type_221642


    @norecursion
    def set_width_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_width_height'
        module_type_store = module_type_store.open_function_context('set_width_height', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_function_name', 'RendererGDK.set_width_height')
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_param_names_list', ['width', 'height'])
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.set_width_height.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.set_width_height', ['width', 'height'], None, None, defaults, varargs, kwargs)

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

        unicode_221643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'unicode', u'w,h is the figure w,h not the pixmap w,h\n        ')
        
        # Assigning a Tuple to a Tuple (line 83):
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'width' (line 83)
        width_221644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'width')
        # Assigning a type to the variable 'tuple_assignment_221496' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_assignment_221496', width_221644)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'height' (line 83)
        height_221645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'height')
        # Assigning a type to the variable 'tuple_assignment_221497' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_assignment_221497', height_221645)
        
        # Assigning a Name to a Attribute (line 83):
        # Getting the type of 'tuple_assignment_221496' (line 83)
        tuple_assignment_221496_221646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_assignment_221496')
        # Getting the type of 'self' (line 83)
        self_221647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
        # Setting the type of the member 'width' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_221647, 'width', tuple_assignment_221496_221646)
        
        # Assigning a Name to a Attribute (line 83):
        # Getting the type of 'tuple_assignment_221497' (line 83)
        tuple_assignment_221497_221648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_assignment_221497')
        # Getting the type of 'self' (line 83)
        self_221649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'self')
        # Setting the type of the member 'height' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), self_221649, 'height', tuple_assignment_221497_221648)
        
        # ################# End of 'set_width_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_width_height' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_221650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221650)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_width_height'
        return stypy_return_type_221650


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 85)
        None_221651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 53), 'None')
        defaults = [None_221651]
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.draw_path.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.draw_path.__dict__.__setitem__('stypy_function_name', 'RendererGDK.draw_path')
        RendererGDK.draw_path.__dict__.__setitem__('stypy_param_names_list', ['gc', 'path', 'transform', 'rgbFace'])
        RendererGDK.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.draw_path', ['gc', 'path', 'transform', 'rgbFace'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Name (line 86):
        
        # Assigning a BinOp to a Name (line 86):
        # Getting the type of 'transform' (line 86)
        transform_221652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'transform')
        
        # Call to translate(...): (line 86)
        # Processing the call arguments (line 86)
        int_221662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 39), 'int')
        # Getting the type of 'self' (line 87)
        self_221663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'self', False)
        # Obtaining the member 'height' of a type (line 87)
        height_221664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 42), self_221663, 'height')
        # Processing the call keyword arguments (line 86)
        kwargs_221665 = {}
        
        # Call to scale(...): (line 86)
        # Processing the call arguments (line 86)
        float_221657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'float')
        float_221658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'float')
        # Processing the call keyword arguments (line 86)
        kwargs_221659 = {}
        
        # Call to Affine2D(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_221654 = {}
        # Getting the type of 'Affine2D' (line 86)
        Affine2D_221653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 86)
        Affine2D_call_result_221655 = invoke(stypy.reporting.localization.Localization(__file__, 86, 32), Affine2D_221653, *[], **kwargs_221654)
        
        # Obtaining the member 'scale' of a type (line 86)
        scale_221656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 32), Affine2D_call_result_221655, 'scale')
        # Calling scale(args, kwargs) (line 86)
        scale_call_result_221660 = invoke(stypy.reporting.localization.Localization(__file__, 86, 32), scale_221656, *[float_221657, float_221658], **kwargs_221659)
        
        # Obtaining the member 'translate' of a type (line 86)
        translate_221661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 32), scale_call_result_221660, 'translate')
        # Calling translate(args, kwargs) (line 86)
        translate_call_result_221666 = invoke(stypy.reporting.localization.Localization(__file__, 86, 32), translate_221661, *[int_221662, height_221664], **kwargs_221665)
        
        # Applying the binary operator '+' (line 86)
        result_add_221667 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 20), '+', transform_221652, translate_call_result_221666)
        
        # Assigning a type to the variable 'transform' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'transform', result_add_221667)
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to to_polygons(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'transform' (line 88)
        transform_221670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'transform', False)
        # Getting the type of 'self' (line 88)
        self_221671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 47), 'self', False)
        # Obtaining the member 'width' of a type (line 88)
        width_221672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 47), self_221671, 'width')
        # Getting the type of 'self' (line 88)
        self_221673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'self', False)
        # Obtaining the member 'height' of a type (line 88)
        height_221674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 59), self_221673, 'height')
        # Processing the call keyword arguments (line 88)
        kwargs_221675 = {}
        # Getting the type of 'path' (line 88)
        path_221668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'path', False)
        # Obtaining the member 'to_polygons' of a type (line 88)
        to_polygons_221669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 19), path_221668, 'to_polygons')
        # Calling to_polygons(args, kwargs) (line 88)
        to_polygons_call_result_221676 = invoke(stypy.reporting.localization.Localization(__file__, 88, 19), to_polygons_221669, *[transform_221670, width_221672, height_221674], **kwargs_221675)
        
        # Assigning a type to the variable 'polygons' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'polygons', to_polygons_call_result_221676)
        
        # Getting the type of 'polygons' (line 89)
        polygons_221677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'polygons')
        # Testing the type of a for loop iterable (line 89)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 8), polygons_221677)
        # Getting the type of the for loop variable (line 89)
        for_loop_var_221678 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 8), polygons_221677)
        # Assigning a type to the variable 'polygon' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'polygon', for_loop_var_221678)
        # SSA begins for a for statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Name (line 92):
        
        # Assigning a ListComp to a Name (line 92):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'polygon' (line 92)
        polygon_221696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 72), 'polygon')
        comprehension_221697 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 23), polygon_221696)
        # Assigning a type to the variable 'x' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 23), comprehension_221697))
        # Assigning a type to the variable 'y' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 23), comprehension_221697))
        
        # Obtaining an instance of the builtin type 'tuple' (line 92)
        tuple_221679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 92)
        # Adding element type (line 92)
        
        # Call to int(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to round(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'x' (line 92)
        x_221683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'x', False)
        # Processing the call keyword arguments (line 92)
        kwargs_221684 = {}
        # Getting the type of 'np' (line 92)
        np_221681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'np', False)
        # Obtaining the member 'round' of a type (line 92)
        round_221682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), np_221681, 'round')
        # Calling round(args, kwargs) (line 92)
        round_call_result_221685 = invoke(stypy.reporting.localization.Localization(__file__, 92, 28), round_221682, *[x_221683], **kwargs_221684)
        
        # Processing the call keyword arguments (line 92)
        kwargs_221686 = {}
        # Getting the type of 'int' (line 92)
        int_221680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'int', False)
        # Calling int(args, kwargs) (line 92)
        int_call_result_221687 = invoke(stypy.reporting.localization.Localization(__file__, 92, 24), int_221680, *[round_call_result_221685], **kwargs_221686)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 24), tuple_221679, int_call_result_221687)
        # Adding element type (line 92)
        
        # Call to int(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to round(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'y' (line 92)
        y_221691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 55), 'y', False)
        # Processing the call keyword arguments (line 92)
        kwargs_221692 = {}
        # Getting the type of 'np' (line 92)
        np_221689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'np', False)
        # Obtaining the member 'round' of a type (line 92)
        round_221690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 46), np_221689, 'round')
        # Calling round(args, kwargs) (line 92)
        round_call_result_221693 = invoke(stypy.reporting.localization.Localization(__file__, 92, 46), round_221690, *[y_221691], **kwargs_221692)
        
        # Processing the call keyword arguments (line 92)
        kwargs_221694 = {}
        # Getting the type of 'int' (line 92)
        int_221688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'int', False)
        # Calling int(args, kwargs) (line 92)
        int_call_result_221695 = invoke(stypy.reporting.localization.Localization(__file__, 92, 42), int_221688, *[round_call_result_221693], **kwargs_221694)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 24), tuple_221679, int_call_result_221695)
        
        list_221698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 23), list_221698, tuple_221679)
        # Assigning a type to the variable 'polygon' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'polygon', list_221698)
        
        # Type idiom detected: calculating its left and rigth part (line 93)
        # Getting the type of 'rgbFace' (line 93)
        rgbFace_221699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'rgbFace')
        # Getting the type of 'None' (line 93)
        None_221700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'None')
        
        (may_be_221701, more_types_in_union_221702) = may_not_be_none(rgbFace_221699, None_221700)

        if may_be_221701:

            if more_types_in_union_221702:
                # Runtime conditional SSA (line 93)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 94):
            
            # Assigning a Attribute to a Name (line 94):
            # Getting the type of 'gc' (line 94)
            gc_221703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'gc')
            # Obtaining the member 'gdkGC' of a type (line 94)
            gdkGC_221704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 28), gc_221703, 'gdkGC')
            # Obtaining the member 'foreground' of a type (line 94)
            foreground_221705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 28), gdkGC_221704, 'foreground')
            # Assigning a type to the variable 'saveColor' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'saveColor', foreground_221705)
            
            # Assigning a Call to a Attribute (line 95):
            
            # Assigning a Call to a Attribute (line 95):
            
            # Call to rgb_to_gdk_color(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'rgbFace' (line 95)
            rgbFace_221708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 58), 'rgbFace', False)
            # Processing the call keyword arguments (line 95)
            kwargs_221709 = {}
            # Getting the type of 'gc' (line 95)
            gc_221706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'gc', False)
            # Obtaining the member 'rgb_to_gdk_color' of a type (line 95)
            rgb_to_gdk_color_221707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 38), gc_221706, 'rgb_to_gdk_color')
            # Calling rgb_to_gdk_color(args, kwargs) (line 95)
            rgb_to_gdk_color_call_result_221710 = invoke(stypy.reporting.localization.Localization(__file__, 95, 38), rgb_to_gdk_color_221707, *[rgbFace_221708], **kwargs_221709)
            
            # Getting the type of 'gc' (line 95)
            gc_221711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'gc')
            # Obtaining the member 'gdkGC' of a type (line 95)
            gdkGC_221712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), gc_221711, 'gdkGC')
            # Setting the type of the member 'foreground' of a type (line 95)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), gdkGC_221712, 'foreground', rgb_to_gdk_color_call_result_221710)
            
            # Call to draw_polygon(...): (line 96)
            # Processing the call arguments (line 96)
            # Getting the type of 'gc' (line 96)
            gc_221716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 46), 'gc', False)
            # Obtaining the member 'gdkGC' of a type (line 96)
            gdkGC_221717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 46), gc_221716, 'gdkGC')
            # Getting the type of 'True' (line 96)
            True_221718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 56), 'True', False)
            # Getting the type of 'polygon' (line 96)
            polygon_221719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 62), 'polygon', False)
            # Processing the call keyword arguments (line 96)
            kwargs_221720 = {}
            # Getting the type of 'self' (line 96)
            self_221713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'self', False)
            # Obtaining the member 'gdkDrawable' of a type (line 96)
            gdkDrawable_221714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), self_221713, 'gdkDrawable')
            # Obtaining the member 'draw_polygon' of a type (line 96)
            draw_polygon_221715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), gdkDrawable_221714, 'draw_polygon')
            # Calling draw_polygon(args, kwargs) (line 96)
            draw_polygon_call_result_221721 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), draw_polygon_221715, *[gdkGC_221717, True_221718, polygon_221719], **kwargs_221720)
            
            
            # Assigning a Name to a Attribute (line 97):
            
            # Assigning a Name to a Attribute (line 97):
            # Getting the type of 'saveColor' (line 97)
            saveColor_221722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'saveColor')
            # Getting the type of 'gc' (line 97)
            gc_221723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'gc')
            # Obtaining the member 'gdkGC' of a type (line 97)
            gdkGC_221724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 16), gc_221723, 'gdkGC')
            # Setting the type of the member 'foreground' of a type (line 97)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 16), gdkGC_221724, 'foreground', saveColor_221722)

            if more_types_in_union_221702:
                # SSA join for if statement (line 93)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'gc' (line 98)
        gc_221725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'gc')
        # Obtaining the member 'gdkGC' of a type (line 98)
        gdkGC_221726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), gc_221725, 'gdkGC')
        # Obtaining the member 'line_width' of a type (line 98)
        line_width_221727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), gdkGC_221726, 'line_width')
        int_221728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 37), 'int')
        # Applying the binary operator '>' (line 98)
        result_gt_221729 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 15), '>', line_width_221727, int_221728)
        
        # Testing the type of an if condition (line 98)
        if_condition_221730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 12), result_gt_221729)
        # Assigning a type to the variable 'if_condition_221730' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'if_condition_221730', if_condition_221730)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to draw_lines(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'gc' (line 99)
        gc_221734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 44), 'gc', False)
        # Obtaining the member 'gdkGC' of a type (line 99)
        gdkGC_221735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 44), gc_221734, 'gdkGC')
        # Getting the type of 'polygon' (line 99)
        polygon_221736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 54), 'polygon', False)
        # Processing the call keyword arguments (line 99)
        kwargs_221737 = {}
        # Getting the type of 'self' (line 99)
        self_221731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'self', False)
        # Obtaining the member 'gdkDrawable' of a type (line 99)
        gdkDrawable_221732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), self_221731, 'gdkDrawable')
        # Obtaining the member 'draw_lines' of a type (line 99)
        draw_lines_221733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), gdkDrawable_221732, 'draw_lines')
        # Calling draw_lines(args, kwargs) (line 99)
        draw_lines_call_result_221738 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), draw_lines_221733, *[gdkGC_221735, polygon_221736], **kwargs_221737)
        
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_221739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221739)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_221739


    @norecursion
    def draw_image(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_image'
        module_type_store = module_type_store.open_function_context('draw_image', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.draw_image.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.draw_image.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.draw_image.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.draw_image.__dict__.__setitem__('stypy_function_name', 'RendererGDK.draw_image')
        RendererGDK.draw_image.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 'im'])
        RendererGDK.draw_image.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.draw_image.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.draw_image.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.draw_image.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.draw_image.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.draw_image.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.draw_image', ['gc', 'x', 'y', 'im'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to get_clip_rectangle(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_221742 = {}
        # Getting the type of 'gc' (line 102)
        gc_221740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'gc', False)
        # Obtaining the member 'get_clip_rectangle' of a type (line 102)
        get_clip_rectangle_221741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), gc_221740, 'get_clip_rectangle')
        # Calling get_clip_rectangle(args, kwargs) (line 102)
        get_clip_rectangle_call_result_221743 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), get_clip_rectangle_221741, *[], **kwargs_221742)
        
        # Assigning a type to the variable 'bbox' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'bbox', get_clip_rectangle_call_result_221743)
        
        
        # Getting the type of 'bbox' (line 104)
        bbox_221744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'bbox')
        # Getting the type of 'None' (line 104)
        None_221745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'None')
        # Applying the binary operator '!=' (line 104)
        result_ne_221746 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), '!=', bbox_221744, None_221745)
        
        # Testing the type of an if condition (line 104)
        if_condition_221747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), result_ne_221746)
        # Assigning a type to the variable 'if_condition_221747' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_221747', if_condition_221747)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 105):
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_221748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'int')
        # Getting the type of 'bbox' (line 105)
        bbox_221749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'bbox')
        # Obtaining the member 'bounds' of a type (line 105)
        bounds_221750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 22), bbox_221749, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___221751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), bounds_221750, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_221752 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), getitem___221751, int_221748)
        
        # Assigning a type to the variable 'tuple_var_assignment_221498' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple_var_assignment_221498', subscript_call_result_221752)
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_221753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'int')
        # Getting the type of 'bbox' (line 105)
        bbox_221754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'bbox')
        # Obtaining the member 'bounds' of a type (line 105)
        bounds_221755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 22), bbox_221754, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___221756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), bounds_221755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_221757 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), getitem___221756, int_221753)
        
        # Assigning a type to the variable 'tuple_var_assignment_221499' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple_var_assignment_221499', subscript_call_result_221757)
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_221758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'int')
        # Getting the type of 'bbox' (line 105)
        bbox_221759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'bbox')
        # Obtaining the member 'bounds' of a type (line 105)
        bounds_221760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 22), bbox_221759, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___221761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), bounds_221760, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_221762 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), getitem___221761, int_221758)
        
        # Assigning a type to the variable 'tuple_var_assignment_221500' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple_var_assignment_221500', subscript_call_result_221762)
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_221763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'int')
        # Getting the type of 'bbox' (line 105)
        bbox_221764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'bbox')
        # Obtaining the member 'bounds' of a type (line 105)
        bounds_221765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 22), bbox_221764, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___221766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), bounds_221765, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_221767 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), getitem___221766, int_221763)
        
        # Assigning a type to the variable 'tuple_var_assignment_221501' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple_var_assignment_221501', subscript_call_result_221767)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_221498' (line 105)
        tuple_var_assignment_221498_221768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple_var_assignment_221498')
        # Assigning a type to the variable 'l' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'l', tuple_var_assignment_221498_221768)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_221499' (line 105)
        tuple_var_assignment_221499_221769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple_var_assignment_221499')
        # Assigning a type to the variable 'b' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 14), 'b', tuple_var_assignment_221499_221769)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_221500' (line 105)
        tuple_var_assignment_221500_221770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple_var_assignment_221500')
        # Assigning a type to the variable 'w' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'w', tuple_var_assignment_221500_221770)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_221501' (line 105)
        tuple_var_assignment_221501_221771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple_var_assignment_221501')
        # Assigning a type to the variable 'h' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'h', tuple_var_assignment_221501_221771)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 110):
        
        # Assigning a Subscript to a Name (line 110):
        
        # Obtaining the type of the subscript
        int_221772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 8), 'int')
        
        # Obtaining the type of the subscript
        int_221773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
        slice_221774 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 110, 21), None, int_221773, None)
        # Getting the type of 'im' (line 110)
        im_221775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'im')
        # Obtaining the member 'shape' of a type (line 110)
        shape_221776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 21), im_221775, 'shape')
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___221777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 21), shape_221776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_221778 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), getitem___221777, slice_221774)
        
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___221779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), subscript_call_result_221778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_221780 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), getitem___221779, int_221772)
        
        # Assigning a type to the variable 'tuple_var_assignment_221502' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'tuple_var_assignment_221502', subscript_call_result_221780)
        
        # Assigning a Subscript to a Name (line 110):
        
        # Obtaining the type of the subscript
        int_221781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 8), 'int')
        
        # Obtaining the type of the subscript
        int_221782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
        slice_221783 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 110, 21), None, int_221782, None)
        # Getting the type of 'im' (line 110)
        im_221784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'im')
        # Obtaining the member 'shape' of a type (line 110)
        shape_221785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 21), im_221784, 'shape')
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___221786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 21), shape_221785, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_221787 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), getitem___221786, slice_221783)
        
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___221788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), subscript_call_result_221787, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_221789 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), getitem___221788, int_221781)
        
        # Assigning a type to the variable 'tuple_var_assignment_221503' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'tuple_var_assignment_221503', subscript_call_result_221789)
        
        # Assigning a Name to a Name (line 110):
        # Getting the type of 'tuple_var_assignment_221502' (line 110)
        tuple_var_assignment_221502_221790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'tuple_var_assignment_221502')
        # Assigning a type to the variable 'rows' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'rows', tuple_var_assignment_221502_221790)
        
        # Assigning a Name to a Name (line 110):
        # Getting the type of 'tuple_var_assignment_221503' (line 110)
        tuple_var_assignment_221503_221791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'tuple_var_assignment_221503')
        # Assigning a type to the variable 'cols' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'cols', tuple_var_assignment_221503_221791)
        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to Pixbuf(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'gtk' (line 112)
        gtk_221795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 112)
        gdk_221796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 32), gtk_221795, 'gdk')
        # Obtaining the member 'COLORSPACE_RGB' of a type (line 112)
        COLORSPACE_RGB_221797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 32), gdk_221796, 'COLORSPACE_RGB')
        # Processing the call keyword arguments (line 112)
        # Getting the type of 'True' (line 113)
        True_221798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 42), 'True', False)
        keyword_221799 = True_221798
        int_221800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 64), 'int')
        keyword_221801 = int_221800
        # Getting the type of 'cols' (line 114)
        cols_221802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'cols', False)
        keyword_221803 = cols_221802
        # Getting the type of 'rows' (line 114)
        rows_221804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 51), 'rows', False)
        keyword_221805 = rows_221804
        kwargs_221806 = {'has_alpha': keyword_221799, 'width': keyword_221803, 'bits_per_sample': keyword_221801, 'height': keyword_221805}
        # Getting the type of 'gtk' (line 112)
        gtk_221792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 112)
        gdk_221793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 17), gtk_221792, 'gdk')
        # Obtaining the member 'Pixbuf' of a type (line 112)
        Pixbuf_221794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 17), gdk_221793, 'Pixbuf')
        # Calling Pixbuf(args, kwargs) (line 112)
        Pixbuf_call_result_221807 = invoke(stypy.reporting.localization.Localization(__file__, 112, 17), Pixbuf_221794, *[COLORSPACE_RGB_221797], **kwargs_221806)
        
        # Assigning a type to the variable 'pixbuf' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'pixbuf', Pixbuf_call_result_221807)
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to pixbuf_get_pixels_array(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'pixbuf' (line 116)
        pixbuf_221809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 40), 'pixbuf', False)
        # Processing the call keyword arguments (line 116)
        kwargs_221810 = {}
        # Getting the type of 'pixbuf_get_pixels_array' (line 116)
        pixbuf_get_pixels_array_221808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'pixbuf_get_pixels_array', False)
        # Calling pixbuf_get_pixels_array(args, kwargs) (line 116)
        pixbuf_get_pixels_array_call_result_221811 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), pixbuf_get_pixels_array_221808, *[pixbuf_221809], **kwargs_221810)
        
        # Assigning a type to the variable 'array' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'array', pixbuf_get_pixels_array_call_result_221811)
        
        # Assigning a Subscript to a Subscript (line 117):
        
        # Assigning a Subscript to a Subscript (line 117):
        
        # Obtaining the type of the subscript
        int_221812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 30), 'int')
        slice_221813 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 25), None, None, int_221812)
        # Getting the type of 'im' (line 117)
        im_221814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'im')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___221815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), im_221814, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_221816 = invoke(stypy.reporting.localization.Localization(__file__, 117, 25), getitem___221815, slice_221813)
        
        # Getting the type of 'array' (line 117)
        array_221817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'array')
        slice_221818 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 8), None, None, None)
        slice_221819 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 8), None, None, None)
        slice_221820 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 8), None, None, None)
        # Storing an element on a container (line 117)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 8), array_221817, ((slice_221818, slice_221819, slice_221820), subscript_call_result_221816))
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to new_gc(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_221823 = {}
        # Getting the type of 'self' (line 119)
        self_221821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 13), 'self', False)
        # Obtaining the member 'new_gc' of a type (line 119)
        new_gc_221822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 13), self_221821, 'new_gc')
        # Calling new_gc(args, kwargs) (line 119)
        new_gc_call_result_221824 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), new_gc_221822, *[], **kwargs_221823)
        
        # Assigning a type to the variable 'gc' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'gc', new_gc_call_result_221824)
        
        # Assigning a BinOp to a Name (line 122):
        
        # Assigning a BinOp to a Name (line 122):
        # Getting the type of 'self' (line 122)
        self_221825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'self')
        # Obtaining the member 'height' of a type (line 122)
        height_221826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), self_221825, 'height')
        # Getting the type of 'y' (line 122)
        y_221827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'y')
        # Applying the binary operator '-' (line 122)
        result_sub_221828 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 12), '-', height_221826, y_221827)
        
        # Getting the type of 'rows' (line 122)
        rows_221829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 26), 'rows')
        # Applying the binary operator '-' (line 122)
        result_sub_221830 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 25), '-', result_sub_221828, rows_221829)
        
        # Assigning a type to the variable 'y' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'y', result_sub_221830)
        
        
        # SSA begins for try-except statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to draw_pixbuf(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'gc' (line 126)
        gc_221834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 42), 'gc', False)
        # Obtaining the member 'gdkGC' of a type (line 126)
        gdkGC_221835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 42), gc_221834, 'gdkGC')
        # Getting the type of 'pixbuf' (line 126)
        pixbuf_221836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 52), 'pixbuf', False)
        int_221837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 60), 'int')
        int_221838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 63), 'int')
        
        # Call to int(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'x' (line 127)
        x_221840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'x', False)
        # Processing the call keyword arguments (line 127)
        kwargs_221841 = {}
        # Getting the type of 'int' (line 127)
        int_221839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'int', False)
        # Calling int(args, kwargs) (line 127)
        int_call_result_221842 = invoke(stypy.reporting.localization.Localization(__file__, 127, 42), int_221839, *[x_221840], **kwargs_221841)
        
        
        # Call to int(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'y' (line 127)
        y_221844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 54), 'y', False)
        # Processing the call keyword arguments (line 127)
        kwargs_221845 = {}
        # Getting the type of 'int' (line 127)
        int_221843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 50), 'int', False)
        # Calling int(args, kwargs) (line 127)
        int_call_result_221846 = invoke(stypy.reporting.localization.Localization(__file__, 127, 50), int_221843, *[y_221844], **kwargs_221845)
        
        # Getting the type of 'cols' (line 127)
        cols_221847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 58), 'cols', False)
        # Getting the type of 'rows' (line 127)
        rows_221848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 64), 'rows', False)
        # Getting the type of 'gdk' (line 128)
        gdk_221849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'gdk', False)
        # Obtaining the member 'RGB_DITHER_NONE' of a type (line 128)
        RGB_DITHER_NONE_221850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 42), gdk_221849, 'RGB_DITHER_NONE')
        int_221851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 63), 'int')
        int_221852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 66), 'int')
        # Processing the call keyword arguments (line 126)
        kwargs_221853 = {}
        # Getting the type of 'self' (line 126)
        self_221831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'self', False)
        # Obtaining the member 'gdkDrawable' of a type (line 126)
        gdkDrawable_221832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), self_221831, 'gdkDrawable')
        # Obtaining the member 'draw_pixbuf' of a type (line 126)
        draw_pixbuf_221833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), gdkDrawable_221832, 'draw_pixbuf')
        # Calling draw_pixbuf(args, kwargs) (line 126)
        draw_pixbuf_call_result_221854 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), draw_pixbuf_221833, *[gdkGC_221835, pixbuf_221836, int_221837, int_221838, int_call_result_221842, int_call_result_221846, cols_221847, rows_221848, RGB_DITHER_NONE_221850, int_221851, int_221852], **kwargs_221853)
        
        # SSA branch for the except part of a try statement (line 124)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 124)
        module_type_store.open_ssa_branch('except')
        
        # Call to render_to_drawable(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'self' (line 131)
        self_221857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 38), 'self', False)
        # Obtaining the member 'gdkDrawable' of a type (line 131)
        gdkDrawable_221858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 38), self_221857, 'gdkDrawable')
        # Getting the type of 'gc' (line 131)
        gc_221859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 56), 'gc', False)
        # Obtaining the member 'gdkGC' of a type (line 131)
        gdkGC_221860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 56), gc_221859, 'gdkGC')
        int_221861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 66), 'int')
        int_221862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 69), 'int')
        
        # Call to int(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'x' (line 132)
        x_221864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 38), 'x', False)
        # Processing the call keyword arguments (line 132)
        kwargs_221865 = {}
        # Getting the type of 'int' (line 132)
        int_221863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'int', False)
        # Calling int(args, kwargs) (line 132)
        int_call_result_221866 = invoke(stypy.reporting.localization.Localization(__file__, 132, 34), int_221863, *[x_221864], **kwargs_221865)
        
        
        # Call to int(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'y' (line 132)
        y_221868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 46), 'y', False)
        # Processing the call keyword arguments (line 132)
        kwargs_221869 = {}
        # Getting the type of 'int' (line 132)
        int_221867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'int', False)
        # Calling int(args, kwargs) (line 132)
        int_call_result_221870 = invoke(stypy.reporting.localization.Localization(__file__, 132, 42), int_221867, *[y_221868], **kwargs_221869)
        
        # Getting the type of 'cols' (line 132)
        cols_221871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 50), 'cols', False)
        # Getting the type of 'rows' (line 132)
        rows_221872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 56), 'rows', False)
        # Getting the type of 'gdk' (line 133)
        gdk_221873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 34), 'gdk', False)
        # Obtaining the member 'RGB_DITHER_NONE' of a type (line 133)
        RGB_DITHER_NONE_221874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 34), gdk_221873, 'RGB_DITHER_NONE')
        int_221875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 55), 'int')
        int_221876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 58), 'int')
        # Processing the call keyword arguments (line 131)
        kwargs_221877 = {}
        # Getting the type of 'pixbuf' (line 131)
        pixbuf_221855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'pixbuf', False)
        # Obtaining the member 'render_to_drawable' of a type (line 131)
        render_to_drawable_221856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), pixbuf_221855, 'render_to_drawable')
        # Calling render_to_drawable(args, kwargs) (line 131)
        render_to_drawable_call_result_221878 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), render_to_drawable_221856, *[gdkDrawable_221858, gdkGC_221860, int_221861, int_221862, int_call_result_221866, int_call_result_221870, cols_221871, rows_221872, RGB_DITHER_NONE_221874, int_221875, int_221876], **kwargs_221877)
        
        # SSA join for try-except statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_image(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_image' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_221879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221879)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_image'
        return stypy_return_type_221879


    @norecursion
    def draw_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 135)
        False_221880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 57), 'False')
        # Getting the type of 'None' (line 135)
        None_221881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 70), 'None')
        defaults = [False_221880, None_221881]
        # Create a new context for function 'draw_text'
        module_type_store = module_type_store.open_function_context('draw_text', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.draw_text.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.draw_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.draw_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.draw_text.__dict__.__setitem__('stypy_function_name', 'RendererGDK.draw_text')
        RendererGDK.draw_text.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'])
        RendererGDK.draw_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.draw_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.draw_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.draw_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.draw_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.draw_text.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.draw_text', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Tuple to a Tuple (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to int(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'x' (line 136)
        x_221883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'x', False)
        # Processing the call keyword arguments (line 136)
        kwargs_221884 = {}
        # Getting the type of 'int' (line 136)
        int_221882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'int', False)
        # Calling int(args, kwargs) (line 136)
        int_call_result_221885 = invoke(stypy.reporting.localization.Localization(__file__, 136, 15), int_221882, *[x_221883], **kwargs_221884)
        
        # Assigning a type to the variable 'tuple_assignment_221504' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_assignment_221504', int_call_result_221885)
        
        # Assigning a Call to a Name (line 136):
        
        # Call to int(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'y' (line 136)
        y_221887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'y', False)
        # Processing the call keyword arguments (line 136)
        kwargs_221888 = {}
        # Getting the type of 'int' (line 136)
        int_221886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'int', False)
        # Calling int(args, kwargs) (line 136)
        int_call_result_221889 = invoke(stypy.reporting.localization.Localization(__file__, 136, 23), int_221886, *[y_221887], **kwargs_221888)
        
        # Assigning a type to the variable 'tuple_assignment_221505' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_assignment_221505', int_call_result_221889)
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'tuple_assignment_221504' (line 136)
        tuple_assignment_221504_221890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_assignment_221504')
        # Assigning a type to the variable 'x' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'x', tuple_assignment_221504_221890)
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'tuple_assignment_221505' (line 136)
        tuple_assignment_221505_221891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_assignment_221505')
        # Assigning a type to the variable 'y' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'y', tuple_assignment_221505_221891)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 138)
        x_221892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'x')
        int_221893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 15), 'int')
        # Applying the binary operator '<' (line 138)
        result_lt_221894 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), '<', x_221892, int_221893)
        
        
        # Getting the type of 'y' (line 138)
        y_221895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'y')
        int_221896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'int')
        # Applying the binary operator '<' (line 138)
        result_lt_221897 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 20), '<', y_221895, int_221896)
        
        # Applying the binary operator 'or' (line 138)
        result_or_keyword_221898 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), 'or', result_lt_221894, result_lt_221897)
        
        # Testing the type of an if condition (line 138)
        if_condition_221899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 8), result_or_keyword_221898)
        # Assigning a type to the variable 'if_condition_221899' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'if_condition_221899', if_condition_221899)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'angle' (line 141)
        angle_221900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'angle')
        
        # Obtaining an instance of the builtin type 'tuple' (line 141)
        tuple_221901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 141)
        # Adding element type (line 141)
        int_221902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 25), tuple_221901, int_221902)
        # Adding element type (line 141)
        int_221903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 25), tuple_221901, int_221903)
        
        # Applying the binary operator 'notin' (line 141)
        result_contains_221904 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), 'notin', angle_221900, tuple_221901)
        
        # Testing the type of an if condition (line 141)
        if_condition_221905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_contains_221904)
        # Assigning a type to the variable 'if_condition_221905' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_221905', if_condition_221905)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 142)
        # Processing the call arguments (line 142)
        unicode_221908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 26), 'unicode', u'backend_gdk: unable to draw text at angles ')
        unicode_221909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 26), 'unicode', u'other than 0 or 90')
        # Applying the binary operator '+' (line 142)
        result_add_221910 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 26), '+', unicode_221908, unicode_221909)
        
        # Processing the call keyword arguments (line 142)
        kwargs_221911 = {}
        # Getting the type of 'warnings' (line 142)
        warnings_221906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 142)
        warn_221907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), warnings_221906, 'warn')
        # Calling warn(args, kwargs) (line 142)
        warn_call_result_221912 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), warn_221907, *[result_add_221910], **kwargs_221911)
        
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'ismath' (line 144)
        ismath_221913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 13), 'ismath')
        # Testing the type of an if condition (line 144)
        if_condition_221914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 13), ismath_221913)
        # Assigning a type to the variable 'if_condition_221914' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 13), 'if_condition_221914', if_condition_221914)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _draw_mathtext(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'gc' (line 145)
        gc_221917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'gc', False)
        # Getting the type of 'x' (line 145)
        x_221918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'x', False)
        # Getting the type of 'y' (line 145)
        y_221919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'y', False)
        # Getting the type of 's' (line 145)
        s_221920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 42), 's', False)
        # Getting the type of 'prop' (line 145)
        prop_221921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 45), 'prop', False)
        # Getting the type of 'angle' (line 145)
        angle_221922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 51), 'angle', False)
        # Processing the call keyword arguments (line 145)
        kwargs_221923 = {}
        # Getting the type of 'self' (line 145)
        self_221915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'self', False)
        # Obtaining the member '_draw_mathtext' of a type (line 145)
        _draw_mathtext_221916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), self_221915, '_draw_mathtext')
        # Calling _draw_mathtext(args, kwargs) (line 145)
        _draw_mathtext_call_result_221924 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), _draw_mathtext_221916, *[gc_221917, x_221918, y_221919, s_221920, prop_221921, angle_221922], **kwargs_221923)
        
        # SSA branch for the else part of an if statement (line 144)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'angle' (line 147)
        angle_221925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'angle')
        int_221926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 20), 'int')
        # Applying the binary operator '==' (line 147)
        result_eq_221927 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 13), '==', angle_221925, int_221926)
        
        # Testing the type of an if condition (line 147)
        if_condition_221928 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 13), result_eq_221927)
        # Assigning a type to the variable 'if_condition_221928' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'if_condition_221928', if_condition_221928)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _draw_rotated_text(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'gc' (line 148)
        gc_221931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 36), 'gc', False)
        # Getting the type of 'x' (line 148)
        x_221932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'x', False)
        # Getting the type of 'y' (line 148)
        y_221933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 43), 'y', False)
        # Getting the type of 's' (line 148)
        s_221934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 46), 's', False)
        # Getting the type of 'prop' (line 148)
        prop_221935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 49), 'prop', False)
        # Getting the type of 'angle' (line 148)
        angle_221936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 55), 'angle', False)
        # Processing the call keyword arguments (line 148)
        kwargs_221937 = {}
        # Getting the type of 'self' (line 148)
        self_221929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'self', False)
        # Obtaining the member '_draw_rotated_text' of a type (line 148)
        _draw_rotated_text_221930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), self_221929, '_draw_rotated_text')
        # Calling _draw_rotated_text(args, kwargs) (line 148)
        _draw_rotated_text_call_result_221938 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), _draw_rotated_text_221930, *[gc_221931, x_221932, y_221933, s_221934, prop_221935, angle_221936], **kwargs_221937)
        
        # SSA branch for the else part of an if statement (line 147)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 151):
        
        # Assigning a Call to a Name:
        
        # Call to _get_pango_layout(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 's' (line 151)
        s_221941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 66), 's', False)
        # Getting the type of 'prop' (line 151)
        prop_221942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 69), 'prop', False)
        # Processing the call keyword arguments (line 151)
        kwargs_221943 = {}
        # Getting the type of 'self' (line 151)
        self_221939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 43), 'self', False)
        # Obtaining the member '_get_pango_layout' of a type (line 151)
        _get_pango_layout_221940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 43), self_221939, '_get_pango_layout')
        # Calling _get_pango_layout(args, kwargs) (line 151)
        _get_pango_layout_call_result_221944 = invoke(stypy.reporting.localization.Localization(__file__, 151, 43), _get_pango_layout_221940, *[s_221941, prop_221942], **kwargs_221943)
        
        # Assigning a type to the variable 'call_assignment_221506' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221506', _get_pango_layout_call_result_221944)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
        # Processing the call keyword arguments
        kwargs_221948 = {}
        # Getting the type of 'call_assignment_221506' (line 151)
        call_assignment_221506_221945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221506', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___221946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), call_assignment_221506_221945, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221949 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221946, *[int_221947], **kwargs_221948)
        
        # Assigning a type to the variable 'call_assignment_221507' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221507', getitem___call_result_221949)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'call_assignment_221507' (line 151)
        call_assignment_221507_221950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221507')
        # Assigning a type to the variable 'layout' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'layout', call_assignment_221507_221950)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
        # Processing the call keyword arguments
        kwargs_221954 = {}
        # Getting the type of 'call_assignment_221506' (line 151)
        call_assignment_221506_221951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221506', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___221952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), call_assignment_221506_221951, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221955 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221952, *[int_221953], **kwargs_221954)
        
        # Assigning a type to the variable 'call_assignment_221508' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221508', getitem___call_result_221955)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'call_assignment_221508' (line 151)
        call_assignment_221508_221956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221508')
        # Assigning a type to the variable 'inkRect' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'inkRect', call_assignment_221508_221956)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
        # Processing the call keyword arguments
        kwargs_221960 = {}
        # Getting the type of 'call_assignment_221506' (line 151)
        call_assignment_221506_221957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221506', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___221958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), call_assignment_221506_221957, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221961 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221958, *[int_221959], **kwargs_221960)
        
        # Assigning a type to the variable 'call_assignment_221509' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221509', getitem___call_result_221961)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'call_assignment_221509' (line 151)
        call_assignment_221509_221962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_221509')
        # Assigning a type to the variable 'logicalRect' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'logicalRect', call_assignment_221509_221962)
        
        # Assigning a Name to a Tuple (line 152):
        
        # Assigning a Subscript to a Name (line 152):
        
        # Obtaining the type of the subscript
        int_221963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
        # Getting the type of 'inkRect' (line 152)
        inkRect_221964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___221965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), inkRect_221964, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_221966 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), getitem___221965, int_221963)
        
        # Assigning a type to the variable 'tuple_var_assignment_221510' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tuple_var_assignment_221510', subscript_call_result_221966)
        
        # Assigning a Subscript to a Name (line 152):
        
        # Obtaining the type of the subscript
        int_221967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
        # Getting the type of 'inkRect' (line 152)
        inkRect_221968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___221969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), inkRect_221968, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_221970 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), getitem___221969, int_221967)
        
        # Assigning a type to the variable 'tuple_var_assignment_221511' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tuple_var_assignment_221511', subscript_call_result_221970)
        
        # Assigning a Subscript to a Name (line 152):
        
        # Obtaining the type of the subscript
        int_221971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
        # Getting the type of 'inkRect' (line 152)
        inkRect_221972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___221973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), inkRect_221972, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_221974 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), getitem___221973, int_221971)
        
        # Assigning a type to the variable 'tuple_var_assignment_221512' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tuple_var_assignment_221512', subscript_call_result_221974)
        
        # Assigning a Subscript to a Name (line 152):
        
        # Obtaining the type of the subscript
        int_221975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
        # Getting the type of 'inkRect' (line 152)
        inkRect_221976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___221977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), inkRect_221976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_221978 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), getitem___221977, int_221975)
        
        # Assigning a type to the variable 'tuple_var_assignment_221513' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tuple_var_assignment_221513', subscript_call_result_221978)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'tuple_var_assignment_221510' (line 152)
        tuple_var_assignment_221510_221979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tuple_var_assignment_221510')
        # Assigning a type to the variable 'l' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'l', tuple_var_assignment_221510_221979)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'tuple_var_assignment_221511' (line 152)
        tuple_var_assignment_221511_221980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tuple_var_assignment_221511')
        # Assigning a type to the variable 'b' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'b', tuple_var_assignment_221511_221980)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'tuple_var_assignment_221512' (line 152)
        tuple_var_assignment_221512_221981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tuple_var_assignment_221512')
        # Assigning a type to the variable 'w' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'w', tuple_var_assignment_221512_221981)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'tuple_var_assignment_221513' (line 152)
        tuple_var_assignment_221513_221982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tuple_var_assignment_221513')
        # Assigning a type to the variable 'h' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'h', tuple_var_assignment_221513_221982)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 153)
        x_221983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'x')
        # Getting the type of 'w' (line 153)
        w_221984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'w')
        # Applying the binary operator '+' (line 153)
        result_add_221985 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 16), '+', x_221983, w_221984)
        
        # Getting the type of 'self' (line 153)
        self_221986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'self')
        # Obtaining the member 'width' of a type (line 153)
        width_221987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 24), self_221986, 'width')
        # Applying the binary operator '>' (line 153)
        result_gt_221988 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 16), '>', result_add_221985, width_221987)
        
        
        # Getting the type of 'y' (line 153)
        y_221989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'y')
        # Getting the type of 'h' (line 153)
        h_221990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 42), 'h')
        # Applying the binary operator '+' (line 153)
        result_add_221991 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 38), '+', y_221989, h_221990)
        
        # Getting the type of 'self' (line 153)
        self_221992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 46), 'self')
        # Obtaining the member 'height' of a type (line 153)
        height_221993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 46), self_221992, 'height')
        # Applying the binary operator '>' (line 153)
        result_gt_221994 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 38), '>', result_add_221991, height_221993)
        
        # Applying the binary operator 'or' (line 153)
        result_or_keyword_221995 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 16), 'or', result_gt_221988, result_gt_221994)
        
        # Testing the type of an if condition (line 153)
        if_condition_221996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 12), result_or_keyword_221995)
        # Assigning a type to the variable 'if_condition_221996' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'if_condition_221996', if_condition_221996)
        # SSA begins for if statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 153)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_layout(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'gc' (line 156)
        gc_222000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 41), 'gc', False)
        # Obtaining the member 'gdkGC' of a type (line 156)
        gdkGC_222001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 41), gc_222000, 'gdkGC')
        # Getting the type of 'x' (line 156)
        x_222002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 51), 'x', False)
        # Getting the type of 'y' (line 156)
        y_222003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 54), 'y', False)
        # Getting the type of 'h' (line 156)
        h_222004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 56), 'h', False)
        # Applying the binary operator '-' (line 156)
        result_sub_222005 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 54), '-', y_222003, h_222004)
        
        # Getting the type of 'b' (line 156)
        b_222006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 58), 'b', False)
        # Applying the binary operator '-' (line 156)
        result_sub_222007 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 57), '-', result_sub_222005, b_222006)
        
        # Getting the type of 'layout' (line 156)
        layout_222008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 61), 'layout', False)
        # Processing the call keyword arguments (line 156)
        kwargs_222009 = {}
        # Getting the type of 'self' (line 156)
        self_221997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'self', False)
        # Obtaining the member 'gdkDrawable' of a type (line 156)
        gdkDrawable_221998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), self_221997, 'gdkDrawable')
        # Obtaining the member 'draw_layout' of a type (line 156)
        draw_layout_221999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), gdkDrawable_221998, 'draw_layout')
        # Calling draw_layout(args, kwargs) (line 156)
        draw_layout_call_result_222010 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), draw_layout_221999, *[gdkGC_222001, x_222002, result_sub_222007, layout_222008], **kwargs_222009)
        
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_text' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_222011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222011)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_text'
        return stypy_return_type_222011


    @norecursion
    def _draw_mathtext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_draw_mathtext'
        module_type_store = module_type_store.open_function_context('_draw_mathtext', 158, 4, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_function_name', 'RendererGDK._draw_mathtext')
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle'])
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK._draw_mathtext.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK._draw_mathtext', ['gc', 'x', 'y', 's', 'prop', 'angle'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 159):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 's' (line 160)
        s_222015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 39), 's', False)
        # Getting the type of 'self' (line 160)
        self_222016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 42), 'self', False)
        # Obtaining the member 'dpi' of a type (line 160)
        dpi_222017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 42), self_222016, 'dpi')
        # Getting the type of 'prop' (line 160)
        prop_222018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 52), 'prop', False)
        # Processing the call keyword arguments (line 160)
        kwargs_222019 = {}
        # Getting the type of 'self' (line 160)
        self_222012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'self', False)
        # Obtaining the member 'mathtext_parser' of a type (line 160)
        mathtext_parser_222013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), self_222012, 'mathtext_parser')
        # Obtaining the member 'parse' of a type (line 160)
        parse_222014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), mathtext_parser_222013, 'parse')
        # Calling parse(args, kwargs) (line 160)
        parse_call_result_222020 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), parse_222014, *[s_222015, dpi_222017, prop_222018], **kwargs_222019)
        
        # Assigning a type to the variable 'call_assignment_221514' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221514', parse_call_result_222020)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222024 = {}
        # Getting the type of 'call_assignment_221514' (line 159)
        call_assignment_221514_222021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221514', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___222022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), call_assignment_221514_222021, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222025 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222022, *[int_222023], **kwargs_222024)
        
        # Assigning a type to the variable 'call_assignment_221515' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221515', getitem___call_result_222025)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'call_assignment_221515' (line 159)
        call_assignment_221515_222026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221515')
        # Assigning a type to the variable 'ox' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'ox', call_assignment_221515_222026)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222030 = {}
        # Getting the type of 'call_assignment_221514' (line 159)
        call_assignment_221514_222027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221514', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___222028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), call_assignment_221514_222027, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222031 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222028, *[int_222029], **kwargs_222030)
        
        # Assigning a type to the variable 'call_assignment_221516' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221516', getitem___call_result_222031)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'call_assignment_221516' (line 159)
        call_assignment_221516_222032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221516')
        # Assigning a type to the variable 'oy' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'oy', call_assignment_221516_222032)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222036 = {}
        # Getting the type of 'call_assignment_221514' (line 159)
        call_assignment_221514_222033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221514', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___222034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), call_assignment_221514_222033, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222037 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222034, *[int_222035], **kwargs_222036)
        
        # Assigning a type to the variable 'call_assignment_221517' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221517', getitem___call_result_222037)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'call_assignment_221517' (line 159)
        call_assignment_221517_222038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221517')
        # Assigning a type to the variable 'width' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'width', call_assignment_221517_222038)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222042 = {}
        # Getting the type of 'call_assignment_221514' (line 159)
        call_assignment_221514_222039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221514', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___222040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), call_assignment_221514_222039, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222043 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222040, *[int_222041], **kwargs_222042)
        
        # Assigning a type to the variable 'call_assignment_221518' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221518', getitem___call_result_222043)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'call_assignment_221518' (line 159)
        call_assignment_221518_222044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221518')
        # Assigning a type to the variable 'height' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'height', call_assignment_221518_222044)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222048 = {}
        # Getting the type of 'call_assignment_221514' (line 159)
        call_assignment_221514_222045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221514', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___222046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), call_assignment_221514_222045, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222049 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222046, *[int_222047], **kwargs_222048)
        
        # Assigning a type to the variable 'call_assignment_221519' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221519', getitem___call_result_222049)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'call_assignment_221519' (line 159)
        call_assignment_221519_222050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221519')
        # Assigning a type to the variable 'descent' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'descent', call_assignment_221519_222050)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222054 = {}
        # Getting the type of 'call_assignment_221514' (line 159)
        call_assignment_221514_222051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221514', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___222052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), call_assignment_221514_222051, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222055 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222052, *[int_222053], **kwargs_222054)
        
        # Assigning a type to the variable 'call_assignment_221520' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221520', getitem___call_result_222055)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'call_assignment_221520' (line 159)
        call_assignment_221520_222056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221520')
        # Assigning a type to the variable 'font_image' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 40), 'font_image', call_assignment_221520_222056)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222060 = {}
        # Getting the type of 'call_assignment_221514' (line 159)
        call_assignment_221514_222057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221514', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___222058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), call_assignment_221514_222057, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222061 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222058, *[int_222059], **kwargs_222060)
        
        # Assigning a type to the variable 'call_assignment_221521' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221521', getitem___call_result_222061)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'call_assignment_221521' (line 159)
        call_assignment_221521_222062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'call_assignment_221521')
        # Assigning a type to the variable 'used_characters' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 52), 'used_characters', call_assignment_221521_222062)
        
        
        # Getting the type of 'angle' (line 162)
        angle_222063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'angle')
        int_222064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'int')
        # Applying the binary operator '==' (line 162)
        result_eq_222065 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), '==', angle_222063, int_222064)
        
        # Testing the type of an if condition (line 162)
        if_condition_222066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), result_eq_222065)
        # Assigning a type to the variable 'if_condition_222066' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_222066', if_condition_222066)
        # SSA begins for if statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 163):
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'height' (line 163)
        height_222067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'height')
        # Assigning a type to the variable 'tuple_assignment_221522' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'tuple_assignment_221522', height_222067)
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'width' (line 163)
        width_222068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 36), 'width')
        # Assigning a type to the variable 'tuple_assignment_221523' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'tuple_assignment_221523', width_222068)
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'tuple_assignment_221522' (line 163)
        tuple_assignment_221522_222069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'tuple_assignment_221522')
        # Assigning a type to the variable 'width' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'width', tuple_assignment_221522_222069)
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'tuple_assignment_221523' (line 163)
        tuple_assignment_221523_222070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'tuple_assignment_221523')
        # Assigning a type to the variable 'height' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'height', tuple_assignment_221523_222070)
        
        # Getting the type of 'x' (line 164)
        x_222071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'x')
        # Getting the type of 'width' (line 164)
        width_222072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'width')
        # Applying the binary operator '-=' (line 164)
        result_isub_222073 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 12), '-=', x_222071, width_222072)
        # Assigning a type to the variable 'x' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'x', result_isub_222073)
        
        # SSA join for if statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'y' (line 165)
        y_222074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'y')
        # Getting the type of 'height' (line 165)
        height_222075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'height')
        # Applying the binary operator '-=' (line 165)
        result_isub_222076 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 8), '-=', y_222074, height_222075)
        # Assigning a type to the variable 'y' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'y', result_isub_222076)
        
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to get_width(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_222079 = {}
        # Getting the type of 'font_image' (line 167)
        font_image_222077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'font_image', False)
        # Obtaining the member 'get_width' of a type (line 167)
        get_width_222078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 14), font_image_222077, 'get_width')
        # Calling get_width(args, kwargs) (line 167)
        get_width_call_result_222080 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), get_width_222078, *[], **kwargs_222079)
        
        # Assigning a type to the variable 'imw' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'imw', get_width_call_result_222080)
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to get_height(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_222083 = {}
        # Getting the type of 'font_image' (line 168)
        font_image_222081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 14), 'font_image', False)
        # Obtaining the member 'get_height' of a type (line 168)
        get_height_222082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 14), font_image_222081, 'get_height')
        # Calling get_height(args, kwargs) (line 168)
        get_height_call_result_222084 = invoke(stypy.reporting.localization.Localization(__file__, 168, 14), get_height_222082, *[], **kwargs_222083)
        
        # Assigning a type to the variable 'imh' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'imh', get_height_call_result_222084)
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to Pixbuf(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'gtk' (line 170)
        gtk_222088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 32), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 170)
        gdk_222089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 32), gtk_222088, 'gdk')
        # Obtaining the member 'COLORSPACE_RGB' of a type (line 170)
        COLORSPACE_RGB_222090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 32), gdk_222089, 'COLORSPACE_RGB')
        # Processing the call keyword arguments (line 170)
        # Getting the type of 'True' (line 170)
        True_222091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 66), 'True', False)
        keyword_222092 = True_222091
        int_222093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 48), 'int')
        keyword_222094 = int_222093
        # Getting the type of 'imw' (line 171)
        imw_222095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 57), 'imw', False)
        keyword_222096 = imw_222095
        # Getting the type of 'imh' (line 171)
        imh_222097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 69), 'imh', False)
        keyword_222098 = imh_222097
        kwargs_222099 = {'has_alpha': keyword_222092, 'width': keyword_222096, 'bits_per_sample': keyword_222094, 'height': keyword_222098}
        # Getting the type of 'gtk' (line 170)
        gtk_222085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 170)
        gdk_222086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 17), gtk_222085, 'gdk')
        # Obtaining the member 'Pixbuf' of a type (line 170)
        Pixbuf_222087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 17), gdk_222086, 'Pixbuf')
        # Calling Pixbuf(args, kwargs) (line 170)
        Pixbuf_call_result_222100 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), Pixbuf_222087, *[COLORSPACE_RGB_222090], **kwargs_222099)
        
        # Assigning a type to the variable 'pixbuf' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'pixbuf', Pixbuf_call_result_222100)
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to pixbuf_get_pixels_array(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'pixbuf' (line 173)
        pixbuf_222102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 40), 'pixbuf', False)
        # Processing the call keyword arguments (line 173)
        kwargs_222103 = {}
        # Getting the type of 'pixbuf_get_pixels_array' (line 173)
        pixbuf_get_pixels_array_222101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'pixbuf_get_pixels_array', False)
        # Calling pixbuf_get_pixels_array(args, kwargs) (line 173)
        pixbuf_get_pixels_array_call_result_222104 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), pixbuf_get_pixels_array_222101, *[pixbuf_222102], **kwargs_222103)
        
        # Assigning a type to the variable 'array' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'array', pixbuf_get_pixels_array_call_result_222104)
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to get_rgb(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_222107 = {}
        # Getting the type of 'gc' (line 175)
        gc_222105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'gc', False)
        # Obtaining the member 'get_rgb' of a type (line 175)
        get_rgb_222106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 14), gc_222105, 'get_rgb')
        # Calling get_rgb(args, kwargs) (line 175)
        get_rgb_call_result_222108 = invoke(stypy.reporting.localization.Localization(__file__, 175, 14), get_rgb_222106, *[], **kwargs_222107)
        
        # Assigning a type to the variable 'rgb' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'rgb', get_rgb_call_result_222108)
        
        # Assigning a Call to a Subscript (line 176):
        
        # Assigning a Call to a Subscript (line 176):
        
        # Call to int(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining the type of the subscript
        int_222110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 31), 'int')
        # Getting the type of 'rgb' (line 176)
        rgb_222111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___222112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 27), rgb_222111, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_222113 = invoke(stypy.reporting.localization.Localization(__file__, 176, 27), getitem___222112, int_222110)
        
        int_222114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 34), 'int')
        # Applying the binary operator '*' (line 176)
        result_mul_222115 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 27), '*', subscript_call_result_222113, int_222114)
        
        # Processing the call keyword arguments (line 176)
        kwargs_222116 = {}
        # Getting the type of 'int' (line 176)
        int_222109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'int', False)
        # Calling int(args, kwargs) (line 176)
        int_call_result_222117 = invoke(stypy.reporting.localization.Localization(__file__, 176, 23), int_222109, *[result_mul_222115], **kwargs_222116)
        
        # Getting the type of 'array' (line 176)
        array_222118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'array')
        slice_222119 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 176, 8), None, None, None)
        slice_222120 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 176, 8), None, None, None)
        int_222121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'int')
        # Storing an element on a container (line 176)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 8), array_222118, ((slice_222119, slice_222120, int_222121), int_call_result_222117))
        
        # Assigning a Call to a Subscript (line 177):
        
        # Assigning a Call to a Subscript (line 177):
        
        # Call to int(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Obtaining the type of the subscript
        int_222123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'int')
        # Getting the type of 'rgb' (line 177)
        rgb_222124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___222125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 27), rgb_222124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_222126 = invoke(stypy.reporting.localization.Localization(__file__, 177, 27), getitem___222125, int_222123)
        
        int_222127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 34), 'int')
        # Applying the binary operator '*' (line 177)
        result_mul_222128 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 27), '*', subscript_call_result_222126, int_222127)
        
        # Processing the call keyword arguments (line 177)
        kwargs_222129 = {}
        # Getting the type of 'int' (line 177)
        int_222122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'int', False)
        # Calling int(args, kwargs) (line 177)
        int_call_result_222130 = invoke(stypy.reporting.localization.Localization(__file__, 177, 23), int_222122, *[result_mul_222128], **kwargs_222129)
        
        # Getting the type of 'array' (line 177)
        array_222131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'array')
        slice_222132 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 177, 8), None, None, None)
        slice_222133 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 177, 8), None, None, None)
        int_222134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 18), 'int')
        # Storing an element on a container (line 177)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 8), array_222131, ((slice_222132, slice_222133, int_222134), int_call_result_222130))
        
        # Assigning a Call to a Subscript (line 178):
        
        # Assigning a Call to a Subscript (line 178):
        
        # Call to int(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining the type of the subscript
        int_222136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 31), 'int')
        # Getting the type of 'rgb' (line 178)
        rgb_222137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___222138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 27), rgb_222137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_222139 = invoke(stypy.reporting.localization.Localization(__file__, 178, 27), getitem___222138, int_222136)
        
        int_222140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'int')
        # Applying the binary operator '*' (line 178)
        result_mul_222141 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 27), '*', subscript_call_result_222139, int_222140)
        
        # Processing the call keyword arguments (line 178)
        kwargs_222142 = {}
        # Getting the type of 'int' (line 178)
        int_222135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'int', False)
        # Calling int(args, kwargs) (line 178)
        int_call_result_222143 = invoke(stypy.reporting.localization.Localization(__file__, 178, 23), int_222135, *[result_mul_222141], **kwargs_222142)
        
        # Getting the type of 'array' (line 178)
        array_222144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'array')
        slice_222145 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 8), None, None, None)
        slice_222146 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 8), None, None, None)
        int_222147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 18), 'int')
        # Storing an element on a container (line 178)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 8), array_222144, ((slice_222145, slice_222146, int_222147), int_call_result_222143))
        
        # Assigning a Call to a Subscript (line 179):
        
        # Assigning a Call to a Subscript (line 179):
        
        # Call to reshape(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_222159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        # Getting the type of 'imh' (line 180)
        imh_222160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 66), 'imh', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 66), tuple_222159, imh_222160)
        # Adding element type (line 180)
        # Getting the type of 'imw' (line 180)
        imw_222161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 71), 'imw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 66), tuple_222159, imw_222161)
        
        # Processing the call keyword arguments (line 180)
        kwargs_222162 = {}
        
        # Call to fromstring(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to as_str(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_222152 = {}
        # Getting the type of 'font_image' (line 180)
        font_image_222150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 26), 'font_image', False)
        # Obtaining the member 'as_str' of a type (line 180)
        as_str_222151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 26), font_image_222150, 'as_str')
        # Calling as_str(args, kwargs) (line 180)
        as_str_call_result_222153 = invoke(stypy.reporting.localization.Localization(__file__, 180, 26), as_str_222151, *[], **kwargs_222152)
        
        # Getting the type of 'np' (line 180)
        np_222154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 47), 'np', False)
        # Obtaining the member 'uint8' of a type (line 180)
        uint8_222155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 47), np_222154, 'uint8')
        # Processing the call keyword arguments (line 180)
        kwargs_222156 = {}
        # Getting the type of 'np' (line 180)
        np_222148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'np', False)
        # Obtaining the member 'fromstring' of a type (line 180)
        fromstring_222149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), np_222148, 'fromstring')
        # Calling fromstring(args, kwargs) (line 180)
        fromstring_call_result_222157 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), fromstring_222149, *[as_str_call_result_222153, uint8_222155], **kwargs_222156)
        
        # Obtaining the member 'reshape' of a type (line 180)
        reshape_222158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), fromstring_call_result_222157, 'reshape')
        # Calling reshape(args, kwargs) (line 180)
        reshape_call_result_222163 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), reshape_222158, *[tuple_222159], **kwargs_222162)
        
        # Getting the type of 'array' (line 179)
        array_222164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'array')
        slice_222165 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 179, 8), None, None, None)
        slice_222166 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 179, 8), None, None, None)
        int_222167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 18), 'int')
        # Storing an element on a container (line 179)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), array_222164, ((slice_222165, slice_222166, int_222167), reshape_call_result_222163))
        
        # Call to draw_pixbuf(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'gc' (line 183)
        gc_222171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 37), 'gc', False)
        # Obtaining the member 'gdkGC' of a type (line 183)
        gdkGC_222172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 37), gc_222171, 'gdkGC')
        # Getting the type of 'pixbuf' (line 183)
        pixbuf_222173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 47), 'pixbuf', False)
        int_222174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 55), 'int')
        int_222175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 58), 'int')
        
        # Call to int(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'x' (line 184)
        x_222177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 41), 'x', False)
        # Processing the call keyword arguments (line 184)
        kwargs_222178 = {}
        # Getting the type of 'int' (line 184)
        int_222176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'int', False)
        # Calling int(args, kwargs) (line 184)
        int_call_result_222179 = invoke(stypy.reporting.localization.Localization(__file__, 184, 37), int_222176, *[x_222177], **kwargs_222178)
        
        
        # Call to int(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'y' (line 184)
        y_222181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 49), 'y', False)
        # Processing the call keyword arguments (line 184)
        kwargs_222182 = {}
        # Getting the type of 'int' (line 184)
        int_222180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 45), 'int', False)
        # Calling int(args, kwargs) (line 184)
        int_call_result_222183 = invoke(stypy.reporting.localization.Localization(__file__, 184, 45), int_222180, *[y_222181], **kwargs_222182)
        
        # Getting the type of 'imw' (line 184)
        imw_222184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 53), 'imw', False)
        # Getting the type of 'imh' (line 184)
        imh_222185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 58), 'imh', False)
        # Getting the type of 'gdk' (line 185)
        gdk_222186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 37), 'gdk', False)
        # Obtaining the member 'RGB_DITHER_NONE' of a type (line 185)
        RGB_DITHER_NONE_222187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 37), gdk_222186, 'RGB_DITHER_NONE')
        int_222188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 58), 'int')
        int_222189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 61), 'int')
        # Processing the call keyword arguments (line 183)
        kwargs_222190 = {}
        # Getting the type of 'self' (line 183)
        self_222168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self', False)
        # Obtaining the member 'gdkDrawable' of a type (line 183)
        gdkDrawable_222169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_222168, 'gdkDrawable')
        # Obtaining the member 'draw_pixbuf' of a type (line 183)
        draw_pixbuf_222170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), gdkDrawable_222169, 'draw_pixbuf')
        # Calling draw_pixbuf(args, kwargs) (line 183)
        draw_pixbuf_call_result_222191 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), draw_pixbuf_222170, *[gdkGC_222172, pixbuf_222173, int_222174, int_222175, int_call_result_222179, int_call_result_222183, imw_222184, imh_222185, RGB_DITHER_NONE_222187, int_222188, int_222189], **kwargs_222190)
        
        
        # ################# End of '_draw_mathtext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_draw_mathtext' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_222192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222192)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_draw_mathtext'
        return stypy_return_type_222192


    @norecursion
    def _draw_rotated_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_draw_rotated_text'
        module_type_store = module_type_store.open_function_context('_draw_rotated_text', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_function_name', 'RendererGDK._draw_rotated_text')
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle'])
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK._draw_rotated_text.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK._draw_rotated_text', ['gc', 'x', 'y', 's', 'prop', 'angle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_draw_rotated_text', localization, ['gc', 'x', 'y', 's', 'prop', 'angle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_draw_rotated_text(...)' code ##################

        unicode_222193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'unicode', u'\n        Draw the text rotated 90 degrees, other angles are not supported\n        ')
        
        # Assigning a Attribute to a Name (line 196):
        
        # Assigning a Attribute to a Name (line 196):
        # Getting the type of 'self' (line 196)
        self_222194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'self')
        # Obtaining the member 'gdkDrawable' of a type (line 196)
        gdkDrawable_222195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 20), self_222194, 'gdkDrawable')
        # Assigning a type to the variable 'gdrawable' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'gdrawable', gdkDrawable_222195)
        
        # Assigning a Attribute to a Name (line 197):
        
        # Assigning a Attribute to a Name (line 197):
        # Getting the type of 'gc' (line 197)
        gc_222196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 14), 'gc')
        # Obtaining the member 'gdkGC' of a type (line 197)
        gdkGC_222197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 14), gc_222196, 'gdkGC')
        # Assigning a type to the variable 'ggc' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'ggc', gdkGC_222197)
        
        # Assigning a Call to a Tuple (line 199):
        
        # Assigning a Call to a Name:
        
        # Call to _get_pango_layout(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 's' (line 199)
        s_222200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 62), 's', False)
        # Getting the type of 'prop' (line 199)
        prop_222201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 65), 'prop', False)
        # Processing the call keyword arguments (line 199)
        kwargs_222202 = {}
        # Getting the type of 'self' (line 199)
        self_222198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'self', False)
        # Obtaining the member '_get_pango_layout' of a type (line 199)
        _get_pango_layout_222199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 39), self_222198, '_get_pango_layout')
        # Calling _get_pango_layout(args, kwargs) (line 199)
        _get_pango_layout_call_result_222203 = invoke(stypy.reporting.localization.Localization(__file__, 199, 39), _get_pango_layout_222199, *[s_222200, prop_222201], **kwargs_222202)
        
        # Assigning a type to the variable 'call_assignment_221524' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221524', _get_pango_layout_call_result_222203)
        
        # Assigning a Call to a Name (line 199):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222207 = {}
        # Getting the type of 'call_assignment_221524' (line 199)
        call_assignment_221524_222204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221524', False)
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___222205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), call_assignment_221524_222204, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222208 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222205, *[int_222206], **kwargs_222207)
        
        # Assigning a type to the variable 'call_assignment_221525' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221525', getitem___call_result_222208)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'call_assignment_221525' (line 199)
        call_assignment_221525_222209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221525')
        # Assigning a type to the variable 'layout' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'layout', call_assignment_221525_222209)
        
        # Assigning a Call to a Name (line 199):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222213 = {}
        # Getting the type of 'call_assignment_221524' (line 199)
        call_assignment_221524_222210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221524', False)
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___222211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), call_assignment_221524_222210, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222214 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222211, *[int_222212], **kwargs_222213)
        
        # Assigning a type to the variable 'call_assignment_221526' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221526', getitem___call_result_222214)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'call_assignment_221526' (line 199)
        call_assignment_221526_222215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221526')
        # Assigning a type to the variable 'inkRect' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'inkRect', call_assignment_221526_222215)
        
        # Assigning a Call to a Name (line 199):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222219 = {}
        # Getting the type of 'call_assignment_221524' (line 199)
        call_assignment_221524_222216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221524', False)
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___222217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), call_assignment_221524_222216, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222220 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222217, *[int_222218], **kwargs_222219)
        
        # Assigning a type to the variable 'call_assignment_221527' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221527', getitem___call_result_222220)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'call_assignment_221527' (line 199)
        call_assignment_221527_222221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'call_assignment_221527')
        # Assigning a type to the variable 'logicalRect' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'logicalRect', call_assignment_221527_222221)
        
        # Assigning a Name to a Tuple (line 200):
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_222222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        # Getting the type of 'inkRect' (line 200)
        inkRect_222223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___222224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), inkRect_222223, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_222225 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___222224, int_222222)
        
        # Assigning a type to the variable 'tuple_var_assignment_221528' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_221528', subscript_call_result_222225)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_222226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        # Getting the type of 'inkRect' (line 200)
        inkRect_222227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___222228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), inkRect_222227, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_222229 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___222228, int_222226)
        
        # Assigning a type to the variable 'tuple_var_assignment_221529' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_221529', subscript_call_result_222229)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_222230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        # Getting the type of 'inkRect' (line 200)
        inkRect_222231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___222232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), inkRect_222231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_222233 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___222232, int_222230)
        
        # Assigning a type to the variable 'tuple_var_assignment_221530' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_221530', subscript_call_result_222233)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_222234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        # Getting the type of 'inkRect' (line 200)
        inkRect_222235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___222236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), inkRect_222235, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_222237 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___222236, int_222234)
        
        # Assigning a type to the variable 'tuple_var_assignment_221531' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_221531', subscript_call_result_222237)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_221528' (line 200)
        tuple_var_assignment_221528_222238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_221528')
        # Assigning a type to the variable 'l' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'l', tuple_var_assignment_221528_222238)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_221529' (line 200)
        tuple_var_assignment_221529_222239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_221529')
        # Assigning a type to the variable 'b' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'b', tuple_var_assignment_221529_222239)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_221530' (line 200)
        tuple_var_assignment_221530_222240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_221530')
        # Assigning a type to the variable 'w' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'w', tuple_var_assignment_221530_222240)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_221531' (line 200)
        tuple_var_assignment_221531_222241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_221531')
        # Assigning a type to the variable 'h' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'h', tuple_var_assignment_221531_222241)
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to int(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'x' (line 201)
        x_222243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'x', False)
        # Getting the type of 'h' (line 201)
        h_222244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'h', False)
        # Applying the binary operator '-' (line 201)
        result_sub_222245 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 16), '-', x_222243, h_222244)
        
        # Processing the call keyword arguments (line 201)
        kwargs_222246 = {}
        # Getting the type of 'int' (line 201)
        int_222242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'int', False)
        # Calling int(args, kwargs) (line 201)
        int_call_result_222247 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), int_222242, *[result_sub_222245], **kwargs_222246)
        
        # Assigning a type to the variable 'x' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'x', int_call_result_222247)
        
        # Assigning a Call to a Name (line 202):
        
        # Assigning a Call to a Name (line 202):
        
        # Call to int(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'y' (line 202)
        y_222249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'y', False)
        # Getting the type of 'w' (line 202)
        w_222250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'w', False)
        # Applying the binary operator '-' (line 202)
        result_sub_222251 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 16), '-', y_222249, w_222250)
        
        # Processing the call keyword arguments (line 202)
        kwargs_222252 = {}
        # Getting the type of 'int' (line 202)
        int_222248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'int', False)
        # Calling int(args, kwargs) (line 202)
        int_call_result_222253 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), int_222248, *[result_sub_222251], **kwargs_222252)
        
        # Assigning a type to the variable 'y' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'y', int_call_result_222253)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 204)
        x_222254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'x')
        int_222255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 16), 'int')
        # Applying the binary operator '<' (line 204)
        result_lt_222256 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 12), '<', x_222254, int_222255)
        
        
        # Getting the type of 'y' (line 204)
        y_222257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'y')
        int_222258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'int')
        # Applying the binary operator '<' (line 204)
        result_lt_222259 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 21), '<', y_222257, int_222258)
        
        # Applying the binary operator 'or' (line 204)
        result_or_keyword_222260 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 12), 'or', result_lt_222256, result_lt_222259)
        
        # Getting the type of 'x' (line 205)
        x_222261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'x')
        # Getting the type of 'w' (line 205)
        w_222262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'w')
        # Applying the binary operator '+' (line 205)
        result_add_222263 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 12), '+', x_222261, w_222262)
        
        # Getting the type of 'self' (line 205)
        self_222264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'self')
        # Obtaining the member 'width' of a type (line 205)
        width_222265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 20), self_222264, 'width')
        # Applying the binary operator '>' (line 205)
        result_gt_222266 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 12), '>', result_add_222263, width_222265)
        
        # Applying the binary operator 'or' (line 204)
        result_or_keyword_222267 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 12), 'or', result_or_keyword_222260, result_gt_222266)
        
        # Getting the type of 'y' (line 205)
        y_222268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 34), 'y')
        # Getting the type of 'h' (line 205)
        h_222269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 38), 'h')
        # Applying the binary operator '+' (line 205)
        result_add_222270 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 34), '+', y_222268, h_222269)
        
        # Getting the type of 'self' (line 205)
        self_222271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 42), 'self')
        # Obtaining the member 'height' of a type (line 205)
        height_222272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 42), self_222271, 'height')
        # Applying the binary operator '>' (line 205)
        result_gt_222273 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 34), '>', result_add_222270, height_222272)
        
        # Applying the binary operator 'or' (line 204)
        result_or_keyword_222274 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 12), 'or', result_or_keyword_222267, result_gt_222273)
        
        # Testing the type of an if condition (line 204)
        if_condition_222275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), result_or_keyword_222274)
        # Assigning a type to the variable 'if_condition_222275' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_222275', if_condition_222275)
        # SSA begins for if statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 204)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 208):
        
        # Assigning a Tuple to a Name (line 208):
        
        # Obtaining an instance of the builtin type 'tuple' (line 208)
        tuple_222276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 208)
        # Adding element type (line 208)
        # Getting the type of 'x' (line 208)
        x_222277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_222276, x_222277)
        # Adding element type (line 208)
        # Getting the type of 'y' (line 208)
        y_222278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_222276, y_222278)
        # Adding element type (line 208)
        # Getting the type of 's' (line 208)
        s_222279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_222276, s_222279)
        # Adding element type (line 208)
        # Getting the type of 'angle' (line 208)
        angle_222280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 21), 'angle')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_222276, angle_222280)
        # Adding element type (line 208)
        
        # Call to hash(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'prop' (line 208)
        prop_222282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), 'prop', False)
        # Processing the call keyword arguments (line 208)
        kwargs_222283 = {}
        # Getting the type of 'hash' (line 208)
        hash_222281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'hash', False)
        # Calling hash(args, kwargs) (line 208)
        hash_call_result_222284 = invoke(stypy.reporting.localization.Localization(__file__, 208, 27), hash_222281, *[prop_222282], **kwargs_222283)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_222276, hash_call_result_222284)
        
        # Assigning a type to the variable 'key' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'key', tuple_222276)
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to get(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'key' (line 209)
        key_222288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 37), 'key', False)
        # Processing the call keyword arguments (line 209)
        kwargs_222289 = {}
        # Getting the type of 'self' (line 209)
        self_222285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'self', False)
        # Obtaining the member 'rotated' of a type (line 209)
        rotated_222286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), self_222285, 'rotated')
        # Obtaining the member 'get' of a type (line 209)
        get_222287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), rotated_222286, 'get')
        # Calling get(args, kwargs) (line 209)
        get_call_result_222290 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), get_222287, *[key_222288], **kwargs_222289)
        
        # Assigning a type to the variable 'imageVert' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'imageVert', get_call_result_222290)
        
        
        # Getting the type of 'imageVert' (line 210)
        imageVert_222291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'imageVert')
        # Getting the type of 'None' (line 210)
        None_222292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), 'None')
        # Applying the binary operator '!=' (line 210)
        result_ne_222293 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 11), '!=', imageVert_222291, None_222292)
        
        # Testing the type of an if condition (line 210)
        if_condition_222294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), result_ne_222293)
        # Assigning a type to the variable 'if_condition_222294' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_222294', if_condition_222294)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to draw_image(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'ggc' (line 211)
        ggc_222297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 33), 'ggc', False)
        # Getting the type of 'imageVert' (line 211)
        imageVert_222298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'imageVert', False)
        int_222299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 49), 'int')
        int_222300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 52), 'int')
        # Getting the type of 'x' (line 211)
        x_222301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 55), 'x', False)
        # Getting the type of 'y' (line 211)
        y_222302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 58), 'y', False)
        # Getting the type of 'h' (line 211)
        h_222303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 61), 'h', False)
        # Getting the type of 'w' (line 211)
        w_222304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 64), 'w', False)
        # Processing the call keyword arguments (line 211)
        kwargs_222305 = {}
        # Getting the type of 'gdrawable' (line 211)
        gdrawable_222295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'gdrawable', False)
        # Obtaining the member 'draw_image' of a type (line 211)
        draw_image_222296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), gdrawable_222295, 'draw_image')
        # Calling draw_image(args, kwargs) (line 211)
        draw_image_call_result_222306 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), draw_image_222296, *[ggc_222297, imageVert_222298, int_222299, int_222300, x_222301, y_222302, h_222303, w_222304], **kwargs_222305)
        
        # Assigning a type to the variable 'stypy_return_type' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to get_image(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'x' (line 214)
        x_222309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 40), 'x', False)
        # Getting the type of 'y' (line 214)
        y_222310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), 'y', False)
        # Getting the type of 'w' (line 214)
        w_222311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 46), 'w', False)
        # Getting the type of 'h' (line 214)
        h_222312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 49), 'h', False)
        # Processing the call keyword arguments (line 214)
        kwargs_222313 = {}
        # Getting the type of 'gdrawable' (line 214)
        gdrawable_222307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'gdrawable', False)
        # Obtaining the member 'get_image' of a type (line 214)
        get_image_222308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), gdrawable_222307, 'get_image')
        # Calling get_image(args, kwargs) (line 214)
        get_image_call_result_222314 = invoke(stypy.reporting.localization.Localization(__file__, 214, 20), get_image_222308, *[x_222309, y_222310, w_222311, h_222312], **kwargs_222313)
        
        # Assigning a type to the variable 'imageBack' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'imageBack', get_image_call_result_222314)
        
        # Assigning a Call to a Name (line 215):
        
        # Assigning a Call to a Name (line 215):
        
        # Call to get_image(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'x' (line 215)
        x_222317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 40), 'x', False)
        # Getting the type of 'y' (line 215)
        y_222318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 43), 'y', False)
        # Getting the type of 'h' (line 215)
        h_222319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 46), 'h', False)
        # Getting the type of 'w' (line 215)
        w_222320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 49), 'w', False)
        # Processing the call keyword arguments (line 215)
        kwargs_222321 = {}
        # Getting the type of 'gdrawable' (line 215)
        gdrawable_222315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'gdrawable', False)
        # Obtaining the member 'get_image' of a type (line 215)
        get_image_222316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 20), gdrawable_222315, 'get_image')
        # Calling get_image(args, kwargs) (line 215)
        get_image_call_result_222322 = invoke(stypy.reporting.localization.Localization(__file__, 215, 20), get_image_222316, *[x_222317, y_222318, h_222319, w_222320], **kwargs_222321)
        
        # Assigning a type to the variable 'imageVert' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'imageVert', get_image_call_result_222322)
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to Image(...): (line 216)
        # Processing the call keyword arguments (line 216)
        # Getting the type of 'gdk' (line 216)
        gdk_222326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'gdk', False)
        # Obtaining the member 'IMAGE_FASTEST' of a type (line 216)
        IMAGE_FASTEST_222327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 39), gdk_222326, 'IMAGE_FASTEST')
        keyword_222328 = IMAGE_FASTEST_222327
        
        # Call to get_visual(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_222331 = {}
        # Getting the type of 'gdrawable' (line 217)
        gdrawable_222329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 41), 'gdrawable', False)
        # Obtaining the member 'get_visual' of a type (line 217)
        get_visual_222330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 41), gdrawable_222329, 'get_visual')
        # Calling get_visual(args, kwargs) (line 217)
        get_visual_call_result_222332 = invoke(stypy.reporting.localization.Localization(__file__, 217, 41), get_visual_222330, *[], **kwargs_222331)
        
        keyword_222333 = get_visual_call_result_222332
        # Getting the type of 'w' (line 218)
        w_222334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 40), 'w', False)
        keyword_222335 = w_222334
        # Getting the type of 'h' (line 218)
        h_222336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 50), 'h', False)
        keyword_222337 = h_222336
        kwargs_222338 = {'width': keyword_222335, 'visual': keyword_222333, 'type': keyword_222328, 'height': keyword_222337}
        # Getting the type of 'gtk' (line 216)
        gtk_222323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 216)
        gdk_222324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 20), gtk_222323, 'gdk')
        # Obtaining the member 'Image' of a type (line 216)
        Image_222325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 20), gdk_222324, 'Image')
        # Calling Image(args, kwargs) (line 216)
        Image_call_result_222339 = invoke(stypy.reporting.localization.Localization(__file__, 216, 20), Image_222325, *[], **kwargs_222338)
        
        # Assigning a type to the variable 'imageFlip' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'imageFlip', Image_call_result_222339)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'imageFlip' (line 219)
        imageFlip_222340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'imageFlip')
        # Getting the type of 'None' (line 219)
        None_222341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'None')
        # Applying the binary operator '==' (line 219)
        result_eq_222342 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), '==', imageFlip_222340, None_222341)
        
        
        # Getting the type of 'imageBack' (line 219)
        imageBack_222343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 32), 'imageBack')
        # Getting the type of 'None' (line 219)
        None_222344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 45), 'None')
        # Applying the binary operator '==' (line 219)
        result_eq_222345 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 32), '==', imageBack_222343, None_222344)
        
        # Applying the binary operator 'or' (line 219)
        result_or_keyword_222346 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), 'or', result_eq_222342, result_eq_222345)
        
        # Getting the type of 'imageVert' (line 219)
        imageVert_222347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 53), 'imageVert')
        # Getting the type of 'None' (line 219)
        None_222348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 66), 'None')
        # Applying the binary operator '==' (line 219)
        result_eq_222349 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 53), '==', imageVert_222347, None_222348)
        
        # Applying the binary operator 'or' (line 219)
        result_or_keyword_222350 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), 'or', result_or_keyword_222346, result_eq_222349)
        
        # Testing the type of an if condition (line 219)
        if_condition_222351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 8), result_or_keyword_222350)
        # Assigning a type to the variable 'if_condition_222351' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'if_condition_222351', if_condition_222351)
        # SSA begins for if statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 220)
        # Processing the call arguments (line 220)
        unicode_222354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 26), 'unicode', u'Could not renderer vertical text')
        # Processing the call keyword arguments (line 220)
        kwargs_222355 = {}
        # Getting the type of 'warnings' (line 220)
        warnings_222352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 220)
        warn_222353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), warnings_222352, 'warn')
        # Calling warn(args, kwargs) (line 220)
        warn_call_result_222356 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), warn_222353, *[unicode_222354], **kwargs_222355)
        
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_colormap(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'self' (line 222)
        self_222359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'self', False)
        # Obtaining the member '_cmap' of a type (line 222)
        _cmap_222360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 31), self_222359, '_cmap')
        # Processing the call keyword arguments (line 222)
        kwargs_222361 = {}
        # Getting the type of 'imageFlip' (line 222)
        imageFlip_222357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'imageFlip', False)
        # Obtaining the member 'set_colormap' of a type (line 222)
        set_colormap_222358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), imageFlip_222357, 'set_colormap')
        # Calling set_colormap(args, kwargs) (line 222)
        set_colormap_call_result_222362 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), set_colormap_222358, *[_cmap_222360], **kwargs_222361)
        
        
        
        # Call to range(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'w' (line 223)
        w_222364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'w', False)
        # Processing the call keyword arguments (line 223)
        kwargs_222365 = {}
        # Getting the type of 'range' (line 223)
        range_222363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'range', False)
        # Calling range(args, kwargs) (line 223)
        range_call_result_222366 = invoke(stypy.reporting.localization.Localization(__file__, 223, 17), range_222363, *[w_222364], **kwargs_222365)
        
        # Testing the type of a for loop iterable (line 223)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 223, 8), range_call_result_222366)
        # Getting the type of the for loop variable (line 223)
        for_loop_var_222367 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 223, 8), range_call_result_222366)
        # Assigning a type to the variable 'i' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'i', for_loop_var_222367)
        # SSA begins for a for statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'h' (line 224)
        h_222369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'h', False)
        # Processing the call keyword arguments (line 224)
        kwargs_222370 = {}
        # Getting the type of 'range' (line 224)
        range_222368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'range', False)
        # Calling range(args, kwargs) (line 224)
        range_call_result_222371 = invoke(stypy.reporting.localization.Localization(__file__, 224, 21), range_222368, *[h_222369], **kwargs_222370)
        
        # Testing the type of a for loop iterable (line 224)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 224, 12), range_call_result_222371)
        # Getting the type of the for loop variable (line 224)
        for_loop_var_222372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 224, 12), range_call_result_222371)
        # Assigning a type to the variable 'j' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'j', for_loop_var_222372)
        # SSA begins for a for statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to put_pixel(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'i' (line 225)
        i_222375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 36), 'i', False)
        # Getting the type of 'j' (line 225)
        j_222376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 39), 'j', False)
        
        # Call to get_pixel(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'j' (line 225)
        j_222379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 62), 'j', False)
        # Getting the type of 'w' (line 225)
        w_222380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 64), 'w', False)
        # Getting the type of 'i' (line 225)
        i_222381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 66), 'i', False)
        # Applying the binary operator '-' (line 225)
        result_sub_222382 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 64), '-', w_222380, i_222381)
        
        int_222383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 68), 'int')
        # Applying the binary operator '-' (line 225)
        result_sub_222384 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 67), '-', result_sub_222382, int_222383)
        
        # Processing the call keyword arguments (line 225)
        kwargs_222385 = {}
        # Getting the type of 'imageVert' (line 225)
        imageVert_222377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 42), 'imageVert', False)
        # Obtaining the member 'get_pixel' of a type (line 225)
        get_pixel_222378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 42), imageVert_222377, 'get_pixel')
        # Calling get_pixel(args, kwargs) (line 225)
        get_pixel_call_result_222386 = invoke(stypy.reporting.localization.Localization(__file__, 225, 42), get_pixel_222378, *[j_222379, result_sub_222384], **kwargs_222385)
        
        # Processing the call keyword arguments (line 225)
        kwargs_222387 = {}
        # Getting the type of 'imageFlip' (line 225)
        imageFlip_222373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'imageFlip', False)
        # Obtaining the member 'put_pixel' of a type (line 225)
        put_pixel_222374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), imageFlip_222373, 'put_pixel')
        # Calling put_pixel(args, kwargs) (line 225)
        put_pixel_call_result_222388 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), put_pixel_222374, *[i_222375, j_222376, get_pixel_call_result_222386], **kwargs_222387)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_image(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'ggc' (line 227)
        ggc_222391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'ggc', False)
        # Getting the type of 'imageFlip' (line 227)
        imageFlip_222392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'imageFlip', False)
        int_222393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 45), 'int')
        int_222394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 48), 'int')
        # Getting the type of 'x' (line 227)
        x_222395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 51), 'x', False)
        # Getting the type of 'y' (line 227)
        y_222396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 54), 'y', False)
        # Getting the type of 'w' (line 227)
        w_222397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 57), 'w', False)
        # Getting the type of 'h' (line 227)
        h_222398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 60), 'h', False)
        # Processing the call keyword arguments (line 227)
        kwargs_222399 = {}
        # Getting the type of 'gdrawable' (line 227)
        gdrawable_222389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'gdrawable', False)
        # Obtaining the member 'draw_image' of a type (line 227)
        draw_image_222390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), gdrawable_222389, 'draw_image')
        # Calling draw_image(args, kwargs) (line 227)
        draw_image_call_result_222400 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), draw_image_222390, *[ggc_222391, imageFlip_222392, int_222393, int_222394, x_222395, y_222396, w_222397, h_222398], **kwargs_222399)
        
        
        # Call to draw_layout(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'ggc' (line 228)
        ggc_222403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'ggc', False)
        # Getting the type of 'x' (line 228)
        x_222404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 35), 'x', False)
        # Getting the type of 'y' (line 228)
        y_222405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 'y', False)
        # Getting the type of 'b' (line 228)
        b_222406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 40), 'b', False)
        # Applying the binary operator '-' (line 228)
        result_sub_222407 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 38), '-', y_222405, b_222406)
        
        # Getting the type of 'layout' (line 228)
        layout_222408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 43), 'layout', False)
        # Processing the call keyword arguments (line 228)
        kwargs_222409 = {}
        # Getting the type of 'gdrawable' (line 228)
        gdrawable_222401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'gdrawable', False)
        # Obtaining the member 'draw_layout' of a type (line 228)
        draw_layout_222402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), gdrawable_222401, 'draw_layout')
        # Calling draw_layout(args, kwargs) (line 228)
        draw_layout_call_result_222410 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), draw_layout_222402, *[ggc_222403, x_222404, result_sub_222407, layout_222408], **kwargs_222409)
        
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to get_image(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'x' (line 230)
        x_222413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 39), 'x', False)
        # Getting the type of 'y' (line 230)
        y_222414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 42), 'y', False)
        # Getting the type of 'w' (line 230)
        w_222415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'w', False)
        # Getting the type of 'h' (line 230)
        h_222416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'h', False)
        # Processing the call keyword arguments (line 230)
        kwargs_222417 = {}
        # Getting the type of 'gdrawable' (line 230)
        gdrawable_222411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'gdrawable', False)
        # Obtaining the member 'get_image' of a type (line 230)
        get_image_222412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 19), gdrawable_222411, 'get_image')
        # Calling get_image(args, kwargs) (line 230)
        get_image_call_result_222418 = invoke(stypy.reporting.localization.Localization(__file__, 230, 19), get_image_222412, *[x_222413, y_222414, w_222415, h_222416], **kwargs_222417)
        
        # Assigning a type to the variable 'imageIn' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'imageIn', get_image_call_result_222418)
        
        
        # Call to range(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'w' (line 231)
        w_222420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'w', False)
        # Processing the call keyword arguments (line 231)
        kwargs_222421 = {}
        # Getting the type of 'range' (line 231)
        range_222419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'range', False)
        # Calling range(args, kwargs) (line 231)
        range_call_result_222422 = invoke(stypy.reporting.localization.Localization(__file__, 231, 17), range_222419, *[w_222420], **kwargs_222421)
        
        # Testing the type of a for loop iterable (line 231)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 231, 8), range_call_result_222422)
        # Getting the type of the for loop variable (line 231)
        for_loop_var_222423 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 231, 8), range_call_result_222422)
        # Assigning a type to the variable 'i' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'i', for_loop_var_222423)
        # SSA begins for a for statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'h' (line 232)
        h_222425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'h', False)
        # Processing the call keyword arguments (line 232)
        kwargs_222426 = {}
        # Getting the type of 'range' (line 232)
        range_222424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 21), 'range', False)
        # Calling range(args, kwargs) (line 232)
        range_call_result_222427 = invoke(stypy.reporting.localization.Localization(__file__, 232, 21), range_222424, *[h_222425], **kwargs_222426)
        
        # Testing the type of a for loop iterable (line 232)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 232, 12), range_call_result_222427)
        # Getting the type of the for loop variable (line 232)
        for_loop_var_222428 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 232, 12), range_call_result_222427)
        # Assigning a type to the variable 'j' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'j', for_loop_var_222428)
        # SSA begins for a for statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to put_pixel(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'j' (line 233)
        j_222431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 36), 'j', False)
        # Getting the type of 'i' (line 233)
        i_222432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 39), 'i', False)
        
        # Call to get_pixel(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'w' (line 233)
        w_222435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 60), 'w', False)
        # Getting the type of 'i' (line 233)
        i_222436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 62), 'i', False)
        # Applying the binary operator '-' (line 233)
        result_sub_222437 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 60), '-', w_222435, i_222436)
        
        int_222438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 64), 'int')
        # Applying the binary operator '-' (line 233)
        result_sub_222439 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 63), '-', result_sub_222437, int_222438)
        
        # Getting the type of 'j' (line 233)
        j_222440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 66), 'j', False)
        # Processing the call keyword arguments (line 233)
        kwargs_222441 = {}
        # Getting the type of 'imageIn' (line 233)
        imageIn_222433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 42), 'imageIn', False)
        # Obtaining the member 'get_pixel' of a type (line 233)
        get_pixel_222434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 42), imageIn_222433, 'get_pixel')
        # Calling get_pixel(args, kwargs) (line 233)
        get_pixel_call_result_222442 = invoke(stypy.reporting.localization.Localization(__file__, 233, 42), get_pixel_222434, *[result_sub_222439, j_222440], **kwargs_222441)
        
        # Processing the call keyword arguments (line 233)
        kwargs_222443 = {}
        # Getting the type of 'imageVert' (line 233)
        imageVert_222429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'imageVert', False)
        # Obtaining the member 'put_pixel' of a type (line 233)
        put_pixel_222430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), imageVert_222429, 'put_pixel')
        # Calling put_pixel(args, kwargs) (line 233)
        put_pixel_call_result_222444 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), put_pixel_222430, *[j_222431, i_222432, get_pixel_call_result_222442], **kwargs_222443)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_image(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'ggc' (line 235)
        ggc_222447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 29), 'ggc', False)
        # Getting the type of 'imageBack' (line 235)
        imageBack_222448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'imageBack', False)
        int_222449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 45), 'int')
        int_222450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 48), 'int')
        # Getting the type of 'x' (line 235)
        x_222451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 51), 'x', False)
        # Getting the type of 'y' (line 235)
        y_222452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 54), 'y', False)
        # Getting the type of 'w' (line 235)
        w_222453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 57), 'w', False)
        # Getting the type of 'h' (line 235)
        h_222454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 60), 'h', False)
        # Processing the call keyword arguments (line 235)
        kwargs_222455 = {}
        # Getting the type of 'gdrawable' (line 235)
        gdrawable_222445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'gdrawable', False)
        # Obtaining the member 'draw_image' of a type (line 235)
        draw_image_222446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), gdrawable_222445, 'draw_image')
        # Calling draw_image(args, kwargs) (line 235)
        draw_image_call_result_222456 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), draw_image_222446, *[ggc_222447, imageBack_222448, int_222449, int_222450, x_222451, y_222452, w_222453, h_222454], **kwargs_222455)
        
        
        # Call to draw_image(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'ggc' (line 236)
        ggc_222459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 29), 'ggc', False)
        # Getting the type of 'imageVert' (line 236)
        imageVert_222460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 34), 'imageVert', False)
        int_222461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 45), 'int')
        int_222462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 48), 'int')
        # Getting the type of 'x' (line 236)
        x_222463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 51), 'x', False)
        # Getting the type of 'y' (line 236)
        y_222464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 54), 'y', False)
        # Getting the type of 'h' (line 236)
        h_222465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 57), 'h', False)
        # Getting the type of 'w' (line 236)
        w_222466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 60), 'w', False)
        # Processing the call keyword arguments (line 236)
        kwargs_222467 = {}
        # Getting the type of 'gdrawable' (line 236)
        gdrawable_222457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'gdrawable', False)
        # Obtaining the member 'draw_image' of a type (line 236)
        draw_image_222458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), gdrawable_222457, 'draw_image')
        # Calling draw_image(args, kwargs) (line 236)
        draw_image_call_result_222468 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), draw_image_222458, *[ggc_222459, imageVert_222460, int_222461, int_222462, x_222463, y_222464, h_222465, w_222466], **kwargs_222467)
        
        
        # Assigning a Name to a Subscript (line 237):
        
        # Assigning a Name to a Subscript (line 237):
        # Getting the type of 'imageVert' (line 237)
        imageVert_222469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 28), 'imageVert')
        # Getting the type of 'self' (line 237)
        self_222470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self')
        # Obtaining the member 'rotated' of a type (line 237)
        rotated_222471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_222470, 'rotated')
        # Getting the type of 'key' (line 237)
        key_222472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'key')
        # Storing an element on a container (line 237)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 8), rotated_222471, (key_222472, imageVert_222469))
        
        # ################# End of '_draw_rotated_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_draw_rotated_text' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_222473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222473)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_draw_rotated_text'
        return stypy_return_type_222473


    @norecursion
    def _get_pango_layout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_pango_layout'
        module_type_store = module_type_store.open_function_context('_get_pango_layout', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_function_name', 'RendererGDK._get_pango_layout')
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_param_names_list', ['s', 'prop'])
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK._get_pango_layout.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK._get_pango_layout', ['s', 'prop'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_pango_layout', localization, ['s', 'prop'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_pango_layout(...)' code ##################

        unicode_222474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, (-1)), 'unicode', u"\n        Create a pango layout instance for Text 's' with properties 'prop'.\n        Return - pango layout (from cache if already exists)\n\n        Note that pango assumes a logical DPI of 96\n        Ref: pango/fonts.c/pango_font_description_set_size() manual page\n        ")
        
        # Assigning a Tuple to a Name (line 251):
        
        # Assigning a Tuple to a Name (line 251):
        
        # Obtaining an instance of the builtin type 'tuple' (line 251)
        tuple_222475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 251)
        # Adding element type (line 251)
        # Getting the type of 'self' (line 251)
        self_222476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 14), 'self')
        # Obtaining the member 'dpi' of a type (line 251)
        dpi_222477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 14), self_222476, 'dpi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 14), tuple_222475, dpi_222477)
        # Adding element type (line 251)
        # Getting the type of 's' (line 251)
        s_222478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 14), tuple_222475, s_222478)
        # Adding element type (line 251)
        
        # Call to hash(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'prop' (line 251)
        prop_222480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 32), 'prop', False)
        # Processing the call keyword arguments (line 251)
        kwargs_222481 = {}
        # Getting the type of 'hash' (line 251)
        hash_222479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'hash', False)
        # Calling hash(args, kwargs) (line 251)
        hash_call_result_222482 = invoke(stypy.reporting.localization.Localization(__file__, 251, 27), hash_222479, *[prop_222480], **kwargs_222481)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 14), tuple_222475, hash_call_result_222482)
        
        # Assigning a type to the variable 'key' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'key', tuple_222475)
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to get(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'key' (line 252)
        key_222486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 33), 'key', False)
        # Processing the call keyword arguments (line 252)
        kwargs_222487 = {}
        # Getting the type of 'self' (line 252)
        self_222483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'self', False)
        # Obtaining the member 'layoutd' of a type (line 252)
        layoutd_222484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), self_222483, 'layoutd')
        # Obtaining the member 'get' of a type (line 252)
        get_222485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), layoutd_222484, 'get')
        # Calling get(args, kwargs) (line 252)
        get_call_result_222488 = invoke(stypy.reporting.localization.Localization(__file__, 252, 16), get_222485, *[key_222486], **kwargs_222487)
        
        # Assigning a type to the variable 'value' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'value', get_call_result_222488)
        
        
        # Getting the type of 'value' (line 253)
        value_222489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'value')
        # Getting the type of 'None' (line 253)
        None_222490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'None')
        # Applying the binary operator '!=' (line 253)
        result_ne_222491 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 11), '!=', value_222489, None_222490)
        
        # Testing the type of an if condition (line 253)
        if_condition_222492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 8), result_ne_222491)
        # Assigning a type to the variable 'if_condition_222492' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'if_condition_222492', if_condition_222492)
        # SSA begins for if statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'value' (line 254)
        value_222493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'stypy_return_type', value_222493)
        # SSA join for if statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 256):
        
        # Assigning a BinOp to a Name (line 256):
        
        # Call to get_size_in_points(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_222496 = {}
        # Getting the type of 'prop' (line 256)
        prop_222494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 256)
        get_size_in_points_222495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 15), prop_222494, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 256)
        get_size_in_points_call_result_222497 = invoke(stypy.reporting.localization.Localization(__file__, 256, 15), get_size_in_points_222495, *[], **kwargs_222496)
        
        # Getting the type of 'self' (line 256)
        self_222498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 43), 'self')
        # Obtaining the member 'dpi' of a type (line 256)
        dpi_222499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 43), self_222498, 'dpi')
        # Applying the binary operator '*' (line 256)
        result_mul_222500 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 15), '*', get_size_in_points_call_result_222497, dpi_222499)
        
        float_222501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 54), 'float')
        # Applying the binary operator 'div' (line 256)
        result_div_222502 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 52), 'div', result_mul_222500, float_222501)
        
        # Assigning a type to the variable 'size' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'size', result_div_222502)
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to round(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'size' (line 257)
        size_222505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 24), 'size', False)
        # Processing the call keyword arguments (line 257)
        kwargs_222506 = {}
        # Getting the type of 'np' (line 257)
        np_222503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'np', False)
        # Obtaining the member 'round' of a type (line 257)
        round_222504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 15), np_222503, 'round')
        # Calling round(args, kwargs) (line 257)
        round_call_result_222507 = invoke(stypy.reporting.localization.Localization(__file__, 257, 15), round_222504, *[size_222505], **kwargs_222506)
        
        # Assigning a type to the variable 'size' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'size', round_call_result_222507)
        
        # Assigning a BinOp to a Name (line 259):
        
        # Assigning a BinOp to a Name (line 259):
        unicode_222508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 19), 'unicode', u'%s, %s %i')
        
        # Obtaining an instance of the builtin type 'tuple' (line 259)
        tuple_222509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 259)
        # Adding element type (line 259)
        
        # Call to get_name(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_222512 = {}
        # Getting the type of 'prop' (line 259)
        prop_222510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'prop', False)
        # Obtaining the member 'get_name' of a type (line 259)
        get_name_222511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 34), prop_222510, 'get_name')
        # Calling get_name(args, kwargs) (line 259)
        get_name_call_result_222513 = invoke(stypy.reporting.localization.Localization(__file__, 259, 34), get_name_222511, *[], **kwargs_222512)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 34), tuple_222509, get_name_call_result_222513)
        # Adding element type (line 259)
        
        # Call to get_style(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_222516 = {}
        # Getting the type of 'prop' (line 259)
        prop_222514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 51), 'prop', False)
        # Obtaining the member 'get_style' of a type (line 259)
        get_style_222515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 51), prop_222514, 'get_style')
        # Calling get_style(args, kwargs) (line 259)
        get_style_call_result_222517 = invoke(stypy.reporting.localization.Localization(__file__, 259, 51), get_style_222515, *[], **kwargs_222516)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 34), tuple_222509, get_style_call_result_222517)
        # Adding element type (line 259)
        # Getting the type of 'size' (line 259)
        size_222518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 69), 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 34), tuple_222509, size_222518)
        
        # Applying the binary operator '%' (line 259)
        result_mod_222519 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 19), '%', unicode_222508, tuple_222509)
        
        # Assigning a type to the variable 'font_str' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'font_str', result_mod_222519)
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to FontDescription(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'font_str' (line 260)
        font_str_222522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'font_str', False)
        # Processing the call keyword arguments (line 260)
        kwargs_222523 = {}
        # Getting the type of 'pango' (line 260)
        pango_222520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'pango', False)
        # Obtaining the member 'FontDescription' of a type (line 260)
        FontDescription_222521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), pango_222520, 'FontDescription')
        # Calling FontDescription(args, kwargs) (line 260)
        FontDescription_call_result_222524 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), FontDescription_222521, *[font_str_222522], **kwargs_222523)
        
        # Assigning a type to the variable 'font' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'font', FontDescription_call_result_222524)
        
        # Call to set_weight(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Obtaining the type of the subscript
        
        # Call to get_weight(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_222529 = {}
        # Getting the type of 'prop' (line 263)
        prop_222527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 41), 'prop', False)
        # Obtaining the member 'get_weight' of a type (line 263)
        get_weight_222528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 41), prop_222527, 'get_weight')
        # Calling get_weight(args, kwargs) (line 263)
        get_weight_call_result_222530 = invoke(stypy.reporting.localization.Localization(__file__, 263, 41), get_weight_222528, *[], **kwargs_222529)
        
        # Getting the type of 'self' (line 263)
        self_222531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'self', False)
        # Obtaining the member 'fontweights' of a type (line 263)
        fontweights_222532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 24), self_222531, 'fontweights')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___222533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 24), fontweights_222532, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_222534 = invoke(stypy.reporting.localization.Localization(__file__, 263, 24), getitem___222533, get_weight_call_result_222530)
        
        # Processing the call keyword arguments (line 263)
        kwargs_222535 = {}
        # Getting the type of 'font' (line 263)
        font_222525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'font', False)
        # Obtaining the member 'set_weight' of a type (line 263)
        set_weight_222526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), font_222525, 'set_weight')
        # Calling set_weight(args, kwargs) (line 263)
        set_weight_call_result_222536 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), set_weight_222526, *[subscript_call_result_222534], **kwargs_222535)
        
        
        # Assigning a Call to a Name (line 265):
        
        # Assigning a Call to a Name (line 265):
        
        # Call to create_pango_layout(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 's' (line 265)
        s_222540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 48), 's', False)
        # Processing the call keyword arguments (line 265)
        kwargs_222541 = {}
        # Getting the type of 'self' (line 265)
        self_222537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'self', False)
        # Obtaining the member 'gtkDA' of a type (line 265)
        gtkDA_222538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 17), self_222537, 'gtkDA')
        # Obtaining the member 'create_pango_layout' of a type (line 265)
        create_pango_layout_222539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 17), gtkDA_222538, 'create_pango_layout')
        # Calling create_pango_layout(args, kwargs) (line 265)
        create_pango_layout_call_result_222542 = invoke(stypy.reporting.localization.Localization(__file__, 265, 17), create_pango_layout_222539, *[s_222540], **kwargs_222541)
        
        # Assigning a type to the variable 'layout' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'layout', create_pango_layout_call_result_222542)
        
        # Call to set_font_description(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'font' (line 266)
        font_222545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 36), 'font', False)
        # Processing the call keyword arguments (line 266)
        kwargs_222546 = {}
        # Getting the type of 'layout' (line 266)
        layout_222543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'layout', False)
        # Obtaining the member 'set_font_description' of a type (line 266)
        set_font_description_222544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), layout_222543, 'set_font_description')
        # Calling set_font_description(args, kwargs) (line 266)
        set_font_description_call_result_222547 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), set_font_description_222544, *[font_222545], **kwargs_222546)
        
        
        # Assigning a Call to a Tuple (line 267):
        
        # Assigning a Call to a Name:
        
        # Call to get_pixel_extents(...): (line 267)
        # Processing the call keyword arguments (line 267)
        kwargs_222550 = {}
        # Getting the type of 'layout' (line 267)
        layout_222548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 31), 'layout', False)
        # Obtaining the member 'get_pixel_extents' of a type (line 267)
        get_pixel_extents_222549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 31), layout_222548, 'get_pixel_extents')
        # Calling get_pixel_extents(args, kwargs) (line 267)
        get_pixel_extents_call_result_222551 = invoke(stypy.reporting.localization.Localization(__file__, 267, 31), get_pixel_extents_222549, *[], **kwargs_222550)
        
        # Assigning a type to the variable 'call_assignment_221532' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'call_assignment_221532', get_pixel_extents_call_result_222551)
        
        # Assigning a Call to a Name (line 267):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222555 = {}
        # Getting the type of 'call_assignment_221532' (line 267)
        call_assignment_221532_222552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'call_assignment_221532', False)
        # Obtaining the member '__getitem__' of a type (line 267)
        getitem___222553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), call_assignment_221532_222552, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222556 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222553, *[int_222554], **kwargs_222555)
        
        # Assigning a type to the variable 'call_assignment_221533' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'call_assignment_221533', getitem___call_result_222556)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'call_assignment_221533' (line 267)
        call_assignment_221533_222557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'call_assignment_221533')
        # Assigning a type to the variable 'inkRect' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'inkRect', call_assignment_221533_222557)
        
        # Assigning a Call to a Name (line 267):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222561 = {}
        # Getting the type of 'call_assignment_221532' (line 267)
        call_assignment_221532_222558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'call_assignment_221532', False)
        # Obtaining the member '__getitem__' of a type (line 267)
        getitem___222559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), call_assignment_221532_222558, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222562 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222559, *[int_222560], **kwargs_222561)
        
        # Assigning a type to the variable 'call_assignment_221534' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'call_assignment_221534', getitem___call_result_222562)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'call_assignment_221534' (line 267)
        call_assignment_221534_222563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'call_assignment_221534')
        # Assigning a type to the variable 'logicalRect' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 17), 'logicalRect', call_assignment_221534_222563)
        
        # Assigning a Tuple to a Subscript (line 269):
        
        # Assigning a Tuple to a Subscript (line 269):
        
        # Obtaining an instance of the builtin type 'tuple' (line 269)
        tuple_222564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 269)
        # Adding element type (line 269)
        # Getting the type of 'layout' (line 269)
        layout_222565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'layout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 28), tuple_222564, layout_222565)
        # Adding element type (line 269)
        # Getting the type of 'inkRect' (line 269)
        inkRect_222566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 36), 'inkRect')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 28), tuple_222564, inkRect_222566)
        # Adding element type (line 269)
        # Getting the type of 'logicalRect' (line 269)
        logicalRect_222567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'logicalRect')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 28), tuple_222564, logicalRect_222567)
        
        # Getting the type of 'self' (line 269)
        self_222568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Obtaining the member 'layoutd' of a type (line 269)
        layoutd_222569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_222568, 'layoutd')
        # Getting the type of 'key' (line 269)
        key_222570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'key')
        # Storing an element on a container (line 269)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 8), layoutd_222569, (key_222570, tuple_222564))
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_222571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        # Adding element type (line 270)
        # Getting the type of 'layout' (line 270)
        layout_222572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'layout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 15), tuple_222571, layout_222572)
        # Adding element type (line 270)
        # Getting the type of 'inkRect' (line 270)
        inkRect_222573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 23), 'inkRect')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 15), tuple_222571, inkRect_222573)
        # Adding element type (line 270)
        # Getting the type of 'logicalRect' (line 270)
        logicalRect_222574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 32), 'logicalRect')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 15), tuple_222571, logicalRect_222574)
        
        # Assigning a type to the variable 'stypy_return_type' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'stypy_return_type', tuple_222571)
        
        # ################# End of '_get_pango_layout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_pango_layout' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_222575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_pango_layout'
        return stypy_return_type_222575


    @norecursion
    def flipy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flipy'
        module_type_store = module_type_store.open_function_context('flipy', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.flipy.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.flipy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.flipy.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.flipy.__dict__.__setitem__('stypy_function_name', 'RendererGDK.flipy')
        RendererGDK.flipy.__dict__.__setitem__('stypy_param_names_list', [])
        RendererGDK.flipy.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.flipy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.flipy.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.flipy.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.flipy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.flipy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.flipy', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'True' (line 273)
        True_222576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'stypy_return_type', True_222576)
        
        # ################# End of 'flipy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flipy' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_222577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222577)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flipy'
        return stypy_return_type_222577


    @norecursion
    def get_canvas_width_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_canvas_width_height'
        module_type_store = module_type_store.open_function_context('get_canvas_width_height', 275, 4, False)
        # Assigning a type to the variable 'self' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_function_name', 'RendererGDK.get_canvas_width_height')
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_param_names_list', [])
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.get_canvas_width_height.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.get_canvas_width_height', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'tuple' (line 276)
        tuple_222578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 276)
        # Adding element type (line 276)
        # Getting the type of 'self' (line 276)
        self_222579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'self')
        # Obtaining the member 'width' of a type (line 276)
        width_222580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 15), self_222579, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 15), tuple_222578, width_222580)
        # Adding element type (line 276)
        # Getting the type of 'self' (line 276)
        self_222581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'self')
        # Obtaining the member 'height' of a type (line 276)
        height_222582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 27), self_222581, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 15), tuple_222578, height_222582)
        
        # Assigning a type to the variable 'stypy_return_type' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'stypy_return_type', tuple_222578)
        
        # ################# End of 'get_canvas_width_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_canvas_width_height' in the type store
        # Getting the type of 'stypy_return_type' (line 275)
        stypy_return_type_222583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222583)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_canvas_width_height'
        return stypy_return_type_222583


    @norecursion
    def get_text_width_height_descent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_text_width_height_descent'
        module_type_store = module_type_store.open_function_context('get_text_width_height_descent', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_function_name', 'RendererGDK.get_text_width_height_descent')
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_param_names_list', ['s', 'prop', 'ismath'])
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.get_text_width_height_descent.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.get_text_width_height_descent', ['s', 'prop', 'ismath'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'ismath' (line 279)
        ismath_222584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'ismath')
        # Testing the type of an if condition (line 279)
        if_condition_222585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), ismath_222584)
        # Assigning a type to the variable 'if_condition_222585' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_222585', if_condition_222585)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 280):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 's' (line 281)
        s_222589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 43), 's', False)
        # Getting the type of 'self' (line 281)
        self_222590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 46), 'self', False)
        # Obtaining the member 'dpi' of a type (line 281)
        dpi_222591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 46), self_222590, 'dpi')
        # Getting the type of 'prop' (line 281)
        prop_222592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 56), 'prop', False)
        # Processing the call keyword arguments (line 281)
        kwargs_222593 = {}
        # Getting the type of 'self' (line 281)
        self_222586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'self', False)
        # Obtaining the member 'mathtext_parser' of a type (line 281)
        mathtext_parser_222587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), self_222586, 'mathtext_parser')
        # Obtaining the member 'parse' of a type (line 281)
        parse_222588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), mathtext_parser_222587, 'parse')
        # Calling parse(args, kwargs) (line 281)
        parse_call_result_222594 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), parse_222588, *[s_222589, dpi_222591, prop_222592], **kwargs_222593)
        
        # Assigning a type to the variable 'call_assignment_221535' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221535', parse_call_result_222594)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
        # Processing the call keyword arguments
        kwargs_222598 = {}
        # Getting the type of 'call_assignment_221535' (line 280)
        call_assignment_221535_222595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221535', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___222596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), call_assignment_221535_222595, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222599 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222596, *[int_222597], **kwargs_222598)
        
        # Assigning a type to the variable 'call_assignment_221536' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221536', getitem___call_result_222599)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'call_assignment_221536' (line 280)
        call_assignment_221536_222600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221536')
        # Assigning a type to the variable 'ox' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'ox', call_assignment_221536_222600)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
        # Processing the call keyword arguments
        kwargs_222604 = {}
        # Getting the type of 'call_assignment_221535' (line 280)
        call_assignment_221535_222601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221535', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___222602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), call_assignment_221535_222601, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222605 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222602, *[int_222603], **kwargs_222604)
        
        # Assigning a type to the variable 'call_assignment_221537' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221537', getitem___call_result_222605)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'call_assignment_221537' (line 280)
        call_assignment_221537_222606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221537')
        # Assigning a type to the variable 'oy' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'oy', call_assignment_221537_222606)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
        # Processing the call keyword arguments
        kwargs_222610 = {}
        # Getting the type of 'call_assignment_221535' (line 280)
        call_assignment_221535_222607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221535', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___222608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), call_assignment_221535_222607, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222611 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222608, *[int_222609], **kwargs_222610)
        
        # Assigning a type to the variable 'call_assignment_221538' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221538', getitem___call_result_222611)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'call_assignment_221538' (line 280)
        call_assignment_221538_222612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221538')
        # Assigning a type to the variable 'width' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 20), 'width', call_assignment_221538_222612)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
        # Processing the call keyword arguments
        kwargs_222616 = {}
        # Getting the type of 'call_assignment_221535' (line 280)
        call_assignment_221535_222613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221535', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___222614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), call_assignment_221535_222613, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222617 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222614, *[int_222615], **kwargs_222616)
        
        # Assigning a type to the variable 'call_assignment_221539' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221539', getitem___call_result_222617)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'call_assignment_221539' (line 280)
        call_assignment_221539_222618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221539')
        # Assigning a type to the variable 'height' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'height', call_assignment_221539_222618)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
        # Processing the call keyword arguments
        kwargs_222622 = {}
        # Getting the type of 'call_assignment_221535' (line 280)
        call_assignment_221535_222619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221535', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___222620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), call_assignment_221535_222619, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222623 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222620, *[int_222621], **kwargs_222622)
        
        # Assigning a type to the variable 'call_assignment_221540' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221540', getitem___call_result_222623)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'call_assignment_221540' (line 280)
        call_assignment_221540_222624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221540')
        # Assigning a type to the variable 'descent' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'descent', call_assignment_221540_222624)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
        # Processing the call keyword arguments
        kwargs_222628 = {}
        # Getting the type of 'call_assignment_221535' (line 280)
        call_assignment_221535_222625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221535', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___222626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), call_assignment_221535_222625, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222629 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222626, *[int_222627], **kwargs_222628)
        
        # Assigning a type to the variable 'call_assignment_221541' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221541', getitem___call_result_222629)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'call_assignment_221541' (line 280)
        call_assignment_221541_222630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221541')
        # Assigning a type to the variable 'font_image' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 44), 'font_image', call_assignment_221541_222630)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
        # Processing the call keyword arguments
        kwargs_222634 = {}
        # Getting the type of 'call_assignment_221535' (line 280)
        call_assignment_221535_222631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221535', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___222632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), call_assignment_221535_222631, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222635 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222632, *[int_222633], **kwargs_222634)
        
        # Assigning a type to the variable 'call_assignment_221542' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221542', getitem___call_result_222635)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'call_assignment_221542' (line 280)
        call_assignment_221542_222636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'call_assignment_221542')
        # Assigning a type to the variable 'used_characters' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 56), 'used_characters', call_assignment_221542_222636)
        
        # Obtaining an instance of the builtin type 'tuple' (line 282)
        tuple_222637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 282)
        # Adding element type (line 282)
        # Getting the type of 'width' (line 282)
        width_222638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 19), tuple_222637, width_222638)
        # Adding element type (line 282)
        # Getting the type of 'height' (line 282)
        height_222639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 19), tuple_222637, height_222639)
        # Adding element type (line 282)
        # Getting the type of 'descent' (line 282)
        descent_222640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 34), 'descent')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 19), tuple_222637, descent_222640)
        
        # Assigning a type to the variable 'stypy_return_type' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', tuple_222637)
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 284):
        
        # Assigning a Call to a Name:
        
        # Call to _get_pango_layout(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 's' (line 284)
        s_222643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 62), 's', False)
        # Getting the type of 'prop' (line 284)
        prop_222644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 65), 'prop', False)
        # Processing the call keyword arguments (line 284)
        kwargs_222645 = {}
        # Getting the type of 'self' (line 284)
        self_222641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 39), 'self', False)
        # Obtaining the member '_get_pango_layout' of a type (line 284)
        _get_pango_layout_222642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 39), self_222641, '_get_pango_layout')
        # Calling _get_pango_layout(args, kwargs) (line 284)
        _get_pango_layout_call_result_222646 = invoke(stypy.reporting.localization.Localization(__file__, 284, 39), _get_pango_layout_222642, *[s_222643, prop_222644], **kwargs_222645)
        
        # Assigning a type to the variable 'call_assignment_221543' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221543', _get_pango_layout_call_result_222646)
        
        # Assigning a Call to a Name (line 284):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222650 = {}
        # Getting the type of 'call_assignment_221543' (line 284)
        call_assignment_221543_222647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221543', False)
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___222648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), call_assignment_221543_222647, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222651 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222648, *[int_222649], **kwargs_222650)
        
        # Assigning a type to the variable 'call_assignment_221544' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221544', getitem___call_result_222651)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'call_assignment_221544' (line 284)
        call_assignment_221544_222652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221544')
        # Assigning a type to the variable 'layout' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'layout', call_assignment_221544_222652)
        
        # Assigning a Call to a Name (line 284):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222656 = {}
        # Getting the type of 'call_assignment_221543' (line 284)
        call_assignment_221543_222653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221543', False)
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___222654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), call_assignment_221543_222653, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222657 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222654, *[int_222655], **kwargs_222656)
        
        # Assigning a type to the variable 'call_assignment_221545' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221545', getitem___call_result_222657)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'call_assignment_221545' (line 284)
        call_assignment_221545_222658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221545')
        # Assigning a type to the variable 'inkRect' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'inkRect', call_assignment_221545_222658)
        
        # Assigning a Call to a Name (line 284):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_222661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 8), 'int')
        # Processing the call keyword arguments
        kwargs_222662 = {}
        # Getting the type of 'call_assignment_221543' (line 284)
        call_assignment_221543_222659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221543', False)
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___222660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), call_assignment_221543_222659, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_222663 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___222660, *[int_222661], **kwargs_222662)
        
        # Assigning a type to the variable 'call_assignment_221546' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221546', getitem___call_result_222663)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'call_assignment_221546' (line 284)
        call_assignment_221546_222664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'call_assignment_221546')
        # Assigning a type to the variable 'logicalRect' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'logicalRect', call_assignment_221546_222664)
        
        # Assigning a Name to a Tuple (line 285):
        
        # Assigning a Subscript to a Name (line 285):
        
        # Obtaining the type of the subscript
        int_222665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 8), 'int')
        # Getting the type of 'inkRect' (line 285)
        inkRect_222666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___222667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), inkRect_222666, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_222668 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), getitem___222667, int_222665)
        
        # Assigning a type to the variable 'tuple_var_assignment_221547' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_221547', subscript_call_result_222668)
        
        # Assigning a Subscript to a Name (line 285):
        
        # Obtaining the type of the subscript
        int_222669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 8), 'int')
        # Getting the type of 'inkRect' (line 285)
        inkRect_222670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___222671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), inkRect_222670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_222672 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), getitem___222671, int_222669)
        
        # Assigning a type to the variable 'tuple_var_assignment_221548' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_221548', subscript_call_result_222672)
        
        # Assigning a Subscript to a Name (line 285):
        
        # Obtaining the type of the subscript
        int_222673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 8), 'int')
        # Getting the type of 'inkRect' (line 285)
        inkRect_222674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___222675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), inkRect_222674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_222676 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), getitem___222675, int_222673)
        
        # Assigning a type to the variable 'tuple_var_assignment_221549' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_221549', subscript_call_result_222676)
        
        # Assigning a Subscript to a Name (line 285):
        
        # Obtaining the type of the subscript
        int_222677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 8), 'int')
        # Getting the type of 'inkRect' (line 285)
        inkRect_222678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'inkRect')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___222679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), inkRect_222678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_222680 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), getitem___222679, int_222677)
        
        # Assigning a type to the variable 'tuple_var_assignment_221550' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_221550', subscript_call_result_222680)
        
        # Assigning a Name to a Name (line 285):
        # Getting the type of 'tuple_var_assignment_221547' (line 285)
        tuple_var_assignment_221547_222681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_221547')
        # Assigning a type to the variable 'l' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'l', tuple_var_assignment_221547_222681)
        
        # Assigning a Name to a Name (line 285):
        # Getting the type of 'tuple_var_assignment_221548' (line 285)
        tuple_var_assignment_221548_222682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_221548')
        # Assigning a type to the variable 'b' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'b', tuple_var_assignment_221548_222682)
        
        # Assigning a Name to a Name (line 285):
        # Getting the type of 'tuple_var_assignment_221549' (line 285)
        tuple_var_assignment_221549_222683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_221549')
        # Assigning a type to the variable 'w' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 14), 'w', tuple_var_assignment_221549_222683)
        
        # Assigning a Name to a Name (line 285):
        # Getting the type of 'tuple_var_assignment_221550' (line 285)
        tuple_var_assignment_221550_222684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_221550')
        # Assigning a type to the variable 'h' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 17), 'h', tuple_var_assignment_221550_222684)
        
        # Assigning a Name to a Tuple (line 286):
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_222685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        # Getting the type of 'logicalRect' (line 286)
        logicalRect_222686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 'logicalRect')
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___222687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), logicalRect_222686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_222688 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___222687, int_222685)
        
        # Assigning a type to the variable 'tuple_var_assignment_221551' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_221551', subscript_call_result_222688)
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_222689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        # Getting the type of 'logicalRect' (line 286)
        logicalRect_222690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 'logicalRect')
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___222691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), logicalRect_222690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_222692 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___222691, int_222689)
        
        # Assigning a type to the variable 'tuple_var_assignment_221552' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_221552', subscript_call_result_222692)
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_222693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        # Getting the type of 'logicalRect' (line 286)
        logicalRect_222694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 'logicalRect')
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___222695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), logicalRect_222694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_222696 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___222695, int_222693)
        
        # Assigning a type to the variable 'tuple_var_assignment_221553' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_221553', subscript_call_result_222696)
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_222697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        # Getting the type of 'logicalRect' (line 286)
        logicalRect_222698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 'logicalRect')
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___222699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), logicalRect_222698, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_222700 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___222699, int_222697)
        
        # Assigning a type to the variable 'tuple_var_assignment_221554' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_221554', subscript_call_result_222700)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_221551' (line 286)
        tuple_var_assignment_221551_222701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_221551')
        # Assigning a type to the variable 'll' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'll', tuple_var_assignment_221551_222701)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_221552' (line 286)
        tuple_var_assignment_221552_222702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_221552')
        # Assigning a type to the variable 'lb' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'lb', tuple_var_assignment_221552_222702)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_221553' (line 286)
        tuple_var_assignment_221553_222703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_221553')
        # Assigning a type to the variable 'lw' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'lw', tuple_var_assignment_221553_222703)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_221554' (line 286)
        tuple_var_assignment_221554_222704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_221554')
        # Assigning a type to the variable 'lh' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'lh', tuple_var_assignment_221554_222704)
        
        # Obtaining an instance of the builtin type 'tuple' (line 288)
        tuple_222705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 288)
        # Adding element type (line 288)
        # Getting the type of 'w' (line 288)
        w_222706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 15), tuple_222705, w_222706)
        # Adding element type (line 288)
        # Getting the type of 'h' (line 288)
        h_222707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'h')
        int_222708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 22), 'int')
        # Applying the binary operator '+' (line 288)
        result_add_222709 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 18), '+', h_222707, int_222708)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 15), tuple_222705, result_add_222709)
        # Adding element type (line 288)
        # Getting the type of 'h' (line 288)
        h_222710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'h')
        # Getting the type of 'lh' (line 288)
        lh_222711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 29), 'lh')
        # Applying the binary operator '-' (line 288)
        result_sub_222712 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 25), '-', h_222710, lh_222711)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 15), tuple_222705, result_sub_222712)
        
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', tuple_222705)
        
        # ################# End of 'get_text_width_height_descent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text_width_height_descent' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_222713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text_width_height_descent'
        return stypy_return_type_222713


    @norecursion
    def new_gc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_gc'
        module_type_store = module_type_store.open_function_context('new_gc', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.new_gc.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.new_gc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.new_gc.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.new_gc.__dict__.__setitem__('stypy_function_name', 'RendererGDK.new_gc')
        RendererGDK.new_gc.__dict__.__setitem__('stypy_param_names_list', [])
        RendererGDK.new_gc.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.new_gc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.new_gc.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.new_gc.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.new_gc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.new_gc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.new_gc', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to GraphicsContextGDK(...): (line 291)
        # Processing the call keyword arguments (line 291)
        # Getting the type of 'self' (line 291)
        self_222715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 43), 'self', False)
        keyword_222716 = self_222715
        kwargs_222717 = {'renderer': keyword_222716}
        # Getting the type of 'GraphicsContextGDK' (line 291)
        GraphicsContextGDK_222714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'GraphicsContextGDK', False)
        # Calling GraphicsContextGDK(args, kwargs) (line 291)
        GraphicsContextGDK_call_result_222718 = invoke(stypy.reporting.localization.Localization(__file__, 291, 15), GraphicsContextGDK_222714, *[], **kwargs_222717)
        
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', GraphicsContextGDK_call_result_222718)
        
        # ################# End of 'new_gc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_gc' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_222719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222719)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_gc'
        return stypy_return_type_222719


    @norecursion
    def points_to_pixels(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'points_to_pixels'
        module_type_store = module_type_store.open_function_context('points_to_pixels', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_localization', localization)
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_function_name', 'RendererGDK.points_to_pixels')
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_param_names_list', ['points'])
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGDK.points_to_pixels.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGDK.points_to_pixels', ['points'], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'points' (line 294)
        points_222720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'points')
        float_222721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'float')
        # Applying the binary operator 'div' (line 294)
        result_div_222722 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), 'div', points_222720, float_222721)
        
        # Getting the type of 'self' (line 294)
        self_222723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 29), 'self')
        # Obtaining the member 'dpi' of a type (line 294)
        dpi_222724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 29), self_222723, 'dpi')
        # Applying the binary operator '*' (line 294)
        result_mul_222725 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 27), '*', result_div_222722, dpi_222724)
        
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', result_mul_222725)
        
        # ################# End of 'points_to_pixels(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'points_to_pixels' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_222726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'points_to_pixels'
        return stypy_return_type_222726


# Assigning a type to the variable 'RendererGDK' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'RendererGDK', RendererGDK)

# Assigning a Dict to a Name (line 43):

# Obtaining an instance of the builtin type 'dict' (line 43)
dict_222727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 43)
# Adding element type (key, value) (line 43)
int_222728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'int')
# Getting the type of 'pango' (line 44)
pango_222729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'pango')
# Obtaining the member 'WEIGHT_ULTRALIGHT' of a type (line 44)
WEIGHT_ULTRALIGHT_222730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), pango_222729, 'WEIGHT_ULTRALIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222728, WEIGHT_ULTRALIGHT_222730))
# Adding element type (key, value) (line 43)
int_222731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'int')
# Getting the type of 'pango' (line 45)
pango_222732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'pango')
# Obtaining the member 'WEIGHT_LIGHT' of a type (line 45)
WEIGHT_LIGHT_222733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 23), pango_222732, 'WEIGHT_LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222731, WEIGHT_LIGHT_222733))
# Adding element type (key, value) (line 43)
int_222734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'int')
# Getting the type of 'pango' (line 46)
pango_222735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'pango')
# Obtaining the member 'WEIGHT_LIGHT' of a type (line 46)
WEIGHT_LIGHT_222736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 23), pango_222735, 'WEIGHT_LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222734, WEIGHT_LIGHT_222736))
# Adding element type (key, value) (line 43)
int_222737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'int')
# Getting the type of 'pango' (line 47)
pango_222738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'pango')
# Obtaining the member 'WEIGHT_NORMAL' of a type (line 47)
WEIGHT_NORMAL_222739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 23), pango_222738, 'WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222737, WEIGHT_NORMAL_222739))
# Adding element type (key, value) (line 43)
int_222740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
# Getting the type of 'pango' (line 48)
pango_222741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'pango')
# Obtaining the member 'WEIGHT_NORMAL' of a type (line 48)
WEIGHT_NORMAL_222742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 23), pango_222741, 'WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222740, WEIGHT_NORMAL_222742))
# Adding element type (key, value) (line 43)
int_222743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
# Getting the type of 'pango' (line 49)
pango_222744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'pango')
# Obtaining the member 'WEIGHT_BOLD' of a type (line 49)
WEIGHT_BOLD_222745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 23), pango_222744, 'WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222743, WEIGHT_BOLD_222745))
# Adding element type (key, value) (line 43)
int_222746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 8), 'int')
# Getting the type of 'pango' (line 50)
pango_222747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), 'pango')
# Obtaining the member 'WEIGHT_BOLD' of a type (line 50)
WEIGHT_BOLD_222748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 23), pango_222747, 'WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222746, WEIGHT_BOLD_222748))
# Adding element type (key, value) (line 43)
int_222749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'int')
# Getting the type of 'pango' (line 51)
pango_222750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'pango')
# Obtaining the member 'WEIGHT_HEAVY' of a type (line 51)
WEIGHT_HEAVY_222751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 23), pango_222750, 'WEIGHT_HEAVY')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222749, WEIGHT_HEAVY_222751))
# Adding element type (key, value) (line 43)
int_222752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'int')
# Getting the type of 'pango' (line 52)
pango_222753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'pango')
# Obtaining the member 'WEIGHT_ULTRABOLD' of a type (line 52)
WEIGHT_ULTRABOLD_222754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 23), pango_222753, 'WEIGHT_ULTRABOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (int_222752, WEIGHT_ULTRABOLD_222754))
# Adding element type (key, value) (line 43)
unicode_222755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'unicode', u'ultralight')
# Getting the type of 'pango' (line 53)
pango_222756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'pango')
# Obtaining the member 'WEIGHT_ULTRALIGHT' of a type (line 53)
WEIGHT_ULTRALIGHT_222757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 23), pango_222756, 'WEIGHT_ULTRALIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222755, WEIGHT_ULTRALIGHT_222757))
# Adding element type (key, value) (line 43)
unicode_222758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'unicode', u'light')
# Getting the type of 'pango' (line 54)
pango_222759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'pango')
# Obtaining the member 'WEIGHT_LIGHT' of a type (line 54)
WEIGHT_LIGHT_222760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 23), pango_222759, 'WEIGHT_LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222758, WEIGHT_LIGHT_222760))
# Adding element type (key, value) (line 43)
unicode_222761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'unicode', u'normal')
# Getting the type of 'pango' (line 55)
pango_222762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'pango')
# Obtaining the member 'WEIGHT_NORMAL' of a type (line 55)
WEIGHT_NORMAL_222763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 23), pango_222762, 'WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222761, WEIGHT_NORMAL_222763))
# Adding element type (key, value) (line 43)
unicode_222764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'unicode', u'medium')
# Getting the type of 'pango' (line 56)
pango_222765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'pango')
# Obtaining the member 'WEIGHT_NORMAL' of a type (line 56)
WEIGHT_NORMAL_222766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 23), pango_222765, 'WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222764, WEIGHT_NORMAL_222766))
# Adding element type (key, value) (line 43)
unicode_222767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'unicode', u'semibold')
# Getting the type of 'pango' (line 57)
pango_222768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'pango')
# Obtaining the member 'WEIGHT_BOLD' of a type (line 57)
WEIGHT_BOLD_222769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), pango_222768, 'WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222767, WEIGHT_BOLD_222769))
# Adding element type (key, value) (line 43)
unicode_222770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'unicode', u'bold')
# Getting the type of 'pango' (line 58)
pango_222771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'pango')
# Obtaining the member 'WEIGHT_BOLD' of a type (line 58)
WEIGHT_BOLD_222772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 23), pango_222771, 'WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222770, WEIGHT_BOLD_222772))
# Adding element type (key, value) (line 43)
unicode_222773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'unicode', u'heavy')
# Getting the type of 'pango' (line 59)
pango_222774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'pango')
# Obtaining the member 'WEIGHT_HEAVY' of a type (line 59)
WEIGHT_HEAVY_222775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), pango_222774, 'WEIGHT_HEAVY')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222773, WEIGHT_HEAVY_222775))
# Adding element type (key, value) (line 43)
unicode_222776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'unicode', u'ultrabold')
# Getting the type of 'pango' (line 60)
pango_222777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'pango')
# Obtaining the member 'WEIGHT_ULTRABOLD' of a type (line 60)
WEIGHT_ULTRABOLD_222778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), pango_222777, 'WEIGHT_ULTRABOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222776, WEIGHT_ULTRABOLD_222778))
# Adding element type (key, value) (line 43)
unicode_222779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'unicode', u'black')
# Getting the type of 'pango' (line 61)
pango_222780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'pango')
# Obtaining the member 'WEIGHT_ULTRABOLD' of a type (line 61)
WEIGHT_ULTRABOLD_222781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 23), pango_222780, 'WEIGHT_ULTRABOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), dict_222727, (unicode_222779, WEIGHT_ULTRABOLD_222781))

# Getting the type of 'RendererGDK'
RendererGDK_222782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RendererGDK')
# Setting the type of the member 'fontweights' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RendererGDK_222782, 'fontweights', dict_222727)

# Assigning a Dict to a Name (line 65):

# Obtaining an instance of the builtin type 'dict' (line 65)
dict_222783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 65)

# Getting the type of 'RendererGDK'
RendererGDK_222784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RendererGDK')
# Setting the type of the member 'layoutd' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RendererGDK_222784, 'layoutd', dict_222783)

# Assigning a Dict to a Name (line 66):

# Obtaining an instance of the builtin type 'dict' (line 66)
dict_222785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 66)

# Getting the type of 'RendererGDK'
RendererGDK_222786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RendererGDK')
# Setting the type of the member 'rotated' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RendererGDK_222786, 'rotated', dict_222785)
# Declaration of the 'GraphicsContextGDK' class
# Getting the type of 'GraphicsContextBase' (line 297)
GraphicsContextBase_222787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'GraphicsContextBase')

class GraphicsContextGDK(GraphicsContextBase_222787, ):
    
    # Assigning a Dict to a Name (line 299):
    
    # Assigning a Dict to a Name (line 301):
    
    # Assigning a Dict to a Name (line 307):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextGDK.__init__', ['renderer'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'self' (line 315)
        self_222790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 37), 'self', False)
        # Processing the call keyword arguments (line 315)
        kwargs_222791 = {}
        # Getting the type of 'GraphicsContextBase' (line 315)
        GraphicsContextBase_222788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'GraphicsContextBase', False)
        # Obtaining the member '__init__' of a type (line 315)
        init___222789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), GraphicsContextBase_222788, '__init__')
        # Calling __init__(args, kwargs) (line 315)
        init___call_result_222792 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), init___222789, *[self_222790], **kwargs_222791)
        
        
        # Assigning a Name to a Attribute (line 316):
        
        # Assigning a Name to a Attribute (line 316):
        # Getting the type of 'renderer' (line 316)
        renderer_222793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 24), 'renderer')
        # Getting the type of 'self' (line 316)
        self_222794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self')
        # Setting the type of the member 'renderer' of a type (line 316)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_222794, 'renderer', renderer_222793)
        
        # Assigning a Call to a Attribute (line 317):
        
        # Assigning a Call to a Attribute (line 317):
        
        # Call to GC(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'renderer' (line 317)
        renderer_222798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 35), 'renderer', False)
        # Obtaining the member 'gdkDrawable' of a type (line 317)
        gdkDrawable_222799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 35), renderer_222798, 'gdkDrawable')
        # Processing the call keyword arguments (line 317)
        kwargs_222800 = {}
        # Getting the type of 'gtk' (line 317)
        gtk_222795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 24), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 317)
        gdk_222796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 24), gtk_222795, 'gdk')
        # Obtaining the member 'GC' of a type (line 317)
        GC_222797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 24), gdk_222796, 'GC')
        # Calling GC(args, kwargs) (line 317)
        GC_call_result_222801 = invoke(stypy.reporting.localization.Localization(__file__, 317, 24), GC_222797, *[gdkDrawable_222799], **kwargs_222800)
        
        # Getting the type of 'self' (line 317)
        self_222802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self')
        # Setting the type of the member 'gdkGC' of a type (line 317)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_222802, 'gdkGC', GC_call_result_222801)
        
        # Assigning a Attribute to a Attribute (line 318):
        
        # Assigning a Attribute to a Attribute (line 318):
        # Getting the type of 'renderer' (line 318)
        renderer_222803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'renderer')
        # Obtaining the member '_cmap' of a type (line 318)
        _cmap_222804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 24), renderer_222803, '_cmap')
        # Getting the type of 'self' (line 318)
        self_222805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'self')
        # Setting the type of the member '_cmap' of a type (line 318)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), self_222805, '_cmap', _cmap_222804)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def rgb_to_gdk_color(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rgb_to_gdk_color'
        module_type_store = module_type_store.open_function_context('rgb_to_gdk_color', 321, 4, False)
        # Assigning a type to the variable 'self' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_function_name', 'GraphicsContextGDK.rgb_to_gdk_color')
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_param_names_list', ['rgb'])
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextGDK.rgb_to_gdk_color.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextGDK.rgb_to_gdk_color', ['rgb'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rgb_to_gdk_color', localization, ['rgb'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rgb_to_gdk_color(...)' code ##################

        unicode_222806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, (-1)), 'unicode', u'\n        rgb - an RGB tuple (three 0.0-1.0 values)\n        return an allocated gtk.gdk.Color\n        ')
        
        
        # SSA begins for try-except statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        
        # Call to tuple(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'rgb' (line 327)
        rgb_222808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 38), 'rgb', False)
        # Processing the call keyword arguments (line 327)
        kwargs_222809 = {}
        # Getting the type of 'tuple' (line 327)
        tuple_222807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), 'tuple', False)
        # Calling tuple(args, kwargs) (line 327)
        tuple_call_result_222810 = invoke(stypy.reporting.localization.Localization(__file__, 327, 32), tuple_222807, *[rgb_222808], **kwargs_222809)
        
        # Getting the type of 'self' (line 327)
        self_222811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'self')
        # Obtaining the member '_cached' of a type (line 327)
        _cached_222812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 19), self_222811, '_cached')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___222813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 19), _cached_222812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_222814 = invoke(stypy.reporting.localization.Localization(__file__, 327, 19), getitem___222813, tuple_call_result_222810)
        
        # Assigning a type to the variable 'stypy_return_type' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'stypy_return_type', subscript_call_result_222814)
        # SSA branch for the except part of a try statement (line 326)
        # SSA branch for the except 'KeyError' branch of a try statement (line 326)
        module_type_store.open_ssa_branch('except')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Subscript (line 329):
        
        # Call to alloc_color(...): (line 330)
        # Processing the call arguments (line 330)
        
        # Call to int(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Obtaining the type of the subscript
        int_222819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 32), 'int')
        # Getting the type of 'rgb' (line 331)
        rgb_222820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___222821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 28), rgb_222820, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 331)
        subscript_call_result_222822 = invoke(stypy.reporting.localization.Localization(__file__, 331, 28), getitem___222821, int_222819)
        
        int_222823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 35), 'int')
        # Applying the binary operator '*' (line 331)
        result_mul_222824 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 28), '*', subscript_call_result_222822, int_222823)
        
        # Processing the call keyword arguments (line 331)
        kwargs_222825 = {}
        # Getting the type of 'int' (line 331)
        int_222818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'int', False)
        # Calling int(args, kwargs) (line 331)
        int_call_result_222826 = invoke(stypy.reporting.localization.Localization(__file__, 331, 24), int_222818, *[result_mul_222824], **kwargs_222825)
        
        
        # Call to int(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Obtaining the type of the subscript
        int_222828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 50), 'int')
        # Getting the type of 'rgb' (line 331)
        rgb_222829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 46), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___222830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 46), rgb_222829, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 331)
        subscript_call_result_222831 = invoke(stypy.reporting.localization.Localization(__file__, 331, 46), getitem___222830, int_222828)
        
        int_222832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 53), 'int')
        # Applying the binary operator '*' (line 331)
        result_mul_222833 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 46), '*', subscript_call_result_222831, int_222832)
        
        # Processing the call keyword arguments (line 331)
        kwargs_222834 = {}
        # Getting the type of 'int' (line 331)
        int_222827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 42), 'int', False)
        # Calling int(args, kwargs) (line 331)
        int_call_result_222835 = invoke(stypy.reporting.localization.Localization(__file__, 331, 42), int_222827, *[result_mul_222833], **kwargs_222834)
        
        
        # Call to int(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Obtaining the type of the subscript
        int_222837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 68), 'int')
        # Getting the type of 'rgb' (line 331)
        rgb_222838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 64), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___222839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 64), rgb_222838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 331)
        subscript_call_result_222840 = invoke(stypy.reporting.localization.Localization(__file__, 331, 64), getitem___222839, int_222837)
        
        int_222841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 71), 'int')
        # Applying the binary operator '*' (line 331)
        result_mul_222842 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 64), '*', subscript_call_result_222840, int_222841)
        
        # Processing the call keyword arguments (line 331)
        kwargs_222843 = {}
        # Getting the type of 'int' (line 331)
        int_222836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 60), 'int', False)
        # Calling int(args, kwargs) (line 331)
        int_call_result_222844 = invoke(stypy.reporting.localization.Localization(__file__, 331, 60), int_222836, *[result_mul_222842], **kwargs_222843)
        
        # Processing the call keyword arguments (line 330)
        kwargs_222845 = {}
        # Getting the type of 'self' (line 330)
        self_222815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 'self', False)
        # Obtaining the member '_cmap' of a type (line 330)
        _cmap_222816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 20), self_222815, '_cmap')
        # Obtaining the member 'alloc_color' of a type (line 330)
        alloc_color_222817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 20), _cmap_222816, 'alloc_color')
        # Calling alloc_color(args, kwargs) (line 330)
        alloc_color_call_result_222846 = invoke(stypy.reporting.localization.Localization(__file__, 330, 20), alloc_color_222817, *[int_call_result_222826, int_call_result_222835, int_call_result_222844], **kwargs_222845)
        
        # Getting the type of 'self' (line 329)
        self_222847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'self')
        # Obtaining the member '_cached' of a type (line 329)
        _cached_222848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 20), self_222847, '_cached')
        
        # Call to tuple(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'rgb' (line 329)
        rgb_222850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 39), 'rgb', False)
        # Processing the call keyword arguments (line 329)
        kwargs_222851 = {}
        # Getting the type of 'tuple' (line 329)
        tuple_222849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'tuple', False)
        # Calling tuple(args, kwargs) (line 329)
        tuple_call_result_222852 = invoke(stypy.reporting.localization.Localization(__file__, 329, 33), tuple_222849, *[rgb_222850], **kwargs_222851)
        
        # Storing an element on a container (line 329)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 20), _cached_222848, (tuple_call_result_222852, alloc_color_call_result_222846))
        
        # Assigning a Subscript to a Name (line 329):
        
        # Obtaining the type of the subscript
        
        # Call to tuple(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'rgb' (line 329)
        rgb_222854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 39), 'rgb', False)
        # Processing the call keyword arguments (line 329)
        kwargs_222855 = {}
        # Getting the type of 'tuple' (line 329)
        tuple_222853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'tuple', False)
        # Calling tuple(args, kwargs) (line 329)
        tuple_call_result_222856 = invoke(stypy.reporting.localization.Localization(__file__, 329, 33), tuple_222853, *[rgb_222854], **kwargs_222855)
        
        # Getting the type of 'self' (line 329)
        self_222857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'self')
        # Obtaining the member '_cached' of a type (line 329)
        _cached_222858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 20), self_222857, '_cached')
        # Obtaining the member '__getitem__' of a type (line 329)
        getitem___222859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 20), _cached_222858, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 329)
        subscript_call_result_222860 = invoke(stypy.reporting.localization.Localization(__file__, 329, 20), getitem___222859, tuple_call_result_222856)
        
        # Assigning a type to the variable 'color' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'color', subscript_call_result_222860)
        # Getting the type of 'color' (line 332)
        color_222861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'color')
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', color_222861)
        # SSA join for try-except statement (line 326)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'rgb_to_gdk_color(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rgb_to_gdk_color' in the type store
        # Getting the type of 'stypy_return_type' (line 321)
        stypy_return_type_222862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222862)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rgb_to_gdk_color'
        return stypy_return_type_222862


    @norecursion
    def set_capstyle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_capstyle'
        module_type_store = module_type_store.open_function_context('set_capstyle', 338, 4, False)
        # Assigning a type to the variable 'self' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_function_name', 'GraphicsContextGDK.set_capstyle')
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_param_names_list', ['cs'])
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextGDK.set_capstyle.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextGDK.set_capstyle', ['cs'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_capstyle(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'self' (line 339)
        self_222865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 41), 'self', False)
        # Getting the type of 'cs' (line 339)
        cs_222866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 47), 'cs', False)
        # Processing the call keyword arguments (line 339)
        kwargs_222867 = {}
        # Getting the type of 'GraphicsContextBase' (line 339)
        GraphicsContextBase_222863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'GraphicsContextBase', False)
        # Obtaining the member 'set_capstyle' of a type (line 339)
        set_capstyle_222864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), GraphicsContextBase_222863, 'set_capstyle')
        # Calling set_capstyle(args, kwargs) (line 339)
        set_capstyle_call_result_222868 = invoke(stypy.reporting.localization.Localization(__file__, 339, 8), set_capstyle_222864, *[self_222865, cs_222866], **kwargs_222867)
        
        
        # Assigning a Subscript to a Attribute (line 340):
        
        # Assigning a Subscript to a Attribute (line 340):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 340)
        self_222869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 42), 'self')
        # Obtaining the member '_capstyle' of a type (line 340)
        _capstyle_222870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 42), self_222869, '_capstyle')
        # Getting the type of 'self' (line 340)
        self_222871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 31), 'self')
        # Obtaining the member '_capd' of a type (line 340)
        _capd_222872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 31), self_222871, '_capd')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___222873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 31), _capd_222872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_222874 = invoke(stypy.reporting.localization.Localization(__file__, 340, 31), getitem___222873, _capstyle_222870)
        
        # Getting the type of 'self' (line 340)
        self_222875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self')
        # Obtaining the member 'gdkGC' of a type (line 340)
        gdkGC_222876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_222875, 'gdkGC')
        # Setting the type of the member 'cap_style' of a type (line 340)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), gdkGC_222876, 'cap_style', subscript_call_result_222874)
        
        # ################# End of 'set_capstyle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_capstyle' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_222877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_capstyle'
        return stypy_return_type_222877


    @norecursion
    def set_clip_rectangle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_clip_rectangle'
        module_type_store = module_type_store.open_function_context('set_clip_rectangle', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_function_name', 'GraphicsContextGDK.set_clip_rectangle')
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_param_names_list', ['rectangle'])
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextGDK.set_clip_rectangle.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextGDK.set_clip_rectangle', ['rectangle'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_clip_rectangle(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'self' (line 344)
        self_222880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 47), 'self', False)
        # Getting the type of 'rectangle' (line 344)
        rectangle_222881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 53), 'rectangle', False)
        # Processing the call keyword arguments (line 344)
        kwargs_222882 = {}
        # Getting the type of 'GraphicsContextBase' (line 344)
        GraphicsContextBase_222878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'GraphicsContextBase', False)
        # Obtaining the member 'set_clip_rectangle' of a type (line 344)
        set_clip_rectangle_222879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), GraphicsContextBase_222878, 'set_clip_rectangle')
        # Calling set_clip_rectangle(args, kwargs) (line 344)
        set_clip_rectangle_call_result_222883 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), set_clip_rectangle_222879, *[self_222880, rectangle_222881], **kwargs_222882)
        
        
        # Type idiom detected: calculating its left and rigth part (line 345)
        # Getting the type of 'rectangle' (line 345)
        rectangle_222884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 11), 'rectangle')
        # Getting the type of 'None' (line 345)
        None_222885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'None')
        
        (may_be_222886, more_types_in_union_222887) = may_be_none(rectangle_222884, None_222885)

        if may_be_222886:

            if more_types_in_union_222887:
                # Runtime conditional SSA (line 345)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 346)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_222887:
                # SSA join for if statement (line 345)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 347):
        
        # Assigning a Subscript to a Name (line 347):
        
        # Obtaining the type of the subscript
        int_222888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 8), 'int')
        # Getting the type of 'rectangle' (line 347)
        rectangle_222889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'rectangle')
        # Obtaining the member 'bounds' of a type (line 347)
        bounds_222890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 18), rectangle_222889, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___222891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), bounds_222890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_222892 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), getitem___222891, int_222888)
        
        # Assigning a type to the variable 'tuple_var_assignment_221555' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'tuple_var_assignment_221555', subscript_call_result_222892)
        
        # Assigning a Subscript to a Name (line 347):
        
        # Obtaining the type of the subscript
        int_222893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 8), 'int')
        # Getting the type of 'rectangle' (line 347)
        rectangle_222894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'rectangle')
        # Obtaining the member 'bounds' of a type (line 347)
        bounds_222895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 18), rectangle_222894, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___222896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), bounds_222895, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_222897 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), getitem___222896, int_222893)
        
        # Assigning a type to the variable 'tuple_var_assignment_221556' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'tuple_var_assignment_221556', subscript_call_result_222897)
        
        # Assigning a Subscript to a Name (line 347):
        
        # Obtaining the type of the subscript
        int_222898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 8), 'int')
        # Getting the type of 'rectangle' (line 347)
        rectangle_222899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'rectangle')
        # Obtaining the member 'bounds' of a type (line 347)
        bounds_222900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 18), rectangle_222899, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___222901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), bounds_222900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_222902 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), getitem___222901, int_222898)
        
        # Assigning a type to the variable 'tuple_var_assignment_221557' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'tuple_var_assignment_221557', subscript_call_result_222902)
        
        # Assigning a Subscript to a Name (line 347):
        
        # Obtaining the type of the subscript
        int_222903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 8), 'int')
        # Getting the type of 'rectangle' (line 347)
        rectangle_222904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'rectangle')
        # Obtaining the member 'bounds' of a type (line 347)
        bounds_222905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 18), rectangle_222904, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___222906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), bounds_222905, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_222907 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), getitem___222906, int_222903)
        
        # Assigning a type to the variable 'tuple_var_assignment_221558' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'tuple_var_assignment_221558', subscript_call_result_222907)
        
        # Assigning a Name to a Name (line 347):
        # Getting the type of 'tuple_var_assignment_221555' (line 347)
        tuple_var_assignment_221555_222908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'tuple_var_assignment_221555')
        # Assigning a type to the variable 'l' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'l', tuple_var_assignment_221555_222908)
        
        # Assigning a Name to a Name (line 347):
        # Getting the type of 'tuple_var_assignment_221556' (line 347)
        tuple_var_assignment_221556_222909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'tuple_var_assignment_221556')
        # Assigning a type to the variable 'b' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 10), 'b', tuple_var_assignment_221556_222909)
        
        # Assigning a Name to a Name (line 347):
        # Getting the type of 'tuple_var_assignment_221557' (line 347)
        tuple_var_assignment_221557_222910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'tuple_var_assignment_221557')
        # Assigning a type to the variable 'w' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'w', tuple_var_assignment_221557_222910)
        
        # Assigning a Name to a Name (line 347):
        # Getting the type of 'tuple_var_assignment_221558' (line 347)
        tuple_var_assignment_221558_222911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'tuple_var_assignment_221558')
        # Assigning a type to the variable 'h' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 14), 'h', tuple_var_assignment_221558_222911)
        
        # Assigning a Tuple to a Name (line 348):
        
        # Assigning a Tuple to a Name (line 348):
        
        # Obtaining an instance of the builtin type 'tuple' (line 348)
        tuple_222912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 348)
        # Adding element type (line 348)
        
        # Call to int(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'l' (line 348)
        l_222914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 25), 'l', False)
        # Processing the call keyword arguments (line 348)
        kwargs_222915 = {}
        # Getting the type of 'int' (line 348)
        int_222913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 21), 'int', False)
        # Calling int(args, kwargs) (line 348)
        int_call_result_222916 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), int_222913, *[l_222914], **kwargs_222915)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 21), tuple_222912, int_call_result_222916)
        # Adding element type (line 348)
        # Getting the type of 'self' (line 348)
        self_222917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 29), 'self')
        # Obtaining the member 'renderer' of a type (line 348)
        renderer_222918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 29), self_222917, 'renderer')
        # Obtaining the member 'height' of a type (line 348)
        height_222919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 29), renderer_222918, 'height')
        
        # Call to int(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'b' (line 348)
        b_222921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 54), 'b', False)
        # Getting the type of 'h' (line 348)
        h_222922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 56), 'h', False)
        # Applying the binary operator '+' (line 348)
        result_add_222923 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 54), '+', b_222921, h_222922)
        
        # Processing the call keyword arguments (line 348)
        kwargs_222924 = {}
        # Getting the type of 'int' (line 348)
        int_222920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 50), 'int', False)
        # Calling int(args, kwargs) (line 348)
        int_call_result_222925 = invoke(stypy.reporting.localization.Localization(__file__, 348, 50), int_222920, *[result_add_222923], **kwargs_222924)
        
        # Applying the binary operator '-' (line 348)
        result_sub_222926 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 29), '-', height_222919, int_call_result_222925)
        
        int_222927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 59), 'int')
        # Applying the binary operator '+' (line 348)
        result_add_222928 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 58), '+', result_sub_222926, int_222927)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 21), tuple_222912, result_add_222928)
        # Adding element type (line 348)
        
        # Call to int(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'w' (line 349)
        w_222930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 25), 'w', False)
        # Processing the call keyword arguments (line 349)
        kwargs_222931 = {}
        # Getting the type of 'int' (line 349)
        int_222929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'int', False)
        # Calling int(args, kwargs) (line 349)
        int_call_result_222932 = invoke(stypy.reporting.localization.Localization(__file__, 349, 21), int_222929, *[w_222930], **kwargs_222931)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 21), tuple_222912, int_call_result_222932)
        # Adding element type (line 348)
        
        # Call to int(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'h' (line 349)
        h_222934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 33), 'h', False)
        # Processing the call keyword arguments (line 349)
        kwargs_222935 = {}
        # Getting the type of 'int' (line 349)
        int_222933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'int', False)
        # Calling int(args, kwargs) (line 349)
        int_call_result_222936 = invoke(stypy.reporting.localization.Localization(__file__, 349, 29), int_222933, *[h_222934], **kwargs_222935)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 21), tuple_222912, int_call_result_222936)
        
        # Assigning a type to the variable 'rectangle' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'rectangle', tuple_222912)
        
        # Call to set_clip_rectangle(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'rectangle' (line 352)
        rectangle_222940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 38), 'rectangle', False)
        # Processing the call keyword arguments (line 352)
        kwargs_222941 = {}
        # Getting the type of 'self' (line 352)
        self_222937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'self', False)
        # Obtaining the member 'gdkGC' of a type (line 352)
        gdkGC_222938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), self_222937, 'gdkGC')
        # Obtaining the member 'set_clip_rectangle' of a type (line 352)
        set_clip_rectangle_222939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), gdkGC_222938, 'set_clip_rectangle')
        # Calling set_clip_rectangle(args, kwargs) (line 352)
        set_clip_rectangle_call_result_222942 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), set_clip_rectangle_222939, *[rectangle_222940], **kwargs_222941)
        
        
        # ################# End of 'set_clip_rectangle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_clip_rectangle' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_222943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222943)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_clip_rectangle'
        return stypy_return_type_222943


    @norecursion
    def set_dashes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_dashes'
        module_type_store = module_type_store.open_function_context('set_dashes', 354, 4, False)
        # Assigning a type to the variable 'self' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_function_name', 'GraphicsContextGDK.set_dashes')
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_param_names_list', ['dash_offset', 'dash_list'])
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextGDK.set_dashes.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextGDK.set_dashes', ['dash_offset', 'dash_list'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_dashes', localization, ['dash_offset', 'dash_list'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_dashes(...)' code ##################

        
        # Call to set_dashes(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'self' (line 355)
        self_222946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 39), 'self', False)
        # Getting the type of 'dash_offset' (line 355)
        dash_offset_222947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 45), 'dash_offset', False)
        # Getting the type of 'dash_list' (line 355)
        dash_list_222948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 58), 'dash_list', False)
        # Processing the call keyword arguments (line 355)
        kwargs_222949 = {}
        # Getting the type of 'GraphicsContextBase' (line 355)
        GraphicsContextBase_222944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'GraphicsContextBase', False)
        # Obtaining the member 'set_dashes' of a type (line 355)
        set_dashes_222945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), GraphicsContextBase_222944, 'set_dashes')
        # Calling set_dashes(args, kwargs) (line 355)
        set_dashes_call_result_222950 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), set_dashes_222945, *[self_222946, dash_offset_222947, dash_list_222948], **kwargs_222949)
        
        
        # Type idiom detected: calculating its left and rigth part (line 357)
        # Getting the type of 'dash_list' (line 357)
        dash_list_222951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'dash_list')
        # Getting the type of 'None' (line 357)
        None_222952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 24), 'None')
        
        (may_be_222953, more_types_in_union_222954) = may_be_none(dash_list_222951, None_222952)

        if may_be_222953:

            if more_types_in_union_222954:
                # Runtime conditional SSA (line 357)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 358):
            
            # Assigning a Attribute to a Attribute (line 358):
            # Getting the type of 'gdk' (line 358)
            gdk_222955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 36), 'gdk')
            # Obtaining the member 'LINE_SOLID' of a type (line 358)
            LINE_SOLID_222956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 36), gdk_222955, 'LINE_SOLID')
            # Getting the type of 'self' (line 358)
            self_222957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'self')
            # Obtaining the member 'gdkGC' of a type (line 358)
            gdkGC_222958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), self_222957, 'gdkGC')
            # Setting the type of the member 'line_style' of a type (line 358)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), gdkGC_222958, 'line_style', LINE_SOLID_222956)

            if more_types_in_union_222954:
                # Runtime conditional SSA for else branch (line 357)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_222953) or more_types_in_union_222954):
            
            # Assigning a Call to a Name (line 360):
            
            # Assigning a Call to a Name (line 360):
            
            # Call to points_to_pixels(...): (line 360)
            # Processing the call arguments (line 360)
            
            # Call to asarray(...): (line 360)
            # Processing the call arguments (line 360)
            # Getting the type of 'dash_list' (line 360)
            dash_list_222964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 63), 'dash_list', False)
            # Processing the call keyword arguments (line 360)
            kwargs_222965 = {}
            # Getting the type of 'np' (line 360)
            np_222962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 52), 'np', False)
            # Obtaining the member 'asarray' of a type (line 360)
            asarray_222963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 52), np_222962, 'asarray')
            # Calling asarray(args, kwargs) (line 360)
            asarray_call_result_222966 = invoke(stypy.reporting.localization.Localization(__file__, 360, 52), asarray_222963, *[dash_list_222964], **kwargs_222965)
            
            # Processing the call keyword arguments (line 360)
            kwargs_222967 = {}
            # Getting the type of 'self' (line 360)
            self_222959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 21), 'self', False)
            # Obtaining the member 'renderer' of a type (line 360)
            renderer_222960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 21), self_222959, 'renderer')
            # Obtaining the member 'points_to_pixels' of a type (line 360)
            points_to_pixels_222961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 21), renderer_222960, 'points_to_pixels')
            # Calling points_to_pixels(args, kwargs) (line 360)
            points_to_pixels_call_result_222968 = invoke(stypy.reporting.localization.Localization(__file__, 360, 21), points_to_pixels_222961, *[asarray_call_result_222966], **kwargs_222967)
            
            # Assigning a type to the variable 'pixels' (line 360)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'pixels', points_to_pixels_call_result_222968)
            
            # Assigning a ListComp to a Name (line 361):
            
            # Assigning a ListComp to a Name (line 361):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'pixels' (line 361)
            pixels_222981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 56), 'pixels')
            comprehension_222982 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 18), pixels_222981)
            # Assigning a type to the variable 'val' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 18), 'val', comprehension_222982)
            
            # Call to max(...): (line 361)
            # Processing the call arguments (line 361)
            int_222970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 22), 'int')
            
            # Call to int(...): (line 361)
            # Processing the call arguments (line 361)
            
            # Call to round(...): (line 361)
            # Processing the call arguments (line 361)
            # Getting the type of 'val' (line 361)
            val_222974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 38), 'val', False)
            # Processing the call keyword arguments (line 361)
            kwargs_222975 = {}
            # Getting the type of 'np' (line 361)
            np_222972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 29), 'np', False)
            # Obtaining the member 'round' of a type (line 361)
            round_222973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 29), np_222972, 'round')
            # Calling round(args, kwargs) (line 361)
            round_call_result_222976 = invoke(stypy.reporting.localization.Localization(__file__, 361, 29), round_222973, *[val_222974], **kwargs_222975)
            
            # Processing the call keyword arguments (line 361)
            kwargs_222977 = {}
            # Getting the type of 'int' (line 361)
            int_222971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 25), 'int', False)
            # Calling int(args, kwargs) (line 361)
            int_call_result_222978 = invoke(stypy.reporting.localization.Localization(__file__, 361, 25), int_222971, *[round_call_result_222976], **kwargs_222977)
            
            # Processing the call keyword arguments (line 361)
            kwargs_222979 = {}
            # Getting the type of 'max' (line 361)
            max_222969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 18), 'max', False)
            # Calling max(args, kwargs) (line 361)
            max_call_result_222980 = invoke(stypy.reporting.localization.Localization(__file__, 361, 18), max_222969, *[int_222970, int_call_result_222978], **kwargs_222979)
            
            list_222983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 18), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 18), list_222983, max_call_result_222980)
            # Assigning a type to the variable 'dl' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'dl', list_222983)
            
            # Call to set_dashes(...): (line 362)
            # Processing the call arguments (line 362)
            # Getting the type of 'dash_offset' (line 362)
            dash_offset_222987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 34), 'dash_offset', False)
            # Getting the type of 'dl' (line 362)
            dl_222988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 47), 'dl', False)
            # Processing the call keyword arguments (line 362)
            kwargs_222989 = {}
            # Getting the type of 'self' (line 362)
            self_222984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'self', False)
            # Obtaining the member 'gdkGC' of a type (line 362)
            gdkGC_222985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), self_222984, 'gdkGC')
            # Obtaining the member 'set_dashes' of a type (line 362)
            set_dashes_222986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), gdkGC_222985, 'set_dashes')
            # Calling set_dashes(args, kwargs) (line 362)
            set_dashes_call_result_222990 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), set_dashes_222986, *[dash_offset_222987, dl_222988], **kwargs_222989)
            
            
            # Assigning a Attribute to a Attribute (line 363):
            
            # Assigning a Attribute to a Attribute (line 363):
            # Getting the type of 'gdk' (line 363)
            gdk_222991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 36), 'gdk')
            # Obtaining the member 'LINE_ON_OFF_DASH' of a type (line 363)
            LINE_ON_OFF_DASH_222992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 36), gdk_222991, 'LINE_ON_OFF_DASH')
            # Getting the type of 'self' (line 363)
            self_222993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'self')
            # Obtaining the member 'gdkGC' of a type (line 363)
            gdkGC_222994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), self_222993, 'gdkGC')
            # Setting the type of the member 'line_style' of a type (line 363)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), gdkGC_222994, 'line_style', LINE_ON_OFF_DASH_222992)

            if (may_be_222953 and more_types_in_union_222954):
                # SSA join for if statement (line 357)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'set_dashes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_dashes' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_222995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222995)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_dashes'
        return stypy_return_type_222995


    @norecursion
    def set_foreground(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 366)
        False_222996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 40), 'False')
        defaults = [False_222996]
        # Create a new context for function 'set_foreground'
        module_type_store = module_type_store.open_function_context('set_foreground', 366, 4, False)
        # Assigning a type to the variable 'self' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_function_name', 'GraphicsContextGDK.set_foreground')
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_param_names_list', ['fg', 'isRGBA'])
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextGDK.set_foreground.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextGDK.set_foreground', ['fg', 'isRGBA'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_foreground(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'self' (line 367)
        self_222999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 43), 'self', False)
        # Getting the type of 'fg' (line 367)
        fg_223000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 49), 'fg', False)
        # Getting the type of 'isRGBA' (line 367)
        isRGBA_223001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 53), 'isRGBA', False)
        # Processing the call keyword arguments (line 367)
        kwargs_223002 = {}
        # Getting the type of 'GraphicsContextBase' (line 367)
        GraphicsContextBase_222997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'GraphicsContextBase', False)
        # Obtaining the member 'set_foreground' of a type (line 367)
        set_foreground_222998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 8), GraphicsContextBase_222997, 'set_foreground')
        # Calling set_foreground(args, kwargs) (line 367)
        set_foreground_call_result_223003 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), set_foreground_222998, *[self_222999, fg_223000, isRGBA_223001], **kwargs_223002)
        
        
        # Assigning a Call to a Attribute (line 368):
        
        # Assigning a Call to a Attribute (line 368):
        
        # Call to rgb_to_gdk_color(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Call to get_rgb(...): (line 368)
        # Processing the call keyword arguments (line 368)
        kwargs_223008 = {}
        # Getting the type of 'self' (line 368)
        self_223006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 54), 'self', False)
        # Obtaining the member 'get_rgb' of a type (line 368)
        get_rgb_223007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 54), self_223006, 'get_rgb')
        # Calling get_rgb(args, kwargs) (line 368)
        get_rgb_call_result_223009 = invoke(stypy.reporting.localization.Localization(__file__, 368, 54), get_rgb_223007, *[], **kwargs_223008)
        
        # Processing the call keyword arguments (line 368)
        kwargs_223010 = {}
        # Getting the type of 'self' (line 368)
        self_223004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 32), 'self', False)
        # Obtaining the member 'rgb_to_gdk_color' of a type (line 368)
        rgb_to_gdk_color_223005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 32), self_223004, 'rgb_to_gdk_color')
        # Calling rgb_to_gdk_color(args, kwargs) (line 368)
        rgb_to_gdk_color_call_result_223011 = invoke(stypy.reporting.localization.Localization(__file__, 368, 32), rgb_to_gdk_color_223005, *[get_rgb_call_result_223009], **kwargs_223010)
        
        # Getting the type of 'self' (line 368)
        self_223012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'self')
        # Obtaining the member 'gdkGC' of a type (line 368)
        gdkGC_223013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), self_223012, 'gdkGC')
        # Setting the type of the member 'foreground' of a type (line 368)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), gdkGC_223013, 'foreground', rgb_to_gdk_color_call_result_223011)
        
        # ################# End of 'set_foreground(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_foreground' in the type store
        # Getting the type of 'stypy_return_type' (line 366)
        stypy_return_type_223014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_foreground'
        return stypy_return_type_223014


    @norecursion
    def set_joinstyle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_joinstyle'
        module_type_store = module_type_store.open_function_context('set_joinstyle', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_function_name', 'GraphicsContextGDK.set_joinstyle')
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_param_names_list', ['js'])
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextGDK.set_joinstyle.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextGDK.set_joinstyle', ['js'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_joinstyle(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'self' (line 372)
        self_223017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 42), 'self', False)
        # Getting the type of 'js' (line 372)
        js_223018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 48), 'js', False)
        # Processing the call keyword arguments (line 372)
        kwargs_223019 = {}
        # Getting the type of 'GraphicsContextBase' (line 372)
        GraphicsContextBase_223015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'GraphicsContextBase', False)
        # Obtaining the member 'set_joinstyle' of a type (line 372)
        set_joinstyle_223016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 8), GraphicsContextBase_223015, 'set_joinstyle')
        # Calling set_joinstyle(args, kwargs) (line 372)
        set_joinstyle_call_result_223020 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), set_joinstyle_223016, *[self_223017, js_223018], **kwargs_223019)
        
        
        # Assigning a Subscript to a Attribute (line 373):
        
        # Assigning a Subscript to a Attribute (line 373):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 373)
        self_223021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 44), 'self')
        # Obtaining the member '_joinstyle' of a type (line 373)
        _joinstyle_223022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 44), self_223021, '_joinstyle')
        # Getting the type of 'self' (line 373)
        self_223023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 32), 'self')
        # Obtaining the member '_joind' of a type (line 373)
        _joind_223024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 32), self_223023, '_joind')
        # Obtaining the member '__getitem__' of a type (line 373)
        getitem___223025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 32), _joind_223024, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 373)
        subscript_call_result_223026 = invoke(stypy.reporting.localization.Localization(__file__, 373, 32), getitem___223025, _joinstyle_223022)
        
        # Getting the type of 'self' (line 373)
        self_223027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'self')
        # Obtaining the member 'gdkGC' of a type (line 373)
        gdkGC_223028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), self_223027, 'gdkGC')
        # Setting the type of the member 'join_style' of a type (line 373)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), gdkGC_223028, 'join_style', subscript_call_result_223026)
        
        # ################# End of 'set_joinstyle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_joinstyle' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_223029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_joinstyle'
        return stypy_return_type_223029


    @norecursion
    def set_linewidth(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_linewidth'
        module_type_store = module_type_store.open_function_context('set_linewidth', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_function_name', 'GraphicsContextGDK.set_linewidth')
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_param_names_list', ['w'])
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextGDK.set_linewidth.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextGDK.set_linewidth', ['w'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_linewidth(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'self' (line 377)
        self_223032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 42), 'self', False)
        # Getting the type of 'w' (line 377)
        w_223033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 48), 'w', False)
        # Processing the call keyword arguments (line 377)
        kwargs_223034 = {}
        # Getting the type of 'GraphicsContextBase' (line 377)
        GraphicsContextBase_223030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'GraphicsContextBase', False)
        # Obtaining the member 'set_linewidth' of a type (line 377)
        set_linewidth_223031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), GraphicsContextBase_223030, 'set_linewidth')
        # Calling set_linewidth(args, kwargs) (line 377)
        set_linewidth_call_result_223035 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), set_linewidth_223031, *[self_223032, w_223033], **kwargs_223034)
        
        
        
        # Getting the type of 'w' (line 378)
        w_223036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 11), 'w')
        int_223037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 16), 'int')
        # Applying the binary operator '==' (line 378)
        result_eq_223038 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 11), '==', w_223036, int_223037)
        
        # Testing the type of an if condition (line 378)
        if_condition_223039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 8), result_eq_223038)
        # Assigning a type to the variable 'if_condition_223039' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'if_condition_223039', if_condition_223039)
        # SSA begins for if statement (line 378)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 379):
        
        # Assigning a Num to a Attribute (line 379):
        int_223040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 36), 'int')
        # Getting the type of 'self' (line 379)
        self_223041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'self')
        # Obtaining the member 'gdkGC' of a type (line 379)
        gdkGC_223042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 12), self_223041, 'gdkGC')
        # Setting the type of the member 'line_width' of a type (line 379)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 12), gdkGC_223042, 'line_width', int_223040)
        # SSA branch for the else part of an if statement (line 378)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 381):
        
        # Assigning a Call to a Name (line 381):
        
        # Call to points_to_pixels(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'w' (line 381)
        w_223046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 52), 'w', False)
        # Processing the call keyword arguments (line 381)
        kwargs_223047 = {}
        # Getting the type of 'self' (line 381)
        self_223043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'self', False)
        # Obtaining the member 'renderer' of a type (line 381)
        renderer_223044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), self_223043, 'renderer')
        # Obtaining the member 'points_to_pixels' of a type (line 381)
        points_to_pixels_223045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), renderer_223044, 'points_to_pixels')
        # Calling points_to_pixels(args, kwargs) (line 381)
        points_to_pixels_call_result_223048 = invoke(stypy.reporting.localization.Localization(__file__, 381, 21), points_to_pixels_223045, *[w_223046], **kwargs_223047)
        
        # Assigning a type to the variable 'pixels' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'pixels', points_to_pixels_call_result_223048)
        
        # Assigning a Call to a Attribute (line 382):
        
        # Assigning a Call to a Attribute (line 382):
        
        # Call to max(...): (line 382)
        # Processing the call arguments (line 382)
        int_223050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 40), 'int')
        
        # Call to int(...): (line 382)
        # Processing the call arguments (line 382)
        
        # Call to round(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'pixels' (line 382)
        pixels_223054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 56), 'pixels', False)
        # Processing the call keyword arguments (line 382)
        kwargs_223055 = {}
        # Getting the type of 'np' (line 382)
        np_223052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 47), 'np', False)
        # Obtaining the member 'round' of a type (line 382)
        round_223053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 47), np_223052, 'round')
        # Calling round(args, kwargs) (line 382)
        round_call_result_223056 = invoke(stypy.reporting.localization.Localization(__file__, 382, 47), round_223053, *[pixels_223054], **kwargs_223055)
        
        # Processing the call keyword arguments (line 382)
        kwargs_223057 = {}
        # Getting the type of 'int' (line 382)
        int_223051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 43), 'int', False)
        # Calling int(args, kwargs) (line 382)
        int_call_result_223058 = invoke(stypy.reporting.localization.Localization(__file__, 382, 43), int_223051, *[round_call_result_223056], **kwargs_223057)
        
        # Processing the call keyword arguments (line 382)
        kwargs_223059 = {}
        # Getting the type of 'max' (line 382)
        max_223049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 36), 'max', False)
        # Calling max(args, kwargs) (line 382)
        max_call_result_223060 = invoke(stypy.reporting.localization.Localization(__file__, 382, 36), max_223049, *[int_223050, int_call_result_223058], **kwargs_223059)
        
        # Getting the type of 'self' (line 382)
        self_223061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'self')
        # Obtaining the member 'gdkGC' of a type (line 382)
        gdkGC_223062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), self_223061, 'gdkGC')
        # Setting the type of the member 'line_width' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), gdkGC_223062, 'line_width', max_call_result_223060)
        # SSA join for if statement (line 378)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_linewidth(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_linewidth' in the type store
        # Getting the type of 'stypy_return_type' (line 376)
        stypy_return_type_223063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_linewidth'
        return stypy_return_type_223063


# Assigning a type to the variable 'GraphicsContextGDK' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'GraphicsContextGDK', GraphicsContextGDK)

# Assigning a Dict to a Name (line 299):

# Obtaining an instance of the builtin type 'dict' (line 299)
dict_223064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 299)

# Getting the type of 'GraphicsContextGDK'
GraphicsContextGDK_223065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GraphicsContextGDK')
# Setting the type of the member '_cached' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GraphicsContextGDK_223065, '_cached', dict_223064)

# Assigning a Dict to a Name (line 301):

# Obtaining an instance of the builtin type 'dict' (line 301)
dict_223066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 301)
# Adding element type (key, value) (line 301)
unicode_223067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 8), 'unicode', u'bevel')
# Getting the type of 'gdk' (line 302)
gdk_223068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 18), 'gdk')
# Obtaining the member 'JOIN_BEVEL' of a type (line 302)
JOIN_BEVEL_223069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 18), gdk_223068, 'JOIN_BEVEL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 13), dict_223066, (unicode_223067, JOIN_BEVEL_223069))
# Adding element type (key, value) (line 301)
unicode_223070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 8), 'unicode', u'miter')
# Getting the type of 'gdk' (line 303)
gdk_223071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 18), 'gdk')
# Obtaining the member 'JOIN_MITER' of a type (line 303)
JOIN_MITER_223072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 18), gdk_223071, 'JOIN_MITER')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 13), dict_223066, (unicode_223070, JOIN_MITER_223072))
# Adding element type (key, value) (line 301)
unicode_223073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 8), 'unicode', u'round')
# Getting the type of 'gdk' (line 304)
gdk_223074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'gdk')
# Obtaining the member 'JOIN_ROUND' of a type (line 304)
JOIN_ROUND_223075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 18), gdk_223074, 'JOIN_ROUND')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 13), dict_223066, (unicode_223073, JOIN_ROUND_223075))

# Getting the type of 'GraphicsContextGDK'
GraphicsContextGDK_223076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GraphicsContextGDK')
# Setting the type of the member '_joind' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GraphicsContextGDK_223076, '_joind', dict_223066)

# Assigning a Dict to a Name (line 307):

# Obtaining an instance of the builtin type 'dict' (line 307)
dict_223077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 307)
# Adding element type (key, value) (line 307)
unicode_223078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 8), 'unicode', u'butt')
# Getting the type of 'gdk' (line 308)
gdk_223079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 23), 'gdk')
# Obtaining the member 'CAP_BUTT' of a type (line 308)
CAP_BUTT_223080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 23), gdk_223079, 'CAP_BUTT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 12), dict_223077, (unicode_223078, CAP_BUTT_223080))
# Adding element type (key, value) (line 307)
unicode_223081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 8), 'unicode', u'projecting')
# Getting the type of 'gdk' (line 309)
gdk_223082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'gdk')
# Obtaining the member 'CAP_PROJECTING' of a type (line 309)
CAP_PROJECTING_223083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 23), gdk_223082, 'CAP_PROJECTING')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 12), dict_223077, (unicode_223081, CAP_PROJECTING_223083))
# Adding element type (key, value) (line 307)
unicode_223084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 8), 'unicode', u'round')
# Getting the type of 'gdk' (line 310)
gdk_223085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 23), 'gdk')
# Obtaining the member 'CAP_ROUND' of a type (line 310)
CAP_ROUND_223086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 23), gdk_223085, 'CAP_ROUND')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 12), dict_223077, (unicode_223084, CAP_ROUND_223086))

# Getting the type of 'GraphicsContextGDK'
GraphicsContextGDK_223087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GraphicsContextGDK')
# Setting the type of the member '_capd' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GraphicsContextGDK_223087, '_capd', dict_223077)
# Declaration of the 'FigureCanvasGDK' class
# Getting the type of 'FigureCanvasBase' (line 385)
FigureCanvasBase_223088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'FigureCanvasBase')

class FigureCanvasGDK(FigureCanvasBase_223088, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 386, 4, False)
        # Assigning a type to the variable 'self' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGDK.__init__', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'self' (line 387)
        self_223091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 34), 'self', False)
        # Getting the type of 'figure' (line 387)
        figure_223092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 40), 'figure', False)
        # Processing the call keyword arguments (line 387)
        kwargs_223093 = {}
        # Getting the type of 'FigureCanvasBase' (line 387)
        FigureCanvasBase_223089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'FigureCanvasBase', False)
        # Obtaining the member '__init__' of a type (line 387)
        init___223090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), FigureCanvasBase_223089, '__init__')
        # Calling __init__(args, kwargs) (line 387)
        init___call_result_223094 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), init___223090, *[self_223091, figure_223092], **kwargs_223093)
        
        
        
        # Getting the type of 'self' (line 388)
        self_223095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), 'self')
        # Obtaining the member '__class__' of a type (line 388)
        class___223096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 11), self_223095, '__class__')
        # Getting the type of 'matplotlib' (line 388)
        matplotlib_223097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 29), 'matplotlib')
        # Obtaining the member 'backends' of a type (line 388)
        backends_223098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 29), matplotlib_223097, 'backends')
        # Obtaining the member 'backend_gdk' of a type (line 388)
        backend_gdk_223099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 29), backends_223098, 'backend_gdk')
        # Obtaining the member 'FigureCanvasGDK' of a type (line 388)
        FigureCanvasGDK_223100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 29), backend_gdk_223099, 'FigureCanvasGDK')
        # Applying the binary operator '==' (line 388)
        result_eq_223101 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 11), '==', class___223096, FigureCanvasGDK_223100)
        
        # Testing the type of an if condition (line 388)
        if_condition_223102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 8), result_eq_223101)
        # Assigning a type to the variable 'if_condition_223102' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'if_condition_223102', if_condition_223102)
        # SSA begins for if statement (line 388)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn_deprecated(...): (line 389)
        # Processing the call arguments (line 389)
        unicode_223104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 28), 'unicode', u'2.0')
        # Processing the call keyword arguments (line 389)
        unicode_223105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 43), 'unicode', u'The GDK backend is deprecated. It is untested, known to be broken and will be removed in Matplotlib 2.2. Use the Agg backend instead. See Matplotlib usage FAQ for more info on backends.')
        keyword_223106 = unicode_223105
        unicode_223107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 40), 'unicode', u'Agg')
        keyword_223108 = unicode_223107
        kwargs_223109 = {'message': keyword_223106, 'alternative': keyword_223108}
        # Getting the type of 'warn_deprecated' (line 389)
        warn_deprecated_223103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'warn_deprecated', False)
        # Calling warn_deprecated(args, kwargs) (line 389)
        warn_deprecated_call_result_223110 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), warn_deprecated_223103, *[unicode_223104], **kwargs_223109)
        
        # SSA join for if statement (line 388)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _renderer_init(...): (line 396)
        # Processing the call keyword arguments (line 396)
        kwargs_223113 = {}
        # Getting the type of 'self' (line 396)
        self_223111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self', False)
        # Obtaining the member '_renderer_init' of a type (line 396)
        _renderer_init_223112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_223111, '_renderer_init')
        # Calling _renderer_init(args, kwargs) (line 396)
        _renderer_init_call_result_223114 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), _renderer_init_223112, *[], **kwargs_223113)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _renderer_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_renderer_init'
        module_type_store = module_type_store.open_function_context('_renderer_init', 398, 4, False)
        # Assigning a type to the variable 'self' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGDK._renderer_init')
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGDK._renderer_init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGDK._renderer_init', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_renderer_init', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_renderer_init(...)' code ##################

        
        # Assigning a Call to a Attribute (line 399):
        
        # Assigning a Call to a Attribute (line 399):
        
        # Call to RendererGDK(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Call to DrawingArea(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_223118 = {}
        # Getting the type of 'gtk' (line 399)
        gtk_223116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 38), 'gtk', False)
        # Obtaining the member 'DrawingArea' of a type (line 399)
        DrawingArea_223117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 38), gtk_223116, 'DrawingArea')
        # Calling DrawingArea(args, kwargs) (line 399)
        DrawingArea_call_result_223119 = invoke(stypy.reporting.localization.Localization(__file__, 399, 38), DrawingArea_223117, *[], **kwargs_223118)
        
        # Getting the type of 'self' (line 399)
        self_223120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 57), 'self', False)
        # Obtaining the member 'figure' of a type (line 399)
        figure_223121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 57), self_223120, 'figure')
        # Obtaining the member 'dpi' of a type (line 399)
        dpi_223122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 57), figure_223121, 'dpi')
        # Processing the call keyword arguments (line 399)
        kwargs_223123 = {}
        # Getting the type of 'RendererGDK' (line 399)
        RendererGDK_223115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 25), 'RendererGDK', False)
        # Calling RendererGDK(args, kwargs) (line 399)
        RendererGDK_call_result_223124 = invoke(stypy.reporting.localization.Localization(__file__, 399, 25), RendererGDK_223115, *[DrawingArea_call_result_223119, dpi_223122], **kwargs_223123)
        
        # Getting the type of 'self' (line 399)
        self_223125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self')
        # Setting the type of the member '_renderer' of a type (line 399)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_223125, '_renderer', RendererGDK_call_result_223124)
        
        # ################# End of '_renderer_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_renderer_init' in the type store
        # Getting the type of 'stypy_return_type' (line 398)
        stypy_return_type_223126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_renderer_init'
        return stypy_return_type_223126


    @norecursion
    def _render_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_render_figure'
        module_type_store = module_type_store.open_function_context('_render_figure', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGDK._render_figure')
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_param_names_list', ['pixmap', 'width', 'height'])
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGDK._render_figure.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGDK._render_figure', ['pixmap', 'width', 'height'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_render_figure', localization, ['pixmap', 'width', 'height'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_render_figure(...)' code ##################

        
        # Call to set_pixmap(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'pixmap' (line 402)
        pixmap_223130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 35), 'pixmap', False)
        # Processing the call keyword arguments (line 402)
        kwargs_223131 = {}
        # Getting the type of 'self' (line 402)
        self_223127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self', False)
        # Obtaining the member '_renderer' of a type (line 402)
        _renderer_223128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_223127, '_renderer')
        # Obtaining the member 'set_pixmap' of a type (line 402)
        set_pixmap_223129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), _renderer_223128, 'set_pixmap')
        # Calling set_pixmap(args, kwargs) (line 402)
        set_pixmap_call_result_223132 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), set_pixmap_223129, *[pixmap_223130], **kwargs_223131)
        
        
        # Call to set_width_height(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'width' (line 403)
        width_223136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 41), 'width', False)
        # Getting the type of 'height' (line 403)
        height_223137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 48), 'height', False)
        # Processing the call keyword arguments (line 403)
        kwargs_223138 = {}
        # Getting the type of 'self' (line 403)
        self_223133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self', False)
        # Obtaining the member '_renderer' of a type (line 403)
        _renderer_223134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_223133, '_renderer')
        # Obtaining the member 'set_width_height' of a type (line 403)
        set_width_height_223135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), _renderer_223134, 'set_width_height')
        # Calling set_width_height(args, kwargs) (line 403)
        set_width_height_call_result_223139 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), set_width_height_223135, *[width_223136, height_223137], **kwargs_223138)
        
        
        # Call to draw(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'self' (line 404)
        self_223143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 26), 'self', False)
        # Obtaining the member '_renderer' of a type (line 404)
        _renderer_223144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 26), self_223143, '_renderer')
        # Processing the call keyword arguments (line 404)
        kwargs_223145 = {}
        # Getting the type of 'self' (line 404)
        self_223140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 404)
        figure_223141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), self_223140, 'figure')
        # Obtaining the member 'draw' of a type (line 404)
        draw_223142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), figure_223141, 'draw')
        # Calling draw(args, kwargs) (line 404)
        draw_call_result_223146 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), draw_223142, *[_renderer_223144], **kwargs_223145)
        
        
        # ################# End of '_render_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_render_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_223147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223147)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_render_figure'
        return stypy_return_type_223147

    
    # Assigning a Call to a Name (line 406):
    
    # Assigning a Str to a Subscript (line 407):
    
    # Assigning a Str to a Subscript (line 408):

    @norecursion
    def print_jpeg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_jpeg'
        module_type_store = module_type_store.open_function_context('print_jpeg', 410, 4, False)
        # Assigning a type to the variable 'self' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGDK.print_jpeg')
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGDK.print_jpeg.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGDK.print_jpeg', ['filename'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_jpeg', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_jpeg(...)' code ##################

        
        # Call to _print_image(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'filename' (line 411)
        filename_223150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 33), 'filename', False)
        unicode_223151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 43), 'unicode', u'jpeg')
        # Processing the call keyword arguments (line 411)
        kwargs_223152 = {}
        # Getting the type of 'self' (line 411)
        self_223148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'self', False)
        # Obtaining the member '_print_image' of a type (line 411)
        _print_image_223149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 15), self_223148, '_print_image')
        # Calling _print_image(args, kwargs) (line 411)
        _print_image_call_result_223153 = invoke(stypy.reporting.localization.Localization(__file__, 411, 15), _print_image_223149, *[filename_223150, unicode_223151], **kwargs_223152)
        
        # Assigning a type to the variable 'stypy_return_type' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'stypy_return_type', _print_image_call_result_223153)
        
        # ################# End of 'print_jpeg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_jpeg' in the type store
        # Getting the type of 'stypy_return_type' (line 410)
        stypy_return_type_223154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223154)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_jpeg'
        return stypy_return_type_223154

    
    # Assigning a Name to a Name (line 412):

    @norecursion
    def print_png(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_png'
        module_type_store = module_type_store.open_function_context('print_png', 414, 4, False)
        # Assigning a type to the variable 'self' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGDK.print_png')
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGDK.print_png.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGDK.print_png', ['filename'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_png', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_png(...)' code ##################

        
        # Call to _print_image(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'filename' (line 415)
        filename_223157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 33), 'filename', False)
        unicode_223158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 43), 'unicode', u'png')
        # Processing the call keyword arguments (line 415)
        kwargs_223159 = {}
        # Getting the type of 'self' (line 415)
        self_223155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'self', False)
        # Obtaining the member '_print_image' of a type (line 415)
        _print_image_223156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 15), self_223155, '_print_image')
        # Calling _print_image(args, kwargs) (line 415)
        _print_image_call_result_223160 = invoke(stypy.reporting.localization.Localization(__file__, 415, 15), _print_image_223156, *[filename_223157, unicode_223158], **kwargs_223159)
        
        # Assigning a type to the variable 'stypy_return_type' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'stypy_return_type', _print_image_call_result_223160)
        
        # ################# End of 'print_png(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_png' in the type store
        # Getting the type of 'stypy_return_type' (line 414)
        stypy_return_type_223161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223161)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_png'
        return stypy_return_type_223161


    @norecursion
    def _print_image(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_print_image'
        module_type_store = module_type_store.open_function_context('_print_image', 417, 4, False)
        # Assigning a type to the variable 'self' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGDK._print_image')
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_param_names_list', ['filename', 'format'])
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGDK._print_image.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGDK._print_image', ['filename', 'format'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_print_image', localization, ['filename', 'format'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_print_image(...)' code ##################

        
        # Assigning a Call to a Tuple (line 418):
        
        # Assigning a Call to a Name:
        
        # Call to get_width_height(...): (line 418)
        # Processing the call keyword arguments (line 418)
        kwargs_223164 = {}
        # Getting the type of 'self' (line 418)
        self_223162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'self', False)
        # Obtaining the member 'get_width_height' of a type (line 418)
        get_width_height_223163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 24), self_223162, 'get_width_height')
        # Calling get_width_height(args, kwargs) (line 418)
        get_width_height_call_result_223165 = invoke(stypy.reporting.localization.Localization(__file__, 418, 24), get_width_height_223163, *[], **kwargs_223164)
        
        # Assigning a type to the variable 'call_assignment_221559' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_221559', get_width_height_call_result_223165)
        
        # Assigning a Call to a Name (line 418):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_223168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 8), 'int')
        # Processing the call keyword arguments
        kwargs_223169 = {}
        # Getting the type of 'call_assignment_221559' (line 418)
        call_assignment_221559_223166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_221559', False)
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___223167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), call_assignment_221559_223166, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_223170 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___223167, *[int_223168], **kwargs_223169)
        
        # Assigning a type to the variable 'call_assignment_221560' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_221560', getitem___call_result_223170)
        
        # Assigning a Name to a Name (line 418):
        # Getting the type of 'call_assignment_221560' (line 418)
        call_assignment_221560_223171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_221560')
        # Assigning a type to the variable 'width' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'width', call_assignment_221560_223171)
        
        # Assigning a Call to a Name (line 418):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_223174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 8), 'int')
        # Processing the call keyword arguments
        kwargs_223175 = {}
        # Getting the type of 'call_assignment_221559' (line 418)
        call_assignment_221559_223172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_221559', False)
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___223173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), call_assignment_221559_223172, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_223176 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___223173, *[int_223174], **kwargs_223175)
        
        # Assigning a type to the variable 'call_assignment_221561' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_221561', getitem___call_result_223176)
        
        # Assigning a Name to a Name (line 418):
        # Getting the type of 'call_assignment_221561' (line 418)
        call_assignment_221561_223177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_221561')
        # Assigning a type to the variable 'height' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'height', call_assignment_221561_223177)
        
        # Assigning a Call to a Name (line 419):
        
        # Assigning a Call to a Name (line 419):
        
        # Call to Pixmap(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'None' (line 419)
        None_223181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 33), 'None', False)
        # Getting the type of 'width' (line 419)
        width_223182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 39), 'width', False)
        # Getting the type of 'height' (line 419)
        height_223183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 46), 'height', False)
        # Processing the call keyword arguments (line 419)
        int_223184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 60), 'int')
        keyword_223185 = int_223184
        kwargs_223186 = {'depth': keyword_223185}
        # Getting the type of 'gtk' (line 419)
        gtk_223178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 17), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 419)
        gdk_223179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 17), gtk_223178, 'gdk')
        # Obtaining the member 'Pixmap' of a type (line 419)
        Pixmap_223180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 17), gdk_223179, 'Pixmap')
        # Calling Pixmap(args, kwargs) (line 419)
        Pixmap_call_result_223187 = invoke(stypy.reporting.localization.Localization(__file__, 419, 17), Pixmap_223180, *[None_223181, width_223182, height_223183], **kwargs_223186)
        
        # Assigning a type to the variable 'pixmap' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'pixmap', Pixmap_call_result_223187)
        
        # Call to _render_figure(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'pixmap' (line 420)
        pixmap_223190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 28), 'pixmap', False)
        # Getting the type of 'width' (line 420)
        width_223191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 36), 'width', False)
        # Getting the type of 'height' (line 420)
        height_223192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 43), 'height', False)
        # Processing the call keyword arguments (line 420)
        kwargs_223193 = {}
        # Getting the type of 'self' (line 420)
        self_223188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'self', False)
        # Obtaining the member '_render_figure' of a type (line 420)
        _render_figure_223189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 8), self_223188, '_render_figure')
        # Calling _render_figure(args, kwargs) (line 420)
        _render_figure_call_result_223194 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), _render_figure_223189, *[pixmap_223190, width_223191, height_223192], **kwargs_223193)
        
        
        # Assigning a Call to a Name (line 424):
        
        # Assigning a Call to a Name (line 424):
        
        # Call to Pixbuf(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'gtk' (line 424)
        gtk_223198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 32), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 424)
        gdk_223199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 32), gtk_223198, 'gdk')
        # Obtaining the member 'COLORSPACE_RGB' of a type (line 424)
        COLORSPACE_RGB_223200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 32), gdk_223199, 'COLORSPACE_RGB')
        int_223201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 56), 'int')
        int_223202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 59), 'int')
        # Getting the type of 'width' (line 425)
        width_223203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 32), 'width', False)
        # Getting the type of 'height' (line 425)
        height_223204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 39), 'height', False)
        # Processing the call keyword arguments (line 424)
        kwargs_223205 = {}
        # Getting the type of 'gtk' (line 424)
        gtk_223195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 17), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 424)
        gdk_223196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 17), gtk_223195, 'gdk')
        # Obtaining the member 'Pixbuf' of a type (line 424)
        Pixbuf_223197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 17), gdk_223196, 'Pixbuf')
        # Calling Pixbuf(args, kwargs) (line 424)
        Pixbuf_call_result_223206 = invoke(stypy.reporting.localization.Localization(__file__, 424, 17), Pixbuf_223197, *[COLORSPACE_RGB_223200, int_223201, int_223202, width_223203, height_223204], **kwargs_223205)
        
        # Assigning a type to the variable 'pixbuf' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'pixbuf', Pixbuf_call_result_223206)
        
        # Call to get_from_drawable(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'pixmap' (line 426)
        pixmap_223209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 33), 'pixmap', False)
        
        # Call to get_colormap(...): (line 426)
        # Processing the call keyword arguments (line 426)
        kwargs_223212 = {}
        # Getting the type of 'pixmap' (line 426)
        pixmap_223210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 41), 'pixmap', False)
        # Obtaining the member 'get_colormap' of a type (line 426)
        get_colormap_223211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 41), pixmap_223210, 'get_colormap')
        # Calling get_colormap(args, kwargs) (line 426)
        get_colormap_call_result_223213 = invoke(stypy.reporting.localization.Localization(__file__, 426, 41), get_colormap_223211, *[], **kwargs_223212)
        
        int_223214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 33), 'int')
        int_223215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 36), 'int')
        int_223216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 39), 'int')
        int_223217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 42), 'int')
        # Getting the type of 'width' (line 427)
        width_223218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 45), 'width', False)
        # Getting the type of 'height' (line 427)
        height_223219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 52), 'height', False)
        # Processing the call keyword arguments (line 426)
        kwargs_223220 = {}
        # Getting the type of 'pixbuf' (line 426)
        pixbuf_223207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'pixbuf', False)
        # Obtaining the member 'get_from_drawable' of a type (line 426)
        get_from_drawable_223208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), pixbuf_223207, 'get_from_drawable')
        # Calling get_from_drawable(args, kwargs) (line 426)
        get_from_drawable_call_result_223221 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), get_from_drawable_223208, *[pixmap_223209, get_colormap_call_result_223213, int_223214, int_223215, int_223216, int_223217, width_223218, height_223219], **kwargs_223220)
        
        
        # Assigning a DictComp to a Name (line 431):
        
        # Assigning a DictComp to a Name (line 431):
        # Calculating dict comprehension
        module_type_store = module_type_store.open_function_context('dict comprehension expression', 431, 19, True)
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'list' (line 431)
        list_223230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 431)
        # Adding element type (line 431)
        unicode_223231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 42), 'unicode', u'quality')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 41), list_223230, unicode_223231)
        
        comprehension_223232 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 19), list_223230)
        # Assigning a type to the variable 'k' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'k', comprehension_223232)
        
        # Getting the type of 'k' (line 431)
        k_223227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 56), 'k')
        # Getting the type of 'kwargs' (line 431)
        kwargs_223228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 61), 'kwargs')
        # Applying the binary operator 'in' (line 431)
        result_contains_223229 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 56), 'in', k_223227, kwargs_223228)
        
        # Getting the type of 'k' (line 431)
        k_223222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'k')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 431)
        k_223223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 29), 'k')
        # Getting the type of 'kwargs' (line 431)
        kwargs_223224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 22), 'kwargs')
        # Obtaining the member '__getitem__' of a type (line 431)
        getitem___223225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 22), kwargs_223224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 431)
        subscript_call_result_223226 = invoke(stypy.reporting.localization.Localization(__file__, 431, 22), getitem___223225, k_223223)
        
        dict_223233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 19), 'dict')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 19), dict_223233, (k_223222, subscript_call_result_223226))
        # Assigning a type to the variable 'options' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'options', dict_223233)
        
        
        # Getting the type of 'format' (line 432)
        format_223234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'format')
        
        # Obtaining an instance of the builtin type 'list' (line 432)
        list_223235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 432)
        # Adding element type (line 432)
        unicode_223236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 22), 'unicode', u'jpg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 21), list_223235, unicode_223236)
        # Adding element type (line 432)
        unicode_223237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 29), 'unicode', u'jpeg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 21), list_223235, unicode_223237)
        
        # Applying the binary operator 'in' (line 432)
        result_contains_223238 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 11), 'in', format_223234, list_223235)
        
        # Testing the type of an if condition (line 432)
        if_condition_223239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 8), result_contains_223238)
        # Assigning a type to the variable 'if_condition_223239' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'if_condition_223239', if_condition_223239)
        # SSA begins for if statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 433)
        # Processing the call arguments (line 433)
        unicode_223242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 31), 'unicode', u'quality')
        
        # Obtaining the type of the subscript
        unicode_223243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 51), 'unicode', u'savefig.jpeg_quality')
        # Getting the type of 'rcParams' (line 433)
        rcParams_223244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 42), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___223245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 42), rcParams_223244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_223246 = invoke(stypy.reporting.localization.Localization(__file__, 433, 42), getitem___223245, unicode_223243)
        
        # Processing the call keyword arguments (line 433)
        kwargs_223247 = {}
        # Getting the type of 'options' (line 433)
        options_223240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 433)
        setdefault_223241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 12), options_223240, 'setdefault')
        # Calling setdefault(args, kwargs) (line 433)
        setdefault_call_result_223248 = invoke(stypy.reporting.localization.Localization(__file__, 433, 12), setdefault_223241, *[unicode_223242, subscript_call_result_223246], **kwargs_223247)
        
        
        # Assigning a Call to a Subscript (line 434):
        
        # Assigning a Call to a Subscript (line 434):
        
        # Call to str(...): (line 434)
        # Processing the call arguments (line 434)
        
        # Obtaining the type of the subscript
        unicode_223250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 45), 'unicode', u'quality')
        # Getting the type of 'options' (line 434)
        options_223251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 37), 'options', False)
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___223252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 37), options_223251, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_223253 = invoke(stypy.reporting.localization.Localization(__file__, 434, 37), getitem___223252, unicode_223250)
        
        # Processing the call keyword arguments (line 434)
        kwargs_223254 = {}
        # Getting the type of 'str' (line 434)
        str_223249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 33), 'str', False)
        # Calling str(args, kwargs) (line 434)
        str_call_result_223255 = invoke(stypy.reporting.localization.Localization(__file__, 434, 33), str_223249, *[subscript_call_result_223253], **kwargs_223254)
        
        # Getting the type of 'options' (line 434)
        options_223256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'options')
        unicode_223257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 20), 'unicode', u'quality')
        # Storing an element on a container (line 434)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 12), options_223256, (unicode_223257, str_call_result_223255))
        # SSA join for if statement (line 432)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to save(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'filename' (line 436)
        filename_223260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'filename', False)
        # Getting the type of 'format' (line 436)
        format_223261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 30), 'format', False)
        # Processing the call keyword arguments (line 436)
        # Getting the type of 'options' (line 436)
        options_223262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 46), 'options', False)
        keyword_223263 = options_223262
        kwargs_223264 = {'options': keyword_223263}
        # Getting the type of 'pixbuf' (line 436)
        pixbuf_223258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'pixbuf', False)
        # Obtaining the member 'save' of a type (line 436)
        save_223259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), pixbuf_223258, 'save')
        # Calling save(args, kwargs) (line 436)
        save_call_result_223265 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), save_223259, *[filename_223260, format_223261], **kwargs_223264)
        
        
        # ################# End of '_print_image(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_print_image' in the type store
        # Getting the type of 'stypy_return_type' (line 417)
        stypy_return_type_223266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_print_image'
        return stypy_return_type_223266


# Assigning a type to the variable 'FigureCanvasGDK' (line 385)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), 'FigureCanvasGDK', FigureCanvasGDK)

# Assigning a Call to a Name (line 406):

# Call to copy(...): (line 406)
# Processing the call keyword arguments (line 406)
kwargs_223270 = {}
# Getting the type of 'FigureCanvasBase' (line 406)
FigureCanvasBase_223267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'FigureCanvasBase', False)
# Obtaining the member 'filetypes' of a type (line 406)
filetypes_223268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 16), FigureCanvasBase_223267, 'filetypes')
# Obtaining the member 'copy' of a type (line 406)
copy_223269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 16), filetypes_223268, 'copy')
# Calling copy(args, kwargs) (line 406)
copy_call_result_223271 = invoke(stypy.reporting.localization.Localization(__file__, 406, 16), copy_223269, *[], **kwargs_223270)

# Getting the type of 'FigureCanvasGDK'
FigureCanvasGDK_223272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGDK')
# Setting the type of the member 'filetypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGDK_223272, 'filetypes', copy_call_result_223271)

# Assigning a Str to a Subscript (line 407):
unicode_223273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 23), 'unicode', u'JPEG')
# Getting the type of 'FigureCanvasGDK'
FigureCanvasGDK_223274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGDK')
# Setting the type of the member 'filetypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGDK_223274, 'filetypes', unicode_223273)

# Assigning a Str to a Subscript (line 408):
unicode_223275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 24), 'unicode', u'JPEG')
# Getting the type of 'FigureCanvasGDK'
FigureCanvasGDK_223276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGDK')
# Setting the type of the member 'filetypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGDK_223276, 'filetypes', unicode_223275)

# Assigning a Name to a Name (line 412):
# Getting the type of 'FigureCanvasGDK'
FigureCanvasGDK_223277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGDK')
# Obtaining the member 'print_jpeg' of a type
print_jpeg_223278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGDK_223277, 'print_jpeg')
# Getting the type of 'FigureCanvasGDK'
FigureCanvasGDK_223279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGDK')
# Setting the type of the member 'print_jpg' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGDK_223279, 'print_jpg', print_jpeg_223278)
# Declaration of the '_BackendGDK' class
# Getting the type of '_Backend' (line 440)
_Backend_223280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 18), '_Backend')

class _BackendGDK(_Backend_223280, ):
    
    # Assigning a Name to a Name (line 441):
    
    # Assigning a Name to a Name (line 442):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 439, 0, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendGDK.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendGDK' (line 439)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), '_BackendGDK', _BackendGDK)

# Assigning a Name to a Name (line 441):
# Getting the type of 'FigureCanvasGDK' (line 441)
FigureCanvasGDK_223281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 19), 'FigureCanvasGDK')
# Getting the type of '_BackendGDK'
_BackendGDK_223282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGDK')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGDK_223282, 'FigureCanvas', FigureCanvasGDK_223281)

# Assigning a Name to a Name (line 442):
# Getting the type of 'FigureManagerBase' (line 442)
FigureManagerBase_223283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'FigureManagerBase')
# Getting the type of '_BackendGDK'
_BackendGDK_223284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGDK')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGDK_223284, 'FigureManager', FigureManagerBase_223283)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
