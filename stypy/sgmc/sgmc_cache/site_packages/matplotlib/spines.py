
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import matplotlib
7: 
8: import matplotlib.artist as martist
9: from matplotlib.artist import allow_rasterization
10: from matplotlib import docstring
11: import matplotlib.transforms as mtransforms
12: import matplotlib.lines as mlines
13: import matplotlib.patches as mpatches
14: import matplotlib.path as mpath
15: import matplotlib.cbook as cbook
16: import numpy as np
17: import warnings
18: 
19: rcParams = matplotlib.rcParams
20: 
21: 
22: class Spine(mpatches.Patch):
23:     '''an axis spine -- the line noting the data area boundaries
24: 
25:     Spines are the lines connecting the axis tick marks and noting the
26:     boundaries of the data area. They can be placed at arbitrary
27:     positions. See function:`~matplotlib.spines.Spine.set_position`
28:     for more information.
29: 
30:     The default position is ``('outward',0)``.
31: 
32:     Spines are subclasses of class:`~matplotlib.patches.Patch`, and
33:     inherit much of their behavior.
34: 
35:     Spines draw a line, a circle, or an arc depending if
36:     function:`~matplotlib.spines.Spine.set_patch_line`,
37:     function:`~matplotlib.spines.Spine.set_patch_circle`, or
38:     function:`~matplotlib.spines.Spine.set_patch_arc` has been called.
39:     Line-like is the default.
40: 
41:     '''
42:     def __str__(self):
43:         return "Spine"
44: 
45:     @docstring.dedent_interpd
46:     def __init__(self, axes, spine_type, path, **kwargs):
47:         '''
48:         - *axes* : the Axes instance containing the spine
49:         - *spine_type* : a string specifying the spine type
50:         - *path* : the path instance used to draw the spine
51: 
52:         Valid kwargs are:
53:         %(Patch)s
54:         '''
55:         super(Spine, self).__init__(**kwargs)
56:         self.axes = axes
57:         self.set_figure(self.axes.figure)
58:         self.spine_type = spine_type
59:         self.set_facecolor('none')
60:         self.set_edgecolor(rcParams['axes.edgecolor'])
61:         self.set_linewidth(rcParams['axes.linewidth'])
62:         self.set_capstyle('projecting')
63:         self.axis = None
64: 
65:         self.set_zorder(2.5)
66:         self.set_transform(self.axes.transData)  # default transform
67: 
68:         self._bounds = None  # default bounds
69:         self._smart_bounds = False
70: 
71:         # Defer initial position determination. (Not much support for
72:         # non-rectangular axes is currently implemented, and this lets
73:         # them pass through the spines machinery without errors.)
74:         self._position = None
75:         if not isinstance(path, matplotlib.path.Path):
76:             msg = "'path' must be an instance of 'matplotlib.path.Path'"
77:             raise ValueError(msg)
78:         self._path = path
79: 
80:         # To support drawing both linear and circular spines, this
81:         # class implements Patch behavior three ways. If
82:         # self._patch_type == 'line', behave like a mpatches.PathPatch
83:         # instance. If self._patch_type == 'circle', behave like a
84:         # mpatches.Ellipse instance. If self._patch_type == 'arc', behave like
85:         # a mpatches.Arc instance.
86:         self._patch_type = 'line'
87: 
88:         # Behavior copied from mpatches.Ellipse:
89:         # Note: This cannot be calculated until this is added to an Axes
90:         self._patch_transform = mtransforms.IdentityTransform()
91: 
92:     def set_smart_bounds(self, value):
93:         '''set the spine and associated axis to have smart bounds'''
94:         self._smart_bounds = value
95: 
96:         # also set the axis if possible
97:         if self.spine_type in ('left', 'right'):
98:             self.axes.yaxis.set_smart_bounds(value)
99:         elif self.spine_type in ('top', 'bottom'):
100:             self.axes.xaxis.set_smart_bounds(value)
101:         self.stale = True
102: 
103:     def get_smart_bounds(self):
104:         '''get whether the spine has smart bounds'''
105:         return self._smart_bounds
106: 
107:     def set_patch_arc(self, center, radius, theta1, theta2):
108:         '''set the spine to be arc-like'''
109:         self._patch_type = 'arc'
110:         self._center = center
111:         self._width = radius * 2
112:         self._height = radius * 2
113:         self._theta1 = theta1
114:         self._theta2 = theta2
115:         self._path = mpath.Path.arc(theta1, theta2)
116:         # arc drawn on axes transform
117:         self.set_transform(self.axes.transAxes)
118:         self.stale = True
119: 
120:     def set_patch_circle(self, center, radius):
121:         '''set the spine to be circular'''
122:         self._patch_type = 'circle'
123:         self._center = center
124:         self._width = radius * 2
125:         self._height = radius * 2
126:         # circle drawn on axes transform
127:         self.set_transform(self.axes.transAxes)
128:         self.stale = True
129: 
130:     def set_patch_line(self):
131:         '''set the spine to be linear'''
132:         self._patch_type = 'line'
133:         self.stale = True
134: 
135:     # Behavior copied from mpatches.Ellipse:
136:     def _recompute_transform(self):
137:         '''NOTE: This cannot be called until after this has been added
138:                  to an Axes, otherwise unit conversion will fail. This
139:                  maxes it very important to call the accessor method and
140:                  not directly access the transformation member variable.
141:         '''
142:         assert self._patch_type in ('arc', 'circle')
143:         center = (self.convert_xunits(self._center[0]),
144:                   self.convert_yunits(self._center[1]))
145:         width = self.convert_xunits(self._width)
146:         height = self.convert_yunits(self._height)
147:         self._patch_transform = mtransforms.Affine2D() \
148:             .scale(width * 0.5, height * 0.5) \
149:             .translate(*center)
150: 
151:     def get_patch_transform(self):
152:         if self._patch_type in ('arc', 'circle'):
153:             self._recompute_transform()
154:             return self._patch_transform
155:         else:
156:             return super(Spine, self).get_patch_transform()
157: 
158:     def get_path(self):
159:         return self._path
160: 
161:     def _ensure_position_is_set(self):
162:         if self._position is None:
163:             # default position
164:             self._position = ('outward', 0.0)  # in points
165:             self.set_position(self._position)
166: 
167:     def register_axis(self, axis):
168:         '''register an axis
169: 
170:         An axis should be registered with its corresponding spine from
171:         the Axes instance. This allows the spine to clear any axis
172:         properties when needed.
173:         '''
174:         self.axis = axis
175:         if self.axis is not None:
176:             self.axis.cla()
177:         self.stale = True
178: 
179:     def cla(self):
180:         '''Clear the current spine'''
181:         self._position = None  # clear position
182:         if self.axis is not None:
183:             self.axis.cla()
184: 
185:     def is_frame_like(self):
186:         '''return True if directly on axes frame
187: 
188:         This is useful for determining if a spine is the edge of an
189:         old style MPL plot. If so, this function will return True.
190:         '''
191:         self._ensure_position_is_set()
192:         position = self._position
193:         if isinstance(position, six.string_types):
194:             if position == 'center':
195:                 position = ('axes', 0.5)
196:             elif position == 'zero':
197:                 position = ('data', 0)
198:         if len(position) != 2:
199:             raise ValueError("position should be 2-tuple")
200:         position_type, amount = position
201:         if position_type == 'outward' and amount == 0:
202:             return True
203:         else:
204:             return False
205: 
206:     def _adjust_location(self):
207:         '''automatically set spine bounds to the view interval'''
208: 
209:         if self.spine_type == 'circle':
210:             return
211: 
212:         if self._bounds is None:
213:             if self.spine_type in ('left', 'right'):
214:                 low, high = self.axes.viewLim.intervaly
215:             elif self.spine_type in ('top', 'bottom'):
216:                 low, high = self.axes.viewLim.intervalx
217:             else:
218:                 raise ValueError('unknown spine spine_type: %s' %
219:                                  self.spine_type)
220: 
221:             if self._smart_bounds:
222:                 # attempt to set bounds in sophisticated way
223: 
224:                 # handle inverted limits
225:                 viewlim_low, viewlim_high = sorted([low, high])
226: 
227:                 if self.spine_type in ('left', 'right'):
228:                     datalim_low, datalim_high = self.axes.dataLim.intervaly
229:                     ticks = self.axes.get_yticks()
230:                 elif self.spine_type in ('top', 'bottom'):
231:                     datalim_low, datalim_high = self.axes.dataLim.intervalx
232:                     ticks = self.axes.get_xticks()
233:                 # handle inverted limits
234:                 ticks = np.sort(ticks)
235:                 datalim_low, datalim_high = sorted([datalim_low, datalim_high])
236: 
237:                 if datalim_low < viewlim_low:
238:                     # Data extends past view. Clip line to view.
239:                     low = viewlim_low
240:                 else:
241:                     # Data ends before view ends.
242:                     cond = (ticks <= datalim_low) & (ticks >= viewlim_low)
243:                     tickvals = ticks[cond]
244:                     if len(tickvals):
245:                         # A tick is less than or equal to lowest data point.
246:                         low = tickvals[-1]
247:                     else:
248:                         # No tick is available
249:                         low = datalim_low
250:                     low = max(low, viewlim_low)
251: 
252:                 if datalim_high > viewlim_high:
253:                     # Data extends past view. Clip line to view.
254:                     high = viewlim_high
255:                 else:
256:                     # Data ends before view ends.
257:                     cond = (ticks >= datalim_high) & (ticks <= viewlim_high)
258:                     tickvals = ticks[cond]
259:                     if len(tickvals):
260:                         # A tick is greater than or equal to highest data
261:                         # point.
262:                         high = tickvals[0]
263:                     else:
264:                         # No tick is available
265:                         high = datalim_high
266:                     high = min(high, viewlim_high)
267: 
268:         else:
269:             low, high = self._bounds
270: 
271:         if self._patch_type == 'arc':
272:             if self.spine_type in ('bottom', 'top'):
273:                 try:
274:                     direction = self.axes.get_theta_direction()
275:                 except AttributeError:
276:                     direction = 1
277:                 try:
278:                     offset = self.axes.get_theta_offset()
279:                 except AttributeError:
280:                     offset = 0
281:                 low = low * direction + offset
282:                 high = high * direction + offset
283:                 if low > high:
284:                     low, high = high, low
285: 
286:                 self._path = mpath.Path.arc(np.rad2deg(low), np.rad2deg(high))
287: 
288:                 if self.spine_type == 'bottom':
289:                     rmin, rmax = self.axes.viewLim.intervaly
290:                     try:
291:                         rorigin = self.axes.get_rorigin()
292:                     except AttributeError:
293:                         rorigin = rmin
294:                     scaled_diameter = (rmin - rorigin) / (rmax - rorigin)
295:                     self._height = scaled_diameter
296:                     self._width = scaled_diameter
297: 
298:             else:
299:                 raise ValueError('unable to set bounds for spine "%s"' %
300:                                  self.spine_type)
301:         else:
302:             v1 = self._path.vertices
303:             assert v1.shape == (2, 2), 'unexpected vertices shape'
304:             if self.spine_type in ['left', 'right']:
305:                 v1[0, 1] = low
306:                 v1[1, 1] = high
307:             elif self.spine_type in ['bottom', 'top']:
308:                 v1[0, 0] = low
309:                 v1[1, 0] = high
310:             else:
311:                 raise ValueError('unable to set bounds for spine "%s"' %
312:                                  self.spine_type)
313: 
314:     @allow_rasterization
315:     def draw(self, renderer):
316:         self._adjust_location()
317:         ret = super(Spine, self).draw(renderer)
318:         self.stale = False
319:         return ret
320: 
321:     def _calc_offset_transform(self):
322:         '''calculate the offset transform performed by the spine'''
323:         self._ensure_position_is_set()
324:         position = self._position
325:         if isinstance(position, six.string_types):
326:             if position == 'center':
327:                 position = ('axes', 0.5)
328:             elif position == 'zero':
329:                 position = ('data', 0)
330:         assert len(position) == 2, "position should be 2-tuple"
331:         position_type, amount = position
332:         assert position_type in ('axes', 'outward', 'data')
333:         if position_type == 'outward':
334:             if amount == 0:
335:                 # short circuit commonest case
336:                 self._spine_transform = ('identity',
337:                                          mtransforms.IdentityTransform())
338:             elif self.spine_type in ['left', 'right', 'top', 'bottom']:
339:                 offset_vec = {'left': (-1, 0),
340:                               'right': (1, 0),
341:                               'bottom': (0, -1),
342:                               'top': (0, 1),
343:                               }[self.spine_type]
344:                 # calculate x and y offset in dots
345:                 offset_x = amount * offset_vec[0] / 72.0
346:                 offset_y = amount * offset_vec[1] / 72.0
347:                 self._spine_transform = ('post',
348:                                          mtransforms.ScaledTranslation(
349:                                              offset_x,
350:                                              offset_y,
351:                                              self.figure.dpi_scale_trans))
352:             else:
353:                 warnings.warn('unknown spine type "%s": no spine '
354:                               'offset performed' % self.spine_type)
355:                 self._spine_transform = ('identity',
356:                                          mtransforms.IdentityTransform())
357:         elif position_type == 'axes':
358:             if self.spine_type in ('left', 'right'):
359:                 self._spine_transform = ('pre',
360:                                          mtransforms.Affine2D.from_values(
361:                                              # keep y unchanged, fix x at
362:                                              # amount
363:                                              0, 0, 0, 1, amount, 0))
364:             elif self.spine_type in ('bottom', 'top'):
365:                 self._spine_transform = ('pre',
366:                                          mtransforms.Affine2D.from_values(
367:                                              # keep x unchanged, fix y at
368:                                              # amount
369:                                              1, 0, 0, 0, 0, amount))
370:             else:
371:                 warnings.warn('unknown spine type "%s": no spine '
372:                               'offset performed' % self.spine_type)
373:                 self._spine_transform = ('identity',
374:                                          mtransforms.IdentityTransform())
375:         elif position_type == 'data':
376:             if self.spine_type in ('right', 'top'):
377:                 # The right and top spines have a default position of 1 in
378:                 # axes coordinates.  When specifying the position in data
379:                 # coordinates, we need to calculate the position relative to 0.
380:                 amount -= 1
381:             if self.spine_type in ('left', 'right'):
382:                 self._spine_transform = ('data',
383:                                          mtransforms.Affine2D().translate(
384:                                              amount, 0))
385:             elif self.spine_type in ('bottom', 'top'):
386:                 self._spine_transform = ('data',
387:                                          mtransforms.Affine2D().translate(
388:                                              0, amount))
389:             else:
390:                 warnings.warn('unknown spine type "%s": no spine '
391:                               'offset performed' % self.spine_type)
392:                 self._spine_transform = ('identity',
393:                                          mtransforms.IdentityTransform())
394: 
395:     def set_position(self, position):
396:         '''set the position of the spine
397: 
398:         Spine position is specified by a 2 tuple of (position type,
399:         amount). The position types are:
400: 
401:         * 'outward' : place the spine out from the data area by the
402:           specified number of points. (Negative values specify placing the
403:           spine inward.)
404: 
405:         * 'axes' : place the spine at the specified Axes coordinate (from
406:           0.0-1.0).
407: 
408:         * 'data' : place the spine at the specified data coordinate.
409: 
410:         Additionally, shorthand notations define a special positions:
411: 
412:         * 'center' -> ('axes',0.5)
413:         * 'zero' -> ('data', 0.0)
414: 
415:         '''
416:         if position in ('center', 'zero'):
417:             # special positions
418:             pass
419:         else:
420:             if len(position) != 2:
421:                 raise ValueError("position should be 'center' or 2-tuple")
422:             if position[0] not in ['outward', 'axes', 'data']:
423:                 msg = ("position[0] should be in [ 'outward' | 'axes' |"
424:                        " 'data' ]")
425:                 raise ValueError(msg)
426:         self._position = position
427:         self._calc_offset_transform()
428: 
429:         self.set_transform(self.get_spine_transform())
430: 
431:         if self.axis is not None:
432:             self.axis.reset_ticks()
433:         self.stale = True
434: 
435:     def get_position(self):
436:         '''get the spine position'''
437:         self._ensure_position_is_set()
438:         return self._position
439: 
440:     def get_spine_transform(self):
441:         '''get the spine transform'''
442:         self._ensure_position_is_set()
443:         what, how = self._spine_transform
444: 
445:         if what == 'data':
446:             # special case data based spine locations
447:             data_xform = self.axes.transScale + \
448:                 (how + self.axes.transLimits + self.axes.transAxes)
449:             if self.spine_type in ['left', 'right']:
450:                 result = mtransforms.blended_transform_factory(
451:                     data_xform, self.axes.transData)
452:             elif self.spine_type in ['top', 'bottom']:
453:                 result = mtransforms.blended_transform_factory(
454:                     self.axes.transData, data_xform)
455:             else:
456:                 raise ValueError('unknown spine spine_type: %s' %
457:                                  self.spine_type)
458:             return result
459: 
460:         if self.spine_type in ['left', 'right']:
461:             base_transform = self.axes.get_yaxis_transform(which='grid')
462:         elif self.spine_type in ['top', 'bottom']:
463:             base_transform = self.axes.get_xaxis_transform(which='grid')
464:         else:
465:             raise ValueError('unknown spine spine_type: %s' %
466:                              self.spine_type)
467: 
468:         if what == 'identity':
469:             return base_transform
470:         elif what == 'post':
471:             return base_transform + how
472:         elif what == 'pre':
473:             return how + base_transform
474:         else:
475:             raise ValueError("unknown spine_transform type: %s" % what)
476: 
477:     def set_bounds(self, low, high):
478:         '''Set the bounds of the spine.'''
479:         if self.spine_type == 'circle':
480:             raise ValueError(
481:                 'set_bounds() method incompatible with circular spines')
482:         self._bounds = (low, high)
483:         self.stale = True
484: 
485:     def get_bounds(self):
486:         '''Get the bounds of the spine.'''
487:         return self._bounds
488: 
489:     @classmethod
490:     def linear_spine(cls, axes, spine_type, **kwargs):
491:         '''
492:         (staticmethod) Returns a linear :class:`Spine`.
493:         '''
494:         # all values of 13 get replaced upon call to set_bounds()
495:         if spine_type == 'left':
496:             path = mpath.Path([(0.0, 13), (0.0, 13)])
497:         elif spine_type == 'right':
498:             path = mpath.Path([(1.0, 13), (1.0, 13)])
499:         elif spine_type == 'bottom':
500:             path = mpath.Path([(13, 0.0), (13, 0.0)])
501:         elif spine_type == 'top':
502:             path = mpath.Path([(13, 1.0), (13, 1.0)])
503:         else:
504:             raise ValueError('unable to make path for spine "%s"' % spine_type)
505:         result = cls(axes, spine_type, path, **kwargs)
506:         result.set_visible(rcParams['axes.spines.{0}'.format(spine_type)])
507: 
508:         return result
509: 
510:     @classmethod
511:     def arc_spine(cls, axes, spine_type, center, radius, theta1, theta2,
512:                   **kwargs):
513:         '''
514:         (classmethod) Returns an arc :class:`Spine`.
515:         '''
516:         path = mpath.Path.arc(theta1, theta2)
517:         result = cls(axes, spine_type, path, **kwargs)
518:         result.set_patch_arc(center, radius, theta1, theta2)
519:         return result
520: 
521:     @classmethod
522:     def circular_spine(cls, axes, center, radius, **kwargs):
523:         '''
524:         (staticmethod) Returns a circular :class:`Spine`.
525:         '''
526:         path = mpath.Path.unit_circle()
527:         spine_type = 'circle'
528:         result = cls(axes, spine_type, path, **kwargs)
529:         result.set_patch_circle(center, radius)
530:         return result
531: 
532:     def set_color(self, c):
533:         '''
534:         Set the edgecolor.
535: 
536:         ACCEPTS: matplotlib color arg or sequence of rgba tuples
537: 
538:         .. seealso::
539: 
540:             :meth:`set_facecolor`, :meth:`set_edgecolor`
541:                For setting the edge or face color individually.
542:         '''
543:         # The facecolor of a spine is always 'none' by default -- let
544:         # the user change it manually if desired.
545:         self.set_edgecolor(c)
546:         self.stale = True
547: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131403 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_131403) is not StypyTypeError):

    if (import_131403 != 'pyd_module'):
        __import__(import_131403)
        sys_modules_131404 = sys.modules[import_131403]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_131404.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_131403)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import matplotlib' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131405 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib')

if (type(import_131405) is not StypyTypeError):

    if (import_131405 != 'pyd_module'):
        __import__(import_131405)
        sys_modules_131406 = sys.modules[import_131405]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', sys_modules_131406.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', import_131405)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import matplotlib.artist' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131407 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.artist')

if (type(import_131407) is not StypyTypeError):

    if (import_131407 != 'pyd_module'):
        __import__(import_131407)
        sys_modules_131408 = sys.modules[import_131407]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'martist', sys_modules_131408.module_type_store, module_type_store)
    else:
        import matplotlib.artist as martist

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'martist', matplotlib.artist, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.artist' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.artist', import_131407)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.artist import allow_rasterization' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131409 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.artist')

if (type(import_131409) is not StypyTypeError):

    if (import_131409 != 'pyd_module'):
        __import__(import_131409)
        sys_modules_131410 = sys.modules[import_131409]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.artist', sys_modules_131410.module_type_store, module_type_store, ['allow_rasterization'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_131410, sys_modules_131410.module_type_store, module_type_store)
    else:
        from matplotlib.artist import allow_rasterization

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.artist', None, module_type_store, ['allow_rasterization'], [allow_rasterization])

else:
    # Assigning a type to the variable 'matplotlib.artist' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.artist', import_131409)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib import docstring' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131411 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib')

if (type(import_131411) is not StypyTypeError):

    if (import_131411 != 'pyd_module'):
        __import__(import_131411)
        sys_modules_131412 = sys.modules[import_131411]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', sys_modules_131412.module_type_store, module_type_store, ['docstring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_131412, sys_modules_131412.module_type_store, module_type_store)
    else:
        from matplotlib import docstring

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', None, module_type_store, ['docstring'], [docstring])

else:
    # Assigning a type to the variable 'matplotlib' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', import_131411)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import matplotlib.transforms' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131413 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.transforms')

if (type(import_131413) is not StypyTypeError):

    if (import_131413 != 'pyd_module'):
        __import__(import_131413)
        sys_modules_131414 = sys.modules[import_131413]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'mtransforms', sys_modules_131414.module_type_store, module_type_store)
    else:
        import matplotlib.transforms as mtransforms

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'mtransforms', matplotlib.transforms, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.transforms', import_131413)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import matplotlib.lines' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131415 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.lines')

if (type(import_131415) is not StypyTypeError):

    if (import_131415 != 'pyd_module'):
        __import__(import_131415)
        sys_modules_131416 = sys.modules[import_131415]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'mlines', sys_modules_131416.module_type_store, module_type_store)
    else:
        import matplotlib.lines as mlines

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'mlines', matplotlib.lines, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.lines' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.lines', import_131415)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import matplotlib.patches' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131417 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.patches')

if (type(import_131417) is not StypyTypeError):

    if (import_131417 != 'pyd_module'):
        __import__(import_131417)
        sys_modules_131418 = sys.modules[import_131417]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'mpatches', sys_modules_131418.module_type_store, module_type_store)
    else:
        import matplotlib.patches as mpatches

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'mpatches', matplotlib.patches, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.patches', import_131417)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import matplotlib.path' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131419 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.path')

if (type(import_131419) is not StypyTypeError):

    if (import_131419 != 'pyd_module'):
        __import__(import_131419)
        sys_modules_131420 = sys.modules[import_131419]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'mpath', sys_modules_131420.module_type_store, module_type_store)
    else:
        import matplotlib.path as mpath

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'mpath', matplotlib.path, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.path' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.path', import_131419)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import matplotlib.cbook' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131421 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.cbook')

if (type(import_131421) is not StypyTypeError):

    if (import_131421 != 'pyd_module'):
        __import__(import_131421)
        sys_modules_131422 = sys.modules[import_131421]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'cbook', sys_modules_131422.module_type_store, module_type_store)
    else:
        import matplotlib.cbook as cbook

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'cbook', matplotlib.cbook, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.cbook', import_131421)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import numpy' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_131423 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy')

if (type(import_131423) is not StypyTypeError):

    if (import_131423 != 'pyd_module'):
        __import__(import_131423)
        sys_modules_131424 = sys.modules[import_131423]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'np', sys_modules_131424.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy', import_131423)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import warnings' statement (line 17)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'warnings', warnings, module_type_store)


# Assigning a Attribute to a Name (line 19):

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'matplotlib' (line 19)
matplotlib_131425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'matplotlib')
# Obtaining the member 'rcParams' of a type (line 19)
rcParams_131426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 11), matplotlib_131425, 'rcParams')
# Assigning a type to the variable 'rcParams' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'rcParams', rcParams_131426)
# Declaration of the 'Spine' class
# Getting the type of 'mpatches' (line 22)
mpatches_131427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'mpatches')
# Obtaining the member 'Patch' of a type (line 22)
Patch_131428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), mpatches_131427, 'Patch')

class Spine(Patch_131428, ):
    unicode_131429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'unicode', u"an axis spine -- the line noting the data area boundaries\n\n    Spines are the lines connecting the axis tick marks and noting the\n    boundaries of the data area. They can be placed at arbitrary\n    positions. See function:`~matplotlib.spines.Spine.set_position`\n    for more information.\n\n    The default position is ``('outward',0)``.\n\n    Spines are subclasses of class:`~matplotlib.patches.Patch`, and\n    inherit much of their behavior.\n\n    Spines draw a line, a circle, or an arc depending if\n    function:`~matplotlib.spines.Spine.set_patch_line`,\n    function:`~matplotlib.spines.Spine.set_patch_circle`, or\n    function:`~matplotlib.spines.Spine.set_patch_arc` has been called.\n    Line-like is the default.\n\n    ")

    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Spine.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Spine.stypy__str__')
        Spine.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        unicode_131430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'unicode', u'Spine')
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', unicode_131430)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_131431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131431)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_131431


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.__init__', ['axes', 'spine_type', 'path'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['axes', 'spine_type', 'path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_131432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'unicode', u'\n        - *axes* : the Axes instance containing the spine\n        - *spine_type* : a string specifying the spine type\n        - *path* : the path instance used to draw the spine\n\n        Valid kwargs are:\n        %(Patch)s\n        ')
        
        # Call to __init__(...): (line 55)
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'kwargs' (line 55)
        kwargs_131439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 38), 'kwargs', False)
        kwargs_131440 = {'kwargs_131439': kwargs_131439}
        
        # Call to super(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'Spine' (line 55)
        Spine_131434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 14), 'Spine', False)
        # Getting the type of 'self' (line 55)
        self_131435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'self', False)
        # Processing the call keyword arguments (line 55)
        kwargs_131436 = {}
        # Getting the type of 'super' (line 55)
        super_131433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'super', False)
        # Calling super(args, kwargs) (line 55)
        super_call_result_131437 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), super_131433, *[Spine_131434, self_131435], **kwargs_131436)
        
        # Obtaining the member '__init__' of a type (line 55)
        init___131438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), super_call_result_131437, '__init__')
        # Calling __init__(args, kwargs) (line 55)
        init___call_result_131441 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), init___131438, *[], **kwargs_131440)
        
        
        # Assigning a Name to a Attribute (line 56):
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'axes' (line 56)
        axes_131442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'axes')
        # Getting the type of 'self' (line 56)
        self_131443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'axes' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_131443, 'axes', axes_131442)
        
        # Call to set_figure(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'self' (line 57)
        self_131446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'self', False)
        # Obtaining the member 'axes' of a type (line 57)
        axes_131447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 24), self_131446, 'axes')
        # Obtaining the member 'figure' of a type (line 57)
        figure_131448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 24), axes_131447, 'figure')
        # Processing the call keyword arguments (line 57)
        kwargs_131449 = {}
        # Getting the type of 'self' (line 57)
        self_131444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'set_figure' of a type (line 57)
        set_figure_131445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_131444, 'set_figure')
        # Calling set_figure(args, kwargs) (line 57)
        set_figure_call_result_131450 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), set_figure_131445, *[figure_131448], **kwargs_131449)
        
        
        # Assigning a Name to a Attribute (line 58):
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'spine_type' (line 58)
        spine_type_131451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'spine_type')
        # Getting the type of 'self' (line 58)
        self_131452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'spine_type' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_131452, 'spine_type', spine_type_131451)
        
        # Call to set_facecolor(...): (line 59)
        # Processing the call arguments (line 59)
        unicode_131455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'unicode', u'none')
        # Processing the call keyword arguments (line 59)
        kwargs_131456 = {}
        # Getting the type of 'self' (line 59)
        self_131453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self', False)
        # Obtaining the member 'set_facecolor' of a type (line 59)
        set_facecolor_131454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_131453, 'set_facecolor')
        # Calling set_facecolor(args, kwargs) (line 59)
        set_facecolor_call_result_131457 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), set_facecolor_131454, *[unicode_131455], **kwargs_131456)
        
        
        # Call to set_edgecolor(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining the type of the subscript
        unicode_131460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'unicode', u'axes.edgecolor')
        # Getting the type of 'rcParams' (line 60)
        rcParams_131461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___131462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 27), rcParams_131461, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_131463 = invoke(stypy.reporting.localization.Localization(__file__, 60, 27), getitem___131462, unicode_131460)
        
        # Processing the call keyword arguments (line 60)
        kwargs_131464 = {}
        # Getting the type of 'self' (line 60)
        self_131458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'set_edgecolor' of a type (line 60)
        set_edgecolor_131459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_131458, 'set_edgecolor')
        # Calling set_edgecolor(args, kwargs) (line 60)
        set_edgecolor_call_result_131465 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), set_edgecolor_131459, *[subscript_call_result_131463], **kwargs_131464)
        
        
        # Call to set_linewidth(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining the type of the subscript
        unicode_131468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 36), 'unicode', u'axes.linewidth')
        # Getting the type of 'rcParams' (line 61)
        rcParams_131469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___131470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 27), rcParams_131469, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_131471 = invoke(stypy.reporting.localization.Localization(__file__, 61, 27), getitem___131470, unicode_131468)
        
        # Processing the call keyword arguments (line 61)
        kwargs_131472 = {}
        # Getting the type of 'self' (line 61)
        self_131466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'set_linewidth' of a type (line 61)
        set_linewidth_131467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_131466, 'set_linewidth')
        # Calling set_linewidth(args, kwargs) (line 61)
        set_linewidth_call_result_131473 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), set_linewidth_131467, *[subscript_call_result_131471], **kwargs_131472)
        
        
        # Call to set_capstyle(...): (line 62)
        # Processing the call arguments (line 62)
        unicode_131476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'unicode', u'projecting')
        # Processing the call keyword arguments (line 62)
        kwargs_131477 = {}
        # Getting the type of 'self' (line 62)
        self_131474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member 'set_capstyle' of a type (line 62)
        set_capstyle_131475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_131474, 'set_capstyle')
        # Calling set_capstyle(args, kwargs) (line 62)
        set_capstyle_call_result_131478 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), set_capstyle_131475, *[unicode_131476], **kwargs_131477)
        
        
        # Assigning a Name to a Attribute (line 63):
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'None' (line 63)
        None_131479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'None')
        # Getting the type of 'self' (line 63)
        self_131480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_131480, 'axis', None_131479)
        
        # Call to set_zorder(...): (line 65)
        # Processing the call arguments (line 65)
        float_131483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'float')
        # Processing the call keyword arguments (line 65)
        kwargs_131484 = {}
        # Getting the type of 'self' (line 65)
        self_131481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', False)
        # Obtaining the member 'set_zorder' of a type (line 65)
        set_zorder_131482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_131481, 'set_zorder')
        # Calling set_zorder(args, kwargs) (line 65)
        set_zorder_call_result_131485 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), set_zorder_131482, *[float_131483], **kwargs_131484)
        
        
        # Call to set_transform(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'self' (line 66)
        self_131488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 27), 'self', False)
        # Obtaining the member 'axes' of a type (line 66)
        axes_131489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 27), self_131488, 'axes')
        # Obtaining the member 'transData' of a type (line 66)
        transData_131490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 27), axes_131489, 'transData')
        # Processing the call keyword arguments (line 66)
        kwargs_131491 = {}
        # Getting the type of 'self' (line 66)
        self_131486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 66)
        set_transform_131487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_131486, 'set_transform')
        # Calling set_transform(args, kwargs) (line 66)
        set_transform_call_result_131492 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), set_transform_131487, *[transData_131490], **kwargs_131491)
        
        
        # Assigning a Name to a Attribute (line 68):
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'None' (line 68)
        None_131493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'None')
        # Getting the type of 'self' (line 68)
        self_131494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member '_bounds' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_131494, '_bounds', None_131493)
        
        # Assigning a Name to a Attribute (line 69):
        
        # Assigning a Name to a Attribute (line 69):
        # Getting the type of 'False' (line 69)
        False_131495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'False')
        # Getting the type of 'self' (line 69)
        self_131496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member '_smart_bounds' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_131496, '_smart_bounds', False_131495)
        
        # Assigning a Name to a Attribute (line 74):
        
        # Assigning a Name to a Attribute (line 74):
        # Getting the type of 'None' (line 74)
        None_131497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'None')
        # Getting the type of 'self' (line 74)
        self_131498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member '_position' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_131498, '_position', None_131497)
        
        
        
        # Call to isinstance(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'path' (line 75)
        path_131500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'path', False)
        # Getting the type of 'matplotlib' (line 75)
        matplotlib_131501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'matplotlib', False)
        # Obtaining the member 'path' of a type (line 75)
        path_131502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), matplotlib_131501, 'path')
        # Obtaining the member 'Path' of a type (line 75)
        Path_131503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), path_131502, 'Path')
        # Processing the call keyword arguments (line 75)
        kwargs_131504 = {}
        # Getting the type of 'isinstance' (line 75)
        isinstance_131499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 75)
        isinstance_call_result_131505 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), isinstance_131499, *[path_131500, Path_131503], **kwargs_131504)
        
        # Applying the 'not' unary operator (line 75)
        result_not__131506 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), 'not', isinstance_call_result_131505)
        
        # Testing the type of an if condition (line 75)
        if_condition_131507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_not__131506)
        # Assigning a type to the variable 'if_condition_131507' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_131507', if_condition_131507)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 76):
        
        # Assigning a Str to a Name (line 76):
        unicode_131508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 18), 'unicode', u"'path' must be an instance of 'matplotlib.path.Path'")
        # Assigning a type to the variable 'msg' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'msg', unicode_131508)
        
        # Call to ValueError(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'msg' (line 77)
        msg_131510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'msg', False)
        # Processing the call keyword arguments (line 77)
        kwargs_131511 = {}
        # Getting the type of 'ValueError' (line 77)
        ValueError_131509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 77)
        ValueError_call_result_131512 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), ValueError_131509, *[msg_131510], **kwargs_131511)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 12), ValueError_call_result_131512, 'raise parameter', BaseException)
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'path' (line 78)
        path_131513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'path')
        # Getting the type of 'self' (line 78)
        self_131514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member '_path' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_131514, '_path', path_131513)
        
        # Assigning a Str to a Attribute (line 86):
        
        # Assigning a Str to a Attribute (line 86):
        unicode_131515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'unicode', u'line')
        # Getting the type of 'self' (line 86)
        self_131516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self')
        # Setting the type of the member '_patch_type' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_131516, '_patch_type', unicode_131515)
        
        # Assigning a Call to a Attribute (line 90):
        
        # Assigning a Call to a Attribute (line 90):
        
        # Call to IdentityTransform(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_131519 = {}
        # Getting the type of 'mtransforms' (line 90)
        mtransforms_131517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'mtransforms', False)
        # Obtaining the member 'IdentityTransform' of a type (line 90)
        IdentityTransform_131518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 32), mtransforms_131517, 'IdentityTransform')
        # Calling IdentityTransform(args, kwargs) (line 90)
        IdentityTransform_call_result_131520 = invoke(stypy.reporting.localization.Localization(__file__, 90, 32), IdentityTransform_131518, *[], **kwargs_131519)
        
        # Getting the type of 'self' (line 90)
        self_131521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member '_patch_transform' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_131521, '_patch_transform', IdentityTransform_call_result_131520)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_smart_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_smart_bounds'
        module_type_store = module_type_store.open_function_context('set_smart_bounds', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_localization', localization)
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_function_name', 'Spine.set_smart_bounds')
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_param_names_list', ['value'])
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.set_smart_bounds.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.set_smart_bounds', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_smart_bounds', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_smart_bounds(...)' code ##################

        unicode_131522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'unicode', u'set the spine and associated axis to have smart bounds')
        
        # Assigning a Name to a Attribute (line 94):
        
        # Assigning a Name to a Attribute (line 94):
        # Getting the type of 'value' (line 94)
        value_131523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 29), 'value')
        # Getting the type of 'self' (line 94)
        self_131524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member '_smart_bounds' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_131524, '_smart_bounds', value_131523)
        
        
        # Getting the type of 'self' (line 97)
        self_131525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'self')
        # Obtaining the member 'spine_type' of a type (line 97)
        spine_type_131526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), self_131525, 'spine_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_131527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        # Adding element type (line 97)
        unicode_131528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 31), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 31), tuple_131527, unicode_131528)
        # Adding element type (line 97)
        unicode_131529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 39), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 31), tuple_131527, unicode_131529)
        
        # Applying the binary operator 'in' (line 97)
        result_contains_131530 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), 'in', spine_type_131526, tuple_131527)
        
        # Testing the type of an if condition (line 97)
        if_condition_131531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_contains_131530)
        # Assigning a type to the variable 'if_condition_131531' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_131531', if_condition_131531)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_smart_bounds(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'value' (line 98)
        value_131536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 45), 'value', False)
        # Processing the call keyword arguments (line 98)
        kwargs_131537 = {}
        # Getting the type of 'self' (line 98)
        self_131532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self', False)
        # Obtaining the member 'axes' of a type (line 98)
        axes_131533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_131532, 'axes')
        # Obtaining the member 'yaxis' of a type (line 98)
        yaxis_131534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), axes_131533, 'yaxis')
        # Obtaining the member 'set_smart_bounds' of a type (line 98)
        set_smart_bounds_131535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), yaxis_131534, 'set_smart_bounds')
        # Calling set_smart_bounds(args, kwargs) (line 98)
        set_smart_bounds_call_result_131538 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), set_smart_bounds_131535, *[value_131536], **kwargs_131537)
        
        # SSA branch for the else part of an if statement (line 97)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 99)
        self_131539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'self')
        # Obtaining the member 'spine_type' of a type (line 99)
        spine_type_131540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), self_131539, 'spine_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_131541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        unicode_131542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 33), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 33), tuple_131541, unicode_131542)
        # Adding element type (line 99)
        unicode_131543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 33), tuple_131541, unicode_131543)
        
        # Applying the binary operator 'in' (line 99)
        result_contains_131544 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 13), 'in', spine_type_131540, tuple_131541)
        
        # Testing the type of an if condition (line 99)
        if_condition_131545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 13), result_contains_131544)
        # Assigning a type to the variable 'if_condition_131545' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'if_condition_131545', if_condition_131545)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_smart_bounds(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'value' (line 100)
        value_131550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 45), 'value', False)
        # Processing the call keyword arguments (line 100)
        kwargs_131551 = {}
        # Getting the type of 'self' (line 100)
        self_131546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self', False)
        # Obtaining the member 'axes' of a type (line 100)
        axes_131547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_131546, 'axes')
        # Obtaining the member 'xaxis' of a type (line 100)
        xaxis_131548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), axes_131547, 'xaxis')
        # Obtaining the member 'set_smart_bounds' of a type (line 100)
        set_smart_bounds_131549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), xaxis_131548, 'set_smart_bounds')
        # Calling set_smart_bounds(args, kwargs) (line 100)
        set_smart_bounds_call_result_131552 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), set_smart_bounds_131549, *[value_131550], **kwargs_131551)
        
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 101):
        
        # Assigning a Name to a Attribute (line 101):
        # Getting the type of 'True' (line 101)
        True_131553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'True')
        # Getting the type of 'self' (line 101)
        self_131554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_131554, 'stale', True_131553)
        
        # ################# End of 'set_smart_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_smart_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_131555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131555)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_smart_bounds'
        return stypy_return_type_131555


    @norecursion
    def get_smart_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_smart_bounds'
        module_type_store = module_type_store.open_function_context('get_smart_bounds', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_localization', localization)
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_function_name', 'Spine.get_smart_bounds')
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.get_smart_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.get_smart_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_smart_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_smart_bounds(...)' code ##################

        unicode_131556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'unicode', u'get whether the spine has smart bounds')
        # Getting the type of 'self' (line 105)
        self_131557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'self')
        # Obtaining the member '_smart_bounds' of a type (line 105)
        _smart_bounds_131558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), self_131557, '_smart_bounds')
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', _smart_bounds_131558)
        
        # ################# End of 'get_smart_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_smart_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_131559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131559)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_smart_bounds'
        return stypy_return_type_131559


    @norecursion
    def set_patch_arc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_patch_arc'
        module_type_store = module_type_store.open_function_context('set_patch_arc', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.set_patch_arc.__dict__.__setitem__('stypy_localization', localization)
        Spine.set_patch_arc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.set_patch_arc.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.set_patch_arc.__dict__.__setitem__('stypy_function_name', 'Spine.set_patch_arc')
        Spine.set_patch_arc.__dict__.__setitem__('stypy_param_names_list', ['center', 'radius', 'theta1', 'theta2'])
        Spine.set_patch_arc.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.set_patch_arc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.set_patch_arc.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.set_patch_arc.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.set_patch_arc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.set_patch_arc.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.set_patch_arc', ['center', 'radius', 'theta1', 'theta2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_patch_arc', localization, ['center', 'radius', 'theta1', 'theta2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_patch_arc(...)' code ##################

        unicode_131560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'unicode', u'set the spine to be arc-like')
        
        # Assigning a Str to a Attribute (line 109):
        
        # Assigning a Str to a Attribute (line 109):
        unicode_131561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 27), 'unicode', u'arc')
        # Getting the type of 'self' (line 109)
        self_131562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member '_patch_type' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_131562, '_patch_type', unicode_131561)
        
        # Assigning a Name to a Attribute (line 110):
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'center' (line 110)
        center_131563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'center')
        # Getting the type of 'self' (line 110)
        self_131564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member '_center' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_131564, '_center', center_131563)
        
        # Assigning a BinOp to a Attribute (line 111):
        
        # Assigning a BinOp to a Attribute (line 111):
        # Getting the type of 'radius' (line 111)
        radius_131565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'radius')
        int_131566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 31), 'int')
        # Applying the binary operator '*' (line 111)
        result_mul_131567 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 22), '*', radius_131565, int_131566)
        
        # Getting the type of 'self' (line 111)
        self_131568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member '_width' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_131568, '_width', result_mul_131567)
        
        # Assigning a BinOp to a Attribute (line 112):
        
        # Assigning a BinOp to a Attribute (line 112):
        # Getting the type of 'radius' (line 112)
        radius_131569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'radius')
        int_131570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'int')
        # Applying the binary operator '*' (line 112)
        result_mul_131571 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 23), '*', radius_131569, int_131570)
        
        # Getting the type of 'self' (line 112)
        self_131572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member '_height' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_131572, '_height', result_mul_131571)
        
        # Assigning a Name to a Attribute (line 113):
        
        # Assigning a Name to a Attribute (line 113):
        # Getting the type of 'theta1' (line 113)
        theta1_131573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'theta1')
        # Getting the type of 'self' (line 113)
        self_131574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member '_theta1' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_131574, '_theta1', theta1_131573)
        
        # Assigning a Name to a Attribute (line 114):
        
        # Assigning a Name to a Attribute (line 114):
        # Getting the type of 'theta2' (line 114)
        theta2_131575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'theta2')
        # Getting the type of 'self' (line 114)
        self_131576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member '_theta2' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_131576, '_theta2', theta2_131575)
        
        # Assigning a Call to a Attribute (line 115):
        
        # Assigning a Call to a Attribute (line 115):
        
        # Call to arc(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'theta1' (line 115)
        theta1_131580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'theta1', False)
        # Getting the type of 'theta2' (line 115)
        theta2_131581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 44), 'theta2', False)
        # Processing the call keyword arguments (line 115)
        kwargs_131582 = {}
        # Getting the type of 'mpath' (line 115)
        mpath_131577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'mpath', False)
        # Obtaining the member 'Path' of a type (line 115)
        Path_131578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 21), mpath_131577, 'Path')
        # Obtaining the member 'arc' of a type (line 115)
        arc_131579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 21), Path_131578, 'arc')
        # Calling arc(args, kwargs) (line 115)
        arc_call_result_131583 = invoke(stypy.reporting.localization.Localization(__file__, 115, 21), arc_131579, *[theta1_131580, theta2_131581], **kwargs_131582)
        
        # Getting the type of 'self' (line 115)
        self_131584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'self')
        # Setting the type of the member '_path' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), self_131584, '_path', arc_call_result_131583)
        
        # Call to set_transform(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_131587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'self', False)
        # Obtaining the member 'axes' of a type (line 117)
        axes_131588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 27), self_131587, 'axes')
        # Obtaining the member 'transAxes' of a type (line 117)
        transAxes_131589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 27), axes_131588, 'transAxes')
        # Processing the call keyword arguments (line 117)
        kwargs_131590 = {}
        # Getting the type of 'self' (line 117)
        self_131585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 117)
        set_transform_131586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_131585, 'set_transform')
        # Calling set_transform(args, kwargs) (line 117)
        set_transform_call_result_131591 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), set_transform_131586, *[transAxes_131589], **kwargs_131590)
        
        
        # Assigning a Name to a Attribute (line 118):
        
        # Assigning a Name to a Attribute (line 118):
        # Getting the type of 'True' (line 118)
        True_131592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'True')
        # Getting the type of 'self' (line 118)
        self_131593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_131593, 'stale', True_131592)
        
        # ################# End of 'set_patch_arc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_patch_arc' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_131594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_patch_arc'
        return stypy_return_type_131594


    @norecursion
    def set_patch_circle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_patch_circle'
        module_type_store = module_type_store.open_function_context('set_patch_circle', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.set_patch_circle.__dict__.__setitem__('stypy_localization', localization)
        Spine.set_patch_circle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.set_patch_circle.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.set_patch_circle.__dict__.__setitem__('stypy_function_name', 'Spine.set_patch_circle')
        Spine.set_patch_circle.__dict__.__setitem__('stypy_param_names_list', ['center', 'radius'])
        Spine.set_patch_circle.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.set_patch_circle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.set_patch_circle.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.set_patch_circle.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.set_patch_circle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.set_patch_circle.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.set_patch_circle', ['center', 'radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_patch_circle', localization, ['center', 'radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_patch_circle(...)' code ##################

        unicode_131595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'unicode', u'set the spine to be circular')
        
        # Assigning a Str to a Attribute (line 122):
        
        # Assigning a Str to a Attribute (line 122):
        unicode_131596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 27), 'unicode', u'circle')
        # Getting the type of 'self' (line 122)
        self_131597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member '_patch_type' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_131597, '_patch_type', unicode_131596)
        
        # Assigning a Name to a Attribute (line 123):
        
        # Assigning a Name to a Attribute (line 123):
        # Getting the type of 'center' (line 123)
        center_131598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'center')
        # Getting the type of 'self' (line 123)
        self_131599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self')
        # Setting the type of the member '_center' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_131599, '_center', center_131598)
        
        # Assigning a BinOp to a Attribute (line 124):
        
        # Assigning a BinOp to a Attribute (line 124):
        # Getting the type of 'radius' (line 124)
        radius_131600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 22), 'radius')
        int_131601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 31), 'int')
        # Applying the binary operator '*' (line 124)
        result_mul_131602 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 22), '*', radius_131600, int_131601)
        
        # Getting the type of 'self' (line 124)
        self_131603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self')
        # Setting the type of the member '_width' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_131603, '_width', result_mul_131602)
        
        # Assigning a BinOp to a Attribute (line 125):
        
        # Assigning a BinOp to a Attribute (line 125):
        # Getting the type of 'radius' (line 125)
        radius_131604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'radius')
        int_131605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 32), 'int')
        # Applying the binary operator '*' (line 125)
        result_mul_131606 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 23), '*', radius_131604, int_131605)
        
        # Getting the type of 'self' (line 125)
        self_131607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self')
        # Setting the type of the member '_height' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_131607, '_height', result_mul_131606)
        
        # Call to set_transform(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'self' (line 127)
        self_131610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'self', False)
        # Obtaining the member 'axes' of a type (line 127)
        axes_131611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), self_131610, 'axes')
        # Obtaining the member 'transAxes' of a type (line 127)
        transAxes_131612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), axes_131611, 'transAxes')
        # Processing the call keyword arguments (line 127)
        kwargs_131613 = {}
        # Getting the type of 'self' (line 127)
        self_131608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 127)
        set_transform_131609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_131608, 'set_transform')
        # Calling set_transform(args, kwargs) (line 127)
        set_transform_call_result_131614 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), set_transform_131609, *[transAxes_131612], **kwargs_131613)
        
        
        # Assigning a Name to a Attribute (line 128):
        
        # Assigning a Name to a Attribute (line 128):
        # Getting the type of 'True' (line 128)
        True_131615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'True')
        # Getting the type of 'self' (line 128)
        self_131616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_131616, 'stale', True_131615)
        
        # ################# End of 'set_patch_circle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_patch_circle' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_131617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131617)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_patch_circle'
        return stypy_return_type_131617


    @norecursion
    def set_patch_line(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_patch_line'
        module_type_store = module_type_store.open_function_context('set_patch_line', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.set_patch_line.__dict__.__setitem__('stypy_localization', localization)
        Spine.set_patch_line.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.set_patch_line.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.set_patch_line.__dict__.__setitem__('stypy_function_name', 'Spine.set_patch_line')
        Spine.set_patch_line.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.set_patch_line.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.set_patch_line.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.set_patch_line.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.set_patch_line.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.set_patch_line.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.set_patch_line.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.set_patch_line', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_patch_line', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_patch_line(...)' code ##################

        unicode_131618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 8), 'unicode', u'set the spine to be linear')
        
        # Assigning a Str to a Attribute (line 132):
        
        # Assigning a Str to a Attribute (line 132):
        unicode_131619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 27), 'unicode', u'line')
        # Getting the type of 'self' (line 132)
        self_131620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self')
        # Setting the type of the member '_patch_type' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_131620, '_patch_type', unicode_131619)
        
        # Assigning a Name to a Attribute (line 133):
        
        # Assigning a Name to a Attribute (line 133):
        # Getting the type of 'True' (line 133)
        True_131621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 21), 'True')
        # Getting the type of 'self' (line 133)
        self_131622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_131622, 'stale', True_131621)
        
        # ################# End of 'set_patch_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_patch_line' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_131623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131623)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_patch_line'
        return stypy_return_type_131623


    @norecursion
    def _recompute_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_recompute_transform'
        module_type_store = module_type_store.open_function_context('_recompute_transform', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine._recompute_transform.__dict__.__setitem__('stypy_localization', localization)
        Spine._recompute_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine._recompute_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine._recompute_transform.__dict__.__setitem__('stypy_function_name', 'Spine._recompute_transform')
        Spine._recompute_transform.__dict__.__setitem__('stypy_param_names_list', [])
        Spine._recompute_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine._recompute_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine._recompute_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine._recompute_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine._recompute_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine._recompute_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine._recompute_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_recompute_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_recompute_transform(...)' code ##################

        unicode_131624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'unicode', u'NOTE: This cannot be called until after this has been added\n                 to an Axes, otherwise unit conversion will fail. This\n                 maxes it very important to call the accessor method and\n                 not directly access the transformation member variable.\n        ')
        # Evaluating assert statement condition
        
        # Getting the type of 'self' (line 142)
        self_131625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'self')
        # Obtaining the member '_patch_type' of a type (line 142)
        _patch_type_131626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), self_131625, '_patch_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_131627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        unicode_131628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 36), 'unicode', u'arc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 36), tuple_131627, unicode_131628)
        # Adding element type (line 142)
        unicode_131629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 43), 'unicode', u'circle')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 36), tuple_131627, unicode_131629)
        
        # Applying the binary operator 'in' (line 142)
        result_contains_131630 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 15), 'in', _patch_type_131626, tuple_131627)
        
        
        # Assigning a Tuple to a Name (line 143):
        
        # Assigning a Tuple to a Name (line 143):
        
        # Obtaining an instance of the builtin type 'tuple' (line 143)
        tuple_131631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 143)
        # Adding element type (line 143)
        
        # Call to convert_xunits(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining the type of the subscript
        int_131634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 51), 'int')
        # Getting the type of 'self' (line 143)
        self_131635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 38), 'self', False)
        # Obtaining the member '_center' of a type (line 143)
        _center_131636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 38), self_131635, '_center')
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___131637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 38), _center_131636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_131638 = invoke(stypy.reporting.localization.Localization(__file__, 143, 38), getitem___131637, int_131634)
        
        # Processing the call keyword arguments (line 143)
        kwargs_131639 = {}
        # Getting the type of 'self' (line 143)
        self_131632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'self', False)
        # Obtaining the member 'convert_xunits' of a type (line 143)
        convert_xunits_131633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 18), self_131632, 'convert_xunits')
        # Calling convert_xunits(args, kwargs) (line 143)
        convert_xunits_call_result_131640 = invoke(stypy.reporting.localization.Localization(__file__, 143, 18), convert_xunits_131633, *[subscript_call_result_131638], **kwargs_131639)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 18), tuple_131631, convert_xunits_call_result_131640)
        # Adding element type (line 143)
        
        # Call to convert_yunits(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining the type of the subscript
        int_131643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 51), 'int')
        # Getting the type of 'self' (line 144)
        self_131644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 38), 'self', False)
        # Obtaining the member '_center' of a type (line 144)
        _center_131645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 38), self_131644, '_center')
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___131646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 38), _center_131645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_131647 = invoke(stypy.reporting.localization.Localization(__file__, 144, 38), getitem___131646, int_131643)
        
        # Processing the call keyword arguments (line 144)
        kwargs_131648 = {}
        # Getting the type of 'self' (line 144)
        self_131641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'self', False)
        # Obtaining the member 'convert_yunits' of a type (line 144)
        convert_yunits_131642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 18), self_131641, 'convert_yunits')
        # Calling convert_yunits(args, kwargs) (line 144)
        convert_yunits_call_result_131649 = invoke(stypy.reporting.localization.Localization(__file__, 144, 18), convert_yunits_131642, *[subscript_call_result_131647], **kwargs_131648)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 18), tuple_131631, convert_yunits_call_result_131649)
        
        # Assigning a type to the variable 'center' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'center', tuple_131631)
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to convert_xunits(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'self' (line 145)
        self_131652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'self', False)
        # Obtaining the member '_width' of a type (line 145)
        _width_131653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 36), self_131652, '_width')
        # Processing the call keyword arguments (line 145)
        kwargs_131654 = {}
        # Getting the type of 'self' (line 145)
        self_131650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'self', False)
        # Obtaining the member 'convert_xunits' of a type (line 145)
        convert_xunits_131651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 16), self_131650, 'convert_xunits')
        # Calling convert_xunits(args, kwargs) (line 145)
        convert_xunits_call_result_131655 = invoke(stypy.reporting.localization.Localization(__file__, 145, 16), convert_xunits_131651, *[_width_131653], **kwargs_131654)
        
        # Assigning a type to the variable 'width' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'width', convert_xunits_call_result_131655)
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to convert_yunits(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_131658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'self', False)
        # Obtaining the member '_height' of a type (line 146)
        _height_131659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 37), self_131658, '_height')
        # Processing the call keyword arguments (line 146)
        kwargs_131660 = {}
        # Getting the type of 'self' (line 146)
        self_131656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'self', False)
        # Obtaining the member 'convert_yunits' of a type (line 146)
        convert_yunits_131657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 17), self_131656, 'convert_yunits')
        # Calling convert_yunits(args, kwargs) (line 146)
        convert_yunits_call_result_131661 = invoke(stypy.reporting.localization.Localization(__file__, 146, 17), convert_yunits_131657, *[_height_131659], **kwargs_131660)
        
        # Assigning a type to the variable 'height' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'height', convert_yunits_call_result_131661)
        
        # Assigning a Call to a Attribute (line 147):
        
        # Assigning a Call to a Attribute (line 147):
        
        # Call to translate(...): (line 147)
        # Getting the type of 'center' (line 149)
        center_131676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'center', False)
        # Processing the call keyword arguments (line 147)
        kwargs_131677 = {}
        
        # Call to scale(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'width' (line 148)
        width_131667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'width', False)
        float_131668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'float')
        # Applying the binary operator '*' (line 148)
        result_mul_131669 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 19), '*', width_131667, float_131668)
        
        # Getting the type of 'height' (line 148)
        height_131670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'height', False)
        float_131671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 41), 'float')
        # Applying the binary operator '*' (line 148)
        result_mul_131672 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 32), '*', height_131670, float_131671)
        
        # Processing the call keyword arguments (line 147)
        kwargs_131673 = {}
        
        # Call to Affine2D(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_131664 = {}
        # Getting the type of 'mtransforms' (line 147)
        mtransforms_131662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 32), 'mtransforms', False)
        # Obtaining the member 'Affine2D' of a type (line 147)
        Affine2D_131663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 32), mtransforms_131662, 'Affine2D')
        # Calling Affine2D(args, kwargs) (line 147)
        Affine2D_call_result_131665 = invoke(stypy.reporting.localization.Localization(__file__, 147, 32), Affine2D_131663, *[], **kwargs_131664)
        
        # Obtaining the member 'scale' of a type (line 147)
        scale_131666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 32), Affine2D_call_result_131665, 'scale')
        # Calling scale(args, kwargs) (line 147)
        scale_call_result_131674 = invoke(stypy.reporting.localization.Localization(__file__, 147, 32), scale_131666, *[result_mul_131669, result_mul_131672], **kwargs_131673)
        
        # Obtaining the member 'translate' of a type (line 147)
        translate_131675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 32), scale_call_result_131674, 'translate')
        # Calling translate(args, kwargs) (line 147)
        translate_call_result_131678 = invoke(stypy.reporting.localization.Localization(__file__, 147, 32), translate_131675, *[center_131676], **kwargs_131677)
        
        # Getting the type of 'self' (line 147)
        self_131679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member '_patch_transform' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_131679, '_patch_transform', translate_call_result_131678)
        
        # ################# End of '_recompute_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_recompute_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_131680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131680)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_recompute_transform'
        return stypy_return_type_131680


    @norecursion
    def get_patch_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_patch_transform'
        module_type_store = module_type_store.open_function_context('get_patch_transform', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.get_patch_transform.__dict__.__setitem__('stypy_localization', localization)
        Spine.get_patch_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.get_patch_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.get_patch_transform.__dict__.__setitem__('stypy_function_name', 'Spine.get_patch_transform')
        Spine.get_patch_transform.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.get_patch_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.get_patch_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.get_patch_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.get_patch_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.get_patch_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.get_patch_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.get_patch_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_patch_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_patch_transform(...)' code ##################

        
        
        # Getting the type of 'self' (line 152)
        self_131681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'self')
        # Obtaining the member '_patch_type' of a type (line 152)
        _patch_type_131682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), self_131681, '_patch_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_131683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        unicode_131684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'unicode', u'arc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 32), tuple_131683, unicode_131684)
        # Adding element type (line 152)
        unicode_131685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 39), 'unicode', u'circle')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 32), tuple_131683, unicode_131685)
        
        # Applying the binary operator 'in' (line 152)
        result_contains_131686 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 11), 'in', _patch_type_131682, tuple_131683)
        
        # Testing the type of an if condition (line 152)
        if_condition_131687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 8), result_contains_131686)
        # Assigning a type to the variable 'if_condition_131687' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'if_condition_131687', if_condition_131687)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _recompute_transform(...): (line 153)
        # Processing the call keyword arguments (line 153)
        kwargs_131690 = {}
        # Getting the type of 'self' (line 153)
        self_131688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'self', False)
        # Obtaining the member '_recompute_transform' of a type (line 153)
        _recompute_transform_131689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), self_131688, '_recompute_transform')
        # Calling _recompute_transform(args, kwargs) (line 153)
        _recompute_transform_call_result_131691 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), _recompute_transform_131689, *[], **kwargs_131690)
        
        # Getting the type of 'self' (line 154)
        self_131692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'self')
        # Obtaining the member '_patch_transform' of a type (line 154)
        _patch_transform_131693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 19), self_131692, '_patch_transform')
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'stypy_return_type', _patch_transform_131693)
        # SSA branch for the else part of an if statement (line 152)
        module_type_store.open_ssa_branch('else')
        
        # Call to get_patch_transform(...): (line 156)
        # Processing the call keyword arguments (line 156)
        kwargs_131700 = {}
        
        # Call to super(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'Spine' (line 156)
        Spine_131695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'Spine', False)
        # Getting the type of 'self' (line 156)
        self_131696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'self', False)
        # Processing the call keyword arguments (line 156)
        kwargs_131697 = {}
        # Getting the type of 'super' (line 156)
        super_131694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'super', False)
        # Calling super(args, kwargs) (line 156)
        super_call_result_131698 = invoke(stypy.reporting.localization.Localization(__file__, 156, 19), super_131694, *[Spine_131695, self_131696], **kwargs_131697)
        
        # Obtaining the member 'get_patch_transform' of a type (line 156)
        get_patch_transform_131699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 19), super_call_result_131698, 'get_patch_transform')
        # Calling get_patch_transform(args, kwargs) (line 156)
        get_patch_transform_call_result_131701 = invoke(stypy.reporting.localization.Localization(__file__, 156, 19), get_patch_transform_131699, *[], **kwargs_131700)
        
        # Assigning a type to the variable 'stypy_return_type' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'stypy_return_type', get_patch_transform_call_result_131701)
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_patch_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_patch_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_131702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_patch_transform'
        return stypy_return_type_131702


    @norecursion
    def get_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_path'
        module_type_store = module_type_store.open_function_context('get_path', 158, 4, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.get_path.__dict__.__setitem__('stypy_localization', localization)
        Spine.get_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.get_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.get_path.__dict__.__setitem__('stypy_function_name', 'Spine.get_path')
        Spine.get_path.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.get_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.get_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.get_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.get_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.get_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.get_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.get_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_path(...)' code ##################

        # Getting the type of 'self' (line 159)
        self_131703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'self')
        # Obtaining the member '_path' of a type (line 159)
        _path_131704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), self_131703, '_path')
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', _path_131704)
        
        # ################# End of 'get_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_path' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_131705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131705)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_path'
        return stypy_return_type_131705


    @norecursion
    def _ensure_position_is_set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ensure_position_is_set'
        module_type_store = module_type_store.open_function_context('_ensure_position_is_set', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_localization', localization)
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_function_name', 'Spine._ensure_position_is_set')
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_param_names_list', [])
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine._ensure_position_is_set.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine._ensure_position_is_set', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ensure_position_is_set', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ensure_position_is_set(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 162)
        # Getting the type of 'self' (line 162)
        self_131706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'self')
        # Obtaining the member '_position' of a type (line 162)
        _position_131707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 11), self_131706, '_position')
        # Getting the type of 'None' (line 162)
        None_131708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'None')
        
        (may_be_131709, more_types_in_union_131710) = may_be_none(_position_131707, None_131708)

        if may_be_131709:

            if more_types_in_union_131710:
                # Runtime conditional SSA (line 162)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Tuple to a Attribute (line 164):
            
            # Assigning a Tuple to a Attribute (line 164):
            
            # Obtaining an instance of the builtin type 'tuple' (line 164)
            tuple_131711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 164)
            # Adding element type (line 164)
            unicode_131712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 30), 'unicode', u'outward')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 30), tuple_131711, unicode_131712)
            # Adding element type (line 164)
            float_131713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 41), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 30), tuple_131711, float_131713)
            
            # Getting the type of 'self' (line 164)
            self_131714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'self')
            # Setting the type of the member '_position' of a type (line 164)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), self_131714, '_position', tuple_131711)
            
            # Call to set_position(...): (line 165)
            # Processing the call arguments (line 165)
            # Getting the type of 'self' (line 165)
            self_131717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'self', False)
            # Obtaining the member '_position' of a type (line 165)
            _position_131718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 30), self_131717, '_position')
            # Processing the call keyword arguments (line 165)
            kwargs_131719 = {}
            # Getting the type of 'self' (line 165)
            self_131715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'self', False)
            # Obtaining the member 'set_position' of a type (line 165)
            set_position_131716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), self_131715, 'set_position')
            # Calling set_position(args, kwargs) (line 165)
            set_position_call_result_131720 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), set_position_131716, *[_position_131718], **kwargs_131719)
            

            if more_types_in_union_131710:
                # SSA join for if statement (line 162)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_ensure_position_is_set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ensure_position_is_set' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_131721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131721)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ensure_position_is_set'
        return stypy_return_type_131721


    @norecursion
    def register_axis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'register_axis'
        module_type_store = module_type_store.open_function_context('register_axis', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.register_axis.__dict__.__setitem__('stypy_localization', localization)
        Spine.register_axis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.register_axis.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.register_axis.__dict__.__setitem__('stypy_function_name', 'Spine.register_axis')
        Spine.register_axis.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        Spine.register_axis.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.register_axis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.register_axis.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.register_axis.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.register_axis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.register_axis.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.register_axis', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'register_axis', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'register_axis(...)' code ##################

        unicode_131722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, (-1)), 'unicode', u'register an axis\n\n        An axis should be registered with its corresponding spine from\n        the Axes instance. This allows the spine to clear any axis\n        properties when needed.\n        ')
        
        # Assigning a Name to a Attribute (line 174):
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'axis' (line 174)
        axis_131723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'axis')
        # Getting the type of 'self' (line 174)
        self_131724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_131724, 'axis', axis_131723)
        
        
        # Getting the type of 'self' (line 175)
        self_131725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'self')
        # Obtaining the member 'axis' of a type (line 175)
        axis_131726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 11), self_131725, 'axis')
        # Getting the type of 'None' (line 175)
        None_131727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'None')
        # Applying the binary operator 'isnot' (line 175)
        result_is_not_131728 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), 'isnot', axis_131726, None_131727)
        
        # Testing the type of an if condition (line 175)
        if_condition_131729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), result_is_not_131728)
        # Assigning a type to the variable 'if_condition_131729' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_131729', if_condition_131729)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to cla(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_131733 = {}
        # Getting the type of 'self' (line 176)
        self_131730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self', False)
        # Obtaining the member 'axis' of a type (line 176)
        axis_131731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_131730, 'axis')
        # Obtaining the member 'cla' of a type (line 176)
        cla_131732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), axis_131731, 'cla')
        # Calling cla(args, kwargs) (line 176)
        cla_call_result_131734 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), cla_131732, *[], **kwargs_131733)
        
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 177):
        
        # Assigning a Name to a Attribute (line 177):
        # Getting the type of 'True' (line 177)
        True_131735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'True')
        # Getting the type of 'self' (line 177)
        self_131736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_131736, 'stale', True_131735)
        
        # ################# End of 'register_axis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'register_axis' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_131737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131737)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'register_axis'
        return stypy_return_type_131737


    @norecursion
    def cla(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cla'
        module_type_store = module_type_store.open_function_context('cla', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.cla.__dict__.__setitem__('stypy_localization', localization)
        Spine.cla.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.cla.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.cla.__dict__.__setitem__('stypy_function_name', 'Spine.cla')
        Spine.cla.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.cla.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.cla.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.cla.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.cla.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.cla.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.cla.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.cla', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cla', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cla(...)' code ##################

        unicode_131738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'unicode', u'Clear the current spine')
        
        # Assigning a Name to a Attribute (line 181):
        
        # Assigning a Name to a Attribute (line 181):
        # Getting the type of 'None' (line 181)
        None_131739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'None')
        # Getting the type of 'self' (line 181)
        self_131740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self')
        # Setting the type of the member '_position' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_131740, '_position', None_131739)
        
        
        # Getting the type of 'self' (line 182)
        self_131741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'self')
        # Obtaining the member 'axis' of a type (line 182)
        axis_131742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 11), self_131741, 'axis')
        # Getting the type of 'None' (line 182)
        None_131743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 28), 'None')
        # Applying the binary operator 'isnot' (line 182)
        result_is_not_131744 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 11), 'isnot', axis_131742, None_131743)
        
        # Testing the type of an if condition (line 182)
        if_condition_131745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), result_is_not_131744)
        # Assigning a type to the variable 'if_condition_131745' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_131745', if_condition_131745)
        # SSA begins for if statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to cla(...): (line 183)
        # Processing the call keyword arguments (line 183)
        kwargs_131749 = {}
        # Getting the type of 'self' (line 183)
        self_131746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'self', False)
        # Obtaining the member 'axis' of a type (line 183)
        axis_131747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), self_131746, 'axis')
        # Obtaining the member 'cla' of a type (line 183)
        cla_131748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), axis_131747, 'cla')
        # Calling cla(args, kwargs) (line 183)
        cla_call_result_131750 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), cla_131748, *[], **kwargs_131749)
        
        # SSA join for if statement (line 182)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'cla(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cla' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_131751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cla'
        return stypy_return_type_131751


    @norecursion
    def is_frame_like(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_frame_like'
        module_type_store = module_type_store.open_function_context('is_frame_like', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.is_frame_like.__dict__.__setitem__('stypy_localization', localization)
        Spine.is_frame_like.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.is_frame_like.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.is_frame_like.__dict__.__setitem__('stypy_function_name', 'Spine.is_frame_like')
        Spine.is_frame_like.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.is_frame_like.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.is_frame_like.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.is_frame_like.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.is_frame_like.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.is_frame_like.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.is_frame_like.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.is_frame_like', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_frame_like', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_frame_like(...)' code ##################

        unicode_131752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'unicode', u'return True if directly on axes frame\n\n        This is useful for determining if a spine is the edge of an\n        old style MPL plot. If so, this function will return True.\n        ')
        
        # Call to _ensure_position_is_set(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_131755 = {}
        # Getting the type of 'self' (line 191)
        self_131753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self', False)
        # Obtaining the member '_ensure_position_is_set' of a type (line 191)
        _ensure_position_is_set_131754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_131753, '_ensure_position_is_set')
        # Calling _ensure_position_is_set(args, kwargs) (line 191)
        _ensure_position_is_set_call_result_131756 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), _ensure_position_is_set_131754, *[], **kwargs_131755)
        
        
        # Assigning a Attribute to a Name (line 192):
        
        # Assigning a Attribute to a Name (line 192):
        # Getting the type of 'self' (line 192)
        self_131757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'self')
        # Obtaining the member '_position' of a type (line 192)
        _position_131758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 19), self_131757, '_position')
        # Assigning a type to the variable 'position' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'position', _position_131758)
        
        
        # Call to isinstance(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'position' (line 193)
        position_131760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'position', False)
        # Getting the type of 'six' (line 193)
        six_131761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 32), 'six', False)
        # Obtaining the member 'string_types' of a type (line 193)
        string_types_131762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 32), six_131761, 'string_types')
        # Processing the call keyword arguments (line 193)
        kwargs_131763 = {}
        # Getting the type of 'isinstance' (line 193)
        isinstance_131759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 193)
        isinstance_call_result_131764 = invoke(stypy.reporting.localization.Localization(__file__, 193, 11), isinstance_131759, *[position_131760, string_types_131762], **kwargs_131763)
        
        # Testing the type of an if condition (line 193)
        if_condition_131765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), isinstance_call_result_131764)
        # Assigning a type to the variable 'if_condition_131765' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_131765', if_condition_131765)
        # SSA begins for if statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'position' (line 194)
        position_131766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'position')
        unicode_131767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 27), 'unicode', u'center')
        # Applying the binary operator '==' (line 194)
        result_eq_131768 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 15), '==', position_131766, unicode_131767)
        
        # Testing the type of an if condition (line 194)
        if_condition_131769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 12), result_eq_131768)
        # Assigning a type to the variable 'if_condition_131769' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'if_condition_131769', if_condition_131769)
        # SSA begins for if statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 195):
        
        # Assigning a Tuple to a Name (line 195):
        
        # Obtaining an instance of the builtin type 'tuple' (line 195)
        tuple_131770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 195)
        # Adding element type (line 195)
        unicode_131771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 28), 'unicode', u'axes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 28), tuple_131770, unicode_131771)
        # Adding element type (line 195)
        float_131772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 28), tuple_131770, float_131772)
        
        # Assigning a type to the variable 'position' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'position', tuple_131770)
        # SSA branch for the else part of an if statement (line 194)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'position' (line 196)
        position_131773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'position')
        unicode_131774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 29), 'unicode', u'zero')
        # Applying the binary operator '==' (line 196)
        result_eq_131775 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 17), '==', position_131773, unicode_131774)
        
        # Testing the type of an if condition (line 196)
        if_condition_131776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 17), result_eq_131775)
        # Assigning a type to the variable 'if_condition_131776' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'if_condition_131776', if_condition_131776)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 197):
        
        # Assigning a Tuple to a Name (line 197):
        
        # Obtaining an instance of the builtin type 'tuple' (line 197)
        tuple_131777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 197)
        # Adding element type (line 197)
        unicode_131778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 28), 'unicode', u'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 28), tuple_131777, unicode_131778)
        # Adding element type (line 197)
        int_131779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 28), tuple_131777, int_131779)
        
        # Assigning a type to the variable 'position' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'position', tuple_131777)
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 194)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 193)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'position' (line 198)
        position_131781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'position', False)
        # Processing the call keyword arguments (line 198)
        kwargs_131782 = {}
        # Getting the type of 'len' (line 198)
        len_131780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'len', False)
        # Calling len(args, kwargs) (line 198)
        len_call_result_131783 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), len_131780, *[position_131781], **kwargs_131782)
        
        int_131784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 28), 'int')
        # Applying the binary operator '!=' (line 198)
        result_ne_131785 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '!=', len_call_result_131783, int_131784)
        
        # Testing the type of an if condition (line 198)
        if_condition_131786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_ne_131785)
        # Assigning a type to the variable 'if_condition_131786' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_131786', if_condition_131786)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 199)
        # Processing the call arguments (line 199)
        unicode_131788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'unicode', u'position should be 2-tuple')
        # Processing the call keyword arguments (line 199)
        kwargs_131789 = {}
        # Getting the type of 'ValueError' (line 199)
        ValueError_131787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 199)
        ValueError_call_result_131790 = invoke(stypy.reporting.localization.Localization(__file__, 199, 18), ValueError_131787, *[unicode_131788], **kwargs_131789)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 199, 12), ValueError_call_result_131790, 'raise parameter', BaseException)
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Tuple (line 200):
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_131791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        # Getting the type of 'position' (line 200)
        position_131792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'position')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___131793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), position_131792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_131794 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___131793, int_131791)
        
        # Assigning a type to the variable 'tuple_var_assignment_131377' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_131377', subscript_call_result_131794)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_131795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        # Getting the type of 'position' (line 200)
        position_131796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'position')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___131797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), position_131796, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_131798 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___131797, int_131795)
        
        # Assigning a type to the variable 'tuple_var_assignment_131378' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_131378', subscript_call_result_131798)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_131377' (line 200)
        tuple_var_assignment_131377_131799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_131377')
        # Assigning a type to the variable 'position_type' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'position_type', tuple_var_assignment_131377_131799)
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_131378' (line 200)
        tuple_var_assignment_131378_131800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_131378')
        # Assigning a type to the variable 'amount' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'amount', tuple_var_assignment_131378_131800)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'position_type' (line 201)
        position_type_131801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'position_type')
        unicode_131802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 28), 'unicode', u'outward')
        # Applying the binary operator '==' (line 201)
        result_eq_131803 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), '==', position_type_131801, unicode_131802)
        
        
        # Getting the type of 'amount' (line 201)
        amount_131804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 42), 'amount')
        int_131805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 52), 'int')
        # Applying the binary operator '==' (line 201)
        result_eq_131806 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 42), '==', amount_131804, int_131805)
        
        # Applying the binary operator 'and' (line 201)
        result_and_keyword_131807 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), 'and', result_eq_131803, result_eq_131806)
        
        # Testing the type of an if condition (line 201)
        if_condition_131808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), result_and_keyword_131807)
        # Assigning a type to the variable 'if_condition_131808' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_131808', if_condition_131808)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 202)
        True_131809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'stypy_return_type', True_131809)
        # SSA branch for the else part of an if statement (line 201)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'False' (line 204)
        False_131810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'stypy_return_type', False_131810)
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'is_frame_like(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_frame_like' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_131811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131811)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_frame_like'
        return stypy_return_type_131811


    @norecursion
    def _adjust_location(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjust_location'
        module_type_store = module_type_store.open_function_context('_adjust_location', 206, 4, False)
        # Assigning a type to the variable 'self' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine._adjust_location.__dict__.__setitem__('stypy_localization', localization)
        Spine._adjust_location.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine._adjust_location.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine._adjust_location.__dict__.__setitem__('stypy_function_name', 'Spine._adjust_location')
        Spine._adjust_location.__dict__.__setitem__('stypy_param_names_list', [])
        Spine._adjust_location.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine._adjust_location.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine._adjust_location.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine._adjust_location.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine._adjust_location.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine._adjust_location.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine._adjust_location', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjust_location', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjust_location(...)' code ##################

        unicode_131812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'unicode', u'automatically set spine bounds to the view interval')
        
        
        # Getting the type of 'self' (line 209)
        self_131813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'self')
        # Obtaining the member 'spine_type' of a type (line 209)
        spine_type_131814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), self_131813, 'spine_type')
        unicode_131815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 30), 'unicode', u'circle')
        # Applying the binary operator '==' (line 209)
        result_eq_131816 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), '==', spine_type_131814, unicode_131815)
        
        # Testing the type of an if condition (line 209)
        if_condition_131817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 8), result_eq_131816)
        # Assigning a type to the variable 'if_condition_131817' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'if_condition_131817', if_condition_131817)
        # SSA begins for if statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 212)
        # Getting the type of 'self' (line 212)
        self_131818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'self')
        # Obtaining the member '_bounds' of a type (line 212)
        _bounds_131819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 11), self_131818, '_bounds')
        # Getting the type of 'None' (line 212)
        None_131820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'None')
        
        (may_be_131821, more_types_in_union_131822) = may_be_none(_bounds_131819, None_131820)

        if may_be_131821:

            if more_types_in_union_131822:
                # Runtime conditional SSA (line 212)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'self' (line 213)
            self_131823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'self')
            # Obtaining the member 'spine_type' of a type (line 213)
            spine_type_131824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), self_131823, 'spine_type')
            
            # Obtaining an instance of the builtin type 'tuple' (line 213)
            tuple_131825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 213)
            # Adding element type (line 213)
            unicode_131826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 35), 'unicode', u'left')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 35), tuple_131825, unicode_131826)
            # Adding element type (line 213)
            unicode_131827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 43), 'unicode', u'right')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 35), tuple_131825, unicode_131827)
            
            # Applying the binary operator 'in' (line 213)
            result_contains_131828 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 15), 'in', spine_type_131824, tuple_131825)
            
            # Testing the type of an if condition (line 213)
            if_condition_131829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 12), result_contains_131828)
            # Assigning a type to the variable 'if_condition_131829' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'if_condition_131829', if_condition_131829)
            # SSA begins for if statement (line 213)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Tuple (line 214):
            
            # Assigning a Subscript to a Name (line 214):
            
            # Obtaining the type of the subscript
            int_131830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 16), 'int')
            # Getting the type of 'self' (line 214)
            self_131831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'self')
            # Obtaining the member 'axes' of a type (line 214)
            axes_131832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), self_131831, 'axes')
            # Obtaining the member 'viewLim' of a type (line 214)
            viewLim_131833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), axes_131832, 'viewLim')
            # Obtaining the member 'intervaly' of a type (line 214)
            intervaly_131834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), viewLim_131833, 'intervaly')
            # Obtaining the member '__getitem__' of a type (line 214)
            getitem___131835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), intervaly_131834, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 214)
            subscript_call_result_131836 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), getitem___131835, int_131830)
            
            # Assigning a type to the variable 'tuple_var_assignment_131379' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'tuple_var_assignment_131379', subscript_call_result_131836)
            
            # Assigning a Subscript to a Name (line 214):
            
            # Obtaining the type of the subscript
            int_131837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 16), 'int')
            # Getting the type of 'self' (line 214)
            self_131838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'self')
            # Obtaining the member 'axes' of a type (line 214)
            axes_131839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), self_131838, 'axes')
            # Obtaining the member 'viewLim' of a type (line 214)
            viewLim_131840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), axes_131839, 'viewLim')
            # Obtaining the member 'intervaly' of a type (line 214)
            intervaly_131841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), viewLim_131840, 'intervaly')
            # Obtaining the member '__getitem__' of a type (line 214)
            getitem___131842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), intervaly_131841, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 214)
            subscript_call_result_131843 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), getitem___131842, int_131837)
            
            # Assigning a type to the variable 'tuple_var_assignment_131380' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'tuple_var_assignment_131380', subscript_call_result_131843)
            
            # Assigning a Name to a Name (line 214):
            # Getting the type of 'tuple_var_assignment_131379' (line 214)
            tuple_var_assignment_131379_131844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'tuple_var_assignment_131379')
            # Assigning a type to the variable 'low' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'low', tuple_var_assignment_131379_131844)
            
            # Assigning a Name to a Name (line 214):
            # Getting the type of 'tuple_var_assignment_131380' (line 214)
            tuple_var_assignment_131380_131845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'tuple_var_assignment_131380')
            # Assigning a type to the variable 'high' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'high', tuple_var_assignment_131380_131845)
            # SSA branch for the else part of an if statement (line 213)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'self' (line 215)
            self_131846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'self')
            # Obtaining the member 'spine_type' of a type (line 215)
            spine_type_131847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 17), self_131846, 'spine_type')
            
            # Obtaining an instance of the builtin type 'tuple' (line 215)
            tuple_131848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 215)
            # Adding element type (line 215)
            unicode_131849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 37), 'unicode', u'top')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 37), tuple_131848, unicode_131849)
            # Adding element type (line 215)
            unicode_131850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 44), 'unicode', u'bottom')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 37), tuple_131848, unicode_131850)
            
            # Applying the binary operator 'in' (line 215)
            result_contains_131851 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 17), 'in', spine_type_131847, tuple_131848)
            
            # Testing the type of an if condition (line 215)
            if_condition_131852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 17), result_contains_131851)
            # Assigning a type to the variable 'if_condition_131852' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'if_condition_131852', if_condition_131852)
            # SSA begins for if statement (line 215)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Tuple (line 216):
            
            # Assigning a Subscript to a Name (line 216):
            
            # Obtaining the type of the subscript
            int_131853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 16), 'int')
            # Getting the type of 'self' (line 216)
            self_131854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'self')
            # Obtaining the member 'axes' of a type (line 216)
            axes_131855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 28), self_131854, 'axes')
            # Obtaining the member 'viewLim' of a type (line 216)
            viewLim_131856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 28), axes_131855, 'viewLim')
            # Obtaining the member 'intervalx' of a type (line 216)
            intervalx_131857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 28), viewLim_131856, 'intervalx')
            # Obtaining the member '__getitem__' of a type (line 216)
            getitem___131858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), intervalx_131857, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 216)
            subscript_call_result_131859 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), getitem___131858, int_131853)
            
            # Assigning a type to the variable 'tuple_var_assignment_131381' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_131381', subscript_call_result_131859)
            
            # Assigning a Subscript to a Name (line 216):
            
            # Obtaining the type of the subscript
            int_131860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 16), 'int')
            # Getting the type of 'self' (line 216)
            self_131861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'self')
            # Obtaining the member 'axes' of a type (line 216)
            axes_131862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 28), self_131861, 'axes')
            # Obtaining the member 'viewLim' of a type (line 216)
            viewLim_131863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 28), axes_131862, 'viewLim')
            # Obtaining the member 'intervalx' of a type (line 216)
            intervalx_131864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 28), viewLim_131863, 'intervalx')
            # Obtaining the member '__getitem__' of a type (line 216)
            getitem___131865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), intervalx_131864, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 216)
            subscript_call_result_131866 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), getitem___131865, int_131860)
            
            # Assigning a type to the variable 'tuple_var_assignment_131382' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_131382', subscript_call_result_131866)
            
            # Assigning a Name to a Name (line 216):
            # Getting the type of 'tuple_var_assignment_131381' (line 216)
            tuple_var_assignment_131381_131867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_131381')
            # Assigning a type to the variable 'low' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'low', tuple_var_assignment_131381_131867)
            
            # Assigning a Name to a Name (line 216):
            # Getting the type of 'tuple_var_assignment_131382' (line 216)
            tuple_var_assignment_131382_131868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_131382')
            # Assigning a type to the variable 'high' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'high', tuple_var_assignment_131382_131868)
            # SSA branch for the else part of an if statement (line 215)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 218)
            # Processing the call arguments (line 218)
            unicode_131870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 33), 'unicode', u'unknown spine spine_type: %s')
            # Getting the type of 'self' (line 219)
            self_131871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 33), 'self', False)
            # Obtaining the member 'spine_type' of a type (line 219)
            spine_type_131872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 33), self_131871, 'spine_type')
            # Applying the binary operator '%' (line 218)
            result_mod_131873 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 33), '%', unicode_131870, spine_type_131872)
            
            # Processing the call keyword arguments (line 218)
            kwargs_131874 = {}
            # Getting the type of 'ValueError' (line 218)
            ValueError_131869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 218)
            ValueError_call_result_131875 = invoke(stypy.reporting.localization.Localization(__file__, 218, 22), ValueError_131869, *[result_mod_131873], **kwargs_131874)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 218, 16), ValueError_call_result_131875, 'raise parameter', BaseException)
            # SSA join for if statement (line 215)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 213)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'self' (line 221)
            self_131876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'self')
            # Obtaining the member '_smart_bounds' of a type (line 221)
            _smart_bounds_131877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 15), self_131876, '_smart_bounds')
            # Testing the type of an if condition (line 221)
            if_condition_131878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 12), _smart_bounds_131877)
            # Assigning a type to the variable 'if_condition_131878' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'if_condition_131878', if_condition_131878)
            # SSA begins for if statement (line 221)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 225):
            
            # Assigning a Call to a Name:
            
            # Call to sorted(...): (line 225)
            # Processing the call arguments (line 225)
            
            # Obtaining an instance of the builtin type 'list' (line 225)
            list_131880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 51), 'list')
            # Adding type elements to the builtin type 'list' instance (line 225)
            # Adding element type (line 225)
            # Getting the type of 'low' (line 225)
            low_131881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 52), 'low', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 51), list_131880, low_131881)
            # Adding element type (line 225)
            # Getting the type of 'high' (line 225)
            high_131882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 57), 'high', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 51), list_131880, high_131882)
            
            # Processing the call keyword arguments (line 225)
            kwargs_131883 = {}
            # Getting the type of 'sorted' (line 225)
            sorted_131879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 44), 'sorted', False)
            # Calling sorted(args, kwargs) (line 225)
            sorted_call_result_131884 = invoke(stypy.reporting.localization.Localization(__file__, 225, 44), sorted_131879, *[list_131880], **kwargs_131883)
            
            # Assigning a type to the variable 'call_assignment_131383' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'call_assignment_131383', sorted_call_result_131884)
            
            # Assigning a Call to a Name (line 225):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_131887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 16), 'int')
            # Processing the call keyword arguments
            kwargs_131888 = {}
            # Getting the type of 'call_assignment_131383' (line 225)
            call_assignment_131383_131885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'call_assignment_131383', False)
            # Obtaining the member '__getitem__' of a type (line 225)
            getitem___131886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), call_assignment_131383_131885, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_131889 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131886, *[int_131887], **kwargs_131888)
            
            # Assigning a type to the variable 'call_assignment_131384' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'call_assignment_131384', getitem___call_result_131889)
            
            # Assigning a Name to a Name (line 225):
            # Getting the type of 'call_assignment_131384' (line 225)
            call_assignment_131384_131890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'call_assignment_131384')
            # Assigning a type to the variable 'viewlim_low' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'viewlim_low', call_assignment_131384_131890)
            
            # Assigning a Call to a Name (line 225):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_131893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 16), 'int')
            # Processing the call keyword arguments
            kwargs_131894 = {}
            # Getting the type of 'call_assignment_131383' (line 225)
            call_assignment_131383_131891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'call_assignment_131383', False)
            # Obtaining the member '__getitem__' of a type (line 225)
            getitem___131892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), call_assignment_131383_131891, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_131895 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131892, *[int_131893], **kwargs_131894)
            
            # Assigning a type to the variable 'call_assignment_131385' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'call_assignment_131385', getitem___call_result_131895)
            
            # Assigning a Name to a Name (line 225):
            # Getting the type of 'call_assignment_131385' (line 225)
            call_assignment_131385_131896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'call_assignment_131385')
            # Assigning a type to the variable 'viewlim_high' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 29), 'viewlim_high', call_assignment_131385_131896)
            
            
            # Getting the type of 'self' (line 227)
            self_131897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'self')
            # Obtaining the member 'spine_type' of a type (line 227)
            spine_type_131898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 19), self_131897, 'spine_type')
            
            # Obtaining an instance of the builtin type 'tuple' (line 227)
            tuple_131899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 39), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 227)
            # Adding element type (line 227)
            unicode_131900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 39), 'unicode', u'left')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 39), tuple_131899, unicode_131900)
            # Adding element type (line 227)
            unicode_131901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 47), 'unicode', u'right')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 39), tuple_131899, unicode_131901)
            
            # Applying the binary operator 'in' (line 227)
            result_contains_131902 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 19), 'in', spine_type_131898, tuple_131899)
            
            # Testing the type of an if condition (line 227)
            if_condition_131903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 16), result_contains_131902)
            # Assigning a type to the variable 'if_condition_131903' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'if_condition_131903', if_condition_131903)
            # SSA begins for if statement (line 227)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Tuple (line 228):
            
            # Assigning a Subscript to a Name (line 228):
            
            # Obtaining the type of the subscript
            int_131904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 20), 'int')
            # Getting the type of 'self' (line 228)
            self_131905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 48), 'self')
            # Obtaining the member 'axes' of a type (line 228)
            axes_131906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 48), self_131905, 'axes')
            # Obtaining the member 'dataLim' of a type (line 228)
            dataLim_131907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 48), axes_131906, 'dataLim')
            # Obtaining the member 'intervaly' of a type (line 228)
            intervaly_131908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 48), dataLim_131907, 'intervaly')
            # Obtaining the member '__getitem__' of a type (line 228)
            getitem___131909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 20), intervaly_131908, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 228)
            subscript_call_result_131910 = invoke(stypy.reporting.localization.Localization(__file__, 228, 20), getitem___131909, int_131904)
            
            # Assigning a type to the variable 'tuple_var_assignment_131386' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'tuple_var_assignment_131386', subscript_call_result_131910)
            
            # Assigning a Subscript to a Name (line 228):
            
            # Obtaining the type of the subscript
            int_131911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 20), 'int')
            # Getting the type of 'self' (line 228)
            self_131912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 48), 'self')
            # Obtaining the member 'axes' of a type (line 228)
            axes_131913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 48), self_131912, 'axes')
            # Obtaining the member 'dataLim' of a type (line 228)
            dataLim_131914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 48), axes_131913, 'dataLim')
            # Obtaining the member 'intervaly' of a type (line 228)
            intervaly_131915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 48), dataLim_131914, 'intervaly')
            # Obtaining the member '__getitem__' of a type (line 228)
            getitem___131916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 20), intervaly_131915, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 228)
            subscript_call_result_131917 = invoke(stypy.reporting.localization.Localization(__file__, 228, 20), getitem___131916, int_131911)
            
            # Assigning a type to the variable 'tuple_var_assignment_131387' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'tuple_var_assignment_131387', subscript_call_result_131917)
            
            # Assigning a Name to a Name (line 228):
            # Getting the type of 'tuple_var_assignment_131386' (line 228)
            tuple_var_assignment_131386_131918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'tuple_var_assignment_131386')
            # Assigning a type to the variable 'datalim_low' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'datalim_low', tuple_var_assignment_131386_131918)
            
            # Assigning a Name to a Name (line 228):
            # Getting the type of 'tuple_var_assignment_131387' (line 228)
            tuple_var_assignment_131387_131919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'tuple_var_assignment_131387')
            # Assigning a type to the variable 'datalim_high' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 33), 'datalim_high', tuple_var_assignment_131387_131919)
            
            # Assigning a Call to a Name (line 229):
            
            # Assigning a Call to a Name (line 229):
            
            # Call to get_yticks(...): (line 229)
            # Processing the call keyword arguments (line 229)
            kwargs_131923 = {}
            # Getting the type of 'self' (line 229)
            self_131920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 28), 'self', False)
            # Obtaining the member 'axes' of a type (line 229)
            axes_131921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 28), self_131920, 'axes')
            # Obtaining the member 'get_yticks' of a type (line 229)
            get_yticks_131922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 28), axes_131921, 'get_yticks')
            # Calling get_yticks(args, kwargs) (line 229)
            get_yticks_call_result_131924 = invoke(stypy.reporting.localization.Localization(__file__, 229, 28), get_yticks_131922, *[], **kwargs_131923)
            
            # Assigning a type to the variable 'ticks' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'ticks', get_yticks_call_result_131924)
            # SSA branch for the else part of an if statement (line 227)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'self' (line 230)
            self_131925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'self')
            # Obtaining the member 'spine_type' of a type (line 230)
            spine_type_131926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 21), self_131925, 'spine_type')
            
            # Obtaining an instance of the builtin type 'tuple' (line 230)
            tuple_131927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 230)
            # Adding element type (line 230)
            unicode_131928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 41), 'unicode', u'top')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 41), tuple_131927, unicode_131928)
            # Adding element type (line 230)
            unicode_131929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 48), 'unicode', u'bottom')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 41), tuple_131927, unicode_131929)
            
            # Applying the binary operator 'in' (line 230)
            result_contains_131930 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 21), 'in', spine_type_131926, tuple_131927)
            
            # Testing the type of an if condition (line 230)
            if_condition_131931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 21), result_contains_131930)
            # Assigning a type to the variable 'if_condition_131931' (line 230)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'if_condition_131931', if_condition_131931)
            # SSA begins for if statement (line 230)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Tuple (line 231):
            
            # Assigning a Subscript to a Name (line 231):
            
            # Obtaining the type of the subscript
            int_131932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 20), 'int')
            # Getting the type of 'self' (line 231)
            self_131933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 48), 'self')
            # Obtaining the member 'axes' of a type (line 231)
            axes_131934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 48), self_131933, 'axes')
            # Obtaining the member 'dataLim' of a type (line 231)
            dataLim_131935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 48), axes_131934, 'dataLim')
            # Obtaining the member 'intervalx' of a type (line 231)
            intervalx_131936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 48), dataLim_131935, 'intervalx')
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___131937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 20), intervalx_131936, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_131938 = invoke(stypy.reporting.localization.Localization(__file__, 231, 20), getitem___131937, int_131932)
            
            # Assigning a type to the variable 'tuple_var_assignment_131388' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'tuple_var_assignment_131388', subscript_call_result_131938)
            
            # Assigning a Subscript to a Name (line 231):
            
            # Obtaining the type of the subscript
            int_131939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 20), 'int')
            # Getting the type of 'self' (line 231)
            self_131940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 48), 'self')
            # Obtaining the member 'axes' of a type (line 231)
            axes_131941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 48), self_131940, 'axes')
            # Obtaining the member 'dataLim' of a type (line 231)
            dataLim_131942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 48), axes_131941, 'dataLim')
            # Obtaining the member 'intervalx' of a type (line 231)
            intervalx_131943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 48), dataLim_131942, 'intervalx')
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___131944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 20), intervalx_131943, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_131945 = invoke(stypy.reporting.localization.Localization(__file__, 231, 20), getitem___131944, int_131939)
            
            # Assigning a type to the variable 'tuple_var_assignment_131389' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'tuple_var_assignment_131389', subscript_call_result_131945)
            
            # Assigning a Name to a Name (line 231):
            # Getting the type of 'tuple_var_assignment_131388' (line 231)
            tuple_var_assignment_131388_131946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'tuple_var_assignment_131388')
            # Assigning a type to the variable 'datalim_low' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'datalim_low', tuple_var_assignment_131388_131946)
            
            # Assigning a Name to a Name (line 231):
            # Getting the type of 'tuple_var_assignment_131389' (line 231)
            tuple_var_assignment_131389_131947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'tuple_var_assignment_131389')
            # Assigning a type to the variable 'datalim_high' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 33), 'datalim_high', tuple_var_assignment_131389_131947)
            
            # Assigning a Call to a Name (line 232):
            
            # Assigning a Call to a Name (line 232):
            
            # Call to get_xticks(...): (line 232)
            # Processing the call keyword arguments (line 232)
            kwargs_131951 = {}
            # Getting the type of 'self' (line 232)
            self_131948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'self', False)
            # Obtaining the member 'axes' of a type (line 232)
            axes_131949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 28), self_131948, 'axes')
            # Obtaining the member 'get_xticks' of a type (line 232)
            get_xticks_131950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 28), axes_131949, 'get_xticks')
            # Calling get_xticks(args, kwargs) (line 232)
            get_xticks_call_result_131952 = invoke(stypy.reporting.localization.Localization(__file__, 232, 28), get_xticks_131950, *[], **kwargs_131951)
            
            # Assigning a type to the variable 'ticks' (line 232)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'ticks', get_xticks_call_result_131952)
            # SSA join for if statement (line 230)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 227)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 234):
            
            # Assigning a Call to a Name (line 234):
            
            # Call to sort(...): (line 234)
            # Processing the call arguments (line 234)
            # Getting the type of 'ticks' (line 234)
            ticks_131955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 32), 'ticks', False)
            # Processing the call keyword arguments (line 234)
            kwargs_131956 = {}
            # Getting the type of 'np' (line 234)
            np_131953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'np', False)
            # Obtaining the member 'sort' of a type (line 234)
            sort_131954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 24), np_131953, 'sort')
            # Calling sort(args, kwargs) (line 234)
            sort_call_result_131957 = invoke(stypy.reporting.localization.Localization(__file__, 234, 24), sort_131954, *[ticks_131955], **kwargs_131956)
            
            # Assigning a type to the variable 'ticks' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'ticks', sort_call_result_131957)
            
            # Assigning a Call to a Tuple (line 235):
            
            # Assigning a Call to a Name:
            
            # Call to sorted(...): (line 235)
            # Processing the call arguments (line 235)
            
            # Obtaining an instance of the builtin type 'list' (line 235)
            list_131959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 51), 'list')
            # Adding type elements to the builtin type 'list' instance (line 235)
            # Adding element type (line 235)
            # Getting the type of 'datalim_low' (line 235)
            datalim_low_131960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 52), 'datalim_low', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 51), list_131959, datalim_low_131960)
            # Adding element type (line 235)
            # Getting the type of 'datalim_high' (line 235)
            datalim_high_131961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 65), 'datalim_high', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 51), list_131959, datalim_high_131961)
            
            # Processing the call keyword arguments (line 235)
            kwargs_131962 = {}
            # Getting the type of 'sorted' (line 235)
            sorted_131958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 44), 'sorted', False)
            # Calling sorted(args, kwargs) (line 235)
            sorted_call_result_131963 = invoke(stypy.reporting.localization.Localization(__file__, 235, 44), sorted_131958, *[list_131959], **kwargs_131962)
            
            # Assigning a type to the variable 'call_assignment_131390' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'call_assignment_131390', sorted_call_result_131963)
            
            # Assigning a Call to a Name (line 235):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_131966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 16), 'int')
            # Processing the call keyword arguments
            kwargs_131967 = {}
            # Getting the type of 'call_assignment_131390' (line 235)
            call_assignment_131390_131964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'call_assignment_131390', False)
            # Obtaining the member '__getitem__' of a type (line 235)
            getitem___131965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), call_assignment_131390_131964, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_131968 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131965, *[int_131966], **kwargs_131967)
            
            # Assigning a type to the variable 'call_assignment_131391' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'call_assignment_131391', getitem___call_result_131968)
            
            # Assigning a Name to a Name (line 235):
            # Getting the type of 'call_assignment_131391' (line 235)
            call_assignment_131391_131969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'call_assignment_131391')
            # Assigning a type to the variable 'datalim_low' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'datalim_low', call_assignment_131391_131969)
            
            # Assigning a Call to a Name (line 235):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_131972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 16), 'int')
            # Processing the call keyword arguments
            kwargs_131973 = {}
            # Getting the type of 'call_assignment_131390' (line 235)
            call_assignment_131390_131970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'call_assignment_131390', False)
            # Obtaining the member '__getitem__' of a type (line 235)
            getitem___131971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), call_assignment_131390_131970, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_131974 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131971, *[int_131972], **kwargs_131973)
            
            # Assigning a type to the variable 'call_assignment_131392' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'call_assignment_131392', getitem___call_result_131974)
            
            # Assigning a Name to a Name (line 235):
            # Getting the type of 'call_assignment_131392' (line 235)
            call_assignment_131392_131975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'call_assignment_131392')
            # Assigning a type to the variable 'datalim_high' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 29), 'datalim_high', call_assignment_131392_131975)
            
            
            # Getting the type of 'datalim_low' (line 237)
            datalim_low_131976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'datalim_low')
            # Getting the type of 'viewlim_low' (line 237)
            viewlim_low_131977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'viewlim_low')
            # Applying the binary operator '<' (line 237)
            result_lt_131978 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 19), '<', datalim_low_131976, viewlim_low_131977)
            
            # Testing the type of an if condition (line 237)
            if_condition_131979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 16), result_lt_131978)
            # Assigning a type to the variable 'if_condition_131979' (line 237)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'if_condition_131979', if_condition_131979)
            # SSA begins for if statement (line 237)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 239):
            
            # Assigning a Name to a Name (line 239):
            # Getting the type of 'viewlim_low' (line 239)
            viewlim_low_131980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 26), 'viewlim_low')
            # Assigning a type to the variable 'low' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'low', viewlim_low_131980)
            # SSA branch for the else part of an if statement (line 237)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 242):
            
            # Assigning a BinOp to a Name (line 242):
            
            # Getting the type of 'ticks' (line 242)
            ticks_131981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'ticks')
            # Getting the type of 'datalim_low' (line 242)
            datalim_low_131982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 37), 'datalim_low')
            # Applying the binary operator '<=' (line 242)
            result_le_131983 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 28), '<=', ticks_131981, datalim_low_131982)
            
            
            # Getting the type of 'ticks' (line 242)
            ticks_131984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 53), 'ticks')
            # Getting the type of 'viewlim_low' (line 242)
            viewlim_low_131985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 62), 'viewlim_low')
            # Applying the binary operator '>=' (line 242)
            result_ge_131986 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 53), '>=', ticks_131984, viewlim_low_131985)
            
            # Applying the binary operator '&' (line 242)
            result_and__131987 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 27), '&', result_le_131983, result_ge_131986)
            
            # Assigning a type to the variable 'cond' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'cond', result_and__131987)
            
            # Assigning a Subscript to a Name (line 243):
            
            # Assigning a Subscript to a Name (line 243):
            
            # Obtaining the type of the subscript
            # Getting the type of 'cond' (line 243)
            cond_131988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 37), 'cond')
            # Getting the type of 'ticks' (line 243)
            ticks_131989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 31), 'ticks')
            # Obtaining the member '__getitem__' of a type (line 243)
            getitem___131990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 31), ticks_131989, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 243)
            subscript_call_result_131991 = invoke(stypy.reporting.localization.Localization(__file__, 243, 31), getitem___131990, cond_131988)
            
            # Assigning a type to the variable 'tickvals' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'tickvals', subscript_call_result_131991)
            
            
            # Call to len(...): (line 244)
            # Processing the call arguments (line 244)
            # Getting the type of 'tickvals' (line 244)
            tickvals_131993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 27), 'tickvals', False)
            # Processing the call keyword arguments (line 244)
            kwargs_131994 = {}
            # Getting the type of 'len' (line 244)
            len_131992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 23), 'len', False)
            # Calling len(args, kwargs) (line 244)
            len_call_result_131995 = invoke(stypy.reporting.localization.Localization(__file__, 244, 23), len_131992, *[tickvals_131993], **kwargs_131994)
            
            # Testing the type of an if condition (line 244)
            if_condition_131996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 20), len_call_result_131995)
            # Assigning a type to the variable 'if_condition_131996' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'if_condition_131996', if_condition_131996)
            # SSA begins for if statement (line 244)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 246):
            
            # Assigning a Subscript to a Name (line 246):
            
            # Obtaining the type of the subscript
            int_131997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 39), 'int')
            # Getting the type of 'tickvals' (line 246)
            tickvals_131998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 30), 'tickvals')
            # Obtaining the member '__getitem__' of a type (line 246)
            getitem___131999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 30), tickvals_131998, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 246)
            subscript_call_result_132000 = invoke(stypy.reporting.localization.Localization(__file__, 246, 30), getitem___131999, int_131997)
            
            # Assigning a type to the variable 'low' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), 'low', subscript_call_result_132000)
            # SSA branch for the else part of an if statement (line 244)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 249):
            
            # Assigning a Name to a Name (line 249):
            # Getting the type of 'datalim_low' (line 249)
            datalim_low_132001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 30), 'datalim_low')
            # Assigning a type to the variable 'low' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 24), 'low', datalim_low_132001)
            # SSA join for if statement (line 244)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 250):
            
            # Assigning a Call to a Name (line 250):
            
            # Call to max(...): (line 250)
            # Processing the call arguments (line 250)
            # Getting the type of 'low' (line 250)
            low_132003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 30), 'low', False)
            # Getting the type of 'viewlim_low' (line 250)
            viewlim_low_132004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 35), 'viewlim_low', False)
            # Processing the call keyword arguments (line 250)
            kwargs_132005 = {}
            # Getting the type of 'max' (line 250)
            max_132002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 26), 'max', False)
            # Calling max(args, kwargs) (line 250)
            max_call_result_132006 = invoke(stypy.reporting.localization.Localization(__file__, 250, 26), max_132002, *[low_132003, viewlim_low_132004], **kwargs_132005)
            
            # Assigning a type to the variable 'low' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'low', max_call_result_132006)
            # SSA join for if statement (line 237)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'datalim_high' (line 252)
            datalim_high_132007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'datalim_high')
            # Getting the type of 'viewlim_high' (line 252)
            viewlim_high_132008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 34), 'viewlim_high')
            # Applying the binary operator '>' (line 252)
            result_gt_132009 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 19), '>', datalim_high_132007, viewlim_high_132008)
            
            # Testing the type of an if condition (line 252)
            if_condition_132010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 16), result_gt_132009)
            # Assigning a type to the variable 'if_condition_132010' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'if_condition_132010', if_condition_132010)
            # SSA begins for if statement (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 254):
            
            # Assigning a Name to a Name (line 254):
            # Getting the type of 'viewlim_high' (line 254)
            viewlim_high_132011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 27), 'viewlim_high')
            # Assigning a type to the variable 'high' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'high', viewlim_high_132011)
            # SSA branch for the else part of an if statement (line 252)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 257):
            
            # Assigning a BinOp to a Name (line 257):
            
            # Getting the type of 'ticks' (line 257)
            ticks_132012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'ticks')
            # Getting the type of 'datalim_high' (line 257)
            datalim_high_132013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 37), 'datalim_high')
            # Applying the binary operator '>=' (line 257)
            result_ge_132014 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 28), '>=', ticks_132012, datalim_high_132013)
            
            
            # Getting the type of 'ticks' (line 257)
            ticks_132015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 54), 'ticks')
            # Getting the type of 'viewlim_high' (line 257)
            viewlim_high_132016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 63), 'viewlim_high')
            # Applying the binary operator '<=' (line 257)
            result_le_132017 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 54), '<=', ticks_132015, viewlim_high_132016)
            
            # Applying the binary operator '&' (line 257)
            result_and__132018 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 27), '&', result_ge_132014, result_le_132017)
            
            # Assigning a type to the variable 'cond' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'cond', result_and__132018)
            
            # Assigning a Subscript to a Name (line 258):
            
            # Assigning a Subscript to a Name (line 258):
            
            # Obtaining the type of the subscript
            # Getting the type of 'cond' (line 258)
            cond_132019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 37), 'cond')
            # Getting the type of 'ticks' (line 258)
            ticks_132020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'ticks')
            # Obtaining the member '__getitem__' of a type (line 258)
            getitem___132021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 31), ticks_132020, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 258)
            subscript_call_result_132022 = invoke(stypy.reporting.localization.Localization(__file__, 258, 31), getitem___132021, cond_132019)
            
            # Assigning a type to the variable 'tickvals' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'tickvals', subscript_call_result_132022)
            
            
            # Call to len(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'tickvals' (line 259)
            tickvals_132024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 27), 'tickvals', False)
            # Processing the call keyword arguments (line 259)
            kwargs_132025 = {}
            # Getting the type of 'len' (line 259)
            len_132023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 23), 'len', False)
            # Calling len(args, kwargs) (line 259)
            len_call_result_132026 = invoke(stypy.reporting.localization.Localization(__file__, 259, 23), len_132023, *[tickvals_132024], **kwargs_132025)
            
            # Testing the type of an if condition (line 259)
            if_condition_132027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 20), len_call_result_132026)
            # Assigning a type to the variable 'if_condition_132027' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'if_condition_132027', if_condition_132027)
            # SSA begins for if statement (line 259)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 262):
            
            # Assigning a Subscript to a Name (line 262):
            
            # Obtaining the type of the subscript
            int_132028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 40), 'int')
            # Getting the type of 'tickvals' (line 262)
            tickvals_132029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 31), 'tickvals')
            # Obtaining the member '__getitem__' of a type (line 262)
            getitem___132030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 31), tickvals_132029, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 262)
            subscript_call_result_132031 = invoke(stypy.reporting.localization.Localization(__file__, 262, 31), getitem___132030, int_132028)
            
            # Assigning a type to the variable 'high' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'high', subscript_call_result_132031)
            # SSA branch for the else part of an if statement (line 259)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 265):
            
            # Assigning a Name to a Name (line 265):
            # Getting the type of 'datalim_high' (line 265)
            datalim_high_132032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 31), 'datalim_high')
            # Assigning a type to the variable 'high' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'high', datalim_high_132032)
            # SSA join for if statement (line 259)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 266):
            
            # Assigning a Call to a Name (line 266):
            
            # Call to min(...): (line 266)
            # Processing the call arguments (line 266)
            # Getting the type of 'high' (line 266)
            high_132034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 31), 'high', False)
            # Getting the type of 'viewlim_high' (line 266)
            viewlim_high_132035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'viewlim_high', False)
            # Processing the call keyword arguments (line 266)
            kwargs_132036 = {}
            # Getting the type of 'min' (line 266)
            min_132033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'min', False)
            # Calling min(args, kwargs) (line 266)
            min_call_result_132037 = invoke(stypy.reporting.localization.Localization(__file__, 266, 27), min_132033, *[high_132034, viewlim_high_132035], **kwargs_132036)
            
            # Assigning a type to the variable 'high' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'high', min_call_result_132037)
            # SSA join for if statement (line 252)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 221)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_131822:
                # Runtime conditional SSA for else branch (line 212)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_131821) or more_types_in_union_131822):
            
            # Assigning a Attribute to a Tuple (line 269):
            
            # Assigning a Subscript to a Name (line 269):
            
            # Obtaining the type of the subscript
            int_132038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 12), 'int')
            # Getting the type of 'self' (line 269)
            self_132039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'self')
            # Obtaining the member '_bounds' of a type (line 269)
            _bounds_132040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 24), self_132039, '_bounds')
            # Obtaining the member '__getitem__' of a type (line 269)
            getitem___132041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), _bounds_132040, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 269)
            subscript_call_result_132042 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), getitem___132041, int_132038)
            
            # Assigning a type to the variable 'tuple_var_assignment_131393' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_131393', subscript_call_result_132042)
            
            # Assigning a Subscript to a Name (line 269):
            
            # Obtaining the type of the subscript
            int_132043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 12), 'int')
            # Getting the type of 'self' (line 269)
            self_132044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'self')
            # Obtaining the member '_bounds' of a type (line 269)
            _bounds_132045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 24), self_132044, '_bounds')
            # Obtaining the member '__getitem__' of a type (line 269)
            getitem___132046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), _bounds_132045, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 269)
            subscript_call_result_132047 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), getitem___132046, int_132043)
            
            # Assigning a type to the variable 'tuple_var_assignment_131394' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_131394', subscript_call_result_132047)
            
            # Assigning a Name to a Name (line 269):
            # Getting the type of 'tuple_var_assignment_131393' (line 269)
            tuple_var_assignment_131393_132048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_131393')
            # Assigning a type to the variable 'low' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'low', tuple_var_assignment_131393_132048)
            
            # Assigning a Name to a Name (line 269):
            # Getting the type of 'tuple_var_assignment_131394' (line 269)
            tuple_var_assignment_131394_132049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_131394')
            # Assigning a type to the variable 'high' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 17), 'high', tuple_var_assignment_131394_132049)

            if (may_be_131821 and more_types_in_union_131822):
                # SSA join for if statement (line 212)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 271)
        self_132050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'self')
        # Obtaining the member '_patch_type' of a type (line 271)
        _patch_type_132051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 11), self_132050, '_patch_type')
        unicode_132052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 31), 'unicode', u'arc')
        # Applying the binary operator '==' (line 271)
        result_eq_132053 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 11), '==', _patch_type_132051, unicode_132052)
        
        # Testing the type of an if condition (line 271)
        if_condition_132054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 8), result_eq_132053)
        # Assigning a type to the variable 'if_condition_132054' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'if_condition_132054', if_condition_132054)
        # SSA begins for if statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 272)
        self_132055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'self')
        # Obtaining the member 'spine_type' of a type (line 272)
        spine_type_132056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), self_132055, 'spine_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 272)
        tuple_132057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 272)
        # Adding element type (line 272)
        unicode_132058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 35), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 35), tuple_132057, unicode_132058)
        # Adding element type (line 272)
        unicode_132059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 45), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 35), tuple_132057, unicode_132059)
        
        # Applying the binary operator 'in' (line 272)
        result_contains_132060 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 15), 'in', spine_type_132056, tuple_132057)
        
        # Testing the type of an if condition (line 272)
        if_condition_132061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), result_contains_132060)
        # Assigning a type to the variable 'if_condition_132061' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_132061', if_condition_132061)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to get_theta_direction(...): (line 274)
        # Processing the call keyword arguments (line 274)
        kwargs_132065 = {}
        # Getting the type of 'self' (line 274)
        self_132062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 32), 'self', False)
        # Obtaining the member 'axes' of a type (line 274)
        axes_132063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 32), self_132062, 'axes')
        # Obtaining the member 'get_theta_direction' of a type (line 274)
        get_theta_direction_132064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 32), axes_132063, 'get_theta_direction')
        # Calling get_theta_direction(args, kwargs) (line 274)
        get_theta_direction_call_result_132066 = invoke(stypy.reporting.localization.Localization(__file__, 274, 32), get_theta_direction_132064, *[], **kwargs_132065)
        
        # Assigning a type to the variable 'direction' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'direction', get_theta_direction_call_result_132066)
        # SSA branch for the except part of a try statement (line 273)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 273)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 276):
        
        # Assigning a Num to a Name (line 276):
        int_132067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 32), 'int')
        # Assigning a type to the variable 'direction' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'direction', int_132067)
        # SSA join for try-except statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to get_theta_offset(...): (line 278)
        # Processing the call keyword arguments (line 278)
        kwargs_132071 = {}
        # Getting the type of 'self' (line 278)
        self_132068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'self', False)
        # Obtaining the member 'axes' of a type (line 278)
        axes_132069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 29), self_132068, 'axes')
        # Obtaining the member 'get_theta_offset' of a type (line 278)
        get_theta_offset_132070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 29), axes_132069, 'get_theta_offset')
        # Calling get_theta_offset(args, kwargs) (line 278)
        get_theta_offset_call_result_132072 = invoke(stypy.reporting.localization.Localization(__file__, 278, 29), get_theta_offset_132070, *[], **kwargs_132071)
        
        # Assigning a type to the variable 'offset' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 20), 'offset', get_theta_offset_call_result_132072)
        # SSA branch for the except part of a try statement (line 277)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 277)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 280):
        
        # Assigning a Num to a Name (line 280):
        int_132073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 29), 'int')
        # Assigning a type to the variable 'offset' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 20), 'offset', int_132073)
        # SSA join for try-except statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 281):
        
        # Assigning a BinOp to a Name (line 281):
        # Getting the type of 'low' (line 281)
        low_132074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'low')
        # Getting the type of 'direction' (line 281)
        direction_132075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'direction')
        # Applying the binary operator '*' (line 281)
        result_mul_132076 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 22), '*', low_132074, direction_132075)
        
        # Getting the type of 'offset' (line 281)
        offset_132077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 40), 'offset')
        # Applying the binary operator '+' (line 281)
        result_add_132078 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 22), '+', result_mul_132076, offset_132077)
        
        # Assigning a type to the variable 'low' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'low', result_add_132078)
        
        # Assigning a BinOp to a Name (line 282):
        
        # Assigning a BinOp to a Name (line 282):
        # Getting the type of 'high' (line 282)
        high_132079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 23), 'high')
        # Getting the type of 'direction' (line 282)
        direction_132080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 30), 'direction')
        # Applying the binary operator '*' (line 282)
        result_mul_132081 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 23), '*', high_132079, direction_132080)
        
        # Getting the type of 'offset' (line 282)
        offset_132082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 42), 'offset')
        # Applying the binary operator '+' (line 282)
        result_add_132083 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 23), '+', result_mul_132081, offset_132082)
        
        # Assigning a type to the variable 'high' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'high', result_add_132083)
        
        
        # Getting the type of 'low' (line 283)
        low_132084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'low')
        # Getting the type of 'high' (line 283)
        high_132085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'high')
        # Applying the binary operator '>' (line 283)
        result_gt_132086 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 19), '>', low_132084, high_132085)
        
        # Testing the type of an if condition (line 283)
        if_condition_132087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 16), result_gt_132086)
        # Assigning a type to the variable 'if_condition_132087' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'if_condition_132087', if_condition_132087)
        # SSA begins for if statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 284):
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'high' (line 284)
        high_132088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 32), 'high')
        # Assigning a type to the variable 'tuple_assignment_131395' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'tuple_assignment_131395', high_132088)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'low' (line 284)
        low_132089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 38), 'low')
        # Assigning a type to the variable 'tuple_assignment_131396' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'tuple_assignment_131396', low_132089)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'tuple_assignment_131395' (line 284)
        tuple_assignment_131395_132090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'tuple_assignment_131395')
        # Assigning a type to the variable 'low' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'low', tuple_assignment_131395_132090)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'tuple_assignment_131396' (line 284)
        tuple_assignment_131396_132091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'tuple_assignment_131396')
        # Assigning a type to the variable 'high' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'high', tuple_assignment_131396_132091)
        # SSA join for if statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 286):
        
        # Assigning a Call to a Attribute (line 286):
        
        # Call to arc(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Call to rad2deg(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'low' (line 286)
        low_132097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 55), 'low', False)
        # Processing the call keyword arguments (line 286)
        kwargs_132098 = {}
        # Getting the type of 'np' (line 286)
        np_132095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 44), 'np', False)
        # Obtaining the member 'rad2deg' of a type (line 286)
        rad2deg_132096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 44), np_132095, 'rad2deg')
        # Calling rad2deg(args, kwargs) (line 286)
        rad2deg_call_result_132099 = invoke(stypy.reporting.localization.Localization(__file__, 286, 44), rad2deg_132096, *[low_132097], **kwargs_132098)
        
        
        # Call to rad2deg(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'high' (line 286)
        high_132102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 72), 'high', False)
        # Processing the call keyword arguments (line 286)
        kwargs_132103 = {}
        # Getting the type of 'np' (line 286)
        np_132100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 61), 'np', False)
        # Obtaining the member 'rad2deg' of a type (line 286)
        rad2deg_132101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 61), np_132100, 'rad2deg')
        # Calling rad2deg(args, kwargs) (line 286)
        rad2deg_call_result_132104 = invoke(stypy.reporting.localization.Localization(__file__, 286, 61), rad2deg_132101, *[high_132102], **kwargs_132103)
        
        # Processing the call keyword arguments (line 286)
        kwargs_132105 = {}
        # Getting the type of 'mpath' (line 286)
        mpath_132092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 29), 'mpath', False)
        # Obtaining the member 'Path' of a type (line 286)
        Path_132093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 29), mpath_132092, 'Path')
        # Obtaining the member 'arc' of a type (line 286)
        arc_132094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 29), Path_132093, 'arc')
        # Calling arc(args, kwargs) (line 286)
        arc_call_result_132106 = invoke(stypy.reporting.localization.Localization(__file__, 286, 29), arc_132094, *[rad2deg_call_result_132099, rad2deg_call_result_132104], **kwargs_132105)
        
        # Getting the type of 'self' (line 286)
        self_132107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'self')
        # Setting the type of the member '_path' of a type (line 286)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 16), self_132107, '_path', arc_call_result_132106)
        
        
        # Getting the type of 'self' (line 288)
        self_132108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'self')
        # Obtaining the member 'spine_type' of a type (line 288)
        spine_type_132109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 19), self_132108, 'spine_type')
        unicode_132110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 38), 'unicode', u'bottom')
        # Applying the binary operator '==' (line 288)
        result_eq_132111 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 19), '==', spine_type_132109, unicode_132110)
        
        # Testing the type of an if condition (line 288)
        if_condition_132112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 16), result_eq_132111)
        # Assigning a type to the variable 'if_condition_132112' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'if_condition_132112', if_condition_132112)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 289):
        
        # Assigning a Subscript to a Name (line 289):
        
        # Obtaining the type of the subscript
        int_132113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 20), 'int')
        # Getting the type of 'self' (line 289)
        self_132114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 33), 'self')
        # Obtaining the member 'axes' of a type (line 289)
        axes_132115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 33), self_132114, 'axes')
        # Obtaining the member 'viewLim' of a type (line 289)
        viewLim_132116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 33), axes_132115, 'viewLim')
        # Obtaining the member 'intervaly' of a type (line 289)
        intervaly_132117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 33), viewLim_132116, 'intervaly')
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___132118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), intervaly_132117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_132119 = invoke(stypy.reporting.localization.Localization(__file__, 289, 20), getitem___132118, int_132113)
        
        # Assigning a type to the variable 'tuple_var_assignment_131397' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'tuple_var_assignment_131397', subscript_call_result_132119)
        
        # Assigning a Subscript to a Name (line 289):
        
        # Obtaining the type of the subscript
        int_132120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 20), 'int')
        # Getting the type of 'self' (line 289)
        self_132121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 33), 'self')
        # Obtaining the member 'axes' of a type (line 289)
        axes_132122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 33), self_132121, 'axes')
        # Obtaining the member 'viewLim' of a type (line 289)
        viewLim_132123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 33), axes_132122, 'viewLim')
        # Obtaining the member 'intervaly' of a type (line 289)
        intervaly_132124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 33), viewLim_132123, 'intervaly')
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___132125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), intervaly_132124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_132126 = invoke(stypy.reporting.localization.Localization(__file__, 289, 20), getitem___132125, int_132120)
        
        # Assigning a type to the variable 'tuple_var_assignment_131398' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'tuple_var_assignment_131398', subscript_call_result_132126)
        
        # Assigning a Name to a Name (line 289):
        # Getting the type of 'tuple_var_assignment_131397' (line 289)
        tuple_var_assignment_131397_132127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'tuple_var_assignment_131397')
        # Assigning a type to the variable 'rmin' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'rmin', tuple_var_assignment_131397_132127)
        
        # Assigning a Name to a Name (line 289):
        # Getting the type of 'tuple_var_assignment_131398' (line 289)
        tuple_var_assignment_131398_132128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'tuple_var_assignment_131398')
        # Assigning a type to the variable 'rmax' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'rmax', tuple_var_assignment_131398_132128)
        
        
        # SSA begins for try-except statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 291):
        
        # Assigning a Call to a Name (line 291):
        
        # Call to get_rorigin(...): (line 291)
        # Processing the call keyword arguments (line 291)
        kwargs_132132 = {}
        # Getting the type of 'self' (line 291)
        self_132129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 34), 'self', False)
        # Obtaining the member 'axes' of a type (line 291)
        axes_132130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 34), self_132129, 'axes')
        # Obtaining the member 'get_rorigin' of a type (line 291)
        get_rorigin_132131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 34), axes_132130, 'get_rorigin')
        # Calling get_rorigin(args, kwargs) (line 291)
        get_rorigin_call_result_132133 = invoke(stypy.reporting.localization.Localization(__file__, 291, 34), get_rorigin_132131, *[], **kwargs_132132)
        
        # Assigning a type to the variable 'rorigin' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'rorigin', get_rorigin_call_result_132133)
        # SSA branch for the except part of a try statement (line 290)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 290)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 293):
        
        # Assigning a Name to a Name (line 293):
        # Getting the type of 'rmin' (line 293)
        rmin_132134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 34), 'rmin')
        # Assigning a type to the variable 'rorigin' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'rorigin', rmin_132134)
        # SSA join for try-except statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 294):
        
        # Assigning a BinOp to a Name (line 294):
        # Getting the type of 'rmin' (line 294)
        rmin_132135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'rmin')
        # Getting the type of 'rorigin' (line 294)
        rorigin_132136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 46), 'rorigin')
        # Applying the binary operator '-' (line 294)
        result_sub_132137 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 39), '-', rmin_132135, rorigin_132136)
        
        # Getting the type of 'rmax' (line 294)
        rmax_132138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 58), 'rmax')
        # Getting the type of 'rorigin' (line 294)
        rorigin_132139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 65), 'rorigin')
        # Applying the binary operator '-' (line 294)
        result_sub_132140 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 58), '-', rmax_132138, rorigin_132139)
        
        # Applying the binary operator 'div' (line 294)
        result_div_132141 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 38), 'div', result_sub_132137, result_sub_132140)
        
        # Assigning a type to the variable 'scaled_diameter' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 20), 'scaled_diameter', result_div_132141)
        
        # Assigning a Name to a Attribute (line 295):
        
        # Assigning a Name to a Attribute (line 295):
        # Getting the type of 'scaled_diameter' (line 295)
        scaled_diameter_132142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'scaled_diameter')
        # Getting the type of 'self' (line 295)
        self_132143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'self')
        # Setting the type of the member '_height' of a type (line 295)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 20), self_132143, '_height', scaled_diameter_132142)
        
        # Assigning a Name to a Attribute (line 296):
        
        # Assigning a Name to a Attribute (line 296):
        # Getting the type of 'scaled_diameter' (line 296)
        scaled_diameter_132144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'scaled_diameter')
        # Getting the type of 'self' (line 296)
        self_132145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'self')
        # Setting the type of the member '_width' of a type (line 296)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 20), self_132145, '_width', scaled_diameter_132144)
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 272)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 299)
        # Processing the call arguments (line 299)
        unicode_132147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 33), 'unicode', u'unable to set bounds for spine "%s"')
        # Getting the type of 'self' (line 300)
        self_132148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 33), 'self', False)
        # Obtaining the member 'spine_type' of a type (line 300)
        spine_type_132149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 33), self_132148, 'spine_type')
        # Applying the binary operator '%' (line 299)
        result_mod_132150 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 33), '%', unicode_132147, spine_type_132149)
        
        # Processing the call keyword arguments (line 299)
        kwargs_132151 = {}
        # Getting the type of 'ValueError' (line 299)
        ValueError_132146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 299)
        ValueError_call_result_132152 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), ValueError_132146, *[result_mod_132150], **kwargs_132151)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 299, 16), ValueError_call_result_132152, 'raise parameter', BaseException)
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 271)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 302):
        
        # Assigning a Attribute to a Name (line 302):
        # Getting the type of 'self' (line 302)
        self_132153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 17), 'self')
        # Obtaining the member '_path' of a type (line 302)
        _path_132154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 17), self_132153, '_path')
        # Obtaining the member 'vertices' of a type (line 302)
        vertices_132155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 17), _path_132154, 'vertices')
        # Assigning a type to the variable 'v1' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'v1', vertices_132155)
        # Evaluating assert statement condition
        
        # Getting the type of 'v1' (line 303)
        v1_132156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'v1')
        # Obtaining the member 'shape' of a type (line 303)
        shape_132157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 19), v1_132156, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 303)
        tuple_132158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 303)
        # Adding element type (line 303)
        int_132159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 32), tuple_132158, int_132159)
        # Adding element type (line 303)
        int_132160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 32), tuple_132158, int_132160)
        
        # Applying the binary operator '==' (line 303)
        result_eq_132161 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 19), '==', shape_132157, tuple_132158)
        
        
        
        # Getting the type of 'self' (line 304)
        self_132162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'self')
        # Obtaining the member 'spine_type' of a type (line 304)
        spine_type_132163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 15), self_132162, 'spine_type')
        
        # Obtaining an instance of the builtin type 'list' (line 304)
        list_132164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 304)
        # Adding element type (line 304)
        unicode_132165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 35), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 34), list_132164, unicode_132165)
        # Adding element type (line 304)
        unicode_132166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 43), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 34), list_132164, unicode_132166)
        
        # Applying the binary operator 'in' (line 304)
        result_contains_132167 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 15), 'in', spine_type_132163, list_132164)
        
        # Testing the type of an if condition (line 304)
        if_condition_132168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 12), result_contains_132167)
        # Assigning a type to the variable 'if_condition_132168' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'if_condition_132168', if_condition_132168)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 305):
        
        # Assigning a Name to a Subscript (line 305):
        # Getting the type of 'low' (line 305)
        low_132169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 27), 'low')
        # Getting the type of 'v1' (line 305)
        v1_132170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'v1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 305)
        tuple_132171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 305)
        # Adding element type (line 305)
        int_132172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 19), tuple_132171, int_132172)
        # Adding element type (line 305)
        int_132173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 19), tuple_132171, int_132173)
        
        # Storing an element on a container (line 305)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 16), v1_132170, (tuple_132171, low_132169))
        
        # Assigning a Name to a Subscript (line 306):
        
        # Assigning a Name to a Subscript (line 306):
        # Getting the type of 'high' (line 306)
        high_132174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 27), 'high')
        # Getting the type of 'v1' (line 306)
        v1_132175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'v1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 306)
        tuple_132176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 306)
        # Adding element type (line 306)
        int_132177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 19), tuple_132176, int_132177)
        # Adding element type (line 306)
        int_132178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 19), tuple_132176, int_132178)
        
        # Storing an element on a container (line 306)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 16), v1_132175, (tuple_132176, high_132174))
        # SSA branch for the else part of an if statement (line 304)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 307)
        self_132179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'self')
        # Obtaining the member 'spine_type' of a type (line 307)
        spine_type_132180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 17), self_132179, 'spine_type')
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_132181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        unicode_132182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 37), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 36), list_132181, unicode_132182)
        # Adding element type (line 307)
        unicode_132183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 47), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 36), list_132181, unicode_132183)
        
        # Applying the binary operator 'in' (line 307)
        result_contains_132184 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 17), 'in', spine_type_132180, list_132181)
        
        # Testing the type of an if condition (line 307)
        if_condition_132185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 17), result_contains_132184)
        # Assigning a type to the variable 'if_condition_132185' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'if_condition_132185', if_condition_132185)
        # SSA begins for if statement (line 307)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 308):
        
        # Assigning a Name to a Subscript (line 308):
        # Getting the type of 'low' (line 308)
        low_132186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 27), 'low')
        # Getting the type of 'v1' (line 308)
        v1_132187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'v1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 308)
        tuple_132188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 308)
        # Adding element type (line 308)
        int_132189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 19), tuple_132188, int_132189)
        # Adding element type (line 308)
        int_132190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 19), tuple_132188, int_132190)
        
        # Storing an element on a container (line 308)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 16), v1_132187, (tuple_132188, low_132186))
        
        # Assigning a Name to a Subscript (line 309):
        
        # Assigning a Name to a Subscript (line 309):
        # Getting the type of 'high' (line 309)
        high_132191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 27), 'high')
        # Getting the type of 'v1' (line 309)
        v1_132192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'v1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 309)
        tuple_132193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 309)
        # Adding element type (line 309)
        int_132194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 19), tuple_132193, int_132194)
        # Adding element type (line 309)
        int_132195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 19), tuple_132193, int_132195)
        
        # Storing an element on a container (line 309)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 16), v1_132192, (tuple_132193, high_132191))
        # SSA branch for the else part of an if statement (line 307)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 311)
        # Processing the call arguments (line 311)
        unicode_132197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 33), 'unicode', u'unable to set bounds for spine "%s"')
        # Getting the type of 'self' (line 312)
        self_132198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 33), 'self', False)
        # Obtaining the member 'spine_type' of a type (line 312)
        spine_type_132199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 33), self_132198, 'spine_type')
        # Applying the binary operator '%' (line 311)
        result_mod_132200 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 33), '%', unicode_132197, spine_type_132199)
        
        # Processing the call keyword arguments (line 311)
        kwargs_132201 = {}
        # Getting the type of 'ValueError' (line 311)
        ValueError_132196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 311)
        ValueError_call_result_132202 = invoke(stypy.reporting.localization.Localization(__file__, 311, 22), ValueError_132196, *[result_mod_132200], **kwargs_132201)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 311, 16), ValueError_call_result_132202, 'raise parameter', BaseException)
        # SSA join for if statement (line 307)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_adjust_location(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjust_location' in the type store
        # Getting the type of 'stypy_return_type' (line 206)
        stypy_return_type_132203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132203)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjust_location'
        return stypy_return_type_132203


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.draw.__dict__.__setitem__('stypy_localization', localization)
        Spine.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.draw.__dict__.__setitem__('stypy_function_name', 'Spine.draw')
        Spine.draw.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Spine.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.draw', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw(...)' code ##################

        
        # Call to _adjust_location(...): (line 316)
        # Processing the call keyword arguments (line 316)
        kwargs_132206 = {}
        # Getting the type of 'self' (line 316)
        self_132204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self', False)
        # Obtaining the member '_adjust_location' of a type (line 316)
        _adjust_location_132205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_132204, '_adjust_location')
        # Calling _adjust_location(args, kwargs) (line 316)
        _adjust_location_call_result_132207 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), _adjust_location_132205, *[], **kwargs_132206)
        
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to draw(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'renderer' (line 317)
        renderer_132214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 38), 'renderer', False)
        # Processing the call keyword arguments (line 317)
        kwargs_132215 = {}
        
        # Call to super(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'Spine' (line 317)
        Spine_132209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'Spine', False)
        # Getting the type of 'self' (line 317)
        self_132210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 27), 'self', False)
        # Processing the call keyword arguments (line 317)
        kwargs_132211 = {}
        # Getting the type of 'super' (line 317)
        super_132208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 14), 'super', False)
        # Calling super(args, kwargs) (line 317)
        super_call_result_132212 = invoke(stypy.reporting.localization.Localization(__file__, 317, 14), super_132208, *[Spine_132209, self_132210], **kwargs_132211)
        
        # Obtaining the member 'draw' of a type (line 317)
        draw_132213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 14), super_call_result_132212, 'draw')
        # Calling draw(args, kwargs) (line 317)
        draw_call_result_132216 = invoke(stypy.reporting.localization.Localization(__file__, 317, 14), draw_132213, *[renderer_132214], **kwargs_132215)
        
        # Assigning a type to the variable 'ret' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'ret', draw_call_result_132216)
        
        # Assigning a Name to a Attribute (line 318):
        
        # Assigning a Name to a Attribute (line 318):
        # Getting the type of 'False' (line 318)
        False_132217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'False')
        # Getting the type of 'self' (line 318)
        self_132218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 318)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), self_132218, 'stale', False_132217)
        # Getting the type of 'ret' (line 319)
        ret_132219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'stypy_return_type', ret_132219)
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_132220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132220)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_132220


    @norecursion
    def _calc_offset_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_calc_offset_transform'
        module_type_store = module_type_store.open_function_context('_calc_offset_transform', 321, 4, False)
        # Assigning a type to the variable 'self' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_localization', localization)
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_function_name', 'Spine._calc_offset_transform')
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_param_names_list', [])
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine._calc_offset_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine._calc_offset_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_calc_offset_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_calc_offset_transform(...)' code ##################

        unicode_132221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 8), 'unicode', u'calculate the offset transform performed by the spine')
        
        # Call to _ensure_position_is_set(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_132224 = {}
        # Getting the type of 'self' (line 323)
        self_132222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self', False)
        # Obtaining the member '_ensure_position_is_set' of a type (line 323)
        _ensure_position_is_set_132223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_132222, '_ensure_position_is_set')
        # Calling _ensure_position_is_set(args, kwargs) (line 323)
        _ensure_position_is_set_call_result_132225 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), _ensure_position_is_set_132223, *[], **kwargs_132224)
        
        
        # Assigning a Attribute to a Name (line 324):
        
        # Assigning a Attribute to a Name (line 324):
        # Getting the type of 'self' (line 324)
        self_132226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'self')
        # Obtaining the member '_position' of a type (line 324)
        _position_132227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 19), self_132226, '_position')
        # Assigning a type to the variable 'position' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'position', _position_132227)
        
        
        # Call to isinstance(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'position' (line 325)
        position_132229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 22), 'position', False)
        # Getting the type of 'six' (line 325)
        six_132230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 32), 'six', False)
        # Obtaining the member 'string_types' of a type (line 325)
        string_types_132231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 32), six_132230, 'string_types')
        # Processing the call keyword arguments (line 325)
        kwargs_132232 = {}
        # Getting the type of 'isinstance' (line 325)
        isinstance_132228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 325)
        isinstance_call_result_132233 = invoke(stypy.reporting.localization.Localization(__file__, 325, 11), isinstance_132228, *[position_132229, string_types_132231], **kwargs_132232)
        
        # Testing the type of an if condition (line 325)
        if_condition_132234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 8), isinstance_call_result_132233)
        # Assigning a type to the variable 'if_condition_132234' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'if_condition_132234', if_condition_132234)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'position' (line 326)
        position_132235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'position')
        unicode_132236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 27), 'unicode', u'center')
        # Applying the binary operator '==' (line 326)
        result_eq_132237 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 15), '==', position_132235, unicode_132236)
        
        # Testing the type of an if condition (line 326)
        if_condition_132238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 12), result_eq_132237)
        # Assigning a type to the variable 'if_condition_132238' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'if_condition_132238', if_condition_132238)
        # SSA begins for if statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 327):
        
        # Assigning a Tuple to a Name (line 327):
        
        # Obtaining an instance of the builtin type 'tuple' (line 327)
        tuple_132239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 327)
        # Adding element type (line 327)
        unicode_132240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 28), 'unicode', u'axes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 28), tuple_132239, unicode_132240)
        # Adding element type (line 327)
        float_132241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 28), tuple_132239, float_132241)
        
        # Assigning a type to the variable 'position' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'position', tuple_132239)
        # SSA branch for the else part of an if statement (line 326)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'position' (line 328)
        position_132242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'position')
        unicode_132243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 29), 'unicode', u'zero')
        # Applying the binary operator '==' (line 328)
        result_eq_132244 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 17), '==', position_132242, unicode_132243)
        
        # Testing the type of an if condition (line 328)
        if_condition_132245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 17), result_eq_132244)
        # Assigning a type to the variable 'if_condition_132245' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'if_condition_132245', if_condition_132245)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 329):
        
        # Assigning a Tuple to a Name (line 329):
        
        # Obtaining an instance of the builtin type 'tuple' (line 329)
        tuple_132246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 329)
        # Adding element type (line 329)
        unicode_132247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 28), 'unicode', u'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 28), tuple_132246, unicode_132247)
        # Adding element type (line 329)
        int_132248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 28), tuple_132246, int_132248)
        
        # Assigning a type to the variable 'position' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'position', tuple_132246)
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 326)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'position' (line 330)
        position_132250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'position', False)
        # Processing the call keyword arguments (line 330)
        kwargs_132251 = {}
        # Getting the type of 'len' (line 330)
        len_132249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'len', False)
        # Calling len(args, kwargs) (line 330)
        len_call_result_132252 = invoke(stypy.reporting.localization.Localization(__file__, 330, 15), len_132249, *[position_132250], **kwargs_132251)
        
        int_132253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 32), 'int')
        # Applying the binary operator '==' (line 330)
        result_eq_132254 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 15), '==', len_call_result_132252, int_132253)
        
        
        # Assigning a Name to a Tuple (line 331):
        
        # Assigning a Subscript to a Name (line 331):
        
        # Obtaining the type of the subscript
        int_132255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 8), 'int')
        # Getting the type of 'position' (line 331)
        position_132256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 32), 'position')
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___132257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), position_132256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 331)
        subscript_call_result_132258 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), getitem___132257, int_132255)
        
        # Assigning a type to the variable 'tuple_var_assignment_131399' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'tuple_var_assignment_131399', subscript_call_result_132258)
        
        # Assigning a Subscript to a Name (line 331):
        
        # Obtaining the type of the subscript
        int_132259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 8), 'int')
        # Getting the type of 'position' (line 331)
        position_132260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 32), 'position')
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___132261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), position_132260, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 331)
        subscript_call_result_132262 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), getitem___132261, int_132259)
        
        # Assigning a type to the variable 'tuple_var_assignment_131400' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'tuple_var_assignment_131400', subscript_call_result_132262)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'tuple_var_assignment_131399' (line 331)
        tuple_var_assignment_131399_132263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'tuple_var_assignment_131399')
        # Assigning a type to the variable 'position_type' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'position_type', tuple_var_assignment_131399_132263)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'tuple_var_assignment_131400' (line 331)
        tuple_var_assignment_131400_132264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'tuple_var_assignment_131400')
        # Assigning a type to the variable 'amount' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 23), 'amount', tuple_var_assignment_131400_132264)
        # Evaluating assert statement condition
        
        # Getting the type of 'position_type' (line 332)
        position_type_132265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'position_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 332)
        tuple_132266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 332)
        # Adding element type (line 332)
        unicode_132267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 33), 'unicode', u'axes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 33), tuple_132266, unicode_132267)
        # Adding element type (line 332)
        unicode_132268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 41), 'unicode', u'outward')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 33), tuple_132266, unicode_132268)
        # Adding element type (line 332)
        unicode_132269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 52), 'unicode', u'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 33), tuple_132266, unicode_132269)
        
        # Applying the binary operator 'in' (line 332)
        result_contains_132270 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 15), 'in', position_type_132265, tuple_132266)
        
        
        
        # Getting the type of 'position_type' (line 333)
        position_type_132271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'position_type')
        unicode_132272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 28), 'unicode', u'outward')
        # Applying the binary operator '==' (line 333)
        result_eq_132273 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 11), '==', position_type_132271, unicode_132272)
        
        # Testing the type of an if condition (line 333)
        if_condition_132274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), result_eq_132273)
        # Assigning a type to the variable 'if_condition_132274' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_132274', if_condition_132274)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'amount' (line 334)
        amount_132275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'amount')
        int_132276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 25), 'int')
        # Applying the binary operator '==' (line 334)
        result_eq_132277 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 15), '==', amount_132275, int_132276)
        
        # Testing the type of an if condition (line 334)
        if_condition_132278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 12), result_eq_132277)
        # Assigning a type to the variable 'if_condition_132278' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'if_condition_132278', if_condition_132278)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 336):
        
        # Assigning a Tuple to a Attribute (line 336):
        
        # Obtaining an instance of the builtin type 'tuple' (line 336)
        tuple_132279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 336)
        # Adding element type (line 336)
        unicode_132280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 41), 'unicode', u'identity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 41), tuple_132279, unicode_132280)
        # Adding element type (line 336)
        
        # Call to IdentityTransform(...): (line 337)
        # Processing the call keyword arguments (line 337)
        kwargs_132283 = {}
        # Getting the type of 'mtransforms' (line 337)
        mtransforms_132281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 41), 'mtransforms', False)
        # Obtaining the member 'IdentityTransform' of a type (line 337)
        IdentityTransform_132282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 41), mtransforms_132281, 'IdentityTransform')
        # Calling IdentityTransform(args, kwargs) (line 337)
        IdentityTransform_call_result_132284 = invoke(stypy.reporting.localization.Localization(__file__, 337, 41), IdentityTransform_132282, *[], **kwargs_132283)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 41), tuple_132279, IdentityTransform_call_result_132284)
        
        # Getting the type of 'self' (line 336)
        self_132285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 336)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 16), self_132285, '_spine_transform', tuple_132279)
        # SSA branch for the else part of an if statement (line 334)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 338)
        self_132286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 17), 'self')
        # Obtaining the member 'spine_type' of a type (line 338)
        spine_type_132287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 17), self_132286, 'spine_type')
        
        # Obtaining an instance of the builtin type 'list' (line 338)
        list_132288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 338)
        # Adding element type (line 338)
        unicode_132289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 37), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 36), list_132288, unicode_132289)
        # Adding element type (line 338)
        unicode_132290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 45), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 36), list_132288, unicode_132290)
        # Adding element type (line 338)
        unicode_132291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 54), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 36), list_132288, unicode_132291)
        # Adding element type (line 338)
        unicode_132292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 61), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 36), list_132288, unicode_132292)
        
        # Applying the binary operator 'in' (line 338)
        result_contains_132293 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 17), 'in', spine_type_132287, list_132288)
        
        # Testing the type of an if condition (line 338)
        if_condition_132294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 17), result_contains_132293)
        # Assigning a type to the variable 'if_condition_132294' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 17), 'if_condition_132294', if_condition_132294)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 339):
        
        # Assigning a Subscript to a Name (line 339):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 343)
        self_132295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 32), 'self')
        # Obtaining the member 'spine_type' of a type (line 343)
        spine_type_132296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 32), self_132295, 'spine_type')
        
        # Obtaining an instance of the builtin type 'dict' (line 339)
        dict_132297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 29), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 339)
        # Adding element type (key, value) (line 339)
        unicode_132298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 30), 'unicode', u'left')
        
        # Obtaining an instance of the builtin type 'tuple' (line 339)
        tuple_132299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 339)
        # Adding element type (line 339)
        int_132300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 39), tuple_132299, int_132300)
        # Adding element type (line 339)
        int_132301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 39), tuple_132299, int_132301)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 29), dict_132297, (unicode_132298, tuple_132299))
        # Adding element type (key, value) (line 339)
        unicode_132302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 30), 'unicode', u'right')
        
        # Obtaining an instance of the builtin type 'tuple' (line 340)
        tuple_132303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 340)
        # Adding element type (line 340)
        int_132304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 40), tuple_132303, int_132304)
        # Adding element type (line 340)
        int_132305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 40), tuple_132303, int_132305)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 29), dict_132297, (unicode_132302, tuple_132303))
        # Adding element type (key, value) (line 339)
        unicode_132306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 30), 'unicode', u'bottom')
        
        # Obtaining an instance of the builtin type 'tuple' (line 341)
        tuple_132307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 341)
        # Adding element type (line 341)
        int_132308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 41), tuple_132307, int_132308)
        # Adding element type (line 341)
        int_132309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 41), tuple_132307, int_132309)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 29), dict_132297, (unicode_132306, tuple_132307))
        # Adding element type (key, value) (line 339)
        unicode_132310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 30), 'unicode', u'top')
        
        # Obtaining an instance of the builtin type 'tuple' (line 342)
        tuple_132311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 342)
        # Adding element type (line 342)
        int_132312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 38), tuple_132311, int_132312)
        # Adding element type (line 342)
        int_132313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 38), tuple_132311, int_132313)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 29), dict_132297, (unicode_132310, tuple_132311))
        
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___132314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 29), dict_132297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 339)
        subscript_call_result_132315 = invoke(stypy.reporting.localization.Localization(__file__, 339, 29), getitem___132314, spine_type_132296)
        
        # Assigning a type to the variable 'offset_vec' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'offset_vec', subscript_call_result_132315)
        
        # Assigning a BinOp to a Name (line 345):
        
        # Assigning a BinOp to a Name (line 345):
        # Getting the type of 'amount' (line 345)
        amount_132316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 27), 'amount')
        
        # Obtaining the type of the subscript
        int_132317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 47), 'int')
        # Getting the type of 'offset_vec' (line 345)
        offset_vec_132318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 36), 'offset_vec')
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___132319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 36), offset_vec_132318, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_132320 = invoke(stypy.reporting.localization.Localization(__file__, 345, 36), getitem___132319, int_132317)
        
        # Applying the binary operator '*' (line 345)
        result_mul_132321 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 27), '*', amount_132316, subscript_call_result_132320)
        
        float_132322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 52), 'float')
        # Applying the binary operator 'div' (line 345)
        result_div_132323 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 50), 'div', result_mul_132321, float_132322)
        
        # Assigning a type to the variable 'offset_x' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 16), 'offset_x', result_div_132323)
        
        # Assigning a BinOp to a Name (line 346):
        
        # Assigning a BinOp to a Name (line 346):
        # Getting the type of 'amount' (line 346)
        amount_132324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 27), 'amount')
        
        # Obtaining the type of the subscript
        int_132325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 47), 'int')
        # Getting the type of 'offset_vec' (line 346)
        offset_vec_132326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 36), 'offset_vec')
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___132327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 36), offset_vec_132326, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_132328 = invoke(stypy.reporting.localization.Localization(__file__, 346, 36), getitem___132327, int_132325)
        
        # Applying the binary operator '*' (line 346)
        result_mul_132329 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 27), '*', amount_132324, subscript_call_result_132328)
        
        float_132330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 52), 'float')
        # Applying the binary operator 'div' (line 346)
        result_div_132331 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 50), 'div', result_mul_132329, float_132330)
        
        # Assigning a type to the variable 'offset_y' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'offset_y', result_div_132331)
        
        # Assigning a Tuple to a Attribute (line 347):
        
        # Assigning a Tuple to a Attribute (line 347):
        
        # Obtaining an instance of the builtin type 'tuple' (line 347)
        tuple_132332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 347)
        # Adding element type (line 347)
        unicode_132333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 41), 'unicode', u'post')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 41), tuple_132332, unicode_132333)
        # Adding element type (line 347)
        
        # Call to ScaledTranslation(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'offset_x' (line 349)
        offset_x_132336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 45), 'offset_x', False)
        # Getting the type of 'offset_y' (line 350)
        offset_y_132337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 45), 'offset_y', False)
        # Getting the type of 'self' (line 351)
        self_132338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 45), 'self', False)
        # Obtaining the member 'figure' of a type (line 351)
        figure_132339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 45), self_132338, 'figure')
        # Obtaining the member 'dpi_scale_trans' of a type (line 351)
        dpi_scale_trans_132340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 45), figure_132339, 'dpi_scale_trans')
        # Processing the call keyword arguments (line 348)
        kwargs_132341 = {}
        # Getting the type of 'mtransforms' (line 348)
        mtransforms_132334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 41), 'mtransforms', False)
        # Obtaining the member 'ScaledTranslation' of a type (line 348)
        ScaledTranslation_132335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 41), mtransforms_132334, 'ScaledTranslation')
        # Calling ScaledTranslation(args, kwargs) (line 348)
        ScaledTranslation_call_result_132342 = invoke(stypy.reporting.localization.Localization(__file__, 348, 41), ScaledTranslation_132335, *[offset_x_132336, offset_y_132337, dpi_scale_trans_132340], **kwargs_132341)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 41), tuple_132332, ScaledTranslation_call_result_132342)
        
        # Getting the type of 'self' (line 347)
        self_132343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 347)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 16), self_132343, '_spine_transform', tuple_132332)
        # SSA branch for the else part of an if statement (line 338)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 353)
        # Processing the call arguments (line 353)
        unicode_132346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 30), 'unicode', u'unknown spine type "%s": no spine offset performed')
        # Getting the type of 'self' (line 354)
        self_132347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 51), 'self', False)
        # Obtaining the member 'spine_type' of a type (line 354)
        spine_type_132348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 51), self_132347, 'spine_type')
        # Applying the binary operator '%' (line 353)
        result_mod_132349 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 30), '%', unicode_132346, spine_type_132348)
        
        # Processing the call keyword arguments (line 353)
        kwargs_132350 = {}
        # Getting the type of 'warnings' (line 353)
        warnings_132344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 353)
        warn_132345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), warnings_132344, 'warn')
        # Calling warn(args, kwargs) (line 353)
        warn_call_result_132351 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), warn_132345, *[result_mod_132349], **kwargs_132350)
        
        
        # Assigning a Tuple to a Attribute (line 355):
        
        # Assigning a Tuple to a Attribute (line 355):
        
        # Obtaining an instance of the builtin type 'tuple' (line 355)
        tuple_132352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 355)
        # Adding element type (line 355)
        unicode_132353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 41), 'unicode', u'identity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 41), tuple_132352, unicode_132353)
        # Adding element type (line 355)
        
        # Call to IdentityTransform(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_132356 = {}
        # Getting the type of 'mtransforms' (line 356)
        mtransforms_132354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 41), 'mtransforms', False)
        # Obtaining the member 'IdentityTransform' of a type (line 356)
        IdentityTransform_132355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 41), mtransforms_132354, 'IdentityTransform')
        # Calling IdentityTransform(args, kwargs) (line 356)
        IdentityTransform_call_result_132357 = invoke(stypy.reporting.localization.Localization(__file__, 356, 41), IdentityTransform_132355, *[], **kwargs_132356)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 41), tuple_132352, IdentityTransform_call_result_132357)
        
        # Getting the type of 'self' (line 355)
        self_132358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), self_132358, '_spine_transform', tuple_132352)
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 333)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'position_type' (line 357)
        position_type_132359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'position_type')
        unicode_132360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 30), 'unicode', u'axes')
        # Applying the binary operator '==' (line 357)
        result_eq_132361 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 13), '==', position_type_132359, unicode_132360)
        
        # Testing the type of an if condition (line 357)
        if_condition_132362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 13), result_eq_132361)
        # Assigning a type to the variable 'if_condition_132362' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'if_condition_132362', if_condition_132362)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 358)
        self_132363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 15), 'self')
        # Obtaining the member 'spine_type' of a type (line 358)
        spine_type_132364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 15), self_132363, 'spine_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 358)
        tuple_132365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 358)
        # Adding element type (line 358)
        unicode_132366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 35), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 35), tuple_132365, unicode_132366)
        # Adding element type (line 358)
        unicode_132367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 43), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 35), tuple_132365, unicode_132367)
        
        # Applying the binary operator 'in' (line 358)
        result_contains_132368 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 15), 'in', spine_type_132364, tuple_132365)
        
        # Testing the type of an if condition (line 358)
        if_condition_132369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 358, 12), result_contains_132368)
        # Assigning a type to the variable 'if_condition_132369' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'if_condition_132369', if_condition_132369)
        # SSA begins for if statement (line 358)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 359):
        
        # Assigning a Tuple to a Attribute (line 359):
        
        # Obtaining an instance of the builtin type 'tuple' (line 359)
        tuple_132370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 359)
        # Adding element type (line 359)
        unicode_132371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 41), 'unicode', u'pre')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 41), tuple_132370, unicode_132371)
        # Adding element type (line 359)
        
        # Call to from_values(...): (line 360)
        # Processing the call arguments (line 360)
        int_132375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 45), 'int')
        int_132376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 48), 'int')
        int_132377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 51), 'int')
        int_132378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 54), 'int')
        # Getting the type of 'amount' (line 363)
        amount_132379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 57), 'amount', False)
        int_132380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 65), 'int')
        # Processing the call keyword arguments (line 360)
        kwargs_132381 = {}
        # Getting the type of 'mtransforms' (line 360)
        mtransforms_132372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 41), 'mtransforms', False)
        # Obtaining the member 'Affine2D' of a type (line 360)
        Affine2D_132373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 41), mtransforms_132372, 'Affine2D')
        # Obtaining the member 'from_values' of a type (line 360)
        from_values_132374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 41), Affine2D_132373, 'from_values')
        # Calling from_values(args, kwargs) (line 360)
        from_values_call_result_132382 = invoke(stypy.reporting.localization.Localization(__file__, 360, 41), from_values_132374, *[int_132375, int_132376, int_132377, int_132378, amount_132379, int_132380], **kwargs_132381)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 41), tuple_132370, from_values_call_result_132382)
        
        # Getting the type of 'self' (line 359)
        self_132383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 359)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), self_132383, '_spine_transform', tuple_132370)
        # SSA branch for the else part of an if statement (line 358)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 364)
        self_132384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'self')
        # Obtaining the member 'spine_type' of a type (line 364)
        spine_type_132385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 17), self_132384, 'spine_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_132386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        unicode_132387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 37), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 37), tuple_132386, unicode_132387)
        # Adding element type (line 364)
        unicode_132388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 47), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 37), tuple_132386, unicode_132388)
        
        # Applying the binary operator 'in' (line 364)
        result_contains_132389 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 17), 'in', spine_type_132385, tuple_132386)
        
        # Testing the type of an if condition (line 364)
        if_condition_132390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 17), result_contains_132389)
        # Assigning a type to the variable 'if_condition_132390' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'if_condition_132390', if_condition_132390)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 365):
        
        # Assigning a Tuple to a Attribute (line 365):
        
        # Obtaining an instance of the builtin type 'tuple' (line 365)
        tuple_132391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 365)
        # Adding element type (line 365)
        unicode_132392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 41), 'unicode', u'pre')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 41), tuple_132391, unicode_132392)
        # Adding element type (line 365)
        
        # Call to from_values(...): (line 366)
        # Processing the call arguments (line 366)
        int_132396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 45), 'int')
        int_132397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 48), 'int')
        int_132398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 51), 'int')
        int_132399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 54), 'int')
        int_132400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 57), 'int')
        # Getting the type of 'amount' (line 369)
        amount_132401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 60), 'amount', False)
        # Processing the call keyword arguments (line 366)
        kwargs_132402 = {}
        # Getting the type of 'mtransforms' (line 366)
        mtransforms_132393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 41), 'mtransforms', False)
        # Obtaining the member 'Affine2D' of a type (line 366)
        Affine2D_132394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 41), mtransforms_132393, 'Affine2D')
        # Obtaining the member 'from_values' of a type (line 366)
        from_values_132395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 41), Affine2D_132394, 'from_values')
        # Calling from_values(args, kwargs) (line 366)
        from_values_call_result_132403 = invoke(stypy.reporting.localization.Localization(__file__, 366, 41), from_values_132395, *[int_132396, int_132397, int_132398, int_132399, int_132400, amount_132401], **kwargs_132402)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 41), tuple_132391, from_values_call_result_132403)
        
        # Getting the type of 'self' (line 365)
        self_132404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 365)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 16), self_132404, '_spine_transform', tuple_132391)
        # SSA branch for the else part of an if statement (line 364)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 371)
        # Processing the call arguments (line 371)
        unicode_132407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 30), 'unicode', u'unknown spine type "%s": no spine offset performed')
        # Getting the type of 'self' (line 372)
        self_132408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 51), 'self', False)
        # Obtaining the member 'spine_type' of a type (line 372)
        spine_type_132409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 51), self_132408, 'spine_type')
        # Applying the binary operator '%' (line 371)
        result_mod_132410 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 30), '%', unicode_132407, spine_type_132409)
        
        # Processing the call keyword arguments (line 371)
        kwargs_132411 = {}
        # Getting the type of 'warnings' (line 371)
        warnings_132405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 371)
        warn_132406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 16), warnings_132405, 'warn')
        # Calling warn(args, kwargs) (line 371)
        warn_call_result_132412 = invoke(stypy.reporting.localization.Localization(__file__, 371, 16), warn_132406, *[result_mod_132410], **kwargs_132411)
        
        
        # Assigning a Tuple to a Attribute (line 373):
        
        # Assigning a Tuple to a Attribute (line 373):
        
        # Obtaining an instance of the builtin type 'tuple' (line 373)
        tuple_132413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 373)
        # Adding element type (line 373)
        unicode_132414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 41), 'unicode', u'identity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 41), tuple_132413, unicode_132414)
        # Adding element type (line 373)
        
        # Call to IdentityTransform(...): (line 374)
        # Processing the call keyword arguments (line 374)
        kwargs_132417 = {}
        # Getting the type of 'mtransforms' (line 374)
        mtransforms_132415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 41), 'mtransforms', False)
        # Obtaining the member 'IdentityTransform' of a type (line 374)
        IdentityTransform_132416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 41), mtransforms_132415, 'IdentityTransform')
        # Calling IdentityTransform(args, kwargs) (line 374)
        IdentityTransform_call_result_132418 = invoke(stypy.reporting.localization.Localization(__file__, 374, 41), IdentityTransform_132416, *[], **kwargs_132417)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 41), tuple_132413, IdentityTransform_call_result_132418)
        
        # Getting the type of 'self' (line 373)
        self_132419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 373)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), self_132419, '_spine_transform', tuple_132413)
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 358)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 357)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'position_type' (line 375)
        position_type_132420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 13), 'position_type')
        unicode_132421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 30), 'unicode', u'data')
        # Applying the binary operator '==' (line 375)
        result_eq_132422 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 13), '==', position_type_132420, unicode_132421)
        
        # Testing the type of an if condition (line 375)
        if_condition_132423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 13), result_eq_132422)
        # Assigning a type to the variable 'if_condition_132423' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 13), 'if_condition_132423', if_condition_132423)
        # SSA begins for if statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 376)
        self_132424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'self')
        # Obtaining the member 'spine_type' of a type (line 376)
        spine_type_132425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 15), self_132424, 'spine_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 376)
        tuple_132426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 376)
        # Adding element type (line 376)
        unicode_132427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 35), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 35), tuple_132426, unicode_132427)
        # Adding element type (line 376)
        unicode_132428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 44), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 35), tuple_132426, unicode_132428)
        
        # Applying the binary operator 'in' (line 376)
        result_contains_132429 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 15), 'in', spine_type_132425, tuple_132426)
        
        # Testing the type of an if condition (line 376)
        if_condition_132430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 12), result_contains_132429)
        # Assigning a type to the variable 'if_condition_132430' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'if_condition_132430', if_condition_132430)
        # SSA begins for if statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'amount' (line 380)
        amount_132431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'amount')
        int_132432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 26), 'int')
        # Applying the binary operator '-=' (line 380)
        result_isub_132433 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 16), '-=', amount_132431, int_132432)
        # Assigning a type to the variable 'amount' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'amount', result_isub_132433)
        
        # SSA join for if statement (line 376)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 381)
        self_132434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'self')
        # Obtaining the member 'spine_type' of a type (line 381)
        spine_type_132435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 15), self_132434, 'spine_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_132436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        unicode_132437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 35), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 35), tuple_132436, unicode_132437)
        # Adding element type (line 381)
        unicode_132438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 43), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 35), tuple_132436, unicode_132438)
        
        # Applying the binary operator 'in' (line 381)
        result_contains_132439 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 15), 'in', spine_type_132435, tuple_132436)
        
        # Testing the type of an if condition (line 381)
        if_condition_132440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 12), result_contains_132439)
        # Assigning a type to the variable 'if_condition_132440' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'if_condition_132440', if_condition_132440)
        # SSA begins for if statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 382):
        
        # Assigning a Tuple to a Attribute (line 382):
        
        # Obtaining an instance of the builtin type 'tuple' (line 382)
        tuple_132441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 382)
        # Adding element type (line 382)
        unicode_132442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 41), 'unicode', u'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 41), tuple_132441, unicode_132442)
        # Adding element type (line 382)
        
        # Call to translate(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'amount' (line 384)
        amount_132448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 45), 'amount', False)
        int_132449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 53), 'int')
        # Processing the call keyword arguments (line 383)
        kwargs_132450 = {}
        
        # Call to Affine2D(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_132445 = {}
        # Getting the type of 'mtransforms' (line 383)
        mtransforms_132443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 41), 'mtransforms', False)
        # Obtaining the member 'Affine2D' of a type (line 383)
        Affine2D_132444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 41), mtransforms_132443, 'Affine2D')
        # Calling Affine2D(args, kwargs) (line 383)
        Affine2D_call_result_132446 = invoke(stypy.reporting.localization.Localization(__file__, 383, 41), Affine2D_132444, *[], **kwargs_132445)
        
        # Obtaining the member 'translate' of a type (line 383)
        translate_132447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 41), Affine2D_call_result_132446, 'translate')
        # Calling translate(args, kwargs) (line 383)
        translate_call_result_132451 = invoke(stypy.reporting.localization.Localization(__file__, 383, 41), translate_132447, *[amount_132448, int_132449], **kwargs_132450)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 41), tuple_132441, translate_call_result_132451)
        
        # Getting the type of 'self' (line 382)
        self_132452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 16), self_132452, '_spine_transform', tuple_132441)
        # SSA branch for the else part of an if statement (line 381)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 385)
        self_132453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 17), 'self')
        # Obtaining the member 'spine_type' of a type (line 385)
        spine_type_132454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 17), self_132453, 'spine_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 385)
        tuple_132455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 385)
        # Adding element type (line 385)
        unicode_132456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 37), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 37), tuple_132455, unicode_132456)
        # Adding element type (line 385)
        unicode_132457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 47), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 37), tuple_132455, unicode_132457)
        
        # Applying the binary operator 'in' (line 385)
        result_contains_132458 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 17), 'in', spine_type_132454, tuple_132455)
        
        # Testing the type of an if condition (line 385)
        if_condition_132459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 17), result_contains_132458)
        # Assigning a type to the variable 'if_condition_132459' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 17), 'if_condition_132459', if_condition_132459)
        # SSA begins for if statement (line 385)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 386):
        
        # Assigning a Tuple to a Attribute (line 386):
        
        # Obtaining an instance of the builtin type 'tuple' (line 386)
        tuple_132460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 386)
        # Adding element type (line 386)
        unicode_132461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 41), 'unicode', u'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 41), tuple_132460, unicode_132461)
        # Adding element type (line 386)
        
        # Call to translate(...): (line 387)
        # Processing the call arguments (line 387)
        int_132467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 45), 'int')
        # Getting the type of 'amount' (line 388)
        amount_132468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 48), 'amount', False)
        # Processing the call keyword arguments (line 387)
        kwargs_132469 = {}
        
        # Call to Affine2D(...): (line 387)
        # Processing the call keyword arguments (line 387)
        kwargs_132464 = {}
        # Getting the type of 'mtransforms' (line 387)
        mtransforms_132462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 41), 'mtransforms', False)
        # Obtaining the member 'Affine2D' of a type (line 387)
        Affine2D_132463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 41), mtransforms_132462, 'Affine2D')
        # Calling Affine2D(args, kwargs) (line 387)
        Affine2D_call_result_132465 = invoke(stypy.reporting.localization.Localization(__file__, 387, 41), Affine2D_132463, *[], **kwargs_132464)
        
        # Obtaining the member 'translate' of a type (line 387)
        translate_132466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 41), Affine2D_call_result_132465, 'translate')
        # Calling translate(args, kwargs) (line 387)
        translate_call_result_132470 = invoke(stypy.reporting.localization.Localization(__file__, 387, 41), translate_132466, *[int_132467, amount_132468], **kwargs_132469)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 41), tuple_132460, translate_call_result_132470)
        
        # Getting the type of 'self' (line 386)
        self_132471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 386)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 16), self_132471, '_spine_transform', tuple_132460)
        # SSA branch for the else part of an if statement (line 385)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 390)
        # Processing the call arguments (line 390)
        unicode_132474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 30), 'unicode', u'unknown spine type "%s": no spine offset performed')
        # Getting the type of 'self' (line 391)
        self_132475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 51), 'self', False)
        # Obtaining the member 'spine_type' of a type (line 391)
        spine_type_132476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 51), self_132475, 'spine_type')
        # Applying the binary operator '%' (line 390)
        result_mod_132477 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 30), '%', unicode_132474, spine_type_132476)
        
        # Processing the call keyword arguments (line 390)
        kwargs_132478 = {}
        # Getting the type of 'warnings' (line 390)
        warnings_132472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 390)
        warn_132473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 16), warnings_132472, 'warn')
        # Calling warn(args, kwargs) (line 390)
        warn_call_result_132479 = invoke(stypy.reporting.localization.Localization(__file__, 390, 16), warn_132473, *[result_mod_132477], **kwargs_132478)
        
        
        # Assigning a Tuple to a Attribute (line 392):
        
        # Assigning a Tuple to a Attribute (line 392):
        
        # Obtaining an instance of the builtin type 'tuple' (line 392)
        tuple_132480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 392)
        # Adding element type (line 392)
        unicode_132481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 41), 'unicode', u'identity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 41), tuple_132480, unicode_132481)
        # Adding element type (line 392)
        
        # Call to IdentityTransform(...): (line 393)
        # Processing the call keyword arguments (line 393)
        kwargs_132484 = {}
        # Getting the type of 'mtransforms' (line 393)
        mtransforms_132482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 41), 'mtransforms', False)
        # Obtaining the member 'IdentityTransform' of a type (line 393)
        IdentityTransform_132483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 41), mtransforms_132482, 'IdentityTransform')
        # Calling IdentityTransform(args, kwargs) (line 393)
        IdentityTransform_call_result_132485 = invoke(stypy.reporting.localization.Localization(__file__, 393, 41), IdentityTransform_132483, *[], **kwargs_132484)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 41), tuple_132480, IdentityTransform_call_result_132485)
        
        # Getting the type of 'self' (line 392)
        self_132486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'self')
        # Setting the type of the member '_spine_transform' of a type (line 392)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 16), self_132486, '_spine_transform', tuple_132480)
        # SSA join for if statement (line 385)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 381)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 375)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_calc_offset_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_calc_offset_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 321)
        stypy_return_type_132487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132487)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_calc_offset_transform'
        return stypy_return_type_132487


    @norecursion
    def set_position(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_position'
        module_type_store = module_type_store.open_function_context('set_position', 395, 4, False)
        # Assigning a type to the variable 'self' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.set_position.__dict__.__setitem__('stypy_localization', localization)
        Spine.set_position.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.set_position.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.set_position.__dict__.__setitem__('stypy_function_name', 'Spine.set_position')
        Spine.set_position.__dict__.__setitem__('stypy_param_names_list', ['position'])
        Spine.set_position.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.set_position.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.set_position.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.set_position.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.set_position.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.set_position.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.set_position', ['position'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_position', localization, ['position'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_position(...)' code ##################

        unicode_132488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, (-1)), 'unicode', u"set the position of the spine\n\n        Spine position is specified by a 2 tuple of (position type,\n        amount). The position types are:\n\n        * 'outward' : place the spine out from the data area by the\n          specified number of points. (Negative values specify placing the\n          spine inward.)\n\n        * 'axes' : place the spine at the specified Axes coordinate (from\n          0.0-1.0).\n\n        * 'data' : place the spine at the specified data coordinate.\n\n        Additionally, shorthand notations define a special positions:\n\n        * 'center' -> ('axes',0.5)\n        * 'zero' -> ('data', 0.0)\n\n        ")
        
        
        # Getting the type of 'position' (line 416)
        position_132489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 11), 'position')
        
        # Obtaining an instance of the builtin type 'tuple' (line 416)
        tuple_132490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 416)
        # Adding element type (line 416)
        unicode_132491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 24), 'unicode', u'center')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 24), tuple_132490, unicode_132491)
        # Adding element type (line 416)
        unicode_132492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 34), 'unicode', u'zero')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 24), tuple_132490, unicode_132492)
        
        # Applying the binary operator 'in' (line 416)
        result_contains_132493 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), 'in', position_132489, tuple_132490)
        
        # Testing the type of an if condition (line 416)
        if_condition_132494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 8), result_contains_132493)
        # Assigning a type to the variable 'if_condition_132494' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'if_condition_132494', if_condition_132494)
        # SSA begins for if statement (line 416)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 416)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'position' (line 420)
        position_132496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 19), 'position', False)
        # Processing the call keyword arguments (line 420)
        kwargs_132497 = {}
        # Getting the type of 'len' (line 420)
        len_132495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'len', False)
        # Calling len(args, kwargs) (line 420)
        len_call_result_132498 = invoke(stypy.reporting.localization.Localization(__file__, 420, 15), len_132495, *[position_132496], **kwargs_132497)
        
        int_132499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 32), 'int')
        # Applying the binary operator '!=' (line 420)
        result_ne_132500 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 15), '!=', len_call_result_132498, int_132499)
        
        # Testing the type of an if condition (line 420)
        if_condition_132501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 12), result_ne_132500)
        # Assigning a type to the variable 'if_condition_132501' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'if_condition_132501', if_condition_132501)
        # SSA begins for if statement (line 420)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 421)
        # Processing the call arguments (line 421)
        unicode_132503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 33), 'unicode', u"position should be 'center' or 2-tuple")
        # Processing the call keyword arguments (line 421)
        kwargs_132504 = {}
        # Getting the type of 'ValueError' (line 421)
        ValueError_132502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 421)
        ValueError_call_result_132505 = invoke(stypy.reporting.localization.Localization(__file__, 421, 22), ValueError_132502, *[unicode_132503], **kwargs_132504)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 421, 16), ValueError_call_result_132505, 'raise parameter', BaseException)
        # SSA join for if statement (line 420)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_132506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 24), 'int')
        # Getting the type of 'position' (line 422)
        position_132507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'position')
        # Obtaining the member '__getitem__' of a type (line 422)
        getitem___132508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), position_132507, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 422)
        subscript_call_result_132509 = invoke(stypy.reporting.localization.Localization(__file__, 422, 15), getitem___132508, int_132506)
        
        
        # Obtaining an instance of the builtin type 'list' (line 422)
        list_132510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 422)
        # Adding element type (line 422)
        unicode_132511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 35), 'unicode', u'outward')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 34), list_132510, unicode_132511)
        # Adding element type (line 422)
        unicode_132512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 46), 'unicode', u'axes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 34), list_132510, unicode_132512)
        # Adding element type (line 422)
        unicode_132513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 54), 'unicode', u'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 34), list_132510, unicode_132513)
        
        # Applying the binary operator 'notin' (line 422)
        result_contains_132514 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 15), 'notin', subscript_call_result_132509, list_132510)
        
        # Testing the type of an if condition (line 422)
        if_condition_132515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 12), result_contains_132514)
        # Assigning a type to the variable 'if_condition_132515' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'if_condition_132515', if_condition_132515)
        # SSA begins for if statement (line 422)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 423):
        
        # Assigning a Str to a Name (line 423):
        unicode_132516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 23), 'unicode', u"position[0] should be in [ 'outward' | 'axes' | 'data' ]")
        # Assigning a type to the variable 'msg' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'msg', unicode_132516)
        
        # Call to ValueError(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'msg' (line 425)
        msg_132518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 33), 'msg', False)
        # Processing the call keyword arguments (line 425)
        kwargs_132519 = {}
        # Getting the type of 'ValueError' (line 425)
        ValueError_132517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 425)
        ValueError_call_result_132520 = invoke(stypy.reporting.localization.Localization(__file__, 425, 22), ValueError_132517, *[msg_132518], **kwargs_132519)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 425, 16), ValueError_call_result_132520, 'raise parameter', BaseException)
        # SSA join for if statement (line 422)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 416)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 426):
        
        # Assigning a Name to a Attribute (line 426):
        # Getting the type of 'position' (line 426)
        position_132521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 25), 'position')
        # Getting the type of 'self' (line 426)
        self_132522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'self')
        # Setting the type of the member '_position' of a type (line 426)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), self_132522, '_position', position_132521)
        
        # Call to _calc_offset_transform(...): (line 427)
        # Processing the call keyword arguments (line 427)
        kwargs_132525 = {}
        # Getting the type of 'self' (line 427)
        self_132523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self', False)
        # Obtaining the member '_calc_offset_transform' of a type (line 427)
        _calc_offset_transform_132524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), self_132523, '_calc_offset_transform')
        # Calling _calc_offset_transform(args, kwargs) (line 427)
        _calc_offset_transform_call_result_132526 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), _calc_offset_transform_132524, *[], **kwargs_132525)
        
        
        # Call to set_transform(...): (line 429)
        # Processing the call arguments (line 429)
        
        # Call to get_spine_transform(...): (line 429)
        # Processing the call keyword arguments (line 429)
        kwargs_132531 = {}
        # Getting the type of 'self' (line 429)
        self_132529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 27), 'self', False)
        # Obtaining the member 'get_spine_transform' of a type (line 429)
        get_spine_transform_132530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 27), self_132529, 'get_spine_transform')
        # Calling get_spine_transform(args, kwargs) (line 429)
        get_spine_transform_call_result_132532 = invoke(stypy.reporting.localization.Localization(__file__, 429, 27), get_spine_transform_132530, *[], **kwargs_132531)
        
        # Processing the call keyword arguments (line 429)
        kwargs_132533 = {}
        # Getting the type of 'self' (line 429)
        self_132527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 429)
        set_transform_132528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_132527, 'set_transform')
        # Calling set_transform(args, kwargs) (line 429)
        set_transform_call_result_132534 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), set_transform_132528, *[get_spine_transform_call_result_132532], **kwargs_132533)
        
        
        
        # Getting the type of 'self' (line 431)
        self_132535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 11), 'self')
        # Obtaining the member 'axis' of a type (line 431)
        axis_132536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 11), self_132535, 'axis')
        # Getting the type of 'None' (line 431)
        None_132537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 28), 'None')
        # Applying the binary operator 'isnot' (line 431)
        result_is_not_132538 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 11), 'isnot', axis_132536, None_132537)
        
        # Testing the type of an if condition (line 431)
        if_condition_132539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 8), result_is_not_132538)
        # Assigning a type to the variable 'if_condition_132539' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'if_condition_132539', if_condition_132539)
        # SSA begins for if statement (line 431)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to reset_ticks(...): (line 432)
        # Processing the call keyword arguments (line 432)
        kwargs_132543 = {}
        # Getting the type of 'self' (line 432)
        self_132540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'self', False)
        # Obtaining the member 'axis' of a type (line 432)
        axis_132541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), self_132540, 'axis')
        # Obtaining the member 'reset_ticks' of a type (line 432)
        reset_ticks_132542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), axis_132541, 'reset_ticks')
        # Calling reset_ticks(args, kwargs) (line 432)
        reset_ticks_call_result_132544 = invoke(stypy.reporting.localization.Localization(__file__, 432, 12), reset_ticks_132542, *[], **kwargs_132543)
        
        # SSA join for if statement (line 431)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 433):
        
        # Assigning a Name to a Attribute (line 433):
        # Getting the type of 'True' (line 433)
        True_132545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 21), 'True')
        # Getting the type of 'self' (line 433)
        self_132546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 433)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_132546, 'stale', True_132545)
        
        # ################# End of 'set_position(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_position' in the type store
        # Getting the type of 'stypy_return_type' (line 395)
        stypy_return_type_132547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132547)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_position'
        return stypy_return_type_132547


    @norecursion
    def get_position(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_position'
        module_type_store = module_type_store.open_function_context('get_position', 435, 4, False)
        # Assigning a type to the variable 'self' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.get_position.__dict__.__setitem__('stypy_localization', localization)
        Spine.get_position.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.get_position.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.get_position.__dict__.__setitem__('stypy_function_name', 'Spine.get_position')
        Spine.get_position.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.get_position.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.get_position.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.get_position.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.get_position.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.get_position.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.get_position.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.get_position', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_position', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_position(...)' code ##################

        unicode_132548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 8), 'unicode', u'get the spine position')
        
        # Call to _ensure_position_is_set(...): (line 437)
        # Processing the call keyword arguments (line 437)
        kwargs_132551 = {}
        # Getting the type of 'self' (line 437)
        self_132549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'self', False)
        # Obtaining the member '_ensure_position_is_set' of a type (line 437)
        _ensure_position_is_set_132550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), self_132549, '_ensure_position_is_set')
        # Calling _ensure_position_is_set(args, kwargs) (line 437)
        _ensure_position_is_set_call_result_132552 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), _ensure_position_is_set_132550, *[], **kwargs_132551)
        
        # Getting the type of 'self' (line 438)
        self_132553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'self')
        # Obtaining the member '_position' of a type (line 438)
        _position_132554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 15), self_132553, '_position')
        # Assigning a type to the variable 'stypy_return_type' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'stypy_return_type', _position_132554)
        
        # ################# End of 'get_position(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_position' in the type store
        # Getting the type of 'stypy_return_type' (line 435)
        stypy_return_type_132555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132555)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_position'
        return stypy_return_type_132555


    @norecursion
    def get_spine_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_spine_transform'
        module_type_store = module_type_store.open_function_context('get_spine_transform', 440, 4, False)
        # Assigning a type to the variable 'self' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.get_spine_transform.__dict__.__setitem__('stypy_localization', localization)
        Spine.get_spine_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.get_spine_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.get_spine_transform.__dict__.__setitem__('stypy_function_name', 'Spine.get_spine_transform')
        Spine.get_spine_transform.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.get_spine_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.get_spine_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.get_spine_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.get_spine_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.get_spine_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.get_spine_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.get_spine_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_spine_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_spine_transform(...)' code ##################

        unicode_132556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 8), 'unicode', u'get the spine transform')
        
        # Call to _ensure_position_is_set(...): (line 442)
        # Processing the call keyword arguments (line 442)
        kwargs_132559 = {}
        # Getting the type of 'self' (line 442)
        self_132557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'self', False)
        # Obtaining the member '_ensure_position_is_set' of a type (line 442)
        _ensure_position_is_set_132558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), self_132557, '_ensure_position_is_set')
        # Calling _ensure_position_is_set(args, kwargs) (line 442)
        _ensure_position_is_set_call_result_132560 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), _ensure_position_is_set_132558, *[], **kwargs_132559)
        
        
        # Assigning a Attribute to a Tuple (line 443):
        
        # Assigning a Subscript to a Name (line 443):
        
        # Obtaining the type of the subscript
        int_132561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
        # Getting the type of 'self' (line 443)
        self_132562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'self')
        # Obtaining the member '_spine_transform' of a type (line 443)
        _spine_transform_132563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 20), self_132562, '_spine_transform')
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___132564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), _spine_transform_132563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_132565 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), getitem___132564, int_132561)
        
        # Assigning a type to the variable 'tuple_var_assignment_131401' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_131401', subscript_call_result_132565)
        
        # Assigning a Subscript to a Name (line 443):
        
        # Obtaining the type of the subscript
        int_132566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
        # Getting the type of 'self' (line 443)
        self_132567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'self')
        # Obtaining the member '_spine_transform' of a type (line 443)
        _spine_transform_132568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 20), self_132567, '_spine_transform')
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___132569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), _spine_transform_132568, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_132570 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), getitem___132569, int_132566)
        
        # Assigning a type to the variable 'tuple_var_assignment_131402' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_131402', subscript_call_result_132570)
        
        # Assigning a Name to a Name (line 443):
        # Getting the type of 'tuple_var_assignment_131401' (line 443)
        tuple_var_assignment_131401_132571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_131401')
        # Assigning a type to the variable 'what' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'what', tuple_var_assignment_131401_132571)
        
        # Assigning a Name to a Name (line 443):
        # Getting the type of 'tuple_var_assignment_131402' (line 443)
        tuple_var_assignment_131402_132572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_131402')
        # Assigning a type to the variable 'how' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 14), 'how', tuple_var_assignment_131402_132572)
        
        
        # Getting the type of 'what' (line 445)
        what_132573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 'what')
        unicode_132574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 19), 'unicode', u'data')
        # Applying the binary operator '==' (line 445)
        result_eq_132575 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 11), '==', what_132573, unicode_132574)
        
        # Testing the type of an if condition (line 445)
        if_condition_132576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 8), result_eq_132575)
        # Assigning a type to the variable 'if_condition_132576' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'if_condition_132576', if_condition_132576)
        # SSA begins for if statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 447):
        
        # Assigning a BinOp to a Name (line 447):
        # Getting the type of 'self' (line 447)
        self_132577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 25), 'self')
        # Obtaining the member 'axes' of a type (line 447)
        axes_132578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 25), self_132577, 'axes')
        # Obtaining the member 'transScale' of a type (line 447)
        transScale_132579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 25), axes_132578, 'transScale')
        # Getting the type of 'how' (line 448)
        how_132580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 17), 'how')
        # Getting the type of 'self' (line 448)
        self_132581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 23), 'self')
        # Obtaining the member 'axes' of a type (line 448)
        axes_132582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 23), self_132581, 'axes')
        # Obtaining the member 'transLimits' of a type (line 448)
        transLimits_132583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 23), axes_132582, 'transLimits')
        # Applying the binary operator '+' (line 448)
        result_add_132584 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 17), '+', how_132580, transLimits_132583)
        
        # Getting the type of 'self' (line 448)
        self_132585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 47), 'self')
        # Obtaining the member 'axes' of a type (line 448)
        axes_132586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 47), self_132585, 'axes')
        # Obtaining the member 'transAxes' of a type (line 448)
        transAxes_132587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 47), axes_132586, 'transAxes')
        # Applying the binary operator '+' (line 448)
        result_add_132588 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 45), '+', result_add_132584, transAxes_132587)
        
        # Applying the binary operator '+' (line 447)
        result_add_132589 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 25), '+', transScale_132579, result_add_132588)
        
        # Assigning a type to the variable 'data_xform' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'data_xform', result_add_132589)
        
        
        # Getting the type of 'self' (line 449)
        self_132590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'self')
        # Obtaining the member 'spine_type' of a type (line 449)
        spine_type_132591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 15), self_132590, 'spine_type')
        
        # Obtaining an instance of the builtin type 'list' (line 449)
        list_132592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 449)
        # Adding element type (line 449)
        unicode_132593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 35), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 34), list_132592, unicode_132593)
        # Adding element type (line 449)
        unicode_132594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 43), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 34), list_132592, unicode_132594)
        
        # Applying the binary operator 'in' (line 449)
        result_contains_132595 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 15), 'in', spine_type_132591, list_132592)
        
        # Testing the type of an if condition (line 449)
        if_condition_132596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 12), result_contains_132595)
        # Assigning a type to the variable 'if_condition_132596' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'if_condition_132596', if_condition_132596)
        # SSA begins for if statement (line 449)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 450):
        
        # Assigning a Call to a Name (line 450):
        
        # Call to blended_transform_factory(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'data_xform' (line 451)
        data_xform_132599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'data_xform', False)
        # Getting the type of 'self' (line 451)
        self_132600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 32), 'self', False)
        # Obtaining the member 'axes' of a type (line 451)
        axes_132601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 32), self_132600, 'axes')
        # Obtaining the member 'transData' of a type (line 451)
        transData_132602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 32), axes_132601, 'transData')
        # Processing the call keyword arguments (line 450)
        kwargs_132603 = {}
        # Getting the type of 'mtransforms' (line 450)
        mtransforms_132597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 25), 'mtransforms', False)
        # Obtaining the member 'blended_transform_factory' of a type (line 450)
        blended_transform_factory_132598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 25), mtransforms_132597, 'blended_transform_factory')
        # Calling blended_transform_factory(args, kwargs) (line 450)
        blended_transform_factory_call_result_132604 = invoke(stypy.reporting.localization.Localization(__file__, 450, 25), blended_transform_factory_132598, *[data_xform_132599, transData_132602], **kwargs_132603)
        
        # Assigning a type to the variable 'result' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'result', blended_transform_factory_call_result_132604)
        # SSA branch for the else part of an if statement (line 449)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 452)
        self_132605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 17), 'self')
        # Obtaining the member 'spine_type' of a type (line 452)
        spine_type_132606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 17), self_132605, 'spine_type')
        
        # Obtaining an instance of the builtin type 'list' (line 452)
        list_132607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 452)
        # Adding element type (line 452)
        unicode_132608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 37), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 36), list_132607, unicode_132608)
        # Adding element type (line 452)
        unicode_132609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 44), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 36), list_132607, unicode_132609)
        
        # Applying the binary operator 'in' (line 452)
        result_contains_132610 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 17), 'in', spine_type_132606, list_132607)
        
        # Testing the type of an if condition (line 452)
        if_condition_132611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 452, 17), result_contains_132610)
        # Assigning a type to the variable 'if_condition_132611' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 17), 'if_condition_132611', if_condition_132611)
        # SSA begins for if statement (line 452)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 453):
        
        # Assigning a Call to a Name (line 453):
        
        # Call to blended_transform_factory(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'self' (line 454)
        self_132614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 20), 'self', False)
        # Obtaining the member 'axes' of a type (line 454)
        axes_132615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 20), self_132614, 'axes')
        # Obtaining the member 'transData' of a type (line 454)
        transData_132616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 20), axes_132615, 'transData')
        # Getting the type of 'data_xform' (line 454)
        data_xform_132617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 41), 'data_xform', False)
        # Processing the call keyword arguments (line 453)
        kwargs_132618 = {}
        # Getting the type of 'mtransforms' (line 453)
        mtransforms_132612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 25), 'mtransforms', False)
        # Obtaining the member 'blended_transform_factory' of a type (line 453)
        blended_transform_factory_132613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 25), mtransforms_132612, 'blended_transform_factory')
        # Calling blended_transform_factory(args, kwargs) (line 453)
        blended_transform_factory_call_result_132619 = invoke(stypy.reporting.localization.Localization(__file__, 453, 25), blended_transform_factory_132613, *[transData_132616, data_xform_132617], **kwargs_132618)
        
        # Assigning a type to the variable 'result' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'result', blended_transform_factory_call_result_132619)
        # SSA branch for the else part of an if statement (line 452)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 456)
        # Processing the call arguments (line 456)
        unicode_132621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 33), 'unicode', u'unknown spine spine_type: %s')
        # Getting the type of 'self' (line 457)
        self_132622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 33), 'self', False)
        # Obtaining the member 'spine_type' of a type (line 457)
        spine_type_132623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 33), self_132622, 'spine_type')
        # Applying the binary operator '%' (line 456)
        result_mod_132624 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 33), '%', unicode_132621, spine_type_132623)
        
        # Processing the call keyword arguments (line 456)
        kwargs_132625 = {}
        # Getting the type of 'ValueError' (line 456)
        ValueError_132620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 456)
        ValueError_call_result_132626 = invoke(stypy.reporting.localization.Localization(__file__, 456, 22), ValueError_132620, *[result_mod_132624], **kwargs_132625)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 456, 16), ValueError_call_result_132626, 'raise parameter', BaseException)
        # SSA join for if statement (line 452)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 449)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 458)
        result_132627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'stypy_return_type', result_132627)
        # SSA join for if statement (line 445)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 460)
        self_132628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'self')
        # Obtaining the member 'spine_type' of a type (line 460)
        spine_type_132629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 11), self_132628, 'spine_type')
        
        # Obtaining an instance of the builtin type 'list' (line 460)
        list_132630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 460)
        # Adding element type (line 460)
        unicode_132631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 31), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 30), list_132630, unicode_132631)
        # Adding element type (line 460)
        unicode_132632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 39), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 30), list_132630, unicode_132632)
        
        # Applying the binary operator 'in' (line 460)
        result_contains_132633 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 11), 'in', spine_type_132629, list_132630)
        
        # Testing the type of an if condition (line 460)
        if_condition_132634 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 8), result_contains_132633)
        # Assigning a type to the variable 'if_condition_132634' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'if_condition_132634', if_condition_132634)
        # SSA begins for if statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 461):
        
        # Assigning a Call to a Name (line 461):
        
        # Call to get_yaxis_transform(...): (line 461)
        # Processing the call keyword arguments (line 461)
        unicode_132638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 65), 'unicode', u'grid')
        keyword_132639 = unicode_132638
        kwargs_132640 = {'which': keyword_132639}
        # Getting the type of 'self' (line 461)
        self_132635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 29), 'self', False)
        # Obtaining the member 'axes' of a type (line 461)
        axes_132636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 29), self_132635, 'axes')
        # Obtaining the member 'get_yaxis_transform' of a type (line 461)
        get_yaxis_transform_132637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 29), axes_132636, 'get_yaxis_transform')
        # Calling get_yaxis_transform(args, kwargs) (line 461)
        get_yaxis_transform_call_result_132641 = invoke(stypy.reporting.localization.Localization(__file__, 461, 29), get_yaxis_transform_132637, *[], **kwargs_132640)
        
        # Assigning a type to the variable 'base_transform' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'base_transform', get_yaxis_transform_call_result_132641)
        # SSA branch for the else part of an if statement (line 460)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 462)
        self_132642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 13), 'self')
        # Obtaining the member 'spine_type' of a type (line 462)
        spine_type_132643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 13), self_132642, 'spine_type')
        
        # Obtaining an instance of the builtin type 'list' (line 462)
        list_132644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 462)
        # Adding element type (line 462)
        unicode_132645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 33), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 32), list_132644, unicode_132645)
        # Adding element type (line 462)
        unicode_132646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 40), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 32), list_132644, unicode_132646)
        
        # Applying the binary operator 'in' (line 462)
        result_contains_132647 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 13), 'in', spine_type_132643, list_132644)
        
        # Testing the type of an if condition (line 462)
        if_condition_132648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 13), result_contains_132647)
        # Assigning a type to the variable 'if_condition_132648' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 13), 'if_condition_132648', if_condition_132648)
        # SSA begins for if statement (line 462)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to get_xaxis_transform(...): (line 463)
        # Processing the call keyword arguments (line 463)
        unicode_132652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 65), 'unicode', u'grid')
        keyword_132653 = unicode_132652
        kwargs_132654 = {'which': keyword_132653}
        # Getting the type of 'self' (line 463)
        self_132649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 29), 'self', False)
        # Obtaining the member 'axes' of a type (line 463)
        axes_132650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 29), self_132649, 'axes')
        # Obtaining the member 'get_xaxis_transform' of a type (line 463)
        get_xaxis_transform_132651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 29), axes_132650, 'get_xaxis_transform')
        # Calling get_xaxis_transform(args, kwargs) (line 463)
        get_xaxis_transform_call_result_132655 = invoke(stypy.reporting.localization.Localization(__file__, 463, 29), get_xaxis_transform_132651, *[], **kwargs_132654)
        
        # Assigning a type to the variable 'base_transform' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'base_transform', get_xaxis_transform_call_result_132655)
        # SSA branch for the else part of an if statement (line 462)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 465)
        # Processing the call arguments (line 465)
        unicode_132657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 29), 'unicode', u'unknown spine spine_type: %s')
        # Getting the type of 'self' (line 466)
        self_132658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 29), 'self', False)
        # Obtaining the member 'spine_type' of a type (line 466)
        spine_type_132659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 29), self_132658, 'spine_type')
        # Applying the binary operator '%' (line 465)
        result_mod_132660 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 29), '%', unicode_132657, spine_type_132659)
        
        # Processing the call keyword arguments (line 465)
        kwargs_132661 = {}
        # Getting the type of 'ValueError' (line 465)
        ValueError_132656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 465)
        ValueError_call_result_132662 = invoke(stypy.reporting.localization.Localization(__file__, 465, 18), ValueError_132656, *[result_mod_132660], **kwargs_132661)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 465, 12), ValueError_call_result_132662, 'raise parameter', BaseException)
        # SSA join for if statement (line 462)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 460)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'what' (line 468)
        what_132663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'what')
        unicode_132664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 19), 'unicode', u'identity')
        # Applying the binary operator '==' (line 468)
        result_eq_132665 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 11), '==', what_132663, unicode_132664)
        
        # Testing the type of an if condition (line 468)
        if_condition_132666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 8), result_eq_132665)
        # Assigning a type to the variable 'if_condition_132666' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'if_condition_132666', if_condition_132666)
        # SSA begins for if statement (line 468)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'base_transform' (line 469)
        base_transform_132667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'base_transform')
        # Assigning a type to the variable 'stypy_return_type' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'stypy_return_type', base_transform_132667)
        # SSA branch for the else part of an if statement (line 468)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'what' (line 470)
        what_132668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 13), 'what')
        unicode_132669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 21), 'unicode', u'post')
        # Applying the binary operator '==' (line 470)
        result_eq_132670 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 13), '==', what_132668, unicode_132669)
        
        # Testing the type of an if condition (line 470)
        if_condition_132671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 13), result_eq_132670)
        # Assigning a type to the variable 'if_condition_132671' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 13), 'if_condition_132671', if_condition_132671)
        # SSA begins for if statement (line 470)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'base_transform' (line 471)
        base_transform_132672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 19), 'base_transform')
        # Getting the type of 'how' (line 471)
        how_132673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 36), 'how')
        # Applying the binary operator '+' (line 471)
        result_add_132674 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 19), '+', base_transform_132672, how_132673)
        
        # Assigning a type to the variable 'stypy_return_type' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'stypy_return_type', result_add_132674)
        # SSA branch for the else part of an if statement (line 470)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'what' (line 472)
        what_132675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 13), 'what')
        unicode_132676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 21), 'unicode', u'pre')
        # Applying the binary operator '==' (line 472)
        result_eq_132677 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 13), '==', what_132675, unicode_132676)
        
        # Testing the type of an if condition (line 472)
        if_condition_132678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 13), result_eq_132677)
        # Assigning a type to the variable 'if_condition_132678' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 13), 'if_condition_132678', if_condition_132678)
        # SSA begins for if statement (line 472)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'how' (line 473)
        how_132679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'how')
        # Getting the type of 'base_transform' (line 473)
        base_transform_132680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 25), 'base_transform')
        # Applying the binary operator '+' (line 473)
        result_add_132681 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 19), '+', how_132679, base_transform_132680)
        
        # Assigning a type to the variable 'stypy_return_type' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'stypy_return_type', result_add_132681)
        # SSA branch for the else part of an if statement (line 472)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 475)
        # Processing the call arguments (line 475)
        unicode_132683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 29), 'unicode', u'unknown spine_transform type: %s')
        # Getting the type of 'what' (line 475)
        what_132684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 66), 'what', False)
        # Applying the binary operator '%' (line 475)
        result_mod_132685 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 29), '%', unicode_132683, what_132684)
        
        # Processing the call keyword arguments (line 475)
        kwargs_132686 = {}
        # Getting the type of 'ValueError' (line 475)
        ValueError_132682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 475)
        ValueError_call_result_132687 = invoke(stypy.reporting.localization.Localization(__file__, 475, 18), ValueError_132682, *[result_mod_132685], **kwargs_132686)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 475, 12), ValueError_call_result_132687, 'raise parameter', BaseException)
        # SSA join for if statement (line 472)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 470)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 468)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_spine_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_spine_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 440)
        stypy_return_type_132688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132688)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_spine_transform'
        return stypy_return_type_132688


    @norecursion
    def set_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_bounds'
        module_type_store = module_type_store.open_function_context('set_bounds', 477, 4, False)
        # Assigning a type to the variable 'self' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.set_bounds.__dict__.__setitem__('stypy_localization', localization)
        Spine.set_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.set_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.set_bounds.__dict__.__setitem__('stypy_function_name', 'Spine.set_bounds')
        Spine.set_bounds.__dict__.__setitem__('stypy_param_names_list', ['low', 'high'])
        Spine.set_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.set_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.set_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.set_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.set_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.set_bounds.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.set_bounds', ['low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_bounds', localization, ['low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_bounds(...)' code ##################

        unicode_132689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 8), 'unicode', u'Set the bounds of the spine.')
        
        
        # Getting the type of 'self' (line 479)
        self_132690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 11), 'self')
        # Obtaining the member 'spine_type' of a type (line 479)
        spine_type_132691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 11), self_132690, 'spine_type')
        unicode_132692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 30), 'unicode', u'circle')
        # Applying the binary operator '==' (line 479)
        result_eq_132693 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 11), '==', spine_type_132691, unicode_132692)
        
        # Testing the type of an if condition (line 479)
        if_condition_132694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 8), result_eq_132693)
        # Assigning a type to the variable 'if_condition_132694' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'if_condition_132694', if_condition_132694)
        # SSA begins for if statement (line 479)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 480)
        # Processing the call arguments (line 480)
        unicode_132696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 16), 'unicode', u'set_bounds() method incompatible with circular spines')
        # Processing the call keyword arguments (line 480)
        kwargs_132697 = {}
        # Getting the type of 'ValueError' (line 480)
        ValueError_132695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 480)
        ValueError_call_result_132698 = invoke(stypy.reporting.localization.Localization(__file__, 480, 18), ValueError_132695, *[unicode_132696], **kwargs_132697)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 480, 12), ValueError_call_result_132698, 'raise parameter', BaseException)
        # SSA join for if statement (line 479)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Attribute (line 482):
        
        # Assigning a Tuple to a Attribute (line 482):
        
        # Obtaining an instance of the builtin type 'tuple' (line 482)
        tuple_132699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 482)
        # Adding element type (line 482)
        # Getting the type of 'low' (line 482)
        low_132700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 24), 'low')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 24), tuple_132699, low_132700)
        # Adding element type (line 482)
        # Getting the type of 'high' (line 482)
        high_132701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 29), 'high')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 24), tuple_132699, high_132701)
        
        # Getting the type of 'self' (line 482)
        self_132702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'self')
        # Setting the type of the member '_bounds' of a type (line 482)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 8), self_132702, '_bounds', tuple_132699)
        
        # Assigning a Name to a Attribute (line 483):
        
        # Assigning a Name to a Attribute (line 483):
        # Getting the type of 'True' (line 483)
        True_132703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 21), 'True')
        # Getting the type of 'self' (line 483)
        self_132704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 483)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), self_132704, 'stale', True_132703)
        
        # ################# End of 'set_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 477)
        stypy_return_type_132705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132705)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_bounds'
        return stypy_return_type_132705


    @norecursion
    def get_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_bounds'
        module_type_store = module_type_store.open_function_context('get_bounds', 485, 4, False)
        # Assigning a type to the variable 'self' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.get_bounds.__dict__.__setitem__('stypy_localization', localization)
        Spine.get_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.get_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.get_bounds.__dict__.__setitem__('stypy_function_name', 'Spine.get_bounds')
        Spine.get_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        Spine.get_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.get_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.get_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.get_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.get_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.get_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.get_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_bounds(...)' code ##################

        unicode_132706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 8), 'unicode', u'Get the bounds of the spine.')
        # Getting the type of 'self' (line 487)
        self_132707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 15), 'self')
        # Obtaining the member '_bounds' of a type (line 487)
        _bounds_132708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 15), self_132707, '_bounds')
        # Assigning a type to the variable 'stypy_return_type' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'stypy_return_type', _bounds_132708)
        
        # ################# End of 'get_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 485)
        stypy_return_type_132709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132709)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_bounds'
        return stypy_return_type_132709


    @norecursion
    def linear_spine(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'linear_spine'
        module_type_store = module_type_store.open_function_context('linear_spine', 489, 4, False)
        # Assigning a type to the variable 'self' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.linear_spine.__dict__.__setitem__('stypy_localization', localization)
        Spine.linear_spine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.linear_spine.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.linear_spine.__dict__.__setitem__('stypy_function_name', 'Spine.linear_spine')
        Spine.linear_spine.__dict__.__setitem__('stypy_param_names_list', ['axes', 'spine_type'])
        Spine.linear_spine.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.linear_spine.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Spine.linear_spine.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.linear_spine.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.linear_spine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.linear_spine.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.linear_spine', ['axes', 'spine_type'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'linear_spine', localization, ['axes', 'spine_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'linear_spine(...)' code ##################

        unicode_132710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, (-1)), 'unicode', u'\n        (staticmethod) Returns a linear :class:`Spine`.\n        ')
        
        
        # Getting the type of 'spine_type' (line 495)
        spine_type_132711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 11), 'spine_type')
        unicode_132712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 25), 'unicode', u'left')
        # Applying the binary operator '==' (line 495)
        result_eq_132713 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 11), '==', spine_type_132711, unicode_132712)
        
        # Testing the type of an if condition (line 495)
        if_condition_132714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 495, 8), result_eq_132713)
        # Assigning a type to the variable 'if_condition_132714' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'if_condition_132714', if_condition_132714)
        # SSA begins for if statement (line 495)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 496):
        
        # Assigning a Call to a Name (line 496):
        
        # Call to Path(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Obtaining an instance of the builtin type 'list' (line 496)
        list_132717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 496)
        # Adding element type (line 496)
        
        # Obtaining an instance of the builtin type 'tuple' (line 496)
        tuple_132718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 496)
        # Adding element type (line 496)
        float_132719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 32), tuple_132718, float_132719)
        # Adding element type (line 496)
        int_132720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 32), tuple_132718, int_132720)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 30), list_132717, tuple_132718)
        # Adding element type (line 496)
        
        # Obtaining an instance of the builtin type 'tuple' (line 496)
        tuple_132721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 496)
        # Adding element type (line 496)
        float_132722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 43), tuple_132721, float_132722)
        # Adding element type (line 496)
        int_132723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 43), tuple_132721, int_132723)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 30), list_132717, tuple_132721)
        
        # Processing the call keyword arguments (line 496)
        kwargs_132724 = {}
        # Getting the type of 'mpath' (line 496)
        mpath_132715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 19), 'mpath', False)
        # Obtaining the member 'Path' of a type (line 496)
        Path_132716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 19), mpath_132715, 'Path')
        # Calling Path(args, kwargs) (line 496)
        Path_call_result_132725 = invoke(stypy.reporting.localization.Localization(__file__, 496, 19), Path_132716, *[list_132717], **kwargs_132724)
        
        # Assigning a type to the variable 'path' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'path', Path_call_result_132725)
        # SSA branch for the else part of an if statement (line 495)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'spine_type' (line 497)
        spine_type_132726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 13), 'spine_type')
        unicode_132727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 27), 'unicode', u'right')
        # Applying the binary operator '==' (line 497)
        result_eq_132728 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 13), '==', spine_type_132726, unicode_132727)
        
        # Testing the type of an if condition (line 497)
        if_condition_132729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 13), result_eq_132728)
        # Assigning a type to the variable 'if_condition_132729' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 13), 'if_condition_132729', if_condition_132729)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 498):
        
        # Assigning a Call to a Name (line 498):
        
        # Call to Path(...): (line 498)
        # Processing the call arguments (line 498)
        
        # Obtaining an instance of the builtin type 'list' (line 498)
        list_132732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 498)
        # Adding element type (line 498)
        
        # Obtaining an instance of the builtin type 'tuple' (line 498)
        tuple_132733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 498)
        # Adding element type (line 498)
        float_132734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 32), tuple_132733, float_132734)
        # Adding element type (line 498)
        int_132735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 32), tuple_132733, int_132735)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 30), list_132732, tuple_132733)
        # Adding element type (line 498)
        
        # Obtaining an instance of the builtin type 'tuple' (line 498)
        tuple_132736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 498)
        # Adding element type (line 498)
        float_132737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 43), tuple_132736, float_132737)
        # Adding element type (line 498)
        int_132738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 43), tuple_132736, int_132738)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 30), list_132732, tuple_132736)
        
        # Processing the call keyword arguments (line 498)
        kwargs_132739 = {}
        # Getting the type of 'mpath' (line 498)
        mpath_132730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 19), 'mpath', False)
        # Obtaining the member 'Path' of a type (line 498)
        Path_132731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 19), mpath_132730, 'Path')
        # Calling Path(args, kwargs) (line 498)
        Path_call_result_132740 = invoke(stypy.reporting.localization.Localization(__file__, 498, 19), Path_132731, *[list_132732], **kwargs_132739)
        
        # Assigning a type to the variable 'path' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'path', Path_call_result_132740)
        # SSA branch for the else part of an if statement (line 497)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'spine_type' (line 499)
        spine_type_132741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'spine_type')
        unicode_132742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 27), 'unicode', u'bottom')
        # Applying the binary operator '==' (line 499)
        result_eq_132743 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 13), '==', spine_type_132741, unicode_132742)
        
        # Testing the type of an if condition (line 499)
        if_condition_132744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 13), result_eq_132743)
        # Assigning a type to the variable 'if_condition_132744' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'if_condition_132744', if_condition_132744)
        # SSA begins for if statement (line 499)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to Path(...): (line 500)
        # Processing the call arguments (line 500)
        
        # Obtaining an instance of the builtin type 'list' (line 500)
        list_132747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 500)
        # Adding element type (line 500)
        
        # Obtaining an instance of the builtin type 'tuple' (line 500)
        tuple_132748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 500)
        # Adding element type (line 500)
        int_132749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 32), tuple_132748, int_132749)
        # Adding element type (line 500)
        float_132750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 32), tuple_132748, float_132750)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 30), list_132747, tuple_132748)
        # Adding element type (line 500)
        
        # Obtaining an instance of the builtin type 'tuple' (line 500)
        tuple_132751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 500)
        # Adding element type (line 500)
        int_132752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 43), tuple_132751, int_132752)
        # Adding element type (line 500)
        float_132753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 43), tuple_132751, float_132753)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 30), list_132747, tuple_132751)
        
        # Processing the call keyword arguments (line 500)
        kwargs_132754 = {}
        # Getting the type of 'mpath' (line 500)
        mpath_132745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 19), 'mpath', False)
        # Obtaining the member 'Path' of a type (line 500)
        Path_132746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 19), mpath_132745, 'Path')
        # Calling Path(args, kwargs) (line 500)
        Path_call_result_132755 = invoke(stypy.reporting.localization.Localization(__file__, 500, 19), Path_132746, *[list_132747], **kwargs_132754)
        
        # Assigning a type to the variable 'path' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'path', Path_call_result_132755)
        # SSA branch for the else part of an if statement (line 499)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'spine_type' (line 501)
        spine_type_132756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 13), 'spine_type')
        unicode_132757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 27), 'unicode', u'top')
        # Applying the binary operator '==' (line 501)
        result_eq_132758 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 13), '==', spine_type_132756, unicode_132757)
        
        # Testing the type of an if condition (line 501)
        if_condition_132759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 13), result_eq_132758)
        # Assigning a type to the variable 'if_condition_132759' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 13), 'if_condition_132759', if_condition_132759)
        # SSA begins for if statement (line 501)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 502):
        
        # Assigning a Call to a Name (line 502):
        
        # Call to Path(...): (line 502)
        # Processing the call arguments (line 502)
        
        # Obtaining an instance of the builtin type 'list' (line 502)
        list_132762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 502)
        # Adding element type (line 502)
        
        # Obtaining an instance of the builtin type 'tuple' (line 502)
        tuple_132763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 502)
        # Adding element type (line 502)
        int_132764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 32), tuple_132763, int_132764)
        # Adding element type (line 502)
        float_132765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 32), tuple_132763, float_132765)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 30), list_132762, tuple_132763)
        # Adding element type (line 502)
        
        # Obtaining an instance of the builtin type 'tuple' (line 502)
        tuple_132766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 502)
        # Adding element type (line 502)
        int_132767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 43), tuple_132766, int_132767)
        # Adding element type (line 502)
        float_132768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 43), tuple_132766, float_132768)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 30), list_132762, tuple_132766)
        
        # Processing the call keyword arguments (line 502)
        kwargs_132769 = {}
        # Getting the type of 'mpath' (line 502)
        mpath_132760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), 'mpath', False)
        # Obtaining the member 'Path' of a type (line 502)
        Path_132761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 19), mpath_132760, 'Path')
        # Calling Path(args, kwargs) (line 502)
        Path_call_result_132770 = invoke(stypy.reporting.localization.Localization(__file__, 502, 19), Path_132761, *[list_132762], **kwargs_132769)
        
        # Assigning a type to the variable 'path' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'path', Path_call_result_132770)
        # SSA branch for the else part of an if statement (line 501)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 504)
        # Processing the call arguments (line 504)
        unicode_132772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 29), 'unicode', u'unable to make path for spine "%s"')
        # Getting the type of 'spine_type' (line 504)
        spine_type_132773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 68), 'spine_type', False)
        # Applying the binary operator '%' (line 504)
        result_mod_132774 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 29), '%', unicode_132772, spine_type_132773)
        
        # Processing the call keyword arguments (line 504)
        kwargs_132775 = {}
        # Getting the type of 'ValueError' (line 504)
        ValueError_132771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 504)
        ValueError_call_result_132776 = invoke(stypy.reporting.localization.Localization(__file__, 504, 18), ValueError_132771, *[result_mod_132774], **kwargs_132775)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 504, 12), ValueError_call_result_132776, 'raise parameter', BaseException)
        # SSA join for if statement (line 501)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 499)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 495)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 505):
        
        # Assigning a Call to a Name (line 505):
        
        # Call to cls(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'axes' (line 505)
        axes_132778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 21), 'axes', False)
        # Getting the type of 'spine_type' (line 505)
        spine_type_132779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 27), 'spine_type', False)
        # Getting the type of 'path' (line 505)
        path_132780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 39), 'path', False)
        # Processing the call keyword arguments (line 505)
        # Getting the type of 'kwargs' (line 505)
        kwargs_132781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 47), 'kwargs', False)
        kwargs_132782 = {'kwargs_132781': kwargs_132781}
        # Getting the type of 'cls' (line 505)
        cls_132777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 17), 'cls', False)
        # Calling cls(args, kwargs) (line 505)
        cls_call_result_132783 = invoke(stypy.reporting.localization.Localization(__file__, 505, 17), cls_132777, *[axes_132778, spine_type_132779, path_132780], **kwargs_132782)
        
        # Assigning a type to the variable 'result' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'result', cls_call_result_132783)
        
        # Call to set_visible(...): (line 506)
        # Processing the call arguments (line 506)
        
        # Obtaining the type of the subscript
        
        # Call to format(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'spine_type' (line 506)
        spine_type_132788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 61), 'spine_type', False)
        # Processing the call keyword arguments (line 506)
        kwargs_132789 = {}
        unicode_132786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 36), 'unicode', u'axes.spines.{0}')
        # Obtaining the member 'format' of a type (line 506)
        format_132787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 36), unicode_132786, 'format')
        # Calling format(args, kwargs) (line 506)
        format_call_result_132790 = invoke(stypy.reporting.localization.Localization(__file__, 506, 36), format_132787, *[spine_type_132788], **kwargs_132789)
        
        # Getting the type of 'rcParams' (line 506)
        rcParams_132791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 27), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 506)
        getitem___132792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 27), rcParams_132791, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 506)
        subscript_call_result_132793 = invoke(stypy.reporting.localization.Localization(__file__, 506, 27), getitem___132792, format_call_result_132790)
        
        # Processing the call keyword arguments (line 506)
        kwargs_132794 = {}
        # Getting the type of 'result' (line 506)
        result_132784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'result', False)
        # Obtaining the member 'set_visible' of a type (line 506)
        set_visible_132785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), result_132784, 'set_visible')
        # Calling set_visible(args, kwargs) (line 506)
        set_visible_call_result_132795 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), set_visible_132785, *[subscript_call_result_132793], **kwargs_132794)
        
        # Getting the type of 'result' (line 508)
        result_132796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'stypy_return_type', result_132796)
        
        # ################# End of 'linear_spine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'linear_spine' in the type store
        # Getting the type of 'stypy_return_type' (line 489)
        stypy_return_type_132797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132797)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'linear_spine'
        return stypy_return_type_132797


    @norecursion
    def arc_spine(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'arc_spine'
        module_type_store = module_type_store.open_function_context('arc_spine', 510, 4, False)
        # Assigning a type to the variable 'self' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.arc_spine.__dict__.__setitem__('stypy_localization', localization)
        Spine.arc_spine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.arc_spine.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.arc_spine.__dict__.__setitem__('stypy_function_name', 'Spine.arc_spine')
        Spine.arc_spine.__dict__.__setitem__('stypy_param_names_list', ['axes', 'spine_type', 'center', 'radius', 'theta1', 'theta2'])
        Spine.arc_spine.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.arc_spine.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Spine.arc_spine.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.arc_spine.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.arc_spine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.arc_spine.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.arc_spine', ['axes', 'spine_type', 'center', 'radius', 'theta1', 'theta2'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'arc_spine', localization, ['axes', 'spine_type', 'center', 'radius', 'theta1', 'theta2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'arc_spine(...)' code ##################

        unicode_132798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, (-1)), 'unicode', u'\n        (classmethod) Returns an arc :class:`Spine`.\n        ')
        
        # Assigning a Call to a Name (line 516):
        
        # Assigning a Call to a Name (line 516):
        
        # Call to arc(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'theta1' (line 516)
        theta1_132802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 30), 'theta1', False)
        # Getting the type of 'theta2' (line 516)
        theta2_132803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 38), 'theta2', False)
        # Processing the call keyword arguments (line 516)
        kwargs_132804 = {}
        # Getting the type of 'mpath' (line 516)
        mpath_132799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'mpath', False)
        # Obtaining the member 'Path' of a type (line 516)
        Path_132800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 15), mpath_132799, 'Path')
        # Obtaining the member 'arc' of a type (line 516)
        arc_132801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 15), Path_132800, 'arc')
        # Calling arc(args, kwargs) (line 516)
        arc_call_result_132805 = invoke(stypy.reporting.localization.Localization(__file__, 516, 15), arc_132801, *[theta1_132802, theta2_132803], **kwargs_132804)
        
        # Assigning a type to the variable 'path' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'path', arc_call_result_132805)
        
        # Assigning a Call to a Name (line 517):
        
        # Assigning a Call to a Name (line 517):
        
        # Call to cls(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'axes' (line 517)
        axes_132807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 21), 'axes', False)
        # Getting the type of 'spine_type' (line 517)
        spine_type_132808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 27), 'spine_type', False)
        # Getting the type of 'path' (line 517)
        path_132809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 39), 'path', False)
        # Processing the call keyword arguments (line 517)
        # Getting the type of 'kwargs' (line 517)
        kwargs_132810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 47), 'kwargs', False)
        kwargs_132811 = {'kwargs_132810': kwargs_132810}
        # Getting the type of 'cls' (line 517)
        cls_132806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 17), 'cls', False)
        # Calling cls(args, kwargs) (line 517)
        cls_call_result_132812 = invoke(stypy.reporting.localization.Localization(__file__, 517, 17), cls_132806, *[axes_132807, spine_type_132808, path_132809], **kwargs_132811)
        
        # Assigning a type to the variable 'result' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'result', cls_call_result_132812)
        
        # Call to set_patch_arc(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'center' (line 518)
        center_132815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 29), 'center', False)
        # Getting the type of 'radius' (line 518)
        radius_132816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 37), 'radius', False)
        # Getting the type of 'theta1' (line 518)
        theta1_132817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 45), 'theta1', False)
        # Getting the type of 'theta2' (line 518)
        theta2_132818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 53), 'theta2', False)
        # Processing the call keyword arguments (line 518)
        kwargs_132819 = {}
        # Getting the type of 'result' (line 518)
        result_132813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'result', False)
        # Obtaining the member 'set_patch_arc' of a type (line 518)
        set_patch_arc_132814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), result_132813, 'set_patch_arc')
        # Calling set_patch_arc(args, kwargs) (line 518)
        set_patch_arc_call_result_132820 = invoke(stypy.reporting.localization.Localization(__file__, 518, 8), set_patch_arc_132814, *[center_132815, radius_132816, theta1_132817, theta2_132818], **kwargs_132819)
        
        # Getting the type of 'result' (line 519)
        result_132821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'stypy_return_type', result_132821)
        
        # ################# End of 'arc_spine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'arc_spine' in the type store
        # Getting the type of 'stypy_return_type' (line 510)
        stypy_return_type_132822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'arc_spine'
        return stypy_return_type_132822


    @norecursion
    def circular_spine(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'circular_spine'
        module_type_store = module_type_store.open_function_context('circular_spine', 521, 4, False)
        # Assigning a type to the variable 'self' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.circular_spine.__dict__.__setitem__('stypy_localization', localization)
        Spine.circular_spine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.circular_spine.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.circular_spine.__dict__.__setitem__('stypy_function_name', 'Spine.circular_spine')
        Spine.circular_spine.__dict__.__setitem__('stypy_param_names_list', ['axes', 'center', 'radius'])
        Spine.circular_spine.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.circular_spine.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Spine.circular_spine.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.circular_spine.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.circular_spine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.circular_spine.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.circular_spine', ['axes', 'center', 'radius'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'circular_spine', localization, ['axes', 'center', 'radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'circular_spine(...)' code ##################

        unicode_132823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, (-1)), 'unicode', u'\n        (staticmethod) Returns a circular :class:`Spine`.\n        ')
        
        # Assigning a Call to a Name (line 526):
        
        # Assigning a Call to a Name (line 526):
        
        # Call to unit_circle(...): (line 526)
        # Processing the call keyword arguments (line 526)
        kwargs_132827 = {}
        # Getting the type of 'mpath' (line 526)
        mpath_132824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'mpath', False)
        # Obtaining the member 'Path' of a type (line 526)
        Path_132825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 15), mpath_132824, 'Path')
        # Obtaining the member 'unit_circle' of a type (line 526)
        unit_circle_132826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 15), Path_132825, 'unit_circle')
        # Calling unit_circle(args, kwargs) (line 526)
        unit_circle_call_result_132828 = invoke(stypy.reporting.localization.Localization(__file__, 526, 15), unit_circle_132826, *[], **kwargs_132827)
        
        # Assigning a type to the variable 'path' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'path', unit_circle_call_result_132828)
        
        # Assigning a Str to a Name (line 527):
        
        # Assigning a Str to a Name (line 527):
        unicode_132829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 21), 'unicode', u'circle')
        # Assigning a type to the variable 'spine_type' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'spine_type', unicode_132829)
        
        # Assigning a Call to a Name (line 528):
        
        # Assigning a Call to a Name (line 528):
        
        # Call to cls(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'axes' (line 528)
        axes_132831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 21), 'axes', False)
        # Getting the type of 'spine_type' (line 528)
        spine_type_132832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 27), 'spine_type', False)
        # Getting the type of 'path' (line 528)
        path_132833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 39), 'path', False)
        # Processing the call keyword arguments (line 528)
        # Getting the type of 'kwargs' (line 528)
        kwargs_132834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 47), 'kwargs', False)
        kwargs_132835 = {'kwargs_132834': kwargs_132834}
        # Getting the type of 'cls' (line 528)
        cls_132830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 17), 'cls', False)
        # Calling cls(args, kwargs) (line 528)
        cls_call_result_132836 = invoke(stypy.reporting.localization.Localization(__file__, 528, 17), cls_132830, *[axes_132831, spine_type_132832, path_132833], **kwargs_132835)
        
        # Assigning a type to the variable 'result' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'result', cls_call_result_132836)
        
        # Call to set_patch_circle(...): (line 529)
        # Processing the call arguments (line 529)
        # Getting the type of 'center' (line 529)
        center_132839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 32), 'center', False)
        # Getting the type of 'radius' (line 529)
        radius_132840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 40), 'radius', False)
        # Processing the call keyword arguments (line 529)
        kwargs_132841 = {}
        # Getting the type of 'result' (line 529)
        result_132837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'result', False)
        # Obtaining the member 'set_patch_circle' of a type (line 529)
        set_patch_circle_132838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 8), result_132837, 'set_patch_circle')
        # Calling set_patch_circle(args, kwargs) (line 529)
        set_patch_circle_call_result_132842 = invoke(stypy.reporting.localization.Localization(__file__, 529, 8), set_patch_circle_132838, *[center_132839, radius_132840], **kwargs_132841)
        
        # Getting the type of 'result' (line 530)
        result_132843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'stypy_return_type', result_132843)
        
        # ################# End of 'circular_spine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'circular_spine' in the type store
        # Getting the type of 'stypy_return_type' (line 521)
        stypy_return_type_132844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'circular_spine'
        return stypy_return_type_132844


    @norecursion
    def set_color(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_color'
        module_type_store = module_type_store.open_function_context('set_color', 532, 4, False)
        # Assigning a type to the variable 'self' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spine.set_color.__dict__.__setitem__('stypy_localization', localization)
        Spine.set_color.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spine.set_color.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spine.set_color.__dict__.__setitem__('stypy_function_name', 'Spine.set_color')
        Spine.set_color.__dict__.__setitem__('stypy_param_names_list', ['c'])
        Spine.set_color.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spine.set_color.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spine.set_color.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spine.set_color.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spine.set_color.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spine.set_color.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spine.set_color', ['c'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_color', localization, ['c'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_color(...)' code ##################

        unicode_132845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, (-1)), 'unicode', u'\n        Set the edgecolor.\n\n        ACCEPTS: matplotlib color arg or sequence of rgba tuples\n\n        .. seealso::\n\n            :meth:`set_facecolor`, :meth:`set_edgecolor`\n               For setting the edge or face color individually.\n        ')
        
        # Call to set_edgecolor(...): (line 545)
        # Processing the call arguments (line 545)
        # Getting the type of 'c' (line 545)
        c_132848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 27), 'c', False)
        # Processing the call keyword arguments (line 545)
        kwargs_132849 = {}
        # Getting the type of 'self' (line 545)
        self_132846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'self', False)
        # Obtaining the member 'set_edgecolor' of a type (line 545)
        set_edgecolor_132847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 8), self_132846, 'set_edgecolor')
        # Calling set_edgecolor(args, kwargs) (line 545)
        set_edgecolor_call_result_132850 = invoke(stypy.reporting.localization.Localization(__file__, 545, 8), set_edgecolor_132847, *[c_132848], **kwargs_132849)
        
        
        # Assigning a Name to a Attribute (line 546):
        
        # Assigning a Name to a Attribute (line 546):
        # Getting the type of 'True' (line 546)
        True_132851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 21), 'True')
        # Getting the type of 'self' (line 546)
        self_132852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 546)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 8), self_132852, 'stale', True_132851)
        
        # ################# End of 'set_color(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_color' in the type store
        # Getting the type of 'stypy_return_type' (line 532)
        stypy_return_type_132853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132853)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_color'
        return stypy_return_type_132853


# Assigning a type to the variable 'Spine' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'Spine', Spine)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
