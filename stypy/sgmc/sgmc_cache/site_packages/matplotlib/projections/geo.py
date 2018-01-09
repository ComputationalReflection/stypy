
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import math
7: 
8: import numpy as np
9: import numpy.ma as ma
10: 
11: import matplotlib
12: rcParams = matplotlib.rcParams
13: from matplotlib.axes import Axes
14: from matplotlib import cbook
15: from matplotlib.patches import Circle
16: from matplotlib.path import Path
17: import matplotlib.spines as mspines
18: import matplotlib.axis as maxis
19: from matplotlib.ticker import Formatter, Locator, NullLocator, FixedLocator, NullFormatter
20: from matplotlib.transforms import Affine2D, Affine2DBase, Bbox, \
21:     BboxTransformTo, IdentityTransform, Transform, TransformWrapper
22: 
23: class GeoAxes(Axes):
24:     '''
25:     An abstract base class for geographic projections
26:     '''
27:     class ThetaFormatter(Formatter):
28:         '''
29:         Used to format the theta tick labels.  Converts the native
30:         unit of radians into degrees and adds a degree symbol.
31:         '''
32:         def __init__(self, round_to=1.0):
33:             self._round_to = round_to
34: 
35:         def __call__(self, x, pos=None):
36:             degrees = (x / np.pi) * 180.0
37:             degrees = np.round(degrees / self._round_to) * self._round_to
38:             if rcParams['text.usetex'] and not rcParams['text.latex.unicode']:
39:                 return r"$%0.0f^\circ$" % degrees
40:             else:
41:                 return "%0.0f\N{DEGREE SIGN}" % degrees
42: 
43:     RESOLUTION = 75
44: 
45:     def _init_axis(self):
46:         self.xaxis = maxis.XAxis(self)
47:         self.yaxis = maxis.YAxis(self)
48:         # Do not register xaxis or yaxis with spines -- as done in
49:         # Axes._init_axis() -- until GeoAxes.xaxis.cla() works.
50:         # self.spines['geo'].register_axis(self.yaxis)
51:         self._update_transScale()
52: 
53:     def cla(self):
54:         Axes.cla(self)
55: 
56:         self.set_longitude_grid(30)
57:         self.set_latitude_grid(15)
58:         self.set_longitude_grid_ends(75)
59:         self.xaxis.set_minor_locator(NullLocator())
60:         self.yaxis.set_minor_locator(NullLocator())
61:         self.xaxis.set_ticks_position('none')
62:         self.yaxis.set_ticks_position('none')
63:         self.yaxis.set_tick_params(label1On=True)
64:         # Why do we need to turn on yaxis tick labels, but
65:         # xaxis tick labels are already on?
66: 
67:         self.grid(rcParams['axes.grid'])
68: 
69:         Axes.set_xlim(self, -np.pi, np.pi)
70:         Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)
71: 
72:     def _set_lim_and_transforms(self):
73:         # A (possibly non-linear) projection on the (already scaled) data
74:         self.transProjection = self._get_core_transform(self.RESOLUTION)
75: 
76:         self.transAffine = self._get_affine_transform()
77: 
78:         self.transAxes = BboxTransformTo(self.bbox)
79: 
80:         # The complete data transformation stack -- from data all the
81:         # way to display coordinates
82:         self.transData = \
83:             self.transProjection + \
84:             self.transAffine + \
85:             self.transAxes
86: 
87:         # This is the transform for longitude ticks.
88:         self._xaxis_pretransform = \
89:             Affine2D() \
90:             .scale(1.0, self._longitude_cap * 2.0) \
91:             .translate(0.0, -self._longitude_cap)
92:         self._xaxis_transform = \
93:             self._xaxis_pretransform + \
94:             self.transData
95:         self._xaxis_text1_transform = \
96:             Affine2D().scale(1.0, 0.0) + \
97:             self.transData + \
98:             Affine2D().translate(0.0, 4.0)
99:         self._xaxis_text2_transform = \
100:             Affine2D().scale(1.0, 0.0) + \
101:             self.transData + \
102:             Affine2D().translate(0.0, -4.0)
103: 
104:         # This is the transform for latitude ticks.
105:         yaxis_stretch = Affine2D().scale(np.pi * 2.0, 1.0).translate(-np.pi, 0.0)
106:         yaxis_space = Affine2D().scale(1.0, 1.1)
107:         self._yaxis_transform = \
108:             yaxis_stretch + \
109:             self.transData
110:         yaxis_text_base = \
111:             yaxis_stretch + \
112:             self.transProjection + \
113:             (yaxis_space + \
114:              self.transAffine + \
115:              self.transAxes)
116:         self._yaxis_text1_transform = \
117:             yaxis_text_base + \
118:             Affine2D().translate(-8.0, 0.0)
119:         self._yaxis_text2_transform = \
120:             yaxis_text_base + \
121:             Affine2D().translate(8.0, 0.0)
122: 
123:     def _get_affine_transform(self):
124:         transform = self._get_core_transform(1)
125:         xscale, _ = transform.transform_point((np.pi, 0))
126:         _, yscale = transform.transform_point((0, np.pi / 2.0))
127:         return Affine2D() \
128:             .scale(0.5 / xscale, 0.5 / yscale) \
129:             .translate(0.5, 0.5)
130: 
131:     def get_xaxis_transform(self,which='grid'):
132:         if which not in ['tick1','tick2','grid']:
133:             msg = "'which' must be on of [ 'tick1' | 'tick2' | 'grid' ]"
134:             raise ValueError(msg)
135:         return self._xaxis_transform
136: 
137:     def get_xaxis_text1_transform(self, pad):
138:         return self._xaxis_text1_transform, 'bottom', 'center'
139: 
140:     def get_xaxis_text2_transform(self, pad):
141:         return self._xaxis_text2_transform, 'top', 'center'
142: 
143:     def get_yaxis_transform(self,which='grid'):
144:         if which not in ['tick1','tick2','grid']:
145:             msg = "'which' must be one of [ 'tick1' | 'tick2' | 'grid' ]"
146:             raise ValueError(msg)
147:         return self._yaxis_transform
148: 
149:     def get_yaxis_text1_transform(self, pad):
150:         return self._yaxis_text1_transform, 'center', 'right'
151: 
152:     def get_yaxis_text2_transform(self, pad):
153:         return self._yaxis_text2_transform, 'center', 'left'
154: 
155:     def _gen_axes_patch(self):
156:         return Circle((0.5, 0.5), 0.5)
157: 
158:     def _gen_axes_spines(self):
159:         return {'geo':mspines.Spine.circular_spine(self,
160:                                                    (0.5, 0.5), 0.5)}
161: 
162:     def set_yscale(self, *args, **kwargs):
163:         if args[0] != 'linear':
164:             raise NotImplementedError
165: 
166:     set_xscale = set_yscale
167: 
168:     def set_xlim(self, *args, **kwargs):
169:         raise TypeError("It is not possible to change axes limits "
170:                         "for geographic projections. Please consider "
171:                         "using Basemap or Cartopy.")
172: 
173:     set_ylim = set_xlim
174: 
175:     def format_coord(self, lon, lat):
176:         'return a format string formatting the coordinate'
177:         lon, lat = np.rad2deg([lon, lat])
178:         if lat >= 0.0:
179:             ns = 'N'
180:         else:
181:             ns = 'S'
182:         if lon >= 0.0:
183:             ew = 'E'
184:         else:
185:             ew = 'W'
186:         return ('%f\N{DEGREE SIGN}%s, %f\N{DEGREE SIGN}%s'
187:                 % (abs(lat), ns, abs(lon), ew))
188: 
189:     def set_longitude_grid(self, degrees):
190:         '''
191:         Set the number of degrees between each longitude grid.
192:         '''
193:         # Skip -180 and 180, which are the fixed limits.
194:         grid = np.arange(-180 + degrees, 180, degrees)
195:         self.xaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
196:         self.xaxis.set_major_formatter(self.ThetaFormatter(degrees))
197: 
198:     def set_latitude_grid(self, degrees):
199:         '''
200:         Set the number of degrees between each latitude grid.
201:         '''
202:         # Skip -90 and 90, which are the fixed limits.
203:         grid = np.arange(-90 + degrees, 90, degrees)
204:         self.yaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
205:         self.yaxis.set_major_formatter(self.ThetaFormatter(degrees))
206: 
207:     def set_longitude_grid_ends(self, degrees):
208:         '''
209:         Set the latitude(s) at which to stop drawing the longitude grids.
210:         '''
211:         self._longitude_cap = degrees * (np.pi / 180.0)
212:         self._xaxis_pretransform \
213:             .clear() \
214:             .scale(1.0, self._longitude_cap * 2.0) \
215:             .translate(0.0, -self._longitude_cap)
216: 
217:     def get_data_ratio(self):
218:         '''
219:         Return the aspect ratio of the data itself.
220:         '''
221:         return 1.0
222: 
223:     ### Interactive panning
224: 
225:     def can_zoom(self):
226:         '''
227:         Return *True* if this axes supports the zoom box button functionality.
228: 
229:         This axes object does not support interactive zoom box.
230:         '''
231:         return False
232: 
233:     def can_pan(self) :
234:         '''
235:         Return *True* if this axes supports the pan/zoom button functionality.
236: 
237:         This axes object does not support interactive pan/zoom.
238:         '''
239:         return False
240: 
241:     def start_pan(self, x, y, button):
242:         pass
243: 
244:     def end_pan(self):
245:         pass
246: 
247:     def drag_pan(self, button, key, x, y):
248:         pass
249: 
250: 
251: class AitoffAxes(GeoAxes):
252:     name = 'aitoff'
253: 
254:     class AitoffTransform(Transform):
255:         '''
256:         The base Aitoff transform.
257:         '''
258:         input_dims = 2
259:         output_dims = 2
260:         is_separable = False
261: 
262:         def __init__(self, resolution):
263:             '''
264:             Create a new Aitoff transform.  Resolution is the number of steps
265:             to interpolate between each input line segment to approximate its
266:             path in curved Aitoff space.
267:             '''
268:             Transform.__init__(self)
269:             self._resolution = resolution
270: 
271:         def transform_non_affine(self, ll):
272:             longitude = ll[:, 0:1]
273:             latitude  = ll[:, 1:2]
274: 
275:             # Pre-compute some values
276:             half_long = longitude / 2.0
277:             cos_latitude = np.cos(latitude)
278: 
279:             alpha = np.arccos(cos_latitude * np.cos(half_long))
280:             # Mask this array or we'll get divide-by-zero errors
281:             alpha = ma.masked_where(alpha == 0.0, alpha)
282:             # The numerators also need to be masked so that masked
283:             # division will be invoked.
284:             # We want unnormalized sinc.  numpy.sinc gives us normalized
285:             sinc_alpha = ma.sin(alpha) / alpha
286: 
287:             x = (cos_latitude * ma.sin(half_long)) / sinc_alpha
288:             y = (ma.sin(latitude) / sinc_alpha)
289:             return np.concatenate((x.filled(0), y.filled(0)), 1)
290:         transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__
291: 
292:         def transform_path_non_affine(self, path):
293:             vertices = path.vertices
294:             ipath = path.interpolated(self._resolution)
295:             return Path(self.transform(ipath.vertices), ipath.codes)
296:         transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__
297: 
298:         def inverted(self):
299:             return AitoffAxes.InvertedAitoffTransform(self._resolution)
300:         inverted.__doc__ = Transform.inverted.__doc__
301: 
302:     class InvertedAitoffTransform(Transform):
303:         input_dims = 2
304:         output_dims = 2
305:         is_separable = False
306: 
307:         def __init__(self, resolution):
308:             Transform.__init__(self)
309:             self._resolution = resolution
310: 
311:         def transform_non_affine(self, xy):
312:             # MGDTODO: Math is hard ;(
313:             return xy
314:         transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__
315: 
316:         def inverted(self):
317:             return AitoffAxes.AitoffTransform(self._resolution)
318:         inverted.__doc__ = Transform.inverted.__doc__
319: 
320:     def __init__(self, *args, **kwargs):
321:         self._longitude_cap = np.pi / 2.0
322:         GeoAxes.__init__(self, *args, **kwargs)
323:         self.set_aspect(0.5, adjustable='box', anchor='C')
324:         self.cla()
325: 
326:     def _get_core_transform(self, resolution):
327:         return self.AitoffTransform(resolution)
328: 
329: 
330: class HammerAxes(GeoAxes):
331:     name = 'hammer'
332: 
333:     class HammerTransform(Transform):
334:         '''
335:         The base Hammer transform.
336:         '''
337:         input_dims = 2
338:         output_dims = 2
339:         is_separable = False
340: 
341:         def __init__(self, resolution):
342:             '''
343:             Create a new Hammer transform.  Resolution is the number of steps
344:             to interpolate between each input line segment to approximate its
345:             path in curved Hammer space.
346:             '''
347:             Transform.__init__(self)
348:             self._resolution = resolution
349: 
350:         def transform_non_affine(self, ll):
351:             longitude = ll[:, 0:1]
352:             latitude  = ll[:, 1:2]
353: 
354:             # Pre-compute some values
355:             half_long = longitude / 2.0
356:             cos_latitude = np.cos(latitude)
357:             sqrt2 = np.sqrt(2.0)
358: 
359:             alpha = np.sqrt(1.0 + cos_latitude * np.cos(half_long))
360:             x = (2.0 * sqrt2) * (cos_latitude * np.sin(half_long)) / alpha
361:             y = (sqrt2 * np.sin(latitude)) / alpha
362:             return np.concatenate((x, y), 1)
363:         transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__
364: 
365:         def transform_path_non_affine(self, path):
366:             vertices = path.vertices
367:             ipath = path.interpolated(self._resolution)
368:             return Path(self.transform(ipath.vertices), ipath.codes)
369:         transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__
370: 
371:         def inverted(self):
372:             return HammerAxes.InvertedHammerTransform(self._resolution)
373:         inverted.__doc__ = Transform.inverted.__doc__
374: 
375:     class InvertedHammerTransform(Transform):
376:         input_dims = 2
377:         output_dims = 2
378:         is_separable = False
379: 
380:         def __init__(self, resolution):
381:             Transform.__init__(self)
382:             self._resolution = resolution
383: 
384:         def transform_non_affine(self, xy):
385:             x, y = xy.T
386:             z = np.sqrt(1 - (x / 4) ** 2 - (y / 2) ** 2)
387:             longitude = 2 * np.arctan((z * x) / (2 * (2 * z ** 2 - 1)))
388:             latitude = np.arcsin(y*z)
389:             return np.column_stack([longitude, latitude])
390:         transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__
391: 
392:         def inverted(self):
393:             return HammerAxes.HammerTransform(self._resolution)
394:         inverted.__doc__ = Transform.inverted.__doc__
395: 
396:     def __init__(self, *args, **kwargs):
397:         self._longitude_cap = np.pi / 2.0
398:         GeoAxes.__init__(self, *args, **kwargs)
399:         self.set_aspect(0.5, adjustable='box', anchor='C')
400:         self.cla()
401: 
402:     def _get_core_transform(self, resolution):
403:         return self.HammerTransform(resolution)
404: 
405: 
406: class MollweideAxes(GeoAxes):
407:     name = 'mollweide'
408: 
409:     class MollweideTransform(Transform):
410:         '''
411:         The base Mollweide transform.
412:         '''
413:         input_dims = 2
414:         output_dims = 2
415:         is_separable = False
416: 
417:         def __init__(self, resolution):
418:             '''
419:             Create a new Mollweide transform.  Resolution is the number of steps
420:             to interpolate between each input line segment to approximate its
421:             path in curved Mollweide space.
422:             '''
423:             Transform.__init__(self)
424:             self._resolution = resolution
425: 
426:         def transform_non_affine(self, ll):
427:             def d(theta):
428:                 delta = -(theta + np.sin(theta) - pi_sin_l) / (1 + np.cos(theta))
429:                 return delta, np.abs(delta) > 0.001
430: 
431:             longitude = ll[:, 0]
432:             latitude  = ll[:, 1]
433: 
434:             clat = np.pi/2 - np.abs(latitude)
435:             ihigh = clat < 0.087 # within 5 degrees of the poles
436:             ilow = ~ihigh
437:             aux = np.empty(latitude.shape, dtype=float)
438: 
439:             if ilow.any():  # Newton-Raphson iteration
440:                 pi_sin_l = np.pi * np.sin(latitude[ilow])
441:                 theta = 2.0 * latitude[ilow]
442:                 delta, large_delta = d(theta)
443:                 while np.any(large_delta):
444:                     theta[large_delta] += delta[large_delta]
445:                     delta, large_delta = d(theta)
446:                 aux[ilow] = theta / 2
447: 
448:             if ihigh.any(): # Taylor series-based approx. solution
449:                 e = clat[ihigh]
450:                 d = 0.5 * (3 * np.pi * e**2) ** (1.0/3)
451:                 aux[ihigh] = (np.pi/2 - d) * np.sign(latitude[ihigh])
452: 
453:             xy = np.empty(ll.shape, dtype=float)
454:             xy[:,0] = (2.0 * np.sqrt(2.0) / np.pi) * longitude * np.cos(aux)
455:             xy[:,1] = np.sqrt(2.0) * np.sin(aux)
456: 
457:             return xy
458:         transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__
459: 
460:         def transform_path_non_affine(self, path):
461:             vertices = path.vertices
462:             ipath = path.interpolated(self._resolution)
463:             return Path(self.transform(ipath.vertices), ipath.codes)
464:         transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__
465: 
466:         def inverted(self):
467:             return MollweideAxes.InvertedMollweideTransform(self._resolution)
468:         inverted.__doc__ = Transform.inverted.__doc__
469: 
470:     class InvertedMollweideTransform(Transform):
471:         input_dims = 2
472:         output_dims = 2
473:         is_separable = False
474: 
475:         def __init__(self, resolution):
476:             Transform.__init__(self)
477:             self._resolution = resolution
478: 
479:         def transform_non_affine(self, xy):
480:             x = xy[:, 0:1]
481:             y = xy[:, 1:2]
482: 
483:             # from Equations (7, 8) of
484:             # http://mathworld.wolfram.com/MollweideProjection.html
485:             theta = np.arcsin(y / np.sqrt(2))
486:             lon = (np.pi / (2 * np.sqrt(2))) * x / np.cos(theta)
487:             lat = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
488: 
489:             return np.concatenate((lon, lat), 1)
490:         transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__
491: 
492:         def inverted(self):
493:             return MollweideAxes.MollweideTransform(self._resolution)
494:         inverted.__doc__ = Transform.inverted.__doc__
495: 
496:     def __init__(self, *args, **kwargs):
497:         self._longitude_cap = np.pi / 2.0
498:         GeoAxes.__init__(self, *args, **kwargs)
499:         self.set_aspect(0.5, adjustable='box', anchor='C')
500:         self.cla()
501: 
502:     def _get_core_transform(self, resolution):
503:         return self.MollweideTransform(resolution)
504: 
505: 
506: class LambertAxes(GeoAxes):
507:     name = 'lambert'
508: 
509:     class LambertTransform(Transform):
510:         '''
511:         The base Lambert transform.
512:         '''
513:         input_dims = 2
514:         output_dims = 2
515:         is_separable = False
516: 
517:         def __init__(self, center_longitude, center_latitude, resolution):
518:             '''
519:             Create a new Lambert transform.  Resolution is the number of steps
520:             to interpolate between each input line segment to approximate its
521:             path in curved Lambert space.
522:             '''
523:             Transform.__init__(self)
524:             self._resolution = resolution
525:             self._center_longitude = center_longitude
526:             self._center_latitude = center_latitude
527: 
528:         def transform_non_affine(self, ll):
529:             longitude = ll[:, 0:1]
530:             latitude  = ll[:, 1:2]
531:             clong = self._center_longitude
532:             clat = self._center_latitude
533:             cos_lat = np.cos(latitude)
534:             sin_lat = np.sin(latitude)
535:             diff_long = longitude - clong
536:             cos_diff_long = np.cos(diff_long)
537: 
538:             inner_k = (1.0 +
539:                        np.sin(clat)*sin_lat +
540:                        np.cos(clat)*cos_lat*cos_diff_long)
541:             # Prevent divide-by-zero problems
542:             inner_k = np.where(inner_k == 0.0, 1e-15, inner_k)
543:             k = np.sqrt(2.0 / inner_k)
544:             x = k*cos_lat*np.sin(diff_long)
545:             y = k*(np.cos(clat)*sin_lat -
546:                    np.sin(clat)*cos_lat*cos_diff_long)
547: 
548:             return np.concatenate((x, y), 1)
549:         transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__
550: 
551:         def transform_path_non_affine(self, path):
552:             vertices = path.vertices
553:             ipath = path.interpolated(self._resolution)
554:             return Path(self.transform(ipath.vertices), ipath.codes)
555:         transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__
556: 
557:         def inverted(self):
558:             return LambertAxes.InvertedLambertTransform(
559:                 self._center_longitude,
560:                 self._center_latitude,
561:                 self._resolution)
562:         inverted.__doc__ = Transform.inverted.__doc__
563: 
564:     class InvertedLambertTransform(Transform):
565:         input_dims = 2
566:         output_dims = 2
567:         is_separable = False
568: 
569:         def __init__(self, center_longitude, center_latitude, resolution):
570:             Transform.__init__(self)
571:             self._resolution = resolution
572:             self._center_longitude = center_longitude
573:             self._center_latitude = center_latitude
574: 
575:         def transform_non_affine(self, xy):
576:             x = xy[:, 0:1]
577:             y = xy[:, 1:2]
578:             clong = self._center_longitude
579:             clat = self._center_latitude
580:             p = np.sqrt(x*x + y*y)
581:             p = np.where(p == 0.0, 1e-9, p)
582:             c = 2.0 * np.arcsin(0.5 * p)
583:             sin_c = np.sin(c)
584:             cos_c = np.cos(c)
585: 
586:             lat = np.arcsin(cos_c*np.sin(clat) +
587:                              ((y*sin_c*np.cos(clat)) / p))
588:             lon = clong + np.arctan(
589:                 (x*sin_c) / (p*np.cos(clat)*cos_c - y*np.sin(clat)*sin_c))
590: 
591:             return np.concatenate((lon, lat), 1)
592:         transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__
593: 
594:         def inverted(self):
595:             return LambertAxes.LambertTransform(
596:                 self._center_longitude,
597:                 self._center_latitude,
598:                 self._resolution)
599:         inverted.__doc__ = Transform.inverted.__doc__
600: 
601:     def __init__(self, *args, **kwargs):
602:         self._longitude_cap = np.pi / 2.0
603:         self._center_longitude = kwargs.pop("center_longitude", 0.0)
604:         self._center_latitude = kwargs.pop("center_latitude", 0.0)
605:         GeoAxes.__init__(self, *args, **kwargs)
606:         self.set_aspect('equal', adjustable='box', anchor='C')
607:         self.cla()
608: 
609:     def cla(self):
610:         GeoAxes.cla(self)
611:         self.yaxis.set_major_formatter(NullFormatter())
612: 
613:     def _get_core_transform(self, resolution):
614:         return self.LambertTransform(
615:             self._center_longitude,
616:             self._center_latitude,
617:             resolution)
618: 
619:     def _get_affine_transform(self):
620:         return Affine2D() \
621:             .scale(0.25) \
622:             .translate(0.5, 0.5)
623: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279217 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_279217) is not StypyTypeError):

    if (import_279217 != 'pyd_module'):
        __import__(import_279217)
        sys_modules_279218 = sys.modules[import_279217]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_279218.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_279217)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import math' statement (line 6)
import math

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279219 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_279219) is not StypyTypeError):

    if (import_279219 != 'pyd_module'):
        __import__(import_279219)
        sys_modules_279220 = sys.modules[import_279219]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_279220.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_279219)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy.ma' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279221 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.ma')

if (type(import_279221) is not StypyTypeError):

    if (import_279221 != 'pyd_module'):
        __import__(import_279221)
        sys_modules_279222 = sys.modules[import_279221]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'ma', sys_modules_279222.module_type_store, module_type_store)
    else:
        import numpy.ma as ma

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'ma', numpy.ma, module_type_store)

else:
    # Assigning a type to the variable 'numpy.ma' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.ma', import_279221)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import matplotlib' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279223 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib')

if (type(import_279223) is not StypyTypeError):

    if (import_279223 != 'pyd_module'):
        __import__(import_279223)
        sys_modules_279224 = sys.modules[import_279223]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib', sys_modules_279224.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib', import_279223)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')


# Assigning a Attribute to a Name (line 12):

# Assigning a Attribute to a Name (line 12):
# Getting the type of 'matplotlib' (line 12)
matplotlib_279225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'matplotlib')
# Obtaining the member 'rcParams' of a type (line 12)
rcParams_279226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 11), matplotlib_279225, 'rcParams')
# Assigning a type to the variable 'rcParams' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'rcParams', rcParams_279226)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib.axes import Axes' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279227 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.axes')

if (type(import_279227) is not StypyTypeError):

    if (import_279227 != 'pyd_module'):
        __import__(import_279227)
        sys_modules_279228 = sys.modules[import_279227]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.axes', sys_modules_279228.module_type_store, module_type_store, ['Axes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_279228, sys_modules_279228.module_type_store, module_type_store)
    else:
        from matplotlib.axes import Axes

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.axes', None, module_type_store, ['Axes'], [Axes])

else:
    # Assigning a type to the variable 'matplotlib.axes' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.axes', import_279227)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib import cbook' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279229 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_279229) is not StypyTypeError):

    if (import_279229 != 'pyd_module'):
        __import__(import_279229)
        sys_modules_279230 = sys.modules[import_279229]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_279230.module_type_store, module_type_store, ['cbook'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_279230, sys_modules_279230.module_type_store, module_type_store)
    else:
        from matplotlib import cbook

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', None, module_type_store, ['cbook'], [cbook])

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_279229)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.patches import Circle' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279231 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.patches')

if (type(import_279231) is not StypyTypeError):

    if (import_279231 != 'pyd_module'):
        __import__(import_279231)
        sys_modules_279232 = sys.modules[import_279231]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.patches', sys_modules_279232.module_type_store, module_type_store, ['Circle'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_279232, sys_modules_279232.module_type_store, module_type_store)
    else:
        from matplotlib.patches import Circle

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.patches', None, module_type_store, ['Circle'], [Circle])

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.patches', import_279231)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib.path import Path' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279233 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.path')

if (type(import_279233) is not StypyTypeError):

    if (import_279233 != 'pyd_module'):
        __import__(import_279233)
        sys_modules_279234 = sys.modules[import_279233]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.path', sys_modules_279234.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_279234, sys_modules_279234.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.path', import_279233)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import matplotlib.spines' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279235 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.spines')

if (type(import_279235) is not StypyTypeError):

    if (import_279235 != 'pyd_module'):
        __import__(import_279235)
        sys_modules_279236 = sys.modules[import_279235]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'mspines', sys_modules_279236.module_type_store, module_type_store)
    else:
        import matplotlib.spines as mspines

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'mspines', matplotlib.spines, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.spines' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.spines', import_279235)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import matplotlib.axis' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279237 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.axis')

if (type(import_279237) is not StypyTypeError):

    if (import_279237 != 'pyd_module'):
        __import__(import_279237)
        sys_modules_279238 = sys.modules[import_279237]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'maxis', sys_modules_279238.module_type_store, module_type_store)
    else:
        import matplotlib.axis as maxis

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'maxis', matplotlib.axis, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.axis' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.axis', import_279237)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from matplotlib.ticker import Formatter, Locator, NullLocator, FixedLocator, NullFormatter' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279239 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.ticker')

if (type(import_279239) is not StypyTypeError):

    if (import_279239 != 'pyd_module'):
        __import__(import_279239)
        sys_modules_279240 = sys.modules[import_279239]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.ticker', sys_modules_279240.module_type_store, module_type_store, ['Formatter', 'Locator', 'NullLocator', 'FixedLocator', 'NullFormatter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_279240, sys_modules_279240.module_type_store, module_type_store)
    else:
        from matplotlib.ticker import Formatter, Locator, NullLocator, FixedLocator, NullFormatter

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.ticker', None, module_type_store, ['Formatter', 'Locator', 'NullLocator', 'FixedLocator', 'NullFormatter'], [Formatter, Locator, NullLocator, FixedLocator, NullFormatter])

else:
    # Assigning a type to the variable 'matplotlib.ticker' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.ticker', import_279239)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from matplotlib.transforms import Affine2D, Affine2DBase, Bbox, BboxTransformTo, IdentityTransform, Transform, TransformWrapper' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_279241 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.transforms')

if (type(import_279241) is not StypyTypeError):

    if (import_279241 != 'pyd_module'):
        __import__(import_279241)
        sys_modules_279242 = sys.modules[import_279241]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.transforms', sys_modules_279242.module_type_store, module_type_store, ['Affine2D', 'Affine2DBase', 'Bbox', 'BboxTransformTo', 'IdentityTransform', 'Transform', 'TransformWrapper'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_279242, sys_modules_279242.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Affine2D, Affine2DBase, Bbox, BboxTransformTo, IdentityTransform, Transform, TransformWrapper

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.transforms', None, module_type_store, ['Affine2D', 'Affine2DBase', 'Bbox', 'BboxTransformTo', 'IdentityTransform', 'Transform', 'TransformWrapper'], [Affine2D, Affine2DBase, Bbox, BboxTransformTo, IdentityTransform, Transform, TransformWrapper])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.transforms', import_279241)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

# Declaration of the 'GeoAxes' class
# Getting the type of 'Axes' (line 23)
Axes_279243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'Axes')

class GeoAxes(Axes_279243, ):
    unicode_279244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'unicode', u'\n    An abstract base class for geographic projections\n    ')
    # Declaration of the 'ThetaFormatter' class
    # Getting the type of 'Formatter' (line 27)
    Formatter_279245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'Formatter')

    class ThetaFormatter(Formatter_279245, ):
        unicode_279246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'unicode', u'\n        Used to format the theta tick labels.  Converts the native\n        unit of radians into degrees and adds a degree symbol.\n        ')

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            float_279247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'float')
            defaults = [float_279247]
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 32, 8, False)
            # Assigning a type to the variable 'self' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'ThetaFormatter.__init__', ['round_to'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['round_to'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Assigning a Name to a Attribute (line 33):
            
            # Assigning a Name to a Attribute (line 33):
            # Getting the type of 'round_to' (line 33)
            round_to_279248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'round_to')
            # Getting the type of 'self' (line 33)
            self_279249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'self')
            # Setting the type of the member '_round_to' of a type (line 33)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), self_279249, '_round_to', round_to_279248)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def __call__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'None' (line 35)
            None_279250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'None')
            defaults = [None_279250]
            # Create a new context for function '__call__'
            module_type_store = module_type_store.open_function_context('__call__', 35, 8, False)
            # Assigning a type to the variable 'self' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_localization', localization)
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_function_name', 'ThetaFormatter.__call__')
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_param_names_list', ['x', 'pos'])
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            ThetaFormatter.__call__.__dict__.__setitem__('stypy_declared_arg_number', 3)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'ThetaFormatter.__call__', ['x', 'pos'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__call__', localization, ['x', 'pos'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__call__(...)' code ##################

            
            # Assigning a BinOp to a Name (line 36):
            
            # Assigning a BinOp to a Name (line 36):
            # Getting the type of 'x' (line 36)
            x_279251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'x')
            # Getting the type of 'np' (line 36)
            np_279252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'np')
            # Obtaining the member 'pi' of a type (line 36)
            pi_279253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), np_279252, 'pi')
            # Applying the binary operator 'div' (line 36)
            result_div_279254 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), 'div', x_279251, pi_279253)
            
            float_279255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'float')
            # Applying the binary operator '*' (line 36)
            result_mul_279256 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 22), '*', result_div_279254, float_279255)
            
            # Assigning a type to the variable 'degrees' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'degrees', result_mul_279256)
            
            # Assigning a BinOp to a Name (line 37):
            
            # Assigning a BinOp to a Name (line 37):
            
            # Call to round(...): (line 37)
            # Processing the call arguments (line 37)
            # Getting the type of 'degrees' (line 37)
            degrees_279259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'degrees', False)
            # Getting the type of 'self' (line 37)
            self_279260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 41), 'self', False)
            # Obtaining the member '_round_to' of a type (line 37)
            _round_to_279261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 41), self_279260, '_round_to')
            # Applying the binary operator 'div' (line 37)
            result_div_279262 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 31), 'div', degrees_279259, _round_to_279261)
            
            # Processing the call keyword arguments (line 37)
            kwargs_279263 = {}
            # Getting the type of 'np' (line 37)
            np_279257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'np', False)
            # Obtaining the member 'round' of a type (line 37)
            round_279258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 22), np_279257, 'round')
            # Calling round(args, kwargs) (line 37)
            round_call_result_279264 = invoke(stypy.reporting.localization.Localization(__file__, 37, 22), round_279258, *[result_div_279262], **kwargs_279263)
            
            # Getting the type of 'self' (line 37)
            self_279265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 59), 'self')
            # Obtaining the member '_round_to' of a type (line 37)
            _round_to_279266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 59), self_279265, '_round_to')
            # Applying the binary operator '*' (line 37)
            result_mul_279267 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 22), '*', round_call_result_279264, _round_to_279266)
            
            # Assigning a type to the variable 'degrees' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'degrees', result_mul_279267)
            
            
            # Evaluating a boolean operation
            
            # Obtaining the type of the subscript
            unicode_279268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'unicode', u'text.usetex')
            # Getting the type of 'rcParams' (line 38)
            rcParams_279269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 38)
            getitem___279270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), rcParams_279269, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 38)
            subscript_call_result_279271 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), getitem___279270, unicode_279268)
            
            
            
            # Obtaining the type of the subscript
            unicode_279272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 56), 'unicode', u'text.latex.unicode')
            # Getting the type of 'rcParams' (line 38)
            rcParams_279273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 47), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 38)
            getitem___279274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 47), rcParams_279273, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 38)
            subscript_call_result_279275 = invoke(stypy.reporting.localization.Localization(__file__, 38, 47), getitem___279274, unicode_279272)
            
            # Applying the 'not' unary operator (line 38)
            result_not__279276 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 43), 'not', subscript_call_result_279275)
            
            # Applying the binary operator 'and' (line 38)
            result_and_keyword_279277 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 15), 'and', subscript_call_result_279271, result_not__279276)
            
            # Testing the type of an if condition (line 38)
            if_condition_279278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 12), result_and_keyword_279277)
            # Assigning a type to the variable 'if_condition_279278' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'if_condition_279278', if_condition_279278)
            # SSA begins for if statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            unicode_279279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'unicode', u'$%0.0f^\\circ$')
            # Getting the type of 'degrees' (line 39)
            degrees_279280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 42), 'degrees')
            # Applying the binary operator '%' (line 39)
            result_mod_279281 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '%', unicode_279279, degrees_279280)
            
            # Assigning a type to the variable 'stypy_return_type' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'stypy_return_type', result_mod_279281)
            # SSA branch for the else part of an if statement (line 38)
            module_type_store.open_ssa_branch('else')
            unicode_279282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'unicode', u'%0.0f\xb0')
            # Getting the type of 'degrees' (line 41)
            degrees_279283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 48), 'degrees')
            # Applying the binary operator '%' (line 41)
            result_mod_279284 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 23), '%', unicode_279282, degrees_279283)
            
            # Assigning a type to the variable 'stypy_return_type' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'stypy_return_type', result_mod_279284)
            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of '__call__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__call__' in the type store
            # Getting the type of 'stypy_return_type' (line 35)
            stypy_return_type_279285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_279285)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__call__'
            return stypy_return_type_279285

    
    # Assigning a type to the variable 'ThetaFormatter' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ThetaFormatter', ThetaFormatter)
    
    # Assigning a Num to a Name (line 43):

    @norecursion
    def _init_axis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_axis'
        module_type_store = module_type_store.open_function_context('_init_axis', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes._init_axis.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes._init_axis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes._init_axis.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes._init_axis.__dict__.__setitem__('stypy_function_name', 'GeoAxes._init_axis')
        GeoAxes._init_axis.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes._init_axis.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes._init_axis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes._init_axis.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes._init_axis.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes._init_axis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes._init_axis.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes._init_axis', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_axis', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_axis(...)' code ##################

        
        # Assigning a Call to a Attribute (line 46):
        
        # Assigning a Call to a Attribute (line 46):
        
        # Call to XAxis(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_279288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'self', False)
        # Processing the call keyword arguments (line 46)
        kwargs_279289 = {}
        # Getting the type of 'maxis' (line 46)
        maxis_279286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'maxis', False)
        # Obtaining the member 'XAxis' of a type (line 46)
        XAxis_279287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 21), maxis_279286, 'XAxis')
        # Calling XAxis(args, kwargs) (line 46)
        XAxis_call_result_279290 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), XAxis_279287, *[self_279288], **kwargs_279289)
        
        # Getting the type of 'self' (line 46)
        self_279291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'xaxis' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_279291, 'xaxis', XAxis_call_result_279290)
        
        # Assigning a Call to a Attribute (line 47):
        
        # Assigning a Call to a Attribute (line 47):
        
        # Call to YAxis(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_279294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'self', False)
        # Processing the call keyword arguments (line 47)
        kwargs_279295 = {}
        # Getting the type of 'maxis' (line 47)
        maxis_279292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'maxis', False)
        # Obtaining the member 'YAxis' of a type (line 47)
        YAxis_279293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 21), maxis_279292, 'YAxis')
        # Calling YAxis(args, kwargs) (line 47)
        YAxis_call_result_279296 = invoke(stypy.reporting.localization.Localization(__file__, 47, 21), YAxis_279293, *[self_279294], **kwargs_279295)
        
        # Getting the type of 'self' (line 47)
        self_279297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'yaxis' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_279297, 'yaxis', YAxis_call_result_279296)
        
        # Call to _update_transScale(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_279300 = {}
        # Getting the type of 'self' (line 51)
        self_279298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self', False)
        # Obtaining the member '_update_transScale' of a type (line 51)
        _update_transScale_279299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_279298, '_update_transScale')
        # Calling _update_transScale(args, kwargs) (line 51)
        _update_transScale_call_result_279301 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), _update_transScale_279299, *[], **kwargs_279300)
        
        
        # ################# End of '_init_axis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_axis' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_279302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_axis'
        return stypy_return_type_279302


    @norecursion
    def cla(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cla'
        module_type_store = module_type_store.open_function_context('cla', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.cla.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.cla.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.cla.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.cla.__dict__.__setitem__('stypy_function_name', 'GeoAxes.cla')
        GeoAxes.cla.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes.cla.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.cla.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.cla.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.cla.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.cla.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.cla.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.cla', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to cla(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_279305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'self', False)
        # Processing the call keyword arguments (line 54)
        kwargs_279306 = {}
        # Getting the type of 'Axes' (line 54)
        Axes_279303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'Axes', False)
        # Obtaining the member 'cla' of a type (line 54)
        cla_279304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), Axes_279303, 'cla')
        # Calling cla(args, kwargs) (line 54)
        cla_call_result_279307 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), cla_279304, *[self_279305], **kwargs_279306)
        
        
        # Call to set_longitude_grid(...): (line 56)
        # Processing the call arguments (line 56)
        int_279310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 32), 'int')
        # Processing the call keyword arguments (line 56)
        kwargs_279311 = {}
        # Getting the type of 'self' (line 56)
        self_279308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self', False)
        # Obtaining the member 'set_longitude_grid' of a type (line 56)
        set_longitude_grid_279309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_279308, 'set_longitude_grid')
        # Calling set_longitude_grid(args, kwargs) (line 56)
        set_longitude_grid_call_result_279312 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), set_longitude_grid_279309, *[int_279310], **kwargs_279311)
        
        
        # Call to set_latitude_grid(...): (line 57)
        # Processing the call arguments (line 57)
        int_279315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'int')
        # Processing the call keyword arguments (line 57)
        kwargs_279316 = {}
        # Getting the type of 'self' (line 57)
        self_279313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'set_latitude_grid' of a type (line 57)
        set_latitude_grid_279314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_279313, 'set_latitude_grid')
        # Calling set_latitude_grid(args, kwargs) (line 57)
        set_latitude_grid_call_result_279317 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), set_latitude_grid_279314, *[int_279315], **kwargs_279316)
        
        
        # Call to set_longitude_grid_ends(...): (line 58)
        # Processing the call arguments (line 58)
        int_279320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 37), 'int')
        # Processing the call keyword arguments (line 58)
        kwargs_279321 = {}
        # Getting the type of 'self' (line 58)
        self_279318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', False)
        # Obtaining the member 'set_longitude_grid_ends' of a type (line 58)
        set_longitude_grid_ends_279319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_279318, 'set_longitude_grid_ends')
        # Calling set_longitude_grid_ends(args, kwargs) (line 58)
        set_longitude_grid_ends_call_result_279322 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), set_longitude_grid_ends_279319, *[int_279320], **kwargs_279321)
        
        
        # Call to set_minor_locator(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Call to NullLocator(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_279327 = {}
        # Getting the type of 'NullLocator' (line 59)
        NullLocator_279326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'NullLocator', False)
        # Calling NullLocator(args, kwargs) (line 59)
        NullLocator_call_result_279328 = invoke(stypy.reporting.localization.Localization(__file__, 59, 37), NullLocator_279326, *[], **kwargs_279327)
        
        # Processing the call keyword arguments (line 59)
        kwargs_279329 = {}
        # Getting the type of 'self' (line 59)
        self_279323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self', False)
        # Obtaining the member 'xaxis' of a type (line 59)
        xaxis_279324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_279323, 'xaxis')
        # Obtaining the member 'set_minor_locator' of a type (line 59)
        set_minor_locator_279325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), xaxis_279324, 'set_minor_locator')
        # Calling set_minor_locator(args, kwargs) (line 59)
        set_minor_locator_call_result_279330 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), set_minor_locator_279325, *[NullLocator_call_result_279328], **kwargs_279329)
        
        
        # Call to set_minor_locator(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to NullLocator(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_279335 = {}
        # Getting the type of 'NullLocator' (line 60)
        NullLocator_279334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'NullLocator', False)
        # Calling NullLocator(args, kwargs) (line 60)
        NullLocator_call_result_279336 = invoke(stypy.reporting.localization.Localization(__file__, 60, 37), NullLocator_279334, *[], **kwargs_279335)
        
        # Processing the call keyword arguments (line 60)
        kwargs_279337 = {}
        # Getting the type of 'self' (line 60)
        self_279331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'yaxis' of a type (line 60)
        yaxis_279332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_279331, 'yaxis')
        # Obtaining the member 'set_minor_locator' of a type (line 60)
        set_minor_locator_279333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), yaxis_279332, 'set_minor_locator')
        # Calling set_minor_locator(args, kwargs) (line 60)
        set_minor_locator_call_result_279338 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), set_minor_locator_279333, *[NullLocator_call_result_279336], **kwargs_279337)
        
        
        # Call to set_ticks_position(...): (line 61)
        # Processing the call arguments (line 61)
        unicode_279342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'unicode', u'none')
        # Processing the call keyword arguments (line 61)
        kwargs_279343 = {}
        # Getting the type of 'self' (line 61)
        self_279339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'xaxis' of a type (line 61)
        xaxis_279340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_279339, 'xaxis')
        # Obtaining the member 'set_ticks_position' of a type (line 61)
        set_ticks_position_279341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), xaxis_279340, 'set_ticks_position')
        # Calling set_ticks_position(args, kwargs) (line 61)
        set_ticks_position_call_result_279344 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), set_ticks_position_279341, *[unicode_279342], **kwargs_279343)
        
        
        # Call to set_ticks_position(...): (line 62)
        # Processing the call arguments (line 62)
        unicode_279348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 38), 'unicode', u'none')
        # Processing the call keyword arguments (line 62)
        kwargs_279349 = {}
        # Getting the type of 'self' (line 62)
        self_279345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member 'yaxis' of a type (line 62)
        yaxis_279346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_279345, 'yaxis')
        # Obtaining the member 'set_ticks_position' of a type (line 62)
        set_ticks_position_279347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), yaxis_279346, 'set_ticks_position')
        # Calling set_ticks_position(args, kwargs) (line 62)
        set_ticks_position_call_result_279350 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), set_ticks_position_279347, *[unicode_279348], **kwargs_279349)
        
        
        # Call to set_tick_params(...): (line 63)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'True' (line 63)
        True_279354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 44), 'True', False)
        keyword_279355 = True_279354
        kwargs_279356 = {'label1On': keyword_279355}
        # Getting the type of 'self' (line 63)
        self_279351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'yaxis' of a type (line 63)
        yaxis_279352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_279351, 'yaxis')
        # Obtaining the member 'set_tick_params' of a type (line 63)
        set_tick_params_279353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), yaxis_279352, 'set_tick_params')
        # Calling set_tick_params(args, kwargs) (line 63)
        set_tick_params_call_result_279357 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), set_tick_params_279353, *[], **kwargs_279356)
        
        
        # Call to grid(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Obtaining the type of the subscript
        unicode_279360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'unicode', u'axes.grid')
        # Getting the type of 'rcParams' (line 67)
        rcParams_279361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 67)
        getitem___279362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), rcParams_279361, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 67)
        subscript_call_result_279363 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), getitem___279362, unicode_279360)
        
        # Processing the call keyword arguments (line 67)
        kwargs_279364 = {}
        # Getting the type of 'self' (line 67)
        self_279358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'grid' of a type (line 67)
        grid_279359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_279358, 'grid')
        # Calling grid(args, kwargs) (line 67)
        grid_call_result_279365 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), grid_279359, *[subscript_call_result_279363], **kwargs_279364)
        
        
        # Call to set_xlim(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'self' (line 69)
        self_279368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'self', False)
        
        # Getting the type of 'np' (line 69)
        np_279369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'np', False)
        # Obtaining the member 'pi' of a type (line 69)
        pi_279370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 29), np_279369, 'pi')
        # Applying the 'usub' unary operator (line 69)
        result___neg___279371 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 28), 'usub', pi_279370)
        
        # Getting the type of 'np' (line 69)
        np_279372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 36), 'np', False)
        # Obtaining the member 'pi' of a type (line 69)
        pi_279373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 36), np_279372, 'pi')
        # Processing the call keyword arguments (line 69)
        kwargs_279374 = {}
        # Getting the type of 'Axes' (line 69)
        Axes_279366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'Axes', False)
        # Obtaining the member 'set_xlim' of a type (line 69)
        set_xlim_279367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), Axes_279366, 'set_xlim')
        # Calling set_xlim(args, kwargs) (line 69)
        set_xlim_call_result_279375 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), set_xlim_279367, *[self_279368, result___neg___279371, pi_279373], **kwargs_279374)
        
        
        # Call to set_ylim(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_279378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'self', False)
        
        # Getting the type of 'np' (line 70)
        np_279379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), 'np', False)
        # Obtaining the member 'pi' of a type (line 70)
        pi_279380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 29), np_279379, 'pi')
        # Applying the 'usub' unary operator (line 70)
        result___neg___279381 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 28), 'usub', pi_279380)
        
        float_279382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'float')
        # Applying the binary operator 'div' (line 70)
        result_div_279383 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 28), 'div', result___neg___279381, float_279382)
        
        # Getting the type of 'np' (line 70)
        np_279384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 42), 'np', False)
        # Obtaining the member 'pi' of a type (line 70)
        pi_279385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 42), np_279384, 'pi')
        float_279386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 50), 'float')
        # Applying the binary operator 'div' (line 70)
        result_div_279387 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 42), 'div', pi_279385, float_279386)
        
        # Processing the call keyword arguments (line 70)
        kwargs_279388 = {}
        # Getting the type of 'Axes' (line 70)
        Axes_279376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'Axes', False)
        # Obtaining the member 'set_ylim' of a type (line 70)
        set_ylim_279377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), Axes_279376, 'set_ylim')
        # Calling set_ylim(args, kwargs) (line 70)
        set_ylim_call_result_279389 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), set_ylim_279377, *[self_279378, result_div_279383, result_div_279387], **kwargs_279388)
        
        
        # ################# End of 'cla(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cla' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_279390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279390)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cla'
        return stypy_return_type_279390


    @norecursion
    def _set_lim_and_transforms(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_lim_and_transforms'
        module_type_store = module_type_store.open_function_context('_set_lim_and_transforms', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_function_name', 'GeoAxes._set_lim_and_transforms')
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes._set_lim_and_transforms.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes._set_lim_and_transforms', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_lim_and_transforms', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_lim_and_transforms(...)' code ##################

        
        # Assigning a Call to a Attribute (line 74):
        
        # Assigning a Call to a Attribute (line 74):
        
        # Call to _get_core_transform(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'self' (line 74)
        self_279393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 56), 'self', False)
        # Obtaining the member 'RESOLUTION' of a type (line 74)
        RESOLUTION_279394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 56), self_279393, 'RESOLUTION')
        # Processing the call keyword arguments (line 74)
        kwargs_279395 = {}
        # Getting the type of 'self' (line 74)
        self_279391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 31), 'self', False)
        # Obtaining the member '_get_core_transform' of a type (line 74)
        _get_core_transform_279392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 31), self_279391, '_get_core_transform')
        # Calling _get_core_transform(args, kwargs) (line 74)
        _get_core_transform_call_result_279396 = invoke(stypy.reporting.localization.Localization(__file__, 74, 31), _get_core_transform_279392, *[RESOLUTION_279394], **kwargs_279395)
        
        # Getting the type of 'self' (line 74)
        self_279397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'transProjection' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_279397, 'transProjection', _get_core_transform_call_result_279396)
        
        # Assigning a Call to a Attribute (line 76):
        
        # Assigning a Call to a Attribute (line 76):
        
        # Call to _get_affine_transform(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_279400 = {}
        # Getting the type of 'self' (line 76)
        self_279398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'self', False)
        # Obtaining the member '_get_affine_transform' of a type (line 76)
        _get_affine_transform_279399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 27), self_279398, '_get_affine_transform')
        # Calling _get_affine_transform(args, kwargs) (line 76)
        _get_affine_transform_call_result_279401 = invoke(stypy.reporting.localization.Localization(__file__, 76, 27), _get_affine_transform_279399, *[], **kwargs_279400)
        
        # Getting the type of 'self' (line 76)
        self_279402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'transAffine' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_279402, 'transAffine', _get_affine_transform_call_result_279401)
        
        # Assigning a Call to a Attribute (line 78):
        
        # Assigning a Call to a Attribute (line 78):
        
        # Call to BboxTransformTo(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'self' (line 78)
        self_279404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 41), 'self', False)
        # Obtaining the member 'bbox' of a type (line 78)
        bbox_279405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 41), self_279404, 'bbox')
        # Processing the call keyword arguments (line 78)
        kwargs_279406 = {}
        # Getting the type of 'BboxTransformTo' (line 78)
        BboxTransformTo_279403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'BboxTransformTo', False)
        # Calling BboxTransformTo(args, kwargs) (line 78)
        BboxTransformTo_call_result_279407 = invoke(stypy.reporting.localization.Localization(__file__, 78, 25), BboxTransformTo_279403, *[bbox_279405], **kwargs_279406)
        
        # Getting the type of 'self' (line 78)
        self_279408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'transAxes' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_279408, 'transAxes', BboxTransformTo_call_result_279407)
        
        # Assigning a BinOp to a Attribute (line 82):
        
        # Assigning a BinOp to a Attribute (line 82):
        # Getting the type of 'self' (line 83)
        self_279409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'self')
        # Obtaining the member 'transProjection' of a type (line 83)
        transProjection_279410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), self_279409, 'transProjection')
        # Getting the type of 'self' (line 84)
        self_279411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self')
        # Obtaining the member 'transAffine' of a type (line 84)
        transAffine_279412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), self_279411, 'transAffine')
        # Applying the binary operator '+' (line 83)
        result_add_279413 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 12), '+', transProjection_279410, transAffine_279412)
        
        # Getting the type of 'self' (line 85)
        self_279414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'self')
        # Obtaining the member 'transAxes' of a type (line 85)
        transAxes_279415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), self_279414, 'transAxes')
        # Applying the binary operator '+' (line 84)
        result_add_279416 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 29), '+', result_add_279413, transAxes_279415)
        
        # Getting the type of 'self' (line 82)
        self_279417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'transData' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_279417, 'transData', result_add_279416)
        
        # Assigning a Call to a Attribute (line 88):
        
        # Assigning a Call to a Attribute (line 88):
        
        # Call to translate(...): (line 89)
        # Processing the call arguments (line 89)
        float_279430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 23), 'float')
        
        # Getting the type of 'self' (line 91)
        self_279431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'self', False)
        # Obtaining the member '_longitude_cap' of a type (line 91)
        _longitude_cap_279432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), self_279431, '_longitude_cap')
        # Applying the 'usub' unary operator (line 91)
        result___neg___279433 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 28), 'usub', _longitude_cap_279432)
        
        # Processing the call keyword arguments (line 89)
        kwargs_279434 = {}
        
        # Call to scale(...): (line 89)
        # Processing the call arguments (line 89)
        float_279422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'float')
        # Getting the type of 'self' (line 90)
        self_279423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'self', False)
        # Obtaining the member '_longitude_cap' of a type (line 90)
        _longitude_cap_279424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 24), self_279423, '_longitude_cap')
        float_279425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 46), 'float')
        # Applying the binary operator '*' (line 90)
        result_mul_279426 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 24), '*', _longitude_cap_279424, float_279425)
        
        # Processing the call keyword arguments (line 89)
        kwargs_279427 = {}
        
        # Call to Affine2D(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_279419 = {}
        # Getting the type of 'Affine2D' (line 89)
        Affine2D_279418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 89)
        Affine2D_call_result_279420 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), Affine2D_279418, *[], **kwargs_279419)
        
        # Obtaining the member 'scale' of a type (line 89)
        scale_279421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), Affine2D_call_result_279420, 'scale')
        # Calling scale(args, kwargs) (line 89)
        scale_call_result_279428 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), scale_279421, *[float_279422, result_mul_279426], **kwargs_279427)
        
        # Obtaining the member 'translate' of a type (line 89)
        translate_279429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), scale_call_result_279428, 'translate')
        # Calling translate(args, kwargs) (line 89)
        translate_call_result_279435 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), translate_279429, *[float_279430, result___neg___279433], **kwargs_279434)
        
        # Getting the type of 'self' (line 88)
        self_279436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Setting the type of the member '_xaxis_pretransform' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_279436, '_xaxis_pretransform', translate_call_result_279435)
        
        # Assigning a BinOp to a Attribute (line 92):
        
        # Assigning a BinOp to a Attribute (line 92):
        # Getting the type of 'self' (line 93)
        self_279437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self')
        # Obtaining the member '_xaxis_pretransform' of a type (line 93)
        _xaxis_pretransform_279438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_279437, '_xaxis_pretransform')
        # Getting the type of 'self' (line 94)
        self_279439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'self')
        # Obtaining the member 'transData' of a type (line 94)
        transData_279440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), self_279439, 'transData')
        # Applying the binary operator '+' (line 93)
        result_add_279441 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '+', _xaxis_pretransform_279438, transData_279440)
        
        # Getting the type of 'self' (line 92)
        self_279442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member '_xaxis_transform' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_279442, '_xaxis_transform', result_add_279441)
        
        # Assigning a BinOp to a Attribute (line 95):
        
        # Assigning a BinOp to a Attribute (line 95):
        
        # Call to scale(...): (line 96)
        # Processing the call arguments (line 96)
        float_279447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 29), 'float')
        float_279448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 34), 'float')
        # Processing the call keyword arguments (line 96)
        kwargs_279449 = {}
        
        # Call to Affine2D(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_279444 = {}
        # Getting the type of 'Affine2D' (line 96)
        Affine2D_279443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 96)
        Affine2D_call_result_279445 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), Affine2D_279443, *[], **kwargs_279444)
        
        # Obtaining the member 'scale' of a type (line 96)
        scale_279446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), Affine2D_call_result_279445, 'scale')
        # Calling scale(args, kwargs) (line 96)
        scale_call_result_279450 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), scale_279446, *[float_279447, float_279448], **kwargs_279449)
        
        # Getting the type of 'self' (line 97)
        self_279451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'self')
        # Obtaining the member 'transData' of a type (line 97)
        transData_279452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), self_279451, 'transData')
        # Applying the binary operator '+' (line 96)
        result_add_279453 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 12), '+', scale_call_result_279450, transData_279452)
        
        
        # Call to translate(...): (line 98)
        # Processing the call arguments (line 98)
        float_279458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 33), 'float')
        float_279459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 38), 'float')
        # Processing the call keyword arguments (line 98)
        kwargs_279460 = {}
        
        # Call to Affine2D(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_279455 = {}
        # Getting the type of 'Affine2D' (line 98)
        Affine2D_279454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 98)
        Affine2D_call_result_279456 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), Affine2D_279454, *[], **kwargs_279455)
        
        # Obtaining the member 'translate' of a type (line 98)
        translate_279457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), Affine2D_call_result_279456, 'translate')
        # Calling translate(args, kwargs) (line 98)
        translate_call_result_279461 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), translate_279457, *[float_279458, float_279459], **kwargs_279460)
        
        # Applying the binary operator '+' (line 97)
        result_add_279462 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 27), '+', result_add_279453, translate_call_result_279461)
        
        # Getting the type of 'self' (line 95)
        self_279463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member '_xaxis_text1_transform' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_279463, '_xaxis_text1_transform', result_add_279462)
        
        # Assigning a BinOp to a Attribute (line 99):
        
        # Assigning a BinOp to a Attribute (line 99):
        
        # Call to scale(...): (line 100)
        # Processing the call arguments (line 100)
        float_279468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'float')
        float_279469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 34), 'float')
        # Processing the call keyword arguments (line 100)
        kwargs_279470 = {}
        
        # Call to Affine2D(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_279465 = {}
        # Getting the type of 'Affine2D' (line 100)
        Affine2D_279464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 100)
        Affine2D_call_result_279466 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), Affine2D_279464, *[], **kwargs_279465)
        
        # Obtaining the member 'scale' of a type (line 100)
        scale_279467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), Affine2D_call_result_279466, 'scale')
        # Calling scale(args, kwargs) (line 100)
        scale_call_result_279471 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), scale_279467, *[float_279468, float_279469], **kwargs_279470)
        
        # Getting the type of 'self' (line 101)
        self_279472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self')
        # Obtaining the member 'transData' of a type (line 101)
        transData_279473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_279472, 'transData')
        # Applying the binary operator '+' (line 100)
        result_add_279474 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 12), '+', scale_call_result_279471, transData_279473)
        
        
        # Call to translate(...): (line 102)
        # Processing the call arguments (line 102)
        float_279479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 33), 'float')
        float_279480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 38), 'float')
        # Processing the call keyword arguments (line 102)
        kwargs_279481 = {}
        
        # Call to Affine2D(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_279476 = {}
        # Getting the type of 'Affine2D' (line 102)
        Affine2D_279475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 102)
        Affine2D_call_result_279477 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), Affine2D_279475, *[], **kwargs_279476)
        
        # Obtaining the member 'translate' of a type (line 102)
        translate_279478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), Affine2D_call_result_279477, 'translate')
        # Calling translate(args, kwargs) (line 102)
        translate_call_result_279482 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), translate_279478, *[float_279479, float_279480], **kwargs_279481)
        
        # Applying the binary operator '+' (line 101)
        result_add_279483 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 27), '+', result_add_279474, translate_call_result_279482)
        
        # Getting the type of 'self' (line 99)
        self_279484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member '_xaxis_text2_transform' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_279484, '_xaxis_text2_transform', result_add_279483)
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to translate(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Getting the type of 'np' (line 105)
        np_279497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 70), 'np', False)
        # Obtaining the member 'pi' of a type (line 105)
        pi_279498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 70), np_279497, 'pi')
        # Applying the 'usub' unary operator (line 105)
        result___neg___279499 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 69), 'usub', pi_279498)
        
        float_279500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 77), 'float')
        # Processing the call keyword arguments (line 105)
        kwargs_279501 = {}
        
        # Call to scale(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'np' (line 105)
        np_279489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 41), 'np', False)
        # Obtaining the member 'pi' of a type (line 105)
        pi_279490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 41), np_279489, 'pi')
        float_279491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 49), 'float')
        # Applying the binary operator '*' (line 105)
        result_mul_279492 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 41), '*', pi_279490, float_279491)
        
        float_279493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 54), 'float')
        # Processing the call keyword arguments (line 105)
        kwargs_279494 = {}
        
        # Call to Affine2D(...): (line 105)
        # Processing the call keyword arguments (line 105)
        kwargs_279486 = {}
        # Getting the type of 'Affine2D' (line 105)
        Affine2D_279485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 105)
        Affine2D_call_result_279487 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), Affine2D_279485, *[], **kwargs_279486)
        
        # Obtaining the member 'scale' of a type (line 105)
        scale_279488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 24), Affine2D_call_result_279487, 'scale')
        # Calling scale(args, kwargs) (line 105)
        scale_call_result_279495 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), scale_279488, *[result_mul_279492, float_279493], **kwargs_279494)
        
        # Obtaining the member 'translate' of a type (line 105)
        translate_279496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 24), scale_call_result_279495, 'translate')
        # Calling translate(args, kwargs) (line 105)
        translate_call_result_279502 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), translate_279496, *[result___neg___279499, float_279500], **kwargs_279501)
        
        # Assigning a type to the variable 'yaxis_stretch' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'yaxis_stretch', translate_call_result_279502)
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to scale(...): (line 106)
        # Processing the call arguments (line 106)
        float_279507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 39), 'float')
        float_279508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 44), 'float')
        # Processing the call keyword arguments (line 106)
        kwargs_279509 = {}
        
        # Call to Affine2D(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_279504 = {}
        # Getting the type of 'Affine2D' (line 106)
        Affine2D_279503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 106)
        Affine2D_call_result_279505 = invoke(stypy.reporting.localization.Localization(__file__, 106, 22), Affine2D_279503, *[], **kwargs_279504)
        
        # Obtaining the member 'scale' of a type (line 106)
        scale_279506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 22), Affine2D_call_result_279505, 'scale')
        # Calling scale(args, kwargs) (line 106)
        scale_call_result_279510 = invoke(stypy.reporting.localization.Localization(__file__, 106, 22), scale_279506, *[float_279507, float_279508], **kwargs_279509)
        
        # Assigning a type to the variable 'yaxis_space' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'yaxis_space', scale_call_result_279510)
        
        # Assigning a BinOp to a Attribute (line 107):
        
        # Assigning a BinOp to a Attribute (line 107):
        # Getting the type of 'yaxis_stretch' (line 108)
        yaxis_stretch_279511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'yaxis_stretch')
        # Getting the type of 'self' (line 109)
        self_279512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'self')
        # Obtaining the member 'transData' of a type (line 109)
        transData_279513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), self_279512, 'transData')
        # Applying the binary operator '+' (line 108)
        result_add_279514 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 12), '+', yaxis_stretch_279511, transData_279513)
        
        # Getting the type of 'self' (line 107)
        self_279515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self')
        # Setting the type of the member '_yaxis_transform' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_279515, '_yaxis_transform', result_add_279514)
        
        # Assigning a BinOp to a Name (line 110):
        
        # Assigning a BinOp to a Name (line 110):
        # Getting the type of 'yaxis_stretch' (line 111)
        yaxis_stretch_279516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'yaxis_stretch')
        # Getting the type of 'self' (line 112)
        self_279517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'self')
        # Obtaining the member 'transProjection' of a type (line 112)
        transProjection_279518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), self_279517, 'transProjection')
        # Applying the binary operator '+' (line 111)
        result_add_279519 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 12), '+', yaxis_stretch_279516, transProjection_279518)
        
        # Getting the type of 'yaxis_space' (line 113)
        yaxis_space_279520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'yaxis_space')
        # Getting the type of 'self' (line 114)
        self_279521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'self')
        # Obtaining the member 'transAffine' of a type (line 114)
        transAffine_279522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), self_279521, 'transAffine')
        # Applying the binary operator '+' (line 113)
        result_add_279523 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 13), '+', yaxis_space_279520, transAffine_279522)
        
        # Getting the type of 'self' (line 115)
        self_279524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'self')
        # Obtaining the member 'transAxes' of a type (line 115)
        transAxes_279525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 13), self_279524, 'transAxes')
        # Applying the binary operator '+' (line 114)
        result_add_279526 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 30), '+', result_add_279523, transAxes_279525)
        
        # Applying the binary operator '+' (line 112)
        result_add_279527 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 33), '+', result_add_279519, result_add_279526)
        
        # Assigning a type to the variable 'yaxis_text_base' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'yaxis_text_base', result_add_279527)
        
        # Assigning a BinOp to a Attribute (line 116):
        
        # Assigning a BinOp to a Attribute (line 116):
        # Getting the type of 'yaxis_text_base' (line 117)
        yaxis_text_base_279528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'yaxis_text_base')
        
        # Call to translate(...): (line 118)
        # Processing the call arguments (line 118)
        float_279533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 33), 'float')
        float_279534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 39), 'float')
        # Processing the call keyword arguments (line 118)
        kwargs_279535 = {}
        
        # Call to Affine2D(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_279530 = {}
        # Getting the type of 'Affine2D' (line 118)
        Affine2D_279529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 118)
        Affine2D_call_result_279531 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), Affine2D_279529, *[], **kwargs_279530)
        
        # Obtaining the member 'translate' of a type (line 118)
        translate_279532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), Affine2D_call_result_279531, 'translate')
        # Calling translate(args, kwargs) (line 118)
        translate_call_result_279536 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), translate_279532, *[float_279533, float_279534], **kwargs_279535)
        
        # Applying the binary operator '+' (line 117)
        result_add_279537 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '+', yaxis_text_base_279528, translate_call_result_279536)
        
        # Getting the type of 'self' (line 116)
        self_279538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member '_yaxis_text1_transform' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_279538, '_yaxis_text1_transform', result_add_279537)
        
        # Assigning a BinOp to a Attribute (line 119):
        
        # Assigning a BinOp to a Attribute (line 119):
        # Getting the type of 'yaxis_text_base' (line 120)
        yaxis_text_base_279539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'yaxis_text_base')
        
        # Call to translate(...): (line 121)
        # Processing the call arguments (line 121)
        float_279544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 33), 'float')
        float_279545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 38), 'float')
        # Processing the call keyword arguments (line 121)
        kwargs_279546 = {}
        
        # Call to Affine2D(...): (line 121)
        # Processing the call keyword arguments (line 121)
        kwargs_279541 = {}
        # Getting the type of 'Affine2D' (line 121)
        Affine2D_279540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 121)
        Affine2D_call_result_279542 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), Affine2D_279540, *[], **kwargs_279541)
        
        # Obtaining the member 'translate' of a type (line 121)
        translate_279543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), Affine2D_call_result_279542, 'translate')
        # Calling translate(args, kwargs) (line 121)
        translate_call_result_279547 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), translate_279543, *[float_279544, float_279545], **kwargs_279546)
        
        # Applying the binary operator '+' (line 120)
        result_add_279548 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), '+', yaxis_text_base_279539, translate_call_result_279547)
        
        # Getting the type of 'self' (line 119)
        self_279549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member '_yaxis_text2_transform' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_279549, '_yaxis_text2_transform', result_add_279548)
        
        # ################# End of '_set_lim_and_transforms(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_lim_and_transforms' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_279550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279550)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_lim_and_transforms'
        return stypy_return_type_279550


    @norecursion
    def _get_affine_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_affine_transform'
        module_type_store = module_type_store.open_function_context('_get_affine_transform', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_function_name', 'GeoAxes._get_affine_transform')
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes._get_affine_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes._get_affine_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_affine_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_affine_transform(...)' code ##################

        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to _get_core_transform(...): (line 124)
        # Processing the call arguments (line 124)
        int_279553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 45), 'int')
        # Processing the call keyword arguments (line 124)
        kwargs_279554 = {}
        # Getting the type of 'self' (line 124)
        self_279551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'self', False)
        # Obtaining the member '_get_core_transform' of a type (line 124)
        _get_core_transform_279552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), self_279551, '_get_core_transform')
        # Calling _get_core_transform(args, kwargs) (line 124)
        _get_core_transform_call_result_279555 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), _get_core_transform_279552, *[int_279553], **kwargs_279554)
        
        # Assigning a type to the variable 'transform' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'transform', _get_core_transform_call_result_279555)
        
        # Assigning a Call to a Tuple (line 125):
        
        # Assigning a Call to a Name:
        
        # Call to transform_point(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_279558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        # Getting the type of 'np' (line 125)
        np_279559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'np', False)
        # Obtaining the member 'pi' of a type (line 125)
        pi_279560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 47), np_279559, 'pi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 47), tuple_279558, pi_279560)
        # Adding element type (line 125)
        int_279561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 47), tuple_279558, int_279561)
        
        # Processing the call keyword arguments (line 125)
        kwargs_279562 = {}
        # Getting the type of 'transform' (line 125)
        transform_279556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'transform', False)
        # Obtaining the member 'transform_point' of a type (line 125)
        transform_point_279557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 20), transform_279556, 'transform_point')
        # Calling transform_point(args, kwargs) (line 125)
        transform_point_call_result_279563 = invoke(stypy.reporting.localization.Localization(__file__, 125, 20), transform_point_279557, *[tuple_279558], **kwargs_279562)
        
        # Assigning a type to the variable 'call_assignment_279200' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'call_assignment_279200', transform_point_call_result_279563)
        
        # Assigning a Call to a Name (line 125):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_279566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 8), 'int')
        # Processing the call keyword arguments
        kwargs_279567 = {}
        # Getting the type of 'call_assignment_279200' (line 125)
        call_assignment_279200_279564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'call_assignment_279200', False)
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___279565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), call_assignment_279200_279564, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_279568 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___279565, *[int_279566], **kwargs_279567)
        
        # Assigning a type to the variable 'call_assignment_279201' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'call_assignment_279201', getitem___call_result_279568)
        
        # Assigning a Name to a Name (line 125):
        # Getting the type of 'call_assignment_279201' (line 125)
        call_assignment_279201_279569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'call_assignment_279201')
        # Assigning a type to the variable 'xscale' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'xscale', call_assignment_279201_279569)
        
        # Assigning a Call to a Name (line 125):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_279572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 8), 'int')
        # Processing the call keyword arguments
        kwargs_279573 = {}
        # Getting the type of 'call_assignment_279200' (line 125)
        call_assignment_279200_279570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'call_assignment_279200', False)
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___279571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), call_assignment_279200_279570, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_279574 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___279571, *[int_279572], **kwargs_279573)
        
        # Assigning a type to the variable 'call_assignment_279202' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'call_assignment_279202', getitem___call_result_279574)
        
        # Assigning a Name to a Name (line 125):
        # Getting the type of 'call_assignment_279202' (line 125)
        call_assignment_279202_279575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'call_assignment_279202')
        # Assigning a type to the variable '_' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), '_', call_assignment_279202_279575)
        
        # Assigning a Call to a Tuple (line 126):
        
        # Assigning a Call to a Name:
        
        # Call to transform_point(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_279578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        int_279579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 47), tuple_279578, int_279579)
        # Adding element type (line 126)
        # Getting the type of 'np' (line 126)
        np_279580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 50), 'np', False)
        # Obtaining the member 'pi' of a type (line 126)
        pi_279581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 50), np_279580, 'pi')
        float_279582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 58), 'float')
        # Applying the binary operator 'div' (line 126)
        result_div_279583 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 50), 'div', pi_279581, float_279582)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 47), tuple_279578, result_div_279583)
        
        # Processing the call keyword arguments (line 126)
        kwargs_279584 = {}
        # Getting the type of 'transform' (line 126)
        transform_279576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'transform', False)
        # Obtaining the member 'transform_point' of a type (line 126)
        transform_point_279577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 20), transform_279576, 'transform_point')
        # Calling transform_point(args, kwargs) (line 126)
        transform_point_call_result_279585 = invoke(stypy.reporting.localization.Localization(__file__, 126, 20), transform_point_279577, *[tuple_279578], **kwargs_279584)
        
        # Assigning a type to the variable 'call_assignment_279203' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'call_assignment_279203', transform_point_call_result_279585)
        
        # Assigning a Call to a Name (line 126):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_279588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 8), 'int')
        # Processing the call keyword arguments
        kwargs_279589 = {}
        # Getting the type of 'call_assignment_279203' (line 126)
        call_assignment_279203_279586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'call_assignment_279203', False)
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___279587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), call_assignment_279203_279586, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_279590 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___279587, *[int_279588], **kwargs_279589)
        
        # Assigning a type to the variable 'call_assignment_279204' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'call_assignment_279204', getitem___call_result_279590)
        
        # Assigning a Name to a Name (line 126):
        # Getting the type of 'call_assignment_279204' (line 126)
        call_assignment_279204_279591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'call_assignment_279204')
        # Assigning a type to the variable '_' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), '_', call_assignment_279204_279591)
        
        # Assigning a Call to a Name (line 126):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_279594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 8), 'int')
        # Processing the call keyword arguments
        kwargs_279595 = {}
        # Getting the type of 'call_assignment_279203' (line 126)
        call_assignment_279203_279592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'call_assignment_279203', False)
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___279593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), call_assignment_279203_279592, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_279596 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___279593, *[int_279594], **kwargs_279595)
        
        # Assigning a type to the variable 'call_assignment_279205' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'call_assignment_279205', getitem___call_result_279596)
        
        # Assigning a Name to a Name (line 126):
        # Getting the type of 'call_assignment_279205' (line 126)
        call_assignment_279205_279597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'call_assignment_279205')
        # Assigning a type to the variable 'yscale' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'yscale', call_assignment_279205_279597)
        
        # Call to translate(...): (line 127)
        # Processing the call arguments (line 127)
        float_279611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 23), 'float')
        float_279612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'float')
        # Processing the call keyword arguments (line 127)
        kwargs_279613 = {}
        
        # Call to scale(...): (line 127)
        # Processing the call arguments (line 127)
        float_279602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'float')
        # Getting the type of 'xscale' (line 128)
        xscale_279603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'xscale', False)
        # Applying the binary operator 'div' (line 128)
        result_div_279604 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 19), 'div', float_279602, xscale_279603)
        
        float_279605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 33), 'float')
        # Getting the type of 'yscale' (line 128)
        yscale_279606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 39), 'yscale', False)
        # Applying the binary operator 'div' (line 128)
        result_div_279607 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 33), 'div', float_279605, yscale_279606)
        
        # Processing the call keyword arguments (line 127)
        kwargs_279608 = {}
        
        # Call to Affine2D(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_279599 = {}
        # Getting the type of 'Affine2D' (line 127)
        Affine2D_279598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 127)
        Affine2D_call_result_279600 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), Affine2D_279598, *[], **kwargs_279599)
        
        # Obtaining the member 'scale' of a type (line 127)
        scale_279601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), Affine2D_call_result_279600, 'scale')
        # Calling scale(args, kwargs) (line 127)
        scale_call_result_279609 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), scale_279601, *[result_div_279604, result_div_279607], **kwargs_279608)
        
        # Obtaining the member 'translate' of a type (line 127)
        translate_279610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), scale_call_result_279609, 'translate')
        # Calling translate(args, kwargs) (line 127)
        translate_call_result_279614 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), translate_279610, *[float_279611, float_279612], **kwargs_279613)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', translate_call_result_279614)
        
        # ################# End of '_get_affine_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_affine_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_279615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_affine_transform'
        return stypy_return_type_279615


    @norecursion
    def get_xaxis_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_279616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 39), 'unicode', u'grid')
        defaults = [unicode_279616]
        # Create a new context for function 'get_xaxis_transform'
        module_type_store = module_type_store.open_function_context('get_xaxis_transform', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_function_name', 'GeoAxes.get_xaxis_transform')
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_param_names_list', ['which'])
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.get_xaxis_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.get_xaxis_transform', ['which'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_xaxis_transform', localization, ['which'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_xaxis_transform(...)' code ##################

        
        
        # Getting the type of 'which' (line 132)
        which_279617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'which')
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_279618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        unicode_279619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'unicode', u'tick1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 24), list_279618, unicode_279619)
        # Adding element type (line 132)
        unicode_279620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 33), 'unicode', u'tick2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 24), list_279618, unicode_279620)
        # Adding element type (line 132)
        unicode_279621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 41), 'unicode', u'grid')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 24), list_279618, unicode_279621)
        
        # Applying the binary operator 'notin' (line 132)
        result_contains_279622 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 11), 'notin', which_279617, list_279618)
        
        # Testing the type of an if condition (line 132)
        if_condition_279623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), result_contains_279622)
        # Assigning a type to the variable 'if_condition_279623' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'if_condition_279623', if_condition_279623)
        # SSA begins for if statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 133):
        
        # Assigning a Str to a Name (line 133):
        unicode_279624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'unicode', u"'which' must be on of [ 'tick1' | 'tick2' | 'grid' ]")
        # Assigning a type to the variable 'msg' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'msg', unicode_279624)
        
        # Call to ValueError(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'msg' (line 134)
        msg_279626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'msg', False)
        # Processing the call keyword arguments (line 134)
        kwargs_279627 = {}
        # Getting the type of 'ValueError' (line 134)
        ValueError_279625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 134)
        ValueError_call_result_279628 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), ValueError_279625, *[msg_279626], **kwargs_279627)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 134, 12), ValueError_call_result_279628, 'raise parameter', BaseException)
        # SSA join for if statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 135)
        self_279629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'self')
        # Obtaining the member '_xaxis_transform' of a type (line 135)
        _xaxis_transform_279630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), self_279629, '_xaxis_transform')
        # Assigning a type to the variable 'stypy_return_type' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', _xaxis_transform_279630)
        
        # ################# End of 'get_xaxis_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_xaxis_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_279631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_xaxis_transform'
        return stypy_return_type_279631


    @norecursion
    def get_xaxis_text1_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_xaxis_text1_transform'
        module_type_store = module_type_store.open_function_context('get_xaxis_text1_transform', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_function_name', 'GeoAxes.get_xaxis_text1_transform')
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_param_names_list', ['pad'])
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.get_xaxis_text1_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.get_xaxis_text1_transform', ['pad'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_xaxis_text1_transform', localization, ['pad'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_xaxis_text1_transform(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 138)
        tuple_279632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 138)
        # Adding element type (line 138)
        # Getting the type of 'self' (line 138)
        self_279633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'self')
        # Obtaining the member '_xaxis_text1_transform' of a type (line 138)
        _xaxis_text1_transform_279634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), self_279633, '_xaxis_text1_transform')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 15), tuple_279632, _xaxis_text1_transform_279634)
        # Adding element type (line 138)
        unicode_279635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 44), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 15), tuple_279632, unicode_279635)
        # Adding element type (line 138)
        unicode_279636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 54), 'unicode', u'center')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 15), tuple_279632, unicode_279636)
        
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', tuple_279632)
        
        # ################# End of 'get_xaxis_text1_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_xaxis_text1_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_279637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279637)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_xaxis_text1_transform'
        return stypy_return_type_279637


    @norecursion
    def get_xaxis_text2_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_xaxis_text2_transform'
        module_type_store = module_type_store.open_function_context('get_xaxis_text2_transform', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_function_name', 'GeoAxes.get_xaxis_text2_transform')
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_param_names_list', ['pad'])
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.get_xaxis_text2_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.get_xaxis_text2_transform', ['pad'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_xaxis_text2_transform', localization, ['pad'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_xaxis_text2_transform(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 141)
        tuple_279638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 141)
        # Adding element type (line 141)
        # Getting the type of 'self' (line 141)
        self_279639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'self')
        # Obtaining the member '_xaxis_text2_transform' of a type (line 141)
        _xaxis_text2_transform_279640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), self_279639, '_xaxis_text2_transform')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 15), tuple_279638, _xaxis_text2_transform_279640)
        # Adding element type (line 141)
        unicode_279641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 44), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 15), tuple_279638, unicode_279641)
        # Adding element type (line 141)
        unicode_279642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 51), 'unicode', u'center')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 15), tuple_279638, unicode_279642)
        
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', tuple_279638)
        
        # ################# End of 'get_xaxis_text2_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_xaxis_text2_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_279643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279643)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_xaxis_text2_transform'
        return stypy_return_type_279643


    @norecursion
    def get_yaxis_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_279644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 39), 'unicode', u'grid')
        defaults = [unicode_279644]
        # Create a new context for function 'get_yaxis_transform'
        module_type_store = module_type_store.open_function_context('get_yaxis_transform', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_function_name', 'GeoAxes.get_yaxis_transform')
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_param_names_list', ['which'])
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.get_yaxis_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.get_yaxis_transform', ['which'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_yaxis_transform', localization, ['which'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_yaxis_transform(...)' code ##################

        
        
        # Getting the type of 'which' (line 144)
        which_279645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'which')
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_279646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        unicode_279647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 25), 'unicode', u'tick1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 24), list_279646, unicode_279647)
        # Adding element type (line 144)
        unicode_279648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 33), 'unicode', u'tick2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 24), list_279646, unicode_279648)
        # Adding element type (line 144)
        unicode_279649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 41), 'unicode', u'grid')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 24), list_279646, unicode_279649)
        
        # Applying the binary operator 'notin' (line 144)
        result_contains_279650 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 11), 'notin', which_279645, list_279646)
        
        # Testing the type of an if condition (line 144)
        if_condition_279651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), result_contains_279650)
        # Assigning a type to the variable 'if_condition_279651' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_279651', if_condition_279651)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 145):
        
        # Assigning a Str to a Name (line 145):
        unicode_279652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'unicode', u"'which' must be one of [ 'tick1' | 'tick2' | 'grid' ]")
        # Assigning a type to the variable 'msg' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'msg', unicode_279652)
        
        # Call to ValueError(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'msg' (line 146)
        msg_279654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'msg', False)
        # Processing the call keyword arguments (line 146)
        kwargs_279655 = {}
        # Getting the type of 'ValueError' (line 146)
        ValueError_279653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 146)
        ValueError_call_result_279656 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), ValueError_279653, *[msg_279654], **kwargs_279655)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 12), ValueError_call_result_279656, 'raise parameter', BaseException)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 147)
        self_279657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'self')
        # Obtaining the member '_yaxis_transform' of a type (line 147)
        _yaxis_transform_279658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 15), self_279657, '_yaxis_transform')
        # Assigning a type to the variable 'stypy_return_type' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stypy_return_type', _yaxis_transform_279658)
        
        # ################# End of 'get_yaxis_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_yaxis_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_279659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279659)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_yaxis_transform'
        return stypy_return_type_279659


    @norecursion
    def get_yaxis_text1_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_yaxis_text1_transform'
        module_type_store = module_type_store.open_function_context('get_yaxis_text1_transform', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_function_name', 'GeoAxes.get_yaxis_text1_transform')
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_param_names_list', ['pad'])
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.get_yaxis_text1_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.get_yaxis_text1_transform', ['pad'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_yaxis_text1_transform', localization, ['pad'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_yaxis_text1_transform(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_279660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        # Getting the type of 'self' (line 150)
        self_279661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'self')
        # Obtaining the member '_yaxis_text1_transform' of a type (line 150)
        _yaxis_text1_transform_279662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 15), self_279661, '_yaxis_text1_transform')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 15), tuple_279660, _yaxis_text1_transform_279662)
        # Adding element type (line 150)
        unicode_279663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 44), 'unicode', u'center')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 15), tuple_279660, unicode_279663)
        # Adding element type (line 150)
        unicode_279664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 54), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 15), tuple_279660, unicode_279664)
        
        # Assigning a type to the variable 'stypy_return_type' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'stypy_return_type', tuple_279660)
        
        # ################# End of 'get_yaxis_text1_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_yaxis_text1_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_279665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279665)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_yaxis_text1_transform'
        return stypy_return_type_279665


    @norecursion
    def get_yaxis_text2_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_yaxis_text2_transform'
        module_type_store = module_type_store.open_function_context('get_yaxis_text2_transform', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_function_name', 'GeoAxes.get_yaxis_text2_transform')
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_param_names_list', ['pad'])
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.get_yaxis_text2_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.get_yaxis_text2_transform', ['pad'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_yaxis_text2_transform', localization, ['pad'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_yaxis_text2_transform(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_279666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        # Getting the type of 'self' (line 153)
        self_279667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'self')
        # Obtaining the member '_yaxis_text2_transform' of a type (line 153)
        _yaxis_text2_transform_279668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 15), self_279667, '_yaxis_text2_transform')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 15), tuple_279666, _yaxis_text2_transform_279668)
        # Adding element type (line 153)
        unicode_279669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 44), 'unicode', u'center')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 15), tuple_279666, unicode_279669)
        # Adding element type (line 153)
        unicode_279670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 54), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 15), tuple_279666, unicode_279670)
        
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', tuple_279666)
        
        # ################# End of 'get_yaxis_text2_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_yaxis_text2_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_279671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_yaxis_text2_transform'
        return stypy_return_type_279671


    @norecursion
    def _gen_axes_patch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_gen_axes_patch'
        module_type_store = module_type_store.open_function_context('_gen_axes_patch', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_function_name', 'GeoAxes._gen_axes_patch')
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes._gen_axes_patch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes._gen_axes_patch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_gen_axes_patch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_gen_axes_patch(...)' code ##################

        
        # Call to Circle(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining an instance of the builtin type 'tuple' (line 156)
        tuple_279673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 156)
        # Adding element type (line 156)
        float_279674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 23), tuple_279673, float_279674)
        # Adding element type (line 156)
        float_279675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 23), tuple_279673, float_279675)
        
        float_279676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 34), 'float')
        # Processing the call keyword arguments (line 156)
        kwargs_279677 = {}
        # Getting the type of 'Circle' (line 156)
        Circle_279672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'Circle', False)
        # Calling Circle(args, kwargs) (line 156)
        Circle_call_result_279678 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), Circle_279672, *[tuple_279673, float_279676], **kwargs_279677)
        
        # Assigning a type to the variable 'stypy_return_type' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', Circle_call_result_279678)
        
        # ################# End of '_gen_axes_patch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_gen_axes_patch' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_279679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_gen_axes_patch'
        return stypy_return_type_279679


    @norecursion
    def _gen_axes_spines(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_gen_axes_spines'
        module_type_store = module_type_store.open_function_context('_gen_axes_spines', 158, 4, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_function_name', 'GeoAxes._gen_axes_spines')
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes._gen_axes_spines.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes._gen_axes_spines', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_gen_axes_spines', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_gen_axes_spines(...)' code ##################

        
        # Obtaining an instance of the builtin type 'dict' (line 159)
        dict_279680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 159)
        # Adding element type (key, value) (line 159)
        unicode_279681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 16), 'unicode', u'geo')
        
        # Call to circular_spine(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'self' (line 159)
        self_279685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'self', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_279686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        float_279687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 52), tuple_279686, float_279687)
        # Adding element type (line 160)
        float_279688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 52), tuple_279686, float_279688)
        
        float_279689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 63), 'float')
        # Processing the call keyword arguments (line 159)
        kwargs_279690 = {}
        # Getting the type of 'mspines' (line 159)
        mspines_279682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'mspines', False)
        # Obtaining the member 'Spine' of a type (line 159)
        Spine_279683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), mspines_279682, 'Spine')
        # Obtaining the member 'circular_spine' of a type (line 159)
        circular_spine_279684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), Spine_279683, 'circular_spine')
        # Calling circular_spine(args, kwargs) (line 159)
        circular_spine_call_result_279691 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), circular_spine_279684, *[self_279685, tuple_279686, float_279689], **kwargs_279690)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 15), dict_279680, (unicode_279681, circular_spine_call_result_279691))
        
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', dict_279680)
        
        # ################# End of '_gen_axes_spines(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_gen_axes_spines' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_279692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_gen_axes_spines'
        return stypy_return_type_279692


    @norecursion
    def set_yscale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_yscale'
        module_type_store = module_type_store.open_function_context('set_yscale', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_function_name', 'GeoAxes.set_yscale')
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.set_yscale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.set_yscale', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_yscale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_yscale(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        int_279693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 16), 'int')
        # Getting the type of 'args' (line 163)
        args_279694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'args')
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___279695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 11), args_279694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_279696 = invoke(stypy.reporting.localization.Localization(__file__, 163, 11), getitem___279695, int_279693)
        
        unicode_279697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 22), 'unicode', u'linear')
        # Applying the binary operator '!=' (line 163)
        result_ne_279698 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), '!=', subscript_call_result_279696, unicode_279697)
        
        # Testing the type of an if condition (line 163)
        if_condition_279699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), result_ne_279698)
        # Assigning a type to the variable 'if_condition_279699' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_279699', if_condition_279699)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplementedError' (line 164)
        NotImplementedError_279700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 164, 12), NotImplementedError_279700, 'raise parameter', BaseException)
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_yscale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_yscale' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_279701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279701)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_yscale'
        return stypy_return_type_279701

    
    # Assigning a Name to a Name (line 166):

    @norecursion
    def set_xlim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_xlim'
        module_type_store = module_type_store.open_function_context('set_xlim', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_function_name', 'GeoAxes.set_xlim')
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.set_xlim.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.set_xlim', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_xlim', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_xlim(...)' code ##################

        
        # Call to TypeError(...): (line 169)
        # Processing the call arguments (line 169)
        unicode_279703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'unicode', u'It is not possible to change axes limits for geographic projections. Please consider using Basemap or Cartopy.')
        # Processing the call keyword arguments (line 169)
        kwargs_279704 = {}
        # Getting the type of 'TypeError' (line 169)
        TypeError_279702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 14), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 169)
        TypeError_call_result_279705 = invoke(stypy.reporting.localization.Localization(__file__, 169, 14), TypeError_279702, *[unicode_279703], **kwargs_279704)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 169, 8), TypeError_call_result_279705, 'raise parameter', BaseException)
        
        # ################# End of 'set_xlim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_xlim' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_279706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279706)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_xlim'
        return stypy_return_type_279706

    
    # Assigning a Name to a Name (line 173):

    @norecursion
    def format_coord(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format_coord'
        module_type_store = module_type_store.open_function_context('format_coord', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.format_coord.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.format_coord.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.format_coord.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.format_coord.__dict__.__setitem__('stypy_function_name', 'GeoAxes.format_coord')
        GeoAxes.format_coord.__dict__.__setitem__('stypy_param_names_list', ['lon', 'lat'])
        GeoAxes.format_coord.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.format_coord.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.format_coord.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.format_coord.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.format_coord.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.format_coord.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.format_coord', ['lon', 'lat'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format_coord', localization, ['lon', 'lat'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format_coord(...)' code ##################

        unicode_279707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'unicode', u'return a format string formatting the coordinate')
        
        # Assigning a Call to a Tuple (line 177):
        
        # Assigning a Call to a Name:
        
        # Call to rad2deg(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_279710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        # Getting the type of 'lon' (line 177)
        lon_279711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'lon', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 30), list_279710, lon_279711)
        # Adding element type (line 177)
        # Getting the type of 'lat' (line 177)
        lat_279712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 36), 'lat', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 30), list_279710, lat_279712)
        
        # Processing the call keyword arguments (line 177)
        kwargs_279713 = {}
        # Getting the type of 'np' (line 177)
        np_279708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'np', False)
        # Obtaining the member 'rad2deg' of a type (line 177)
        rad2deg_279709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 19), np_279708, 'rad2deg')
        # Calling rad2deg(args, kwargs) (line 177)
        rad2deg_call_result_279714 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), rad2deg_279709, *[list_279710], **kwargs_279713)
        
        # Assigning a type to the variable 'call_assignment_279206' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_279206', rad2deg_call_result_279714)
        
        # Assigning a Call to a Name (line 177):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_279717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 8), 'int')
        # Processing the call keyword arguments
        kwargs_279718 = {}
        # Getting the type of 'call_assignment_279206' (line 177)
        call_assignment_279206_279715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_279206', False)
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___279716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), call_assignment_279206_279715, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_279719 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___279716, *[int_279717], **kwargs_279718)
        
        # Assigning a type to the variable 'call_assignment_279207' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_279207', getitem___call_result_279719)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'call_assignment_279207' (line 177)
        call_assignment_279207_279720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_279207')
        # Assigning a type to the variable 'lon' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'lon', call_assignment_279207_279720)
        
        # Assigning a Call to a Name (line 177):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_279723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 8), 'int')
        # Processing the call keyword arguments
        kwargs_279724 = {}
        # Getting the type of 'call_assignment_279206' (line 177)
        call_assignment_279206_279721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_279206', False)
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___279722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), call_assignment_279206_279721, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_279725 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___279722, *[int_279723], **kwargs_279724)
        
        # Assigning a type to the variable 'call_assignment_279208' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_279208', getitem___call_result_279725)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'call_assignment_279208' (line 177)
        call_assignment_279208_279726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_279208')
        # Assigning a type to the variable 'lat' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'lat', call_assignment_279208_279726)
        
        
        # Getting the type of 'lat' (line 178)
        lat_279727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'lat')
        float_279728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 18), 'float')
        # Applying the binary operator '>=' (line 178)
        result_ge_279729 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 11), '>=', lat_279727, float_279728)
        
        # Testing the type of an if condition (line 178)
        if_condition_279730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 8), result_ge_279729)
        # Assigning a type to the variable 'if_condition_279730' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'if_condition_279730', if_condition_279730)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 179):
        
        # Assigning a Str to a Name (line 179):
        unicode_279731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 17), 'unicode', u'N')
        # Assigning a type to the variable 'ns' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'ns', unicode_279731)
        # SSA branch for the else part of an if statement (line 178)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 181):
        
        # Assigning a Str to a Name (line 181):
        unicode_279732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 17), 'unicode', u'S')
        # Assigning a type to the variable 'ns' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'ns', unicode_279732)
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'lon' (line 182)
        lon_279733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'lon')
        float_279734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 18), 'float')
        # Applying the binary operator '>=' (line 182)
        result_ge_279735 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 11), '>=', lon_279733, float_279734)
        
        # Testing the type of an if condition (line 182)
        if_condition_279736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), result_ge_279735)
        # Assigning a type to the variable 'if_condition_279736' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_279736', if_condition_279736)
        # SSA begins for if statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 183):
        
        # Assigning a Str to a Name (line 183):
        unicode_279737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 17), 'unicode', u'E')
        # Assigning a type to the variable 'ew' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'ew', unicode_279737)
        # SSA branch for the else part of an if statement (line 182)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 185):
        
        # Assigning a Str to a Name (line 185):
        unicode_279738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 17), 'unicode', u'W')
        # Assigning a type to the variable 'ew' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'ew', unicode_279738)
        # SSA join for if statement (line 182)
        module_type_store = module_type_store.join_ssa_context()
        
        unicode_279739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 16), 'unicode', u'%f\xb0%s, %f\xb0%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_279740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        
        # Call to abs(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'lat' (line 187)
        lat_279742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'lat', False)
        # Processing the call keyword arguments (line 187)
        kwargs_279743 = {}
        # Getting the type of 'abs' (line 187)
        abs_279741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'abs', False)
        # Calling abs(args, kwargs) (line 187)
        abs_call_result_279744 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), abs_279741, *[lat_279742], **kwargs_279743)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 19), tuple_279740, abs_call_result_279744)
        # Adding element type (line 187)
        # Getting the type of 'ns' (line 187)
        ns_279745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 29), 'ns')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 19), tuple_279740, ns_279745)
        # Adding element type (line 187)
        
        # Call to abs(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'lon' (line 187)
        lon_279747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 37), 'lon', False)
        # Processing the call keyword arguments (line 187)
        kwargs_279748 = {}
        # Getting the type of 'abs' (line 187)
        abs_279746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'abs', False)
        # Calling abs(args, kwargs) (line 187)
        abs_call_result_279749 = invoke(stypy.reporting.localization.Localization(__file__, 187, 33), abs_279746, *[lon_279747], **kwargs_279748)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 19), tuple_279740, abs_call_result_279749)
        # Adding element type (line 187)
        # Getting the type of 'ew' (line 187)
        ew_279750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 43), 'ew')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 19), tuple_279740, ew_279750)
        
        # Applying the binary operator '%' (line 186)
        result_mod_279751 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 16), '%', unicode_279739, tuple_279740)
        
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type', result_mod_279751)
        
        # ################# End of 'format_coord(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_coord' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_279752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279752)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_coord'
        return stypy_return_type_279752


    @norecursion
    def set_longitude_grid(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_longitude_grid'
        module_type_store = module_type_store.open_function_context('set_longitude_grid', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_function_name', 'GeoAxes.set_longitude_grid')
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_param_names_list', ['degrees'])
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.set_longitude_grid.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.set_longitude_grid', ['degrees'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_longitude_grid', localization, ['degrees'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_longitude_grid(...)' code ##################

        unicode_279753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, (-1)), 'unicode', u'\n        Set the number of degrees between each longitude grid.\n        ')
        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to arange(...): (line 194)
        # Processing the call arguments (line 194)
        int_279756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 25), 'int')
        # Getting the type of 'degrees' (line 194)
        degrees_279757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 32), 'degrees', False)
        # Applying the binary operator '+' (line 194)
        result_add_279758 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 25), '+', int_279756, degrees_279757)
        
        int_279759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 41), 'int')
        # Getting the type of 'degrees' (line 194)
        degrees_279760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 46), 'degrees', False)
        # Processing the call keyword arguments (line 194)
        kwargs_279761 = {}
        # Getting the type of 'np' (line 194)
        np_279754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'np', False)
        # Obtaining the member 'arange' of a type (line 194)
        arange_279755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), np_279754, 'arange')
        # Calling arange(args, kwargs) (line 194)
        arange_call_result_279762 = invoke(stypy.reporting.localization.Localization(__file__, 194, 15), arange_279755, *[result_add_279758, int_279759, degrees_279760], **kwargs_279761)
        
        # Assigning a type to the variable 'grid' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'grid', arange_call_result_279762)
        
        # Call to set_major_locator(...): (line 195)
        # Processing the call arguments (line 195)
        
        # Call to FixedLocator(...): (line 195)
        # Processing the call arguments (line 195)
        
        # Call to deg2rad(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'grid' (line 195)
        grid_279769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 61), 'grid', False)
        # Processing the call keyword arguments (line 195)
        kwargs_279770 = {}
        # Getting the type of 'np' (line 195)
        np_279767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 50), 'np', False)
        # Obtaining the member 'deg2rad' of a type (line 195)
        deg2rad_279768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 50), np_279767, 'deg2rad')
        # Calling deg2rad(args, kwargs) (line 195)
        deg2rad_call_result_279771 = invoke(stypy.reporting.localization.Localization(__file__, 195, 50), deg2rad_279768, *[grid_279769], **kwargs_279770)
        
        # Processing the call keyword arguments (line 195)
        kwargs_279772 = {}
        # Getting the type of 'FixedLocator' (line 195)
        FixedLocator_279766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 37), 'FixedLocator', False)
        # Calling FixedLocator(args, kwargs) (line 195)
        FixedLocator_call_result_279773 = invoke(stypy.reporting.localization.Localization(__file__, 195, 37), FixedLocator_279766, *[deg2rad_call_result_279771], **kwargs_279772)
        
        # Processing the call keyword arguments (line 195)
        kwargs_279774 = {}
        # Getting the type of 'self' (line 195)
        self_279763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self', False)
        # Obtaining the member 'xaxis' of a type (line 195)
        xaxis_279764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_279763, 'xaxis')
        # Obtaining the member 'set_major_locator' of a type (line 195)
        set_major_locator_279765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), xaxis_279764, 'set_major_locator')
        # Calling set_major_locator(args, kwargs) (line 195)
        set_major_locator_call_result_279775 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), set_major_locator_279765, *[FixedLocator_call_result_279773], **kwargs_279774)
        
        
        # Call to set_major_formatter(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to ThetaFormatter(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'degrees' (line 196)
        degrees_279781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 59), 'degrees', False)
        # Processing the call keyword arguments (line 196)
        kwargs_279782 = {}
        # Getting the type of 'self' (line 196)
        self_279779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 'self', False)
        # Obtaining the member 'ThetaFormatter' of a type (line 196)
        ThetaFormatter_279780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 39), self_279779, 'ThetaFormatter')
        # Calling ThetaFormatter(args, kwargs) (line 196)
        ThetaFormatter_call_result_279783 = invoke(stypy.reporting.localization.Localization(__file__, 196, 39), ThetaFormatter_279780, *[degrees_279781], **kwargs_279782)
        
        # Processing the call keyword arguments (line 196)
        kwargs_279784 = {}
        # Getting the type of 'self' (line 196)
        self_279776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self', False)
        # Obtaining the member 'xaxis' of a type (line 196)
        xaxis_279777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_279776, 'xaxis')
        # Obtaining the member 'set_major_formatter' of a type (line 196)
        set_major_formatter_279778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), xaxis_279777, 'set_major_formatter')
        # Calling set_major_formatter(args, kwargs) (line 196)
        set_major_formatter_call_result_279785 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), set_major_formatter_279778, *[ThetaFormatter_call_result_279783], **kwargs_279784)
        
        
        # ################# End of 'set_longitude_grid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_longitude_grid' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_279786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279786)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_longitude_grid'
        return stypy_return_type_279786


    @norecursion
    def set_latitude_grid(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_latitude_grid'
        module_type_store = module_type_store.open_function_context('set_latitude_grid', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_function_name', 'GeoAxes.set_latitude_grid')
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_param_names_list', ['degrees'])
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.set_latitude_grid.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.set_latitude_grid', ['degrees'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_latitude_grid', localization, ['degrees'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_latitude_grid(...)' code ##################

        unicode_279787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'unicode', u'\n        Set the number of degrees between each latitude grid.\n        ')
        
        # Assigning a Call to a Name (line 203):
        
        # Assigning a Call to a Name (line 203):
        
        # Call to arange(...): (line 203)
        # Processing the call arguments (line 203)
        int_279790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 25), 'int')
        # Getting the type of 'degrees' (line 203)
        degrees_279791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 31), 'degrees', False)
        # Applying the binary operator '+' (line 203)
        result_add_279792 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 25), '+', int_279790, degrees_279791)
        
        int_279793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 40), 'int')
        # Getting the type of 'degrees' (line 203)
        degrees_279794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 44), 'degrees', False)
        # Processing the call keyword arguments (line 203)
        kwargs_279795 = {}
        # Getting the type of 'np' (line 203)
        np_279788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'np', False)
        # Obtaining the member 'arange' of a type (line 203)
        arange_279789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), np_279788, 'arange')
        # Calling arange(args, kwargs) (line 203)
        arange_call_result_279796 = invoke(stypy.reporting.localization.Localization(__file__, 203, 15), arange_279789, *[result_add_279792, int_279793, degrees_279794], **kwargs_279795)
        
        # Assigning a type to the variable 'grid' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'grid', arange_call_result_279796)
        
        # Call to set_major_locator(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Call to FixedLocator(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Call to deg2rad(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'grid' (line 204)
        grid_279803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 61), 'grid', False)
        # Processing the call keyword arguments (line 204)
        kwargs_279804 = {}
        # Getting the type of 'np' (line 204)
        np_279801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 50), 'np', False)
        # Obtaining the member 'deg2rad' of a type (line 204)
        deg2rad_279802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 50), np_279801, 'deg2rad')
        # Calling deg2rad(args, kwargs) (line 204)
        deg2rad_call_result_279805 = invoke(stypy.reporting.localization.Localization(__file__, 204, 50), deg2rad_279802, *[grid_279803], **kwargs_279804)
        
        # Processing the call keyword arguments (line 204)
        kwargs_279806 = {}
        # Getting the type of 'FixedLocator' (line 204)
        FixedLocator_279800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 37), 'FixedLocator', False)
        # Calling FixedLocator(args, kwargs) (line 204)
        FixedLocator_call_result_279807 = invoke(stypy.reporting.localization.Localization(__file__, 204, 37), FixedLocator_279800, *[deg2rad_call_result_279805], **kwargs_279806)
        
        # Processing the call keyword arguments (line 204)
        kwargs_279808 = {}
        # Getting the type of 'self' (line 204)
        self_279797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self', False)
        # Obtaining the member 'yaxis' of a type (line 204)
        yaxis_279798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_279797, 'yaxis')
        # Obtaining the member 'set_major_locator' of a type (line 204)
        set_major_locator_279799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), yaxis_279798, 'set_major_locator')
        # Calling set_major_locator(args, kwargs) (line 204)
        set_major_locator_call_result_279809 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), set_major_locator_279799, *[FixedLocator_call_result_279807], **kwargs_279808)
        
        
        # Call to set_major_formatter(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Call to ThetaFormatter(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'degrees' (line 205)
        degrees_279815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 59), 'degrees', False)
        # Processing the call keyword arguments (line 205)
        kwargs_279816 = {}
        # Getting the type of 'self' (line 205)
        self_279813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 39), 'self', False)
        # Obtaining the member 'ThetaFormatter' of a type (line 205)
        ThetaFormatter_279814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 39), self_279813, 'ThetaFormatter')
        # Calling ThetaFormatter(args, kwargs) (line 205)
        ThetaFormatter_call_result_279817 = invoke(stypy.reporting.localization.Localization(__file__, 205, 39), ThetaFormatter_279814, *[degrees_279815], **kwargs_279816)
        
        # Processing the call keyword arguments (line 205)
        kwargs_279818 = {}
        # Getting the type of 'self' (line 205)
        self_279810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self', False)
        # Obtaining the member 'yaxis' of a type (line 205)
        yaxis_279811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_279810, 'yaxis')
        # Obtaining the member 'set_major_formatter' of a type (line 205)
        set_major_formatter_279812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), yaxis_279811, 'set_major_formatter')
        # Calling set_major_formatter(args, kwargs) (line 205)
        set_major_formatter_call_result_279819 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), set_major_formatter_279812, *[ThetaFormatter_call_result_279817], **kwargs_279818)
        
        
        # ################# End of 'set_latitude_grid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_latitude_grid' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_279820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279820)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_latitude_grid'
        return stypy_return_type_279820


    @norecursion
    def set_longitude_grid_ends(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_longitude_grid_ends'
        module_type_store = module_type_store.open_function_context('set_longitude_grid_ends', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_function_name', 'GeoAxes.set_longitude_grid_ends')
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_param_names_list', ['degrees'])
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.set_longitude_grid_ends.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.set_longitude_grid_ends', ['degrees'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_longitude_grid_ends', localization, ['degrees'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_longitude_grid_ends(...)' code ##################

        unicode_279821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'unicode', u'\n        Set the latitude(s) at which to stop drawing the longitude grids.\n        ')
        
        # Assigning a BinOp to a Attribute (line 211):
        
        # Assigning a BinOp to a Attribute (line 211):
        # Getting the type of 'degrees' (line 211)
        degrees_279822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'degrees')
        # Getting the type of 'np' (line 211)
        np_279823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'np')
        # Obtaining the member 'pi' of a type (line 211)
        pi_279824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 41), np_279823, 'pi')
        float_279825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 49), 'float')
        # Applying the binary operator 'div' (line 211)
        result_div_279826 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 41), 'div', pi_279824, float_279825)
        
        # Applying the binary operator '*' (line 211)
        result_mul_279827 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 30), '*', degrees_279822, result_div_279826)
        
        # Getting the type of 'self' (line 211)
        self_279828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self')
        # Setting the type of the member '_longitude_cap' of a type (line 211)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_279828, '_longitude_cap', result_mul_279827)
        
        # Call to translate(...): (line 212)
        # Processing the call arguments (line 212)
        float_279843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 23), 'float')
        
        # Getting the type of 'self' (line 215)
        self_279844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 29), 'self', False)
        # Obtaining the member '_longitude_cap' of a type (line 215)
        _longitude_cap_279845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 29), self_279844, '_longitude_cap')
        # Applying the 'usub' unary operator (line 215)
        result___neg___279846 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 28), 'usub', _longitude_cap_279845)
        
        # Processing the call keyword arguments (line 212)
        kwargs_279847 = {}
        
        # Call to scale(...): (line 212)
        # Processing the call arguments (line 212)
        float_279835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 19), 'float')
        # Getting the type of 'self' (line 214)
        self_279836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'self', False)
        # Obtaining the member '_longitude_cap' of a type (line 214)
        _longitude_cap_279837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 24), self_279836, '_longitude_cap')
        float_279838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 46), 'float')
        # Applying the binary operator '*' (line 214)
        result_mul_279839 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 24), '*', _longitude_cap_279837, float_279838)
        
        # Processing the call keyword arguments (line 212)
        kwargs_279840 = {}
        
        # Call to clear(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_279832 = {}
        # Getting the type of 'self' (line 212)
        self_279829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self', False)
        # Obtaining the member '_xaxis_pretransform' of a type (line 212)
        _xaxis_pretransform_279830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_279829, '_xaxis_pretransform')
        # Obtaining the member 'clear' of a type (line 212)
        clear_279831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), _xaxis_pretransform_279830, 'clear')
        # Calling clear(args, kwargs) (line 212)
        clear_call_result_279833 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), clear_279831, *[], **kwargs_279832)
        
        # Obtaining the member 'scale' of a type (line 212)
        scale_279834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), clear_call_result_279833, 'scale')
        # Calling scale(args, kwargs) (line 212)
        scale_call_result_279841 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), scale_279834, *[float_279835, result_mul_279839], **kwargs_279840)
        
        # Obtaining the member 'translate' of a type (line 212)
        translate_279842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), scale_call_result_279841, 'translate')
        # Calling translate(args, kwargs) (line 212)
        translate_call_result_279848 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), translate_279842, *[float_279843, result___neg___279846], **kwargs_279847)
        
        
        # ################# End of 'set_longitude_grid_ends(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_longitude_grid_ends' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_279849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_longitude_grid_ends'
        return stypy_return_type_279849


    @norecursion
    def get_data_ratio(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_data_ratio'
        module_type_store = module_type_store.open_function_context('get_data_ratio', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_function_name', 'GeoAxes.get_data_ratio')
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.get_data_ratio.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.get_data_ratio', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_data_ratio', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_data_ratio(...)' code ##################

        unicode_279850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'unicode', u'\n        Return the aspect ratio of the data itself.\n        ')
        float_279851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 15), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', float_279851)
        
        # ################# End of 'get_data_ratio(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_data_ratio' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_279852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279852)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_data_ratio'
        return stypy_return_type_279852


    @norecursion
    def can_zoom(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'can_zoom'
        module_type_store = module_type_store.open_function_context('can_zoom', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_function_name', 'GeoAxes.can_zoom')
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.can_zoom.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.can_zoom', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'can_zoom', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'can_zoom(...)' code ##################

        unicode_279853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'unicode', u'\n        Return *True* if this axes supports the zoom box button functionality.\n\n        This axes object does not support interactive zoom box.\n        ')
        # Getting the type of 'False' (line 231)
        False_279854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', False_279854)
        
        # ################# End of 'can_zoom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_zoom' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_279855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279855)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_zoom'
        return stypy_return_type_279855


    @norecursion
    def can_pan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'can_pan'
        module_type_store = module_type_store.open_function_context('can_pan', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.can_pan.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.can_pan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.can_pan.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.can_pan.__dict__.__setitem__('stypy_function_name', 'GeoAxes.can_pan')
        GeoAxes.can_pan.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes.can_pan.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.can_pan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.can_pan.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.can_pan.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.can_pan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.can_pan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.can_pan', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'can_pan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'can_pan(...)' code ##################

        unicode_279856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'unicode', u'\n        Return *True* if this axes supports the pan/zoom button functionality.\n\n        This axes object does not support interactive pan/zoom.\n        ')
        # Getting the type of 'False' (line 239)
        False_279857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'stypy_return_type', False_279857)
        
        # ################# End of 'can_pan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_pan' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_279858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279858)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_pan'
        return stypy_return_type_279858


    @norecursion
    def start_pan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'start_pan'
        module_type_store = module_type_store.open_function_context('start_pan', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.start_pan.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.start_pan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.start_pan.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.start_pan.__dict__.__setitem__('stypy_function_name', 'GeoAxes.start_pan')
        GeoAxes.start_pan.__dict__.__setitem__('stypy_param_names_list', ['x', 'y', 'button'])
        GeoAxes.start_pan.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.start_pan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.start_pan.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.start_pan.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.start_pan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.start_pan.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.start_pan', ['x', 'y', 'button'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'start_pan', localization, ['x', 'y', 'button'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'start_pan(...)' code ##################

        pass
        
        # ################# End of 'start_pan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'start_pan' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_279859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279859)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'start_pan'
        return stypy_return_type_279859


    @norecursion
    def end_pan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'end_pan'
        module_type_store = module_type_store.open_function_context('end_pan', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.end_pan.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.end_pan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.end_pan.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.end_pan.__dict__.__setitem__('stypy_function_name', 'GeoAxes.end_pan')
        GeoAxes.end_pan.__dict__.__setitem__('stypy_param_names_list', [])
        GeoAxes.end_pan.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.end_pan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.end_pan.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.end_pan.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.end_pan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.end_pan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.end_pan', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'end_pan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'end_pan(...)' code ##################

        pass
        
        # ################# End of 'end_pan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'end_pan' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_279860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'end_pan'
        return stypy_return_type_279860


    @norecursion
    def drag_pan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'drag_pan'
        module_type_store = module_type_store.open_function_context('drag_pan', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_localization', localization)
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_type_store', module_type_store)
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_function_name', 'GeoAxes.drag_pan')
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_param_names_list', ['button', 'key', 'x', 'y'])
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_varargs_param_name', None)
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_call_defaults', defaults)
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_call_varargs', varargs)
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GeoAxes.drag_pan.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.drag_pan', ['button', 'key', 'x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'drag_pan', localization, ['button', 'key', 'x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'drag_pan(...)' code ##################

        pass
        
        # ################# End of 'drag_pan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'drag_pan' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_279861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279861)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'drag_pan'
        return stypy_return_type_279861


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 0, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GeoAxes.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'GeoAxes' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'GeoAxes', GeoAxes)

# Assigning a Num to a Name (line 43):
int_279862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'int')
# Getting the type of 'GeoAxes'
GeoAxes_279863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GeoAxes')
# Setting the type of the member 'RESOLUTION' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GeoAxes_279863, 'RESOLUTION', int_279862)

# Assigning a Name to a Name (line 166):
# Getting the type of 'GeoAxes'
GeoAxes_279864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GeoAxes')
# Obtaining the member 'set_yscale' of a type
set_yscale_279865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GeoAxes_279864, 'set_yscale')
# Getting the type of 'GeoAxes'
GeoAxes_279866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GeoAxes')
# Setting the type of the member 'set_xscale' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GeoAxes_279866, 'set_xscale', set_yscale_279865)

# Assigning a Name to a Name (line 173):
# Getting the type of 'GeoAxes'
GeoAxes_279867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GeoAxes')
# Obtaining the member 'set_xlim' of a type
set_xlim_279868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GeoAxes_279867, 'set_xlim')
# Getting the type of 'GeoAxes'
GeoAxes_279869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GeoAxes')
# Setting the type of the member 'set_ylim' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GeoAxes_279869, 'set_ylim', set_xlim_279868)
# Declaration of the 'AitoffAxes' class
# Getting the type of 'GeoAxes' (line 251)
GeoAxes_279870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 17), 'GeoAxes')

class AitoffAxes(GeoAxes_279870, ):
    
    # Assigning a Str to a Name (line 252):
    # Declaration of the 'AitoffTransform' class
    # Getting the type of 'Transform' (line 254)
    Transform_279871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'Transform')

    class AitoffTransform(Transform_279871, ):
        unicode_279872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'unicode', u'\n        The base Aitoff transform.\n        ')
        
        # Assigning a Num to a Name (line 258):
        
        # Assigning a Num to a Name (line 258):
        int_279873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 21), 'int')
        # Assigning a type to the variable 'input_dims' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'input_dims', int_279873)
        
        # Assigning a Num to a Name (line 259):
        
        # Assigning a Num to a Name (line 259):
        int_279874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 22), 'int')
        # Assigning a type to the variable 'output_dims' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'output_dims', int_279874)
        
        # Assigning a Name to a Name (line 260):
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'False' (line 260)
        False_279875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'False')
        # Assigning a type to the variable 'is_separable' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'is_separable', False_279875)

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 262, 8, False)
            # Assigning a type to the variable 'self' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'AitoffTransform.__init__', ['resolution'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['resolution'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            unicode_279876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'unicode', u'\n            Create a new Aitoff transform.  Resolution is the number of steps\n            to interpolate between each input line segment to approximate its\n            path in curved Aitoff space.\n            ')
            
            # Call to __init__(...): (line 268)
            # Processing the call arguments (line 268)
            # Getting the type of 'self' (line 268)
            self_279879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 31), 'self', False)
            # Processing the call keyword arguments (line 268)
            kwargs_279880 = {}
            # Getting the type of 'Transform' (line 268)
            Transform_279877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'Transform', False)
            # Obtaining the member '__init__' of a type (line 268)
            init___279878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), Transform_279877, '__init__')
            # Calling __init__(args, kwargs) (line 268)
            init___call_result_279881 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), init___279878, *[self_279879], **kwargs_279880)
            
            
            # Assigning a Name to a Attribute (line 269):
            
            # Assigning a Name to a Attribute (line 269):
            # Getting the type of 'resolution' (line 269)
            resolution_279882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 31), 'resolution')
            # Getting the type of 'self' (line 269)
            self_279883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'self')
            # Setting the type of the member '_resolution' of a type (line 269)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), self_279883, '_resolution', resolution_279882)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_non_affine'
            module_type_store = module_type_store.open_function_context('transform_non_affine', 271, 8, False)
            # Assigning a type to the variable 'self' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'AitoffTransform.transform_non_affine')
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['ll'])
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            AitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'AitoffTransform.transform_non_affine', ['ll'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_non_affine', localization, ['ll'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_non_affine(...)' code ##################

            
            # Assigning a Subscript to a Name (line 272):
            
            # Assigning a Subscript to a Name (line 272):
            
            # Obtaining the type of the subscript
            slice_279884 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 272, 24), None, None, None)
            int_279885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 30), 'int')
            int_279886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 32), 'int')
            slice_279887 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 272, 24), int_279885, int_279886, None)
            # Getting the type of 'll' (line 272)
            ll_279888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 24), 'll')
            # Obtaining the member '__getitem__' of a type (line 272)
            getitem___279889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 24), ll_279888, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 272)
            subscript_call_result_279890 = invoke(stypy.reporting.localization.Localization(__file__, 272, 24), getitem___279889, (slice_279884, slice_279887))
            
            # Assigning a type to the variable 'longitude' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'longitude', subscript_call_result_279890)
            
            # Assigning a Subscript to a Name (line 273):
            
            # Assigning a Subscript to a Name (line 273):
            
            # Obtaining the type of the subscript
            slice_279891 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 273, 24), None, None, None)
            int_279892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 30), 'int')
            int_279893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 32), 'int')
            slice_279894 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 273, 24), int_279892, int_279893, None)
            # Getting the type of 'll' (line 273)
            ll_279895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'll')
            # Obtaining the member '__getitem__' of a type (line 273)
            getitem___279896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 24), ll_279895, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 273)
            subscript_call_result_279897 = invoke(stypy.reporting.localization.Localization(__file__, 273, 24), getitem___279896, (slice_279891, slice_279894))
            
            # Assigning a type to the variable 'latitude' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'latitude', subscript_call_result_279897)
            
            # Assigning a BinOp to a Name (line 276):
            
            # Assigning a BinOp to a Name (line 276):
            # Getting the type of 'longitude' (line 276)
            longitude_279898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'longitude')
            float_279899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 36), 'float')
            # Applying the binary operator 'div' (line 276)
            result_div_279900 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 24), 'div', longitude_279898, float_279899)
            
            # Assigning a type to the variable 'half_long' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'half_long', result_div_279900)
            
            # Assigning a Call to a Name (line 277):
            
            # Assigning a Call to a Name (line 277):
            
            # Call to cos(...): (line 277)
            # Processing the call arguments (line 277)
            # Getting the type of 'latitude' (line 277)
            latitude_279903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 34), 'latitude', False)
            # Processing the call keyword arguments (line 277)
            kwargs_279904 = {}
            # Getting the type of 'np' (line 277)
            np_279901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'np', False)
            # Obtaining the member 'cos' of a type (line 277)
            cos_279902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 27), np_279901, 'cos')
            # Calling cos(args, kwargs) (line 277)
            cos_call_result_279905 = invoke(stypy.reporting.localization.Localization(__file__, 277, 27), cos_279902, *[latitude_279903], **kwargs_279904)
            
            # Assigning a type to the variable 'cos_latitude' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'cos_latitude', cos_call_result_279905)
            
            # Assigning a Call to a Name (line 279):
            
            # Assigning a Call to a Name (line 279):
            
            # Call to arccos(...): (line 279)
            # Processing the call arguments (line 279)
            # Getting the type of 'cos_latitude' (line 279)
            cos_latitude_279908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 30), 'cos_latitude', False)
            
            # Call to cos(...): (line 279)
            # Processing the call arguments (line 279)
            # Getting the type of 'half_long' (line 279)
            half_long_279911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 52), 'half_long', False)
            # Processing the call keyword arguments (line 279)
            kwargs_279912 = {}
            # Getting the type of 'np' (line 279)
            np_279909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 45), 'np', False)
            # Obtaining the member 'cos' of a type (line 279)
            cos_279910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 45), np_279909, 'cos')
            # Calling cos(args, kwargs) (line 279)
            cos_call_result_279913 = invoke(stypy.reporting.localization.Localization(__file__, 279, 45), cos_279910, *[half_long_279911], **kwargs_279912)
            
            # Applying the binary operator '*' (line 279)
            result_mul_279914 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 30), '*', cos_latitude_279908, cos_call_result_279913)
            
            # Processing the call keyword arguments (line 279)
            kwargs_279915 = {}
            # Getting the type of 'np' (line 279)
            np_279906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'np', False)
            # Obtaining the member 'arccos' of a type (line 279)
            arccos_279907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), np_279906, 'arccos')
            # Calling arccos(args, kwargs) (line 279)
            arccos_call_result_279916 = invoke(stypy.reporting.localization.Localization(__file__, 279, 20), arccos_279907, *[result_mul_279914], **kwargs_279915)
            
            # Assigning a type to the variable 'alpha' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'alpha', arccos_call_result_279916)
            
            # Assigning a Call to a Name (line 281):
            
            # Assigning a Call to a Name (line 281):
            
            # Call to masked_where(...): (line 281)
            # Processing the call arguments (line 281)
            
            # Getting the type of 'alpha' (line 281)
            alpha_279919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 36), 'alpha', False)
            float_279920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 45), 'float')
            # Applying the binary operator '==' (line 281)
            result_eq_279921 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 36), '==', alpha_279919, float_279920)
            
            # Getting the type of 'alpha' (line 281)
            alpha_279922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 50), 'alpha', False)
            # Processing the call keyword arguments (line 281)
            kwargs_279923 = {}
            # Getting the type of 'ma' (line 281)
            ma_279917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'ma', False)
            # Obtaining the member 'masked_where' of a type (line 281)
            masked_where_279918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 20), ma_279917, 'masked_where')
            # Calling masked_where(args, kwargs) (line 281)
            masked_where_call_result_279924 = invoke(stypy.reporting.localization.Localization(__file__, 281, 20), masked_where_279918, *[result_eq_279921, alpha_279922], **kwargs_279923)
            
            # Assigning a type to the variable 'alpha' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'alpha', masked_where_call_result_279924)
            
            # Assigning a BinOp to a Name (line 285):
            
            # Assigning a BinOp to a Name (line 285):
            
            # Call to sin(...): (line 285)
            # Processing the call arguments (line 285)
            # Getting the type of 'alpha' (line 285)
            alpha_279927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 32), 'alpha', False)
            # Processing the call keyword arguments (line 285)
            kwargs_279928 = {}
            # Getting the type of 'ma' (line 285)
            ma_279925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'ma', False)
            # Obtaining the member 'sin' of a type (line 285)
            sin_279926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 25), ma_279925, 'sin')
            # Calling sin(args, kwargs) (line 285)
            sin_call_result_279929 = invoke(stypy.reporting.localization.Localization(__file__, 285, 25), sin_279926, *[alpha_279927], **kwargs_279928)
            
            # Getting the type of 'alpha' (line 285)
            alpha_279930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 41), 'alpha')
            # Applying the binary operator 'div' (line 285)
            result_div_279931 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 25), 'div', sin_call_result_279929, alpha_279930)
            
            # Assigning a type to the variable 'sinc_alpha' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'sinc_alpha', result_div_279931)
            
            # Assigning a BinOp to a Name (line 287):
            
            # Assigning a BinOp to a Name (line 287):
            # Getting the type of 'cos_latitude' (line 287)
            cos_latitude_279932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 17), 'cos_latitude')
            
            # Call to sin(...): (line 287)
            # Processing the call arguments (line 287)
            # Getting the type of 'half_long' (line 287)
            half_long_279935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 39), 'half_long', False)
            # Processing the call keyword arguments (line 287)
            kwargs_279936 = {}
            # Getting the type of 'ma' (line 287)
            ma_279933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'ma', False)
            # Obtaining the member 'sin' of a type (line 287)
            sin_279934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 32), ma_279933, 'sin')
            # Calling sin(args, kwargs) (line 287)
            sin_call_result_279937 = invoke(stypy.reporting.localization.Localization(__file__, 287, 32), sin_279934, *[half_long_279935], **kwargs_279936)
            
            # Applying the binary operator '*' (line 287)
            result_mul_279938 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 17), '*', cos_latitude_279932, sin_call_result_279937)
            
            # Getting the type of 'sinc_alpha' (line 287)
            sinc_alpha_279939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 53), 'sinc_alpha')
            # Applying the binary operator 'div' (line 287)
            result_div_279940 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 16), 'div', result_mul_279938, sinc_alpha_279939)
            
            # Assigning a type to the variable 'x' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'x', result_div_279940)
            
            # Assigning a BinOp to a Name (line 288):
            
            # Assigning a BinOp to a Name (line 288):
            
            # Call to sin(...): (line 288)
            # Processing the call arguments (line 288)
            # Getting the type of 'latitude' (line 288)
            latitude_279943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'latitude', False)
            # Processing the call keyword arguments (line 288)
            kwargs_279944 = {}
            # Getting the type of 'ma' (line 288)
            ma_279941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 17), 'ma', False)
            # Obtaining the member 'sin' of a type (line 288)
            sin_279942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 17), ma_279941, 'sin')
            # Calling sin(args, kwargs) (line 288)
            sin_call_result_279945 = invoke(stypy.reporting.localization.Localization(__file__, 288, 17), sin_279942, *[latitude_279943], **kwargs_279944)
            
            # Getting the type of 'sinc_alpha' (line 288)
            sinc_alpha_279946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 36), 'sinc_alpha')
            # Applying the binary operator 'div' (line 288)
            result_div_279947 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 17), 'div', sin_call_result_279945, sinc_alpha_279946)
            
            # Assigning a type to the variable 'y' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'y', result_div_279947)
            
            # Call to concatenate(...): (line 289)
            # Processing the call arguments (line 289)
            
            # Obtaining an instance of the builtin type 'tuple' (line 289)
            tuple_279950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 289)
            # Adding element type (line 289)
            
            # Call to filled(...): (line 289)
            # Processing the call arguments (line 289)
            int_279953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 44), 'int')
            # Processing the call keyword arguments (line 289)
            kwargs_279954 = {}
            # Getting the type of 'x' (line 289)
            x_279951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'x', False)
            # Obtaining the member 'filled' of a type (line 289)
            filled_279952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 35), x_279951, 'filled')
            # Calling filled(args, kwargs) (line 289)
            filled_call_result_279955 = invoke(stypy.reporting.localization.Localization(__file__, 289, 35), filled_279952, *[int_279953], **kwargs_279954)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 35), tuple_279950, filled_call_result_279955)
            # Adding element type (line 289)
            
            # Call to filled(...): (line 289)
            # Processing the call arguments (line 289)
            int_279958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 57), 'int')
            # Processing the call keyword arguments (line 289)
            kwargs_279959 = {}
            # Getting the type of 'y' (line 289)
            y_279956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 48), 'y', False)
            # Obtaining the member 'filled' of a type (line 289)
            filled_279957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 48), y_279956, 'filled')
            # Calling filled(args, kwargs) (line 289)
            filled_call_result_279960 = invoke(stypy.reporting.localization.Localization(__file__, 289, 48), filled_279957, *[int_279958], **kwargs_279959)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 35), tuple_279950, filled_call_result_279960)
            
            int_279961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 62), 'int')
            # Processing the call keyword arguments (line 289)
            kwargs_279962 = {}
            # Getting the type of 'np' (line 289)
            np_279948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 19), 'np', False)
            # Obtaining the member 'concatenate' of a type (line 289)
            concatenate_279949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 19), np_279948, 'concatenate')
            # Calling concatenate(args, kwargs) (line 289)
            concatenate_call_result_279963 = invoke(stypy.reporting.localization.Localization(__file__, 289, 19), concatenate_279949, *[tuple_279950, int_279961], **kwargs_279962)
            
            # Assigning a type to the variable 'stypy_return_type' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'stypy_return_type', concatenate_call_result_279963)
            
            # ################# End of 'transform_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 271)
            stypy_return_type_279964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_279964)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_non_affine'
            return stypy_return_type_279964

        
        # Assigning a Attribute to a Attribute (line 290):
        
        # Assigning a Attribute to a Attribute (line 290):
        # Getting the type of 'Transform' (line 290)
        Transform_279965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 39), 'Transform')
        # Obtaining the member 'transform_non_affine' of a type (line 290)
        transform_non_affine_279966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 39), Transform_279965, 'transform_non_affine')
        # Obtaining the member '__doc__' of a type (line 290)
        doc___279967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 39), transform_non_affine_279966, '__doc__')
        # Getting the type of 'transform_non_affine' (line 290)
        transform_non_affine_279968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'transform_non_affine')
        # Setting the type of the member '__doc__' of a type (line 290)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), transform_non_affine_279968, '__doc__', doc___279967)

        @norecursion
        def transform_path_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_path_non_affine'
            module_type_store = module_type_store.open_function_context('transform_path_non_affine', 292, 8, False)
            # Assigning a type to the variable 'self' (line 293)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_localization', localization)
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_function_name', 'AitoffTransform.transform_path_non_affine')
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_param_names_list', ['path'])
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            AitoffTransform.transform_path_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'AitoffTransform.transform_path_non_affine', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_path_non_affine', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_path_non_affine(...)' code ##################

            
            # Assigning a Attribute to a Name (line 293):
            
            # Assigning a Attribute to a Name (line 293):
            # Getting the type of 'path' (line 293)
            path_279969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 23), 'path')
            # Obtaining the member 'vertices' of a type (line 293)
            vertices_279970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 23), path_279969, 'vertices')
            # Assigning a type to the variable 'vertices' (line 293)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'vertices', vertices_279970)
            
            # Assigning a Call to a Name (line 294):
            
            # Assigning a Call to a Name (line 294):
            
            # Call to interpolated(...): (line 294)
            # Processing the call arguments (line 294)
            # Getting the type of 'self' (line 294)
            self_279973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 38), 'self', False)
            # Obtaining the member '_resolution' of a type (line 294)
            _resolution_279974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 38), self_279973, '_resolution')
            # Processing the call keyword arguments (line 294)
            kwargs_279975 = {}
            # Getting the type of 'path' (line 294)
            path_279971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 20), 'path', False)
            # Obtaining the member 'interpolated' of a type (line 294)
            interpolated_279972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 20), path_279971, 'interpolated')
            # Calling interpolated(args, kwargs) (line 294)
            interpolated_call_result_279976 = invoke(stypy.reporting.localization.Localization(__file__, 294, 20), interpolated_279972, *[_resolution_279974], **kwargs_279975)
            
            # Assigning a type to the variable 'ipath' (line 294)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'ipath', interpolated_call_result_279976)
            
            # Call to Path(...): (line 295)
            # Processing the call arguments (line 295)
            
            # Call to transform(...): (line 295)
            # Processing the call arguments (line 295)
            # Getting the type of 'ipath' (line 295)
            ipath_279980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'ipath', False)
            # Obtaining the member 'vertices' of a type (line 295)
            vertices_279981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 39), ipath_279980, 'vertices')
            # Processing the call keyword arguments (line 295)
            kwargs_279982 = {}
            # Getting the type of 'self' (line 295)
            self_279978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'self', False)
            # Obtaining the member 'transform' of a type (line 295)
            transform_279979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 24), self_279978, 'transform')
            # Calling transform(args, kwargs) (line 295)
            transform_call_result_279983 = invoke(stypy.reporting.localization.Localization(__file__, 295, 24), transform_279979, *[vertices_279981], **kwargs_279982)
            
            # Getting the type of 'ipath' (line 295)
            ipath_279984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 56), 'ipath', False)
            # Obtaining the member 'codes' of a type (line 295)
            codes_279985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 56), ipath_279984, 'codes')
            # Processing the call keyword arguments (line 295)
            kwargs_279986 = {}
            # Getting the type of 'Path' (line 295)
            Path_279977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 295)
            Path_call_result_279987 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), Path_279977, *[transform_call_result_279983, codes_279985], **kwargs_279986)
            
            # Assigning a type to the variable 'stypy_return_type' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'stypy_return_type', Path_call_result_279987)
            
            # ################# End of 'transform_path_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_path_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 292)
            stypy_return_type_279988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_279988)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_path_non_affine'
            return stypy_return_type_279988

        
        # Assigning a Attribute to a Attribute (line 296):
        
        # Assigning a Attribute to a Attribute (line 296):
        # Getting the type of 'Transform' (line 296)
        Transform_279989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 44), 'Transform')
        # Obtaining the member 'transform_path_non_affine' of a type (line 296)
        transform_path_non_affine_279990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 44), Transform_279989, 'transform_path_non_affine')
        # Obtaining the member '__doc__' of a type (line 296)
        doc___279991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 44), transform_path_non_affine_279990, '__doc__')
        # Getting the type of 'transform_path_non_affine' (line 296)
        transform_path_non_affine_279992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'transform_path_non_affine')
        # Setting the type of the member '__doc__' of a type (line 296)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), transform_path_non_affine_279992, '__doc__', doc___279991)

        @norecursion
        def inverted(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inverted'
            module_type_store = module_type_store.open_function_context('inverted', 298, 8, False)
            # Assigning a type to the variable 'self' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            AitoffTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
            AitoffTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            AitoffTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
            AitoffTransform.inverted.__dict__.__setitem__('stypy_function_name', 'AitoffTransform.inverted')
            AitoffTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
            AitoffTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
            AitoffTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
            AitoffTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
            AitoffTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
            AitoffTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            AitoffTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'AitoffTransform.inverted', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inverted', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inverted(...)' code ##################

            
            # Call to InvertedAitoffTransform(...): (line 299)
            # Processing the call arguments (line 299)
            # Getting the type of 'self' (line 299)
            self_279995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 54), 'self', False)
            # Obtaining the member '_resolution' of a type (line 299)
            _resolution_279996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 54), self_279995, '_resolution')
            # Processing the call keyword arguments (line 299)
            kwargs_279997 = {}
            # Getting the type of 'AitoffAxes' (line 299)
            AitoffAxes_279993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 19), 'AitoffAxes', False)
            # Obtaining the member 'InvertedAitoffTransform' of a type (line 299)
            InvertedAitoffTransform_279994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 19), AitoffAxes_279993, 'InvertedAitoffTransform')
            # Calling InvertedAitoffTransform(args, kwargs) (line 299)
            InvertedAitoffTransform_call_result_279998 = invoke(stypy.reporting.localization.Localization(__file__, 299, 19), InvertedAitoffTransform_279994, *[_resolution_279996], **kwargs_279997)
            
            # Assigning a type to the variable 'stypy_return_type' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'stypy_return_type', InvertedAitoffTransform_call_result_279998)
            
            # ################# End of 'inverted(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inverted' in the type store
            # Getting the type of 'stypy_return_type' (line 298)
            stypy_return_type_279999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_279999)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inverted'
            return stypy_return_type_279999

        
        # Assigning a Attribute to a Attribute (line 300):
        
        # Assigning a Attribute to a Attribute (line 300):
        # Getting the type of 'Transform' (line 300)
        Transform_280000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'Transform')
        # Obtaining the member 'inverted' of a type (line 300)
        inverted_280001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 27), Transform_280000, 'inverted')
        # Obtaining the member '__doc__' of a type (line 300)
        doc___280002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 27), inverted_280001, '__doc__')
        # Getting the type of 'inverted' (line 300)
        inverted_280003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'inverted')
        # Setting the type of the member '__doc__' of a type (line 300)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), inverted_280003, '__doc__', doc___280002)
    
    # Assigning a type to the variable 'AitoffTransform' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'AitoffTransform', AitoffTransform)
    # Declaration of the 'InvertedAitoffTransform' class
    # Getting the type of 'Transform' (line 302)
    Transform_280004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 34), 'Transform')

    class InvertedAitoffTransform(Transform_280004, ):
        
        # Assigning a Num to a Name (line 303):
        
        # Assigning a Num to a Name (line 303):
        int_280005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 21), 'int')
        # Assigning a type to the variable 'input_dims' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'input_dims', int_280005)
        
        # Assigning a Num to a Name (line 304):
        
        # Assigning a Num to a Name (line 304):
        int_280006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 22), 'int')
        # Assigning a type to the variable 'output_dims' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'output_dims', int_280006)
        
        # Assigning a Name to a Name (line 305):
        
        # Assigning a Name to a Name (line 305):
        # Getting the type of 'False' (line 305)
        False_280007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 23), 'False')
        # Assigning a type to the variable 'is_separable' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'is_separable', False_280007)

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 307, 8, False)
            # Assigning a type to the variable 'self' (line 308)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedAitoffTransform.__init__', ['resolution'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['resolution'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Call to __init__(...): (line 308)
            # Processing the call arguments (line 308)
            # Getting the type of 'self' (line 308)
            self_280010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 31), 'self', False)
            # Processing the call keyword arguments (line 308)
            kwargs_280011 = {}
            # Getting the type of 'Transform' (line 308)
            Transform_280008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'Transform', False)
            # Obtaining the member '__init__' of a type (line 308)
            init___280009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), Transform_280008, '__init__')
            # Calling __init__(args, kwargs) (line 308)
            init___call_result_280012 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), init___280009, *[self_280010], **kwargs_280011)
            
            
            # Assigning a Name to a Attribute (line 309):
            
            # Assigning a Name to a Attribute (line 309):
            # Getting the type of 'resolution' (line 309)
            resolution_280013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 'resolution')
            # Getting the type of 'self' (line 309)
            self_280014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'self')
            # Setting the type of the member '_resolution' of a type (line 309)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), self_280014, '_resolution', resolution_280013)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_non_affine'
            module_type_store = module_type_store.open_function_context('transform_non_affine', 311, 8, False)
            # Assigning a type to the variable 'self' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'InvertedAitoffTransform.transform_non_affine')
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['xy'])
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            InvertedAitoffTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedAitoffTransform.transform_non_affine', ['xy'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_non_affine', localization, ['xy'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_non_affine(...)' code ##################

            # Getting the type of 'xy' (line 313)
            xy_280015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'xy')
            # Assigning a type to the variable 'stypy_return_type' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'stypy_return_type', xy_280015)
            
            # ################# End of 'transform_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 311)
            stypy_return_type_280016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280016)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_non_affine'
            return stypy_return_type_280016

        
        # Assigning a Attribute to a Attribute (line 314):
        
        # Assigning a Attribute to a Attribute (line 314):
        # Getting the type of 'Transform' (line 314)
        Transform_280017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 39), 'Transform')
        # Obtaining the member 'transform_non_affine' of a type (line 314)
        transform_non_affine_280018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 39), Transform_280017, 'transform_non_affine')
        # Obtaining the member '__doc__' of a type (line 314)
        doc___280019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 39), transform_non_affine_280018, '__doc__')
        # Getting the type of 'transform_non_affine' (line 314)
        transform_non_affine_280020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'transform_non_affine')
        # Setting the type of the member '__doc__' of a type (line 314)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), transform_non_affine_280020, '__doc__', doc___280019)

        @norecursion
        def inverted(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inverted'
            module_type_store = module_type_store.open_function_context('inverted', 316, 8, False)
            # Assigning a type to the variable 'self' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedAitoffTransform.inverted')
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            InvertedAitoffTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedAitoffTransform.inverted', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inverted', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inverted(...)' code ##################

            
            # Call to AitoffTransform(...): (line 317)
            # Processing the call arguments (line 317)
            # Getting the type of 'self' (line 317)
            self_280023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 46), 'self', False)
            # Obtaining the member '_resolution' of a type (line 317)
            _resolution_280024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 46), self_280023, '_resolution')
            # Processing the call keyword arguments (line 317)
            kwargs_280025 = {}
            # Getting the type of 'AitoffAxes' (line 317)
            AitoffAxes_280021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'AitoffAxes', False)
            # Obtaining the member 'AitoffTransform' of a type (line 317)
            AitoffTransform_280022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 19), AitoffAxes_280021, 'AitoffTransform')
            # Calling AitoffTransform(args, kwargs) (line 317)
            AitoffTransform_call_result_280026 = invoke(stypy.reporting.localization.Localization(__file__, 317, 19), AitoffTransform_280022, *[_resolution_280024], **kwargs_280025)
            
            # Assigning a type to the variable 'stypy_return_type' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'stypy_return_type', AitoffTransform_call_result_280026)
            
            # ################# End of 'inverted(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inverted' in the type store
            # Getting the type of 'stypy_return_type' (line 316)
            stypy_return_type_280027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280027)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inverted'
            return stypy_return_type_280027

        
        # Assigning a Attribute to a Attribute (line 318):
        
        # Assigning a Attribute to a Attribute (line 318):
        # Getting the type of 'Transform' (line 318)
        Transform_280028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'Transform')
        # Obtaining the member 'inverted' of a type (line 318)
        inverted_280029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 27), Transform_280028, 'inverted')
        # Obtaining the member '__doc__' of a type (line 318)
        doc___280030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 27), inverted_280029, '__doc__')
        # Getting the type of 'inverted' (line 318)
        inverted_280031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'inverted')
        # Setting the type of the member '__doc__' of a type (line 318)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), inverted_280031, '__doc__', doc___280030)
    
    # Assigning a type to the variable 'InvertedAitoffTransform' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'InvertedAitoffTransform', InvertedAitoffTransform)

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 320, 4, False)
        # Assigning a type to the variable 'self' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AitoffAxes.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Attribute (line 321):
        
        # Assigning a BinOp to a Attribute (line 321):
        # Getting the type of 'np' (line 321)
        np_280032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'np')
        # Obtaining the member 'pi' of a type (line 321)
        pi_280033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 30), np_280032, 'pi')
        float_280034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 38), 'float')
        # Applying the binary operator 'div' (line 321)
        result_div_280035 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 30), 'div', pi_280033, float_280034)
        
        # Getting the type of 'self' (line 321)
        self_280036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self')
        # Setting the type of the member '_longitude_cap' of a type (line 321)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_280036, '_longitude_cap', result_div_280035)
        
        # Call to __init__(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'self' (line 322)
        self_280039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'self', False)
        # Getting the type of 'args' (line 322)
        args_280040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 32), 'args', False)
        # Processing the call keyword arguments (line 322)
        # Getting the type of 'kwargs' (line 322)
        kwargs_280041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 40), 'kwargs', False)
        kwargs_280042 = {'kwargs_280041': kwargs_280041}
        # Getting the type of 'GeoAxes' (line 322)
        GeoAxes_280037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'GeoAxes', False)
        # Obtaining the member '__init__' of a type (line 322)
        init___280038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), GeoAxes_280037, '__init__')
        # Calling __init__(args, kwargs) (line 322)
        init___call_result_280043 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), init___280038, *[self_280039, args_280040], **kwargs_280042)
        
        
        # Call to set_aspect(...): (line 323)
        # Processing the call arguments (line 323)
        float_280046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 24), 'float')
        # Processing the call keyword arguments (line 323)
        unicode_280047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 40), 'unicode', u'box')
        keyword_280048 = unicode_280047
        unicode_280049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 54), 'unicode', u'C')
        keyword_280050 = unicode_280049
        kwargs_280051 = {'adjustable': keyword_280048, 'anchor': keyword_280050}
        # Getting the type of 'self' (line 323)
        self_280044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self', False)
        # Obtaining the member 'set_aspect' of a type (line 323)
        set_aspect_280045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_280044, 'set_aspect')
        # Calling set_aspect(args, kwargs) (line 323)
        set_aspect_call_result_280052 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), set_aspect_280045, *[float_280046], **kwargs_280051)
        
        
        # Call to cla(...): (line 324)
        # Processing the call keyword arguments (line 324)
        kwargs_280055 = {}
        # Getting the type of 'self' (line 324)
        self_280053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'self', False)
        # Obtaining the member 'cla' of a type (line 324)
        cla_280054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), self_280053, 'cla')
        # Calling cla(args, kwargs) (line 324)
        cla_call_result_280056 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), cla_280054, *[], **kwargs_280055)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _get_core_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_core_transform'
        module_type_store = module_type_store.open_function_context('_get_core_transform', 326, 4, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_localization', localization)
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_function_name', 'AitoffAxes._get_core_transform')
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_param_names_list', ['resolution'])
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AitoffAxes._get_core_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AitoffAxes._get_core_transform', ['resolution'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_core_transform', localization, ['resolution'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_core_transform(...)' code ##################

        
        # Call to AitoffTransform(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'resolution' (line 327)
        resolution_280059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 36), 'resolution', False)
        # Processing the call keyword arguments (line 327)
        kwargs_280060 = {}
        # Getting the type of 'self' (line 327)
        self_280057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'self', False)
        # Obtaining the member 'AitoffTransform' of a type (line 327)
        AitoffTransform_280058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 15), self_280057, 'AitoffTransform')
        # Calling AitoffTransform(args, kwargs) (line 327)
        AitoffTransform_call_result_280061 = invoke(stypy.reporting.localization.Localization(__file__, 327, 15), AitoffTransform_280058, *[resolution_280059], **kwargs_280060)
        
        # Assigning a type to the variable 'stypy_return_type' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stypy_return_type', AitoffTransform_call_result_280061)
        
        # ################# End of '_get_core_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_core_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 326)
        stypy_return_type_280062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_280062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_core_transform'
        return stypy_return_type_280062


# Assigning a type to the variable 'AitoffAxes' (line 251)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'AitoffAxes', AitoffAxes)

# Assigning a Str to a Name (line 252):
unicode_280063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 11), 'unicode', u'aitoff')
# Getting the type of 'AitoffAxes'
AitoffAxes_280064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AitoffAxes')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AitoffAxes_280064, 'name', unicode_280063)
# Declaration of the 'HammerAxes' class
# Getting the type of 'GeoAxes' (line 330)
GeoAxes_280065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'GeoAxes')

class HammerAxes(GeoAxes_280065, ):
    
    # Assigning a Str to a Name (line 331):
    # Declaration of the 'HammerTransform' class
    # Getting the type of 'Transform' (line 333)
    Transform_280066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 26), 'Transform')

    class HammerTransform(Transform_280066, ):
        unicode_280067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, (-1)), 'unicode', u'\n        The base Hammer transform.\n        ')
        
        # Assigning a Num to a Name (line 337):
        
        # Assigning a Num to a Name (line 337):
        int_280068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 21), 'int')
        # Assigning a type to the variable 'input_dims' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'input_dims', int_280068)
        
        # Assigning a Num to a Name (line 338):
        
        # Assigning a Num to a Name (line 338):
        int_280069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 22), 'int')
        # Assigning a type to the variable 'output_dims' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'output_dims', int_280069)
        
        # Assigning a Name to a Name (line 339):
        
        # Assigning a Name to a Name (line 339):
        # Getting the type of 'False' (line 339)
        False_280070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'False')
        # Assigning a type to the variable 'is_separable' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'is_separable', False_280070)

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 341, 8, False)
            # Assigning a type to the variable 'self' (line 342)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'HammerTransform.__init__', ['resolution'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['resolution'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            unicode_280071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, (-1)), 'unicode', u'\n            Create a new Hammer transform.  Resolution is the number of steps\n            to interpolate between each input line segment to approximate its\n            path in curved Hammer space.\n            ')
            
            # Call to __init__(...): (line 347)
            # Processing the call arguments (line 347)
            # Getting the type of 'self' (line 347)
            self_280074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 31), 'self', False)
            # Processing the call keyword arguments (line 347)
            kwargs_280075 = {}
            # Getting the type of 'Transform' (line 347)
            Transform_280072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'Transform', False)
            # Obtaining the member '__init__' of a type (line 347)
            init___280073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), Transform_280072, '__init__')
            # Calling __init__(args, kwargs) (line 347)
            init___call_result_280076 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), init___280073, *[self_280074], **kwargs_280075)
            
            
            # Assigning a Name to a Attribute (line 348):
            
            # Assigning a Name to a Attribute (line 348):
            # Getting the type of 'resolution' (line 348)
            resolution_280077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 31), 'resolution')
            # Getting the type of 'self' (line 348)
            self_280078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'self')
            # Setting the type of the member '_resolution' of a type (line 348)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), self_280078, '_resolution', resolution_280077)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_non_affine'
            module_type_store = module_type_store.open_function_context('transform_non_affine', 350, 8, False)
            # Assigning a type to the variable 'self' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'HammerTransform.transform_non_affine')
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['ll'])
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            HammerTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'HammerTransform.transform_non_affine', ['ll'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_non_affine', localization, ['ll'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_non_affine(...)' code ##################

            
            # Assigning a Subscript to a Name (line 351):
            
            # Assigning a Subscript to a Name (line 351):
            
            # Obtaining the type of the subscript
            slice_280079 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 351, 24), None, None, None)
            int_280080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 30), 'int')
            int_280081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 32), 'int')
            slice_280082 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 351, 24), int_280080, int_280081, None)
            # Getting the type of 'll' (line 351)
            ll_280083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'll')
            # Obtaining the member '__getitem__' of a type (line 351)
            getitem___280084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 24), ll_280083, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 351)
            subscript_call_result_280085 = invoke(stypy.reporting.localization.Localization(__file__, 351, 24), getitem___280084, (slice_280079, slice_280082))
            
            # Assigning a type to the variable 'longitude' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'longitude', subscript_call_result_280085)
            
            # Assigning a Subscript to a Name (line 352):
            
            # Assigning a Subscript to a Name (line 352):
            
            # Obtaining the type of the subscript
            slice_280086 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 352, 24), None, None, None)
            int_280087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 30), 'int')
            int_280088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 32), 'int')
            slice_280089 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 352, 24), int_280087, int_280088, None)
            # Getting the type of 'll' (line 352)
            ll_280090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 24), 'll')
            # Obtaining the member '__getitem__' of a type (line 352)
            getitem___280091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 24), ll_280090, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 352)
            subscript_call_result_280092 = invoke(stypy.reporting.localization.Localization(__file__, 352, 24), getitem___280091, (slice_280086, slice_280089))
            
            # Assigning a type to the variable 'latitude' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'latitude', subscript_call_result_280092)
            
            # Assigning a BinOp to a Name (line 355):
            
            # Assigning a BinOp to a Name (line 355):
            # Getting the type of 'longitude' (line 355)
            longitude_280093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'longitude')
            float_280094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 36), 'float')
            # Applying the binary operator 'div' (line 355)
            result_div_280095 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 24), 'div', longitude_280093, float_280094)
            
            # Assigning a type to the variable 'half_long' (line 355)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'half_long', result_div_280095)
            
            # Assigning a Call to a Name (line 356):
            
            # Assigning a Call to a Name (line 356):
            
            # Call to cos(...): (line 356)
            # Processing the call arguments (line 356)
            # Getting the type of 'latitude' (line 356)
            latitude_280098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'latitude', False)
            # Processing the call keyword arguments (line 356)
            kwargs_280099 = {}
            # Getting the type of 'np' (line 356)
            np_280096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 27), 'np', False)
            # Obtaining the member 'cos' of a type (line 356)
            cos_280097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 27), np_280096, 'cos')
            # Calling cos(args, kwargs) (line 356)
            cos_call_result_280100 = invoke(stypy.reporting.localization.Localization(__file__, 356, 27), cos_280097, *[latitude_280098], **kwargs_280099)
            
            # Assigning a type to the variable 'cos_latitude' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'cos_latitude', cos_call_result_280100)
            
            # Assigning a Call to a Name (line 357):
            
            # Assigning a Call to a Name (line 357):
            
            # Call to sqrt(...): (line 357)
            # Processing the call arguments (line 357)
            float_280103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 28), 'float')
            # Processing the call keyword arguments (line 357)
            kwargs_280104 = {}
            # Getting the type of 'np' (line 357)
            np_280101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 357)
            sqrt_280102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 20), np_280101, 'sqrt')
            # Calling sqrt(args, kwargs) (line 357)
            sqrt_call_result_280105 = invoke(stypy.reporting.localization.Localization(__file__, 357, 20), sqrt_280102, *[float_280103], **kwargs_280104)
            
            # Assigning a type to the variable 'sqrt2' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'sqrt2', sqrt_call_result_280105)
            
            # Assigning a Call to a Name (line 359):
            
            # Assigning a Call to a Name (line 359):
            
            # Call to sqrt(...): (line 359)
            # Processing the call arguments (line 359)
            float_280108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 28), 'float')
            # Getting the type of 'cos_latitude' (line 359)
            cos_latitude_280109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 34), 'cos_latitude', False)
            
            # Call to cos(...): (line 359)
            # Processing the call arguments (line 359)
            # Getting the type of 'half_long' (line 359)
            half_long_280112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 56), 'half_long', False)
            # Processing the call keyword arguments (line 359)
            kwargs_280113 = {}
            # Getting the type of 'np' (line 359)
            np_280110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 49), 'np', False)
            # Obtaining the member 'cos' of a type (line 359)
            cos_280111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 49), np_280110, 'cos')
            # Calling cos(args, kwargs) (line 359)
            cos_call_result_280114 = invoke(stypy.reporting.localization.Localization(__file__, 359, 49), cos_280111, *[half_long_280112], **kwargs_280113)
            
            # Applying the binary operator '*' (line 359)
            result_mul_280115 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 34), '*', cos_latitude_280109, cos_call_result_280114)
            
            # Applying the binary operator '+' (line 359)
            result_add_280116 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 28), '+', float_280108, result_mul_280115)
            
            # Processing the call keyword arguments (line 359)
            kwargs_280117 = {}
            # Getting the type of 'np' (line 359)
            np_280106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 20), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 359)
            sqrt_280107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 20), np_280106, 'sqrt')
            # Calling sqrt(args, kwargs) (line 359)
            sqrt_call_result_280118 = invoke(stypy.reporting.localization.Localization(__file__, 359, 20), sqrt_280107, *[result_add_280116], **kwargs_280117)
            
            # Assigning a type to the variable 'alpha' (line 359)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'alpha', sqrt_call_result_280118)
            
            # Assigning a BinOp to a Name (line 360):
            
            # Assigning a BinOp to a Name (line 360):
            float_280119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 17), 'float')
            # Getting the type of 'sqrt2' (line 360)
            sqrt2_280120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'sqrt2')
            # Applying the binary operator '*' (line 360)
            result_mul_280121 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 17), '*', float_280119, sqrt2_280120)
            
            # Getting the type of 'cos_latitude' (line 360)
            cos_latitude_280122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 33), 'cos_latitude')
            
            # Call to sin(...): (line 360)
            # Processing the call arguments (line 360)
            # Getting the type of 'half_long' (line 360)
            half_long_280125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 55), 'half_long', False)
            # Processing the call keyword arguments (line 360)
            kwargs_280126 = {}
            # Getting the type of 'np' (line 360)
            np_280123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 48), 'np', False)
            # Obtaining the member 'sin' of a type (line 360)
            sin_280124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 48), np_280123, 'sin')
            # Calling sin(args, kwargs) (line 360)
            sin_call_result_280127 = invoke(stypy.reporting.localization.Localization(__file__, 360, 48), sin_280124, *[half_long_280125], **kwargs_280126)
            
            # Applying the binary operator '*' (line 360)
            result_mul_280128 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 33), '*', cos_latitude_280122, sin_call_result_280127)
            
            # Applying the binary operator '*' (line 360)
            result_mul_280129 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 16), '*', result_mul_280121, result_mul_280128)
            
            # Getting the type of 'alpha' (line 360)
            alpha_280130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 69), 'alpha')
            # Applying the binary operator 'div' (line 360)
            result_div_280131 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 67), 'div', result_mul_280129, alpha_280130)
            
            # Assigning a type to the variable 'x' (line 360)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'x', result_div_280131)
            
            # Assigning a BinOp to a Name (line 361):
            
            # Assigning a BinOp to a Name (line 361):
            # Getting the type of 'sqrt2' (line 361)
            sqrt2_280132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 17), 'sqrt2')
            
            # Call to sin(...): (line 361)
            # Processing the call arguments (line 361)
            # Getting the type of 'latitude' (line 361)
            latitude_280135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 32), 'latitude', False)
            # Processing the call keyword arguments (line 361)
            kwargs_280136 = {}
            # Getting the type of 'np' (line 361)
            np_280133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 25), 'np', False)
            # Obtaining the member 'sin' of a type (line 361)
            sin_280134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 25), np_280133, 'sin')
            # Calling sin(args, kwargs) (line 361)
            sin_call_result_280137 = invoke(stypy.reporting.localization.Localization(__file__, 361, 25), sin_280134, *[latitude_280135], **kwargs_280136)
            
            # Applying the binary operator '*' (line 361)
            result_mul_280138 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 17), '*', sqrt2_280132, sin_call_result_280137)
            
            # Getting the type of 'alpha' (line 361)
            alpha_280139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 45), 'alpha')
            # Applying the binary operator 'div' (line 361)
            result_div_280140 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 16), 'div', result_mul_280138, alpha_280139)
            
            # Assigning a type to the variable 'y' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'y', result_div_280140)
            
            # Call to concatenate(...): (line 362)
            # Processing the call arguments (line 362)
            
            # Obtaining an instance of the builtin type 'tuple' (line 362)
            tuple_280143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 362)
            # Adding element type (line 362)
            # Getting the type of 'x' (line 362)
            x_280144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 35), tuple_280143, x_280144)
            # Adding element type (line 362)
            # Getting the type of 'y' (line 362)
            y_280145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 38), 'y', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 35), tuple_280143, y_280145)
            
            int_280146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 42), 'int')
            # Processing the call keyword arguments (line 362)
            kwargs_280147 = {}
            # Getting the type of 'np' (line 362)
            np_280141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'np', False)
            # Obtaining the member 'concatenate' of a type (line 362)
            concatenate_280142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 19), np_280141, 'concatenate')
            # Calling concatenate(args, kwargs) (line 362)
            concatenate_call_result_280148 = invoke(stypy.reporting.localization.Localization(__file__, 362, 19), concatenate_280142, *[tuple_280143, int_280146], **kwargs_280147)
            
            # Assigning a type to the variable 'stypy_return_type' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'stypy_return_type', concatenate_call_result_280148)
            
            # ################# End of 'transform_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 350)
            stypy_return_type_280149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280149)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_non_affine'
            return stypy_return_type_280149

        
        # Assigning a Attribute to a Attribute (line 363):
        
        # Assigning a Attribute to a Attribute (line 363):
        # Getting the type of 'Transform' (line 363)
        Transform_280150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 39), 'Transform')
        # Obtaining the member 'transform_non_affine' of a type (line 363)
        transform_non_affine_280151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 39), Transform_280150, 'transform_non_affine')
        # Obtaining the member '__doc__' of a type (line 363)
        doc___280152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 39), transform_non_affine_280151, '__doc__')
        # Getting the type of 'transform_non_affine' (line 363)
        transform_non_affine_280153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'transform_non_affine')
        # Setting the type of the member '__doc__' of a type (line 363)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), transform_non_affine_280153, '__doc__', doc___280152)

        @norecursion
        def transform_path_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_path_non_affine'
            module_type_store = module_type_store.open_function_context('transform_path_non_affine', 365, 8, False)
            # Assigning a type to the variable 'self' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_localization', localization)
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_function_name', 'HammerTransform.transform_path_non_affine')
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_param_names_list', ['path'])
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            HammerTransform.transform_path_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'HammerTransform.transform_path_non_affine', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_path_non_affine', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_path_non_affine(...)' code ##################

            
            # Assigning a Attribute to a Name (line 366):
            
            # Assigning a Attribute to a Name (line 366):
            # Getting the type of 'path' (line 366)
            path_280154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 23), 'path')
            # Obtaining the member 'vertices' of a type (line 366)
            vertices_280155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 23), path_280154, 'vertices')
            # Assigning a type to the variable 'vertices' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'vertices', vertices_280155)
            
            # Assigning a Call to a Name (line 367):
            
            # Assigning a Call to a Name (line 367):
            
            # Call to interpolated(...): (line 367)
            # Processing the call arguments (line 367)
            # Getting the type of 'self' (line 367)
            self_280158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 38), 'self', False)
            # Obtaining the member '_resolution' of a type (line 367)
            _resolution_280159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 38), self_280158, '_resolution')
            # Processing the call keyword arguments (line 367)
            kwargs_280160 = {}
            # Getting the type of 'path' (line 367)
            path_280156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'path', False)
            # Obtaining the member 'interpolated' of a type (line 367)
            interpolated_280157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 20), path_280156, 'interpolated')
            # Calling interpolated(args, kwargs) (line 367)
            interpolated_call_result_280161 = invoke(stypy.reporting.localization.Localization(__file__, 367, 20), interpolated_280157, *[_resolution_280159], **kwargs_280160)
            
            # Assigning a type to the variable 'ipath' (line 367)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'ipath', interpolated_call_result_280161)
            
            # Call to Path(...): (line 368)
            # Processing the call arguments (line 368)
            
            # Call to transform(...): (line 368)
            # Processing the call arguments (line 368)
            # Getting the type of 'ipath' (line 368)
            ipath_280165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 39), 'ipath', False)
            # Obtaining the member 'vertices' of a type (line 368)
            vertices_280166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 39), ipath_280165, 'vertices')
            # Processing the call keyword arguments (line 368)
            kwargs_280167 = {}
            # Getting the type of 'self' (line 368)
            self_280163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'self', False)
            # Obtaining the member 'transform' of a type (line 368)
            transform_280164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 24), self_280163, 'transform')
            # Calling transform(args, kwargs) (line 368)
            transform_call_result_280168 = invoke(stypy.reporting.localization.Localization(__file__, 368, 24), transform_280164, *[vertices_280166], **kwargs_280167)
            
            # Getting the type of 'ipath' (line 368)
            ipath_280169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 56), 'ipath', False)
            # Obtaining the member 'codes' of a type (line 368)
            codes_280170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 56), ipath_280169, 'codes')
            # Processing the call keyword arguments (line 368)
            kwargs_280171 = {}
            # Getting the type of 'Path' (line 368)
            Path_280162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 368)
            Path_call_result_280172 = invoke(stypy.reporting.localization.Localization(__file__, 368, 19), Path_280162, *[transform_call_result_280168, codes_280170], **kwargs_280171)
            
            # Assigning a type to the variable 'stypy_return_type' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'stypy_return_type', Path_call_result_280172)
            
            # ################# End of 'transform_path_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_path_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 365)
            stypy_return_type_280173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280173)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_path_non_affine'
            return stypy_return_type_280173

        
        # Assigning a Attribute to a Attribute (line 369):
        
        # Assigning a Attribute to a Attribute (line 369):
        # Getting the type of 'Transform' (line 369)
        Transform_280174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 44), 'Transform')
        # Obtaining the member 'transform_path_non_affine' of a type (line 369)
        transform_path_non_affine_280175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 44), Transform_280174, 'transform_path_non_affine')
        # Obtaining the member '__doc__' of a type (line 369)
        doc___280176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 44), transform_path_non_affine_280175, '__doc__')
        # Getting the type of 'transform_path_non_affine' (line 369)
        transform_path_non_affine_280177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'transform_path_non_affine')
        # Setting the type of the member '__doc__' of a type (line 369)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), transform_path_non_affine_280177, '__doc__', doc___280176)

        @norecursion
        def inverted(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inverted'
            module_type_store = module_type_store.open_function_context('inverted', 371, 8, False)
            # Assigning a type to the variable 'self' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            HammerTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
            HammerTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            HammerTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
            HammerTransform.inverted.__dict__.__setitem__('stypy_function_name', 'HammerTransform.inverted')
            HammerTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
            HammerTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
            HammerTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
            HammerTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
            HammerTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
            HammerTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            HammerTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'HammerTransform.inverted', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inverted', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inverted(...)' code ##################

            
            # Call to InvertedHammerTransform(...): (line 372)
            # Processing the call arguments (line 372)
            # Getting the type of 'self' (line 372)
            self_280180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 54), 'self', False)
            # Obtaining the member '_resolution' of a type (line 372)
            _resolution_280181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 54), self_280180, '_resolution')
            # Processing the call keyword arguments (line 372)
            kwargs_280182 = {}
            # Getting the type of 'HammerAxes' (line 372)
            HammerAxes_280178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 19), 'HammerAxes', False)
            # Obtaining the member 'InvertedHammerTransform' of a type (line 372)
            InvertedHammerTransform_280179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 19), HammerAxes_280178, 'InvertedHammerTransform')
            # Calling InvertedHammerTransform(args, kwargs) (line 372)
            InvertedHammerTransform_call_result_280183 = invoke(stypy.reporting.localization.Localization(__file__, 372, 19), InvertedHammerTransform_280179, *[_resolution_280181], **kwargs_280182)
            
            # Assigning a type to the variable 'stypy_return_type' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'stypy_return_type', InvertedHammerTransform_call_result_280183)
            
            # ################# End of 'inverted(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inverted' in the type store
            # Getting the type of 'stypy_return_type' (line 371)
            stypy_return_type_280184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280184)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inverted'
            return stypy_return_type_280184

        
        # Assigning a Attribute to a Attribute (line 373):
        
        # Assigning a Attribute to a Attribute (line 373):
        # Getting the type of 'Transform' (line 373)
        Transform_280185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 27), 'Transform')
        # Obtaining the member 'inverted' of a type (line 373)
        inverted_280186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 27), Transform_280185, 'inverted')
        # Obtaining the member '__doc__' of a type (line 373)
        doc___280187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 27), inverted_280186, '__doc__')
        # Getting the type of 'inverted' (line 373)
        inverted_280188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'inverted')
        # Setting the type of the member '__doc__' of a type (line 373)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), inverted_280188, '__doc__', doc___280187)
    
    # Assigning a type to the variable 'HammerTransform' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'HammerTransform', HammerTransform)
    # Declaration of the 'InvertedHammerTransform' class
    # Getting the type of 'Transform' (line 375)
    Transform_280189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 34), 'Transform')

    class InvertedHammerTransform(Transform_280189, ):
        
        # Assigning a Num to a Name (line 376):
        
        # Assigning a Num to a Name (line 376):
        int_280190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 21), 'int')
        # Assigning a type to the variable 'input_dims' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'input_dims', int_280190)
        
        # Assigning a Num to a Name (line 377):
        
        # Assigning a Num to a Name (line 377):
        int_280191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 22), 'int')
        # Assigning a type to the variable 'output_dims' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'output_dims', int_280191)
        
        # Assigning a Name to a Name (line 378):
        
        # Assigning a Name to a Name (line 378):
        # Getting the type of 'False' (line 378)
        False_280192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 23), 'False')
        # Assigning a type to the variable 'is_separable' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'is_separable', False_280192)

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 380, 8, False)
            # Assigning a type to the variable 'self' (line 381)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedHammerTransform.__init__', ['resolution'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['resolution'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Call to __init__(...): (line 381)
            # Processing the call arguments (line 381)
            # Getting the type of 'self' (line 381)
            self_280195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 31), 'self', False)
            # Processing the call keyword arguments (line 381)
            kwargs_280196 = {}
            # Getting the type of 'Transform' (line 381)
            Transform_280193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'Transform', False)
            # Obtaining the member '__init__' of a type (line 381)
            init___280194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 12), Transform_280193, '__init__')
            # Calling __init__(args, kwargs) (line 381)
            init___call_result_280197 = invoke(stypy.reporting.localization.Localization(__file__, 381, 12), init___280194, *[self_280195], **kwargs_280196)
            
            
            # Assigning a Name to a Attribute (line 382):
            
            # Assigning a Name to a Attribute (line 382):
            # Getting the type of 'resolution' (line 382)
            resolution_280198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 31), 'resolution')
            # Getting the type of 'self' (line 382)
            self_280199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'self')
            # Setting the type of the member '_resolution' of a type (line 382)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), self_280199, '_resolution', resolution_280198)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_non_affine'
            module_type_store = module_type_store.open_function_context('transform_non_affine', 384, 8, False)
            # Assigning a type to the variable 'self' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'InvertedHammerTransform.transform_non_affine')
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['xy'])
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            InvertedHammerTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedHammerTransform.transform_non_affine', ['xy'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_non_affine', localization, ['xy'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_non_affine(...)' code ##################

            
            # Assigning a Attribute to a Tuple (line 385):
            
            # Assigning a Subscript to a Name (line 385):
            
            # Obtaining the type of the subscript
            int_280200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 12), 'int')
            # Getting the type of 'xy' (line 385)
            xy_280201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'xy')
            # Obtaining the member 'T' of a type (line 385)
            T_280202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 19), xy_280201, 'T')
            # Obtaining the member '__getitem__' of a type (line 385)
            getitem___280203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 12), T_280202, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 385)
            subscript_call_result_280204 = invoke(stypy.reporting.localization.Localization(__file__, 385, 12), getitem___280203, int_280200)
            
            # Assigning a type to the variable 'tuple_var_assignment_279209' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'tuple_var_assignment_279209', subscript_call_result_280204)
            
            # Assigning a Subscript to a Name (line 385):
            
            # Obtaining the type of the subscript
            int_280205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 12), 'int')
            # Getting the type of 'xy' (line 385)
            xy_280206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'xy')
            # Obtaining the member 'T' of a type (line 385)
            T_280207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 19), xy_280206, 'T')
            # Obtaining the member '__getitem__' of a type (line 385)
            getitem___280208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 12), T_280207, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 385)
            subscript_call_result_280209 = invoke(stypy.reporting.localization.Localization(__file__, 385, 12), getitem___280208, int_280205)
            
            # Assigning a type to the variable 'tuple_var_assignment_279210' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'tuple_var_assignment_279210', subscript_call_result_280209)
            
            # Assigning a Name to a Name (line 385):
            # Getting the type of 'tuple_var_assignment_279209' (line 385)
            tuple_var_assignment_279209_280210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'tuple_var_assignment_279209')
            # Assigning a type to the variable 'x' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'x', tuple_var_assignment_279209_280210)
            
            # Assigning a Name to a Name (line 385):
            # Getting the type of 'tuple_var_assignment_279210' (line 385)
            tuple_var_assignment_279210_280211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'tuple_var_assignment_279210')
            # Assigning a type to the variable 'y' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 15), 'y', tuple_var_assignment_279210_280211)
            
            # Assigning a Call to a Name (line 386):
            
            # Assigning a Call to a Name (line 386):
            
            # Call to sqrt(...): (line 386)
            # Processing the call arguments (line 386)
            int_280214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 24), 'int')
            # Getting the type of 'x' (line 386)
            x_280215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 29), 'x', False)
            int_280216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 33), 'int')
            # Applying the binary operator 'div' (line 386)
            result_div_280217 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 29), 'div', x_280215, int_280216)
            
            int_280218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 39), 'int')
            # Applying the binary operator '**' (line 386)
            result_pow_280219 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 28), '**', result_div_280217, int_280218)
            
            # Applying the binary operator '-' (line 386)
            result_sub_280220 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 24), '-', int_280214, result_pow_280219)
            
            # Getting the type of 'y' (line 386)
            y_280221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 44), 'y', False)
            int_280222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 48), 'int')
            # Applying the binary operator 'div' (line 386)
            result_div_280223 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 44), 'div', y_280221, int_280222)
            
            int_280224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 54), 'int')
            # Applying the binary operator '**' (line 386)
            result_pow_280225 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 43), '**', result_div_280223, int_280224)
            
            # Applying the binary operator '-' (line 386)
            result_sub_280226 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 41), '-', result_sub_280220, result_pow_280225)
            
            # Processing the call keyword arguments (line 386)
            kwargs_280227 = {}
            # Getting the type of 'np' (line 386)
            np_280212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 386)
            sqrt_280213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 16), np_280212, 'sqrt')
            # Calling sqrt(args, kwargs) (line 386)
            sqrt_call_result_280228 = invoke(stypy.reporting.localization.Localization(__file__, 386, 16), sqrt_280213, *[result_sub_280226], **kwargs_280227)
            
            # Assigning a type to the variable 'z' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'z', sqrt_call_result_280228)
            
            # Assigning a BinOp to a Name (line 387):
            
            # Assigning a BinOp to a Name (line 387):
            int_280229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 24), 'int')
            
            # Call to arctan(...): (line 387)
            # Processing the call arguments (line 387)
            # Getting the type of 'z' (line 387)
            z_280232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 39), 'z', False)
            # Getting the type of 'x' (line 387)
            x_280233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 43), 'x', False)
            # Applying the binary operator '*' (line 387)
            result_mul_280234 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 39), '*', z_280232, x_280233)
            
            int_280235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 49), 'int')
            int_280236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 54), 'int')
            # Getting the type of 'z' (line 387)
            z_280237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 58), 'z', False)
            int_280238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 63), 'int')
            # Applying the binary operator '**' (line 387)
            result_pow_280239 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 58), '**', z_280237, int_280238)
            
            # Applying the binary operator '*' (line 387)
            result_mul_280240 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 54), '*', int_280236, result_pow_280239)
            
            int_280241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 67), 'int')
            # Applying the binary operator '-' (line 387)
            result_sub_280242 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 54), '-', result_mul_280240, int_280241)
            
            # Applying the binary operator '*' (line 387)
            result_mul_280243 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 49), '*', int_280235, result_sub_280242)
            
            # Applying the binary operator 'div' (line 387)
            result_div_280244 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 38), 'div', result_mul_280234, result_mul_280243)
            
            # Processing the call keyword arguments (line 387)
            kwargs_280245 = {}
            # Getting the type of 'np' (line 387)
            np_280230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 28), 'np', False)
            # Obtaining the member 'arctan' of a type (line 387)
            arctan_280231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 28), np_280230, 'arctan')
            # Calling arctan(args, kwargs) (line 387)
            arctan_call_result_280246 = invoke(stypy.reporting.localization.Localization(__file__, 387, 28), arctan_280231, *[result_div_280244], **kwargs_280245)
            
            # Applying the binary operator '*' (line 387)
            result_mul_280247 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 24), '*', int_280229, arctan_call_result_280246)
            
            # Assigning a type to the variable 'longitude' (line 387)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'longitude', result_mul_280247)
            
            # Assigning a Call to a Name (line 388):
            
            # Assigning a Call to a Name (line 388):
            
            # Call to arcsin(...): (line 388)
            # Processing the call arguments (line 388)
            # Getting the type of 'y' (line 388)
            y_280250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 33), 'y', False)
            # Getting the type of 'z' (line 388)
            z_280251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 35), 'z', False)
            # Applying the binary operator '*' (line 388)
            result_mul_280252 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 33), '*', y_280250, z_280251)
            
            # Processing the call keyword arguments (line 388)
            kwargs_280253 = {}
            # Getting the type of 'np' (line 388)
            np_280248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'np', False)
            # Obtaining the member 'arcsin' of a type (line 388)
            arcsin_280249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 23), np_280248, 'arcsin')
            # Calling arcsin(args, kwargs) (line 388)
            arcsin_call_result_280254 = invoke(stypy.reporting.localization.Localization(__file__, 388, 23), arcsin_280249, *[result_mul_280252], **kwargs_280253)
            
            # Assigning a type to the variable 'latitude' (line 388)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'latitude', arcsin_call_result_280254)
            
            # Call to column_stack(...): (line 389)
            # Processing the call arguments (line 389)
            
            # Obtaining an instance of the builtin type 'list' (line 389)
            list_280257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 35), 'list')
            # Adding type elements to the builtin type 'list' instance (line 389)
            # Adding element type (line 389)
            # Getting the type of 'longitude' (line 389)
            longitude_280258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 36), 'longitude', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_280257, longitude_280258)
            # Adding element type (line 389)
            # Getting the type of 'latitude' (line 389)
            latitude_280259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 47), 'latitude', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_280257, latitude_280259)
            
            # Processing the call keyword arguments (line 389)
            kwargs_280260 = {}
            # Getting the type of 'np' (line 389)
            np_280255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'np', False)
            # Obtaining the member 'column_stack' of a type (line 389)
            column_stack_280256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 19), np_280255, 'column_stack')
            # Calling column_stack(args, kwargs) (line 389)
            column_stack_call_result_280261 = invoke(stypy.reporting.localization.Localization(__file__, 389, 19), column_stack_280256, *[list_280257], **kwargs_280260)
            
            # Assigning a type to the variable 'stypy_return_type' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'stypy_return_type', column_stack_call_result_280261)
            
            # ################# End of 'transform_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 384)
            stypy_return_type_280262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280262)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_non_affine'
            return stypy_return_type_280262

        
        # Assigning a Attribute to a Attribute (line 390):
        
        # Assigning a Attribute to a Attribute (line 390):
        # Getting the type of 'Transform' (line 390)
        Transform_280263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 39), 'Transform')
        # Obtaining the member 'transform_non_affine' of a type (line 390)
        transform_non_affine_280264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 39), Transform_280263, 'transform_non_affine')
        # Obtaining the member '__doc__' of a type (line 390)
        doc___280265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 39), transform_non_affine_280264, '__doc__')
        # Getting the type of 'transform_non_affine' (line 390)
        transform_non_affine_280266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'transform_non_affine')
        # Setting the type of the member '__doc__' of a type (line 390)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), transform_non_affine_280266, '__doc__', doc___280265)

        @norecursion
        def inverted(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inverted'
            module_type_store = module_type_store.open_function_context('inverted', 392, 8, False)
            # Assigning a type to the variable 'self' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedHammerTransform.inverted')
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            InvertedHammerTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedHammerTransform.inverted', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inverted', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inverted(...)' code ##################

            
            # Call to HammerTransform(...): (line 393)
            # Processing the call arguments (line 393)
            # Getting the type of 'self' (line 393)
            self_280269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 46), 'self', False)
            # Obtaining the member '_resolution' of a type (line 393)
            _resolution_280270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 46), self_280269, '_resolution')
            # Processing the call keyword arguments (line 393)
            kwargs_280271 = {}
            # Getting the type of 'HammerAxes' (line 393)
            HammerAxes_280267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 19), 'HammerAxes', False)
            # Obtaining the member 'HammerTransform' of a type (line 393)
            HammerTransform_280268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 19), HammerAxes_280267, 'HammerTransform')
            # Calling HammerTransform(args, kwargs) (line 393)
            HammerTransform_call_result_280272 = invoke(stypy.reporting.localization.Localization(__file__, 393, 19), HammerTransform_280268, *[_resolution_280270], **kwargs_280271)
            
            # Assigning a type to the variable 'stypy_return_type' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'stypy_return_type', HammerTransform_call_result_280272)
            
            # ################# End of 'inverted(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inverted' in the type store
            # Getting the type of 'stypy_return_type' (line 392)
            stypy_return_type_280273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280273)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inverted'
            return stypy_return_type_280273

        
        # Assigning a Attribute to a Attribute (line 394):
        
        # Assigning a Attribute to a Attribute (line 394):
        # Getting the type of 'Transform' (line 394)
        Transform_280274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 27), 'Transform')
        # Obtaining the member 'inverted' of a type (line 394)
        inverted_280275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 27), Transform_280274, 'inverted')
        # Obtaining the member '__doc__' of a type (line 394)
        doc___280276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 27), inverted_280275, '__doc__')
        # Getting the type of 'inverted' (line 394)
        inverted_280277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'inverted')
        # Setting the type of the member '__doc__' of a type (line 394)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), inverted_280277, '__doc__', doc___280276)
    
    # Assigning a type to the variable 'InvertedHammerTransform' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'InvertedHammerTransform', InvertedHammerTransform)

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 396, 4, False)
        # Assigning a type to the variable 'self' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HammerAxes.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Attribute (line 397):
        
        # Assigning a BinOp to a Attribute (line 397):
        # Getting the type of 'np' (line 397)
        np_280278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 30), 'np')
        # Obtaining the member 'pi' of a type (line 397)
        pi_280279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 30), np_280278, 'pi')
        float_280280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 38), 'float')
        # Applying the binary operator 'div' (line 397)
        result_div_280281 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 30), 'div', pi_280279, float_280280)
        
        # Getting the type of 'self' (line 397)
        self_280282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self')
        # Setting the type of the member '_longitude_cap' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_280282, '_longitude_cap', result_div_280281)
        
        # Call to __init__(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'self' (line 398)
        self_280285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 25), 'self', False)
        # Getting the type of 'args' (line 398)
        args_280286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 32), 'args', False)
        # Processing the call keyword arguments (line 398)
        # Getting the type of 'kwargs' (line 398)
        kwargs_280287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 40), 'kwargs', False)
        kwargs_280288 = {'kwargs_280287': kwargs_280287}
        # Getting the type of 'GeoAxes' (line 398)
        GeoAxes_280283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'GeoAxes', False)
        # Obtaining the member '__init__' of a type (line 398)
        init___280284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), GeoAxes_280283, '__init__')
        # Calling __init__(args, kwargs) (line 398)
        init___call_result_280289 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), init___280284, *[self_280285, args_280286], **kwargs_280288)
        
        
        # Call to set_aspect(...): (line 399)
        # Processing the call arguments (line 399)
        float_280292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 24), 'float')
        # Processing the call keyword arguments (line 399)
        unicode_280293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 40), 'unicode', u'box')
        keyword_280294 = unicode_280293
        unicode_280295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 54), 'unicode', u'C')
        keyword_280296 = unicode_280295
        kwargs_280297 = {'adjustable': keyword_280294, 'anchor': keyword_280296}
        # Getting the type of 'self' (line 399)
        self_280290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self', False)
        # Obtaining the member 'set_aspect' of a type (line 399)
        set_aspect_280291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_280290, 'set_aspect')
        # Calling set_aspect(args, kwargs) (line 399)
        set_aspect_call_result_280298 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), set_aspect_280291, *[float_280292], **kwargs_280297)
        
        
        # Call to cla(...): (line 400)
        # Processing the call keyword arguments (line 400)
        kwargs_280301 = {}
        # Getting the type of 'self' (line 400)
        self_280299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'self', False)
        # Obtaining the member 'cla' of a type (line 400)
        cla_280300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), self_280299, 'cla')
        # Calling cla(args, kwargs) (line 400)
        cla_call_result_280302 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), cla_280300, *[], **kwargs_280301)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _get_core_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_core_transform'
        module_type_store = module_type_store.open_function_context('_get_core_transform', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_localization', localization)
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_function_name', 'HammerAxes._get_core_transform')
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_param_names_list', ['resolution'])
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HammerAxes._get_core_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HammerAxes._get_core_transform', ['resolution'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_core_transform', localization, ['resolution'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_core_transform(...)' code ##################

        
        # Call to HammerTransform(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'resolution' (line 403)
        resolution_280305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 36), 'resolution', False)
        # Processing the call keyword arguments (line 403)
        kwargs_280306 = {}
        # Getting the type of 'self' (line 403)
        self_280303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'self', False)
        # Obtaining the member 'HammerTransform' of a type (line 403)
        HammerTransform_280304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 15), self_280303, 'HammerTransform')
        # Calling HammerTransform(args, kwargs) (line 403)
        HammerTransform_call_result_280307 = invoke(stypy.reporting.localization.Localization(__file__, 403, 15), HammerTransform_280304, *[resolution_280305], **kwargs_280306)
        
        # Assigning a type to the variable 'stypy_return_type' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'stypy_return_type', HammerTransform_call_result_280307)
        
        # ################# End of '_get_core_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_core_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_280308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_280308)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_core_transform'
        return stypy_return_type_280308


# Assigning a type to the variable 'HammerAxes' (line 330)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 0), 'HammerAxes', HammerAxes)

# Assigning a Str to a Name (line 331):
unicode_280309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 11), 'unicode', u'hammer')
# Getting the type of 'HammerAxes'
HammerAxes_280310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HammerAxes')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HammerAxes_280310, 'name', unicode_280309)
# Declaration of the 'MollweideAxes' class
# Getting the type of 'GeoAxes' (line 406)
GeoAxes_280311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'GeoAxes')

class MollweideAxes(GeoAxes_280311, ):
    
    # Assigning a Str to a Name (line 407):
    # Declaration of the 'MollweideTransform' class
    # Getting the type of 'Transform' (line 409)
    Transform_280312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 29), 'Transform')

    class MollweideTransform(Transform_280312, ):
        unicode_280313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, (-1)), 'unicode', u'\n        The base Mollweide transform.\n        ')
        
        # Assigning a Num to a Name (line 413):
        
        # Assigning a Num to a Name (line 413):
        int_280314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 21), 'int')
        # Assigning a type to the variable 'input_dims' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'input_dims', int_280314)
        
        # Assigning a Num to a Name (line 414):
        
        # Assigning a Num to a Name (line 414):
        int_280315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 22), 'int')
        # Assigning a type to the variable 'output_dims' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'output_dims', int_280315)
        
        # Assigning a Name to a Name (line 415):
        
        # Assigning a Name to a Name (line 415):
        # Getting the type of 'False' (line 415)
        False_280316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 23), 'False')
        # Assigning a type to the variable 'is_separable' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'is_separable', False_280316)

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 417, 8, False)
            # Assigning a type to the variable 'self' (line 418)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'MollweideTransform.__init__', ['resolution'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['resolution'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            unicode_280317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, (-1)), 'unicode', u'\n            Create a new Mollweide transform.  Resolution is the number of steps\n            to interpolate between each input line segment to approximate its\n            path in curved Mollweide space.\n            ')
            
            # Call to __init__(...): (line 423)
            # Processing the call arguments (line 423)
            # Getting the type of 'self' (line 423)
            self_280320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 31), 'self', False)
            # Processing the call keyword arguments (line 423)
            kwargs_280321 = {}
            # Getting the type of 'Transform' (line 423)
            Transform_280318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'Transform', False)
            # Obtaining the member '__init__' of a type (line 423)
            init___280319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), Transform_280318, '__init__')
            # Calling __init__(args, kwargs) (line 423)
            init___call_result_280322 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), init___280319, *[self_280320], **kwargs_280321)
            
            
            # Assigning a Name to a Attribute (line 424):
            
            # Assigning a Name to a Attribute (line 424):
            # Getting the type of 'resolution' (line 424)
            resolution_280323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 31), 'resolution')
            # Getting the type of 'self' (line 424)
            self_280324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'self')
            # Setting the type of the member '_resolution' of a type (line 424)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), self_280324, '_resolution', resolution_280323)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_non_affine'
            module_type_store = module_type_store.open_function_context('transform_non_affine', 426, 8, False)
            # Assigning a type to the variable 'self' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'MollweideTransform.transform_non_affine')
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['ll'])
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            MollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'MollweideTransform.transform_non_affine', ['ll'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_non_affine', localization, ['ll'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_non_affine(...)' code ##################


            @norecursion
            def d(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'd'
                module_type_store = module_type_store.open_function_context('d', 427, 12, False)
                
                # Passed parameters checking function
                d.stypy_localization = localization
                d.stypy_type_of_self = None
                d.stypy_type_store = module_type_store
                d.stypy_function_name = 'd'
                d.stypy_param_names_list = ['theta']
                d.stypy_varargs_param_name = None
                d.stypy_kwargs_param_name = None
                d.stypy_call_defaults = defaults
                d.stypy_call_varargs = varargs
                d.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'd', ['theta'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'd', localization, ['theta'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'd(...)' code ##################

                
                # Assigning a BinOp to a Name (line 428):
                
                # Assigning a BinOp to a Name (line 428):
                
                # Getting the type of 'theta' (line 428)
                theta_280325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'theta')
                
                # Call to sin(...): (line 428)
                # Processing the call arguments (line 428)
                # Getting the type of 'theta' (line 428)
                theta_280328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 41), 'theta', False)
                # Processing the call keyword arguments (line 428)
                kwargs_280329 = {}
                # Getting the type of 'np' (line 428)
                np_280326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 34), 'np', False)
                # Obtaining the member 'sin' of a type (line 428)
                sin_280327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 34), np_280326, 'sin')
                # Calling sin(args, kwargs) (line 428)
                sin_call_result_280330 = invoke(stypy.reporting.localization.Localization(__file__, 428, 34), sin_280327, *[theta_280328], **kwargs_280329)
                
                # Applying the binary operator '+' (line 428)
                result_add_280331 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 26), '+', theta_280325, sin_call_result_280330)
                
                # Getting the type of 'pi_sin_l' (line 428)
                pi_sin_l_280332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 50), 'pi_sin_l')
                # Applying the binary operator '-' (line 428)
                result_sub_280333 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 48), '-', result_add_280331, pi_sin_l_280332)
                
                # Applying the 'usub' unary operator (line 428)
                result___neg___280334 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 24), 'usub', result_sub_280333)
                
                int_280335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 63), 'int')
                
                # Call to cos(...): (line 428)
                # Processing the call arguments (line 428)
                # Getting the type of 'theta' (line 428)
                theta_280338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 74), 'theta', False)
                # Processing the call keyword arguments (line 428)
                kwargs_280339 = {}
                # Getting the type of 'np' (line 428)
                np_280336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 67), 'np', False)
                # Obtaining the member 'cos' of a type (line 428)
                cos_280337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 67), np_280336, 'cos')
                # Calling cos(args, kwargs) (line 428)
                cos_call_result_280340 = invoke(stypy.reporting.localization.Localization(__file__, 428, 67), cos_280337, *[theta_280338], **kwargs_280339)
                
                # Applying the binary operator '+' (line 428)
                result_add_280341 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 63), '+', int_280335, cos_call_result_280340)
                
                # Applying the binary operator 'div' (line 428)
                result_div_280342 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 24), 'div', result___neg___280334, result_add_280341)
                
                # Assigning a type to the variable 'delta' (line 428)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 'delta', result_div_280342)
                
                # Obtaining an instance of the builtin type 'tuple' (line 429)
                tuple_280343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 429)
                # Adding element type (line 429)
                # Getting the type of 'delta' (line 429)
                delta_280344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 23), 'delta')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 23), tuple_280343, delta_280344)
                # Adding element type (line 429)
                
                
                # Call to abs(...): (line 429)
                # Processing the call arguments (line 429)
                # Getting the type of 'delta' (line 429)
                delta_280347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 37), 'delta', False)
                # Processing the call keyword arguments (line 429)
                kwargs_280348 = {}
                # Getting the type of 'np' (line 429)
                np_280345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 30), 'np', False)
                # Obtaining the member 'abs' of a type (line 429)
                abs_280346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 30), np_280345, 'abs')
                # Calling abs(args, kwargs) (line 429)
                abs_call_result_280349 = invoke(stypy.reporting.localization.Localization(__file__, 429, 30), abs_280346, *[delta_280347], **kwargs_280348)
                
                float_280350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 46), 'float')
                # Applying the binary operator '>' (line 429)
                result_gt_280351 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 30), '>', abs_call_result_280349, float_280350)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 23), tuple_280343, result_gt_280351)
                
                # Assigning a type to the variable 'stypy_return_type' (line 429)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'stypy_return_type', tuple_280343)
                
                # ################# End of 'd(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'd' in the type store
                # Getting the type of 'stypy_return_type' (line 427)
                stypy_return_type_280352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_280352)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'd'
                return stypy_return_type_280352

            # Assigning a type to the variable 'd' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'd', d)
            
            # Assigning a Subscript to a Name (line 431):
            
            # Assigning a Subscript to a Name (line 431):
            
            # Obtaining the type of the subscript
            slice_280353 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 431, 24), None, None, None)
            int_280354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 30), 'int')
            # Getting the type of 'll' (line 431)
            ll_280355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 24), 'll')
            # Obtaining the member '__getitem__' of a type (line 431)
            getitem___280356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 24), ll_280355, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 431)
            subscript_call_result_280357 = invoke(stypy.reporting.localization.Localization(__file__, 431, 24), getitem___280356, (slice_280353, int_280354))
            
            # Assigning a type to the variable 'longitude' (line 431)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'longitude', subscript_call_result_280357)
            
            # Assigning a Subscript to a Name (line 432):
            
            # Assigning a Subscript to a Name (line 432):
            
            # Obtaining the type of the subscript
            slice_280358 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 432, 24), None, None, None)
            int_280359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 30), 'int')
            # Getting the type of 'll' (line 432)
            ll_280360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'll')
            # Obtaining the member '__getitem__' of a type (line 432)
            getitem___280361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 24), ll_280360, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 432)
            subscript_call_result_280362 = invoke(stypy.reporting.localization.Localization(__file__, 432, 24), getitem___280361, (slice_280358, int_280359))
            
            # Assigning a type to the variable 'latitude' (line 432)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'latitude', subscript_call_result_280362)
            
            # Assigning a BinOp to a Name (line 434):
            
            # Assigning a BinOp to a Name (line 434):
            # Getting the type of 'np' (line 434)
            np_280363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'np')
            # Obtaining the member 'pi' of a type (line 434)
            pi_280364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 19), np_280363, 'pi')
            int_280365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 25), 'int')
            # Applying the binary operator 'div' (line 434)
            result_div_280366 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 19), 'div', pi_280364, int_280365)
            
            
            # Call to abs(...): (line 434)
            # Processing the call arguments (line 434)
            # Getting the type of 'latitude' (line 434)
            latitude_280369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 36), 'latitude', False)
            # Processing the call keyword arguments (line 434)
            kwargs_280370 = {}
            # Getting the type of 'np' (line 434)
            np_280367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 29), 'np', False)
            # Obtaining the member 'abs' of a type (line 434)
            abs_280368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 29), np_280367, 'abs')
            # Calling abs(args, kwargs) (line 434)
            abs_call_result_280371 = invoke(stypy.reporting.localization.Localization(__file__, 434, 29), abs_280368, *[latitude_280369], **kwargs_280370)
            
            # Applying the binary operator '-' (line 434)
            result_sub_280372 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 19), '-', result_div_280366, abs_call_result_280371)
            
            # Assigning a type to the variable 'clat' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'clat', result_sub_280372)
            
            # Assigning a Compare to a Name (line 435):
            
            # Assigning a Compare to a Name (line 435):
            
            # Getting the type of 'clat' (line 435)
            clat_280373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'clat')
            float_280374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 27), 'float')
            # Applying the binary operator '<' (line 435)
            result_lt_280375 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 20), '<', clat_280373, float_280374)
            
            # Assigning a type to the variable 'ihigh' (line 435)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'ihigh', result_lt_280375)
            
            # Assigning a UnaryOp to a Name (line 436):
            
            # Assigning a UnaryOp to a Name (line 436):
            
            # Getting the type of 'ihigh' (line 436)
            ihigh_280376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'ihigh')
            # Applying the '~' unary operator (line 436)
            result_inv_280377 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 19), '~', ihigh_280376)
            
            # Assigning a type to the variable 'ilow' (line 436)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'ilow', result_inv_280377)
            
            # Assigning a Call to a Name (line 437):
            
            # Assigning a Call to a Name (line 437):
            
            # Call to empty(...): (line 437)
            # Processing the call arguments (line 437)
            # Getting the type of 'latitude' (line 437)
            latitude_280380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 27), 'latitude', False)
            # Obtaining the member 'shape' of a type (line 437)
            shape_280381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 27), latitude_280380, 'shape')
            # Processing the call keyword arguments (line 437)
            # Getting the type of 'float' (line 437)
            float_280382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 49), 'float', False)
            keyword_280383 = float_280382
            kwargs_280384 = {'dtype': keyword_280383}
            # Getting the type of 'np' (line 437)
            np_280378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 18), 'np', False)
            # Obtaining the member 'empty' of a type (line 437)
            empty_280379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 18), np_280378, 'empty')
            # Calling empty(args, kwargs) (line 437)
            empty_call_result_280385 = invoke(stypy.reporting.localization.Localization(__file__, 437, 18), empty_280379, *[shape_280381], **kwargs_280384)
            
            # Assigning a type to the variable 'aux' (line 437)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'aux', empty_call_result_280385)
            
            
            # Call to any(...): (line 439)
            # Processing the call keyword arguments (line 439)
            kwargs_280388 = {}
            # Getting the type of 'ilow' (line 439)
            ilow_280386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 15), 'ilow', False)
            # Obtaining the member 'any' of a type (line 439)
            any_280387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 15), ilow_280386, 'any')
            # Calling any(args, kwargs) (line 439)
            any_call_result_280389 = invoke(stypy.reporting.localization.Localization(__file__, 439, 15), any_280387, *[], **kwargs_280388)
            
            # Testing the type of an if condition (line 439)
            if_condition_280390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 12), any_call_result_280389)
            # Assigning a type to the variable 'if_condition_280390' (line 439)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'if_condition_280390', if_condition_280390)
            # SSA begins for if statement (line 439)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 440):
            
            # Assigning a BinOp to a Name (line 440):
            # Getting the type of 'np' (line 440)
            np_280391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 27), 'np')
            # Obtaining the member 'pi' of a type (line 440)
            pi_280392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 27), np_280391, 'pi')
            
            # Call to sin(...): (line 440)
            # Processing the call arguments (line 440)
            
            # Obtaining the type of the subscript
            # Getting the type of 'ilow' (line 440)
            ilow_280395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 51), 'ilow', False)
            # Getting the type of 'latitude' (line 440)
            latitude_280396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 42), 'latitude', False)
            # Obtaining the member '__getitem__' of a type (line 440)
            getitem___280397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 42), latitude_280396, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 440)
            subscript_call_result_280398 = invoke(stypy.reporting.localization.Localization(__file__, 440, 42), getitem___280397, ilow_280395)
            
            # Processing the call keyword arguments (line 440)
            kwargs_280399 = {}
            # Getting the type of 'np' (line 440)
            np_280393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 35), 'np', False)
            # Obtaining the member 'sin' of a type (line 440)
            sin_280394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 35), np_280393, 'sin')
            # Calling sin(args, kwargs) (line 440)
            sin_call_result_280400 = invoke(stypy.reporting.localization.Localization(__file__, 440, 35), sin_280394, *[subscript_call_result_280398], **kwargs_280399)
            
            # Applying the binary operator '*' (line 440)
            result_mul_280401 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 27), '*', pi_280392, sin_call_result_280400)
            
            # Assigning a type to the variable 'pi_sin_l' (line 440)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'pi_sin_l', result_mul_280401)
            
            # Assigning a BinOp to a Name (line 441):
            
            # Assigning a BinOp to a Name (line 441):
            float_280402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 24), 'float')
            
            # Obtaining the type of the subscript
            # Getting the type of 'ilow' (line 441)
            ilow_280403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 39), 'ilow')
            # Getting the type of 'latitude' (line 441)
            latitude_280404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 'latitude')
            # Obtaining the member '__getitem__' of a type (line 441)
            getitem___280405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 30), latitude_280404, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 441)
            subscript_call_result_280406 = invoke(stypy.reporting.localization.Localization(__file__, 441, 30), getitem___280405, ilow_280403)
            
            # Applying the binary operator '*' (line 441)
            result_mul_280407 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 24), '*', float_280402, subscript_call_result_280406)
            
            # Assigning a type to the variable 'theta' (line 441)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'theta', result_mul_280407)
            
            # Assigning a Call to a Tuple (line 442):
            
            # Assigning a Call to a Name:
            
            # Call to d(...): (line 442)
            # Processing the call arguments (line 442)
            # Getting the type of 'theta' (line 442)
            theta_280409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 39), 'theta', False)
            # Processing the call keyword arguments (line 442)
            kwargs_280410 = {}
            # Getting the type of 'd' (line 442)
            d_280408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 37), 'd', False)
            # Calling d(args, kwargs) (line 442)
            d_call_result_280411 = invoke(stypy.reporting.localization.Localization(__file__, 442, 37), d_280408, *[theta_280409], **kwargs_280410)
            
            # Assigning a type to the variable 'call_assignment_279211' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'call_assignment_279211', d_call_result_280411)
            
            # Assigning a Call to a Name (line 442):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_280414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 16), 'int')
            # Processing the call keyword arguments
            kwargs_280415 = {}
            # Getting the type of 'call_assignment_279211' (line 442)
            call_assignment_279211_280412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'call_assignment_279211', False)
            # Obtaining the member '__getitem__' of a type (line 442)
            getitem___280413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 16), call_assignment_279211_280412, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_280416 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___280413, *[int_280414], **kwargs_280415)
            
            # Assigning a type to the variable 'call_assignment_279212' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'call_assignment_279212', getitem___call_result_280416)
            
            # Assigning a Name to a Name (line 442):
            # Getting the type of 'call_assignment_279212' (line 442)
            call_assignment_279212_280417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'call_assignment_279212')
            # Assigning a type to the variable 'delta' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'delta', call_assignment_279212_280417)
            
            # Assigning a Call to a Name (line 442):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_280420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 16), 'int')
            # Processing the call keyword arguments
            kwargs_280421 = {}
            # Getting the type of 'call_assignment_279211' (line 442)
            call_assignment_279211_280418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'call_assignment_279211', False)
            # Obtaining the member '__getitem__' of a type (line 442)
            getitem___280419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 16), call_assignment_279211_280418, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_280422 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___280419, *[int_280420], **kwargs_280421)
            
            # Assigning a type to the variable 'call_assignment_279213' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'call_assignment_279213', getitem___call_result_280422)
            
            # Assigning a Name to a Name (line 442):
            # Getting the type of 'call_assignment_279213' (line 442)
            call_assignment_279213_280423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'call_assignment_279213')
            # Assigning a type to the variable 'large_delta' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'large_delta', call_assignment_279213_280423)
            
            
            # Call to any(...): (line 443)
            # Processing the call arguments (line 443)
            # Getting the type of 'large_delta' (line 443)
            large_delta_280426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 29), 'large_delta', False)
            # Processing the call keyword arguments (line 443)
            kwargs_280427 = {}
            # Getting the type of 'np' (line 443)
            np_280424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 22), 'np', False)
            # Obtaining the member 'any' of a type (line 443)
            any_280425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 22), np_280424, 'any')
            # Calling any(args, kwargs) (line 443)
            any_call_result_280428 = invoke(stypy.reporting.localization.Localization(__file__, 443, 22), any_280425, *[large_delta_280426], **kwargs_280427)
            
            # Testing the type of an if condition (line 443)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 16), any_call_result_280428)
            # SSA begins for while statement (line 443)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Getting the type of 'theta' (line 444)
            theta_280429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'theta')
            
            # Obtaining the type of the subscript
            # Getting the type of 'large_delta' (line 444)
            large_delta_280430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 26), 'large_delta')
            # Getting the type of 'theta' (line 444)
            theta_280431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'theta')
            # Obtaining the member '__getitem__' of a type (line 444)
            getitem___280432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 20), theta_280431, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 444)
            subscript_call_result_280433 = invoke(stypy.reporting.localization.Localization(__file__, 444, 20), getitem___280432, large_delta_280430)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'large_delta' (line 444)
            large_delta_280434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 48), 'large_delta')
            # Getting the type of 'delta' (line 444)
            delta_280435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 42), 'delta')
            # Obtaining the member '__getitem__' of a type (line 444)
            getitem___280436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 42), delta_280435, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 444)
            subscript_call_result_280437 = invoke(stypy.reporting.localization.Localization(__file__, 444, 42), getitem___280436, large_delta_280434)
            
            # Applying the binary operator '+=' (line 444)
            result_iadd_280438 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), '+=', subscript_call_result_280433, subscript_call_result_280437)
            # Getting the type of 'theta' (line 444)
            theta_280439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'theta')
            # Getting the type of 'large_delta' (line 444)
            large_delta_280440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 26), 'large_delta')
            # Storing an element on a container (line 444)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 20), theta_280439, (large_delta_280440, result_iadd_280438))
            
            
            # Assigning a Call to a Tuple (line 445):
            
            # Assigning a Call to a Name:
            
            # Call to d(...): (line 445)
            # Processing the call arguments (line 445)
            # Getting the type of 'theta' (line 445)
            theta_280442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 43), 'theta', False)
            # Processing the call keyword arguments (line 445)
            kwargs_280443 = {}
            # Getting the type of 'd' (line 445)
            d_280441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 41), 'd', False)
            # Calling d(args, kwargs) (line 445)
            d_call_result_280444 = invoke(stypy.reporting.localization.Localization(__file__, 445, 41), d_280441, *[theta_280442], **kwargs_280443)
            
            # Assigning a type to the variable 'call_assignment_279214' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'call_assignment_279214', d_call_result_280444)
            
            # Assigning a Call to a Name (line 445):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_280447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 20), 'int')
            # Processing the call keyword arguments
            kwargs_280448 = {}
            # Getting the type of 'call_assignment_279214' (line 445)
            call_assignment_279214_280445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'call_assignment_279214', False)
            # Obtaining the member '__getitem__' of a type (line 445)
            getitem___280446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 20), call_assignment_279214_280445, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_280449 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___280446, *[int_280447], **kwargs_280448)
            
            # Assigning a type to the variable 'call_assignment_279215' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'call_assignment_279215', getitem___call_result_280449)
            
            # Assigning a Name to a Name (line 445):
            # Getting the type of 'call_assignment_279215' (line 445)
            call_assignment_279215_280450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'call_assignment_279215')
            # Assigning a type to the variable 'delta' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'delta', call_assignment_279215_280450)
            
            # Assigning a Call to a Name (line 445):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_280453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 20), 'int')
            # Processing the call keyword arguments
            kwargs_280454 = {}
            # Getting the type of 'call_assignment_279214' (line 445)
            call_assignment_279214_280451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'call_assignment_279214', False)
            # Obtaining the member '__getitem__' of a type (line 445)
            getitem___280452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 20), call_assignment_279214_280451, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_280455 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___280452, *[int_280453], **kwargs_280454)
            
            # Assigning a type to the variable 'call_assignment_279216' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'call_assignment_279216', getitem___call_result_280455)
            
            # Assigning a Name to a Name (line 445):
            # Getting the type of 'call_assignment_279216' (line 445)
            call_assignment_279216_280456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'call_assignment_279216')
            # Assigning a type to the variable 'large_delta' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 27), 'large_delta', call_assignment_279216_280456)
            # SSA join for while statement (line 443)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Subscript (line 446):
            
            # Assigning a BinOp to a Subscript (line 446):
            # Getting the type of 'theta' (line 446)
            theta_280457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 28), 'theta')
            int_280458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 36), 'int')
            # Applying the binary operator 'div' (line 446)
            result_div_280459 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 28), 'div', theta_280457, int_280458)
            
            # Getting the type of 'aux' (line 446)
            aux_280460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'aux')
            # Getting the type of 'ilow' (line 446)
            ilow_280461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 20), 'ilow')
            # Storing an element on a container (line 446)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 16), aux_280460, (ilow_280461, result_div_280459))
            # SSA join for if statement (line 439)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Call to any(...): (line 448)
            # Processing the call keyword arguments (line 448)
            kwargs_280464 = {}
            # Getting the type of 'ihigh' (line 448)
            ihigh_280462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'ihigh', False)
            # Obtaining the member 'any' of a type (line 448)
            any_280463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 15), ihigh_280462, 'any')
            # Calling any(args, kwargs) (line 448)
            any_call_result_280465 = invoke(stypy.reporting.localization.Localization(__file__, 448, 15), any_280463, *[], **kwargs_280464)
            
            # Testing the type of an if condition (line 448)
            if_condition_280466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 12), any_call_result_280465)
            # Assigning a type to the variable 'if_condition_280466' (line 448)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'if_condition_280466', if_condition_280466)
            # SSA begins for if statement (line 448)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 449):
            
            # Assigning a Subscript to a Name (line 449):
            
            # Obtaining the type of the subscript
            # Getting the type of 'ihigh' (line 449)
            ihigh_280467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 25), 'ihigh')
            # Getting the type of 'clat' (line 449)
            clat_280468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 20), 'clat')
            # Obtaining the member '__getitem__' of a type (line 449)
            getitem___280469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 20), clat_280468, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 449)
            subscript_call_result_280470 = invoke(stypy.reporting.localization.Localization(__file__, 449, 20), getitem___280469, ihigh_280467)
            
            # Assigning a type to the variable 'e' (line 449)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 16), 'e', subscript_call_result_280470)
            
            # Assigning a BinOp to a Name (line 450):
            
            # Assigning a BinOp to a Name (line 450):
            float_280471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 20), 'float')
            int_280472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 27), 'int')
            # Getting the type of 'np' (line 450)
            np_280473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 31), 'np')
            # Obtaining the member 'pi' of a type (line 450)
            pi_280474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 31), np_280473, 'pi')
            # Applying the binary operator '*' (line 450)
            result_mul_280475 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 27), '*', int_280472, pi_280474)
            
            # Getting the type of 'e' (line 450)
            e_280476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 39), 'e')
            int_280477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 42), 'int')
            # Applying the binary operator '**' (line 450)
            result_pow_280478 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 39), '**', e_280476, int_280477)
            
            # Applying the binary operator '*' (line 450)
            result_mul_280479 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 37), '*', result_mul_280475, result_pow_280478)
            
            float_280480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 49), 'float')
            int_280481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 53), 'int')
            # Applying the binary operator 'div' (line 450)
            result_div_280482 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 49), 'div', float_280480, int_280481)
            
            # Applying the binary operator '**' (line 450)
            result_pow_280483 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 26), '**', result_mul_280479, result_div_280482)
            
            # Applying the binary operator '*' (line 450)
            result_mul_280484 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 20), '*', float_280471, result_pow_280483)
            
            # Assigning a type to the variable 'd' (line 450)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'd', result_mul_280484)
            
            # Assigning a BinOp to a Subscript (line 451):
            
            # Assigning a BinOp to a Subscript (line 451):
            # Getting the type of 'np' (line 451)
            np_280485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 30), 'np')
            # Obtaining the member 'pi' of a type (line 451)
            pi_280486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 30), np_280485, 'pi')
            int_280487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 36), 'int')
            # Applying the binary operator 'div' (line 451)
            result_div_280488 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 30), 'div', pi_280486, int_280487)
            
            # Getting the type of 'd' (line 451)
            d_280489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 40), 'd')
            # Applying the binary operator '-' (line 451)
            result_sub_280490 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 30), '-', result_div_280488, d_280489)
            
            
            # Call to sign(...): (line 451)
            # Processing the call arguments (line 451)
            
            # Obtaining the type of the subscript
            # Getting the type of 'ihigh' (line 451)
            ihigh_280493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 62), 'ihigh', False)
            # Getting the type of 'latitude' (line 451)
            latitude_280494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 53), 'latitude', False)
            # Obtaining the member '__getitem__' of a type (line 451)
            getitem___280495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 53), latitude_280494, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 451)
            subscript_call_result_280496 = invoke(stypy.reporting.localization.Localization(__file__, 451, 53), getitem___280495, ihigh_280493)
            
            # Processing the call keyword arguments (line 451)
            kwargs_280497 = {}
            # Getting the type of 'np' (line 451)
            np_280491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 45), 'np', False)
            # Obtaining the member 'sign' of a type (line 451)
            sign_280492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 45), np_280491, 'sign')
            # Calling sign(args, kwargs) (line 451)
            sign_call_result_280498 = invoke(stypy.reporting.localization.Localization(__file__, 451, 45), sign_280492, *[subscript_call_result_280496], **kwargs_280497)
            
            # Applying the binary operator '*' (line 451)
            result_mul_280499 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 29), '*', result_sub_280490, sign_call_result_280498)
            
            # Getting the type of 'aux' (line 451)
            aux_280500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'aux')
            # Getting the type of 'ihigh' (line 451)
            ihigh_280501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'ihigh')
            # Storing an element on a container (line 451)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 16), aux_280500, (ihigh_280501, result_mul_280499))
            # SSA join for if statement (line 448)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 453):
            
            # Assigning a Call to a Name (line 453):
            
            # Call to empty(...): (line 453)
            # Processing the call arguments (line 453)
            # Getting the type of 'll' (line 453)
            ll_280504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 26), 'll', False)
            # Obtaining the member 'shape' of a type (line 453)
            shape_280505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 26), ll_280504, 'shape')
            # Processing the call keyword arguments (line 453)
            # Getting the type of 'float' (line 453)
            float_280506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 42), 'float', False)
            keyword_280507 = float_280506
            kwargs_280508 = {'dtype': keyword_280507}
            # Getting the type of 'np' (line 453)
            np_280502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 17), 'np', False)
            # Obtaining the member 'empty' of a type (line 453)
            empty_280503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 17), np_280502, 'empty')
            # Calling empty(args, kwargs) (line 453)
            empty_call_result_280509 = invoke(stypy.reporting.localization.Localization(__file__, 453, 17), empty_280503, *[shape_280505], **kwargs_280508)
            
            # Assigning a type to the variable 'xy' (line 453)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'xy', empty_call_result_280509)
            
            # Assigning a BinOp to a Subscript (line 454):
            
            # Assigning a BinOp to a Subscript (line 454):
            float_280510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 23), 'float')
            
            # Call to sqrt(...): (line 454)
            # Processing the call arguments (line 454)
            float_280513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 37), 'float')
            # Processing the call keyword arguments (line 454)
            kwargs_280514 = {}
            # Getting the type of 'np' (line 454)
            np_280511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 29), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 454)
            sqrt_280512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 29), np_280511, 'sqrt')
            # Calling sqrt(args, kwargs) (line 454)
            sqrt_call_result_280515 = invoke(stypy.reporting.localization.Localization(__file__, 454, 29), sqrt_280512, *[float_280513], **kwargs_280514)
            
            # Applying the binary operator '*' (line 454)
            result_mul_280516 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 23), '*', float_280510, sqrt_call_result_280515)
            
            # Getting the type of 'np' (line 454)
            np_280517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 44), 'np')
            # Obtaining the member 'pi' of a type (line 454)
            pi_280518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 44), np_280517, 'pi')
            # Applying the binary operator 'div' (line 454)
            result_div_280519 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 42), 'div', result_mul_280516, pi_280518)
            
            # Getting the type of 'longitude' (line 454)
            longitude_280520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 53), 'longitude')
            # Applying the binary operator '*' (line 454)
            result_mul_280521 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 22), '*', result_div_280519, longitude_280520)
            
            
            # Call to cos(...): (line 454)
            # Processing the call arguments (line 454)
            # Getting the type of 'aux' (line 454)
            aux_280524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 72), 'aux', False)
            # Processing the call keyword arguments (line 454)
            kwargs_280525 = {}
            # Getting the type of 'np' (line 454)
            np_280522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 65), 'np', False)
            # Obtaining the member 'cos' of a type (line 454)
            cos_280523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 65), np_280522, 'cos')
            # Calling cos(args, kwargs) (line 454)
            cos_call_result_280526 = invoke(stypy.reporting.localization.Localization(__file__, 454, 65), cos_280523, *[aux_280524], **kwargs_280525)
            
            # Applying the binary operator '*' (line 454)
            result_mul_280527 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 63), '*', result_mul_280521, cos_call_result_280526)
            
            # Getting the type of 'xy' (line 454)
            xy_280528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'xy')
            slice_280529 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 454, 12), None, None, None)
            int_280530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 17), 'int')
            # Storing an element on a container (line 454)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 12), xy_280528, ((slice_280529, int_280530), result_mul_280527))
            
            # Assigning a BinOp to a Subscript (line 455):
            
            # Assigning a BinOp to a Subscript (line 455):
            
            # Call to sqrt(...): (line 455)
            # Processing the call arguments (line 455)
            float_280533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 30), 'float')
            # Processing the call keyword arguments (line 455)
            kwargs_280534 = {}
            # Getting the type of 'np' (line 455)
            np_280531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 22), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 455)
            sqrt_280532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 22), np_280531, 'sqrt')
            # Calling sqrt(args, kwargs) (line 455)
            sqrt_call_result_280535 = invoke(stypy.reporting.localization.Localization(__file__, 455, 22), sqrt_280532, *[float_280533], **kwargs_280534)
            
            
            # Call to sin(...): (line 455)
            # Processing the call arguments (line 455)
            # Getting the type of 'aux' (line 455)
            aux_280538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 44), 'aux', False)
            # Processing the call keyword arguments (line 455)
            kwargs_280539 = {}
            # Getting the type of 'np' (line 455)
            np_280536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 37), 'np', False)
            # Obtaining the member 'sin' of a type (line 455)
            sin_280537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 37), np_280536, 'sin')
            # Calling sin(args, kwargs) (line 455)
            sin_call_result_280540 = invoke(stypy.reporting.localization.Localization(__file__, 455, 37), sin_280537, *[aux_280538], **kwargs_280539)
            
            # Applying the binary operator '*' (line 455)
            result_mul_280541 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 22), '*', sqrt_call_result_280535, sin_call_result_280540)
            
            # Getting the type of 'xy' (line 455)
            xy_280542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'xy')
            slice_280543 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 455, 12), None, None, None)
            int_280544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 17), 'int')
            # Storing an element on a container (line 455)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), xy_280542, ((slice_280543, int_280544), result_mul_280541))
            # Getting the type of 'xy' (line 457)
            xy_280545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'xy')
            # Assigning a type to the variable 'stypy_return_type' (line 457)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'stypy_return_type', xy_280545)
            
            # ################# End of 'transform_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 426)
            stypy_return_type_280546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280546)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_non_affine'
            return stypy_return_type_280546

        
        # Assigning a Attribute to a Attribute (line 458):
        
        # Assigning a Attribute to a Attribute (line 458):
        # Getting the type of 'Transform' (line 458)
        Transform_280547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 39), 'Transform')
        # Obtaining the member 'transform_non_affine' of a type (line 458)
        transform_non_affine_280548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 39), Transform_280547, 'transform_non_affine')
        # Obtaining the member '__doc__' of a type (line 458)
        doc___280549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 39), transform_non_affine_280548, '__doc__')
        # Getting the type of 'transform_non_affine' (line 458)
        transform_non_affine_280550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'transform_non_affine')
        # Setting the type of the member '__doc__' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), transform_non_affine_280550, '__doc__', doc___280549)

        @norecursion
        def transform_path_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_path_non_affine'
            module_type_store = module_type_store.open_function_context('transform_path_non_affine', 460, 8, False)
            # Assigning a type to the variable 'self' (line 461)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_localization', localization)
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_function_name', 'MollweideTransform.transform_path_non_affine')
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_param_names_list', ['path'])
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            MollweideTransform.transform_path_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'MollweideTransform.transform_path_non_affine', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_path_non_affine', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_path_non_affine(...)' code ##################

            
            # Assigning a Attribute to a Name (line 461):
            
            # Assigning a Attribute to a Name (line 461):
            # Getting the type of 'path' (line 461)
            path_280551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 23), 'path')
            # Obtaining the member 'vertices' of a type (line 461)
            vertices_280552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 23), path_280551, 'vertices')
            # Assigning a type to the variable 'vertices' (line 461)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'vertices', vertices_280552)
            
            # Assigning a Call to a Name (line 462):
            
            # Assigning a Call to a Name (line 462):
            
            # Call to interpolated(...): (line 462)
            # Processing the call arguments (line 462)
            # Getting the type of 'self' (line 462)
            self_280555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'self', False)
            # Obtaining the member '_resolution' of a type (line 462)
            _resolution_280556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 38), self_280555, '_resolution')
            # Processing the call keyword arguments (line 462)
            kwargs_280557 = {}
            # Getting the type of 'path' (line 462)
            path_280553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'path', False)
            # Obtaining the member 'interpolated' of a type (line 462)
            interpolated_280554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 20), path_280553, 'interpolated')
            # Calling interpolated(args, kwargs) (line 462)
            interpolated_call_result_280558 = invoke(stypy.reporting.localization.Localization(__file__, 462, 20), interpolated_280554, *[_resolution_280556], **kwargs_280557)
            
            # Assigning a type to the variable 'ipath' (line 462)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'ipath', interpolated_call_result_280558)
            
            # Call to Path(...): (line 463)
            # Processing the call arguments (line 463)
            
            # Call to transform(...): (line 463)
            # Processing the call arguments (line 463)
            # Getting the type of 'ipath' (line 463)
            ipath_280562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 39), 'ipath', False)
            # Obtaining the member 'vertices' of a type (line 463)
            vertices_280563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 39), ipath_280562, 'vertices')
            # Processing the call keyword arguments (line 463)
            kwargs_280564 = {}
            # Getting the type of 'self' (line 463)
            self_280560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'self', False)
            # Obtaining the member 'transform' of a type (line 463)
            transform_280561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), self_280560, 'transform')
            # Calling transform(args, kwargs) (line 463)
            transform_call_result_280565 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), transform_280561, *[vertices_280563], **kwargs_280564)
            
            # Getting the type of 'ipath' (line 463)
            ipath_280566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 56), 'ipath', False)
            # Obtaining the member 'codes' of a type (line 463)
            codes_280567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 56), ipath_280566, 'codes')
            # Processing the call keyword arguments (line 463)
            kwargs_280568 = {}
            # Getting the type of 'Path' (line 463)
            Path_280559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 463)
            Path_call_result_280569 = invoke(stypy.reporting.localization.Localization(__file__, 463, 19), Path_280559, *[transform_call_result_280565, codes_280567], **kwargs_280568)
            
            # Assigning a type to the variable 'stypy_return_type' (line 463)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'stypy_return_type', Path_call_result_280569)
            
            # ################# End of 'transform_path_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_path_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 460)
            stypy_return_type_280570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280570)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_path_non_affine'
            return stypy_return_type_280570

        
        # Assigning a Attribute to a Attribute (line 464):
        
        # Assigning a Attribute to a Attribute (line 464):
        # Getting the type of 'Transform' (line 464)
        Transform_280571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 44), 'Transform')
        # Obtaining the member 'transform_path_non_affine' of a type (line 464)
        transform_path_non_affine_280572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 44), Transform_280571, 'transform_path_non_affine')
        # Obtaining the member '__doc__' of a type (line 464)
        doc___280573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 44), transform_path_non_affine_280572, '__doc__')
        # Getting the type of 'transform_path_non_affine' (line 464)
        transform_path_non_affine_280574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'transform_path_non_affine')
        # Setting the type of the member '__doc__' of a type (line 464)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 8), transform_path_non_affine_280574, '__doc__', doc___280573)

        @norecursion
        def inverted(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inverted'
            module_type_store = module_type_store.open_function_context('inverted', 466, 8, False)
            # Assigning a type to the variable 'self' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            MollweideTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
            MollweideTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            MollweideTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
            MollweideTransform.inverted.__dict__.__setitem__('stypy_function_name', 'MollweideTransform.inverted')
            MollweideTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
            MollweideTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
            MollweideTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
            MollweideTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
            MollweideTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
            MollweideTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            MollweideTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'MollweideTransform.inverted', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inverted', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inverted(...)' code ##################

            
            # Call to InvertedMollweideTransform(...): (line 467)
            # Processing the call arguments (line 467)
            # Getting the type of 'self' (line 467)
            self_280577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 60), 'self', False)
            # Obtaining the member '_resolution' of a type (line 467)
            _resolution_280578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 60), self_280577, '_resolution')
            # Processing the call keyword arguments (line 467)
            kwargs_280579 = {}
            # Getting the type of 'MollweideAxes' (line 467)
            MollweideAxes_280575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 19), 'MollweideAxes', False)
            # Obtaining the member 'InvertedMollweideTransform' of a type (line 467)
            InvertedMollweideTransform_280576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 19), MollweideAxes_280575, 'InvertedMollweideTransform')
            # Calling InvertedMollweideTransform(args, kwargs) (line 467)
            InvertedMollweideTransform_call_result_280580 = invoke(stypy.reporting.localization.Localization(__file__, 467, 19), InvertedMollweideTransform_280576, *[_resolution_280578], **kwargs_280579)
            
            # Assigning a type to the variable 'stypy_return_type' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'stypy_return_type', InvertedMollweideTransform_call_result_280580)
            
            # ################# End of 'inverted(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inverted' in the type store
            # Getting the type of 'stypy_return_type' (line 466)
            stypy_return_type_280581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280581)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inverted'
            return stypy_return_type_280581

        
        # Assigning a Attribute to a Attribute (line 468):
        
        # Assigning a Attribute to a Attribute (line 468):
        # Getting the type of 'Transform' (line 468)
        Transform_280582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 27), 'Transform')
        # Obtaining the member 'inverted' of a type (line 468)
        inverted_280583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 27), Transform_280582, 'inverted')
        # Obtaining the member '__doc__' of a type (line 468)
        doc___280584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 27), inverted_280583, '__doc__')
        # Getting the type of 'inverted' (line 468)
        inverted_280585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'inverted')
        # Setting the type of the member '__doc__' of a type (line 468)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), inverted_280585, '__doc__', doc___280584)
    
    # Assigning a type to the variable 'MollweideTransform' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'MollweideTransform', MollweideTransform)
    # Declaration of the 'InvertedMollweideTransform' class
    # Getting the type of 'Transform' (line 470)
    Transform_280586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 37), 'Transform')

    class InvertedMollweideTransform(Transform_280586, ):
        
        # Assigning a Num to a Name (line 471):
        
        # Assigning a Num to a Name (line 471):
        int_280587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 21), 'int')
        # Assigning a type to the variable 'input_dims' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'input_dims', int_280587)
        
        # Assigning a Num to a Name (line 472):
        
        # Assigning a Num to a Name (line 472):
        int_280588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 22), 'int')
        # Assigning a type to the variable 'output_dims' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'output_dims', int_280588)
        
        # Assigning a Name to a Name (line 473):
        
        # Assigning a Name to a Name (line 473):
        # Getting the type of 'False' (line 473)
        False_280589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 23), 'False')
        # Assigning a type to the variable 'is_separable' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'is_separable', False_280589)

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 475, 8, False)
            # Assigning a type to the variable 'self' (line 476)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedMollweideTransform.__init__', ['resolution'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['resolution'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Call to __init__(...): (line 476)
            # Processing the call arguments (line 476)
            # Getting the type of 'self' (line 476)
            self_280592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 31), 'self', False)
            # Processing the call keyword arguments (line 476)
            kwargs_280593 = {}
            # Getting the type of 'Transform' (line 476)
            Transform_280590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'Transform', False)
            # Obtaining the member '__init__' of a type (line 476)
            init___280591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), Transform_280590, '__init__')
            # Calling __init__(args, kwargs) (line 476)
            init___call_result_280594 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), init___280591, *[self_280592], **kwargs_280593)
            
            
            # Assigning a Name to a Attribute (line 477):
            
            # Assigning a Name to a Attribute (line 477):
            # Getting the type of 'resolution' (line 477)
            resolution_280595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 31), 'resolution')
            # Getting the type of 'self' (line 477)
            self_280596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'self')
            # Setting the type of the member '_resolution' of a type (line 477)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 12), self_280596, '_resolution', resolution_280595)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_non_affine'
            module_type_store = module_type_store.open_function_context('transform_non_affine', 479, 8, False)
            # Assigning a type to the variable 'self' (line 480)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'InvertedMollweideTransform.transform_non_affine')
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['xy'])
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            InvertedMollweideTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedMollweideTransform.transform_non_affine', ['xy'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_non_affine', localization, ['xy'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_non_affine(...)' code ##################

            
            # Assigning a Subscript to a Name (line 480):
            
            # Assigning a Subscript to a Name (line 480):
            
            # Obtaining the type of the subscript
            slice_280597 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 480, 16), None, None, None)
            int_280598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 22), 'int')
            int_280599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 24), 'int')
            slice_280600 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 480, 16), int_280598, int_280599, None)
            # Getting the type of 'xy' (line 480)
            xy_280601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'xy')
            # Obtaining the member '__getitem__' of a type (line 480)
            getitem___280602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), xy_280601, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 480)
            subscript_call_result_280603 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), getitem___280602, (slice_280597, slice_280600))
            
            # Assigning a type to the variable 'x' (line 480)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'x', subscript_call_result_280603)
            
            # Assigning a Subscript to a Name (line 481):
            
            # Assigning a Subscript to a Name (line 481):
            
            # Obtaining the type of the subscript
            slice_280604 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 481, 16), None, None, None)
            int_280605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 22), 'int')
            int_280606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 24), 'int')
            slice_280607 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 481, 16), int_280605, int_280606, None)
            # Getting the type of 'xy' (line 481)
            xy_280608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 16), 'xy')
            # Obtaining the member '__getitem__' of a type (line 481)
            getitem___280609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 16), xy_280608, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 481)
            subscript_call_result_280610 = invoke(stypy.reporting.localization.Localization(__file__, 481, 16), getitem___280609, (slice_280604, slice_280607))
            
            # Assigning a type to the variable 'y' (line 481)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'y', subscript_call_result_280610)
            
            # Assigning a Call to a Name (line 485):
            
            # Assigning a Call to a Name (line 485):
            
            # Call to arcsin(...): (line 485)
            # Processing the call arguments (line 485)
            # Getting the type of 'y' (line 485)
            y_280613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 30), 'y', False)
            
            # Call to sqrt(...): (line 485)
            # Processing the call arguments (line 485)
            int_280616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 42), 'int')
            # Processing the call keyword arguments (line 485)
            kwargs_280617 = {}
            # Getting the type of 'np' (line 485)
            np_280614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 34), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 485)
            sqrt_280615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 34), np_280614, 'sqrt')
            # Calling sqrt(args, kwargs) (line 485)
            sqrt_call_result_280618 = invoke(stypy.reporting.localization.Localization(__file__, 485, 34), sqrt_280615, *[int_280616], **kwargs_280617)
            
            # Applying the binary operator 'div' (line 485)
            result_div_280619 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 30), 'div', y_280613, sqrt_call_result_280618)
            
            # Processing the call keyword arguments (line 485)
            kwargs_280620 = {}
            # Getting the type of 'np' (line 485)
            np_280611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 20), 'np', False)
            # Obtaining the member 'arcsin' of a type (line 485)
            arcsin_280612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 20), np_280611, 'arcsin')
            # Calling arcsin(args, kwargs) (line 485)
            arcsin_call_result_280621 = invoke(stypy.reporting.localization.Localization(__file__, 485, 20), arcsin_280612, *[result_div_280619], **kwargs_280620)
            
            # Assigning a type to the variable 'theta' (line 485)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'theta', arcsin_call_result_280621)
            
            # Assigning a BinOp to a Name (line 486):
            
            # Assigning a BinOp to a Name (line 486):
            # Getting the type of 'np' (line 486)
            np_280622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 19), 'np')
            # Obtaining the member 'pi' of a type (line 486)
            pi_280623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 19), np_280622, 'pi')
            int_280624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 28), 'int')
            
            # Call to sqrt(...): (line 486)
            # Processing the call arguments (line 486)
            int_280627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 40), 'int')
            # Processing the call keyword arguments (line 486)
            kwargs_280628 = {}
            # Getting the type of 'np' (line 486)
            np_280625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 32), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 486)
            sqrt_280626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 32), np_280625, 'sqrt')
            # Calling sqrt(args, kwargs) (line 486)
            sqrt_call_result_280629 = invoke(stypy.reporting.localization.Localization(__file__, 486, 32), sqrt_280626, *[int_280627], **kwargs_280628)
            
            # Applying the binary operator '*' (line 486)
            result_mul_280630 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 28), '*', int_280624, sqrt_call_result_280629)
            
            # Applying the binary operator 'div' (line 486)
            result_div_280631 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 19), 'div', pi_280623, result_mul_280630)
            
            # Getting the type of 'x' (line 486)
            x_280632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 47), 'x')
            # Applying the binary operator '*' (line 486)
            result_mul_280633 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 18), '*', result_div_280631, x_280632)
            
            
            # Call to cos(...): (line 486)
            # Processing the call arguments (line 486)
            # Getting the type of 'theta' (line 486)
            theta_280636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 58), 'theta', False)
            # Processing the call keyword arguments (line 486)
            kwargs_280637 = {}
            # Getting the type of 'np' (line 486)
            np_280634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 51), 'np', False)
            # Obtaining the member 'cos' of a type (line 486)
            cos_280635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 51), np_280634, 'cos')
            # Calling cos(args, kwargs) (line 486)
            cos_call_result_280638 = invoke(stypy.reporting.localization.Localization(__file__, 486, 51), cos_280635, *[theta_280636], **kwargs_280637)
            
            # Applying the binary operator 'div' (line 486)
            result_div_280639 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 49), 'div', result_mul_280633, cos_call_result_280638)
            
            # Assigning a type to the variable 'lon' (line 486)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'lon', result_div_280639)
            
            # Assigning a Call to a Name (line 487):
            
            # Assigning a Call to a Name (line 487):
            
            # Call to arcsin(...): (line 487)
            # Processing the call arguments (line 487)
            int_280642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 29), 'int')
            # Getting the type of 'theta' (line 487)
            theta_280643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 33), 'theta', False)
            # Applying the binary operator '*' (line 487)
            result_mul_280644 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 29), '*', int_280642, theta_280643)
            
            
            # Call to sin(...): (line 487)
            # Processing the call arguments (line 487)
            int_280647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 48), 'int')
            # Getting the type of 'theta' (line 487)
            theta_280648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 52), 'theta', False)
            # Applying the binary operator '*' (line 487)
            result_mul_280649 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 48), '*', int_280647, theta_280648)
            
            # Processing the call keyword arguments (line 487)
            kwargs_280650 = {}
            # Getting the type of 'np' (line 487)
            np_280645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 41), 'np', False)
            # Obtaining the member 'sin' of a type (line 487)
            sin_280646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 41), np_280645, 'sin')
            # Calling sin(args, kwargs) (line 487)
            sin_call_result_280651 = invoke(stypy.reporting.localization.Localization(__file__, 487, 41), sin_280646, *[result_mul_280649], **kwargs_280650)
            
            # Applying the binary operator '+' (line 487)
            result_add_280652 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 29), '+', result_mul_280644, sin_call_result_280651)
            
            # Getting the type of 'np' (line 487)
            np_280653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 62), 'np', False)
            # Obtaining the member 'pi' of a type (line 487)
            pi_280654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 62), np_280653, 'pi')
            # Applying the binary operator 'div' (line 487)
            result_div_280655 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 28), 'div', result_add_280652, pi_280654)
            
            # Processing the call keyword arguments (line 487)
            kwargs_280656 = {}
            # Getting the type of 'np' (line 487)
            np_280640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 18), 'np', False)
            # Obtaining the member 'arcsin' of a type (line 487)
            arcsin_280641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 18), np_280640, 'arcsin')
            # Calling arcsin(args, kwargs) (line 487)
            arcsin_call_result_280657 = invoke(stypy.reporting.localization.Localization(__file__, 487, 18), arcsin_280641, *[result_div_280655], **kwargs_280656)
            
            # Assigning a type to the variable 'lat' (line 487)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'lat', arcsin_call_result_280657)
            
            # Call to concatenate(...): (line 489)
            # Processing the call arguments (line 489)
            
            # Obtaining an instance of the builtin type 'tuple' (line 489)
            tuple_280660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 489)
            # Adding element type (line 489)
            # Getting the type of 'lon' (line 489)
            lon_280661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 35), 'lon', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 35), tuple_280660, lon_280661)
            # Adding element type (line 489)
            # Getting the type of 'lat' (line 489)
            lat_280662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 40), 'lat', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 35), tuple_280660, lat_280662)
            
            int_280663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 46), 'int')
            # Processing the call keyword arguments (line 489)
            kwargs_280664 = {}
            # Getting the type of 'np' (line 489)
            np_280658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 19), 'np', False)
            # Obtaining the member 'concatenate' of a type (line 489)
            concatenate_280659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 19), np_280658, 'concatenate')
            # Calling concatenate(args, kwargs) (line 489)
            concatenate_call_result_280665 = invoke(stypy.reporting.localization.Localization(__file__, 489, 19), concatenate_280659, *[tuple_280660, int_280663], **kwargs_280664)
            
            # Assigning a type to the variable 'stypy_return_type' (line 489)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'stypy_return_type', concatenate_call_result_280665)
            
            # ################# End of 'transform_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 479)
            stypy_return_type_280666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280666)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_non_affine'
            return stypy_return_type_280666

        
        # Assigning a Attribute to a Attribute (line 490):
        
        # Assigning a Attribute to a Attribute (line 490):
        # Getting the type of 'Transform' (line 490)
        Transform_280667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 39), 'Transform')
        # Obtaining the member 'transform_non_affine' of a type (line 490)
        transform_non_affine_280668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 39), Transform_280667, 'transform_non_affine')
        # Obtaining the member '__doc__' of a type (line 490)
        doc___280669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 39), transform_non_affine_280668, '__doc__')
        # Getting the type of 'transform_non_affine' (line 490)
        transform_non_affine_280670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'transform_non_affine')
        # Setting the type of the member '__doc__' of a type (line 490)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), transform_non_affine_280670, '__doc__', doc___280669)

        @norecursion
        def inverted(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inverted'
            module_type_store = module_type_store.open_function_context('inverted', 492, 8, False)
            # Assigning a type to the variable 'self' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedMollweideTransform.inverted')
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            InvertedMollweideTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedMollweideTransform.inverted', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inverted', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inverted(...)' code ##################

            
            # Call to MollweideTransform(...): (line 493)
            # Processing the call arguments (line 493)
            # Getting the type of 'self' (line 493)
            self_280673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 52), 'self', False)
            # Obtaining the member '_resolution' of a type (line 493)
            _resolution_280674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 52), self_280673, '_resolution')
            # Processing the call keyword arguments (line 493)
            kwargs_280675 = {}
            # Getting the type of 'MollweideAxes' (line 493)
            MollweideAxes_280671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 19), 'MollweideAxes', False)
            # Obtaining the member 'MollweideTransform' of a type (line 493)
            MollweideTransform_280672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 19), MollweideAxes_280671, 'MollweideTransform')
            # Calling MollweideTransform(args, kwargs) (line 493)
            MollweideTransform_call_result_280676 = invoke(stypy.reporting.localization.Localization(__file__, 493, 19), MollweideTransform_280672, *[_resolution_280674], **kwargs_280675)
            
            # Assigning a type to the variable 'stypy_return_type' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'stypy_return_type', MollweideTransform_call_result_280676)
            
            # ################# End of 'inverted(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inverted' in the type store
            # Getting the type of 'stypy_return_type' (line 492)
            stypy_return_type_280677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280677)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inverted'
            return stypy_return_type_280677

        
        # Assigning a Attribute to a Attribute (line 494):
        
        # Assigning a Attribute to a Attribute (line 494):
        # Getting the type of 'Transform' (line 494)
        Transform_280678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 27), 'Transform')
        # Obtaining the member 'inverted' of a type (line 494)
        inverted_280679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 27), Transform_280678, 'inverted')
        # Obtaining the member '__doc__' of a type (line 494)
        doc___280680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 27), inverted_280679, '__doc__')
        # Getting the type of 'inverted' (line 494)
        inverted_280681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'inverted')
        # Setting the type of the member '__doc__' of a type (line 494)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), inverted_280681, '__doc__', doc___280680)
    
    # Assigning a type to the variable 'InvertedMollweideTransform' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'InvertedMollweideTransform', InvertedMollweideTransform)

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 496, 4, False)
        # Assigning a type to the variable 'self' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MollweideAxes.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Attribute (line 497):
        
        # Assigning a BinOp to a Attribute (line 497):
        # Getting the type of 'np' (line 497)
        np_280682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 30), 'np')
        # Obtaining the member 'pi' of a type (line 497)
        pi_280683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 30), np_280682, 'pi')
        float_280684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 38), 'float')
        # Applying the binary operator 'div' (line 497)
        result_div_280685 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 30), 'div', pi_280683, float_280684)
        
        # Getting the type of 'self' (line 497)
        self_280686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'self')
        # Setting the type of the member '_longitude_cap' of a type (line 497)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), self_280686, '_longitude_cap', result_div_280685)
        
        # Call to __init__(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'self' (line 498)
        self_280689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 25), 'self', False)
        # Getting the type of 'args' (line 498)
        args_280690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 32), 'args', False)
        # Processing the call keyword arguments (line 498)
        # Getting the type of 'kwargs' (line 498)
        kwargs_280691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 40), 'kwargs', False)
        kwargs_280692 = {'kwargs_280691': kwargs_280691}
        # Getting the type of 'GeoAxes' (line 498)
        GeoAxes_280687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'GeoAxes', False)
        # Obtaining the member '__init__' of a type (line 498)
        init___280688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), GeoAxes_280687, '__init__')
        # Calling __init__(args, kwargs) (line 498)
        init___call_result_280693 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), init___280688, *[self_280689, args_280690], **kwargs_280692)
        
        
        # Call to set_aspect(...): (line 499)
        # Processing the call arguments (line 499)
        float_280696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 24), 'float')
        # Processing the call keyword arguments (line 499)
        unicode_280697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 40), 'unicode', u'box')
        keyword_280698 = unicode_280697
        unicode_280699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 54), 'unicode', u'C')
        keyword_280700 = unicode_280699
        kwargs_280701 = {'adjustable': keyword_280698, 'anchor': keyword_280700}
        # Getting the type of 'self' (line 499)
        self_280694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'self', False)
        # Obtaining the member 'set_aspect' of a type (line 499)
        set_aspect_280695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), self_280694, 'set_aspect')
        # Calling set_aspect(args, kwargs) (line 499)
        set_aspect_call_result_280702 = invoke(stypy.reporting.localization.Localization(__file__, 499, 8), set_aspect_280695, *[float_280696], **kwargs_280701)
        
        
        # Call to cla(...): (line 500)
        # Processing the call keyword arguments (line 500)
        kwargs_280705 = {}
        # Getting the type of 'self' (line 500)
        self_280703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'self', False)
        # Obtaining the member 'cla' of a type (line 500)
        cla_280704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), self_280703, 'cla')
        # Calling cla(args, kwargs) (line 500)
        cla_call_result_280706 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), cla_280704, *[], **kwargs_280705)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _get_core_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_core_transform'
        module_type_store = module_type_store.open_function_context('_get_core_transform', 502, 4, False)
        # Assigning a type to the variable 'self' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_localization', localization)
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_function_name', 'MollweideAxes._get_core_transform')
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_param_names_list', ['resolution'])
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MollweideAxes._get_core_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MollweideAxes._get_core_transform', ['resolution'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_core_transform', localization, ['resolution'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_core_transform(...)' code ##################

        
        # Call to MollweideTransform(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'resolution' (line 503)
        resolution_280709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 39), 'resolution', False)
        # Processing the call keyword arguments (line 503)
        kwargs_280710 = {}
        # Getting the type of 'self' (line 503)
        self_280707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 15), 'self', False)
        # Obtaining the member 'MollweideTransform' of a type (line 503)
        MollweideTransform_280708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 15), self_280707, 'MollweideTransform')
        # Calling MollweideTransform(args, kwargs) (line 503)
        MollweideTransform_call_result_280711 = invoke(stypy.reporting.localization.Localization(__file__, 503, 15), MollweideTransform_280708, *[resolution_280709], **kwargs_280710)
        
        # Assigning a type to the variable 'stypy_return_type' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'stypy_return_type', MollweideTransform_call_result_280711)
        
        # ################# End of '_get_core_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_core_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 502)
        stypy_return_type_280712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_280712)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_core_transform'
        return stypy_return_type_280712


# Assigning a type to the variable 'MollweideAxes' (line 406)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'MollweideAxes', MollweideAxes)

# Assigning a Str to a Name (line 407):
unicode_280713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 11), 'unicode', u'mollweide')
# Getting the type of 'MollweideAxes'
MollweideAxes_280714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MollweideAxes')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MollweideAxes_280714, 'name', unicode_280713)
# Declaration of the 'LambertAxes' class
# Getting the type of 'GeoAxes' (line 506)
GeoAxes_280715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 18), 'GeoAxes')

class LambertAxes(GeoAxes_280715, ):
    
    # Assigning a Str to a Name (line 507):
    # Declaration of the 'LambertTransform' class
    # Getting the type of 'Transform' (line 509)
    Transform_280716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 27), 'Transform')

    class LambertTransform(Transform_280716, ):
        unicode_280717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, (-1)), 'unicode', u'\n        The base Lambert transform.\n        ')
        
        # Assigning a Num to a Name (line 513):
        
        # Assigning a Num to a Name (line 513):
        int_280718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 21), 'int')
        # Assigning a type to the variable 'input_dims' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'input_dims', int_280718)
        
        # Assigning a Num to a Name (line 514):
        
        # Assigning a Num to a Name (line 514):
        int_280719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 22), 'int')
        # Assigning a type to the variable 'output_dims' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'output_dims', int_280719)
        
        # Assigning a Name to a Name (line 515):
        
        # Assigning a Name to a Name (line 515):
        # Getting the type of 'False' (line 515)
        False_280720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 23), 'False')
        # Assigning a type to the variable 'is_separable' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'is_separable', False_280720)

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 517, 8, False)
            # Assigning a type to the variable 'self' (line 518)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'LambertTransform.__init__', ['center_longitude', 'center_latitude', 'resolution'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['center_longitude', 'center_latitude', 'resolution'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            unicode_280721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, (-1)), 'unicode', u'\n            Create a new Lambert transform.  Resolution is the number of steps\n            to interpolate between each input line segment to approximate its\n            path in curved Lambert space.\n            ')
            
            # Call to __init__(...): (line 523)
            # Processing the call arguments (line 523)
            # Getting the type of 'self' (line 523)
            self_280724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 31), 'self', False)
            # Processing the call keyword arguments (line 523)
            kwargs_280725 = {}
            # Getting the type of 'Transform' (line 523)
            Transform_280722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'Transform', False)
            # Obtaining the member '__init__' of a type (line 523)
            init___280723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 12), Transform_280722, '__init__')
            # Calling __init__(args, kwargs) (line 523)
            init___call_result_280726 = invoke(stypy.reporting.localization.Localization(__file__, 523, 12), init___280723, *[self_280724], **kwargs_280725)
            
            
            # Assigning a Name to a Attribute (line 524):
            
            # Assigning a Name to a Attribute (line 524):
            # Getting the type of 'resolution' (line 524)
            resolution_280727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 31), 'resolution')
            # Getting the type of 'self' (line 524)
            self_280728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'self')
            # Setting the type of the member '_resolution' of a type (line 524)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 12), self_280728, '_resolution', resolution_280727)
            
            # Assigning a Name to a Attribute (line 525):
            
            # Assigning a Name to a Attribute (line 525):
            # Getting the type of 'center_longitude' (line 525)
            center_longitude_280729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 37), 'center_longitude')
            # Getting the type of 'self' (line 525)
            self_280730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'self')
            # Setting the type of the member '_center_longitude' of a type (line 525)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 12), self_280730, '_center_longitude', center_longitude_280729)
            
            # Assigning a Name to a Attribute (line 526):
            
            # Assigning a Name to a Attribute (line 526):
            # Getting the type of 'center_latitude' (line 526)
            center_latitude_280731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 36), 'center_latitude')
            # Getting the type of 'self' (line 526)
            self_280732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'self')
            # Setting the type of the member '_center_latitude' of a type (line 526)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 12), self_280732, '_center_latitude', center_latitude_280731)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_non_affine'
            module_type_store = module_type_store.open_function_context('transform_non_affine', 528, 8, False)
            # Assigning a type to the variable 'self' (line 529)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'LambertTransform.transform_non_affine')
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['ll'])
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            LambertTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'LambertTransform.transform_non_affine', ['ll'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_non_affine', localization, ['ll'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_non_affine(...)' code ##################

            
            # Assigning a Subscript to a Name (line 529):
            
            # Assigning a Subscript to a Name (line 529):
            
            # Obtaining the type of the subscript
            slice_280733 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 529, 24), None, None, None)
            int_280734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 30), 'int')
            int_280735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 32), 'int')
            slice_280736 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 529, 24), int_280734, int_280735, None)
            # Getting the type of 'll' (line 529)
            ll_280737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 24), 'll')
            # Obtaining the member '__getitem__' of a type (line 529)
            getitem___280738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 24), ll_280737, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 529)
            subscript_call_result_280739 = invoke(stypy.reporting.localization.Localization(__file__, 529, 24), getitem___280738, (slice_280733, slice_280736))
            
            # Assigning a type to the variable 'longitude' (line 529)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'longitude', subscript_call_result_280739)
            
            # Assigning a Subscript to a Name (line 530):
            
            # Assigning a Subscript to a Name (line 530):
            
            # Obtaining the type of the subscript
            slice_280740 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 530, 24), None, None, None)
            int_280741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 30), 'int')
            int_280742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 32), 'int')
            slice_280743 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 530, 24), int_280741, int_280742, None)
            # Getting the type of 'll' (line 530)
            ll_280744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 24), 'll')
            # Obtaining the member '__getitem__' of a type (line 530)
            getitem___280745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 24), ll_280744, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 530)
            subscript_call_result_280746 = invoke(stypy.reporting.localization.Localization(__file__, 530, 24), getitem___280745, (slice_280740, slice_280743))
            
            # Assigning a type to the variable 'latitude' (line 530)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'latitude', subscript_call_result_280746)
            
            # Assigning a Attribute to a Name (line 531):
            
            # Assigning a Attribute to a Name (line 531):
            # Getting the type of 'self' (line 531)
            self_280747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 20), 'self')
            # Obtaining the member '_center_longitude' of a type (line 531)
            _center_longitude_280748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 20), self_280747, '_center_longitude')
            # Assigning a type to the variable 'clong' (line 531)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'clong', _center_longitude_280748)
            
            # Assigning a Attribute to a Name (line 532):
            
            # Assigning a Attribute to a Name (line 532):
            # Getting the type of 'self' (line 532)
            self_280749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 19), 'self')
            # Obtaining the member '_center_latitude' of a type (line 532)
            _center_latitude_280750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 19), self_280749, '_center_latitude')
            # Assigning a type to the variable 'clat' (line 532)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'clat', _center_latitude_280750)
            
            # Assigning a Call to a Name (line 533):
            
            # Assigning a Call to a Name (line 533):
            
            # Call to cos(...): (line 533)
            # Processing the call arguments (line 533)
            # Getting the type of 'latitude' (line 533)
            latitude_280753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 29), 'latitude', False)
            # Processing the call keyword arguments (line 533)
            kwargs_280754 = {}
            # Getting the type of 'np' (line 533)
            np_280751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'np', False)
            # Obtaining the member 'cos' of a type (line 533)
            cos_280752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 22), np_280751, 'cos')
            # Calling cos(args, kwargs) (line 533)
            cos_call_result_280755 = invoke(stypy.reporting.localization.Localization(__file__, 533, 22), cos_280752, *[latitude_280753], **kwargs_280754)
            
            # Assigning a type to the variable 'cos_lat' (line 533)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'cos_lat', cos_call_result_280755)
            
            # Assigning a Call to a Name (line 534):
            
            # Assigning a Call to a Name (line 534):
            
            # Call to sin(...): (line 534)
            # Processing the call arguments (line 534)
            # Getting the type of 'latitude' (line 534)
            latitude_280758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 29), 'latitude', False)
            # Processing the call keyword arguments (line 534)
            kwargs_280759 = {}
            # Getting the type of 'np' (line 534)
            np_280756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 22), 'np', False)
            # Obtaining the member 'sin' of a type (line 534)
            sin_280757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 22), np_280756, 'sin')
            # Calling sin(args, kwargs) (line 534)
            sin_call_result_280760 = invoke(stypy.reporting.localization.Localization(__file__, 534, 22), sin_280757, *[latitude_280758], **kwargs_280759)
            
            # Assigning a type to the variable 'sin_lat' (line 534)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'sin_lat', sin_call_result_280760)
            
            # Assigning a BinOp to a Name (line 535):
            
            # Assigning a BinOp to a Name (line 535):
            # Getting the type of 'longitude' (line 535)
            longitude_280761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 24), 'longitude')
            # Getting the type of 'clong' (line 535)
            clong_280762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 36), 'clong')
            # Applying the binary operator '-' (line 535)
            result_sub_280763 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 24), '-', longitude_280761, clong_280762)
            
            # Assigning a type to the variable 'diff_long' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'diff_long', result_sub_280763)
            
            # Assigning a Call to a Name (line 536):
            
            # Assigning a Call to a Name (line 536):
            
            # Call to cos(...): (line 536)
            # Processing the call arguments (line 536)
            # Getting the type of 'diff_long' (line 536)
            diff_long_280766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 35), 'diff_long', False)
            # Processing the call keyword arguments (line 536)
            kwargs_280767 = {}
            # Getting the type of 'np' (line 536)
            np_280764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 28), 'np', False)
            # Obtaining the member 'cos' of a type (line 536)
            cos_280765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 28), np_280764, 'cos')
            # Calling cos(args, kwargs) (line 536)
            cos_call_result_280768 = invoke(stypy.reporting.localization.Localization(__file__, 536, 28), cos_280765, *[diff_long_280766], **kwargs_280767)
            
            # Assigning a type to the variable 'cos_diff_long' (line 536)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'cos_diff_long', cos_call_result_280768)
            
            # Assigning a BinOp to a Name (line 538):
            
            # Assigning a BinOp to a Name (line 538):
            float_280769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 23), 'float')
            
            # Call to sin(...): (line 539)
            # Processing the call arguments (line 539)
            # Getting the type of 'clat' (line 539)
            clat_280772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 30), 'clat', False)
            # Processing the call keyword arguments (line 539)
            kwargs_280773 = {}
            # Getting the type of 'np' (line 539)
            np_280770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 23), 'np', False)
            # Obtaining the member 'sin' of a type (line 539)
            sin_280771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 23), np_280770, 'sin')
            # Calling sin(args, kwargs) (line 539)
            sin_call_result_280774 = invoke(stypy.reporting.localization.Localization(__file__, 539, 23), sin_280771, *[clat_280772], **kwargs_280773)
            
            # Getting the type of 'sin_lat' (line 539)
            sin_lat_280775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 36), 'sin_lat')
            # Applying the binary operator '*' (line 539)
            result_mul_280776 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 23), '*', sin_call_result_280774, sin_lat_280775)
            
            # Applying the binary operator '+' (line 538)
            result_add_280777 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 23), '+', float_280769, result_mul_280776)
            
            
            # Call to cos(...): (line 540)
            # Processing the call arguments (line 540)
            # Getting the type of 'clat' (line 540)
            clat_280780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 30), 'clat', False)
            # Processing the call keyword arguments (line 540)
            kwargs_280781 = {}
            # Getting the type of 'np' (line 540)
            np_280778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 23), 'np', False)
            # Obtaining the member 'cos' of a type (line 540)
            cos_280779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 23), np_280778, 'cos')
            # Calling cos(args, kwargs) (line 540)
            cos_call_result_280782 = invoke(stypy.reporting.localization.Localization(__file__, 540, 23), cos_280779, *[clat_280780], **kwargs_280781)
            
            # Getting the type of 'cos_lat' (line 540)
            cos_lat_280783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 36), 'cos_lat')
            # Applying the binary operator '*' (line 540)
            result_mul_280784 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 23), '*', cos_call_result_280782, cos_lat_280783)
            
            # Getting the type of 'cos_diff_long' (line 540)
            cos_diff_long_280785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 44), 'cos_diff_long')
            # Applying the binary operator '*' (line 540)
            result_mul_280786 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 43), '*', result_mul_280784, cos_diff_long_280785)
            
            # Applying the binary operator '+' (line 539)
            result_add_280787 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 44), '+', result_add_280777, result_mul_280786)
            
            # Assigning a type to the variable 'inner_k' (line 538)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'inner_k', result_add_280787)
            
            # Assigning a Call to a Name (line 542):
            
            # Assigning a Call to a Name (line 542):
            
            # Call to where(...): (line 542)
            # Processing the call arguments (line 542)
            
            # Getting the type of 'inner_k' (line 542)
            inner_k_280790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 31), 'inner_k', False)
            float_280791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 42), 'float')
            # Applying the binary operator '==' (line 542)
            result_eq_280792 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 31), '==', inner_k_280790, float_280791)
            
            float_280793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 47), 'float')
            # Getting the type of 'inner_k' (line 542)
            inner_k_280794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 54), 'inner_k', False)
            # Processing the call keyword arguments (line 542)
            kwargs_280795 = {}
            # Getting the type of 'np' (line 542)
            np_280788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 22), 'np', False)
            # Obtaining the member 'where' of a type (line 542)
            where_280789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 22), np_280788, 'where')
            # Calling where(args, kwargs) (line 542)
            where_call_result_280796 = invoke(stypy.reporting.localization.Localization(__file__, 542, 22), where_280789, *[result_eq_280792, float_280793, inner_k_280794], **kwargs_280795)
            
            # Assigning a type to the variable 'inner_k' (line 542)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'inner_k', where_call_result_280796)
            
            # Assigning a Call to a Name (line 543):
            
            # Assigning a Call to a Name (line 543):
            
            # Call to sqrt(...): (line 543)
            # Processing the call arguments (line 543)
            float_280799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 24), 'float')
            # Getting the type of 'inner_k' (line 543)
            inner_k_280800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 30), 'inner_k', False)
            # Applying the binary operator 'div' (line 543)
            result_div_280801 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 24), 'div', float_280799, inner_k_280800)
            
            # Processing the call keyword arguments (line 543)
            kwargs_280802 = {}
            # Getting the type of 'np' (line 543)
            np_280797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 16), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 543)
            sqrt_280798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 16), np_280797, 'sqrt')
            # Calling sqrt(args, kwargs) (line 543)
            sqrt_call_result_280803 = invoke(stypy.reporting.localization.Localization(__file__, 543, 16), sqrt_280798, *[result_div_280801], **kwargs_280802)
            
            # Assigning a type to the variable 'k' (line 543)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'k', sqrt_call_result_280803)
            
            # Assigning a BinOp to a Name (line 544):
            
            # Assigning a BinOp to a Name (line 544):
            # Getting the type of 'k' (line 544)
            k_280804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'k')
            # Getting the type of 'cos_lat' (line 544)
            cos_lat_280805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 18), 'cos_lat')
            # Applying the binary operator '*' (line 544)
            result_mul_280806 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 16), '*', k_280804, cos_lat_280805)
            
            
            # Call to sin(...): (line 544)
            # Processing the call arguments (line 544)
            # Getting the type of 'diff_long' (line 544)
            diff_long_280809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 33), 'diff_long', False)
            # Processing the call keyword arguments (line 544)
            kwargs_280810 = {}
            # Getting the type of 'np' (line 544)
            np_280807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 26), 'np', False)
            # Obtaining the member 'sin' of a type (line 544)
            sin_280808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 26), np_280807, 'sin')
            # Calling sin(args, kwargs) (line 544)
            sin_call_result_280811 = invoke(stypy.reporting.localization.Localization(__file__, 544, 26), sin_280808, *[diff_long_280809], **kwargs_280810)
            
            # Applying the binary operator '*' (line 544)
            result_mul_280812 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 25), '*', result_mul_280806, sin_call_result_280811)
            
            # Assigning a type to the variable 'x' (line 544)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'x', result_mul_280812)
            
            # Assigning a BinOp to a Name (line 545):
            
            # Assigning a BinOp to a Name (line 545):
            # Getting the type of 'k' (line 545)
            k_280813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'k')
            
            # Call to cos(...): (line 545)
            # Processing the call arguments (line 545)
            # Getting the type of 'clat' (line 545)
            clat_280816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 26), 'clat', False)
            # Processing the call keyword arguments (line 545)
            kwargs_280817 = {}
            # Getting the type of 'np' (line 545)
            np_280814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 19), 'np', False)
            # Obtaining the member 'cos' of a type (line 545)
            cos_280815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 19), np_280814, 'cos')
            # Calling cos(args, kwargs) (line 545)
            cos_call_result_280818 = invoke(stypy.reporting.localization.Localization(__file__, 545, 19), cos_280815, *[clat_280816], **kwargs_280817)
            
            # Getting the type of 'sin_lat' (line 545)
            sin_lat_280819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 32), 'sin_lat')
            # Applying the binary operator '*' (line 545)
            result_mul_280820 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 19), '*', cos_call_result_280818, sin_lat_280819)
            
            
            # Call to sin(...): (line 546)
            # Processing the call arguments (line 546)
            # Getting the type of 'clat' (line 546)
            clat_280823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 26), 'clat', False)
            # Processing the call keyword arguments (line 546)
            kwargs_280824 = {}
            # Getting the type of 'np' (line 546)
            np_280821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 19), 'np', False)
            # Obtaining the member 'sin' of a type (line 546)
            sin_280822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 19), np_280821, 'sin')
            # Calling sin(args, kwargs) (line 546)
            sin_call_result_280825 = invoke(stypy.reporting.localization.Localization(__file__, 546, 19), sin_280822, *[clat_280823], **kwargs_280824)
            
            # Getting the type of 'cos_lat' (line 546)
            cos_lat_280826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 32), 'cos_lat')
            # Applying the binary operator '*' (line 546)
            result_mul_280827 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 19), '*', sin_call_result_280825, cos_lat_280826)
            
            # Getting the type of 'cos_diff_long' (line 546)
            cos_diff_long_280828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 40), 'cos_diff_long')
            # Applying the binary operator '*' (line 546)
            result_mul_280829 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 39), '*', result_mul_280827, cos_diff_long_280828)
            
            # Applying the binary operator '-' (line 545)
            result_sub_280830 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 19), '-', result_mul_280820, result_mul_280829)
            
            # Applying the binary operator '*' (line 545)
            result_mul_280831 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 16), '*', k_280813, result_sub_280830)
            
            # Assigning a type to the variable 'y' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'y', result_mul_280831)
            
            # Call to concatenate(...): (line 548)
            # Processing the call arguments (line 548)
            
            # Obtaining an instance of the builtin type 'tuple' (line 548)
            tuple_280834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 548)
            # Adding element type (line 548)
            # Getting the type of 'x' (line 548)
            x_280835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 35), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 35), tuple_280834, x_280835)
            # Adding element type (line 548)
            # Getting the type of 'y' (line 548)
            y_280836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 38), 'y', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 35), tuple_280834, y_280836)
            
            int_280837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 42), 'int')
            # Processing the call keyword arguments (line 548)
            kwargs_280838 = {}
            # Getting the type of 'np' (line 548)
            np_280832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'np', False)
            # Obtaining the member 'concatenate' of a type (line 548)
            concatenate_280833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 19), np_280832, 'concatenate')
            # Calling concatenate(args, kwargs) (line 548)
            concatenate_call_result_280839 = invoke(stypy.reporting.localization.Localization(__file__, 548, 19), concatenate_280833, *[tuple_280834, int_280837], **kwargs_280838)
            
            # Assigning a type to the variable 'stypy_return_type' (line 548)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'stypy_return_type', concatenate_call_result_280839)
            
            # ################# End of 'transform_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 528)
            stypy_return_type_280840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280840)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_non_affine'
            return stypy_return_type_280840

        
        # Assigning a Attribute to a Attribute (line 549):
        
        # Assigning a Attribute to a Attribute (line 549):
        # Getting the type of 'Transform' (line 549)
        Transform_280841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 39), 'Transform')
        # Obtaining the member 'transform_non_affine' of a type (line 549)
        transform_non_affine_280842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 39), Transform_280841, 'transform_non_affine')
        # Obtaining the member '__doc__' of a type (line 549)
        doc___280843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 39), transform_non_affine_280842, '__doc__')
        # Getting the type of 'transform_non_affine' (line 549)
        transform_non_affine_280844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'transform_non_affine')
        # Setting the type of the member '__doc__' of a type (line 549)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 8), transform_non_affine_280844, '__doc__', doc___280843)

        @norecursion
        def transform_path_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_path_non_affine'
            module_type_store = module_type_store.open_function_context('transform_path_non_affine', 551, 8, False)
            # Assigning a type to the variable 'self' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_localization', localization)
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_function_name', 'LambertTransform.transform_path_non_affine')
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_param_names_list', ['path'])
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            LambertTransform.transform_path_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'LambertTransform.transform_path_non_affine', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_path_non_affine', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_path_non_affine(...)' code ##################

            
            # Assigning a Attribute to a Name (line 552):
            
            # Assigning a Attribute to a Name (line 552):
            # Getting the type of 'path' (line 552)
            path_280845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 23), 'path')
            # Obtaining the member 'vertices' of a type (line 552)
            vertices_280846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 23), path_280845, 'vertices')
            # Assigning a type to the variable 'vertices' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'vertices', vertices_280846)
            
            # Assigning a Call to a Name (line 553):
            
            # Assigning a Call to a Name (line 553):
            
            # Call to interpolated(...): (line 553)
            # Processing the call arguments (line 553)
            # Getting the type of 'self' (line 553)
            self_280849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 38), 'self', False)
            # Obtaining the member '_resolution' of a type (line 553)
            _resolution_280850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 38), self_280849, '_resolution')
            # Processing the call keyword arguments (line 553)
            kwargs_280851 = {}
            # Getting the type of 'path' (line 553)
            path_280847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 20), 'path', False)
            # Obtaining the member 'interpolated' of a type (line 553)
            interpolated_280848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 20), path_280847, 'interpolated')
            # Calling interpolated(args, kwargs) (line 553)
            interpolated_call_result_280852 = invoke(stypy.reporting.localization.Localization(__file__, 553, 20), interpolated_280848, *[_resolution_280850], **kwargs_280851)
            
            # Assigning a type to the variable 'ipath' (line 553)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'ipath', interpolated_call_result_280852)
            
            # Call to Path(...): (line 554)
            # Processing the call arguments (line 554)
            
            # Call to transform(...): (line 554)
            # Processing the call arguments (line 554)
            # Getting the type of 'ipath' (line 554)
            ipath_280856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 39), 'ipath', False)
            # Obtaining the member 'vertices' of a type (line 554)
            vertices_280857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 39), ipath_280856, 'vertices')
            # Processing the call keyword arguments (line 554)
            kwargs_280858 = {}
            # Getting the type of 'self' (line 554)
            self_280854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 24), 'self', False)
            # Obtaining the member 'transform' of a type (line 554)
            transform_280855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 24), self_280854, 'transform')
            # Calling transform(args, kwargs) (line 554)
            transform_call_result_280859 = invoke(stypy.reporting.localization.Localization(__file__, 554, 24), transform_280855, *[vertices_280857], **kwargs_280858)
            
            # Getting the type of 'ipath' (line 554)
            ipath_280860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 56), 'ipath', False)
            # Obtaining the member 'codes' of a type (line 554)
            codes_280861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 56), ipath_280860, 'codes')
            # Processing the call keyword arguments (line 554)
            kwargs_280862 = {}
            # Getting the type of 'Path' (line 554)
            Path_280853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 554)
            Path_call_result_280863 = invoke(stypy.reporting.localization.Localization(__file__, 554, 19), Path_280853, *[transform_call_result_280859, codes_280861], **kwargs_280862)
            
            # Assigning a type to the variable 'stypy_return_type' (line 554)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'stypy_return_type', Path_call_result_280863)
            
            # ################# End of 'transform_path_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_path_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 551)
            stypy_return_type_280864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280864)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_path_non_affine'
            return stypy_return_type_280864

        
        # Assigning a Attribute to a Attribute (line 555):
        
        # Assigning a Attribute to a Attribute (line 555):
        # Getting the type of 'Transform' (line 555)
        Transform_280865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 44), 'Transform')
        # Obtaining the member 'transform_path_non_affine' of a type (line 555)
        transform_path_non_affine_280866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 44), Transform_280865, 'transform_path_non_affine')
        # Obtaining the member '__doc__' of a type (line 555)
        doc___280867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 44), transform_path_non_affine_280866, '__doc__')
        # Getting the type of 'transform_path_non_affine' (line 555)
        transform_path_non_affine_280868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'transform_path_non_affine')
        # Setting the type of the member '__doc__' of a type (line 555)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 8), transform_path_non_affine_280868, '__doc__', doc___280867)

        @norecursion
        def inverted(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inverted'
            module_type_store = module_type_store.open_function_context('inverted', 557, 8, False)
            # Assigning a type to the variable 'self' (line 558)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            LambertTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
            LambertTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            LambertTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
            LambertTransform.inverted.__dict__.__setitem__('stypy_function_name', 'LambertTransform.inverted')
            LambertTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
            LambertTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
            LambertTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
            LambertTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
            LambertTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
            LambertTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            LambertTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'LambertTransform.inverted', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inverted', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inverted(...)' code ##################

            
            # Call to InvertedLambertTransform(...): (line 558)
            # Processing the call arguments (line 558)
            # Getting the type of 'self' (line 559)
            self_280871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 16), 'self', False)
            # Obtaining the member '_center_longitude' of a type (line 559)
            _center_longitude_280872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 16), self_280871, '_center_longitude')
            # Getting the type of 'self' (line 560)
            self_280873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'self', False)
            # Obtaining the member '_center_latitude' of a type (line 560)
            _center_latitude_280874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 16), self_280873, '_center_latitude')
            # Getting the type of 'self' (line 561)
            self_280875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'self', False)
            # Obtaining the member '_resolution' of a type (line 561)
            _resolution_280876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 16), self_280875, '_resolution')
            # Processing the call keyword arguments (line 558)
            kwargs_280877 = {}
            # Getting the type of 'LambertAxes' (line 558)
            LambertAxes_280869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 19), 'LambertAxes', False)
            # Obtaining the member 'InvertedLambertTransform' of a type (line 558)
            InvertedLambertTransform_280870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 19), LambertAxes_280869, 'InvertedLambertTransform')
            # Calling InvertedLambertTransform(args, kwargs) (line 558)
            InvertedLambertTransform_call_result_280878 = invoke(stypy.reporting.localization.Localization(__file__, 558, 19), InvertedLambertTransform_280870, *[_center_longitude_280872, _center_latitude_280874, _resolution_280876], **kwargs_280877)
            
            # Assigning a type to the variable 'stypy_return_type' (line 558)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'stypy_return_type', InvertedLambertTransform_call_result_280878)
            
            # ################# End of 'inverted(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inverted' in the type store
            # Getting the type of 'stypy_return_type' (line 557)
            stypy_return_type_280879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_280879)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inverted'
            return stypy_return_type_280879

        
        # Assigning a Attribute to a Attribute (line 562):
        
        # Assigning a Attribute to a Attribute (line 562):
        # Getting the type of 'Transform' (line 562)
        Transform_280880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 27), 'Transform')
        # Obtaining the member 'inverted' of a type (line 562)
        inverted_280881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 27), Transform_280880, 'inverted')
        # Obtaining the member '__doc__' of a type (line 562)
        doc___280882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 27), inverted_280881, '__doc__')
        # Getting the type of 'inverted' (line 562)
        inverted_280883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'inverted')
        # Setting the type of the member '__doc__' of a type (line 562)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 8), inverted_280883, '__doc__', doc___280882)
    
    # Assigning a type to the variable 'LambertTransform' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'LambertTransform', LambertTransform)
    # Declaration of the 'InvertedLambertTransform' class
    # Getting the type of 'Transform' (line 564)
    Transform_280884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 35), 'Transform')

    class InvertedLambertTransform(Transform_280884, ):
        
        # Assigning a Num to a Name (line 565):
        
        # Assigning a Num to a Name (line 565):
        int_280885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 21), 'int')
        # Assigning a type to the variable 'input_dims' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'input_dims', int_280885)
        
        # Assigning a Num to a Name (line 566):
        
        # Assigning a Num to a Name (line 566):
        int_280886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 22), 'int')
        # Assigning a type to the variable 'output_dims' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'output_dims', int_280886)
        
        # Assigning a Name to a Name (line 567):
        
        # Assigning a Name to a Name (line 567):
        # Getting the type of 'False' (line 567)
        False_280887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'False')
        # Assigning a type to the variable 'is_separable' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'is_separable', False_280887)

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 569, 8, False)
            # Assigning a type to the variable 'self' (line 570)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLambertTransform.__init__', ['center_longitude', 'center_latitude', 'resolution'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['center_longitude', 'center_latitude', 'resolution'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Call to __init__(...): (line 570)
            # Processing the call arguments (line 570)
            # Getting the type of 'self' (line 570)
            self_280890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 31), 'self', False)
            # Processing the call keyword arguments (line 570)
            kwargs_280891 = {}
            # Getting the type of 'Transform' (line 570)
            Transform_280888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'Transform', False)
            # Obtaining the member '__init__' of a type (line 570)
            init___280889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 12), Transform_280888, '__init__')
            # Calling __init__(args, kwargs) (line 570)
            init___call_result_280892 = invoke(stypy.reporting.localization.Localization(__file__, 570, 12), init___280889, *[self_280890], **kwargs_280891)
            
            
            # Assigning a Name to a Attribute (line 571):
            
            # Assigning a Name to a Attribute (line 571):
            # Getting the type of 'resolution' (line 571)
            resolution_280893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 31), 'resolution')
            # Getting the type of 'self' (line 571)
            self_280894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'self')
            # Setting the type of the member '_resolution' of a type (line 571)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 12), self_280894, '_resolution', resolution_280893)
            
            # Assigning a Name to a Attribute (line 572):
            
            # Assigning a Name to a Attribute (line 572):
            # Getting the type of 'center_longitude' (line 572)
            center_longitude_280895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 37), 'center_longitude')
            # Getting the type of 'self' (line 572)
            self_280896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'self')
            # Setting the type of the member '_center_longitude' of a type (line 572)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 12), self_280896, '_center_longitude', center_longitude_280895)
            
            # Assigning a Name to a Attribute (line 573):
            
            # Assigning a Name to a Attribute (line 573):
            # Getting the type of 'center_latitude' (line 573)
            center_latitude_280897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 36), 'center_latitude')
            # Getting the type of 'self' (line 573)
            self_280898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'self')
            # Setting the type of the member '_center_latitude' of a type (line 573)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 12), self_280898, '_center_latitude', center_latitude_280897)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'transform_non_affine'
            module_type_store = module_type_store.open_function_context('transform_non_affine', 575, 8, False)
            # Assigning a type to the variable 'self' (line 576)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'InvertedLambertTransform.transform_non_affine')
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['xy'])
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            InvertedLambertTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLambertTransform.transform_non_affine', ['xy'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'transform_non_affine', localization, ['xy'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'transform_non_affine(...)' code ##################

            
            # Assigning a Subscript to a Name (line 576):
            
            # Assigning a Subscript to a Name (line 576):
            
            # Obtaining the type of the subscript
            slice_280899 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 576, 16), None, None, None)
            int_280900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 22), 'int')
            int_280901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 24), 'int')
            slice_280902 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 576, 16), int_280900, int_280901, None)
            # Getting the type of 'xy' (line 576)
            xy_280903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'xy')
            # Obtaining the member '__getitem__' of a type (line 576)
            getitem___280904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 16), xy_280903, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 576)
            subscript_call_result_280905 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), getitem___280904, (slice_280899, slice_280902))
            
            # Assigning a type to the variable 'x' (line 576)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'x', subscript_call_result_280905)
            
            # Assigning a Subscript to a Name (line 577):
            
            # Assigning a Subscript to a Name (line 577):
            
            # Obtaining the type of the subscript
            slice_280906 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 577, 16), None, None, None)
            int_280907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 22), 'int')
            int_280908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 24), 'int')
            slice_280909 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 577, 16), int_280907, int_280908, None)
            # Getting the type of 'xy' (line 577)
            xy_280910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'xy')
            # Obtaining the member '__getitem__' of a type (line 577)
            getitem___280911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 16), xy_280910, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 577)
            subscript_call_result_280912 = invoke(stypy.reporting.localization.Localization(__file__, 577, 16), getitem___280911, (slice_280906, slice_280909))
            
            # Assigning a type to the variable 'y' (line 577)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'y', subscript_call_result_280912)
            
            # Assigning a Attribute to a Name (line 578):
            
            # Assigning a Attribute to a Name (line 578):
            # Getting the type of 'self' (line 578)
            self_280913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 20), 'self')
            # Obtaining the member '_center_longitude' of a type (line 578)
            _center_longitude_280914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 20), self_280913, '_center_longitude')
            # Assigning a type to the variable 'clong' (line 578)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'clong', _center_longitude_280914)
            
            # Assigning a Attribute to a Name (line 579):
            
            # Assigning a Attribute to a Name (line 579):
            # Getting the type of 'self' (line 579)
            self_280915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 19), 'self')
            # Obtaining the member '_center_latitude' of a type (line 579)
            _center_latitude_280916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 19), self_280915, '_center_latitude')
            # Assigning a type to the variable 'clat' (line 579)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'clat', _center_latitude_280916)
            
            # Assigning a Call to a Name (line 580):
            
            # Assigning a Call to a Name (line 580):
            
            # Call to sqrt(...): (line 580)
            # Processing the call arguments (line 580)
            # Getting the type of 'x' (line 580)
            x_280919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 24), 'x', False)
            # Getting the type of 'x' (line 580)
            x_280920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 26), 'x', False)
            # Applying the binary operator '*' (line 580)
            result_mul_280921 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 24), '*', x_280919, x_280920)
            
            # Getting the type of 'y' (line 580)
            y_280922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 30), 'y', False)
            # Getting the type of 'y' (line 580)
            y_280923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 32), 'y', False)
            # Applying the binary operator '*' (line 580)
            result_mul_280924 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 30), '*', y_280922, y_280923)
            
            # Applying the binary operator '+' (line 580)
            result_add_280925 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 24), '+', result_mul_280921, result_mul_280924)
            
            # Processing the call keyword arguments (line 580)
            kwargs_280926 = {}
            # Getting the type of 'np' (line 580)
            np_280917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 16), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 580)
            sqrt_280918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 16), np_280917, 'sqrt')
            # Calling sqrt(args, kwargs) (line 580)
            sqrt_call_result_280927 = invoke(stypy.reporting.localization.Localization(__file__, 580, 16), sqrt_280918, *[result_add_280925], **kwargs_280926)
            
            # Assigning a type to the variable 'p' (line 580)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'p', sqrt_call_result_280927)
            
            # Assigning a Call to a Name (line 581):
            
            # Assigning a Call to a Name (line 581):
            
            # Call to where(...): (line 581)
            # Processing the call arguments (line 581)
            
            # Getting the type of 'p' (line 581)
            p_280930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'p', False)
            float_280931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 30), 'float')
            # Applying the binary operator '==' (line 581)
            result_eq_280932 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 25), '==', p_280930, float_280931)
            
            float_280933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 35), 'float')
            # Getting the type of 'p' (line 581)
            p_280934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 41), 'p', False)
            # Processing the call keyword arguments (line 581)
            kwargs_280935 = {}
            # Getting the type of 'np' (line 581)
            np_280928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'np', False)
            # Obtaining the member 'where' of a type (line 581)
            where_280929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 16), np_280928, 'where')
            # Calling where(args, kwargs) (line 581)
            where_call_result_280936 = invoke(stypy.reporting.localization.Localization(__file__, 581, 16), where_280929, *[result_eq_280932, float_280933, p_280934], **kwargs_280935)
            
            # Assigning a type to the variable 'p' (line 581)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'p', where_call_result_280936)
            
            # Assigning a BinOp to a Name (line 582):
            
            # Assigning a BinOp to a Name (line 582):
            float_280937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 16), 'float')
            
            # Call to arcsin(...): (line 582)
            # Processing the call arguments (line 582)
            float_280940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 32), 'float')
            # Getting the type of 'p' (line 582)
            p_280941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 38), 'p', False)
            # Applying the binary operator '*' (line 582)
            result_mul_280942 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 32), '*', float_280940, p_280941)
            
            # Processing the call keyword arguments (line 582)
            kwargs_280943 = {}
            # Getting the type of 'np' (line 582)
            np_280938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 22), 'np', False)
            # Obtaining the member 'arcsin' of a type (line 582)
            arcsin_280939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 22), np_280938, 'arcsin')
            # Calling arcsin(args, kwargs) (line 582)
            arcsin_call_result_280944 = invoke(stypy.reporting.localization.Localization(__file__, 582, 22), arcsin_280939, *[result_mul_280942], **kwargs_280943)
            
            # Applying the binary operator '*' (line 582)
            result_mul_280945 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 16), '*', float_280937, arcsin_call_result_280944)
            
            # Assigning a type to the variable 'c' (line 582)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'c', result_mul_280945)
            
            # Assigning a Call to a Name (line 583):
            
            # Assigning a Call to a Name (line 583):
            
            # Call to sin(...): (line 583)
            # Processing the call arguments (line 583)
            # Getting the type of 'c' (line 583)
            c_280948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 27), 'c', False)
            # Processing the call keyword arguments (line 583)
            kwargs_280949 = {}
            # Getting the type of 'np' (line 583)
            np_280946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'np', False)
            # Obtaining the member 'sin' of a type (line 583)
            sin_280947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 20), np_280946, 'sin')
            # Calling sin(args, kwargs) (line 583)
            sin_call_result_280950 = invoke(stypy.reporting.localization.Localization(__file__, 583, 20), sin_280947, *[c_280948], **kwargs_280949)
            
            # Assigning a type to the variable 'sin_c' (line 583)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'sin_c', sin_call_result_280950)
            
            # Assigning a Call to a Name (line 584):
            
            # Assigning a Call to a Name (line 584):
            
            # Call to cos(...): (line 584)
            # Processing the call arguments (line 584)
            # Getting the type of 'c' (line 584)
            c_280953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 27), 'c', False)
            # Processing the call keyword arguments (line 584)
            kwargs_280954 = {}
            # Getting the type of 'np' (line 584)
            np_280951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'np', False)
            # Obtaining the member 'cos' of a type (line 584)
            cos_280952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 20), np_280951, 'cos')
            # Calling cos(args, kwargs) (line 584)
            cos_call_result_280955 = invoke(stypy.reporting.localization.Localization(__file__, 584, 20), cos_280952, *[c_280953], **kwargs_280954)
            
            # Assigning a type to the variable 'cos_c' (line 584)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'cos_c', cos_call_result_280955)
            
            # Assigning a Call to a Name (line 586):
            
            # Assigning a Call to a Name (line 586):
            
            # Call to arcsin(...): (line 586)
            # Processing the call arguments (line 586)
            # Getting the type of 'cos_c' (line 586)
            cos_c_280958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'cos_c', False)
            
            # Call to sin(...): (line 586)
            # Processing the call arguments (line 586)
            # Getting the type of 'clat' (line 586)
            clat_280961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 41), 'clat', False)
            # Processing the call keyword arguments (line 586)
            kwargs_280962 = {}
            # Getting the type of 'np' (line 586)
            np_280959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 34), 'np', False)
            # Obtaining the member 'sin' of a type (line 586)
            sin_280960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 34), np_280959, 'sin')
            # Calling sin(args, kwargs) (line 586)
            sin_call_result_280963 = invoke(stypy.reporting.localization.Localization(__file__, 586, 34), sin_280960, *[clat_280961], **kwargs_280962)
            
            # Applying the binary operator '*' (line 586)
            result_mul_280964 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 28), '*', cos_c_280958, sin_call_result_280963)
            
            # Getting the type of 'y' (line 587)
            y_280965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 31), 'y', False)
            # Getting the type of 'sin_c' (line 587)
            sin_c_280966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 33), 'sin_c', False)
            # Applying the binary operator '*' (line 587)
            result_mul_280967 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 31), '*', y_280965, sin_c_280966)
            
            
            # Call to cos(...): (line 587)
            # Processing the call arguments (line 587)
            # Getting the type of 'clat' (line 587)
            clat_280970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 46), 'clat', False)
            # Processing the call keyword arguments (line 587)
            kwargs_280971 = {}
            # Getting the type of 'np' (line 587)
            np_280968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 39), 'np', False)
            # Obtaining the member 'cos' of a type (line 587)
            cos_280969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 39), np_280968, 'cos')
            # Calling cos(args, kwargs) (line 587)
            cos_call_result_280972 = invoke(stypy.reporting.localization.Localization(__file__, 587, 39), cos_280969, *[clat_280970], **kwargs_280971)
            
            # Applying the binary operator '*' (line 587)
            result_mul_280973 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 38), '*', result_mul_280967, cos_call_result_280972)
            
            # Getting the type of 'p' (line 587)
            p_280974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 55), 'p', False)
            # Applying the binary operator 'div' (line 587)
            result_div_280975 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 30), 'div', result_mul_280973, p_280974)
            
            # Applying the binary operator '+' (line 586)
            result_add_280976 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 28), '+', result_mul_280964, result_div_280975)
            
            # Processing the call keyword arguments (line 586)
            kwargs_280977 = {}
            # Getting the type of 'np' (line 586)
            np_280956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 18), 'np', False)
            # Obtaining the member 'arcsin' of a type (line 586)
            arcsin_280957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 18), np_280956, 'arcsin')
            # Calling arcsin(args, kwargs) (line 586)
            arcsin_call_result_280978 = invoke(stypy.reporting.localization.Localization(__file__, 586, 18), arcsin_280957, *[result_add_280976], **kwargs_280977)
            
            # Assigning a type to the variable 'lat' (line 586)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'lat', arcsin_call_result_280978)
            
            # Assigning a BinOp to a Name (line 588):
            
            # Assigning a BinOp to a Name (line 588):
            # Getting the type of 'clong' (line 588)
            clong_280979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 18), 'clong')
            
            # Call to arctan(...): (line 588)
            # Processing the call arguments (line 588)
            # Getting the type of 'x' (line 589)
            x_280982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 17), 'x', False)
            # Getting the type of 'sin_c' (line 589)
            sin_c_280983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 19), 'sin_c', False)
            # Applying the binary operator '*' (line 589)
            result_mul_280984 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 17), '*', x_280982, sin_c_280983)
            
            # Getting the type of 'p' (line 589)
            p_280985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 29), 'p', False)
            
            # Call to cos(...): (line 589)
            # Processing the call arguments (line 589)
            # Getting the type of 'clat' (line 589)
            clat_280988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 38), 'clat', False)
            # Processing the call keyword arguments (line 589)
            kwargs_280989 = {}
            # Getting the type of 'np' (line 589)
            np_280986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 31), 'np', False)
            # Obtaining the member 'cos' of a type (line 589)
            cos_280987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 31), np_280986, 'cos')
            # Calling cos(args, kwargs) (line 589)
            cos_call_result_280990 = invoke(stypy.reporting.localization.Localization(__file__, 589, 31), cos_280987, *[clat_280988], **kwargs_280989)
            
            # Applying the binary operator '*' (line 589)
            result_mul_280991 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 29), '*', p_280985, cos_call_result_280990)
            
            # Getting the type of 'cos_c' (line 589)
            cos_c_280992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'cos_c', False)
            # Applying the binary operator '*' (line 589)
            result_mul_280993 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 43), '*', result_mul_280991, cos_c_280992)
            
            # Getting the type of 'y' (line 589)
            y_280994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 52), 'y', False)
            
            # Call to sin(...): (line 589)
            # Processing the call arguments (line 589)
            # Getting the type of 'clat' (line 589)
            clat_280997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 61), 'clat', False)
            # Processing the call keyword arguments (line 589)
            kwargs_280998 = {}
            # Getting the type of 'np' (line 589)
            np_280995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 54), 'np', False)
            # Obtaining the member 'sin' of a type (line 589)
            sin_280996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 54), np_280995, 'sin')
            # Calling sin(args, kwargs) (line 589)
            sin_call_result_280999 = invoke(stypy.reporting.localization.Localization(__file__, 589, 54), sin_280996, *[clat_280997], **kwargs_280998)
            
            # Applying the binary operator '*' (line 589)
            result_mul_281000 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 52), '*', y_280994, sin_call_result_280999)
            
            # Getting the type of 'sin_c' (line 589)
            sin_c_281001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 67), 'sin_c', False)
            # Applying the binary operator '*' (line 589)
            result_mul_281002 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 66), '*', result_mul_281000, sin_c_281001)
            
            # Applying the binary operator '-' (line 589)
            result_sub_281003 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 29), '-', result_mul_280993, result_mul_281002)
            
            # Applying the binary operator 'div' (line 589)
            result_div_281004 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 16), 'div', result_mul_280984, result_sub_281003)
            
            # Processing the call keyword arguments (line 588)
            kwargs_281005 = {}
            # Getting the type of 'np' (line 588)
            np_280980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 26), 'np', False)
            # Obtaining the member 'arctan' of a type (line 588)
            arctan_280981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 26), np_280980, 'arctan')
            # Calling arctan(args, kwargs) (line 588)
            arctan_call_result_281006 = invoke(stypy.reporting.localization.Localization(__file__, 588, 26), arctan_280981, *[result_div_281004], **kwargs_281005)
            
            # Applying the binary operator '+' (line 588)
            result_add_281007 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 18), '+', clong_280979, arctan_call_result_281006)
            
            # Assigning a type to the variable 'lon' (line 588)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'lon', result_add_281007)
            
            # Call to concatenate(...): (line 591)
            # Processing the call arguments (line 591)
            
            # Obtaining an instance of the builtin type 'tuple' (line 591)
            tuple_281010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 591)
            # Adding element type (line 591)
            # Getting the type of 'lon' (line 591)
            lon_281011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 35), 'lon', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 35), tuple_281010, lon_281011)
            # Adding element type (line 591)
            # Getting the type of 'lat' (line 591)
            lat_281012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 40), 'lat', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 35), tuple_281010, lat_281012)
            
            int_281013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 46), 'int')
            # Processing the call keyword arguments (line 591)
            kwargs_281014 = {}
            # Getting the type of 'np' (line 591)
            np_281008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 19), 'np', False)
            # Obtaining the member 'concatenate' of a type (line 591)
            concatenate_281009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 19), np_281008, 'concatenate')
            # Calling concatenate(args, kwargs) (line 591)
            concatenate_call_result_281015 = invoke(stypy.reporting.localization.Localization(__file__, 591, 19), concatenate_281009, *[tuple_281010, int_281013], **kwargs_281014)
            
            # Assigning a type to the variable 'stypy_return_type' (line 591)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'stypy_return_type', concatenate_call_result_281015)
            
            # ################# End of 'transform_non_affine(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'transform_non_affine' in the type store
            # Getting the type of 'stypy_return_type' (line 575)
            stypy_return_type_281016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_281016)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'transform_non_affine'
            return stypy_return_type_281016

        
        # Assigning a Attribute to a Attribute (line 592):
        
        # Assigning a Attribute to a Attribute (line 592):
        # Getting the type of 'Transform' (line 592)
        Transform_281017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 39), 'Transform')
        # Obtaining the member 'transform_non_affine' of a type (line 592)
        transform_non_affine_281018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 39), Transform_281017, 'transform_non_affine')
        # Obtaining the member '__doc__' of a type (line 592)
        doc___281019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 39), transform_non_affine_281018, '__doc__')
        # Getting the type of 'transform_non_affine' (line 592)
        transform_non_affine_281020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'transform_non_affine')
        # Setting the type of the member '__doc__' of a type (line 592)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 8), transform_non_affine_281020, '__doc__', doc___281019)

        @norecursion
        def inverted(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inverted'
            module_type_store = module_type_store.open_function_context('inverted', 594, 8, False)
            # Assigning a type to the variable 'self' (line 595)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedLambertTransform.inverted')
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            InvertedLambertTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLambertTransform.inverted', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inverted', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inverted(...)' code ##################

            
            # Call to LambertTransform(...): (line 595)
            # Processing the call arguments (line 595)
            # Getting the type of 'self' (line 596)
            self_281023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'self', False)
            # Obtaining the member '_center_longitude' of a type (line 596)
            _center_longitude_281024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 16), self_281023, '_center_longitude')
            # Getting the type of 'self' (line 597)
            self_281025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 16), 'self', False)
            # Obtaining the member '_center_latitude' of a type (line 597)
            _center_latitude_281026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 16), self_281025, '_center_latitude')
            # Getting the type of 'self' (line 598)
            self_281027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'self', False)
            # Obtaining the member '_resolution' of a type (line 598)
            _resolution_281028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 16), self_281027, '_resolution')
            # Processing the call keyword arguments (line 595)
            kwargs_281029 = {}
            # Getting the type of 'LambertAxes' (line 595)
            LambertAxes_281021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 19), 'LambertAxes', False)
            # Obtaining the member 'LambertTransform' of a type (line 595)
            LambertTransform_281022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 19), LambertAxes_281021, 'LambertTransform')
            # Calling LambertTransform(args, kwargs) (line 595)
            LambertTransform_call_result_281030 = invoke(stypy.reporting.localization.Localization(__file__, 595, 19), LambertTransform_281022, *[_center_longitude_281024, _center_latitude_281026, _resolution_281028], **kwargs_281029)
            
            # Assigning a type to the variable 'stypy_return_type' (line 595)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'stypy_return_type', LambertTransform_call_result_281030)
            
            # ################# End of 'inverted(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inverted' in the type store
            # Getting the type of 'stypy_return_type' (line 594)
            stypy_return_type_281031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_281031)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inverted'
            return stypy_return_type_281031

        
        # Assigning a Attribute to a Attribute (line 599):
        
        # Assigning a Attribute to a Attribute (line 599):
        # Getting the type of 'Transform' (line 599)
        Transform_281032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 27), 'Transform')
        # Obtaining the member 'inverted' of a type (line 599)
        inverted_281033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 27), Transform_281032, 'inverted')
        # Obtaining the member '__doc__' of a type (line 599)
        doc___281034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 27), inverted_281033, '__doc__')
        # Getting the type of 'inverted' (line 599)
        inverted_281035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'inverted')
        # Setting the type of the member '__doc__' of a type (line 599)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 8), inverted_281035, '__doc__', doc___281034)
    
    # Assigning a type to the variable 'InvertedLambertTransform' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'InvertedLambertTransform', InvertedLambertTransform)

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 601, 4, False)
        # Assigning a type to the variable 'self' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LambertAxes.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Attribute (line 602):
        
        # Assigning a BinOp to a Attribute (line 602):
        # Getting the type of 'np' (line 602)
        np_281036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 30), 'np')
        # Obtaining the member 'pi' of a type (line 602)
        pi_281037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 30), np_281036, 'pi')
        float_281038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 38), 'float')
        # Applying the binary operator 'div' (line 602)
        result_div_281039 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 30), 'div', pi_281037, float_281038)
        
        # Getting the type of 'self' (line 602)
        self_281040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'self')
        # Setting the type of the member '_longitude_cap' of a type (line 602)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 8), self_281040, '_longitude_cap', result_div_281039)
        
        # Assigning a Call to a Attribute (line 603):
        
        # Assigning a Call to a Attribute (line 603):
        
        # Call to pop(...): (line 603)
        # Processing the call arguments (line 603)
        unicode_281043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 44), 'unicode', u'center_longitude')
        float_281044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 64), 'float')
        # Processing the call keyword arguments (line 603)
        kwargs_281045 = {}
        # Getting the type of 'kwargs' (line 603)
        kwargs_281041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 33), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 603)
        pop_281042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 33), kwargs_281041, 'pop')
        # Calling pop(args, kwargs) (line 603)
        pop_call_result_281046 = invoke(stypy.reporting.localization.Localization(__file__, 603, 33), pop_281042, *[unicode_281043, float_281044], **kwargs_281045)
        
        # Getting the type of 'self' (line 603)
        self_281047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'self')
        # Setting the type of the member '_center_longitude' of a type (line 603)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 8), self_281047, '_center_longitude', pop_call_result_281046)
        
        # Assigning a Call to a Attribute (line 604):
        
        # Assigning a Call to a Attribute (line 604):
        
        # Call to pop(...): (line 604)
        # Processing the call arguments (line 604)
        unicode_281050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 43), 'unicode', u'center_latitude')
        float_281051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 62), 'float')
        # Processing the call keyword arguments (line 604)
        kwargs_281052 = {}
        # Getting the type of 'kwargs' (line 604)
        kwargs_281048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 32), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 604)
        pop_281049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 32), kwargs_281048, 'pop')
        # Calling pop(args, kwargs) (line 604)
        pop_call_result_281053 = invoke(stypy.reporting.localization.Localization(__file__, 604, 32), pop_281049, *[unicode_281050, float_281051], **kwargs_281052)
        
        # Getting the type of 'self' (line 604)
        self_281054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'self')
        # Setting the type of the member '_center_latitude' of a type (line 604)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 8), self_281054, '_center_latitude', pop_call_result_281053)
        
        # Call to __init__(...): (line 605)
        # Processing the call arguments (line 605)
        # Getting the type of 'self' (line 605)
        self_281057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 25), 'self', False)
        # Getting the type of 'args' (line 605)
        args_281058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 32), 'args', False)
        # Processing the call keyword arguments (line 605)
        # Getting the type of 'kwargs' (line 605)
        kwargs_281059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 40), 'kwargs', False)
        kwargs_281060 = {'kwargs_281059': kwargs_281059}
        # Getting the type of 'GeoAxes' (line 605)
        GeoAxes_281055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'GeoAxes', False)
        # Obtaining the member '__init__' of a type (line 605)
        init___281056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 8), GeoAxes_281055, '__init__')
        # Calling __init__(args, kwargs) (line 605)
        init___call_result_281061 = invoke(stypy.reporting.localization.Localization(__file__, 605, 8), init___281056, *[self_281057, args_281058], **kwargs_281060)
        
        
        # Call to set_aspect(...): (line 606)
        # Processing the call arguments (line 606)
        unicode_281064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 24), 'unicode', u'equal')
        # Processing the call keyword arguments (line 606)
        unicode_281065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 44), 'unicode', u'box')
        keyword_281066 = unicode_281065
        unicode_281067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 58), 'unicode', u'C')
        keyword_281068 = unicode_281067
        kwargs_281069 = {'adjustable': keyword_281066, 'anchor': keyword_281068}
        # Getting the type of 'self' (line 606)
        self_281062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'self', False)
        # Obtaining the member 'set_aspect' of a type (line 606)
        set_aspect_281063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 8), self_281062, 'set_aspect')
        # Calling set_aspect(args, kwargs) (line 606)
        set_aspect_call_result_281070 = invoke(stypy.reporting.localization.Localization(__file__, 606, 8), set_aspect_281063, *[unicode_281064], **kwargs_281069)
        
        
        # Call to cla(...): (line 607)
        # Processing the call keyword arguments (line 607)
        kwargs_281073 = {}
        # Getting the type of 'self' (line 607)
        self_281071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'self', False)
        # Obtaining the member 'cla' of a type (line 607)
        cla_281072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 8), self_281071, 'cla')
        # Calling cla(args, kwargs) (line 607)
        cla_call_result_281074 = invoke(stypy.reporting.localization.Localization(__file__, 607, 8), cla_281072, *[], **kwargs_281073)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def cla(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cla'
        module_type_store = module_type_store.open_function_context('cla', 609, 4, False)
        # Assigning a type to the variable 'self' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LambertAxes.cla.__dict__.__setitem__('stypy_localization', localization)
        LambertAxes.cla.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LambertAxes.cla.__dict__.__setitem__('stypy_type_store', module_type_store)
        LambertAxes.cla.__dict__.__setitem__('stypy_function_name', 'LambertAxes.cla')
        LambertAxes.cla.__dict__.__setitem__('stypy_param_names_list', [])
        LambertAxes.cla.__dict__.__setitem__('stypy_varargs_param_name', None)
        LambertAxes.cla.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LambertAxes.cla.__dict__.__setitem__('stypy_call_defaults', defaults)
        LambertAxes.cla.__dict__.__setitem__('stypy_call_varargs', varargs)
        LambertAxes.cla.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LambertAxes.cla.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LambertAxes.cla', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to cla(...): (line 610)
        # Processing the call arguments (line 610)
        # Getting the type of 'self' (line 610)
        self_281077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'self', False)
        # Processing the call keyword arguments (line 610)
        kwargs_281078 = {}
        # Getting the type of 'GeoAxes' (line 610)
        GeoAxes_281075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'GeoAxes', False)
        # Obtaining the member 'cla' of a type (line 610)
        cla_281076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 8), GeoAxes_281075, 'cla')
        # Calling cla(args, kwargs) (line 610)
        cla_call_result_281079 = invoke(stypy.reporting.localization.Localization(__file__, 610, 8), cla_281076, *[self_281077], **kwargs_281078)
        
        
        # Call to set_major_formatter(...): (line 611)
        # Processing the call arguments (line 611)
        
        # Call to NullFormatter(...): (line 611)
        # Processing the call keyword arguments (line 611)
        kwargs_281084 = {}
        # Getting the type of 'NullFormatter' (line 611)
        NullFormatter_281083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 39), 'NullFormatter', False)
        # Calling NullFormatter(args, kwargs) (line 611)
        NullFormatter_call_result_281085 = invoke(stypy.reporting.localization.Localization(__file__, 611, 39), NullFormatter_281083, *[], **kwargs_281084)
        
        # Processing the call keyword arguments (line 611)
        kwargs_281086 = {}
        # Getting the type of 'self' (line 611)
        self_281080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'self', False)
        # Obtaining the member 'yaxis' of a type (line 611)
        yaxis_281081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 8), self_281080, 'yaxis')
        # Obtaining the member 'set_major_formatter' of a type (line 611)
        set_major_formatter_281082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 8), yaxis_281081, 'set_major_formatter')
        # Calling set_major_formatter(args, kwargs) (line 611)
        set_major_formatter_call_result_281087 = invoke(stypy.reporting.localization.Localization(__file__, 611, 8), set_major_formatter_281082, *[NullFormatter_call_result_281085], **kwargs_281086)
        
        
        # ################# End of 'cla(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cla' in the type store
        # Getting the type of 'stypy_return_type' (line 609)
        stypy_return_type_281088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_281088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cla'
        return stypy_return_type_281088


    @norecursion
    def _get_core_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_core_transform'
        module_type_store = module_type_store.open_function_context('_get_core_transform', 613, 4, False)
        # Assigning a type to the variable 'self' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_localization', localization)
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_function_name', 'LambertAxes._get_core_transform')
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_param_names_list', ['resolution'])
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LambertAxes._get_core_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LambertAxes._get_core_transform', ['resolution'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_core_transform', localization, ['resolution'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_core_transform(...)' code ##################

        
        # Call to LambertTransform(...): (line 614)
        # Processing the call arguments (line 614)
        # Getting the type of 'self' (line 615)
        self_281091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'self', False)
        # Obtaining the member '_center_longitude' of a type (line 615)
        _center_longitude_281092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 12), self_281091, '_center_longitude')
        # Getting the type of 'self' (line 616)
        self_281093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'self', False)
        # Obtaining the member '_center_latitude' of a type (line 616)
        _center_latitude_281094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 12), self_281093, '_center_latitude')
        # Getting the type of 'resolution' (line 617)
        resolution_281095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'resolution', False)
        # Processing the call keyword arguments (line 614)
        kwargs_281096 = {}
        # Getting the type of 'self' (line 614)
        self_281089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 15), 'self', False)
        # Obtaining the member 'LambertTransform' of a type (line 614)
        LambertTransform_281090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 15), self_281089, 'LambertTransform')
        # Calling LambertTransform(args, kwargs) (line 614)
        LambertTransform_call_result_281097 = invoke(stypy.reporting.localization.Localization(__file__, 614, 15), LambertTransform_281090, *[_center_longitude_281092, _center_latitude_281094, resolution_281095], **kwargs_281096)
        
        # Assigning a type to the variable 'stypy_return_type' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'stypy_return_type', LambertTransform_call_result_281097)
        
        # ################# End of '_get_core_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_core_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 613)
        stypy_return_type_281098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_281098)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_core_transform'
        return stypy_return_type_281098


    @norecursion
    def _get_affine_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_affine_transform'
        module_type_store = module_type_store.open_function_context('_get_affine_transform', 619, 4, False)
        # Assigning a type to the variable 'self' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_localization', localization)
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_function_name', 'LambertAxes._get_affine_transform')
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_param_names_list', [])
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LambertAxes._get_affine_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LambertAxes._get_affine_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_affine_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_affine_transform(...)' code ##################

        
        # Call to translate(...): (line 620)
        # Processing the call arguments (line 620)
        float_281107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 23), 'float')
        float_281108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 28), 'float')
        # Processing the call keyword arguments (line 620)
        kwargs_281109 = {}
        
        # Call to scale(...): (line 620)
        # Processing the call arguments (line 620)
        float_281103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 19), 'float')
        # Processing the call keyword arguments (line 620)
        kwargs_281104 = {}
        
        # Call to Affine2D(...): (line 620)
        # Processing the call keyword arguments (line 620)
        kwargs_281100 = {}
        # Getting the type of 'Affine2D' (line 620)
        Affine2D_281099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 620)
        Affine2D_call_result_281101 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), Affine2D_281099, *[], **kwargs_281100)
        
        # Obtaining the member 'scale' of a type (line 620)
        scale_281102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), Affine2D_call_result_281101, 'scale')
        # Calling scale(args, kwargs) (line 620)
        scale_call_result_281105 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), scale_281102, *[float_281103], **kwargs_281104)
        
        # Obtaining the member 'translate' of a type (line 620)
        translate_281106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), scale_call_result_281105, 'translate')
        # Calling translate(args, kwargs) (line 620)
        translate_call_result_281110 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), translate_281106, *[float_281107, float_281108], **kwargs_281109)
        
        # Assigning a type to the variable 'stypy_return_type' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'stypy_return_type', translate_call_result_281110)
        
        # ################# End of '_get_affine_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_affine_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 619)
        stypy_return_type_281111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_281111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_affine_transform'
        return stypy_return_type_281111


# Assigning a type to the variable 'LambertAxes' (line 506)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 0), 'LambertAxes', LambertAxes)

# Assigning a Str to a Name (line 507):
unicode_281112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 11), 'unicode', u'lambert')
# Getting the type of 'LambertAxes'
LambertAxes_281113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LambertAxes')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LambertAxes_281113, 'name', unicode_281112)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
