
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Streamline plotting for 2D vector fields.
3: 
4: '''
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: 
8: import six
9: from six.moves import xrange
10: 
11: import numpy as np
12: import matplotlib
13: import matplotlib.cm as cm
14: import matplotlib.colors as mcolors
15: import matplotlib.collections as mcollections
16: import matplotlib.lines as mlines
17: import matplotlib.patches as patches
18: 
19: 
20: __all__ = ['streamplot']
21: 
22: 
23: def streamplot(axes, x, y, u, v, density=1, linewidth=None, color=None,
24:                cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
25:                minlength=0.1, transform=None, zorder=None, start_points=None,
26:                maxlength=4.0, integration_direction='both'):
27:     '''Draws streamlines of a vector flow.
28: 
29:     *x*, *y* : 1d arrays
30:         an *evenly spaced* grid.
31:     *u*, *v* : 2d arrays
32:         x and y-velocities. Number of rows should match length of y, and
33:         the number of columns should match x.
34:     *density* : float or 2-tuple
35:         Controls the closeness of streamlines. When `density = 1`, the domain
36:         is divided into a 30x30 grid---*density* linearly scales this grid.
37:         Each cell in the grid can have, at most, one traversing streamline.
38:         For different densities in each direction, use [density_x, density_y].
39:     *linewidth* : numeric or 2d array
40:         vary linewidth when given a 2d array with the same shape as velocities.
41:     *color* : matplotlib color code, or 2d array
42:         Streamline color. When given an array with the same shape as
43:         velocities, *color* values are converted to colors using *cmap*.
44:     *cmap* : :class:`~matplotlib.colors.Colormap`
45:         Colormap used to plot streamlines and arrows. Only necessary when using
46:         an array input for *color*.
47:     *norm* : :class:`~matplotlib.colors.Normalize`
48:         Normalize object used to scale luminance data to 0, 1. If None, stretch
49:         (min, max) to (0, 1). Only necessary when *color* is an array.
50:     *arrowsize* : float
51:         Factor scale arrow size.
52:     *arrowstyle* : str
53:         Arrow style specification.
54:         See :class:`~matplotlib.patches.FancyArrowPatch`.
55:     *minlength* : float
56:         Minimum length of streamline in axes coordinates.
57:     *start_points*: Nx2 array
58:         Coordinates of starting points for the streamlines.
59:         In data coordinates, the same as the ``x`` and ``y`` arrays.
60:     *zorder* : int
61:         any number
62:     *maxlength* : float
63:         Maximum length of streamline in axes coordinates.
64:     *integration_direction* : ['forward', 'backward', 'both']
65:         Integrate the streamline in forward, backward or both directions.
66: 
67:     Returns:
68: 
69:         *stream_container* : StreamplotSet
70:             Container object with attributes
71: 
72:                 - lines: `matplotlib.collections.LineCollection` of streamlines
73: 
74:                 - arrows: collection of `matplotlib.patches.FancyArrowPatch`
75:                   objects representing arrows half-way along stream
76:                   lines.
77: 
78:             This container will probably change in the future to allow changes
79:             to the colormap, alpha, etc. for both lines and arrows, but these
80:             changes should be backward compatible.
81: 
82:     '''
83:     grid = Grid(x, y)
84:     mask = StreamMask(density)
85:     dmap = DomainMap(grid, mask)
86: 
87:     if zorder is None:
88:         zorder = mlines.Line2D.zorder
89: 
90:     # default to data coordinates
91:     if transform is None:
92:         transform = axes.transData
93: 
94:     if color is None:
95:         color = axes._get_lines.get_next_color()
96: 
97:     if linewidth is None:
98:         linewidth = matplotlib.rcParams['lines.linewidth']
99: 
100:     line_kw = {}
101:     arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)
102: 
103:     if integration_direction not in ['both', 'forward', 'backward']:
104:         errstr = ("Integration direction '%s' not recognised. "
105:                   "Expected 'both', 'forward' or 'backward'." %
106:                   integration_direction)
107:         raise ValueError(errstr)
108: 
109:     if integration_direction == 'both':
110:         maxlength /= 2.
111: 
112:     use_multicolor_lines = isinstance(color, np.ndarray)
113:     if use_multicolor_lines:
114:         if color.shape != grid.shape:
115:             msg = "If 'color' is given, must have the shape of 'Grid(x,y)'"
116:             raise ValueError(msg)
117:         line_colors = []
118:         color = np.ma.masked_invalid(color)
119:     else:
120:         line_kw['color'] = color
121:         arrow_kw['color'] = color
122: 
123:     if isinstance(linewidth, np.ndarray):
124:         if linewidth.shape != grid.shape:
125:             msg = "If 'linewidth' is given, must have the shape of 'Grid(x,y)'"
126:             raise ValueError(msg)
127:         line_kw['linewidth'] = []
128:     else:
129:         line_kw['linewidth'] = linewidth
130:         arrow_kw['linewidth'] = linewidth
131: 
132:     line_kw['zorder'] = zorder
133:     arrow_kw['zorder'] = zorder
134: 
135:     ## Sanity checks.
136:     if (u.shape != grid.shape) or (v.shape != grid.shape):
137:         msg = "'u' and 'v' must be of shape 'Grid(x,y)'"
138:         raise ValueError(msg)
139: 
140:     u = np.ma.masked_invalid(u)
141:     v = np.ma.masked_invalid(v)
142: 
143:     integrate = get_integrator(u, v, dmap, minlength, maxlength,
144:                                integration_direction)
145: 
146:     trajectories = []
147:     if start_points is None:
148:         for xm, ym in _gen_starting_points(mask.shape):
149:             if mask[ym, xm] == 0:
150:                 xg, yg = dmap.mask2grid(xm, ym)
151:                 t = integrate(xg, yg)
152:                 if t is not None:
153:                     trajectories.append(t)
154:     else:
155:         sp2 = np.asanyarray(start_points, dtype=float).copy()
156: 
157:         # Check if start_points are outside the data boundaries
158:         for xs, ys in sp2:
159:             if not (grid.x_origin <= xs <= grid.x_origin + grid.width
160:                     and grid.y_origin <= ys <= grid.y_origin + grid.height):
161:                 raise ValueError("Starting point ({}, {}) outside of data "
162:                                  "boundaries".format(xs, ys))
163: 
164:         # Convert start_points from data to array coords
165:         # Shift the seed points from the bottom left of the data so that
166:         # data2grid works properly.
167:         sp2[:, 0] -= grid.x_origin
168:         sp2[:, 1] -= grid.y_origin
169: 
170:         for xs, ys in sp2:
171:             xg, yg = dmap.data2grid(xs, ys)
172:             t = integrate(xg, yg)
173:             if t is not None:
174:                 trajectories.append(t)
175: 
176:     if use_multicolor_lines:
177:         if norm is None:
178:             norm = mcolors.Normalize(color.min(), color.max())
179:         if cmap is None:
180:             cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
181:         else:
182:             cmap = cm.get_cmap(cmap)
183: 
184:     streamlines = []
185:     arrows = []
186:     for t in trajectories:
187:         tgx = np.array(t[0])
188:         tgy = np.array(t[1])
189:         # Rescale from grid-coordinates to data-coordinates.
190:         tx, ty = dmap.grid2data(*np.array(t))
191:         tx += grid.x_origin
192:         ty += grid.y_origin
193: 
194:         points = np.transpose([tx, ty]).reshape(-1, 1, 2)
195:         streamlines.extend(np.hstack([points[:-1], points[1:]]))
196: 
197:         # Add arrows half way along each trajectory.
198:         s = np.cumsum(np.sqrt(np.diff(tx) ** 2 + np.diff(ty) ** 2))
199:         n = np.searchsorted(s, s[-1] / 2.)
200:         arrow_tail = (tx[n], ty[n])
201:         arrow_head = (np.mean(tx[n:n + 2]), np.mean(ty[n:n + 2]))
202: 
203:         if isinstance(linewidth, np.ndarray):
204:             line_widths = interpgrid(linewidth, tgx, tgy)[:-1]
205:             line_kw['linewidth'].extend(line_widths)
206:             arrow_kw['linewidth'] = line_widths[n]
207: 
208:         if use_multicolor_lines:
209:             color_values = interpgrid(color, tgx, tgy)[:-1]
210:             line_colors.append(color_values)
211:             arrow_kw['color'] = cmap(norm(color_values[n]))
212: 
213:         p = patches.FancyArrowPatch(
214:             arrow_tail, arrow_head, transform=transform, **arrow_kw)
215:         axes.add_patch(p)
216:         arrows.append(p)
217: 
218:     lc = mcollections.LineCollection(
219:         streamlines, transform=transform, **line_kw)
220:     lc.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
221:     lc.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
222:     if use_multicolor_lines:
223:         lc.set_array(np.ma.hstack(line_colors))
224:         lc.set_cmap(cmap)
225:         lc.set_norm(norm)
226:     axes.add_collection(lc)
227:     axes.autoscale_view()
228: 
229:     ac = matplotlib.collections.PatchCollection(arrows)
230:     stream_container = StreamplotSet(lc, ac)
231:     return stream_container
232: 
233: 
234: class StreamplotSet(object):
235: 
236:     def __init__(self, lines, arrows, **kwargs):
237:         self.lines = lines
238:         self.arrows = arrows
239: 
240: 
241: # Coordinate definitions
242: # ========================
243: 
244: class DomainMap(object):
245:     '''Map representing different coordinate systems.
246: 
247:     Coordinate definitions:
248: 
249:     * axes-coordinates goes from 0 to 1 in the domain.
250:     * data-coordinates are specified by the input x-y coordinates.
251:     * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
252:       where N and M match the shape of the input data.
253:     * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
254:       where N and M are user-specified to control the density of streamlines.
255: 
256:     This class also has methods for adding trajectories to the StreamMask.
257:     Before adding a trajectory, run `start_trajectory` to keep track of regions
258:     crossed by a given trajectory. Later, if you decide the trajectory is bad
259:     (e.g., if the trajectory is very short) just call `undo_trajectory`.
260:     '''
261: 
262:     def __init__(self, grid, mask):
263:         self.grid = grid
264:         self.mask = mask
265:         # Constants for conversion between grid- and mask-coordinates
266:         self.x_grid2mask = float(mask.nx - 1) / grid.nx
267:         self.y_grid2mask = float(mask.ny - 1) / grid.ny
268: 
269:         self.x_mask2grid = 1. / self.x_grid2mask
270:         self.y_mask2grid = 1. / self.y_grid2mask
271: 
272:         self.x_data2grid = 1. / grid.dx
273:         self.y_data2grid = 1. / grid.dy
274: 
275:     def grid2mask(self, xi, yi):
276:         '''Return nearest space in mask-coords from given grid-coords.'''
277:         return (int((xi * self.x_grid2mask) + 0.5),
278:                 int((yi * self.y_grid2mask) + 0.5))
279: 
280:     def mask2grid(self, xm, ym):
281:         return xm * self.x_mask2grid, ym * self.y_mask2grid
282: 
283:     def data2grid(self, xd, yd):
284:         return xd * self.x_data2grid, yd * self.y_data2grid
285: 
286:     def grid2data(self, xg, yg):
287:         return xg / self.x_data2grid, yg / self.y_data2grid
288: 
289:     def start_trajectory(self, xg, yg):
290:         xm, ym = self.grid2mask(xg, yg)
291:         self.mask._start_trajectory(xm, ym)
292: 
293:     def reset_start_point(self, xg, yg):
294:         xm, ym = self.grid2mask(xg, yg)
295:         self.mask._current_xy = (xm, ym)
296: 
297:     def update_trajectory(self, xg, yg):
298:         if not self.grid.within_grid(xg, yg):
299:             raise InvalidIndexError
300:         xm, ym = self.grid2mask(xg, yg)
301:         self.mask._update_trajectory(xm, ym)
302: 
303:     def undo_trajectory(self):
304:         self.mask._undo_trajectory()
305: 
306: 
307: class Grid(object):
308:     '''Grid of data.'''
309:     def __init__(self, x, y):
310: 
311:         if x.ndim == 1:
312:             pass
313:         elif x.ndim == 2:
314:             x_row = x[0, :]
315:             if not np.allclose(x_row, x):
316:                 raise ValueError("The rows of 'x' must be equal")
317:             x = x_row
318:         else:
319:             raise ValueError("'x' can have at maximum 2 dimensions")
320: 
321:         if y.ndim == 1:
322:             pass
323:         elif y.ndim == 2:
324:             y_col = y[:, 0]
325:             if not np.allclose(y_col, y.T):
326:                 raise ValueError("The columns of 'y' must be equal")
327:             y = y_col
328:         else:
329:             raise ValueError("'y' can have at maximum 2 dimensions")
330: 
331:         self.nx = len(x)
332:         self.ny = len(y)
333: 
334:         self.dx = x[1] - x[0]
335:         self.dy = y[1] - y[0]
336: 
337:         self.x_origin = x[0]
338:         self.y_origin = y[0]
339: 
340:         self.width = x[-1] - x[0]
341:         self.height = y[-1] - y[0]
342: 
343:     @property
344:     def shape(self):
345:         return self.ny, self.nx
346: 
347:     def within_grid(self, xi, yi):
348:         '''Return True if point is a valid index of grid.'''
349:         # Note that xi/yi can be floats; so, for example, we can't simply check
350:         # `xi < self.nx` since `xi` can be `self.nx - 1 < xi < self.nx`
351:         return xi >= 0 and xi <= self.nx - 1 and yi >= 0 and yi <= self.ny - 1
352: 
353: 
354: class StreamMask(object):
355:     '''Mask to keep track of discrete regions crossed by streamlines.
356: 
357:     The resolution of this grid determines the approximate spacing between
358:     trajectories. Streamlines are only allowed to pass through zeroed cells:
359:     When a streamline enters a cell, that cell is set to 1, and no new
360:     streamlines are allowed to enter.
361:     '''
362: 
363:     def __init__(self, density):
364:         if np.isscalar(density):
365:             if density <= 0:
366:                 raise ValueError("If a scalar, 'density' must be positive")
367:             self.nx = self.ny = int(30 * density)
368:         else:
369:             if len(density) != 2:
370:                 raise ValueError("'density' can have at maximum 2 dimensions")
371:             self.nx = int(30 * density[0])
372:             self.ny = int(30 * density[1])
373:         self._mask = np.zeros((self.ny, self.nx))
374:         self.shape = self._mask.shape
375: 
376:         self._current_xy = None
377: 
378:     def __getitem__(self, *args):
379:         return self._mask.__getitem__(*args)
380: 
381:     def _start_trajectory(self, xm, ym):
382:         '''Start recording streamline trajectory'''
383:         self._traj = []
384:         self._update_trajectory(xm, ym)
385: 
386:     def _undo_trajectory(self):
387:         '''Remove current trajectory from mask'''
388:         for t in self._traj:
389:             self._mask.__setitem__(t, 0)
390: 
391:     def _update_trajectory(self, xm, ym):
392:         '''Update current trajectory position in mask.
393: 
394:         If the new position has already been filled, raise `InvalidIndexError`.
395:         '''
396:         if self._current_xy != (xm, ym):
397:             if self[ym, xm] == 0:
398:                 self._traj.append((ym, xm))
399:                 self._mask[ym, xm] = 1
400:                 self._current_xy = (xm, ym)
401:             else:
402:                 raise InvalidIndexError
403: 
404: 
405: class InvalidIndexError(Exception):
406:     pass
407: 
408: 
409: class TerminateTrajectory(Exception):
410:     pass
411: 
412: 
413: # Integrator definitions
414: #========================
415: 
416: def get_integrator(u, v, dmap, minlength, maxlength, integration_direction):
417: 
418:     # rescale velocity onto grid-coordinates for integrations.
419:     u, v = dmap.data2grid(u, v)
420: 
421:     # speed (path length) will be in axes-coordinates
422:     u_ax = u / dmap.grid.nx
423:     v_ax = v / dmap.grid.ny
424:     speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)
425: 
426:     def forward_time(xi, yi):
427:         ds_dt = interpgrid(speed, xi, yi)
428:         if ds_dt == 0:
429:             raise TerminateTrajectory()
430:         dt_ds = 1. / ds_dt
431:         ui = interpgrid(u, xi, yi)
432:         vi = interpgrid(v, xi, yi)
433:         return ui * dt_ds, vi * dt_ds
434: 
435:     def backward_time(xi, yi):
436:         dxi, dyi = forward_time(xi, yi)
437:         return -dxi, -dyi
438: 
439:     def integrate(x0, y0):
440:         '''Return x, y grid-coordinates of trajectory based on starting point.
441: 
442:         Integrate both forward and backward in time from starting point in
443:         grid coordinates.
444: 
445:         Integration is terminated when a trajectory reaches a domain boundary
446:         or when it crosses into an already occupied cell in the StreamMask. The
447:         resulting trajectory is None if it is shorter than `minlength`.
448:         '''
449: 
450:         stotal, x_traj, y_traj = 0., [], []
451: 
452:         try:
453:             dmap.start_trajectory(x0, y0)
454:         except InvalidIndexError:
455:             return None
456:         if integration_direction in ['both', 'backward']:
457:             s, xt, yt = _integrate_rk12(x0, y0, dmap, backward_time, maxlength)
458:             stotal += s
459:             x_traj += xt[::-1]
460:             y_traj += yt[::-1]
461: 
462:         if integration_direction in ['both', 'forward']:
463:             dmap.reset_start_point(x0, y0)
464:             s, xt, yt = _integrate_rk12(x0, y0, dmap, forward_time, maxlength)
465:             if len(x_traj) > 0:
466:                 xt = xt[1:]
467:                 yt = yt[1:]
468:             stotal += s
469:             x_traj += xt
470:             y_traj += yt
471: 
472:         if stotal > minlength:
473:             return x_traj, y_traj
474:         else:  # reject short trajectories
475:             dmap.undo_trajectory()
476:             return None
477: 
478:     return integrate
479: 
480: 
481: def _integrate_rk12(x0, y0, dmap, f, maxlength):
482:     '''2nd-order Runge-Kutta algorithm with adaptive step size.
483: 
484:     This method is also referred to as the improved Euler's method, or Heun's
485:     method. This method is favored over higher-order methods because:
486: 
487:     1. To get decent looking trajectories and to sample every mask cell
488:        on the trajectory we need a small timestep, so a lower order
489:        solver doesn't hurt us unless the data is *very* high resolution.
490:        In fact, for cases where the user inputs
491:        data smaller or of similar grid size to the mask grid, the higher
492:        order corrections are negligible because of the very fast linear
493:        interpolation used in `interpgrid`.
494: 
495:     2. For high resolution input data (i.e. beyond the mask
496:        resolution), we must reduce the timestep. Therefore, an adaptive
497:        timestep is more suited to the problem as this would be very hard
498:        to judge automatically otherwise.
499: 
500:     This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
501:     solvers in most setups on my machine. I would recommend removing the
502:     other two to keep things simple.
503:     '''
504:     # This error is below that needed to match the RK4 integrator. It
505:     # is set for visual reasons -- too low and corners start
506:     # appearing ugly and jagged. Can be tuned.
507:     maxerror = 0.003
508: 
509:     # This limit is important (for all integrators) to avoid the
510:     # trajectory skipping some mask cells. We could relax this
511:     # condition if we use the code which is commented out below to
512:     # increment the location gradually. However, due to the efficient
513:     # nature of the interpolation, this doesn't boost speed by much
514:     # for quite a bit of complexity.
515:     maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 0.1)
516: 
517:     ds = maxds
518:     stotal = 0
519:     xi = x0
520:     yi = y0
521:     xf_traj = []
522:     yf_traj = []
523: 
524:     while dmap.grid.within_grid(xi, yi):
525:         xf_traj.append(xi)
526:         yf_traj.append(yi)
527:         try:
528:             k1x, k1y = f(xi, yi)
529:             k2x, k2y = f(xi + ds * k1x,
530:                          yi + ds * k1y)
531:         except IndexError:
532:             # Out of the domain on one of the intermediate integration steps.
533:             # Take an Euler step to the boundary to improve neatness.
534:             ds, xf_traj, yf_traj = _euler_step(xf_traj, yf_traj, dmap, f)
535:             stotal += ds
536:             break
537:         except TerminateTrajectory:
538:             break
539: 
540:         dx1 = ds * k1x
541:         dy1 = ds * k1y
542:         dx2 = ds * 0.5 * (k1x + k2x)
543:         dy2 = ds * 0.5 * (k1y + k2y)
544: 
545:         nx, ny = dmap.grid.shape
546:         # Error is normalized to the axes coordinates
547:         error = np.sqrt(((dx2 - dx1) / nx) ** 2 + ((dy2 - dy1) / ny) ** 2)
548: 
549:         # Only save step if within error tolerance
550:         if error < maxerror:
551:             xi += dx2
552:             yi += dy2
553:             try:
554:                 dmap.update_trajectory(xi, yi)
555:             except InvalidIndexError:
556:                 break
557:             if (stotal + ds) > maxlength:
558:                 break
559:             stotal += ds
560: 
561:         # recalculate stepsize based on step error
562:         if error == 0:
563:             ds = maxds
564:         else:
565:             ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)
566: 
567:     return stotal, xf_traj, yf_traj
568: 
569: 
570: def _euler_step(xf_traj, yf_traj, dmap, f):
571:     '''Simple Euler integration step that extends streamline to boundary.'''
572:     ny, nx = dmap.grid.shape
573:     xi = xf_traj[-1]
574:     yi = yf_traj[-1]
575:     cx, cy = f(xi, yi)
576:     if cx == 0:
577:         dsx = np.inf
578:     elif cx < 0:
579:         dsx = xi / -cx
580:     else:
581:         dsx = (nx - 1 - xi) / cx
582:     if cy == 0:
583:         dsy = np.inf
584:     elif cy < 0:
585:         dsy = yi / -cy
586:     else:
587:         dsy = (ny - 1 - yi) / cy
588:     ds = min(dsx, dsy)
589:     xf_traj.append(xi + cx * ds)
590:     yf_traj.append(yi + cy * ds)
591:     return ds, xf_traj, yf_traj
592: 
593: 
594: # Utility functions
595: # ========================
596: 
597: def interpgrid(a, xi, yi):
598:     '''Fast 2D, linear interpolation on an integer grid'''
599: 
600:     Ny, Nx = np.shape(a)
601:     if isinstance(xi, np.ndarray):
602:         x = xi.astype(int)
603:         y = yi.astype(int)
604:         # Check that xn, yn don't exceed max index
605:         xn = np.clip(x + 1, 0, Nx - 1)
606:         yn = np.clip(y + 1, 0, Ny - 1)
607:     else:
608:         x = int(xi)
609:         y = int(yi)
610:         # conditional is faster than clipping for integers
611:         if x == (Nx - 2):
612:             xn = x
613:         else:
614:             xn = x + 1
615:         if y == (Ny - 2):
616:             yn = y
617:         else:
618:             yn = y + 1
619: 
620:     a00 = a[y, x]
621:     a01 = a[y, xn]
622:     a10 = a[yn, x]
623:     a11 = a[yn, xn]
624:     xt = xi - x
625:     yt = yi - y
626:     a0 = a00 * (1 - xt) + a01 * xt
627:     a1 = a10 * (1 - xt) + a11 * xt
628:     ai = a0 * (1 - yt) + a1 * yt
629: 
630:     if not isinstance(xi, np.ndarray):
631:         if np.ma.is_masked(ai):
632:             raise TerminateTrajectory
633: 
634:     return ai
635: 
636: 
637: def _gen_starting_points(shape):
638:     '''Yield starting points for streamlines.
639: 
640:     Trying points on the boundary first gives higher quality streamlines.
641:     This algorithm starts with a point on the mask corner and spirals inward.
642:     This algorithm is inefficient, but fast compared to rest of streamplot.
643:     '''
644:     ny, nx = shape
645:     xfirst = 0
646:     yfirst = 1
647:     xlast = nx - 1
648:     ylast = ny - 1
649:     x, y = 0, 0
650:     i = 0
651:     direction = 'right'
652:     for i in xrange(nx * ny):
653: 
654:         yield x, y
655: 
656:         if direction == 'right':
657:             x += 1
658:             if x >= xlast:
659:                 xlast -= 1
660:                 direction = 'up'
661:         elif direction == 'up':
662:             y += 1
663:             if y >= ylast:
664:                 ylast -= 1
665:                 direction = 'left'
666:         elif direction == 'left':
667:             x -= 1
668:             if x <= xfirst:
669:                 xfirst += 1
670:                 direction = 'down'
671:         elif direction == 'down':
672:             y -= 1
673:             if y <= yfirst:
674:                 yfirst += 1
675:                 direction = 'right'
676: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_133226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'unicode', u'\nStreamline plotting for 2D vector fields.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import six' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133227 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_133227) is not StypyTypeError):

    if (import_133227 != 'pyd_module'):
        __import__(import_133227)
        sys_modules_133228 = sys.modules[import_133227]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_133228.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_133227)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from six.moves import xrange' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133229 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves')

if (type(import_133229) is not StypyTypeError):

    if (import_133229 != 'pyd_module'):
        __import__(import_133229)
        sys_modules_133230 = sys.modules[import_133229]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', sys_modules_133230.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_133230, sys_modules_133230.module_type_store, module_type_store)
    else:
        from six.moves import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'six.moves' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', import_133229)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133231 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_133231) is not StypyTypeError):

    if (import_133231 != 'pyd_module'):
        __import__(import_133231)
        sys_modules_133232 = sys.modules[import_133231]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_133232.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_133231)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import matplotlib' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133233 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib')

if (type(import_133233) is not StypyTypeError):

    if (import_133233 != 'pyd_module'):
        __import__(import_133233)
        sys_modules_133234 = sys.modules[import_133233]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', sys_modules_133234.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', import_133233)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import matplotlib.cm' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133235 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.cm')

if (type(import_133235) is not StypyTypeError):

    if (import_133235 != 'pyd_module'):
        __import__(import_133235)
        sys_modules_133236 = sys.modules[import_133235]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'cm', sys_modules_133236.module_type_store, module_type_store)
    else:
        import matplotlib.cm as cm

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'cm', matplotlib.cm, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cm' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.cm', import_133235)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import matplotlib.colors' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133237 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.colors')

if (type(import_133237) is not StypyTypeError):

    if (import_133237 != 'pyd_module'):
        __import__(import_133237)
        sys_modules_133238 = sys.modules[import_133237]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'mcolors', sys_modules_133238.module_type_store, module_type_store)
    else:
        import matplotlib.colors as mcolors

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'mcolors', matplotlib.colors, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.colors' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.colors', import_133237)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import matplotlib.collections' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133239 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.collections')

if (type(import_133239) is not StypyTypeError):

    if (import_133239 != 'pyd_module'):
        __import__(import_133239)
        sys_modules_133240 = sys.modules[import_133239]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'mcollections', sys_modules_133240.module_type_store, module_type_store)
    else:
        import matplotlib.collections as mcollections

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'mcollections', matplotlib.collections, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.collections' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.collections', import_133239)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import matplotlib.lines' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133241 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.lines')

if (type(import_133241) is not StypyTypeError):

    if (import_133241 != 'pyd_module'):
        __import__(import_133241)
        sys_modules_133242 = sys.modules[import_133241]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'mlines', sys_modules_133242.module_type_store, module_type_store)
    else:
        import matplotlib.lines as mlines

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'mlines', matplotlib.lines, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.lines' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.lines', import_133241)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import matplotlib.patches' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_133243 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.patches')

if (type(import_133243) is not StypyTypeError):

    if (import_133243 != 'pyd_module'):
        __import__(import_133243)
        sys_modules_133244 = sys.modules[import_133243]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'patches', sys_modules_133244.module_type_store, module_type_store)
    else:
        import matplotlib.patches as patches

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'patches', matplotlib.patches, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.patches', import_133243)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


# Assigning a List to a Name (line 20):

# Assigning a List to a Name (line 20):
__all__ = [u'streamplot']
module_type_store.set_exportable_members([u'streamplot'])

# Obtaining an instance of the builtin type 'list' (line 20)
list_133245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
unicode_133246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'unicode', u'streamplot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_133245, unicode_133246)

# Assigning a type to the variable '__all__' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '__all__', list_133245)

@norecursion
def streamplot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_133247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 41), 'int')
    # Getting the type of 'None' (line 23)
    None_133248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 54), 'None')
    # Getting the type of 'None' (line 23)
    None_133249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 66), 'None')
    # Getting the type of 'None' (line 24)
    None_133250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'None')
    # Getting the type of 'None' (line 24)
    None_133251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'None')
    int_133252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 47), 'int')
    unicode_133253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 61), 'unicode', u'-|>')
    float_133254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'float')
    # Getting the type of 'None' (line 25)
    None_133255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 40), 'None')
    # Getting the type of 'None' (line 25)
    None_133256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 53), 'None')
    # Getting the type of 'None' (line 25)
    None_133257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 72), 'None')
    float_133258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'float')
    unicode_133259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 52), 'unicode', u'both')
    defaults = [int_133247, None_133248, None_133249, None_133250, None_133251, int_133252, unicode_133253, float_133254, None_133255, None_133256, None_133257, float_133258, unicode_133259]
    # Create a new context for function 'streamplot'
    module_type_store = module_type_store.open_function_context('streamplot', 23, 0, False)
    
    # Passed parameters checking function
    streamplot.stypy_localization = localization
    streamplot.stypy_type_of_self = None
    streamplot.stypy_type_store = module_type_store
    streamplot.stypy_function_name = 'streamplot'
    streamplot.stypy_param_names_list = ['axes', 'x', 'y', 'u', 'v', 'density', 'linewidth', 'color', 'cmap', 'norm', 'arrowsize', 'arrowstyle', 'minlength', 'transform', 'zorder', 'start_points', 'maxlength', 'integration_direction']
    streamplot.stypy_varargs_param_name = None
    streamplot.stypy_kwargs_param_name = None
    streamplot.stypy_call_defaults = defaults
    streamplot.stypy_call_varargs = varargs
    streamplot.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'streamplot', ['axes', 'x', 'y', 'u', 'v', 'density', 'linewidth', 'color', 'cmap', 'norm', 'arrowsize', 'arrowstyle', 'minlength', 'transform', 'zorder', 'start_points', 'maxlength', 'integration_direction'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'streamplot', localization, ['axes', 'x', 'y', 'u', 'v', 'density', 'linewidth', 'color', 'cmap', 'norm', 'arrowsize', 'arrowstyle', 'minlength', 'transform', 'zorder', 'start_points', 'maxlength', 'integration_direction'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'streamplot(...)' code ##################

    unicode_133260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'unicode', u"Draws streamlines of a vector flow.\n\n    *x*, *y* : 1d arrays\n        an *evenly spaced* grid.\n    *u*, *v* : 2d arrays\n        x and y-velocities. Number of rows should match length of y, and\n        the number of columns should match x.\n    *density* : float or 2-tuple\n        Controls the closeness of streamlines. When `density = 1`, the domain\n        is divided into a 30x30 grid---*density* linearly scales this grid.\n        Each cell in the grid can have, at most, one traversing streamline.\n        For different densities in each direction, use [density_x, density_y].\n    *linewidth* : numeric or 2d array\n        vary linewidth when given a 2d array with the same shape as velocities.\n    *color* : matplotlib color code, or 2d array\n        Streamline color. When given an array with the same shape as\n        velocities, *color* values are converted to colors using *cmap*.\n    *cmap* : :class:`~matplotlib.colors.Colormap`\n        Colormap used to plot streamlines and arrows. Only necessary when using\n        an array input for *color*.\n    *norm* : :class:`~matplotlib.colors.Normalize`\n        Normalize object used to scale luminance data to 0, 1. If None, stretch\n        (min, max) to (0, 1). Only necessary when *color* is an array.\n    *arrowsize* : float\n        Factor scale arrow size.\n    *arrowstyle* : str\n        Arrow style specification.\n        See :class:`~matplotlib.patches.FancyArrowPatch`.\n    *minlength* : float\n        Minimum length of streamline in axes coordinates.\n    *start_points*: Nx2 array\n        Coordinates of starting points for the streamlines.\n        In data coordinates, the same as the ``x`` and ``y`` arrays.\n    *zorder* : int\n        any number\n    *maxlength* : float\n        Maximum length of streamline in axes coordinates.\n    *integration_direction* : ['forward', 'backward', 'both']\n        Integrate the streamline in forward, backward or both directions.\n\n    Returns:\n\n        *stream_container* : StreamplotSet\n            Container object with attributes\n\n                - lines: `matplotlib.collections.LineCollection` of streamlines\n\n                - arrows: collection of `matplotlib.patches.FancyArrowPatch`\n                  objects representing arrows half-way along stream\n                  lines.\n\n            This container will probably change in the future to allow changes\n            to the colormap, alpha, etc. for both lines and arrows, but these\n            changes should be backward compatible.\n\n    ")
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to Grid(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'x' (line 83)
    x_133262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'x', False)
    # Getting the type of 'y' (line 83)
    y_133263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'y', False)
    # Processing the call keyword arguments (line 83)
    kwargs_133264 = {}
    # Getting the type of 'Grid' (line 83)
    Grid_133261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'Grid', False)
    # Calling Grid(args, kwargs) (line 83)
    Grid_call_result_133265 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), Grid_133261, *[x_133262, y_133263], **kwargs_133264)
    
    # Assigning a type to the variable 'grid' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'grid', Grid_call_result_133265)
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to StreamMask(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'density' (line 84)
    density_133267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'density', False)
    # Processing the call keyword arguments (line 84)
    kwargs_133268 = {}
    # Getting the type of 'StreamMask' (line 84)
    StreamMask_133266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'StreamMask', False)
    # Calling StreamMask(args, kwargs) (line 84)
    StreamMask_call_result_133269 = invoke(stypy.reporting.localization.Localization(__file__, 84, 11), StreamMask_133266, *[density_133267], **kwargs_133268)
    
    # Assigning a type to the variable 'mask' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'mask', StreamMask_call_result_133269)
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to DomainMap(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'grid' (line 85)
    grid_133271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'grid', False)
    # Getting the type of 'mask' (line 85)
    mask_133272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'mask', False)
    # Processing the call keyword arguments (line 85)
    kwargs_133273 = {}
    # Getting the type of 'DomainMap' (line 85)
    DomainMap_133270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'DomainMap', False)
    # Calling DomainMap(args, kwargs) (line 85)
    DomainMap_call_result_133274 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), DomainMap_133270, *[grid_133271, mask_133272], **kwargs_133273)
    
    # Assigning a type to the variable 'dmap' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'dmap', DomainMap_call_result_133274)
    
    # Type idiom detected: calculating its left and rigth part (line 87)
    # Getting the type of 'zorder' (line 87)
    zorder_133275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 7), 'zorder')
    # Getting the type of 'None' (line 87)
    None_133276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'None')
    
    (may_be_133277, more_types_in_union_133278) = may_be_none(zorder_133275, None_133276)

    if may_be_133277:

        if more_types_in_union_133278:
            # Runtime conditional SSA (line 87)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 88):
        
        # Assigning a Attribute to a Name (line 88):
        # Getting the type of 'mlines' (line 88)
        mlines_133279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'mlines')
        # Obtaining the member 'Line2D' of a type (line 88)
        Line2D_133280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), mlines_133279, 'Line2D')
        # Obtaining the member 'zorder' of a type (line 88)
        zorder_133281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), Line2D_133280, 'zorder')
        # Assigning a type to the variable 'zorder' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'zorder', zorder_133281)

        if more_types_in_union_133278:
            # SSA join for if statement (line 87)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 91)
    # Getting the type of 'transform' (line 91)
    transform_133282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 7), 'transform')
    # Getting the type of 'None' (line 91)
    None_133283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'None')
    
    (may_be_133284, more_types_in_union_133285) = may_be_none(transform_133282, None_133283)

    if may_be_133284:

        if more_types_in_union_133285:
            # Runtime conditional SSA (line 91)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 92):
        
        # Assigning a Attribute to a Name (line 92):
        # Getting the type of 'axes' (line 92)
        axes_133286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'axes')
        # Obtaining the member 'transData' of a type (line 92)
        transData_133287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 20), axes_133286, 'transData')
        # Assigning a type to the variable 'transform' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'transform', transData_133287)

        if more_types_in_union_133285:
            # SSA join for if statement (line 91)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 94)
    # Getting the type of 'color' (line 94)
    color_133288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'color')
    # Getting the type of 'None' (line 94)
    None_133289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'None')
    
    (may_be_133290, more_types_in_union_133291) = may_be_none(color_133288, None_133289)

    if may_be_133290:

        if more_types_in_union_133291:
            # Runtime conditional SSA (line 94)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to get_next_color(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_133295 = {}
        # Getting the type of 'axes' (line 95)
        axes_133292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'axes', False)
        # Obtaining the member '_get_lines' of a type (line 95)
        _get_lines_133293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), axes_133292, '_get_lines')
        # Obtaining the member 'get_next_color' of a type (line 95)
        get_next_color_133294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), _get_lines_133293, 'get_next_color')
        # Calling get_next_color(args, kwargs) (line 95)
        get_next_color_call_result_133296 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), get_next_color_133294, *[], **kwargs_133295)
        
        # Assigning a type to the variable 'color' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'color', get_next_color_call_result_133296)

        if more_types_in_union_133291:
            # SSA join for if statement (line 94)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 97)
    # Getting the type of 'linewidth' (line 97)
    linewidth_133297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 7), 'linewidth')
    # Getting the type of 'None' (line 97)
    None_133298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'None')
    
    (may_be_133299, more_types_in_union_133300) = may_be_none(linewidth_133297, None_133298)

    if may_be_133299:

        if more_types_in_union_133300:
            # Runtime conditional SSA (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 98):
        
        # Assigning a Subscript to a Name (line 98):
        
        # Obtaining the type of the subscript
        unicode_133301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 40), 'unicode', u'lines.linewidth')
        # Getting the type of 'matplotlib' (line 98)
        matplotlib_133302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'matplotlib')
        # Obtaining the member 'rcParams' of a type (line 98)
        rcParams_133303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), matplotlib_133302, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___133304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), rcParams_133303, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_133305 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), getitem___133304, unicode_133301)
        
        # Assigning a type to the variable 'linewidth' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'linewidth', subscript_call_result_133305)

        if more_types_in_union_133300:
            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Dict to a Name (line 100):
    
    # Assigning a Dict to a Name (line 100):
    
    # Obtaining an instance of the builtin type 'dict' (line 100)
    dict_133306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 100)
    
    # Assigning a type to the variable 'line_kw' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'line_kw', dict_133306)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to dict(...): (line 101)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'arrowstyle' (line 101)
    arrowstyle_133308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'arrowstyle', False)
    keyword_133309 = arrowstyle_133308
    int_133310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 58), 'int')
    # Getting the type of 'arrowsize' (line 101)
    arrowsize_133311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 63), 'arrowsize', False)
    # Applying the binary operator '*' (line 101)
    result_mul_133312 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 58), '*', int_133310, arrowsize_133311)
    
    keyword_133313 = result_mul_133312
    kwargs_133314 = {'mutation_scale': keyword_133313, 'arrowstyle': keyword_133309}
    # Getting the type of 'dict' (line 101)
    dict_133307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'dict', False)
    # Calling dict(args, kwargs) (line 101)
    dict_call_result_133315 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), dict_133307, *[], **kwargs_133314)
    
    # Assigning a type to the variable 'arrow_kw' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'arrow_kw', dict_call_result_133315)
    
    
    # Getting the type of 'integration_direction' (line 103)
    integration_direction_133316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'integration_direction')
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_133317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    unicode_133318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 37), 'unicode', u'both')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), list_133317, unicode_133318)
    # Adding element type (line 103)
    unicode_133319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 45), 'unicode', u'forward')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), list_133317, unicode_133319)
    # Adding element type (line 103)
    unicode_133320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 56), 'unicode', u'backward')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), list_133317, unicode_133320)
    
    # Applying the binary operator 'notin' (line 103)
    result_contains_133321 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), 'notin', integration_direction_133316, list_133317)
    
    # Testing the type of an if condition (line 103)
    if_condition_133322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), result_contains_133321)
    # Assigning a type to the variable 'if_condition_133322' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_133322', if_condition_133322)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 104):
    
    # Assigning a BinOp to a Name (line 104):
    unicode_133323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'unicode', u"Integration direction '%s' not recognised. Expected 'both', 'forward' or 'backward'.")
    # Getting the type of 'integration_direction' (line 106)
    integration_direction_133324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'integration_direction')
    # Applying the binary operator '%' (line 104)
    result_mod_133325 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 18), '%', unicode_133323, integration_direction_133324)
    
    # Assigning a type to the variable 'errstr' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'errstr', result_mod_133325)
    
    # Call to ValueError(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'errstr' (line 107)
    errstr_133327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'errstr', False)
    # Processing the call keyword arguments (line 107)
    kwargs_133328 = {}
    # Getting the type of 'ValueError' (line 107)
    ValueError_133326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 107)
    ValueError_call_result_133329 = invoke(stypy.reporting.localization.Localization(__file__, 107, 14), ValueError_133326, *[errstr_133327], **kwargs_133328)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 107, 8), ValueError_call_result_133329, 'raise parameter', BaseException)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'integration_direction' (line 109)
    integration_direction_133330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'integration_direction')
    unicode_133331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 32), 'unicode', u'both')
    # Applying the binary operator '==' (line 109)
    result_eq_133332 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 7), '==', integration_direction_133330, unicode_133331)
    
    # Testing the type of an if condition (line 109)
    if_condition_133333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), result_eq_133332)
    # Assigning a type to the variable 'if_condition_133333' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'if_condition_133333', if_condition_133333)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'maxlength' (line 110)
    maxlength_133334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'maxlength')
    float_133335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'float')
    # Applying the binary operator 'div=' (line 110)
    result_div_133336 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 8), 'div=', maxlength_133334, float_133335)
    # Assigning a type to the variable 'maxlength' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'maxlength', result_div_133336)
    
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to isinstance(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'color' (line 112)
    color_133338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'color', False)
    # Getting the type of 'np' (line 112)
    np_133339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 45), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 112)
    ndarray_133340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 45), np_133339, 'ndarray')
    # Processing the call keyword arguments (line 112)
    kwargs_133341 = {}
    # Getting the type of 'isinstance' (line 112)
    isinstance_133337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 112)
    isinstance_call_result_133342 = invoke(stypy.reporting.localization.Localization(__file__, 112, 27), isinstance_133337, *[color_133338, ndarray_133340], **kwargs_133341)
    
    # Assigning a type to the variable 'use_multicolor_lines' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'use_multicolor_lines', isinstance_call_result_133342)
    
    # Getting the type of 'use_multicolor_lines' (line 113)
    use_multicolor_lines_133343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'use_multicolor_lines')
    # Testing the type of an if condition (line 113)
    if_condition_133344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 4), use_multicolor_lines_133343)
    # Assigning a type to the variable 'if_condition_133344' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'if_condition_133344', if_condition_133344)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'color' (line 114)
    color_133345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'color')
    # Obtaining the member 'shape' of a type (line 114)
    shape_133346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 11), color_133345, 'shape')
    # Getting the type of 'grid' (line 114)
    grid_133347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'grid')
    # Obtaining the member 'shape' of a type (line 114)
    shape_133348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 26), grid_133347, 'shape')
    # Applying the binary operator '!=' (line 114)
    result_ne_133349 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 11), '!=', shape_133346, shape_133348)
    
    # Testing the type of an if condition (line 114)
    if_condition_133350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 8), result_ne_133349)
    # Assigning a type to the variable 'if_condition_133350' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'if_condition_133350', if_condition_133350)
    # SSA begins for if statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 115):
    
    # Assigning a Str to a Name (line 115):
    unicode_133351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 18), 'unicode', u"If 'color' is given, must have the shape of 'Grid(x,y)'")
    # Assigning a type to the variable 'msg' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'msg', unicode_133351)
    
    # Call to ValueError(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'msg' (line 116)
    msg_133353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'msg', False)
    # Processing the call keyword arguments (line 116)
    kwargs_133354 = {}
    # Getting the type of 'ValueError' (line 116)
    ValueError_133352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 116)
    ValueError_call_result_133355 = invoke(stypy.reporting.localization.Localization(__file__, 116, 18), ValueError_133352, *[msg_133353], **kwargs_133354)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 116, 12), ValueError_call_result_133355, 'raise parameter', BaseException)
    # SSA join for if statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 117):
    
    # Assigning a List to a Name (line 117):
    
    # Obtaining an instance of the builtin type 'list' (line 117)
    list_133356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 117)
    
    # Assigning a type to the variable 'line_colors' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'line_colors', list_133356)
    
    # Assigning a Call to a Name (line 118):
    
    # Assigning a Call to a Name (line 118):
    
    # Call to masked_invalid(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'color' (line 118)
    color_133360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 37), 'color', False)
    # Processing the call keyword arguments (line 118)
    kwargs_133361 = {}
    # Getting the type of 'np' (line 118)
    np_133357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'np', False)
    # Obtaining the member 'ma' of a type (line 118)
    ma_133358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), np_133357, 'ma')
    # Obtaining the member 'masked_invalid' of a type (line 118)
    masked_invalid_133359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), ma_133358, 'masked_invalid')
    # Calling masked_invalid(args, kwargs) (line 118)
    masked_invalid_call_result_133362 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), masked_invalid_133359, *[color_133360], **kwargs_133361)
    
    # Assigning a type to the variable 'color' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'color', masked_invalid_call_result_133362)
    # SSA branch for the else part of an if statement (line 113)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 120):
    
    # Assigning a Name to a Subscript (line 120):
    # Getting the type of 'color' (line 120)
    color_133363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'color')
    # Getting the type of 'line_kw' (line 120)
    line_kw_133364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'line_kw')
    unicode_133365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'unicode', u'color')
    # Storing an element on a container (line 120)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), line_kw_133364, (unicode_133365, color_133363))
    
    # Assigning a Name to a Subscript (line 121):
    
    # Assigning a Name to a Subscript (line 121):
    # Getting the type of 'color' (line 121)
    color_133366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'color')
    # Getting the type of 'arrow_kw' (line 121)
    arrow_kw_133367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'arrow_kw')
    unicode_133368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'unicode', u'color')
    # Storing an element on a container (line 121)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), arrow_kw_133367, (unicode_133368, color_133366))
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'linewidth' (line 123)
    linewidth_133370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'linewidth', False)
    # Getting the type of 'np' (line 123)
    np_133371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 123)
    ndarray_133372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 29), np_133371, 'ndarray')
    # Processing the call keyword arguments (line 123)
    kwargs_133373 = {}
    # Getting the type of 'isinstance' (line 123)
    isinstance_133369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 123)
    isinstance_call_result_133374 = invoke(stypy.reporting.localization.Localization(__file__, 123, 7), isinstance_133369, *[linewidth_133370, ndarray_133372], **kwargs_133373)
    
    # Testing the type of an if condition (line 123)
    if_condition_133375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 4), isinstance_call_result_133374)
    # Assigning a type to the variable 'if_condition_133375' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'if_condition_133375', if_condition_133375)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'linewidth' (line 124)
    linewidth_133376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'linewidth')
    # Obtaining the member 'shape' of a type (line 124)
    shape_133377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 11), linewidth_133376, 'shape')
    # Getting the type of 'grid' (line 124)
    grid_133378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'grid')
    # Obtaining the member 'shape' of a type (line 124)
    shape_133379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 30), grid_133378, 'shape')
    # Applying the binary operator '!=' (line 124)
    result_ne_133380 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), '!=', shape_133377, shape_133379)
    
    # Testing the type of an if condition (line 124)
    if_condition_133381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_ne_133380)
    # Assigning a type to the variable 'if_condition_133381' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_133381', if_condition_133381)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 125):
    
    # Assigning a Str to a Name (line 125):
    unicode_133382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'unicode', u"If 'linewidth' is given, must have the shape of 'Grid(x,y)'")
    # Assigning a type to the variable 'msg' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'msg', unicode_133382)
    
    # Call to ValueError(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'msg' (line 126)
    msg_133384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 29), 'msg', False)
    # Processing the call keyword arguments (line 126)
    kwargs_133385 = {}
    # Getting the type of 'ValueError' (line 126)
    ValueError_133383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 126)
    ValueError_call_result_133386 = invoke(stypy.reporting.localization.Localization(__file__, 126, 18), ValueError_133383, *[msg_133384], **kwargs_133385)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 126, 12), ValueError_call_result_133386, 'raise parameter', BaseException)
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Subscript (line 127):
    
    # Assigning a List to a Subscript (line 127):
    
    # Obtaining an instance of the builtin type 'list' (line 127)
    list_133387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 127)
    
    # Getting the type of 'line_kw' (line 127)
    line_kw_133388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'line_kw')
    unicode_133389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 16), 'unicode', u'linewidth')
    # Storing an element on a container (line 127)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 8), line_kw_133388, (unicode_133389, list_133387))
    # SSA branch for the else part of an if statement (line 123)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 129):
    
    # Assigning a Name to a Subscript (line 129):
    # Getting the type of 'linewidth' (line 129)
    linewidth_133390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 31), 'linewidth')
    # Getting the type of 'line_kw' (line 129)
    line_kw_133391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'line_kw')
    unicode_133392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'unicode', u'linewidth')
    # Storing an element on a container (line 129)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 8), line_kw_133391, (unicode_133392, linewidth_133390))
    
    # Assigning a Name to a Subscript (line 130):
    
    # Assigning a Name to a Subscript (line 130):
    # Getting the type of 'linewidth' (line 130)
    linewidth_133393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 32), 'linewidth')
    # Getting the type of 'arrow_kw' (line 130)
    arrow_kw_133394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'arrow_kw')
    unicode_133395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 17), 'unicode', u'linewidth')
    # Storing an element on a container (line 130)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 8), arrow_kw_133394, (unicode_133395, linewidth_133393))
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 132):
    
    # Assigning a Name to a Subscript (line 132):
    # Getting the type of 'zorder' (line 132)
    zorder_133396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'zorder')
    # Getting the type of 'line_kw' (line 132)
    line_kw_133397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'line_kw')
    unicode_133398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'unicode', u'zorder')
    # Storing an element on a container (line 132)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 4), line_kw_133397, (unicode_133398, zorder_133396))
    
    # Assigning a Name to a Subscript (line 133):
    
    # Assigning a Name to a Subscript (line 133):
    # Getting the type of 'zorder' (line 133)
    zorder_133399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'zorder')
    # Getting the type of 'arrow_kw' (line 133)
    arrow_kw_133400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'arrow_kw')
    unicode_133401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 13), 'unicode', u'zorder')
    # Storing an element on a container (line 133)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), arrow_kw_133400, (unicode_133401, zorder_133399))
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'u' (line 136)
    u_133402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'u')
    # Obtaining the member 'shape' of a type (line 136)
    shape_133403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), u_133402, 'shape')
    # Getting the type of 'grid' (line 136)
    grid_133404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'grid')
    # Obtaining the member 'shape' of a type (line 136)
    shape_133405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), grid_133404, 'shape')
    # Applying the binary operator '!=' (line 136)
    result_ne_133406 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 8), '!=', shape_133403, shape_133405)
    
    
    # Getting the type of 'v' (line 136)
    v_133407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 35), 'v')
    # Obtaining the member 'shape' of a type (line 136)
    shape_133408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 35), v_133407, 'shape')
    # Getting the type of 'grid' (line 136)
    grid_133409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 46), 'grid')
    # Obtaining the member 'shape' of a type (line 136)
    shape_133410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 46), grid_133409, 'shape')
    # Applying the binary operator '!=' (line 136)
    result_ne_133411 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 35), '!=', shape_133408, shape_133410)
    
    # Applying the binary operator 'or' (line 136)
    result_or_keyword_133412 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 7), 'or', result_ne_133406, result_ne_133411)
    
    # Testing the type of an if condition (line 136)
    if_condition_133413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 4), result_or_keyword_133412)
    # Assigning a type to the variable 'if_condition_133413' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'if_condition_133413', if_condition_133413)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 137):
    
    # Assigning a Str to a Name (line 137):
    unicode_133414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 14), 'unicode', u"'u' and 'v' must be of shape 'Grid(x,y)'")
    # Assigning a type to the variable 'msg' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'msg', unicode_133414)
    
    # Call to ValueError(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'msg' (line 138)
    msg_133416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'msg', False)
    # Processing the call keyword arguments (line 138)
    kwargs_133417 = {}
    # Getting the type of 'ValueError' (line 138)
    ValueError_133415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 138)
    ValueError_call_result_133418 = invoke(stypy.reporting.localization.Localization(__file__, 138, 14), ValueError_133415, *[msg_133416], **kwargs_133417)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 138, 8), ValueError_call_result_133418, 'raise parameter', BaseException)
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to masked_invalid(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'u' (line 140)
    u_133422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'u', False)
    # Processing the call keyword arguments (line 140)
    kwargs_133423 = {}
    # Getting the type of 'np' (line 140)
    np_133419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'np', False)
    # Obtaining the member 'ma' of a type (line 140)
    ma_133420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), np_133419, 'ma')
    # Obtaining the member 'masked_invalid' of a type (line 140)
    masked_invalid_133421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), ma_133420, 'masked_invalid')
    # Calling masked_invalid(args, kwargs) (line 140)
    masked_invalid_call_result_133424 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), masked_invalid_133421, *[u_133422], **kwargs_133423)
    
    # Assigning a type to the variable 'u' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'u', masked_invalid_call_result_133424)
    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 141):
    
    # Call to masked_invalid(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'v' (line 141)
    v_133428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'v', False)
    # Processing the call keyword arguments (line 141)
    kwargs_133429 = {}
    # Getting the type of 'np' (line 141)
    np_133425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'np', False)
    # Obtaining the member 'ma' of a type (line 141)
    ma_133426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), np_133425, 'ma')
    # Obtaining the member 'masked_invalid' of a type (line 141)
    masked_invalid_133427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), ma_133426, 'masked_invalid')
    # Calling masked_invalid(args, kwargs) (line 141)
    masked_invalid_call_result_133430 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), masked_invalid_133427, *[v_133428], **kwargs_133429)
    
    # Assigning a type to the variable 'v' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'v', masked_invalid_call_result_133430)
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to get_integrator(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'u' (line 143)
    u_133432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'u', False)
    # Getting the type of 'v' (line 143)
    v_133433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 34), 'v', False)
    # Getting the type of 'dmap' (line 143)
    dmap_133434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 37), 'dmap', False)
    # Getting the type of 'minlength' (line 143)
    minlength_133435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 43), 'minlength', False)
    # Getting the type of 'maxlength' (line 143)
    maxlength_133436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 54), 'maxlength', False)
    # Getting the type of 'integration_direction' (line 144)
    integration_direction_133437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 31), 'integration_direction', False)
    # Processing the call keyword arguments (line 143)
    kwargs_133438 = {}
    # Getting the type of 'get_integrator' (line 143)
    get_integrator_133431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'get_integrator', False)
    # Calling get_integrator(args, kwargs) (line 143)
    get_integrator_call_result_133439 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), get_integrator_133431, *[u_133432, v_133433, dmap_133434, minlength_133435, maxlength_133436, integration_direction_133437], **kwargs_133438)
    
    # Assigning a type to the variable 'integrate' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'integrate', get_integrator_call_result_133439)
    
    # Assigning a List to a Name (line 146):
    
    # Assigning a List to a Name (line 146):
    
    # Obtaining an instance of the builtin type 'list' (line 146)
    list_133440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 146)
    
    # Assigning a type to the variable 'trajectories' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'trajectories', list_133440)
    
    # Type idiom detected: calculating its left and rigth part (line 147)
    # Getting the type of 'start_points' (line 147)
    start_points_133441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'start_points')
    # Getting the type of 'None' (line 147)
    None_133442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'None')
    
    (may_be_133443, more_types_in_union_133444) = may_be_none(start_points_133441, None_133442)

    if may_be_133443:

        if more_types_in_union_133444:
            # Runtime conditional SSA (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to _gen_starting_points(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'mask' (line 148)
        mask_133446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 43), 'mask', False)
        # Obtaining the member 'shape' of a type (line 148)
        shape_133447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 43), mask_133446, 'shape')
        # Processing the call keyword arguments (line 148)
        kwargs_133448 = {}
        # Getting the type of '_gen_starting_points' (line 148)
        _gen_starting_points_133445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), '_gen_starting_points', False)
        # Calling _gen_starting_points(args, kwargs) (line 148)
        _gen_starting_points_call_result_133449 = invoke(stypy.reporting.localization.Localization(__file__, 148, 22), _gen_starting_points_133445, *[shape_133447], **kwargs_133448)
        
        # Testing the type of a for loop iterable (line 148)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 8), _gen_starting_points_call_result_133449)
        # Getting the type of the for loop variable (line 148)
        for_loop_var_133450 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 8), _gen_starting_points_call_result_133449)
        # Assigning a type to the variable 'xm' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'xm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 8), for_loop_var_133450))
        # Assigning a type to the variable 'ym' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'ym', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 8), for_loop_var_133450))
        # SSA begins for a for statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_133451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        # Getting the type of 'ym' (line 149)
        ym_133452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'ym')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 20), tuple_133451, ym_133452)
        # Adding element type (line 149)
        # Getting the type of 'xm' (line 149)
        xm_133453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'xm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 20), tuple_133451, xm_133453)
        
        # Getting the type of 'mask' (line 149)
        mask_133454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'mask')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___133455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), mask_133454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_133456 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), getitem___133455, tuple_133451)
        
        int_133457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'int')
        # Applying the binary operator '==' (line 149)
        result_eq_133458 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), '==', subscript_call_result_133456, int_133457)
        
        # Testing the type of an if condition (line 149)
        if_condition_133459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 12), result_eq_133458)
        # Assigning a type to the variable 'if_condition_133459' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'if_condition_133459', if_condition_133459)
        # SSA begins for if statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 150):
        
        # Assigning a Call to a Name:
        
        # Call to mask2grid(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'xm' (line 150)
        xm_133462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 40), 'xm', False)
        # Getting the type of 'ym' (line 150)
        ym_133463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 44), 'ym', False)
        # Processing the call keyword arguments (line 150)
        kwargs_133464 = {}
        # Getting the type of 'dmap' (line 150)
        dmap_133460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 25), 'dmap', False)
        # Obtaining the member 'mask2grid' of a type (line 150)
        mask2grid_133461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 25), dmap_133460, 'mask2grid')
        # Calling mask2grid(args, kwargs) (line 150)
        mask2grid_call_result_133465 = invoke(stypy.reporting.localization.Localization(__file__, 150, 25), mask2grid_133461, *[xm_133462, ym_133463], **kwargs_133464)
        
        # Assigning a type to the variable 'call_assignment_133167' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_133167', mask2grid_call_result_133465)
        
        # Assigning a Call to a Name (line 150):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_133468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'int')
        # Processing the call keyword arguments
        kwargs_133469 = {}
        # Getting the type of 'call_assignment_133167' (line 150)
        call_assignment_133167_133466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_133167', False)
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___133467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), call_assignment_133167_133466, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_133470 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133467, *[int_133468], **kwargs_133469)
        
        # Assigning a type to the variable 'call_assignment_133168' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_133168', getitem___call_result_133470)
        
        # Assigning a Name to a Name (line 150):
        # Getting the type of 'call_assignment_133168' (line 150)
        call_assignment_133168_133471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_133168')
        # Assigning a type to the variable 'xg' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'xg', call_assignment_133168_133471)
        
        # Assigning a Call to a Name (line 150):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_133474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'int')
        # Processing the call keyword arguments
        kwargs_133475 = {}
        # Getting the type of 'call_assignment_133167' (line 150)
        call_assignment_133167_133472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_133167', False)
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___133473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), call_assignment_133167_133472, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_133476 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133473, *[int_133474], **kwargs_133475)
        
        # Assigning a type to the variable 'call_assignment_133169' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_133169', getitem___call_result_133476)
        
        # Assigning a Name to a Name (line 150):
        # Getting the type of 'call_assignment_133169' (line 150)
        call_assignment_133169_133477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'call_assignment_133169')
        # Assigning a type to the variable 'yg' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'yg', call_assignment_133169_133477)
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to integrate(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'xg' (line 151)
        xg_133479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'xg', False)
        # Getting the type of 'yg' (line 151)
        yg_133480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'yg', False)
        # Processing the call keyword arguments (line 151)
        kwargs_133481 = {}
        # Getting the type of 'integrate' (line 151)
        integrate_133478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'integrate', False)
        # Calling integrate(args, kwargs) (line 151)
        integrate_call_result_133482 = invoke(stypy.reporting.localization.Localization(__file__, 151, 20), integrate_133478, *[xg_133479, yg_133480], **kwargs_133481)
        
        # Assigning a type to the variable 't' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 't', integrate_call_result_133482)
        
        # Type idiom detected: calculating its left and rigth part (line 152)
        # Getting the type of 't' (line 152)
        t_133483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 't')
        # Getting the type of 'None' (line 152)
        None_133484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'None')
        
        (may_be_133485, more_types_in_union_133486) = may_not_be_none(t_133483, None_133484)

        if may_be_133485:

            if more_types_in_union_133486:
                # Runtime conditional SSA (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 153)
            # Processing the call arguments (line 153)
            # Getting the type of 't' (line 153)
            t_133489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 40), 't', False)
            # Processing the call keyword arguments (line 153)
            kwargs_133490 = {}
            # Getting the type of 'trajectories' (line 153)
            trajectories_133487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'trajectories', False)
            # Obtaining the member 'append' of a type (line 153)
            append_133488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 20), trajectories_133487, 'append')
            # Calling append(args, kwargs) (line 153)
            append_call_result_133491 = invoke(stypy.reporting.localization.Localization(__file__, 153, 20), append_133488, *[t_133489], **kwargs_133490)
            

            if more_types_in_union_133486:
                # SSA join for if statement (line 152)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 149)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_133444:
            # Runtime conditional SSA for else branch (line 147)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_133443) or more_types_in_union_133444):
        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to copy(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_133500 = {}
        
        # Call to asanyarray(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'start_points' (line 155)
        start_points_133494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'start_points', False)
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'float' (line 155)
        float_133495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 48), 'float', False)
        keyword_133496 = float_133495
        kwargs_133497 = {'dtype': keyword_133496}
        # Getting the type of 'np' (line 155)
        np_133492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'np', False)
        # Obtaining the member 'asanyarray' of a type (line 155)
        asanyarray_133493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 14), np_133492, 'asanyarray')
        # Calling asanyarray(args, kwargs) (line 155)
        asanyarray_call_result_133498 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), asanyarray_133493, *[start_points_133494], **kwargs_133497)
        
        # Obtaining the member 'copy' of a type (line 155)
        copy_133499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 14), asanyarray_call_result_133498, 'copy')
        # Calling copy(args, kwargs) (line 155)
        copy_call_result_133501 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), copy_133499, *[], **kwargs_133500)
        
        # Assigning a type to the variable 'sp2' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'sp2', copy_call_result_133501)
        
        # Getting the type of 'sp2' (line 158)
        sp2_133502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'sp2')
        # Testing the type of a for loop iterable (line 158)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 8), sp2_133502)
        # Getting the type of the for loop variable (line 158)
        for_loop_var_133503 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 8), sp2_133502)
        # Assigning a type to the variable 'xs' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'xs', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), for_loop_var_133503))
        # Assigning a type to the variable 'ys' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'ys', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), for_loop_var_133503))
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'grid' (line 159)
        grid_133504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'grid')
        # Obtaining the member 'x_origin' of a type (line 159)
        x_origin_133505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 20), grid_133504, 'x_origin')
        # Getting the type of 'xs' (line 159)
        xs_133506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'xs')
        # Applying the binary operator '<=' (line 159)
        result_le_133507 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), '<=', x_origin_133505, xs_133506)
        # Getting the type of 'grid' (line 159)
        grid_133508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 43), 'grid')
        # Obtaining the member 'x_origin' of a type (line 159)
        x_origin_133509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 43), grid_133508, 'x_origin')
        # Getting the type of 'grid' (line 159)
        grid_133510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'grid')
        # Obtaining the member 'width' of a type (line 159)
        width_133511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 59), grid_133510, 'width')
        # Applying the binary operator '+' (line 159)
        result_add_133512 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 43), '+', x_origin_133509, width_133511)
        
        # Applying the binary operator '<=' (line 159)
        result_le_133513 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), '<=', xs_133506, result_add_133512)
        # Applying the binary operator '&' (line 159)
        result_and__133514 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), '&', result_le_133507, result_le_133513)
        
        
        # Getting the type of 'grid' (line 160)
        grid_133515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'grid')
        # Obtaining the member 'y_origin' of a type (line 160)
        y_origin_133516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 24), grid_133515, 'y_origin')
        # Getting the type of 'ys' (line 160)
        ys_133517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 41), 'ys')
        # Applying the binary operator '<=' (line 160)
        result_le_133518 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 24), '<=', y_origin_133516, ys_133517)
        # Getting the type of 'grid' (line 160)
        grid_133519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 47), 'grid')
        # Obtaining the member 'y_origin' of a type (line 160)
        y_origin_133520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 47), grid_133519, 'y_origin')
        # Getting the type of 'grid' (line 160)
        grid_133521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 63), 'grid')
        # Obtaining the member 'height' of a type (line 160)
        height_133522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 63), grid_133521, 'height')
        # Applying the binary operator '+' (line 160)
        result_add_133523 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 47), '+', y_origin_133520, height_133522)
        
        # Applying the binary operator '<=' (line 160)
        result_le_133524 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 24), '<=', ys_133517, result_add_133523)
        # Applying the binary operator '&' (line 160)
        result_and__133525 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 24), '&', result_le_133518, result_le_133524)
        
        # Applying the binary operator 'and' (line 159)
        result_and_keyword_133526 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), 'and', result_and__133514, result_and__133525)
        
        # Applying the 'not' unary operator (line 159)
        result_not__133527 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 15), 'not', result_and_keyword_133526)
        
        # Testing the type of an if condition (line 159)
        if_condition_133528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 12), result_not__133527)
        # Assigning a type to the variable 'if_condition_133528' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'if_condition_133528', if_condition_133528)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Call to format(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'xs' (line 162)
        xs_133532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 53), 'xs', False)
        # Getting the type of 'ys' (line 162)
        ys_133533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 57), 'ys', False)
        # Processing the call keyword arguments (line 161)
        kwargs_133534 = {}
        unicode_133530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 33), 'unicode', u'Starting point ({}, {}) outside of data boundaries')
        # Obtaining the member 'format' of a type (line 161)
        format_133531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 33), unicode_133530, 'format')
        # Calling format(args, kwargs) (line 161)
        format_call_result_133535 = invoke(stypy.reporting.localization.Localization(__file__, 161, 33), format_133531, *[xs_133532, ys_133533], **kwargs_133534)
        
        # Processing the call keyword arguments (line 161)
        kwargs_133536 = {}
        # Getting the type of 'ValueError' (line 161)
        ValueError_133529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 161)
        ValueError_call_result_133537 = invoke(stypy.reporting.localization.Localization(__file__, 161, 22), ValueError_133529, *[format_call_result_133535], **kwargs_133536)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 161, 16), ValueError_call_result_133537, 'raise parameter', BaseException)
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'sp2' (line 167)
        sp2_133538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'sp2')
        
        # Obtaining the type of the subscript
        slice_133539 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 8), None, None, None)
        int_133540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'int')
        # Getting the type of 'sp2' (line 167)
        sp2_133541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'sp2')
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___133542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), sp2_133541, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_133543 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___133542, (slice_133539, int_133540))
        
        # Getting the type of 'grid' (line 167)
        grid_133544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'grid')
        # Obtaining the member 'x_origin' of a type (line 167)
        x_origin_133545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 21), grid_133544, 'x_origin')
        # Applying the binary operator '-=' (line 167)
        result_isub_133546 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 8), '-=', subscript_call_result_133543, x_origin_133545)
        # Getting the type of 'sp2' (line 167)
        sp2_133547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'sp2')
        slice_133548 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 8), None, None, None)
        int_133549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'int')
        # Storing an element on a container (line 167)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 8), sp2_133547, ((slice_133548, int_133549), result_isub_133546))
        
        
        # Getting the type of 'sp2' (line 168)
        sp2_133550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'sp2')
        
        # Obtaining the type of the subscript
        slice_133551 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 168, 8), None, None, None)
        int_133552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 15), 'int')
        # Getting the type of 'sp2' (line 168)
        sp2_133553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'sp2')
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___133554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), sp2_133553, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_133555 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___133554, (slice_133551, int_133552))
        
        # Getting the type of 'grid' (line 168)
        grid_133556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'grid')
        # Obtaining the member 'y_origin' of a type (line 168)
        y_origin_133557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 21), grid_133556, 'y_origin')
        # Applying the binary operator '-=' (line 168)
        result_isub_133558 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 8), '-=', subscript_call_result_133555, y_origin_133557)
        # Getting the type of 'sp2' (line 168)
        sp2_133559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'sp2')
        slice_133560 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 168, 8), None, None, None)
        int_133561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 15), 'int')
        # Storing an element on a container (line 168)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), sp2_133559, ((slice_133560, int_133561), result_isub_133558))
        
        
        # Getting the type of 'sp2' (line 170)
        sp2_133562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'sp2')
        # Testing the type of a for loop iterable (line 170)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 170, 8), sp2_133562)
        # Getting the type of the for loop variable (line 170)
        for_loop_var_133563 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 170, 8), sp2_133562)
        # Assigning a type to the variable 'xs' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'xs', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), for_loop_var_133563))
        # Assigning a type to the variable 'ys' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ys', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), for_loop_var_133563))
        # SSA begins for a for statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 171):
        
        # Assigning a Call to a Name:
        
        # Call to data2grid(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'xs' (line 171)
        xs_133566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 36), 'xs', False)
        # Getting the type of 'ys' (line 171)
        ys_133567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 40), 'ys', False)
        # Processing the call keyword arguments (line 171)
        kwargs_133568 = {}
        # Getting the type of 'dmap' (line 171)
        dmap_133564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 21), 'dmap', False)
        # Obtaining the member 'data2grid' of a type (line 171)
        data2grid_133565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 21), dmap_133564, 'data2grid')
        # Calling data2grid(args, kwargs) (line 171)
        data2grid_call_result_133569 = invoke(stypy.reporting.localization.Localization(__file__, 171, 21), data2grid_133565, *[xs_133566, ys_133567], **kwargs_133568)
        
        # Assigning a type to the variable 'call_assignment_133170' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'call_assignment_133170', data2grid_call_result_133569)
        
        # Assigning a Call to a Name (line 171):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_133572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 12), 'int')
        # Processing the call keyword arguments
        kwargs_133573 = {}
        # Getting the type of 'call_assignment_133170' (line 171)
        call_assignment_133170_133570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'call_assignment_133170', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___133571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), call_assignment_133170_133570, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_133574 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133571, *[int_133572], **kwargs_133573)
        
        # Assigning a type to the variable 'call_assignment_133171' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'call_assignment_133171', getitem___call_result_133574)
        
        # Assigning a Name to a Name (line 171):
        # Getting the type of 'call_assignment_133171' (line 171)
        call_assignment_133171_133575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'call_assignment_133171')
        # Assigning a type to the variable 'xg' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'xg', call_assignment_133171_133575)
        
        # Assigning a Call to a Name (line 171):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_133578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 12), 'int')
        # Processing the call keyword arguments
        kwargs_133579 = {}
        # Getting the type of 'call_assignment_133170' (line 171)
        call_assignment_133170_133576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'call_assignment_133170', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___133577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), call_assignment_133170_133576, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_133580 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133577, *[int_133578], **kwargs_133579)
        
        # Assigning a type to the variable 'call_assignment_133172' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'call_assignment_133172', getitem___call_result_133580)
        
        # Assigning a Name to a Name (line 171):
        # Getting the type of 'call_assignment_133172' (line 171)
        call_assignment_133172_133581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'call_assignment_133172')
        # Assigning a type to the variable 'yg' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'yg', call_assignment_133172_133581)
        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to integrate(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'xg' (line 172)
        xg_133583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 26), 'xg', False)
        # Getting the type of 'yg' (line 172)
        yg_133584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 30), 'yg', False)
        # Processing the call keyword arguments (line 172)
        kwargs_133585 = {}
        # Getting the type of 'integrate' (line 172)
        integrate_133582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'integrate', False)
        # Calling integrate(args, kwargs) (line 172)
        integrate_call_result_133586 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), integrate_133582, *[xg_133583, yg_133584], **kwargs_133585)
        
        # Assigning a type to the variable 't' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 't', integrate_call_result_133586)
        
        # Type idiom detected: calculating its left and rigth part (line 173)
        # Getting the type of 't' (line 173)
        t_133587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 't')
        # Getting the type of 'None' (line 173)
        None_133588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'None')
        
        (may_be_133589, more_types_in_union_133590) = may_not_be_none(t_133587, None_133588)

        if may_be_133589:

            if more_types_in_union_133590:
                # Runtime conditional SSA (line 173)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 174)
            # Processing the call arguments (line 174)
            # Getting the type of 't' (line 174)
            t_133593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 36), 't', False)
            # Processing the call keyword arguments (line 174)
            kwargs_133594 = {}
            # Getting the type of 'trajectories' (line 174)
            trajectories_133591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'trajectories', False)
            # Obtaining the member 'append' of a type (line 174)
            append_133592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), trajectories_133591, 'append')
            # Calling append(args, kwargs) (line 174)
            append_call_result_133595 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), append_133592, *[t_133593], **kwargs_133594)
            

            if more_types_in_union_133590:
                # SSA join for if statement (line 173)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_133443 and more_types_in_union_133444):
            # SSA join for if statement (line 147)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'use_multicolor_lines' (line 176)
    use_multicolor_lines_133596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'use_multicolor_lines')
    # Testing the type of an if condition (line 176)
    if_condition_133597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 4), use_multicolor_lines_133596)
    # Assigning a type to the variable 'if_condition_133597' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'if_condition_133597', if_condition_133597)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 177)
    # Getting the type of 'norm' (line 177)
    norm_133598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'norm')
    # Getting the type of 'None' (line 177)
    None_133599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'None')
    
    (may_be_133600, more_types_in_union_133601) = may_be_none(norm_133598, None_133599)

    if may_be_133600:

        if more_types_in_union_133601:
            # Runtime conditional SSA (line 177)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to Normalize(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Call to min(...): (line 178)
        # Processing the call keyword arguments (line 178)
        kwargs_133606 = {}
        # Getting the type of 'color' (line 178)
        color_133604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 37), 'color', False)
        # Obtaining the member 'min' of a type (line 178)
        min_133605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 37), color_133604, 'min')
        # Calling min(args, kwargs) (line 178)
        min_call_result_133607 = invoke(stypy.reporting.localization.Localization(__file__, 178, 37), min_133605, *[], **kwargs_133606)
        
        
        # Call to max(...): (line 178)
        # Processing the call keyword arguments (line 178)
        kwargs_133610 = {}
        # Getting the type of 'color' (line 178)
        color_133608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 50), 'color', False)
        # Obtaining the member 'max' of a type (line 178)
        max_133609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 50), color_133608, 'max')
        # Calling max(args, kwargs) (line 178)
        max_call_result_133611 = invoke(stypy.reporting.localization.Localization(__file__, 178, 50), max_133609, *[], **kwargs_133610)
        
        # Processing the call keyword arguments (line 178)
        kwargs_133612 = {}
        # Getting the type of 'mcolors' (line 178)
        mcolors_133602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'mcolors', False)
        # Obtaining the member 'Normalize' of a type (line 178)
        Normalize_133603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 19), mcolors_133602, 'Normalize')
        # Calling Normalize(args, kwargs) (line 178)
        Normalize_call_result_133613 = invoke(stypy.reporting.localization.Localization(__file__, 178, 19), Normalize_133603, *[min_call_result_133607, max_call_result_133611], **kwargs_133612)
        
        # Assigning a type to the variable 'norm' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'norm', Normalize_call_result_133613)

        if more_types_in_union_133601:
            # SSA join for if statement (line 177)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 179)
    # Getting the type of 'cmap' (line 179)
    cmap_133614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'cmap')
    # Getting the type of 'None' (line 179)
    None_133615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'None')
    
    (may_be_133616, more_types_in_union_133617) = may_be_none(cmap_133614, None_133615)

    if may_be_133616:

        if more_types_in_union_133617:
            # Runtime conditional SSA (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to get_cmap(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Obtaining the type of the subscript
        unicode_133620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 51), 'unicode', u'image.cmap')
        # Getting the type of 'matplotlib' (line 180)
        matplotlib_133621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'matplotlib', False)
        # Obtaining the member 'rcParams' of a type (line 180)
        rcParams_133622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 31), matplotlib_133621, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___133623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 31), rcParams_133622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_133624 = invoke(stypy.reporting.localization.Localization(__file__, 180, 31), getitem___133623, unicode_133620)
        
        # Processing the call keyword arguments (line 180)
        kwargs_133625 = {}
        # Getting the type of 'cm' (line 180)
        cm_133618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'cm', False)
        # Obtaining the member 'get_cmap' of a type (line 180)
        get_cmap_133619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 19), cm_133618, 'get_cmap')
        # Calling get_cmap(args, kwargs) (line 180)
        get_cmap_call_result_133626 = invoke(stypy.reporting.localization.Localization(__file__, 180, 19), get_cmap_133619, *[subscript_call_result_133624], **kwargs_133625)
        
        # Assigning a type to the variable 'cmap' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'cmap', get_cmap_call_result_133626)

        if more_types_in_union_133617:
            # Runtime conditional SSA for else branch (line 179)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_133616) or more_types_in_union_133617):
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to get_cmap(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'cmap' (line 182)
        cmap_133629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'cmap', False)
        # Processing the call keyword arguments (line 182)
        kwargs_133630 = {}
        # Getting the type of 'cm' (line 182)
        cm_133627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'cm', False)
        # Obtaining the member 'get_cmap' of a type (line 182)
        get_cmap_133628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 19), cm_133627, 'get_cmap')
        # Calling get_cmap(args, kwargs) (line 182)
        get_cmap_call_result_133631 = invoke(stypy.reporting.localization.Localization(__file__, 182, 19), get_cmap_133628, *[cmap_133629], **kwargs_133630)
        
        # Assigning a type to the variable 'cmap' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'cmap', get_cmap_call_result_133631)

        if (may_be_133616 and more_types_in_union_133617):
            # SSA join for if statement (line 179)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 184):
    
    # Assigning a List to a Name (line 184):
    
    # Obtaining an instance of the builtin type 'list' (line 184)
    list_133632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 184)
    
    # Assigning a type to the variable 'streamlines' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'streamlines', list_133632)
    
    # Assigning a List to a Name (line 185):
    
    # Assigning a List to a Name (line 185):
    
    # Obtaining an instance of the builtin type 'list' (line 185)
    list_133633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 185)
    
    # Assigning a type to the variable 'arrows' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'arrows', list_133633)
    
    # Getting the type of 'trajectories' (line 186)
    trajectories_133634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'trajectories')
    # Testing the type of a for loop iterable (line 186)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 186, 4), trajectories_133634)
    # Getting the type of the for loop variable (line 186)
    for_loop_var_133635 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 186, 4), trajectories_133634)
    # Assigning a type to the variable 't' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 't', for_loop_var_133635)
    # SSA begins for a for statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 187):
    
    # Assigning a Call to a Name (line 187):
    
    # Call to array(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Obtaining the type of the subscript
    int_133638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 25), 'int')
    # Getting the type of 't' (line 187)
    t_133639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 't', False)
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___133640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), t_133639, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_133641 = invoke(stypy.reporting.localization.Localization(__file__, 187, 23), getitem___133640, int_133638)
    
    # Processing the call keyword arguments (line 187)
    kwargs_133642 = {}
    # Getting the type of 'np' (line 187)
    np_133636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 187)
    array_133637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 14), np_133636, 'array')
    # Calling array(args, kwargs) (line 187)
    array_call_result_133643 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), array_133637, *[subscript_call_result_133641], **kwargs_133642)
    
    # Assigning a type to the variable 'tgx' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tgx', array_call_result_133643)
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to array(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Obtaining the type of the subscript
    int_133646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 25), 'int')
    # Getting the type of 't' (line 188)
    t_133647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 't', False)
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___133648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 23), t_133647, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_133649 = invoke(stypy.reporting.localization.Localization(__file__, 188, 23), getitem___133648, int_133646)
    
    # Processing the call keyword arguments (line 188)
    kwargs_133650 = {}
    # Getting the type of 'np' (line 188)
    np_133644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 188)
    array_133645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 14), np_133644, 'array')
    # Calling array(args, kwargs) (line 188)
    array_call_result_133651 = invoke(stypy.reporting.localization.Localization(__file__, 188, 14), array_133645, *[subscript_call_result_133649], **kwargs_133650)
    
    # Assigning a type to the variable 'tgy' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'tgy', array_call_result_133651)
    
    # Assigning a Call to a Tuple (line 190):
    
    # Assigning a Call to a Name:
    
    # Call to grid2data(...): (line 190)
    
    # Call to array(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 't' (line 190)
    t_133656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 42), 't', False)
    # Processing the call keyword arguments (line 190)
    kwargs_133657 = {}
    # Getting the type of 'np' (line 190)
    np_133654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'np', False)
    # Obtaining the member 'array' of a type (line 190)
    array_133655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 33), np_133654, 'array')
    # Calling array(args, kwargs) (line 190)
    array_call_result_133658 = invoke(stypy.reporting.localization.Localization(__file__, 190, 33), array_133655, *[t_133656], **kwargs_133657)
    
    # Processing the call keyword arguments (line 190)
    kwargs_133659 = {}
    # Getting the type of 'dmap' (line 190)
    dmap_133652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'dmap', False)
    # Obtaining the member 'grid2data' of a type (line 190)
    grid2data_133653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 17), dmap_133652, 'grid2data')
    # Calling grid2data(args, kwargs) (line 190)
    grid2data_call_result_133660 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), grid2data_133653, *[array_call_result_133658], **kwargs_133659)
    
    # Assigning a type to the variable 'call_assignment_133173' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_133173', grid2data_call_result_133660)
    
    # Assigning a Call to a Name (line 190):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_133663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
    # Processing the call keyword arguments
    kwargs_133664 = {}
    # Getting the type of 'call_assignment_133173' (line 190)
    call_assignment_133173_133661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_133173', False)
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___133662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), call_assignment_133173_133661, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_133665 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133662, *[int_133663], **kwargs_133664)
    
    # Assigning a type to the variable 'call_assignment_133174' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_133174', getitem___call_result_133665)
    
    # Assigning a Name to a Name (line 190):
    # Getting the type of 'call_assignment_133174' (line 190)
    call_assignment_133174_133666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_133174')
    # Assigning a type to the variable 'tx' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'tx', call_assignment_133174_133666)
    
    # Assigning a Call to a Name (line 190):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_133669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
    # Processing the call keyword arguments
    kwargs_133670 = {}
    # Getting the type of 'call_assignment_133173' (line 190)
    call_assignment_133173_133667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_133173', False)
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___133668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), call_assignment_133173_133667, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_133671 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133668, *[int_133669], **kwargs_133670)
    
    # Assigning a type to the variable 'call_assignment_133175' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_133175', getitem___call_result_133671)
    
    # Assigning a Name to a Name (line 190):
    # Getting the type of 'call_assignment_133175' (line 190)
    call_assignment_133175_133672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_133175')
    # Assigning a type to the variable 'ty' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'ty', call_assignment_133175_133672)
    
    # Getting the type of 'tx' (line 191)
    tx_133673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tx')
    # Getting the type of 'grid' (line 191)
    grid_133674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), 'grid')
    # Obtaining the member 'x_origin' of a type (line 191)
    x_origin_133675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 14), grid_133674, 'x_origin')
    # Applying the binary operator '+=' (line 191)
    result_iadd_133676 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 8), '+=', tx_133673, x_origin_133675)
    # Assigning a type to the variable 'tx' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tx', result_iadd_133676)
    
    
    # Getting the type of 'ty' (line 192)
    ty_133677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'ty')
    # Getting the type of 'grid' (line 192)
    grid_133678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 14), 'grid')
    # Obtaining the member 'y_origin' of a type (line 192)
    y_origin_133679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 14), grid_133678, 'y_origin')
    # Applying the binary operator '+=' (line 192)
    result_iadd_133680 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 8), '+=', ty_133677, y_origin_133679)
    # Assigning a type to the variable 'ty' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'ty', result_iadd_133680)
    
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to reshape(...): (line 194)
    # Processing the call arguments (line 194)
    int_133689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 48), 'int')
    int_133690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 52), 'int')
    int_133691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 55), 'int')
    # Processing the call keyword arguments (line 194)
    kwargs_133692 = {}
    
    # Call to transpose(...): (line 194)
    # Processing the call arguments (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_133683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    # Getting the type of 'tx' (line 194)
    tx_133684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 31), 'tx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 30), list_133683, tx_133684)
    # Adding element type (line 194)
    # Getting the type of 'ty' (line 194)
    ty_133685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 35), 'ty', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 30), list_133683, ty_133685)
    
    # Processing the call keyword arguments (line 194)
    kwargs_133686 = {}
    # Getting the type of 'np' (line 194)
    np_133681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'np', False)
    # Obtaining the member 'transpose' of a type (line 194)
    transpose_133682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), np_133681, 'transpose')
    # Calling transpose(args, kwargs) (line 194)
    transpose_call_result_133687 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), transpose_133682, *[list_133683], **kwargs_133686)
    
    # Obtaining the member 'reshape' of a type (line 194)
    reshape_133688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), transpose_call_result_133687, 'reshape')
    # Calling reshape(args, kwargs) (line 194)
    reshape_call_result_133693 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), reshape_133688, *[int_133689, int_133690, int_133691], **kwargs_133692)
    
    # Assigning a type to the variable 'points' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'points', reshape_call_result_133693)
    
    # Call to extend(...): (line 195)
    # Processing the call arguments (line 195)
    
    # Call to hstack(...): (line 195)
    # Processing the call arguments (line 195)
    
    # Obtaining an instance of the builtin type 'list' (line 195)
    list_133698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 195)
    # Adding element type (line 195)
    
    # Obtaining the type of the subscript
    int_133699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 46), 'int')
    slice_133700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 195, 38), None, int_133699, None)
    # Getting the type of 'points' (line 195)
    points_133701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 38), 'points', False)
    # Obtaining the member '__getitem__' of a type (line 195)
    getitem___133702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 38), points_133701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 195)
    subscript_call_result_133703 = invoke(stypy.reporting.localization.Localization(__file__, 195, 38), getitem___133702, slice_133700)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 37), list_133698, subscript_call_result_133703)
    # Adding element type (line 195)
    
    # Obtaining the type of the subscript
    int_133704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 58), 'int')
    slice_133705 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 195, 51), int_133704, None, None)
    # Getting the type of 'points' (line 195)
    points_133706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 51), 'points', False)
    # Obtaining the member '__getitem__' of a type (line 195)
    getitem___133707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 51), points_133706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 195)
    subscript_call_result_133708 = invoke(stypy.reporting.localization.Localization(__file__, 195, 51), getitem___133707, slice_133705)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 37), list_133698, subscript_call_result_133708)
    
    # Processing the call keyword arguments (line 195)
    kwargs_133709 = {}
    # Getting the type of 'np' (line 195)
    np_133696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'np', False)
    # Obtaining the member 'hstack' of a type (line 195)
    hstack_133697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 27), np_133696, 'hstack')
    # Calling hstack(args, kwargs) (line 195)
    hstack_call_result_133710 = invoke(stypy.reporting.localization.Localization(__file__, 195, 27), hstack_133697, *[list_133698], **kwargs_133709)
    
    # Processing the call keyword arguments (line 195)
    kwargs_133711 = {}
    # Getting the type of 'streamlines' (line 195)
    streamlines_133694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'streamlines', False)
    # Obtaining the member 'extend' of a type (line 195)
    extend_133695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), streamlines_133694, 'extend')
    # Calling extend(args, kwargs) (line 195)
    extend_call_result_133712 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), extend_133695, *[hstack_call_result_133710], **kwargs_133711)
    
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to cumsum(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Call to sqrt(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Call to diff(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'tx' (line 198)
    tx_133719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 38), 'tx', False)
    # Processing the call keyword arguments (line 198)
    kwargs_133720 = {}
    # Getting the type of 'np' (line 198)
    np_133717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'np', False)
    # Obtaining the member 'diff' of a type (line 198)
    diff_133718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 30), np_133717, 'diff')
    # Calling diff(args, kwargs) (line 198)
    diff_call_result_133721 = invoke(stypy.reporting.localization.Localization(__file__, 198, 30), diff_133718, *[tx_133719], **kwargs_133720)
    
    int_133722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 45), 'int')
    # Applying the binary operator '**' (line 198)
    result_pow_133723 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 30), '**', diff_call_result_133721, int_133722)
    
    
    # Call to diff(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'ty' (line 198)
    ty_133726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 57), 'ty', False)
    # Processing the call keyword arguments (line 198)
    kwargs_133727 = {}
    # Getting the type of 'np' (line 198)
    np_133724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 49), 'np', False)
    # Obtaining the member 'diff' of a type (line 198)
    diff_133725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 49), np_133724, 'diff')
    # Calling diff(args, kwargs) (line 198)
    diff_call_result_133728 = invoke(stypy.reporting.localization.Localization(__file__, 198, 49), diff_133725, *[ty_133726], **kwargs_133727)
    
    int_133729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 64), 'int')
    # Applying the binary operator '**' (line 198)
    result_pow_133730 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 49), '**', diff_call_result_133728, int_133729)
    
    # Applying the binary operator '+' (line 198)
    result_add_133731 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 30), '+', result_pow_133723, result_pow_133730)
    
    # Processing the call keyword arguments (line 198)
    kwargs_133732 = {}
    # Getting the type of 'np' (line 198)
    np_133715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 22), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 198)
    sqrt_133716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 22), np_133715, 'sqrt')
    # Calling sqrt(args, kwargs) (line 198)
    sqrt_call_result_133733 = invoke(stypy.reporting.localization.Localization(__file__, 198, 22), sqrt_133716, *[result_add_133731], **kwargs_133732)
    
    # Processing the call keyword arguments (line 198)
    kwargs_133734 = {}
    # Getting the type of 'np' (line 198)
    np_133713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 198)
    cumsum_133714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), np_133713, 'cumsum')
    # Calling cumsum(args, kwargs) (line 198)
    cumsum_call_result_133735 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), cumsum_133714, *[sqrt_call_result_133733], **kwargs_133734)
    
    # Assigning a type to the variable 's' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 's', cumsum_call_result_133735)
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to searchsorted(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 's' (line 199)
    s_133738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 's', False)
    
    # Obtaining the type of the subscript
    int_133739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 33), 'int')
    # Getting the type of 's' (line 199)
    s_133740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 's', False)
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___133741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 31), s_133740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_133742 = invoke(stypy.reporting.localization.Localization(__file__, 199, 31), getitem___133741, int_133739)
    
    float_133743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'float')
    # Applying the binary operator 'div' (line 199)
    result_div_133744 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 31), 'div', subscript_call_result_133742, float_133743)
    
    # Processing the call keyword arguments (line 199)
    kwargs_133745 = {}
    # Getting the type of 'np' (line 199)
    np_133736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'np', False)
    # Obtaining the member 'searchsorted' of a type (line 199)
    searchsorted_133737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), np_133736, 'searchsorted')
    # Calling searchsorted(args, kwargs) (line 199)
    searchsorted_call_result_133746 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), searchsorted_133737, *[s_133738, result_div_133744], **kwargs_133745)
    
    # Assigning a type to the variable 'n' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'n', searchsorted_call_result_133746)
    
    # Assigning a Tuple to a Name (line 200):
    
    # Assigning a Tuple to a Name (line 200):
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_133747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 200)
    n_133748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'n')
    # Getting the type of 'tx' (line 200)
    tx_133749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'tx')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___133750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), tx_133749, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_133751 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), getitem___133750, n_133748)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 22), tuple_133747, subscript_call_result_133751)
    # Adding element type (line 200)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 200)
    n_133752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'n')
    # Getting the type of 'ty' (line 200)
    ty_133753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 29), 'ty')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___133754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 29), ty_133753, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_133755 = invoke(stypy.reporting.localization.Localization(__file__, 200, 29), getitem___133754, n_133752)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 22), tuple_133747, subscript_call_result_133755)
    
    # Assigning a type to the variable 'arrow_tail' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'arrow_tail', tuple_133747)
    
    # Assigning a Tuple to a Name (line 201):
    
    # Assigning a Tuple to a Name (line 201):
    
    # Obtaining an instance of the builtin type 'tuple' (line 201)
    tuple_133756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 201)
    # Adding element type (line 201)
    
    # Call to mean(...): (line 201)
    # Processing the call arguments (line 201)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 201)
    n_133759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'n', False)
    # Getting the type of 'n' (line 201)
    n_133760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 35), 'n', False)
    int_133761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 39), 'int')
    # Applying the binary operator '+' (line 201)
    result_add_133762 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 35), '+', n_133760, int_133761)
    
    slice_133763 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 201, 30), n_133759, result_add_133762, None)
    # Getting the type of 'tx' (line 201)
    tx_133764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 30), 'tx', False)
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___133765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 30), tx_133764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_133766 = invoke(stypy.reporting.localization.Localization(__file__, 201, 30), getitem___133765, slice_133763)
    
    # Processing the call keyword arguments (line 201)
    kwargs_133767 = {}
    # Getting the type of 'np' (line 201)
    np_133757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 22), 'np', False)
    # Obtaining the member 'mean' of a type (line 201)
    mean_133758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 22), np_133757, 'mean')
    # Calling mean(args, kwargs) (line 201)
    mean_call_result_133768 = invoke(stypy.reporting.localization.Localization(__file__, 201, 22), mean_133758, *[subscript_call_result_133766], **kwargs_133767)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 22), tuple_133756, mean_call_result_133768)
    # Adding element type (line 201)
    
    # Call to mean(...): (line 201)
    # Processing the call arguments (line 201)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 201)
    n_133771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 55), 'n', False)
    # Getting the type of 'n' (line 201)
    n_133772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'n', False)
    int_133773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 61), 'int')
    # Applying the binary operator '+' (line 201)
    result_add_133774 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 57), '+', n_133772, int_133773)
    
    slice_133775 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 201, 52), n_133771, result_add_133774, None)
    # Getting the type of 'ty' (line 201)
    ty_133776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 52), 'ty', False)
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___133777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 52), ty_133776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_133778 = invoke(stypy.reporting.localization.Localization(__file__, 201, 52), getitem___133777, slice_133775)
    
    # Processing the call keyword arguments (line 201)
    kwargs_133779 = {}
    # Getting the type of 'np' (line 201)
    np_133769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 44), 'np', False)
    # Obtaining the member 'mean' of a type (line 201)
    mean_133770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 44), np_133769, 'mean')
    # Calling mean(args, kwargs) (line 201)
    mean_call_result_133780 = invoke(stypy.reporting.localization.Localization(__file__, 201, 44), mean_133770, *[subscript_call_result_133778], **kwargs_133779)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 22), tuple_133756, mean_call_result_133780)
    
    # Assigning a type to the variable 'arrow_head' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'arrow_head', tuple_133756)
    
    
    # Call to isinstance(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'linewidth' (line 203)
    linewidth_133782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'linewidth', False)
    # Getting the type of 'np' (line 203)
    np_133783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 33), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 203)
    ndarray_133784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 33), np_133783, 'ndarray')
    # Processing the call keyword arguments (line 203)
    kwargs_133785 = {}
    # Getting the type of 'isinstance' (line 203)
    isinstance_133781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 203)
    isinstance_call_result_133786 = invoke(stypy.reporting.localization.Localization(__file__, 203, 11), isinstance_133781, *[linewidth_133782, ndarray_133784], **kwargs_133785)
    
    # Testing the type of an if condition (line 203)
    if_condition_133787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), isinstance_call_result_133786)
    # Assigning a type to the variable 'if_condition_133787' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_133787', if_condition_133787)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 204):
    
    # Assigning a Subscript to a Name (line 204):
    
    # Obtaining the type of the subscript
    int_133788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 59), 'int')
    slice_133789 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 204, 26), None, int_133788, None)
    
    # Call to interpgrid(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'linewidth' (line 204)
    linewidth_133791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 37), 'linewidth', False)
    # Getting the type of 'tgx' (line 204)
    tgx_133792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 48), 'tgx', False)
    # Getting the type of 'tgy' (line 204)
    tgy_133793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 53), 'tgy', False)
    # Processing the call keyword arguments (line 204)
    kwargs_133794 = {}
    # Getting the type of 'interpgrid' (line 204)
    interpgrid_133790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'interpgrid', False)
    # Calling interpgrid(args, kwargs) (line 204)
    interpgrid_call_result_133795 = invoke(stypy.reporting.localization.Localization(__file__, 204, 26), interpgrid_133790, *[linewidth_133791, tgx_133792, tgy_133793], **kwargs_133794)
    
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___133796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 26), interpgrid_call_result_133795, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_133797 = invoke(stypy.reporting.localization.Localization(__file__, 204, 26), getitem___133796, slice_133789)
    
    # Assigning a type to the variable 'line_widths' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'line_widths', subscript_call_result_133797)
    
    # Call to extend(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'line_widths' (line 205)
    line_widths_133803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 40), 'line_widths', False)
    # Processing the call keyword arguments (line 205)
    kwargs_133804 = {}
    
    # Obtaining the type of the subscript
    unicode_133798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 20), 'unicode', u'linewidth')
    # Getting the type of 'line_kw' (line 205)
    line_kw_133799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'line_kw', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___133800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), line_kw_133799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_133801 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), getitem___133800, unicode_133798)
    
    # Obtaining the member 'extend' of a type (line 205)
    extend_133802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), subscript_call_result_133801, 'extend')
    # Calling extend(args, kwargs) (line 205)
    extend_call_result_133805 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), extend_133802, *[line_widths_133803], **kwargs_133804)
    
    
    # Assigning a Subscript to a Subscript (line 206):
    
    # Assigning a Subscript to a Subscript (line 206):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 206)
    n_133806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 48), 'n')
    # Getting the type of 'line_widths' (line 206)
    line_widths_133807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 36), 'line_widths')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___133808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 36), line_widths_133807, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_133809 = invoke(stypy.reporting.localization.Localization(__file__, 206, 36), getitem___133808, n_133806)
    
    # Getting the type of 'arrow_kw' (line 206)
    arrow_kw_133810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'arrow_kw')
    unicode_133811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 21), 'unicode', u'linewidth')
    # Storing an element on a container (line 206)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 12), arrow_kw_133810, (unicode_133811, subscript_call_result_133809))
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'use_multicolor_lines' (line 208)
    use_multicolor_lines_133812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'use_multicolor_lines')
    # Testing the type of an if condition (line 208)
    if_condition_133813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), use_multicolor_lines_133812)
    # Assigning a type to the variable 'if_condition_133813' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_133813', if_condition_133813)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 209):
    
    # Assigning a Subscript to a Name (line 209):
    
    # Obtaining the type of the subscript
    int_133814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 56), 'int')
    slice_133815 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 209, 27), None, int_133814, None)
    
    # Call to interpgrid(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'color' (line 209)
    color_133817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 38), 'color', False)
    # Getting the type of 'tgx' (line 209)
    tgx_133818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 45), 'tgx', False)
    # Getting the type of 'tgy' (line 209)
    tgy_133819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 50), 'tgy', False)
    # Processing the call keyword arguments (line 209)
    kwargs_133820 = {}
    # Getting the type of 'interpgrid' (line 209)
    interpgrid_133816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'interpgrid', False)
    # Calling interpgrid(args, kwargs) (line 209)
    interpgrid_call_result_133821 = invoke(stypy.reporting.localization.Localization(__file__, 209, 27), interpgrid_133816, *[color_133817, tgx_133818, tgy_133819], **kwargs_133820)
    
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___133822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 27), interpgrid_call_result_133821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_133823 = invoke(stypy.reporting.localization.Localization(__file__, 209, 27), getitem___133822, slice_133815)
    
    # Assigning a type to the variable 'color_values' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'color_values', subscript_call_result_133823)
    
    # Call to append(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'color_values' (line 210)
    color_values_133826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 31), 'color_values', False)
    # Processing the call keyword arguments (line 210)
    kwargs_133827 = {}
    # Getting the type of 'line_colors' (line 210)
    line_colors_133824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'line_colors', False)
    # Obtaining the member 'append' of a type (line 210)
    append_133825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), line_colors_133824, 'append')
    # Calling append(args, kwargs) (line 210)
    append_call_result_133828 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), append_133825, *[color_values_133826], **kwargs_133827)
    
    
    # Assigning a Call to a Subscript (line 211):
    
    # Assigning a Call to a Subscript (line 211):
    
    # Call to cmap(...): (line 211)
    # Processing the call arguments (line 211)
    
    # Call to norm(...): (line 211)
    # Processing the call arguments (line 211)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 211)
    n_133831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 55), 'n', False)
    # Getting the type of 'color_values' (line 211)
    color_values_133832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 42), 'color_values', False)
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___133833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 42), color_values_133832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_133834 = invoke(stypy.reporting.localization.Localization(__file__, 211, 42), getitem___133833, n_133831)
    
    # Processing the call keyword arguments (line 211)
    kwargs_133835 = {}
    # Getting the type of 'norm' (line 211)
    norm_133830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 37), 'norm', False)
    # Calling norm(args, kwargs) (line 211)
    norm_call_result_133836 = invoke(stypy.reporting.localization.Localization(__file__, 211, 37), norm_133830, *[subscript_call_result_133834], **kwargs_133835)
    
    # Processing the call keyword arguments (line 211)
    kwargs_133837 = {}
    # Getting the type of 'cmap' (line 211)
    cmap_133829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 32), 'cmap', False)
    # Calling cmap(args, kwargs) (line 211)
    cmap_call_result_133838 = invoke(stypy.reporting.localization.Localization(__file__, 211, 32), cmap_133829, *[norm_call_result_133836], **kwargs_133837)
    
    # Getting the type of 'arrow_kw' (line 211)
    arrow_kw_133839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'arrow_kw')
    unicode_133840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 21), 'unicode', u'color')
    # Storing an element on a container (line 211)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 12), arrow_kw_133839, (unicode_133840, cmap_call_result_133838))
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to FancyArrowPatch(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'arrow_tail' (line 214)
    arrow_tail_133843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'arrow_tail', False)
    # Getting the type of 'arrow_head' (line 214)
    arrow_head_133844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'arrow_head', False)
    # Processing the call keyword arguments (line 213)
    # Getting the type of 'transform' (line 214)
    transform_133845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 46), 'transform', False)
    keyword_133846 = transform_133845
    # Getting the type of 'arrow_kw' (line 214)
    arrow_kw_133847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 59), 'arrow_kw', False)
    kwargs_133848 = {'arrow_kw_133847': arrow_kw_133847, 'transform': keyword_133846}
    # Getting the type of 'patches' (line 213)
    patches_133841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'patches', False)
    # Obtaining the member 'FancyArrowPatch' of a type (line 213)
    FancyArrowPatch_133842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), patches_133841, 'FancyArrowPatch')
    # Calling FancyArrowPatch(args, kwargs) (line 213)
    FancyArrowPatch_call_result_133849 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), FancyArrowPatch_133842, *[arrow_tail_133843, arrow_head_133844], **kwargs_133848)
    
    # Assigning a type to the variable 'p' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'p', FancyArrowPatch_call_result_133849)
    
    # Call to add_patch(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'p' (line 215)
    p_133852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'p', False)
    # Processing the call keyword arguments (line 215)
    kwargs_133853 = {}
    # Getting the type of 'axes' (line 215)
    axes_133850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'axes', False)
    # Obtaining the member 'add_patch' of a type (line 215)
    add_patch_133851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), axes_133850, 'add_patch')
    # Calling add_patch(args, kwargs) (line 215)
    add_patch_call_result_133854 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), add_patch_133851, *[p_133852], **kwargs_133853)
    
    
    # Call to append(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'p' (line 216)
    p_133857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 22), 'p', False)
    # Processing the call keyword arguments (line 216)
    kwargs_133858 = {}
    # Getting the type of 'arrows' (line 216)
    arrows_133855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'arrows', False)
    # Obtaining the member 'append' of a type (line 216)
    append_133856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), arrows_133855, 'append')
    # Calling append(args, kwargs) (line 216)
    append_call_result_133859 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), append_133856, *[p_133857], **kwargs_133858)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to LineCollection(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'streamlines' (line 219)
    streamlines_133862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'streamlines', False)
    # Processing the call keyword arguments (line 218)
    # Getting the type of 'transform' (line 219)
    transform_133863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'transform', False)
    keyword_133864 = transform_133863
    # Getting the type of 'line_kw' (line 219)
    line_kw_133865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 44), 'line_kw', False)
    kwargs_133866 = {'line_kw_133865': line_kw_133865, 'transform': keyword_133864}
    # Getting the type of 'mcollections' (line 218)
    mcollections_133860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 9), 'mcollections', False)
    # Obtaining the member 'LineCollection' of a type (line 218)
    LineCollection_133861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 9), mcollections_133860, 'LineCollection')
    # Calling LineCollection(args, kwargs) (line 218)
    LineCollection_call_result_133867 = invoke(stypy.reporting.localization.Localization(__file__, 218, 9), LineCollection_133861, *[streamlines_133862], **kwargs_133866)
    
    # Assigning a type to the variable 'lc' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'lc', LineCollection_call_result_133867)
    
    # Assigning a List to a Subscript (line 220):
    
    # Assigning a List to a Subscript (line 220):
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_133868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    # Adding element type (line 220)
    # Getting the type of 'grid' (line 220)
    grid_133869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'grid')
    # Obtaining the member 'x_origin' of a type (line 220)
    x_origin_133870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 28), grid_133869, 'x_origin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 27), list_133868, x_origin_133870)
    # Adding element type (line 220)
    # Getting the type of 'grid' (line 220)
    grid_133871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 43), 'grid')
    # Obtaining the member 'x_origin' of a type (line 220)
    x_origin_133872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 43), grid_133871, 'x_origin')
    # Getting the type of 'grid' (line 220)
    grid_133873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 59), 'grid')
    # Obtaining the member 'width' of a type (line 220)
    width_133874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 59), grid_133873, 'width')
    # Applying the binary operator '+' (line 220)
    result_add_133875 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 43), '+', x_origin_133872, width_133874)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 27), list_133868, result_add_133875)
    
    # Getting the type of 'lc' (line 220)
    lc_133876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'lc')
    # Obtaining the member 'sticky_edges' of a type (line 220)
    sticky_edges_133877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 4), lc_133876, 'sticky_edges')
    # Obtaining the member 'x' of a type (line 220)
    x_133878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 4), sticky_edges_133877, 'x')
    slice_133879 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 220, 4), None, None, None)
    # Storing an element on a container (line 220)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 4), x_133878, (slice_133879, list_133868))
    
    # Assigning a List to a Subscript (line 221):
    
    # Assigning a List to a Subscript (line 221):
    
    # Obtaining an instance of the builtin type 'list' (line 221)
    list_133880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'grid' (line 221)
    grid_133881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 28), 'grid')
    # Obtaining the member 'y_origin' of a type (line 221)
    y_origin_133882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 28), grid_133881, 'y_origin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 27), list_133880, y_origin_133882)
    # Adding element type (line 221)
    # Getting the type of 'grid' (line 221)
    grid_133883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), 'grid')
    # Obtaining the member 'y_origin' of a type (line 221)
    y_origin_133884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 43), grid_133883, 'y_origin')
    # Getting the type of 'grid' (line 221)
    grid_133885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 59), 'grid')
    # Obtaining the member 'height' of a type (line 221)
    height_133886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 59), grid_133885, 'height')
    # Applying the binary operator '+' (line 221)
    result_add_133887 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 43), '+', y_origin_133884, height_133886)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 27), list_133880, result_add_133887)
    
    # Getting the type of 'lc' (line 221)
    lc_133888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'lc')
    # Obtaining the member 'sticky_edges' of a type (line 221)
    sticky_edges_133889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 4), lc_133888, 'sticky_edges')
    # Obtaining the member 'y' of a type (line 221)
    y_133890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 4), sticky_edges_133889, 'y')
    slice_133891 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 221, 4), None, None, None)
    # Storing an element on a container (line 221)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 4), y_133890, (slice_133891, list_133880))
    
    # Getting the type of 'use_multicolor_lines' (line 222)
    use_multicolor_lines_133892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 7), 'use_multicolor_lines')
    # Testing the type of an if condition (line 222)
    if_condition_133893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 4), use_multicolor_lines_133892)
    # Assigning a type to the variable 'if_condition_133893' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'if_condition_133893', if_condition_133893)
    # SSA begins for if statement (line 222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_array(...): (line 223)
    # Processing the call arguments (line 223)
    
    # Call to hstack(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'line_colors' (line 223)
    line_colors_133899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 34), 'line_colors', False)
    # Processing the call keyword arguments (line 223)
    kwargs_133900 = {}
    # Getting the type of 'np' (line 223)
    np_133896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'np', False)
    # Obtaining the member 'ma' of a type (line 223)
    ma_133897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 21), np_133896, 'ma')
    # Obtaining the member 'hstack' of a type (line 223)
    hstack_133898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 21), ma_133897, 'hstack')
    # Calling hstack(args, kwargs) (line 223)
    hstack_call_result_133901 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), hstack_133898, *[line_colors_133899], **kwargs_133900)
    
    # Processing the call keyword arguments (line 223)
    kwargs_133902 = {}
    # Getting the type of 'lc' (line 223)
    lc_133894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'lc', False)
    # Obtaining the member 'set_array' of a type (line 223)
    set_array_133895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), lc_133894, 'set_array')
    # Calling set_array(args, kwargs) (line 223)
    set_array_call_result_133903 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), set_array_133895, *[hstack_call_result_133901], **kwargs_133902)
    
    
    # Call to set_cmap(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'cmap' (line 224)
    cmap_133906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'cmap', False)
    # Processing the call keyword arguments (line 224)
    kwargs_133907 = {}
    # Getting the type of 'lc' (line 224)
    lc_133904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'lc', False)
    # Obtaining the member 'set_cmap' of a type (line 224)
    set_cmap_133905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), lc_133904, 'set_cmap')
    # Calling set_cmap(args, kwargs) (line 224)
    set_cmap_call_result_133908 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), set_cmap_133905, *[cmap_133906], **kwargs_133907)
    
    
    # Call to set_norm(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'norm' (line 225)
    norm_133911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'norm', False)
    # Processing the call keyword arguments (line 225)
    kwargs_133912 = {}
    # Getting the type of 'lc' (line 225)
    lc_133909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'lc', False)
    # Obtaining the member 'set_norm' of a type (line 225)
    set_norm_133910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), lc_133909, 'set_norm')
    # Calling set_norm(args, kwargs) (line 225)
    set_norm_call_result_133913 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), set_norm_133910, *[norm_133911], **kwargs_133912)
    
    # SSA join for if statement (line 222)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add_collection(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'lc' (line 226)
    lc_133916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'lc', False)
    # Processing the call keyword arguments (line 226)
    kwargs_133917 = {}
    # Getting the type of 'axes' (line 226)
    axes_133914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'axes', False)
    # Obtaining the member 'add_collection' of a type (line 226)
    add_collection_133915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), axes_133914, 'add_collection')
    # Calling add_collection(args, kwargs) (line 226)
    add_collection_call_result_133918 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), add_collection_133915, *[lc_133916], **kwargs_133917)
    
    
    # Call to autoscale_view(...): (line 227)
    # Processing the call keyword arguments (line 227)
    kwargs_133921 = {}
    # Getting the type of 'axes' (line 227)
    axes_133919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'axes', False)
    # Obtaining the member 'autoscale_view' of a type (line 227)
    autoscale_view_133920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), axes_133919, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 227)
    autoscale_view_call_result_133922 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), autoscale_view_133920, *[], **kwargs_133921)
    
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Call to PatchCollection(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'arrows' (line 229)
    arrows_133926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 48), 'arrows', False)
    # Processing the call keyword arguments (line 229)
    kwargs_133927 = {}
    # Getting the type of 'matplotlib' (line 229)
    matplotlib_133923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 9), 'matplotlib', False)
    # Obtaining the member 'collections' of a type (line 229)
    collections_133924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 9), matplotlib_133923, 'collections')
    # Obtaining the member 'PatchCollection' of a type (line 229)
    PatchCollection_133925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 9), collections_133924, 'PatchCollection')
    # Calling PatchCollection(args, kwargs) (line 229)
    PatchCollection_call_result_133928 = invoke(stypy.reporting.localization.Localization(__file__, 229, 9), PatchCollection_133925, *[arrows_133926], **kwargs_133927)
    
    # Assigning a type to the variable 'ac' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'ac', PatchCollection_call_result_133928)
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to StreamplotSet(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'lc' (line 230)
    lc_133930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 37), 'lc', False)
    # Getting the type of 'ac' (line 230)
    ac_133931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 41), 'ac', False)
    # Processing the call keyword arguments (line 230)
    kwargs_133932 = {}
    # Getting the type of 'StreamplotSet' (line 230)
    StreamplotSet_133929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'StreamplotSet', False)
    # Calling StreamplotSet(args, kwargs) (line 230)
    StreamplotSet_call_result_133933 = invoke(stypy.reporting.localization.Localization(__file__, 230, 23), StreamplotSet_133929, *[lc_133930, ac_133931], **kwargs_133932)
    
    # Assigning a type to the variable 'stream_container' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stream_container', StreamplotSet_call_result_133933)
    # Getting the type of 'stream_container' (line 231)
    stream_container_133934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'stream_container')
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type', stream_container_133934)
    
    # ################# End of 'streamplot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'streamplot' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_133935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133935)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'streamplot'
    return stypy_return_type_133935

# Assigning a type to the variable 'streamplot' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'streamplot', streamplot)
# Declaration of the 'StreamplotSet' class

class StreamplotSet(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StreamplotSet.__init__', ['lines', 'arrows'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['lines', 'arrows'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 237):
        
        # Assigning a Name to a Attribute (line 237):
        # Getting the type of 'lines' (line 237)
        lines_133936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'lines')
        # Getting the type of 'self' (line 237)
        self_133937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self')
        # Setting the type of the member 'lines' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_133937, 'lines', lines_133936)
        
        # Assigning a Name to a Attribute (line 238):
        
        # Assigning a Name to a Attribute (line 238):
        # Getting the type of 'arrows' (line 238)
        arrows_133938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'arrows')
        # Getting the type of 'self' (line 238)
        self_133939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self')
        # Setting the type of the member 'arrows' of a type (line 238)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_133939, 'arrows', arrows_133938)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'StreamplotSet' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'StreamplotSet', StreamplotSet)
# Declaration of the 'DomainMap' class

class DomainMap(object, ):
    unicode_133940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, (-1)), 'unicode', u'Map representing different coordinate systems.\n\n    Coordinate definitions:\n\n    * axes-coordinates goes from 0 to 1 in the domain.\n    * data-coordinates are specified by the input x-y coordinates.\n    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,\n      where N and M match the shape of the input data.\n    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,\n      where N and M are user-specified to control the density of streamlines.\n\n    This class also has methods for adding trajectories to the StreamMask.\n    Before adding a trajectory, run `start_trajectory` to keep track of regions\n    crossed by a given trajectory. Later, if you decide the trajectory is bad\n    (e.g., if the trajectory is very short) just call `undo_trajectory`.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 262, 4, False)
        # Assigning a type to the variable 'self' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.__init__', ['grid', 'mask'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['grid', 'mask'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 263):
        
        # Assigning a Name to a Attribute (line 263):
        # Getting the type of 'grid' (line 263)
        grid_133941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'grid')
        # Getting the type of 'self' (line 263)
        self_133942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self')
        # Setting the type of the member 'grid' of a type (line 263)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), self_133942, 'grid', grid_133941)
        
        # Assigning a Name to a Attribute (line 264):
        
        # Assigning a Name to a Attribute (line 264):
        # Getting the type of 'mask' (line 264)
        mask_133943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'mask')
        # Getting the type of 'self' (line 264)
        self_133944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self')
        # Setting the type of the member 'mask' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_133944, 'mask', mask_133943)
        
        # Assigning a BinOp to a Attribute (line 266):
        
        # Assigning a BinOp to a Attribute (line 266):
        
        # Call to float(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'mask' (line 266)
        mask_133946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'mask', False)
        # Obtaining the member 'nx' of a type (line 266)
        nx_133947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 33), mask_133946, 'nx')
        int_133948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 43), 'int')
        # Applying the binary operator '-' (line 266)
        result_sub_133949 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 33), '-', nx_133947, int_133948)
        
        # Processing the call keyword arguments (line 266)
        kwargs_133950 = {}
        # Getting the type of 'float' (line 266)
        float_133945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'float', False)
        # Calling float(args, kwargs) (line 266)
        float_call_result_133951 = invoke(stypy.reporting.localization.Localization(__file__, 266, 27), float_133945, *[result_sub_133949], **kwargs_133950)
        
        # Getting the type of 'grid' (line 266)
        grid_133952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 48), 'grid')
        # Obtaining the member 'nx' of a type (line 266)
        nx_133953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 48), grid_133952, 'nx')
        # Applying the binary operator 'div' (line 266)
        result_div_133954 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 27), 'div', float_call_result_133951, nx_133953)
        
        # Getting the type of 'self' (line 266)
        self_133955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self')
        # Setting the type of the member 'x_grid2mask' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_133955, 'x_grid2mask', result_div_133954)
        
        # Assigning a BinOp to a Attribute (line 267):
        
        # Assigning a BinOp to a Attribute (line 267):
        
        # Call to float(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'mask' (line 267)
        mask_133957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 33), 'mask', False)
        # Obtaining the member 'ny' of a type (line 267)
        ny_133958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 33), mask_133957, 'ny')
        int_133959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 43), 'int')
        # Applying the binary operator '-' (line 267)
        result_sub_133960 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 33), '-', ny_133958, int_133959)
        
        # Processing the call keyword arguments (line 267)
        kwargs_133961 = {}
        # Getting the type of 'float' (line 267)
        float_133956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 27), 'float', False)
        # Calling float(args, kwargs) (line 267)
        float_call_result_133962 = invoke(stypy.reporting.localization.Localization(__file__, 267, 27), float_133956, *[result_sub_133960], **kwargs_133961)
        
        # Getting the type of 'grid' (line 267)
        grid_133963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 48), 'grid')
        # Obtaining the member 'ny' of a type (line 267)
        ny_133964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 48), grid_133963, 'ny')
        # Applying the binary operator 'div' (line 267)
        result_div_133965 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 27), 'div', float_call_result_133962, ny_133964)
        
        # Getting the type of 'self' (line 267)
        self_133966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self')
        # Setting the type of the member 'y_grid2mask' of a type (line 267)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_133966, 'y_grid2mask', result_div_133965)
        
        # Assigning a BinOp to a Attribute (line 269):
        
        # Assigning a BinOp to a Attribute (line 269):
        float_133967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 27), 'float')
        # Getting the type of 'self' (line 269)
        self_133968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'self')
        # Obtaining the member 'x_grid2mask' of a type (line 269)
        x_grid2mask_133969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 32), self_133968, 'x_grid2mask')
        # Applying the binary operator 'div' (line 269)
        result_div_133970 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 27), 'div', float_133967, x_grid2mask_133969)
        
        # Getting the type of 'self' (line 269)
        self_133971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Setting the type of the member 'x_mask2grid' of a type (line 269)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_133971, 'x_mask2grid', result_div_133970)
        
        # Assigning a BinOp to a Attribute (line 270):
        
        # Assigning a BinOp to a Attribute (line 270):
        float_133972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 27), 'float')
        # Getting the type of 'self' (line 270)
        self_133973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 32), 'self')
        # Obtaining the member 'y_grid2mask' of a type (line 270)
        y_grid2mask_133974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 32), self_133973, 'y_grid2mask')
        # Applying the binary operator 'div' (line 270)
        result_div_133975 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 27), 'div', float_133972, y_grid2mask_133974)
        
        # Getting the type of 'self' (line 270)
        self_133976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Setting the type of the member 'y_mask2grid' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_133976, 'y_mask2grid', result_div_133975)
        
        # Assigning a BinOp to a Attribute (line 272):
        
        # Assigning a BinOp to a Attribute (line 272):
        float_133977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 27), 'float')
        # Getting the type of 'grid' (line 272)
        grid_133978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 32), 'grid')
        # Obtaining the member 'dx' of a type (line 272)
        dx_133979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 32), grid_133978, 'dx')
        # Applying the binary operator 'div' (line 272)
        result_div_133980 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 27), 'div', float_133977, dx_133979)
        
        # Getting the type of 'self' (line 272)
        self_133981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member 'x_data2grid' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_133981, 'x_data2grid', result_div_133980)
        
        # Assigning a BinOp to a Attribute (line 273):
        
        # Assigning a BinOp to a Attribute (line 273):
        float_133982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'float')
        # Getting the type of 'grid' (line 273)
        grid_133983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 32), 'grid')
        # Obtaining the member 'dy' of a type (line 273)
        dy_133984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 32), grid_133983, 'dy')
        # Applying the binary operator 'div' (line 273)
        result_div_133985 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 27), 'div', float_133982, dy_133984)
        
        # Getting the type of 'self' (line 273)
        self_133986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self')
        # Setting the type of the member 'y_data2grid' of a type (line 273)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_133986, 'y_data2grid', result_div_133985)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def grid2mask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'grid2mask'
        module_type_store = module_type_store.open_function_context('grid2mask', 275, 4, False)
        # Assigning a type to the variable 'self' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DomainMap.grid2mask.__dict__.__setitem__('stypy_localization', localization)
        DomainMap.grid2mask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DomainMap.grid2mask.__dict__.__setitem__('stypy_type_store', module_type_store)
        DomainMap.grid2mask.__dict__.__setitem__('stypy_function_name', 'DomainMap.grid2mask')
        DomainMap.grid2mask.__dict__.__setitem__('stypy_param_names_list', ['xi', 'yi'])
        DomainMap.grid2mask.__dict__.__setitem__('stypy_varargs_param_name', None)
        DomainMap.grid2mask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DomainMap.grid2mask.__dict__.__setitem__('stypy_call_defaults', defaults)
        DomainMap.grid2mask.__dict__.__setitem__('stypy_call_varargs', varargs)
        DomainMap.grid2mask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DomainMap.grid2mask.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.grid2mask', ['xi', 'yi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'grid2mask', localization, ['xi', 'yi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'grid2mask(...)' code ##################

        unicode_133987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 8), 'unicode', u'Return nearest space in mask-coords from given grid-coords.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 277)
        tuple_133988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 277)
        # Adding element type (line 277)
        
        # Call to int(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'xi' (line 277)
        xi_133990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'xi', False)
        # Getting the type of 'self' (line 277)
        self_133991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 26), 'self', False)
        # Obtaining the member 'x_grid2mask' of a type (line 277)
        x_grid2mask_133992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 26), self_133991, 'x_grid2mask')
        # Applying the binary operator '*' (line 277)
        result_mul_133993 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 21), '*', xi_133990, x_grid2mask_133992)
        
        float_133994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 46), 'float')
        # Applying the binary operator '+' (line 277)
        result_add_133995 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 20), '+', result_mul_133993, float_133994)
        
        # Processing the call keyword arguments (line 277)
        kwargs_133996 = {}
        # Getting the type of 'int' (line 277)
        int_133989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'int', False)
        # Calling int(args, kwargs) (line 277)
        int_call_result_133997 = invoke(stypy.reporting.localization.Localization(__file__, 277, 16), int_133989, *[result_add_133995], **kwargs_133996)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 16), tuple_133988, int_call_result_133997)
        # Adding element type (line 277)
        
        # Call to int(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'yi' (line 278)
        yi_133999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'yi', False)
        # Getting the type of 'self' (line 278)
        self_134000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 26), 'self', False)
        # Obtaining the member 'y_grid2mask' of a type (line 278)
        y_grid2mask_134001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 26), self_134000, 'y_grid2mask')
        # Applying the binary operator '*' (line 278)
        result_mul_134002 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 21), '*', yi_133999, y_grid2mask_134001)
        
        float_134003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 46), 'float')
        # Applying the binary operator '+' (line 278)
        result_add_134004 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 20), '+', result_mul_134002, float_134003)
        
        # Processing the call keyword arguments (line 278)
        kwargs_134005 = {}
        # Getting the type of 'int' (line 278)
        int_133998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'int', False)
        # Calling int(args, kwargs) (line 278)
        int_call_result_134006 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), int_133998, *[result_add_134004], **kwargs_134005)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 16), tuple_133988, int_call_result_134006)
        
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'stypy_return_type', tuple_133988)
        
        # ################# End of 'grid2mask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'grid2mask' in the type store
        # Getting the type of 'stypy_return_type' (line 275)
        stypy_return_type_134007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134007)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'grid2mask'
        return stypy_return_type_134007


    @norecursion
    def mask2grid(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mask2grid'
        module_type_store = module_type_store.open_function_context('mask2grid', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DomainMap.mask2grid.__dict__.__setitem__('stypy_localization', localization)
        DomainMap.mask2grid.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DomainMap.mask2grid.__dict__.__setitem__('stypy_type_store', module_type_store)
        DomainMap.mask2grid.__dict__.__setitem__('stypy_function_name', 'DomainMap.mask2grid')
        DomainMap.mask2grid.__dict__.__setitem__('stypy_param_names_list', ['xm', 'ym'])
        DomainMap.mask2grid.__dict__.__setitem__('stypy_varargs_param_name', None)
        DomainMap.mask2grid.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DomainMap.mask2grid.__dict__.__setitem__('stypy_call_defaults', defaults)
        DomainMap.mask2grid.__dict__.__setitem__('stypy_call_varargs', varargs)
        DomainMap.mask2grid.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DomainMap.mask2grid.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.mask2grid', ['xm', 'ym'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mask2grid', localization, ['xm', 'ym'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mask2grid(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 281)
        tuple_134008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 281)
        # Adding element type (line 281)
        # Getting the type of 'xm' (line 281)
        xm_134009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'xm')
        # Getting the type of 'self' (line 281)
        self_134010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'self')
        # Obtaining the member 'x_mask2grid' of a type (line 281)
        x_mask2grid_134011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 20), self_134010, 'x_mask2grid')
        # Applying the binary operator '*' (line 281)
        result_mul_134012 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 15), '*', xm_134009, x_mask2grid_134011)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 15), tuple_134008, result_mul_134012)
        # Adding element type (line 281)
        # Getting the type of 'ym' (line 281)
        ym_134013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 38), 'ym')
        # Getting the type of 'self' (line 281)
        self_134014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 43), 'self')
        # Obtaining the member 'y_mask2grid' of a type (line 281)
        y_mask2grid_134015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 43), self_134014, 'y_mask2grid')
        # Applying the binary operator '*' (line 281)
        result_mul_134016 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 38), '*', ym_134013, y_mask2grid_134015)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 15), tuple_134008, result_mul_134016)
        
        # Assigning a type to the variable 'stypy_return_type' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'stypy_return_type', tuple_134008)
        
        # ################# End of 'mask2grid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mask2grid' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_134017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134017)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mask2grid'
        return stypy_return_type_134017


    @norecursion
    def data2grid(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'data2grid'
        module_type_store = module_type_store.open_function_context('data2grid', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DomainMap.data2grid.__dict__.__setitem__('stypy_localization', localization)
        DomainMap.data2grid.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DomainMap.data2grid.__dict__.__setitem__('stypy_type_store', module_type_store)
        DomainMap.data2grid.__dict__.__setitem__('stypy_function_name', 'DomainMap.data2grid')
        DomainMap.data2grid.__dict__.__setitem__('stypy_param_names_list', ['xd', 'yd'])
        DomainMap.data2grid.__dict__.__setitem__('stypy_varargs_param_name', None)
        DomainMap.data2grid.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DomainMap.data2grid.__dict__.__setitem__('stypy_call_defaults', defaults)
        DomainMap.data2grid.__dict__.__setitem__('stypy_call_varargs', varargs)
        DomainMap.data2grid.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DomainMap.data2grid.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.data2grid', ['xd', 'yd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'data2grid', localization, ['xd', 'yd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'data2grid(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_134018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        # Getting the type of 'xd' (line 284)
        xd_134019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), 'xd')
        # Getting the type of 'self' (line 284)
        self_134020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'self')
        # Obtaining the member 'x_data2grid' of a type (line 284)
        x_data2grid_134021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 20), self_134020, 'x_data2grid')
        # Applying the binary operator '*' (line 284)
        result_mul_134022 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 15), '*', xd_134019, x_data2grid_134021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), tuple_134018, result_mul_134022)
        # Adding element type (line 284)
        # Getting the type of 'yd' (line 284)
        yd_134023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 38), 'yd')
        # Getting the type of 'self' (line 284)
        self_134024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 43), 'self')
        # Obtaining the member 'y_data2grid' of a type (line 284)
        y_data2grid_134025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 43), self_134024, 'y_data2grid')
        # Applying the binary operator '*' (line 284)
        result_mul_134026 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 38), '*', yd_134023, y_data2grid_134025)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), tuple_134018, result_mul_134026)
        
        # Assigning a type to the variable 'stypy_return_type' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'stypy_return_type', tuple_134018)
        
        # ################# End of 'data2grid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'data2grid' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_134027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'data2grid'
        return stypy_return_type_134027


    @norecursion
    def grid2data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'grid2data'
        module_type_store = module_type_store.open_function_context('grid2data', 286, 4, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DomainMap.grid2data.__dict__.__setitem__('stypy_localization', localization)
        DomainMap.grid2data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DomainMap.grid2data.__dict__.__setitem__('stypy_type_store', module_type_store)
        DomainMap.grid2data.__dict__.__setitem__('stypy_function_name', 'DomainMap.grid2data')
        DomainMap.grid2data.__dict__.__setitem__('stypy_param_names_list', ['xg', 'yg'])
        DomainMap.grid2data.__dict__.__setitem__('stypy_varargs_param_name', None)
        DomainMap.grid2data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DomainMap.grid2data.__dict__.__setitem__('stypy_call_defaults', defaults)
        DomainMap.grid2data.__dict__.__setitem__('stypy_call_varargs', varargs)
        DomainMap.grid2data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DomainMap.grid2data.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.grid2data', ['xg', 'yg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'grid2data', localization, ['xg', 'yg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'grid2data(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 287)
        tuple_134028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 287)
        # Adding element type (line 287)
        # Getting the type of 'xg' (line 287)
        xg_134029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'xg')
        # Getting the type of 'self' (line 287)
        self_134030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'self')
        # Obtaining the member 'x_data2grid' of a type (line 287)
        x_data2grid_134031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 20), self_134030, 'x_data2grid')
        # Applying the binary operator 'div' (line 287)
        result_div_134032 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 15), 'div', xg_134029, x_data2grid_134031)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 15), tuple_134028, result_div_134032)
        # Adding element type (line 287)
        # Getting the type of 'yg' (line 287)
        yg_134033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 38), 'yg')
        # Getting the type of 'self' (line 287)
        self_134034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 43), 'self')
        # Obtaining the member 'y_data2grid' of a type (line 287)
        y_data2grid_134035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 43), self_134034, 'y_data2grid')
        # Applying the binary operator 'div' (line 287)
        result_div_134036 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 38), 'div', yg_134033, y_data2grid_134035)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 15), tuple_134028, result_div_134036)
        
        # Assigning a type to the variable 'stypy_return_type' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'stypy_return_type', tuple_134028)
        
        # ################# End of 'grid2data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'grid2data' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_134037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134037)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'grid2data'
        return stypy_return_type_134037


    @norecursion
    def start_trajectory(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'start_trajectory'
        module_type_store = module_type_store.open_function_context('start_trajectory', 289, 4, False)
        # Assigning a type to the variable 'self' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_localization', localization)
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_type_store', module_type_store)
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_function_name', 'DomainMap.start_trajectory')
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_param_names_list', ['xg', 'yg'])
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_varargs_param_name', None)
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_call_defaults', defaults)
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_call_varargs', varargs)
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DomainMap.start_trajectory.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.start_trajectory', ['xg', 'yg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'start_trajectory', localization, ['xg', 'yg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'start_trajectory(...)' code ##################

        
        # Assigning a Call to a Tuple (line 290):
        
        # Assigning a Call to a Name:
        
        # Call to grid2mask(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'xg' (line 290)
        xg_134040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'xg', False)
        # Getting the type of 'yg' (line 290)
        yg_134041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 36), 'yg', False)
        # Processing the call keyword arguments (line 290)
        kwargs_134042 = {}
        # Getting the type of 'self' (line 290)
        self_134038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'self', False)
        # Obtaining the member 'grid2mask' of a type (line 290)
        grid2mask_134039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 17), self_134038, 'grid2mask')
        # Calling grid2mask(args, kwargs) (line 290)
        grid2mask_call_result_134043 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), grid2mask_134039, *[xg_134040, yg_134041], **kwargs_134042)
        
        # Assigning a type to the variable 'call_assignment_133176' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_133176', grid2mask_call_result_134043)
        
        # Assigning a Call to a Name (line 290):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        # Processing the call keyword arguments
        kwargs_134047 = {}
        # Getting the type of 'call_assignment_133176' (line 290)
        call_assignment_133176_134044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_133176', False)
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___134045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), call_assignment_133176_134044, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134048 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134045, *[int_134046], **kwargs_134047)
        
        # Assigning a type to the variable 'call_assignment_133177' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_133177', getitem___call_result_134048)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'call_assignment_133177' (line 290)
        call_assignment_133177_134049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_133177')
        # Assigning a type to the variable 'xm' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'xm', call_assignment_133177_134049)
        
        # Assigning a Call to a Name (line 290):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        # Processing the call keyword arguments
        kwargs_134053 = {}
        # Getting the type of 'call_assignment_133176' (line 290)
        call_assignment_133176_134050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_133176', False)
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___134051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), call_assignment_133176_134050, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134054 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134051, *[int_134052], **kwargs_134053)
        
        # Assigning a type to the variable 'call_assignment_133178' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_133178', getitem___call_result_134054)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'call_assignment_133178' (line 290)
        call_assignment_133178_134055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_133178')
        # Assigning a type to the variable 'ym' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'ym', call_assignment_133178_134055)
        
        # Call to _start_trajectory(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'xm' (line 291)
        xm_134059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 36), 'xm', False)
        # Getting the type of 'ym' (line 291)
        ym_134060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 40), 'ym', False)
        # Processing the call keyword arguments (line 291)
        kwargs_134061 = {}
        # Getting the type of 'self' (line 291)
        self_134056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'self', False)
        # Obtaining the member 'mask' of a type (line 291)
        mask_134057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), self_134056, 'mask')
        # Obtaining the member '_start_trajectory' of a type (line 291)
        _start_trajectory_134058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), mask_134057, '_start_trajectory')
        # Calling _start_trajectory(args, kwargs) (line 291)
        _start_trajectory_call_result_134062 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), _start_trajectory_134058, *[xm_134059, ym_134060], **kwargs_134061)
        
        
        # ################# End of 'start_trajectory(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'start_trajectory' in the type store
        # Getting the type of 'stypy_return_type' (line 289)
        stypy_return_type_134063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'start_trajectory'
        return stypy_return_type_134063


    @norecursion
    def reset_start_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset_start_point'
        module_type_store = module_type_store.open_function_context('reset_start_point', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_localization', localization)
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_function_name', 'DomainMap.reset_start_point')
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_param_names_list', ['xg', 'yg'])
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DomainMap.reset_start_point.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.reset_start_point', ['xg', 'yg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset_start_point', localization, ['xg', 'yg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset_start_point(...)' code ##################

        
        # Assigning a Call to a Tuple (line 294):
        
        # Assigning a Call to a Name:
        
        # Call to grid2mask(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'xg' (line 294)
        xg_134066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 32), 'xg', False)
        # Getting the type of 'yg' (line 294)
        yg_134067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 36), 'yg', False)
        # Processing the call keyword arguments (line 294)
        kwargs_134068 = {}
        # Getting the type of 'self' (line 294)
        self_134064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'self', False)
        # Obtaining the member 'grid2mask' of a type (line 294)
        grid2mask_134065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 17), self_134064, 'grid2mask')
        # Calling grid2mask(args, kwargs) (line 294)
        grid2mask_call_result_134069 = invoke(stypy.reporting.localization.Localization(__file__, 294, 17), grid2mask_134065, *[xg_134066, yg_134067], **kwargs_134068)
        
        # Assigning a type to the variable 'call_assignment_133179' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'call_assignment_133179', grid2mask_call_result_134069)
        
        # Assigning a Call to a Name (line 294):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 8), 'int')
        # Processing the call keyword arguments
        kwargs_134073 = {}
        # Getting the type of 'call_assignment_133179' (line 294)
        call_assignment_133179_134070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'call_assignment_133179', False)
        # Obtaining the member '__getitem__' of a type (line 294)
        getitem___134071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), call_assignment_133179_134070, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134074 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134071, *[int_134072], **kwargs_134073)
        
        # Assigning a type to the variable 'call_assignment_133180' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'call_assignment_133180', getitem___call_result_134074)
        
        # Assigning a Name to a Name (line 294):
        # Getting the type of 'call_assignment_133180' (line 294)
        call_assignment_133180_134075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'call_assignment_133180')
        # Assigning a type to the variable 'xm' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'xm', call_assignment_133180_134075)
        
        # Assigning a Call to a Name (line 294):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 8), 'int')
        # Processing the call keyword arguments
        kwargs_134079 = {}
        # Getting the type of 'call_assignment_133179' (line 294)
        call_assignment_133179_134076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'call_assignment_133179', False)
        # Obtaining the member '__getitem__' of a type (line 294)
        getitem___134077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), call_assignment_133179_134076, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134080 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134077, *[int_134078], **kwargs_134079)
        
        # Assigning a type to the variable 'call_assignment_133181' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'call_assignment_133181', getitem___call_result_134080)
        
        # Assigning a Name to a Name (line 294):
        # Getting the type of 'call_assignment_133181' (line 294)
        call_assignment_133181_134081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'call_assignment_133181')
        # Assigning a type to the variable 'ym' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'ym', call_assignment_133181_134081)
        
        # Assigning a Tuple to a Attribute (line 295):
        
        # Assigning a Tuple to a Attribute (line 295):
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_134082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        # Getting the type of 'xm' (line 295)
        xm_134083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 33), 'xm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 33), tuple_134082, xm_134083)
        # Adding element type (line 295)
        # Getting the type of 'ym' (line 295)
        ym_134084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 37), 'ym')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 33), tuple_134082, ym_134084)
        
        # Getting the type of 'self' (line 295)
        self_134085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self')
        # Obtaining the member 'mask' of a type (line 295)
        mask_134086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_134085, 'mask')
        # Setting the type of the member '_current_xy' of a type (line 295)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), mask_134086, '_current_xy', tuple_134082)
        
        # ################# End of 'reset_start_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset_start_point' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_134087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134087)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset_start_point'
        return stypy_return_type_134087


    @norecursion
    def update_trajectory(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_trajectory'
        module_type_store = module_type_store.open_function_context('update_trajectory', 297, 4, False)
        # Assigning a type to the variable 'self' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_localization', localization)
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_type_store', module_type_store)
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_function_name', 'DomainMap.update_trajectory')
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_param_names_list', ['xg', 'yg'])
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_varargs_param_name', None)
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_call_defaults', defaults)
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_call_varargs', varargs)
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DomainMap.update_trajectory.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.update_trajectory', ['xg', 'yg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_trajectory', localization, ['xg', 'yg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_trajectory(...)' code ##################

        
        
        
        # Call to within_grid(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'xg' (line 298)
        xg_134091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 37), 'xg', False)
        # Getting the type of 'yg' (line 298)
        yg_134092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'yg', False)
        # Processing the call keyword arguments (line 298)
        kwargs_134093 = {}
        # Getting the type of 'self' (line 298)
        self_134088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'self', False)
        # Obtaining the member 'grid' of a type (line 298)
        grid_134089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 15), self_134088, 'grid')
        # Obtaining the member 'within_grid' of a type (line 298)
        within_grid_134090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 15), grid_134089, 'within_grid')
        # Calling within_grid(args, kwargs) (line 298)
        within_grid_call_result_134094 = invoke(stypy.reporting.localization.Localization(__file__, 298, 15), within_grid_134090, *[xg_134091, yg_134092], **kwargs_134093)
        
        # Applying the 'not' unary operator (line 298)
        result_not__134095 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 11), 'not', within_grid_call_result_134094)
        
        # Testing the type of an if condition (line 298)
        if_condition_134096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), result_not__134095)
        # Assigning a type to the variable 'if_condition_134096' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_134096', if_condition_134096)
        # SSA begins for if statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'InvalidIndexError' (line 299)
        InvalidIndexError_134097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'InvalidIndexError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 299, 12), InvalidIndexError_134097, 'raise parameter', BaseException)
        # SSA join for if statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 300):
        
        # Assigning a Call to a Name:
        
        # Call to grid2mask(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'xg' (line 300)
        xg_134100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 32), 'xg', False)
        # Getting the type of 'yg' (line 300)
        yg_134101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 36), 'yg', False)
        # Processing the call keyword arguments (line 300)
        kwargs_134102 = {}
        # Getting the type of 'self' (line 300)
        self_134098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 17), 'self', False)
        # Obtaining the member 'grid2mask' of a type (line 300)
        grid2mask_134099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 17), self_134098, 'grid2mask')
        # Calling grid2mask(args, kwargs) (line 300)
        grid2mask_call_result_134103 = invoke(stypy.reporting.localization.Localization(__file__, 300, 17), grid2mask_134099, *[xg_134100, yg_134101], **kwargs_134102)
        
        # Assigning a type to the variable 'call_assignment_133182' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'call_assignment_133182', grid2mask_call_result_134103)
        
        # Assigning a Call to a Name (line 300):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 8), 'int')
        # Processing the call keyword arguments
        kwargs_134107 = {}
        # Getting the type of 'call_assignment_133182' (line 300)
        call_assignment_133182_134104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'call_assignment_133182', False)
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___134105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), call_assignment_133182_134104, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134108 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134105, *[int_134106], **kwargs_134107)
        
        # Assigning a type to the variable 'call_assignment_133183' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'call_assignment_133183', getitem___call_result_134108)
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'call_assignment_133183' (line 300)
        call_assignment_133183_134109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'call_assignment_133183')
        # Assigning a type to the variable 'xm' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'xm', call_assignment_133183_134109)
        
        # Assigning a Call to a Name (line 300):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 8), 'int')
        # Processing the call keyword arguments
        kwargs_134113 = {}
        # Getting the type of 'call_assignment_133182' (line 300)
        call_assignment_133182_134110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'call_assignment_133182', False)
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___134111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), call_assignment_133182_134110, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134114 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134111, *[int_134112], **kwargs_134113)
        
        # Assigning a type to the variable 'call_assignment_133184' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'call_assignment_133184', getitem___call_result_134114)
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'call_assignment_133184' (line 300)
        call_assignment_133184_134115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'call_assignment_133184')
        # Assigning a type to the variable 'ym' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'ym', call_assignment_133184_134115)
        
        # Call to _update_trajectory(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'xm' (line 301)
        xm_134119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 37), 'xm', False)
        # Getting the type of 'ym' (line 301)
        ym_134120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 41), 'ym', False)
        # Processing the call keyword arguments (line 301)
        kwargs_134121 = {}
        # Getting the type of 'self' (line 301)
        self_134116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'self', False)
        # Obtaining the member 'mask' of a type (line 301)
        mask_134117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), self_134116, 'mask')
        # Obtaining the member '_update_trajectory' of a type (line 301)
        _update_trajectory_134118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), mask_134117, '_update_trajectory')
        # Calling _update_trajectory(args, kwargs) (line 301)
        _update_trajectory_call_result_134122 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), _update_trajectory_134118, *[xm_134119, ym_134120], **kwargs_134121)
        
        
        # ################# End of 'update_trajectory(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_trajectory' in the type store
        # Getting the type of 'stypy_return_type' (line 297)
        stypy_return_type_134123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_trajectory'
        return stypy_return_type_134123


    @norecursion
    def undo_trajectory(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'undo_trajectory'
        module_type_store = module_type_store.open_function_context('undo_trajectory', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_localization', localization)
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_type_store', module_type_store)
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_function_name', 'DomainMap.undo_trajectory')
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_param_names_list', [])
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_varargs_param_name', None)
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_call_defaults', defaults)
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_call_varargs', varargs)
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DomainMap.undo_trajectory.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DomainMap.undo_trajectory', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'undo_trajectory', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'undo_trajectory(...)' code ##################

        
        # Call to _undo_trajectory(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_134127 = {}
        # Getting the type of 'self' (line 304)
        self_134124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'self', False)
        # Obtaining the member 'mask' of a type (line 304)
        mask_134125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), self_134124, 'mask')
        # Obtaining the member '_undo_trajectory' of a type (line 304)
        _undo_trajectory_134126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), mask_134125, '_undo_trajectory')
        # Calling _undo_trajectory(args, kwargs) (line 304)
        _undo_trajectory_call_result_134128 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), _undo_trajectory_134126, *[], **kwargs_134127)
        
        
        # ################# End of 'undo_trajectory(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'undo_trajectory' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_134129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'undo_trajectory'
        return stypy_return_type_134129


# Assigning a type to the variable 'DomainMap' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'DomainMap', DomainMap)
# Declaration of the 'Grid' class

class Grid(object, ):
    unicode_134130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 4), 'unicode', u'Grid of data.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Grid.__init__', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Getting the type of 'x' (line 311)
        x_134131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'x')
        # Obtaining the member 'ndim' of a type (line 311)
        ndim_134132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 11), x_134131, 'ndim')
        int_134133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 21), 'int')
        # Applying the binary operator '==' (line 311)
        result_eq_134134 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 11), '==', ndim_134132, int_134133)
        
        # Testing the type of an if condition (line 311)
        if_condition_134135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), result_eq_134134)
        # Assigning a type to the variable 'if_condition_134135' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_134135', if_condition_134135)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 311)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'x' (line 313)
        x_134136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'x')
        # Obtaining the member 'ndim' of a type (line 313)
        ndim_134137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 13), x_134136, 'ndim')
        int_134138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 23), 'int')
        # Applying the binary operator '==' (line 313)
        result_eq_134139 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 13), '==', ndim_134137, int_134138)
        
        # Testing the type of an if condition (line 313)
        if_condition_134140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 13), result_eq_134139)
        # Assigning a type to the variable 'if_condition_134140' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'if_condition_134140', if_condition_134140)
        # SSA begins for if statement (line 313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 314):
        
        # Assigning a Subscript to a Name (line 314):
        
        # Obtaining the type of the subscript
        int_134141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 22), 'int')
        slice_134142 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 20), None, None, None)
        # Getting the type of 'x' (line 314)
        x_134143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'x')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___134144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 20), x_134143, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_134145 = invoke(stypy.reporting.localization.Localization(__file__, 314, 20), getitem___134144, (int_134141, slice_134142))
        
        # Assigning a type to the variable 'x_row' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'x_row', subscript_call_result_134145)
        
        
        
        # Call to allclose(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'x_row' (line 315)
        x_row_134148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 31), 'x_row', False)
        # Getting the type of 'x' (line 315)
        x_134149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 38), 'x', False)
        # Processing the call keyword arguments (line 315)
        kwargs_134150 = {}
        # Getting the type of 'np' (line 315)
        np_134146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'np', False)
        # Obtaining the member 'allclose' of a type (line 315)
        allclose_134147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 19), np_134146, 'allclose')
        # Calling allclose(args, kwargs) (line 315)
        allclose_call_result_134151 = invoke(stypy.reporting.localization.Localization(__file__, 315, 19), allclose_134147, *[x_row_134148, x_134149], **kwargs_134150)
        
        # Applying the 'not' unary operator (line 315)
        result_not__134152 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 15), 'not', allclose_call_result_134151)
        
        # Testing the type of an if condition (line 315)
        if_condition_134153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 12), result_not__134152)
        # Assigning a type to the variable 'if_condition_134153' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'if_condition_134153', if_condition_134153)
        # SSA begins for if statement (line 315)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 316)
        # Processing the call arguments (line 316)
        unicode_134155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 33), 'unicode', u"The rows of 'x' must be equal")
        # Processing the call keyword arguments (line 316)
        kwargs_134156 = {}
        # Getting the type of 'ValueError' (line 316)
        ValueError_134154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 316)
        ValueError_call_result_134157 = invoke(stypy.reporting.localization.Localization(__file__, 316, 22), ValueError_134154, *[unicode_134155], **kwargs_134156)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 316, 16), ValueError_call_result_134157, 'raise parameter', BaseException)
        # SSA join for if statement (line 315)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 317):
        
        # Assigning a Name to a Name (line 317):
        # Getting the type of 'x_row' (line 317)
        x_row_134158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 'x_row')
        # Assigning a type to the variable 'x' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'x', x_row_134158)
        # SSA branch for the else part of an if statement (line 313)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 319)
        # Processing the call arguments (line 319)
        unicode_134160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 29), 'unicode', u"'x' can have at maximum 2 dimensions")
        # Processing the call keyword arguments (line 319)
        kwargs_134161 = {}
        # Getting the type of 'ValueError' (line 319)
        ValueError_134159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 319)
        ValueError_call_result_134162 = invoke(stypy.reporting.localization.Localization(__file__, 319, 18), ValueError_134159, *[unicode_134160], **kwargs_134161)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 319, 12), ValueError_call_result_134162, 'raise parameter', BaseException)
        # SSA join for if statement (line 313)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'y' (line 321)
        y_134163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 'y')
        # Obtaining the member 'ndim' of a type (line 321)
        ndim_134164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 11), y_134163, 'ndim')
        int_134165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 21), 'int')
        # Applying the binary operator '==' (line 321)
        result_eq_134166 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 11), '==', ndim_134164, int_134165)
        
        # Testing the type of an if condition (line 321)
        if_condition_134167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 8), result_eq_134166)
        # Assigning a type to the variable 'if_condition_134167' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'if_condition_134167', if_condition_134167)
        # SSA begins for if statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 321)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'y' (line 323)
        y_134168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'y')
        # Obtaining the member 'ndim' of a type (line 323)
        ndim_134169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 13), y_134168, 'ndim')
        int_134170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 23), 'int')
        # Applying the binary operator '==' (line 323)
        result_eq_134171 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 13), '==', ndim_134169, int_134170)
        
        # Testing the type of an if condition (line 323)
        if_condition_134172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 13), result_eq_134171)
        # Assigning a type to the variable 'if_condition_134172' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'if_condition_134172', if_condition_134172)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 324):
        
        # Assigning a Subscript to a Name (line 324):
        
        # Obtaining the type of the subscript
        slice_134173 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 324, 20), None, None, None)
        int_134174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 25), 'int')
        # Getting the type of 'y' (line 324)
        y_134175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'y')
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___134176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 20), y_134175, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 324)
        subscript_call_result_134177 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), getitem___134176, (slice_134173, int_134174))
        
        # Assigning a type to the variable 'y_col' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'y_col', subscript_call_result_134177)
        
        
        
        # Call to allclose(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'y_col' (line 325)
        y_col_134180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 31), 'y_col', False)
        # Getting the type of 'y' (line 325)
        y_134181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'y', False)
        # Obtaining the member 'T' of a type (line 325)
        T_134182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 38), y_134181, 'T')
        # Processing the call keyword arguments (line 325)
        kwargs_134183 = {}
        # Getting the type of 'np' (line 325)
        np_134178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'np', False)
        # Obtaining the member 'allclose' of a type (line 325)
        allclose_134179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 19), np_134178, 'allclose')
        # Calling allclose(args, kwargs) (line 325)
        allclose_call_result_134184 = invoke(stypy.reporting.localization.Localization(__file__, 325, 19), allclose_134179, *[y_col_134180, T_134182], **kwargs_134183)
        
        # Applying the 'not' unary operator (line 325)
        result_not__134185 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 15), 'not', allclose_call_result_134184)
        
        # Testing the type of an if condition (line 325)
        if_condition_134186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 12), result_not__134185)
        # Assigning a type to the variable 'if_condition_134186' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'if_condition_134186', if_condition_134186)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 326)
        # Processing the call arguments (line 326)
        unicode_134188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 33), 'unicode', u"The columns of 'y' must be equal")
        # Processing the call keyword arguments (line 326)
        kwargs_134189 = {}
        # Getting the type of 'ValueError' (line 326)
        ValueError_134187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 326)
        ValueError_call_result_134190 = invoke(stypy.reporting.localization.Localization(__file__, 326, 22), ValueError_134187, *[unicode_134188], **kwargs_134189)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 326, 16), ValueError_call_result_134190, 'raise parameter', BaseException)
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 327):
        
        # Assigning a Name to a Name (line 327):
        # Getting the type of 'y_col' (line 327)
        y_col_134191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'y_col')
        # Assigning a type to the variable 'y' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'y', y_col_134191)
        # SSA branch for the else part of an if statement (line 323)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 329)
        # Processing the call arguments (line 329)
        unicode_134193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 29), 'unicode', u"'y' can have at maximum 2 dimensions")
        # Processing the call keyword arguments (line 329)
        kwargs_134194 = {}
        # Getting the type of 'ValueError' (line 329)
        ValueError_134192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 329)
        ValueError_call_result_134195 = invoke(stypy.reporting.localization.Localization(__file__, 329, 18), ValueError_134192, *[unicode_134193], **kwargs_134194)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 329, 12), ValueError_call_result_134195, 'raise parameter', BaseException)
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 321)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 331):
        
        # Assigning a Call to a Attribute (line 331):
        
        # Call to len(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'x' (line 331)
        x_134197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 22), 'x', False)
        # Processing the call keyword arguments (line 331)
        kwargs_134198 = {}
        # Getting the type of 'len' (line 331)
        len_134196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 18), 'len', False)
        # Calling len(args, kwargs) (line 331)
        len_call_result_134199 = invoke(stypy.reporting.localization.Localization(__file__, 331, 18), len_134196, *[x_134197], **kwargs_134198)
        
        # Getting the type of 'self' (line 331)
        self_134200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'self')
        # Setting the type of the member 'nx' of a type (line 331)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), self_134200, 'nx', len_call_result_134199)
        
        # Assigning a Call to a Attribute (line 332):
        
        # Assigning a Call to a Attribute (line 332):
        
        # Call to len(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'y' (line 332)
        y_134202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 22), 'y', False)
        # Processing the call keyword arguments (line 332)
        kwargs_134203 = {}
        # Getting the type of 'len' (line 332)
        len_134201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'len', False)
        # Calling len(args, kwargs) (line 332)
        len_call_result_134204 = invoke(stypy.reporting.localization.Localization(__file__, 332, 18), len_134201, *[y_134202], **kwargs_134203)
        
        # Getting the type of 'self' (line 332)
        self_134205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self')
        # Setting the type of the member 'ny' of a type (line 332)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_134205, 'ny', len_call_result_134204)
        
        # Assigning a BinOp to a Attribute (line 334):
        
        # Assigning a BinOp to a Attribute (line 334):
        
        # Obtaining the type of the subscript
        int_134206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 20), 'int')
        # Getting the type of 'x' (line 334)
        x_134207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 18), 'x')
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___134208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 18), x_134207, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_134209 = invoke(stypy.reporting.localization.Localization(__file__, 334, 18), getitem___134208, int_134206)
        
        
        # Obtaining the type of the subscript
        int_134210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 27), 'int')
        # Getting the type of 'x' (line 334)
        x_134211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 25), 'x')
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___134212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 25), x_134211, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_134213 = invoke(stypy.reporting.localization.Localization(__file__, 334, 25), getitem___134212, int_134210)
        
        # Applying the binary operator '-' (line 334)
        result_sub_134214 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 18), '-', subscript_call_result_134209, subscript_call_result_134213)
        
        # Getting the type of 'self' (line 334)
        self_134215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'self')
        # Setting the type of the member 'dx' of a type (line 334)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), self_134215, 'dx', result_sub_134214)
        
        # Assigning a BinOp to a Attribute (line 335):
        
        # Assigning a BinOp to a Attribute (line 335):
        
        # Obtaining the type of the subscript
        int_134216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 20), 'int')
        # Getting the type of 'y' (line 335)
        y_134217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 18), 'y')
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___134218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 18), y_134217, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_134219 = invoke(stypy.reporting.localization.Localization(__file__, 335, 18), getitem___134218, int_134216)
        
        
        # Obtaining the type of the subscript
        int_134220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 27), 'int')
        # Getting the type of 'y' (line 335)
        y_134221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'y')
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___134222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 25), y_134221, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_134223 = invoke(stypy.reporting.localization.Localization(__file__, 335, 25), getitem___134222, int_134220)
        
        # Applying the binary operator '-' (line 335)
        result_sub_134224 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 18), '-', subscript_call_result_134219, subscript_call_result_134223)
        
        # Getting the type of 'self' (line 335)
        self_134225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'self')
        # Setting the type of the member 'dy' of a type (line 335)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), self_134225, 'dy', result_sub_134224)
        
        # Assigning a Subscript to a Attribute (line 337):
        
        # Assigning a Subscript to a Attribute (line 337):
        
        # Obtaining the type of the subscript
        int_134226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 26), 'int')
        # Getting the type of 'x' (line 337)
        x_134227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'x')
        # Obtaining the member '__getitem__' of a type (line 337)
        getitem___134228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 24), x_134227, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 337)
        subscript_call_result_134229 = invoke(stypy.reporting.localization.Localization(__file__, 337, 24), getitem___134228, int_134226)
        
        # Getting the type of 'self' (line 337)
        self_134230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'self')
        # Setting the type of the member 'x_origin' of a type (line 337)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), self_134230, 'x_origin', subscript_call_result_134229)
        
        # Assigning a Subscript to a Attribute (line 338):
        
        # Assigning a Subscript to a Attribute (line 338):
        
        # Obtaining the type of the subscript
        int_134231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 26), 'int')
        # Getting the type of 'y' (line 338)
        y_134232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'y')
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___134233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 24), y_134232, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_134234 = invoke(stypy.reporting.localization.Localization(__file__, 338, 24), getitem___134233, int_134231)
        
        # Getting the type of 'self' (line 338)
        self_134235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'self')
        # Setting the type of the member 'y_origin' of a type (line 338)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), self_134235, 'y_origin', subscript_call_result_134234)
        
        # Assigning a BinOp to a Attribute (line 340):
        
        # Assigning a BinOp to a Attribute (line 340):
        
        # Obtaining the type of the subscript
        int_134236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 23), 'int')
        # Getting the type of 'x' (line 340)
        x_134237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 21), 'x')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___134238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 21), x_134237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_134239 = invoke(stypy.reporting.localization.Localization(__file__, 340, 21), getitem___134238, int_134236)
        
        
        # Obtaining the type of the subscript
        int_134240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 31), 'int')
        # Getting the type of 'x' (line 340)
        x_134241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 29), 'x')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___134242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 29), x_134241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_134243 = invoke(stypy.reporting.localization.Localization(__file__, 340, 29), getitem___134242, int_134240)
        
        # Applying the binary operator '-' (line 340)
        result_sub_134244 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 21), '-', subscript_call_result_134239, subscript_call_result_134243)
        
        # Getting the type of 'self' (line 340)
        self_134245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self')
        # Setting the type of the member 'width' of a type (line 340)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_134245, 'width', result_sub_134244)
        
        # Assigning a BinOp to a Attribute (line 341):
        
        # Assigning a BinOp to a Attribute (line 341):
        
        # Obtaining the type of the subscript
        int_134246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 24), 'int')
        # Getting the type of 'y' (line 341)
        y_134247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 22), 'y')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___134248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 22), y_134247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_134249 = invoke(stypy.reporting.localization.Localization(__file__, 341, 22), getitem___134248, int_134246)
        
        
        # Obtaining the type of the subscript
        int_134250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 32), 'int')
        # Getting the type of 'y' (line 341)
        y_134251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'y')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___134252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 30), y_134251, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_134253 = invoke(stypy.reporting.localization.Localization(__file__, 341, 30), getitem___134252, int_134250)
        
        # Applying the binary operator '-' (line 341)
        result_sub_134254 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 22), '-', subscript_call_result_134249, subscript_call_result_134253)
        
        # Getting the type of 'self' (line 341)
        self_134255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self')
        # Setting the type of the member 'height' of a type (line 341)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_134255, 'height', result_sub_134254)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shape'
        module_type_store = module_type_store.open_function_context('shape', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Grid.shape.__dict__.__setitem__('stypy_localization', localization)
        Grid.shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Grid.shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        Grid.shape.__dict__.__setitem__('stypy_function_name', 'Grid.shape')
        Grid.shape.__dict__.__setitem__('stypy_param_names_list', [])
        Grid.shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        Grid.shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Grid.shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        Grid.shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        Grid.shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Grid.shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Grid.shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shape(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 345)
        tuple_134256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 345)
        # Adding element type (line 345)
        # Getting the type of 'self' (line 345)
        self_134257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 15), 'self')
        # Obtaining the member 'ny' of a type (line 345)
        ny_134258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 15), self_134257, 'ny')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 15), tuple_134256, ny_134258)
        # Adding element type (line 345)
        # Getting the type of 'self' (line 345)
        self_134259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'self')
        # Obtaining the member 'nx' of a type (line 345)
        nx_134260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 24), self_134259, 'nx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 15), tuple_134256, nx_134260)
        
        # Assigning a type to the variable 'stypy_return_type' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'stypy_return_type', tuple_134256)
        
        # ################# End of 'shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shape' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_134261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shape'
        return stypy_return_type_134261


    @norecursion
    def within_grid(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'within_grid'
        module_type_store = module_type_store.open_function_context('within_grid', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Grid.within_grid.__dict__.__setitem__('stypy_localization', localization)
        Grid.within_grid.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Grid.within_grid.__dict__.__setitem__('stypy_type_store', module_type_store)
        Grid.within_grid.__dict__.__setitem__('stypy_function_name', 'Grid.within_grid')
        Grid.within_grid.__dict__.__setitem__('stypy_param_names_list', ['xi', 'yi'])
        Grid.within_grid.__dict__.__setitem__('stypy_varargs_param_name', None)
        Grid.within_grid.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Grid.within_grid.__dict__.__setitem__('stypy_call_defaults', defaults)
        Grid.within_grid.__dict__.__setitem__('stypy_call_varargs', varargs)
        Grid.within_grid.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Grid.within_grid.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Grid.within_grid', ['xi', 'yi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'within_grid', localization, ['xi', 'yi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'within_grid(...)' code ##################

        unicode_134262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 8), 'unicode', u'Return True if point is a valid index of grid.')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'xi' (line 351)
        xi_134263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), 'xi')
        int_134264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 21), 'int')
        # Applying the binary operator '>=' (line 351)
        result_ge_134265 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), '>=', xi_134263, int_134264)
        
        
        # Getting the type of 'xi' (line 351)
        xi_134266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 27), 'xi')
        # Getting the type of 'self' (line 351)
        self_134267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 33), 'self')
        # Obtaining the member 'nx' of a type (line 351)
        nx_134268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 33), self_134267, 'nx')
        int_134269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 43), 'int')
        # Applying the binary operator '-' (line 351)
        result_sub_134270 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 33), '-', nx_134268, int_134269)
        
        # Applying the binary operator '<=' (line 351)
        result_le_134271 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 27), '<=', xi_134266, result_sub_134270)
        
        # Applying the binary operator 'and' (line 351)
        result_and_keyword_134272 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), 'and', result_ge_134265, result_le_134271)
        
        # Getting the type of 'yi' (line 351)
        yi_134273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 49), 'yi')
        int_134274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 55), 'int')
        # Applying the binary operator '>=' (line 351)
        result_ge_134275 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 49), '>=', yi_134273, int_134274)
        
        # Applying the binary operator 'and' (line 351)
        result_and_keyword_134276 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), 'and', result_and_keyword_134272, result_ge_134275)
        
        # Getting the type of 'yi' (line 351)
        yi_134277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 61), 'yi')
        # Getting the type of 'self' (line 351)
        self_134278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 67), 'self')
        # Obtaining the member 'ny' of a type (line 351)
        ny_134279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 67), self_134278, 'ny')
        int_134280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 77), 'int')
        # Applying the binary operator '-' (line 351)
        result_sub_134281 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 67), '-', ny_134279, int_134280)
        
        # Applying the binary operator '<=' (line 351)
        result_le_134282 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 61), '<=', yi_134277, result_sub_134281)
        
        # Applying the binary operator 'and' (line 351)
        result_and_keyword_134283 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), 'and', result_and_keyword_134276, result_le_134282)
        
        # Assigning a type to the variable 'stypy_return_type' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'stypy_return_type', result_and_keyword_134283)
        
        # ################# End of 'within_grid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'within_grid' in the type store
        # Getting the type of 'stypy_return_type' (line 347)
        stypy_return_type_134284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'within_grid'
        return stypy_return_type_134284


# Assigning a type to the variable 'Grid' (line 307)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'Grid', Grid)
# Declaration of the 'StreamMask' class

class StreamMask(object, ):
    unicode_134285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, (-1)), 'unicode', u'Mask to keep track of discrete regions crossed by streamlines.\n\n    The resolution of this grid determines the approximate spacing between\n    trajectories. Streamlines are only allowed to pass through zeroed cells:\n    When a streamline enters a cell, that cell is set to 1, and no new\n    streamlines are allowed to enter.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 363, 4, False)
        # Assigning a type to the variable 'self' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StreamMask.__init__', ['density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Call to isscalar(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'density' (line 364)
        density_134288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'density', False)
        # Processing the call keyword arguments (line 364)
        kwargs_134289 = {}
        # Getting the type of 'np' (line 364)
        np_134286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 11), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 364)
        isscalar_134287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 11), np_134286, 'isscalar')
        # Calling isscalar(args, kwargs) (line 364)
        isscalar_call_result_134290 = invoke(stypy.reporting.localization.Localization(__file__, 364, 11), isscalar_134287, *[density_134288], **kwargs_134289)
        
        # Testing the type of an if condition (line 364)
        if_condition_134291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 8), isscalar_call_result_134290)
        # Assigning a type to the variable 'if_condition_134291' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'if_condition_134291', if_condition_134291)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'density' (line 365)
        density_134292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'density')
        int_134293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 26), 'int')
        # Applying the binary operator '<=' (line 365)
        result_le_134294 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 15), '<=', density_134292, int_134293)
        
        # Testing the type of an if condition (line 365)
        if_condition_134295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 12), result_le_134294)
        # Assigning a type to the variable 'if_condition_134295' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'if_condition_134295', if_condition_134295)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 366)
        # Processing the call arguments (line 366)
        unicode_134297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 33), 'unicode', u"If a scalar, 'density' must be positive")
        # Processing the call keyword arguments (line 366)
        kwargs_134298 = {}
        # Getting the type of 'ValueError' (line 366)
        ValueError_134296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 366)
        ValueError_call_result_134299 = invoke(stypy.reporting.localization.Localization(__file__, 366, 22), ValueError_134296, *[unicode_134297], **kwargs_134298)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 366, 16), ValueError_call_result_134299, 'raise parameter', BaseException)
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Attribute (line 367):
        
        # Call to int(...): (line 367)
        # Processing the call arguments (line 367)
        int_134301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 36), 'int')
        # Getting the type of 'density' (line 367)
        density_134302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 41), 'density', False)
        # Applying the binary operator '*' (line 367)
        result_mul_134303 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 36), '*', int_134301, density_134302)
        
        # Processing the call keyword arguments (line 367)
        kwargs_134304 = {}
        # Getting the type of 'int' (line 367)
        int_134300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 32), 'int', False)
        # Calling int(args, kwargs) (line 367)
        int_call_result_134305 = invoke(stypy.reporting.localization.Localization(__file__, 367, 32), int_134300, *[result_mul_134303], **kwargs_134304)
        
        # Getting the type of 'self' (line 367)
        self_134306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'self')
        # Setting the type of the member 'ny' of a type (line 367)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 22), self_134306, 'ny', int_call_result_134305)
        
        # Assigning a Attribute to a Attribute (line 367):
        # Getting the type of 'self' (line 367)
        self_134307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'self')
        # Obtaining the member 'ny' of a type (line 367)
        ny_134308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 22), self_134307, 'ny')
        # Getting the type of 'self' (line 367)
        self_134309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'self')
        # Setting the type of the member 'nx' of a type (line 367)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), self_134309, 'nx', ny_134308)
        # SSA branch for the else part of an if statement (line 364)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'density' (line 369)
        density_134311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'density', False)
        # Processing the call keyword arguments (line 369)
        kwargs_134312 = {}
        # Getting the type of 'len' (line 369)
        len_134310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'len', False)
        # Calling len(args, kwargs) (line 369)
        len_call_result_134313 = invoke(stypy.reporting.localization.Localization(__file__, 369, 15), len_134310, *[density_134311], **kwargs_134312)
        
        int_134314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 31), 'int')
        # Applying the binary operator '!=' (line 369)
        result_ne_134315 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 15), '!=', len_call_result_134313, int_134314)
        
        # Testing the type of an if condition (line 369)
        if_condition_134316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 12), result_ne_134315)
        # Assigning a type to the variable 'if_condition_134316' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'if_condition_134316', if_condition_134316)
        # SSA begins for if statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 370)
        # Processing the call arguments (line 370)
        unicode_134318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 33), 'unicode', u"'density' can have at maximum 2 dimensions")
        # Processing the call keyword arguments (line 370)
        kwargs_134319 = {}
        # Getting the type of 'ValueError' (line 370)
        ValueError_134317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 370)
        ValueError_call_result_134320 = invoke(stypy.reporting.localization.Localization(__file__, 370, 22), ValueError_134317, *[unicode_134318], **kwargs_134319)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 370, 16), ValueError_call_result_134320, 'raise parameter', BaseException)
        # SSA join for if statement (line 369)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 371):
        
        # Assigning a Call to a Attribute (line 371):
        
        # Call to int(...): (line 371)
        # Processing the call arguments (line 371)
        int_134322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 26), 'int')
        
        # Obtaining the type of the subscript
        int_134323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 39), 'int')
        # Getting the type of 'density' (line 371)
        density_134324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 31), 'density', False)
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___134325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 31), density_134324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_134326 = invoke(stypy.reporting.localization.Localization(__file__, 371, 31), getitem___134325, int_134323)
        
        # Applying the binary operator '*' (line 371)
        result_mul_134327 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 26), '*', int_134322, subscript_call_result_134326)
        
        # Processing the call keyword arguments (line 371)
        kwargs_134328 = {}
        # Getting the type of 'int' (line 371)
        int_134321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 22), 'int', False)
        # Calling int(args, kwargs) (line 371)
        int_call_result_134329 = invoke(stypy.reporting.localization.Localization(__file__, 371, 22), int_134321, *[result_mul_134327], **kwargs_134328)
        
        # Getting the type of 'self' (line 371)
        self_134330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'self')
        # Setting the type of the member 'nx' of a type (line 371)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 12), self_134330, 'nx', int_call_result_134329)
        
        # Assigning a Call to a Attribute (line 372):
        
        # Assigning a Call to a Attribute (line 372):
        
        # Call to int(...): (line 372)
        # Processing the call arguments (line 372)
        int_134332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 26), 'int')
        
        # Obtaining the type of the subscript
        int_134333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 39), 'int')
        # Getting the type of 'density' (line 372)
        density_134334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 31), 'density', False)
        # Obtaining the member '__getitem__' of a type (line 372)
        getitem___134335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 31), density_134334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 372)
        subscript_call_result_134336 = invoke(stypy.reporting.localization.Localization(__file__, 372, 31), getitem___134335, int_134333)
        
        # Applying the binary operator '*' (line 372)
        result_mul_134337 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 26), '*', int_134332, subscript_call_result_134336)
        
        # Processing the call keyword arguments (line 372)
        kwargs_134338 = {}
        # Getting the type of 'int' (line 372)
        int_134331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'int', False)
        # Calling int(args, kwargs) (line 372)
        int_call_result_134339 = invoke(stypy.reporting.localization.Localization(__file__, 372, 22), int_134331, *[result_mul_134337], **kwargs_134338)
        
        # Getting the type of 'self' (line 372)
        self_134340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'self')
        # Setting the type of the member 'ny' of a type (line 372)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), self_134340, 'ny', int_call_result_134339)
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 373):
        
        # Assigning a Call to a Attribute (line 373):
        
        # Call to zeros(...): (line 373)
        # Processing the call arguments (line 373)
        
        # Obtaining an instance of the builtin type 'tuple' (line 373)
        tuple_134343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 373)
        # Adding element type (line 373)
        # Getting the type of 'self' (line 373)
        self_134344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 31), 'self', False)
        # Obtaining the member 'ny' of a type (line 373)
        ny_134345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 31), self_134344, 'ny')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 31), tuple_134343, ny_134345)
        # Adding element type (line 373)
        # Getting the type of 'self' (line 373)
        self_134346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 40), 'self', False)
        # Obtaining the member 'nx' of a type (line 373)
        nx_134347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 40), self_134346, 'nx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 31), tuple_134343, nx_134347)
        
        # Processing the call keyword arguments (line 373)
        kwargs_134348 = {}
        # Getting the type of 'np' (line 373)
        np_134341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 21), 'np', False)
        # Obtaining the member 'zeros' of a type (line 373)
        zeros_134342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 21), np_134341, 'zeros')
        # Calling zeros(args, kwargs) (line 373)
        zeros_call_result_134349 = invoke(stypy.reporting.localization.Localization(__file__, 373, 21), zeros_134342, *[tuple_134343], **kwargs_134348)
        
        # Getting the type of 'self' (line 373)
        self_134350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'self')
        # Setting the type of the member '_mask' of a type (line 373)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), self_134350, '_mask', zeros_call_result_134349)
        
        # Assigning a Attribute to a Attribute (line 374):
        
        # Assigning a Attribute to a Attribute (line 374):
        # Getting the type of 'self' (line 374)
        self_134351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 'self')
        # Obtaining the member '_mask' of a type (line 374)
        _mask_134352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 21), self_134351, '_mask')
        # Obtaining the member 'shape' of a type (line 374)
        shape_134353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 21), _mask_134352, 'shape')
        # Getting the type of 'self' (line 374)
        self_134354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'self')
        # Setting the type of the member 'shape' of a type (line 374)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), self_134354, 'shape', shape_134353)
        
        # Assigning a Name to a Attribute (line 376):
        
        # Assigning a Name to a Attribute (line 376):
        # Getting the type of 'None' (line 376)
        None_134355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 27), 'None')
        # Getting the type of 'self' (line 376)
        self_134356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'self')
        # Setting the type of the member '_current_xy' of a type (line 376)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), self_134356, '_current_xy', None_134355)
        
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
        module_type_store = module_type_store.open_function_context('__getitem__', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StreamMask.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        StreamMask.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StreamMask.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        StreamMask.__getitem__.__dict__.__setitem__('stypy_function_name', 'StreamMask.__getitem__')
        StreamMask.__getitem__.__dict__.__setitem__('stypy_param_names_list', [])
        StreamMask.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        StreamMask.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StreamMask.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        StreamMask.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        StreamMask.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StreamMask.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StreamMask.__getitem__', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Call to __getitem__(...): (line 379)
        # Getting the type of 'args' (line 379)
        args_134360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 39), 'args', False)
        # Processing the call keyword arguments (line 379)
        kwargs_134361 = {}
        # Getting the type of 'self' (line 379)
        self_134357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'self', False)
        # Obtaining the member '_mask' of a type (line 379)
        _mask_134358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), self_134357, '_mask')
        # Obtaining the member '__getitem__' of a type (line 379)
        getitem___134359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), _mask_134358, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 379)
        getitem___call_result_134362 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), getitem___134359, *[args_134360], **kwargs_134361)
        
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', getitem___call_result_134362)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_134363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134363)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_134363


    @norecursion
    def _start_trajectory(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_start_trajectory'
        module_type_store = module_type_store.open_function_context('_start_trajectory', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_localization', localization)
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_type_store', module_type_store)
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_function_name', 'StreamMask._start_trajectory')
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_param_names_list', ['xm', 'ym'])
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_varargs_param_name', None)
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_call_defaults', defaults)
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_call_varargs', varargs)
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StreamMask._start_trajectory.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StreamMask._start_trajectory', ['xm', 'ym'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_start_trajectory', localization, ['xm', 'ym'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_start_trajectory(...)' code ##################

        unicode_134364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 8), 'unicode', u'Start recording streamline trajectory')
        
        # Assigning a List to a Attribute (line 383):
        
        # Assigning a List to a Attribute (line 383):
        
        # Obtaining an instance of the builtin type 'list' (line 383)
        list_134365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 383)
        
        # Getting the type of 'self' (line 383)
        self_134366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member '_traj' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_134366, '_traj', list_134365)
        
        # Call to _update_trajectory(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'xm' (line 384)
        xm_134369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 32), 'xm', False)
        # Getting the type of 'ym' (line 384)
        ym_134370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 36), 'ym', False)
        # Processing the call keyword arguments (line 384)
        kwargs_134371 = {}
        # Getting the type of 'self' (line 384)
        self_134367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self', False)
        # Obtaining the member '_update_trajectory' of a type (line 384)
        _update_trajectory_134368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_134367, '_update_trajectory')
        # Calling _update_trajectory(args, kwargs) (line 384)
        _update_trajectory_call_result_134372 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), _update_trajectory_134368, *[xm_134369, ym_134370], **kwargs_134371)
        
        
        # ################# End of '_start_trajectory(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_start_trajectory' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_134373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134373)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_start_trajectory'
        return stypy_return_type_134373


    @norecursion
    def _undo_trajectory(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_undo_trajectory'
        module_type_store = module_type_store.open_function_context('_undo_trajectory', 386, 4, False)
        # Assigning a type to the variable 'self' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_localization', localization)
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_type_store', module_type_store)
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_function_name', 'StreamMask._undo_trajectory')
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_param_names_list', [])
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_varargs_param_name', None)
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_call_defaults', defaults)
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_call_varargs', varargs)
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StreamMask._undo_trajectory.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StreamMask._undo_trajectory', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_undo_trajectory', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_undo_trajectory(...)' code ##################

        unicode_134374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 8), 'unicode', u'Remove current trajectory from mask')
        
        # Getting the type of 'self' (line 388)
        self_134375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 17), 'self')
        # Obtaining the member '_traj' of a type (line 388)
        _traj_134376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 17), self_134375, '_traj')
        # Testing the type of a for loop iterable (line 388)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 388, 8), _traj_134376)
        # Getting the type of the for loop variable (line 388)
        for_loop_var_134377 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 388, 8), _traj_134376)
        # Assigning a type to the variable 't' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 't', for_loop_var_134377)
        # SSA begins for a for statement (line 388)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to __setitem__(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 't' (line 389)
        t_134381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 35), 't', False)
        int_134382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 38), 'int')
        # Processing the call keyword arguments (line 389)
        kwargs_134383 = {}
        # Getting the type of 'self' (line 389)
        self_134378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'self', False)
        # Obtaining the member '_mask' of a type (line 389)
        _mask_134379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), self_134378, '_mask')
        # Obtaining the member '__setitem__' of a type (line 389)
        setitem___134380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), _mask_134379, '__setitem__')
        # Calling __setitem__(args, kwargs) (line 389)
        setitem___call_result_134384 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), setitem___134380, *[t_134381, int_134382], **kwargs_134383)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_undo_trajectory(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_undo_trajectory' in the type store
        # Getting the type of 'stypy_return_type' (line 386)
        stypy_return_type_134385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_undo_trajectory'
        return stypy_return_type_134385


    @norecursion
    def _update_trajectory(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_trajectory'
        module_type_store = module_type_store.open_function_context('_update_trajectory', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_localization', localization)
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_type_store', module_type_store)
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_function_name', 'StreamMask._update_trajectory')
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_param_names_list', ['xm', 'ym'])
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_varargs_param_name', None)
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_call_defaults', defaults)
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_call_varargs', varargs)
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StreamMask._update_trajectory.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StreamMask._update_trajectory', ['xm', 'ym'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_trajectory', localization, ['xm', 'ym'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_trajectory(...)' code ##################

        unicode_134386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, (-1)), 'unicode', u'Update current trajectory position in mask.\n\n        If the new position has already been filled, raise `InvalidIndexError`.\n        ')
        
        
        # Getting the type of 'self' (line 396)
        self_134387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 11), 'self')
        # Obtaining the member '_current_xy' of a type (line 396)
        _current_xy_134388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 11), self_134387, '_current_xy')
        
        # Obtaining an instance of the builtin type 'tuple' (line 396)
        tuple_134389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 396)
        # Adding element type (line 396)
        # Getting the type of 'xm' (line 396)
        xm_134390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 32), 'xm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 32), tuple_134389, xm_134390)
        # Adding element type (line 396)
        # Getting the type of 'ym' (line 396)
        ym_134391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 36), 'ym')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 32), tuple_134389, ym_134391)
        
        # Applying the binary operator '!=' (line 396)
        result_ne_134392 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 11), '!=', _current_xy_134388, tuple_134389)
        
        # Testing the type of an if condition (line 396)
        if_condition_134393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 8), result_ne_134392)
        # Assigning a type to the variable 'if_condition_134393' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'if_condition_134393', if_condition_134393)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 397)
        tuple_134394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 397)
        # Adding element type (line 397)
        # Getting the type of 'ym' (line 397)
        ym_134395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 20), 'ym')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 20), tuple_134394, ym_134395)
        # Adding element type (line 397)
        # Getting the type of 'xm' (line 397)
        xm_134396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'xm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 20), tuple_134394, xm_134396)
        
        # Getting the type of 'self' (line 397)
        self_134397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'self')
        # Obtaining the member '__getitem__' of a type (line 397)
        getitem___134398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 15), self_134397, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 397)
        subscript_call_result_134399 = invoke(stypy.reporting.localization.Localization(__file__, 397, 15), getitem___134398, tuple_134394)
        
        int_134400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 31), 'int')
        # Applying the binary operator '==' (line 397)
        result_eq_134401 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 15), '==', subscript_call_result_134399, int_134400)
        
        # Testing the type of an if condition (line 397)
        if_condition_134402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 12), result_eq_134401)
        # Assigning a type to the variable 'if_condition_134402' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'if_condition_134402', if_condition_134402)
        # SSA begins for if statement (line 397)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Obtaining an instance of the builtin type 'tuple' (line 398)
        tuple_134406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 398)
        # Adding element type (line 398)
        # Getting the type of 'ym' (line 398)
        ym_134407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 35), 'ym', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 35), tuple_134406, ym_134407)
        # Adding element type (line 398)
        # Getting the type of 'xm' (line 398)
        xm_134408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 39), 'xm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 35), tuple_134406, xm_134408)
        
        # Processing the call keyword arguments (line 398)
        kwargs_134409 = {}
        # Getting the type of 'self' (line 398)
        self_134403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'self', False)
        # Obtaining the member '_traj' of a type (line 398)
        _traj_134404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 16), self_134403, '_traj')
        # Obtaining the member 'append' of a type (line 398)
        append_134405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 16), _traj_134404, 'append')
        # Calling append(args, kwargs) (line 398)
        append_call_result_134410 = invoke(stypy.reporting.localization.Localization(__file__, 398, 16), append_134405, *[tuple_134406], **kwargs_134409)
        
        
        # Assigning a Num to a Subscript (line 399):
        
        # Assigning a Num to a Subscript (line 399):
        int_134411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 37), 'int')
        # Getting the type of 'self' (line 399)
        self_134412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'self')
        # Obtaining the member '_mask' of a type (line 399)
        _mask_134413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 16), self_134412, '_mask')
        
        # Obtaining an instance of the builtin type 'tuple' (line 399)
        tuple_134414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 399)
        # Adding element type (line 399)
        # Getting the type of 'ym' (line 399)
        ym_134415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 27), 'ym')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 27), tuple_134414, ym_134415)
        # Adding element type (line 399)
        # Getting the type of 'xm' (line 399)
        xm_134416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'xm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 27), tuple_134414, xm_134416)
        
        # Storing an element on a container (line 399)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 16), _mask_134413, (tuple_134414, int_134411))
        
        # Assigning a Tuple to a Attribute (line 400):
        
        # Assigning a Tuple to a Attribute (line 400):
        
        # Obtaining an instance of the builtin type 'tuple' (line 400)
        tuple_134417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 400)
        # Adding element type (line 400)
        # Getting the type of 'xm' (line 400)
        xm_134418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 36), 'xm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 36), tuple_134417, xm_134418)
        # Adding element type (line 400)
        # Getting the type of 'ym' (line 400)
        ym_134419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 40), 'ym')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 36), tuple_134417, ym_134419)
        
        # Getting the type of 'self' (line 400)
        self_134420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'self')
        # Setting the type of the member '_current_xy' of a type (line 400)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), self_134420, '_current_xy', tuple_134417)
        # SSA branch for the else part of an if statement (line 397)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'InvalidIndexError' (line 402)
        InvalidIndexError_134421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 22), 'InvalidIndexError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 402, 16), InvalidIndexError_134421, 'raise parameter', BaseException)
        # SSA join for if statement (line 397)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_update_trajectory(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_trajectory' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_134422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_trajectory'
        return stypy_return_type_134422


# Assigning a type to the variable 'StreamMask' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'StreamMask', StreamMask)
# Declaration of the 'InvalidIndexError' class
# Getting the type of 'Exception' (line 405)
Exception_134423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 24), 'Exception')

class InvalidIndexError(Exception_134423, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 405, 0, False)
        # Assigning a type to the variable 'self' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvalidIndexError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InvalidIndexError' (line 405)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 0), 'InvalidIndexError', InvalidIndexError)
# Declaration of the 'TerminateTrajectory' class
# Getting the type of 'Exception' (line 409)
Exception_134424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 26), 'Exception')

class TerminateTrajectory(Exception_134424, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 409, 0, False)
        # Assigning a type to the variable 'self' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TerminateTrajectory.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TerminateTrajectory' (line 409)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'TerminateTrajectory', TerminateTrajectory)

@norecursion
def get_integrator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_integrator'
    module_type_store = module_type_store.open_function_context('get_integrator', 416, 0, False)
    
    # Passed parameters checking function
    get_integrator.stypy_localization = localization
    get_integrator.stypy_type_of_self = None
    get_integrator.stypy_type_store = module_type_store
    get_integrator.stypy_function_name = 'get_integrator'
    get_integrator.stypy_param_names_list = ['u', 'v', 'dmap', 'minlength', 'maxlength', 'integration_direction']
    get_integrator.stypy_varargs_param_name = None
    get_integrator.stypy_kwargs_param_name = None
    get_integrator.stypy_call_defaults = defaults
    get_integrator.stypy_call_varargs = varargs
    get_integrator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_integrator', ['u', 'v', 'dmap', 'minlength', 'maxlength', 'integration_direction'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_integrator', localization, ['u', 'v', 'dmap', 'minlength', 'maxlength', 'integration_direction'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_integrator(...)' code ##################

    
    # Assigning a Call to a Tuple (line 419):
    
    # Assigning a Call to a Name:
    
    # Call to data2grid(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'u' (line 419)
    u_134427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 26), 'u', False)
    # Getting the type of 'v' (line 419)
    v_134428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 29), 'v', False)
    # Processing the call keyword arguments (line 419)
    kwargs_134429 = {}
    # Getting the type of 'dmap' (line 419)
    dmap_134425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'dmap', False)
    # Obtaining the member 'data2grid' of a type (line 419)
    data2grid_134426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 11), dmap_134425, 'data2grid')
    # Calling data2grid(args, kwargs) (line 419)
    data2grid_call_result_134430 = invoke(stypy.reporting.localization.Localization(__file__, 419, 11), data2grid_134426, *[u_134427, v_134428], **kwargs_134429)
    
    # Assigning a type to the variable 'call_assignment_133185' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'call_assignment_133185', data2grid_call_result_134430)
    
    # Assigning a Call to a Name (line 419):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 4), 'int')
    # Processing the call keyword arguments
    kwargs_134434 = {}
    # Getting the type of 'call_assignment_133185' (line 419)
    call_assignment_133185_134431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'call_assignment_133185', False)
    # Obtaining the member '__getitem__' of a type (line 419)
    getitem___134432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 4), call_assignment_133185_134431, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134435 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134432, *[int_134433], **kwargs_134434)
    
    # Assigning a type to the variable 'call_assignment_133186' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'call_assignment_133186', getitem___call_result_134435)
    
    # Assigning a Name to a Name (line 419):
    # Getting the type of 'call_assignment_133186' (line 419)
    call_assignment_133186_134436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'call_assignment_133186')
    # Assigning a type to the variable 'u' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'u', call_assignment_133186_134436)
    
    # Assigning a Call to a Name (line 419):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 4), 'int')
    # Processing the call keyword arguments
    kwargs_134440 = {}
    # Getting the type of 'call_assignment_133185' (line 419)
    call_assignment_133185_134437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'call_assignment_133185', False)
    # Obtaining the member '__getitem__' of a type (line 419)
    getitem___134438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 4), call_assignment_133185_134437, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134441 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134438, *[int_134439], **kwargs_134440)
    
    # Assigning a type to the variable 'call_assignment_133187' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'call_assignment_133187', getitem___call_result_134441)
    
    # Assigning a Name to a Name (line 419):
    # Getting the type of 'call_assignment_133187' (line 419)
    call_assignment_133187_134442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'call_assignment_133187')
    # Assigning a type to the variable 'v' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 7), 'v', call_assignment_133187_134442)
    
    # Assigning a BinOp to a Name (line 422):
    
    # Assigning a BinOp to a Name (line 422):
    # Getting the type of 'u' (line 422)
    u_134443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), 'u')
    # Getting the type of 'dmap' (line 422)
    dmap_134444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'dmap')
    # Obtaining the member 'grid' of a type (line 422)
    grid_134445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), dmap_134444, 'grid')
    # Obtaining the member 'nx' of a type (line 422)
    nx_134446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), grid_134445, 'nx')
    # Applying the binary operator 'div' (line 422)
    result_div_134447 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 11), 'div', u_134443, nx_134446)
    
    # Assigning a type to the variable 'u_ax' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'u_ax', result_div_134447)
    
    # Assigning a BinOp to a Name (line 423):
    
    # Assigning a BinOp to a Name (line 423):
    # Getting the type of 'v' (line 423)
    v_134448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'v')
    # Getting the type of 'dmap' (line 423)
    dmap_134449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'dmap')
    # Obtaining the member 'grid' of a type (line 423)
    grid_134450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 15), dmap_134449, 'grid')
    # Obtaining the member 'ny' of a type (line 423)
    ny_134451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 15), grid_134450, 'ny')
    # Applying the binary operator 'div' (line 423)
    result_div_134452 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 11), 'div', v_134448, ny_134451)
    
    # Assigning a type to the variable 'v_ax' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'v_ax', result_div_134452)
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to sqrt(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'u_ax' (line 424)
    u_ax_134456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 23), 'u_ax', False)
    int_134457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 31), 'int')
    # Applying the binary operator '**' (line 424)
    result_pow_134458 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 23), '**', u_ax_134456, int_134457)
    
    # Getting the type of 'v_ax' (line 424)
    v_ax_134459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 35), 'v_ax', False)
    int_134460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 43), 'int')
    # Applying the binary operator '**' (line 424)
    result_pow_134461 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 35), '**', v_ax_134459, int_134460)
    
    # Applying the binary operator '+' (line 424)
    result_add_134462 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 23), '+', result_pow_134458, result_pow_134461)
    
    # Processing the call keyword arguments (line 424)
    kwargs_134463 = {}
    # Getting the type of 'np' (line 424)
    np_134453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'np', False)
    # Obtaining the member 'ma' of a type (line 424)
    ma_134454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), np_134453, 'ma')
    # Obtaining the member 'sqrt' of a type (line 424)
    sqrt_134455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), ma_134454, 'sqrt')
    # Calling sqrt(args, kwargs) (line 424)
    sqrt_call_result_134464 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), sqrt_134455, *[result_add_134462], **kwargs_134463)
    
    # Assigning a type to the variable 'speed' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'speed', sqrt_call_result_134464)

    @norecursion
    def forward_time(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'forward_time'
        module_type_store = module_type_store.open_function_context('forward_time', 426, 4, False)
        
        # Passed parameters checking function
        forward_time.stypy_localization = localization
        forward_time.stypy_type_of_self = None
        forward_time.stypy_type_store = module_type_store
        forward_time.stypy_function_name = 'forward_time'
        forward_time.stypy_param_names_list = ['xi', 'yi']
        forward_time.stypy_varargs_param_name = None
        forward_time.stypy_kwargs_param_name = None
        forward_time.stypy_call_defaults = defaults
        forward_time.stypy_call_varargs = varargs
        forward_time.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'forward_time', ['xi', 'yi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'forward_time', localization, ['xi', 'yi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'forward_time(...)' code ##################

        
        # Assigning a Call to a Name (line 427):
        
        # Assigning a Call to a Name (line 427):
        
        # Call to interpgrid(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'speed' (line 427)
        speed_134466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 27), 'speed', False)
        # Getting the type of 'xi' (line 427)
        xi_134467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 34), 'xi', False)
        # Getting the type of 'yi' (line 427)
        yi_134468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 38), 'yi', False)
        # Processing the call keyword arguments (line 427)
        kwargs_134469 = {}
        # Getting the type of 'interpgrid' (line 427)
        interpgrid_134465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'interpgrid', False)
        # Calling interpgrid(args, kwargs) (line 427)
        interpgrid_call_result_134470 = invoke(stypy.reporting.localization.Localization(__file__, 427, 16), interpgrid_134465, *[speed_134466, xi_134467, yi_134468], **kwargs_134469)
        
        # Assigning a type to the variable 'ds_dt' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'ds_dt', interpgrid_call_result_134470)
        
        
        # Getting the type of 'ds_dt' (line 428)
        ds_dt_134471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'ds_dt')
        int_134472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 20), 'int')
        # Applying the binary operator '==' (line 428)
        result_eq_134473 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), '==', ds_dt_134471, int_134472)
        
        # Testing the type of an if condition (line 428)
        if_condition_134474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_eq_134473)
        # Assigning a type to the variable 'if_condition_134474' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_134474', if_condition_134474)
        # SSA begins for if statement (line 428)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TerminateTrajectory(...): (line 429)
        # Processing the call keyword arguments (line 429)
        kwargs_134476 = {}
        # Getting the type of 'TerminateTrajectory' (line 429)
        TerminateTrajectory_134475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 18), 'TerminateTrajectory', False)
        # Calling TerminateTrajectory(args, kwargs) (line 429)
        TerminateTrajectory_call_result_134477 = invoke(stypy.reporting.localization.Localization(__file__, 429, 18), TerminateTrajectory_134475, *[], **kwargs_134476)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 429, 12), TerminateTrajectory_call_result_134477, 'raise parameter', BaseException)
        # SSA join for if statement (line 428)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 430):
        
        # Assigning a BinOp to a Name (line 430):
        float_134478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 16), 'float')
        # Getting the type of 'ds_dt' (line 430)
        ds_dt_134479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 21), 'ds_dt')
        # Applying the binary operator 'div' (line 430)
        result_div_134480 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 16), 'div', float_134478, ds_dt_134479)
        
        # Assigning a type to the variable 'dt_ds' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'dt_ds', result_div_134480)
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to interpgrid(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'u' (line 431)
        u_134482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 24), 'u', False)
        # Getting the type of 'xi' (line 431)
        xi_134483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 27), 'xi', False)
        # Getting the type of 'yi' (line 431)
        yi_134484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 31), 'yi', False)
        # Processing the call keyword arguments (line 431)
        kwargs_134485 = {}
        # Getting the type of 'interpgrid' (line 431)
        interpgrid_134481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 13), 'interpgrid', False)
        # Calling interpgrid(args, kwargs) (line 431)
        interpgrid_call_result_134486 = invoke(stypy.reporting.localization.Localization(__file__, 431, 13), interpgrid_134481, *[u_134482, xi_134483, yi_134484], **kwargs_134485)
        
        # Assigning a type to the variable 'ui' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'ui', interpgrid_call_result_134486)
        
        # Assigning a Call to a Name (line 432):
        
        # Assigning a Call to a Name (line 432):
        
        # Call to interpgrid(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'v' (line 432)
        v_134488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'v', False)
        # Getting the type of 'xi' (line 432)
        xi_134489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 27), 'xi', False)
        # Getting the type of 'yi' (line 432)
        yi_134490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 31), 'yi', False)
        # Processing the call keyword arguments (line 432)
        kwargs_134491 = {}
        # Getting the type of 'interpgrid' (line 432)
        interpgrid_134487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 13), 'interpgrid', False)
        # Calling interpgrid(args, kwargs) (line 432)
        interpgrid_call_result_134492 = invoke(stypy.reporting.localization.Localization(__file__, 432, 13), interpgrid_134487, *[v_134488, xi_134489, yi_134490], **kwargs_134491)
        
        # Assigning a type to the variable 'vi' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'vi', interpgrid_call_result_134492)
        
        # Obtaining an instance of the builtin type 'tuple' (line 433)
        tuple_134493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 433)
        # Adding element type (line 433)
        # Getting the type of 'ui' (line 433)
        ui_134494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'ui')
        # Getting the type of 'dt_ds' (line 433)
        dt_ds_134495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'dt_ds')
        # Applying the binary operator '*' (line 433)
        result_mul_134496 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 15), '*', ui_134494, dt_ds_134495)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 15), tuple_134493, result_mul_134496)
        # Adding element type (line 433)
        # Getting the type of 'vi' (line 433)
        vi_134497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'vi')
        # Getting the type of 'dt_ds' (line 433)
        dt_ds_134498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 32), 'dt_ds')
        # Applying the binary operator '*' (line 433)
        result_mul_134499 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 27), '*', vi_134497, dt_ds_134498)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 15), tuple_134493, result_mul_134499)
        
        # Assigning a type to the variable 'stypy_return_type' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'stypy_return_type', tuple_134493)
        
        # ################# End of 'forward_time(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'forward_time' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_134500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'forward_time'
        return stypy_return_type_134500

    # Assigning a type to the variable 'forward_time' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'forward_time', forward_time)

    @norecursion
    def backward_time(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'backward_time'
        module_type_store = module_type_store.open_function_context('backward_time', 435, 4, False)
        
        # Passed parameters checking function
        backward_time.stypy_localization = localization
        backward_time.stypy_type_of_self = None
        backward_time.stypy_type_store = module_type_store
        backward_time.stypy_function_name = 'backward_time'
        backward_time.stypy_param_names_list = ['xi', 'yi']
        backward_time.stypy_varargs_param_name = None
        backward_time.stypy_kwargs_param_name = None
        backward_time.stypy_call_defaults = defaults
        backward_time.stypy_call_varargs = varargs
        backward_time.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'backward_time', ['xi', 'yi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'backward_time', localization, ['xi', 'yi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'backward_time(...)' code ##################

        
        # Assigning a Call to a Tuple (line 436):
        
        # Assigning a Call to a Name:
        
        # Call to forward_time(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'xi' (line 436)
        xi_134502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 32), 'xi', False)
        # Getting the type of 'yi' (line 436)
        yi_134503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 36), 'yi', False)
        # Processing the call keyword arguments (line 436)
        kwargs_134504 = {}
        # Getting the type of 'forward_time' (line 436)
        forward_time_134501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 19), 'forward_time', False)
        # Calling forward_time(args, kwargs) (line 436)
        forward_time_call_result_134505 = invoke(stypy.reporting.localization.Localization(__file__, 436, 19), forward_time_134501, *[xi_134502, yi_134503], **kwargs_134504)
        
        # Assigning a type to the variable 'call_assignment_133188' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'call_assignment_133188', forward_time_call_result_134505)
        
        # Assigning a Call to a Name (line 436):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 8), 'int')
        # Processing the call keyword arguments
        kwargs_134509 = {}
        # Getting the type of 'call_assignment_133188' (line 436)
        call_assignment_133188_134506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'call_assignment_133188', False)
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___134507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), call_assignment_133188_134506, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134510 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134507, *[int_134508], **kwargs_134509)
        
        # Assigning a type to the variable 'call_assignment_133189' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'call_assignment_133189', getitem___call_result_134510)
        
        # Assigning a Name to a Name (line 436):
        # Getting the type of 'call_assignment_133189' (line 436)
        call_assignment_133189_134511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'call_assignment_133189')
        # Assigning a type to the variable 'dxi' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'dxi', call_assignment_133189_134511)
        
        # Assigning a Call to a Name (line 436):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 8), 'int')
        # Processing the call keyword arguments
        kwargs_134515 = {}
        # Getting the type of 'call_assignment_133188' (line 436)
        call_assignment_133188_134512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'call_assignment_133188', False)
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___134513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), call_assignment_133188_134512, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134516 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134513, *[int_134514], **kwargs_134515)
        
        # Assigning a type to the variable 'call_assignment_133190' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'call_assignment_133190', getitem___call_result_134516)
        
        # Assigning a Name to a Name (line 436):
        # Getting the type of 'call_assignment_133190' (line 436)
        call_assignment_133190_134517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'call_assignment_133190')
        # Assigning a type to the variable 'dyi' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 13), 'dyi', call_assignment_133190_134517)
        
        # Obtaining an instance of the builtin type 'tuple' (line 437)
        tuple_134518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 437)
        # Adding element type (line 437)
        
        # Getting the type of 'dxi' (line 437)
        dxi_134519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'dxi')
        # Applying the 'usub' unary operator (line 437)
        result___neg___134520 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 15), 'usub', dxi_134519)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 15), tuple_134518, result___neg___134520)
        # Adding element type (line 437)
        
        # Getting the type of 'dyi' (line 437)
        dyi_134521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 22), 'dyi')
        # Applying the 'usub' unary operator (line 437)
        result___neg___134522 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 21), 'usub', dyi_134521)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 15), tuple_134518, result___neg___134522)
        
        # Assigning a type to the variable 'stypy_return_type' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', tuple_134518)
        
        # ################# End of 'backward_time(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'backward_time' in the type store
        # Getting the type of 'stypy_return_type' (line 435)
        stypy_return_type_134523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134523)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'backward_time'
        return stypy_return_type_134523

    # Assigning a type to the variable 'backward_time' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'backward_time', backward_time)

    @norecursion
    def integrate(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'integrate'
        module_type_store = module_type_store.open_function_context('integrate', 439, 4, False)
        
        # Passed parameters checking function
        integrate.stypy_localization = localization
        integrate.stypy_type_of_self = None
        integrate.stypy_type_store = module_type_store
        integrate.stypy_function_name = 'integrate'
        integrate.stypy_param_names_list = ['x0', 'y0']
        integrate.stypy_varargs_param_name = None
        integrate.stypy_kwargs_param_name = None
        integrate.stypy_call_defaults = defaults
        integrate.stypy_call_varargs = varargs
        integrate.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'integrate', ['x0', 'y0'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate', localization, ['x0', 'y0'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate(...)' code ##################

        unicode_134524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, (-1)), 'unicode', u'Return x, y grid-coordinates of trajectory based on starting point.\n\n        Integrate both forward and backward in time from starting point in\n        grid coordinates.\n\n        Integration is terminated when a trajectory reaches a domain boundary\n        or when it crosses into an already occupied cell in the StreamMask. The\n        resulting trajectory is None if it is shorter than `minlength`.\n        ')
        
        # Assigning a Tuple to a Tuple (line 450):
        
        # Assigning a Num to a Name (line 450):
        float_134525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 33), 'float')
        # Assigning a type to the variable 'tuple_assignment_133191' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_assignment_133191', float_134525)
        
        # Assigning a List to a Name (line 450):
        
        # Obtaining an instance of the builtin type 'list' (line 450)
        list_134526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 450)
        
        # Assigning a type to the variable 'tuple_assignment_133192' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_assignment_133192', list_134526)
        
        # Assigning a List to a Name (line 450):
        
        # Obtaining an instance of the builtin type 'list' (line 450)
        list_134527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 450)
        
        # Assigning a type to the variable 'tuple_assignment_133193' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_assignment_133193', list_134527)
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'tuple_assignment_133191' (line 450)
        tuple_assignment_133191_134528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_assignment_133191')
        # Assigning a type to the variable 'stotal' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'stotal', tuple_assignment_133191_134528)
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'tuple_assignment_133192' (line 450)
        tuple_assignment_133192_134529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_assignment_133192')
        # Assigning a type to the variable 'x_traj' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'x_traj', tuple_assignment_133192_134529)
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'tuple_assignment_133193' (line 450)
        tuple_assignment_133193_134530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_assignment_133193')
        # Assigning a type to the variable 'y_traj' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'y_traj', tuple_assignment_133193_134530)
        
        
        # SSA begins for try-except statement (line 452)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to start_trajectory(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'x0' (line 453)
        x0_134533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'x0', False)
        # Getting the type of 'y0' (line 453)
        y0_134534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 38), 'y0', False)
        # Processing the call keyword arguments (line 453)
        kwargs_134535 = {}
        # Getting the type of 'dmap' (line 453)
        dmap_134531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'dmap', False)
        # Obtaining the member 'start_trajectory' of a type (line 453)
        start_trajectory_134532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 12), dmap_134531, 'start_trajectory')
        # Calling start_trajectory(args, kwargs) (line 453)
        start_trajectory_call_result_134536 = invoke(stypy.reporting.localization.Localization(__file__, 453, 12), start_trajectory_134532, *[x0_134533, y0_134534], **kwargs_134535)
        
        # SSA branch for the except part of a try statement (line 452)
        # SSA branch for the except 'InvalidIndexError' branch of a try statement (line 452)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 455)
        None_134537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'stypy_return_type', None_134537)
        # SSA join for try-except statement (line 452)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'integration_direction' (line 456)
        integration_direction_134538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'integration_direction')
        
        # Obtaining an instance of the builtin type 'list' (line 456)
        list_134539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 456)
        # Adding element type (line 456)
        unicode_134540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 37), 'unicode', u'both')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 36), list_134539, unicode_134540)
        # Adding element type (line 456)
        unicode_134541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 45), 'unicode', u'backward')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 36), list_134539, unicode_134541)
        
        # Applying the binary operator 'in' (line 456)
        result_contains_134542 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 11), 'in', integration_direction_134538, list_134539)
        
        # Testing the type of an if condition (line 456)
        if_condition_134543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 8), result_contains_134542)
        # Assigning a type to the variable 'if_condition_134543' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'if_condition_134543', if_condition_134543)
        # SSA begins for if statement (line 456)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 457):
        
        # Assigning a Call to a Name:
        
        # Call to _integrate_rk12(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'x0' (line 457)
        x0_134545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 40), 'x0', False)
        # Getting the type of 'y0' (line 457)
        y0_134546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 44), 'y0', False)
        # Getting the type of 'dmap' (line 457)
        dmap_134547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 48), 'dmap', False)
        # Getting the type of 'backward_time' (line 457)
        backward_time_134548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 54), 'backward_time', False)
        # Getting the type of 'maxlength' (line 457)
        maxlength_134549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 69), 'maxlength', False)
        # Processing the call keyword arguments (line 457)
        kwargs_134550 = {}
        # Getting the type of '_integrate_rk12' (line 457)
        _integrate_rk12_134544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 24), '_integrate_rk12', False)
        # Calling _integrate_rk12(args, kwargs) (line 457)
        _integrate_rk12_call_result_134551 = invoke(stypy.reporting.localization.Localization(__file__, 457, 24), _integrate_rk12_134544, *[x0_134545, y0_134546, dmap_134547, backward_time_134548, maxlength_134549], **kwargs_134550)
        
        # Assigning a type to the variable 'call_assignment_133194' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133194', _integrate_rk12_call_result_134551)
        
        # Assigning a Call to a Name (line 457):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 12), 'int')
        # Processing the call keyword arguments
        kwargs_134555 = {}
        # Getting the type of 'call_assignment_133194' (line 457)
        call_assignment_133194_134552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133194', False)
        # Obtaining the member '__getitem__' of a type (line 457)
        getitem___134553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), call_assignment_133194_134552, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134556 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134553, *[int_134554], **kwargs_134555)
        
        # Assigning a type to the variable 'call_assignment_133195' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133195', getitem___call_result_134556)
        
        # Assigning a Name to a Name (line 457):
        # Getting the type of 'call_assignment_133195' (line 457)
        call_assignment_133195_134557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133195')
        # Assigning a type to the variable 's' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 's', call_assignment_133195_134557)
        
        # Assigning a Call to a Name (line 457):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 12), 'int')
        # Processing the call keyword arguments
        kwargs_134561 = {}
        # Getting the type of 'call_assignment_133194' (line 457)
        call_assignment_133194_134558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133194', False)
        # Obtaining the member '__getitem__' of a type (line 457)
        getitem___134559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), call_assignment_133194_134558, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134562 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134559, *[int_134560], **kwargs_134561)
        
        # Assigning a type to the variable 'call_assignment_133196' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133196', getitem___call_result_134562)
        
        # Assigning a Name to a Name (line 457):
        # Getting the type of 'call_assignment_133196' (line 457)
        call_assignment_133196_134563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133196')
        # Assigning a type to the variable 'xt' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'xt', call_assignment_133196_134563)
        
        # Assigning a Call to a Name (line 457):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 12), 'int')
        # Processing the call keyword arguments
        kwargs_134567 = {}
        # Getting the type of 'call_assignment_133194' (line 457)
        call_assignment_133194_134564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133194', False)
        # Obtaining the member '__getitem__' of a type (line 457)
        getitem___134565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), call_assignment_133194_134564, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134568 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134565, *[int_134566], **kwargs_134567)
        
        # Assigning a type to the variable 'call_assignment_133197' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133197', getitem___call_result_134568)
        
        # Assigning a Name to a Name (line 457):
        # Getting the type of 'call_assignment_133197' (line 457)
        call_assignment_133197_134569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'call_assignment_133197')
        # Assigning a type to the variable 'yt' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'yt', call_assignment_133197_134569)
        
        # Getting the type of 'stotal' (line 458)
        stotal_134570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'stotal')
        # Getting the type of 's' (line 458)
        s_134571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 22), 's')
        # Applying the binary operator '+=' (line 458)
        result_iadd_134572 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 12), '+=', stotal_134570, s_134571)
        # Assigning a type to the variable 'stotal' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'stotal', result_iadd_134572)
        
        
        # Getting the type of 'x_traj' (line 459)
        x_traj_134573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'x_traj')
        
        # Obtaining the type of the subscript
        int_134574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 27), 'int')
        slice_134575 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 459, 22), None, None, int_134574)
        # Getting the type of 'xt' (line 459)
        xt_134576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 22), 'xt')
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___134577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 22), xt_134576, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_134578 = invoke(stypy.reporting.localization.Localization(__file__, 459, 22), getitem___134577, slice_134575)
        
        # Applying the binary operator '+=' (line 459)
        result_iadd_134579 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 12), '+=', x_traj_134573, subscript_call_result_134578)
        # Assigning a type to the variable 'x_traj' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'x_traj', result_iadd_134579)
        
        
        # Getting the type of 'y_traj' (line 460)
        y_traj_134580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'y_traj')
        
        # Obtaining the type of the subscript
        int_134581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 27), 'int')
        slice_134582 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 460, 22), None, None, int_134581)
        # Getting the type of 'yt' (line 460)
        yt_134583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 22), 'yt')
        # Obtaining the member '__getitem__' of a type (line 460)
        getitem___134584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 22), yt_134583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 460)
        subscript_call_result_134585 = invoke(stypy.reporting.localization.Localization(__file__, 460, 22), getitem___134584, slice_134582)
        
        # Applying the binary operator '+=' (line 460)
        result_iadd_134586 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 12), '+=', y_traj_134580, subscript_call_result_134585)
        # Assigning a type to the variable 'y_traj' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'y_traj', result_iadd_134586)
        
        # SSA join for if statement (line 456)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'integration_direction' (line 462)
        integration_direction_134587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 11), 'integration_direction')
        
        # Obtaining an instance of the builtin type 'list' (line 462)
        list_134588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 462)
        # Adding element type (line 462)
        unicode_134589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 37), 'unicode', u'both')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 36), list_134588, unicode_134589)
        # Adding element type (line 462)
        unicode_134590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 45), 'unicode', u'forward')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 36), list_134588, unicode_134590)
        
        # Applying the binary operator 'in' (line 462)
        result_contains_134591 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 11), 'in', integration_direction_134587, list_134588)
        
        # Testing the type of an if condition (line 462)
        if_condition_134592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 8), result_contains_134591)
        # Assigning a type to the variable 'if_condition_134592' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'if_condition_134592', if_condition_134592)
        # SSA begins for if statement (line 462)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to reset_start_point(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'x0' (line 463)
        x0_134595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 35), 'x0', False)
        # Getting the type of 'y0' (line 463)
        y0_134596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 39), 'y0', False)
        # Processing the call keyword arguments (line 463)
        kwargs_134597 = {}
        # Getting the type of 'dmap' (line 463)
        dmap_134593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'dmap', False)
        # Obtaining the member 'reset_start_point' of a type (line 463)
        reset_start_point_134594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 12), dmap_134593, 'reset_start_point')
        # Calling reset_start_point(args, kwargs) (line 463)
        reset_start_point_call_result_134598 = invoke(stypy.reporting.localization.Localization(__file__, 463, 12), reset_start_point_134594, *[x0_134595, y0_134596], **kwargs_134597)
        
        
        # Assigning a Call to a Tuple (line 464):
        
        # Assigning a Call to a Name:
        
        # Call to _integrate_rk12(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'x0' (line 464)
        x0_134600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 40), 'x0', False)
        # Getting the type of 'y0' (line 464)
        y0_134601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 44), 'y0', False)
        # Getting the type of 'dmap' (line 464)
        dmap_134602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 48), 'dmap', False)
        # Getting the type of 'forward_time' (line 464)
        forward_time_134603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 54), 'forward_time', False)
        # Getting the type of 'maxlength' (line 464)
        maxlength_134604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 68), 'maxlength', False)
        # Processing the call keyword arguments (line 464)
        kwargs_134605 = {}
        # Getting the type of '_integrate_rk12' (line 464)
        _integrate_rk12_134599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), '_integrate_rk12', False)
        # Calling _integrate_rk12(args, kwargs) (line 464)
        _integrate_rk12_call_result_134606 = invoke(stypy.reporting.localization.Localization(__file__, 464, 24), _integrate_rk12_134599, *[x0_134600, y0_134601, dmap_134602, forward_time_134603, maxlength_134604], **kwargs_134605)
        
        # Assigning a type to the variable 'call_assignment_133198' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133198', _integrate_rk12_call_result_134606)
        
        # Assigning a Call to a Name (line 464):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 12), 'int')
        # Processing the call keyword arguments
        kwargs_134610 = {}
        # Getting the type of 'call_assignment_133198' (line 464)
        call_assignment_133198_134607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133198', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___134608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 12), call_assignment_133198_134607, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134611 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134608, *[int_134609], **kwargs_134610)
        
        # Assigning a type to the variable 'call_assignment_133199' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133199', getitem___call_result_134611)
        
        # Assigning a Name to a Name (line 464):
        # Getting the type of 'call_assignment_133199' (line 464)
        call_assignment_133199_134612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133199')
        # Assigning a type to the variable 's' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 's', call_assignment_133199_134612)
        
        # Assigning a Call to a Name (line 464):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 12), 'int')
        # Processing the call keyword arguments
        kwargs_134616 = {}
        # Getting the type of 'call_assignment_133198' (line 464)
        call_assignment_133198_134613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133198', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___134614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 12), call_assignment_133198_134613, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134617 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134614, *[int_134615], **kwargs_134616)
        
        # Assigning a type to the variable 'call_assignment_133200' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133200', getitem___call_result_134617)
        
        # Assigning a Name to a Name (line 464):
        # Getting the type of 'call_assignment_133200' (line 464)
        call_assignment_133200_134618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133200')
        # Assigning a type to the variable 'xt' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), 'xt', call_assignment_133200_134618)
        
        # Assigning a Call to a Name (line 464):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_134621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 12), 'int')
        # Processing the call keyword arguments
        kwargs_134622 = {}
        # Getting the type of 'call_assignment_133198' (line 464)
        call_assignment_133198_134619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133198', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___134620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 12), call_assignment_133198_134619, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_134623 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134620, *[int_134621], **kwargs_134622)
        
        # Assigning a type to the variable 'call_assignment_133201' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133201', getitem___call_result_134623)
        
        # Assigning a Name to a Name (line 464):
        # Getting the type of 'call_assignment_133201' (line 464)
        call_assignment_133201_134624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'call_assignment_133201')
        # Assigning a type to the variable 'yt' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 19), 'yt', call_assignment_133201_134624)
        
        
        
        # Call to len(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'x_traj' (line 465)
        x_traj_134626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 19), 'x_traj', False)
        # Processing the call keyword arguments (line 465)
        kwargs_134627 = {}
        # Getting the type of 'len' (line 465)
        len_134625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'len', False)
        # Calling len(args, kwargs) (line 465)
        len_call_result_134628 = invoke(stypy.reporting.localization.Localization(__file__, 465, 15), len_134625, *[x_traj_134626], **kwargs_134627)
        
        int_134629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 29), 'int')
        # Applying the binary operator '>' (line 465)
        result_gt_134630 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 15), '>', len_call_result_134628, int_134629)
        
        # Testing the type of an if condition (line 465)
        if_condition_134631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 465, 12), result_gt_134630)
        # Assigning a type to the variable 'if_condition_134631' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'if_condition_134631', if_condition_134631)
        # SSA begins for if statement (line 465)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 466):
        
        # Assigning a Subscript to a Name (line 466):
        
        # Obtaining the type of the subscript
        int_134632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 24), 'int')
        slice_134633 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 466, 21), int_134632, None, None)
        # Getting the type of 'xt' (line 466)
        xt_134634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 21), 'xt')
        # Obtaining the member '__getitem__' of a type (line 466)
        getitem___134635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 21), xt_134634, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 466)
        subscript_call_result_134636 = invoke(stypy.reporting.localization.Localization(__file__, 466, 21), getitem___134635, slice_134633)
        
        # Assigning a type to the variable 'xt' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'xt', subscript_call_result_134636)
        
        # Assigning a Subscript to a Name (line 467):
        
        # Assigning a Subscript to a Name (line 467):
        
        # Obtaining the type of the subscript
        int_134637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 24), 'int')
        slice_134638 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 467, 21), int_134637, None, None)
        # Getting the type of 'yt' (line 467)
        yt_134639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'yt')
        # Obtaining the member '__getitem__' of a type (line 467)
        getitem___134640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 21), yt_134639, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 467)
        subscript_call_result_134641 = invoke(stypy.reporting.localization.Localization(__file__, 467, 21), getitem___134640, slice_134638)
        
        # Assigning a type to the variable 'yt' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'yt', subscript_call_result_134641)
        # SSA join for if statement (line 465)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'stotal' (line 468)
        stotal_134642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'stotal')
        # Getting the type of 's' (line 468)
        s_134643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 22), 's')
        # Applying the binary operator '+=' (line 468)
        result_iadd_134644 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 12), '+=', stotal_134642, s_134643)
        # Assigning a type to the variable 'stotal' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'stotal', result_iadd_134644)
        
        
        # Getting the type of 'x_traj' (line 469)
        x_traj_134645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'x_traj')
        # Getting the type of 'xt' (line 469)
        xt_134646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 22), 'xt')
        # Applying the binary operator '+=' (line 469)
        result_iadd_134647 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 12), '+=', x_traj_134645, xt_134646)
        # Assigning a type to the variable 'x_traj' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'x_traj', result_iadd_134647)
        
        
        # Getting the type of 'y_traj' (line 470)
        y_traj_134648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'y_traj')
        # Getting the type of 'yt' (line 470)
        yt_134649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 22), 'yt')
        # Applying the binary operator '+=' (line 470)
        result_iadd_134650 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 12), '+=', y_traj_134648, yt_134649)
        # Assigning a type to the variable 'y_traj' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'y_traj', result_iadd_134650)
        
        # SSA join for if statement (line 462)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'stotal' (line 472)
        stotal_134651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 11), 'stotal')
        # Getting the type of 'minlength' (line 472)
        minlength_134652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 20), 'minlength')
        # Applying the binary operator '>' (line 472)
        result_gt_134653 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 11), '>', stotal_134651, minlength_134652)
        
        # Testing the type of an if condition (line 472)
        if_condition_134654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 8), result_gt_134653)
        # Assigning a type to the variable 'if_condition_134654' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'if_condition_134654', if_condition_134654)
        # SSA begins for if statement (line 472)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 473)
        tuple_134655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 473)
        # Adding element type (line 473)
        # Getting the type of 'x_traj' (line 473)
        x_traj_134656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'x_traj')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 19), tuple_134655, x_traj_134656)
        # Adding element type (line 473)
        # Getting the type of 'y_traj' (line 473)
        y_traj_134657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 27), 'y_traj')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 19), tuple_134655, y_traj_134657)
        
        # Assigning a type to the variable 'stypy_return_type' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'stypy_return_type', tuple_134655)
        # SSA branch for the else part of an if statement (line 472)
        module_type_store.open_ssa_branch('else')
        
        # Call to undo_trajectory(...): (line 475)
        # Processing the call keyword arguments (line 475)
        kwargs_134660 = {}
        # Getting the type of 'dmap' (line 475)
        dmap_134658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'dmap', False)
        # Obtaining the member 'undo_trajectory' of a type (line 475)
        undo_trajectory_134659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), dmap_134658, 'undo_trajectory')
        # Calling undo_trajectory(args, kwargs) (line 475)
        undo_trajectory_call_result_134661 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), undo_trajectory_134659, *[], **kwargs_134660)
        
        # Getting the type of 'None' (line 476)
        None_134662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'stypy_return_type', None_134662)
        # SSA join for if statement (line 472)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'integrate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_134663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134663)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate'
        return stypy_return_type_134663

    # Assigning a type to the variable 'integrate' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'integrate', integrate)
    # Getting the type of 'integrate' (line 478)
    integrate_134664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'integrate')
    # Assigning a type to the variable 'stypy_return_type' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type', integrate_134664)
    
    # ################# End of 'get_integrator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_integrator' in the type store
    # Getting the type of 'stypy_return_type' (line 416)
    stypy_return_type_134665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_134665)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_integrator'
    return stypy_return_type_134665

# Assigning a type to the variable 'get_integrator' (line 416)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'get_integrator', get_integrator)

@norecursion
def _integrate_rk12(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_integrate_rk12'
    module_type_store = module_type_store.open_function_context('_integrate_rk12', 481, 0, False)
    
    # Passed parameters checking function
    _integrate_rk12.stypy_localization = localization
    _integrate_rk12.stypy_type_of_self = None
    _integrate_rk12.stypy_type_store = module_type_store
    _integrate_rk12.stypy_function_name = '_integrate_rk12'
    _integrate_rk12.stypy_param_names_list = ['x0', 'y0', 'dmap', 'f', 'maxlength']
    _integrate_rk12.stypy_varargs_param_name = None
    _integrate_rk12.stypy_kwargs_param_name = None
    _integrate_rk12.stypy_call_defaults = defaults
    _integrate_rk12.stypy_call_varargs = varargs
    _integrate_rk12.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_integrate_rk12', ['x0', 'y0', 'dmap', 'f', 'maxlength'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_integrate_rk12', localization, ['x0', 'y0', 'dmap', 'f', 'maxlength'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_integrate_rk12(...)' code ##################

    unicode_134666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, (-1)), 'unicode', u"2nd-order Runge-Kutta algorithm with adaptive step size.\n\n    This method is also referred to as the improved Euler's method, or Heun's\n    method. This method is favored over higher-order methods because:\n\n    1. To get decent looking trajectories and to sample every mask cell\n       on the trajectory we need a small timestep, so a lower order\n       solver doesn't hurt us unless the data is *very* high resolution.\n       In fact, for cases where the user inputs\n       data smaller or of similar grid size to the mask grid, the higher\n       order corrections are negligible because of the very fast linear\n       interpolation used in `interpgrid`.\n\n    2. For high resolution input data (i.e. beyond the mask\n       resolution), we must reduce the timestep. Therefore, an adaptive\n       timestep is more suited to the problem as this would be very hard\n       to judge automatically otherwise.\n\n    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45\n    solvers in most setups on my machine. I would recommend removing the\n    other two to keep things simple.\n    ")
    
    # Assigning a Num to a Name (line 507):
    
    # Assigning a Num to a Name (line 507):
    float_134667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 15), 'float')
    # Assigning a type to the variable 'maxerror' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'maxerror', float_134667)
    
    # Assigning a Call to a Name (line 515):
    
    # Assigning a Call to a Name (line 515):
    
    # Call to min(...): (line 515)
    # Processing the call arguments (line 515)
    float_134669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 16), 'float')
    # Getting the type of 'dmap' (line 515)
    dmap_134670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 21), 'dmap', False)
    # Obtaining the member 'mask' of a type (line 515)
    mask_134671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 21), dmap_134670, 'mask')
    # Obtaining the member 'nx' of a type (line 515)
    nx_134672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 21), mask_134671, 'nx')
    # Applying the binary operator 'div' (line 515)
    result_div_134673 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 16), 'div', float_134669, nx_134672)
    
    float_134674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 35), 'float')
    # Getting the type of 'dmap' (line 515)
    dmap_134675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 40), 'dmap', False)
    # Obtaining the member 'mask' of a type (line 515)
    mask_134676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 40), dmap_134675, 'mask')
    # Obtaining the member 'ny' of a type (line 515)
    ny_134677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 40), mask_134676, 'ny')
    # Applying the binary operator 'div' (line 515)
    result_div_134678 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 35), 'div', float_134674, ny_134677)
    
    float_134679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 54), 'float')
    # Processing the call keyword arguments (line 515)
    kwargs_134680 = {}
    # Getting the type of 'min' (line 515)
    min_134668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'min', False)
    # Calling min(args, kwargs) (line 515)
    min_call_result_134681 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), min_134668, *[result_div_134673, result_div_134678, float_134679], **kwargs_134680)
    
    # Assigning a type to the variable 'maxds' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'maxds', min_call_result_134681)
    
    # Assigning a Name to a Name (line 517):
    
    # Assigning a Name to a Name (line 517):
    # Getting the type of 'maxds' (line 517)
    maxds_134682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 9), 'maxds')
    # Assigning a type to the variable 'ds' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'ds', maxds_134682)
    
    # Assigning a Num to a Name (line 518):
    
    # Assigning a Num to a Name (line 518):
    int_134683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 13), 'int')
    # Assigning a type to the variable 'stotal' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'stotal', int_134683)
    
    # Assigning a Name to a Name (line 519):
    
    # Assigning a Name to a Name (line 519):
    # Getting the type of 'x0' (line 519)
    x0_134684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 9), 'x0')
    # Assigning a type to the variable 'xi' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'xi', x0_134684)
    
    # Assigning a Name to a Name (line 520):
    
    # Assigning a Name to a Name (line 520):
    # Getting the type of 'y0' (line 520)
    y0_134685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 9), 'y0')
    # Assigning a type to the variable 'yi' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'yi', y0_134685)
    
    # Assigning a List to a Name (line 521):
    
    # Assigning a List to a Name (line 521):
    
    # Obtaining an instance of the builtin type 'list' (line 521)
    list_134686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 521)
    
    # Assigning a type to the variable 'xf_traj' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'xf_traj', list_134686)
    
    # Assigning a List to a Name (line 522):
    
    # Assigning a List to a Name (line 522):
    
    # Obtaining an instance of the builtin type 'list' (line 522)
    list_134687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 522)
    
    # Assigning a type to the variable 'yf_traj' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'yf_traj', list_134687)
    
    
    # Call to within_grid(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'xi' (line 524)
    xi_134691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 32), 'xi', False)
    # Getting the type of 'yi' (line 524)
    yi_134692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 36), 'yi', False)
    # Processing the call keyword arguments (line 524)
    kwargs_134693 = {}
    # Getting the type of 'dmap' (line 524)
    dmap_134688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 10), 'dmap', False)
    # Obtaining the member 'grid' of a type (line 524)
    grid_134689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 10), dmap_134688, 'grid')
    # Obtaining the member 'within_grid' of a type (line 524)
    within_grid_134690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 10), grid_134689, 'within_grid')
    # Calling within_grid(args, kwargs) (line 524)
    within_grid_call_result_134694 = invoke(stypy.reporting.localization.Localization(__file__, 524, 10), within_grid_134690, *[xi_134691, yi_134692], **kwargs_134693)
    
    # Testing the type of an if condition (line 524)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 524, 4), within_grid_call_result_134694)
    # SSA begins for while statement (line 524)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to append(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'xi' (line 525)
    xi_134697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'xi', False)
    # Processing the call keyword arguments (line 525)
    kwargs_134698 = {}
    # Getting the type of 'xf_traj' (line 525)
    xf_traj_134695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'xf_traj', False)
    # Obtaining the member 'append' of a type (line 525)
    append_134696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 8), xf_traj_134695, 'append')
    # Calling append(args, kwargs) (line 525)
    append_call_result_134699 = invoke(stypy.reporting.localization.Localization(__file__, 525, 8), append_134696, *[xi_134697], **kwargs_134698)
    
    
    # Call to append(...): (line 526)
    # Processing the call arguments (line 526)
    # Getting the type of 'yi' (line 526)
    yi_134702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 23), 'yi', False)
    # Processing the call keyword arguments (line 526)
    kwargs_134703 = {}
    # Getting the type of 'yf_traj' (line 526)
    yf_traj_134700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'yf_traj', False)
    # Obtaining the member 'append' of a type (line 526)
    append_134701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 8), yf_traj_134700, 'append')
    # Calling append(args, kwargs) (line 526)
    append_call_result_134704 = invoke(stypy.reporting.localization.Localization(__file__, 526, 8), append_134701, *[yi_134702], **kwargs_134703)
    
    
    
    # SSA begins for try-except statement (line 527)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 528):
    
    # Assigning a Call to a Name:
    
    # Call to f(...): (line 528)
    # Processing the call arguments (line 528)
    # Getting the type of 'xi' (line 528)
    xi_134706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 25), 'xi', False)
    # Getting the type of 'yi' (line 528)
    yi_134707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 29), 'yi', False)
    # Processing the call keyword arguments (line 528)
    kwargs_134708 = {}
    # Getting the type of 'f' (line 528)
    f_134705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 23), 'f', False)
    # Calling f(args, kwargs) (line 528)
    f_call_result_134709 = invoke(stypy.reporting.localization.Localization(__file__, 528, 23), f_134705, *[xi_134706, yi_134707], **kwargs_134708)
    
    # Assigning a type to the variable 'call_assignment_133202' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'call_assignment_133202', f_call_result_134709)
    
    # Assigning a Call to a Name (line 528):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 12), 'int')
    # Processing the call keyword arguments
    kwargs_134713 = {}
    # Getting the type of 'call_assignment_133202' (line 528)
    call_assignment_133202_134710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'call_assignment_133202', False)
    # Obtaining the member '__getitem__' of a type (line 528)
    getitem___134711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 12), call_assignment_133202_134710, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134714 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134711, *[int_134712], **kwargs_134713)
    
    # Assigning a type to the variable 'call_assignment_133203' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'call_assignment_133203', getitem___call_result_134714)
    
    # Assigning a Name to a Name (line 528):
    # Getting the type of 'call_assignment_133203' (line 528)
    call_assignment_133203_134715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'call_assignment_133203')
    # Assigning a type to the variable 'k1x' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'k1x', call_assignment_133203_134715)
    
    # Assigning a Call to a Name (line 528):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 12), 'int')
    # Processing the call keyword arguments
    kwargs_134719 = {}
    # Getting the type of 'call_assignment_133202' (line 528)
    call_assignment_133202_134716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'call_assignment_133202', False)
    # Obtaining the member '__getitem__' of a type (line 528)
    getitem___134717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 12), call_assignment_133202_134716, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134720 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134717, *[int_134718], **kwargs_134719)
    
    # Assigning a type to the variable 'call_assignment_133204' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'call_assignment_133204', getitem___call_result_134720)
    
    # Assigning a Name to a Name (line 528):
    # Getting the type of 'call_assignment_133204' (line 528)
    call_assignment_133204_134721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'call_assignment_133204')
    # Assigning a type to the variable 'k1y' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 17), 'k1y', call_assignment_133204_134721)
    
    # Assigning a Call to a Tuple (line 529):
    
    # Assigning a Call to a Name:
    
    # Call to f(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'xi' (line 529)
    xi_134723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 25), 'xi', False)
    # Getting the type of 'ds' (line 529)
    ds_134724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 30), 'ds', False)
    # Getting the type of 'k1x' (line 529)
    k1x_134725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 35), 'k1x', False)
    # Applying the binary operator '*' (line 529)
    result_mul_134726 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 30), '*', ds_134724, k1x_134725)
    
    # Applying the binary operator '+' (line 529)
    result_add_134727 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 25), '+', xi_134723, result_mul_134726)
    
    # Getting the type of 'yi' (line 530)
    yi_134728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 25), 'yi', False)
    # Getting the type of 'ds' (line 530)
    ds_134729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 30), 'ds', False)
    # Getting the type of 'k1y' (line 530)
    k1y_134730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 35), 'k1y', False)
    # Applying the binary operator '*' (line 530)
    result_mul_134731 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 30), '*', ds_134729, k1y_134730)
    
    # Applying the binary operator '+' (line 530)
    result_add_134732 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 25), '+', yi_134728, result_mul_134731)
    
    # Processing the call keyword arguments (line 529)
    kwargs_134733 = {}
    # Getting the type of 'f' (line 529)
    f_134722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 23), 'f', False)
    # Calling f(args, kwargs) (line 529)
    f_call_result_134734 = invoke(stypy.reporting.localization.Localization(__file__, 529, 23), f_134722, *[result_add_134727, result_add_134732], **kwargs_134733)
    
    # Assigning a type to the variable 'call_assignment_133205' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'call_assignment_133205', f_call_result_134734)
    
    # Assigning a Call to a Name (line 529):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 12), 'int')
    # Processing the call keyword arguments
    kwargs_134738 = {}
    # Getting the type of 'call_assignment_133205' (line 529)
    call_assignment_133205_134735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'call_assignment_133205', False)
    # Obtaining the member '__getitem__' of a type (line 529)
    getitem___134736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), call_assignment_133205_134735, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134739 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134736, *[int_134737], **kwargs_134738)
    
    # Assigning a type to the variable 'call_assignment_133206' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'call_assignment_133206', getitem___call_result_134739)
    
    # Assigning a Name to a Name (line 529):
    # Getting the type of 'call_assignment_133206' (line 529)
    call_assignment_133206_134740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'call_assignment_133206')
    # Assigning a type to the variable 'k2x' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'k2x', call_assignment_133206_134740)
    
    # Assigning a Call to a Name (line 529):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 12), 'int')
    # Processing the call keyword arguments
    kwargs_134744 = {}
    # Getting the type of 'call_assignment_133205' (line 529)
    call_assignment_133205_134741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'call_assignment_133205', False)
    # Obtaining the member '__getitem__' of a type (line 529)
    getitem___134742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), call_assignment_133205_134741, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134745 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134742, *[int_134743], **kwargs_134744)
    
    # Assigning a type to the variable 'call_assignment_133207' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'call_assignment_133207', getitem___call_result_134745)
    
    # Assigning a Name to a Name (line 529):
    # Getting the type of 'call_assignment_133207' (line 529)
    call_assignment_133207_134746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'call_assignment_133207')
    # Assigning a type to the variable 'k2y' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 17), 'k2y', call_assignment_133207_134746)
    # SSA branch for the except part of a try statement (line 527)
    # SSA branch for the except 'IndexError' branch of a try statement (line 527)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Tuple (line 534):
    
    # Assigning a Call to a Name:
    
    # Call to _euler_step(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 'xf_traj' (line 534)
    xf_traj_134748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 47), 'xf_traj', False)
    # Getting the type of 'yf_traj' (line 534)
    yf_traj_134749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 56), 'yf_traj', False)
    # Getting the type of 'dmap' (line 534)
    dmap_134750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 65), 'dmap', False)
    # Getting the type of 'f' (line 534)
    f_134751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 71), 'f', False)
    # Processing the call keyword arguments (line 534)
    kwargs_134752 = {}
    # Getting the type of '_euler_step' (line 534)
    _euler_step_134747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 35), '_euler_step', False)
    # Calling _euler_step(args, kwargs) (line 534)
    _euler_step_call_result_134753 = invoke(stypy.reporting.localization.Localization(__file__, 534, 35), _euler_step_134747, *[xf_traj_134748, yf_traj_134749, dmap_134750, f_134751], **kwargs_134752)
    
    # Assigning a type to the variable 'call_assignment_133208' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133208', _euler_step_call_result_134753)
    
    # Assigning a Call to a Name (line 534):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 12), 'int')
    # Processing the call keyword arguments
    kwargs_134757 = {}
    # Getting the type of 'call_assignment_133208' (line 534)
    call_assignment_133208_134754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133208', False)
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___134755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 12), call_assignment_133208_134754, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134758 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134755, *[int_134756], **kwargs_134757)
    
    # Assigning a type to the variable 'call_assignment_133209' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133209', getitem___call_result_134758)
    
    # Assigning a Name to a Name (line 534):
    # Getting the type of 'call_assignment_133209' (line 534)
    call_assignment_133209_134759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133209')
    # Assigning a type to the variable 'ds' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'ds', call_assignment_133209_134759)
    
    # Assigning a Call to a Name (line 534):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 12), 'int')
    # Processing the call keyword arguments
    kwargs_134763 = {}
    # Getting the type of 'call_assignment_133208' (line 534)
    call_assignment_133208_134760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133208', False)
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___134761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 12), call_assignment_133208_134760, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134764 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134761, *[int_134762], **kwargs_134763)
    
    # Assigning a type to the variable 'call_assignment_133210' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133210', getitem___call_result_134764)
    
    # Assigning a Name to a Name (line 534):
    # Getting the type of 'call_assignment_133210' (line 534)
    call_assignment_133210_134765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133210')
    # Assigning a type to the variable 'xf_traj' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'xf_traj', call_assignment_133210_134765)
    
    # Assigning a Call to a Name (line 534):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 12), 'int')
    # Processing the call keyword arguments
    kwargs_134769 = {}
    # Getting the type of 'call_assignment_133208' (line 534)
    call_assignment_133208_134766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133208', False)
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___134767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 12), call_assignment_133208_134766, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134770 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134767, *[int_134768], **kwargs_134769)
    
    # Assigning a type to the variable 'call_assignment_133211' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133211', getitem___call_result_134770)
    
    # Assigning a Name to a Name (line 534):
    # Getting the type of 'call_assignment_133211' (line 534)
    call_assignment_133211_134771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'call_assignment_133211')
    # Assigning a type to the variable 'yf_traj' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 25), 'yf_traj', call_assignment_133211_134771)
    
    # Getting the type of 'stotal' (line 535)
    stotal_134772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'stotal')
    # Getting the type of 'ds' (line 535)
    ds_134773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 22), 'ds')
    # Applying the binary operator '+=' (line 535)
    result_iadd_134774 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 12), '+=', stotal_134772, ds_134773)
    # Assigning a type to the variable 'stotal' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'stotal', result_iadd_134774)
    
    # SSA branch for the except 'TerminateTrajectory' branch of a try statement (line 527)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 527)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 540):
    
    # Assigning a BinOp to a Name (line 540):
    # Getting the type of 'ds' (line 540)
    ds_134775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 14), 'ds')
    # Getting the type of 'k1x' (line 540)
    k1x_134776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 19), 'k1x')
    # Applying the binary operator '*' (line 540)
    result_mul_134777 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 14), '*', ds_134775, k1x_134776)
    
    # Assigning a type to the variable 'dx1' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'dx1', result_mul_134777)
    
    # Assigning a BinOp to a Name (line 541):
    
    # Assigning a BinOp to a Name (line 541):
    # Getting the type of 'ds' (line 541)
    ds_134778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 14), 'ds')
    # Getting the type of 'k1y' (line 541)
    k1y_134779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 19), 'k1y')
    # Applying the binary operator '*' (line 541)
    result_mul_134780 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 14), '*', ds_134778, k1y_134779)
    
    # Assigning a type to the variable 'dy1' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'dy1', result_mul_134780)
    
    # Assigning a BinOp to a Name (line 542):
    
    # Assigning a BinOp to a Name (line 542):
    # Getting the type of 'ds' (line 542)
    ds_134781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 14), 'ds')
    float_134782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 19), 'float')
    # Applying the binary operator '*' (line 542)
    result_mul_134783 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 14), '*', ds_134781, float_134782)
    
    # Getting the type of 'k1x' (line 542)
    k1x_134784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 26), 'k1x')
    # Getting the type of 'k2x' (line 542)
    k2x_134785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'k2x')
    # Applying the binary operator '+' (line 542)
    result_add_134786 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 26), '+', k1x_134784, k2x_134785)
    
    # Applying the binary operator '*' (line 542)
    result_mul_134787 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 23), '*', result_mul_134783, result_add_134786)
    
    # Assigning a type to the variable 'dx2' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'dx2', result_mul_134787)
    
    # Assigning a BinOp to a Name (line 543):
    
    # Assigning a BinOp to a Name (line 543):
    # Getting the type of 'ds' (line 543)
    ds_134788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 14), 'ds')
    float_134789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 19), 'float')
    # Applying the binary operator '*' (line 543)
    result_mul_134790 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 14), '*', ds_134788, float_134789)
    
    # Getting the type of 'k1y' (line 543)
    k1y_134791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 26), 'k1y')
    # Getting the type of 'k2y' (line 543)
    k2y_134792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 32), 'k2y')
    # Applying the binary operator '+' (line 543)
    result_add_134793 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 26), '+', k1y_134791, k2y_134792)
    
    # Applying the binary operator '*' (line 543)
    result_mul_134794 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 23), '*', result_mul_134790, result_add_134793)
    
    # Assigning a type to the variable 'dy2' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'dy2', result_mul_134794)
    
    # Assigning a Attribute to a Tuple (line 545):
    
    # Assigning a Subscript to a Name (line 545):
    
    # Obtaining the type of the subscript
    int_134795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 8), 'int')
    # Getting the type of 'dmap' (line 545)
    dmap_134796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 17), 'dmap')
    # Obtaining the member 'grid' of a type (line 545)
    grid_134797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 17), dmap_134796, 'grid')
    # Obtaining the member 'shape' of a type (line 545)
    shape_134798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 17), grid_134797, 'shape')
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___134799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 8), shape_134798, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_134800 = invoke(stypy.reporting.localization.Localization(__file__, 545, 8), getitem___134799, int_134795)
    
    # Assigning a type to the variable 'tuple_var_assignment_133212' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'tuple_var_assignment_133212', subscript_call_result_134800)
    
    # Assigning a Subscript to a Name (line 545):
    
    # Obtaining the type of the subscript
    int_134801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 8), 'int')
    # Getting the type of 'dmap' (line 545)
    dmap_134802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 17), 'dmap')
    # Obtaining the member 'grid' of a type (line 545)
    grid_134803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 17), dmap_134802, 'grid')
    # Obtaining the member 'shape' of a type (line 545)
    shape_134804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 17), grid_134803, 'shape')
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___134805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 8), shape_134804, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_134806 = invoke(stypy.reporting.localization.Localization(__file__, 545, 8), getitem___134805, int_134801)
    
    # Assigning a type to the variable 'tuple_var_assignment_133213' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'tuple_var_assignment_133213', subscript_call_result_134806)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_var_assignment_133212' (line 545)
    tuple_var_assignment_133212_134807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'tuple_var_assignment_133212')
    # Assigning a type to the variable 'nx' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'nx', tuple_var_assignment_133212_134807)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_var_assignment_133213' (line 545)
    tuple_var_assignment_133213_134808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'tuple_var_assignment_133213')
    # Assigning a type to the variable 'ny' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'ny', tuple_var_assignment_133213_134808)
    
    # Assigning a Call to a Name (line 547):
    
    # Assigning a Call to a Name (line 547):
    
    # Call to sqrt(...): (line 547)
    # Processing the call arguments (line 547)
    # Getting the type of 'dx2' (line 547)
    dx2_134811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 26), 'dx2', False)
    # Getting the type of 'dx1' (line 547)
    dx1_134812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 32), 'dx1', False)
    # Applying the binary operator '-' (line 547)
    result_sub_134813 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 26), '-', dx2_134811, dx1_134812)
    
    # Getting the type of 'nx' (line 547)
    nx_134814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 39), 'nx', False)
    # Applying the binary operator 'div' (line 547)
    result_div_134815 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 25), 'div', result_sub_134813, nx_134814)
    
    int_134816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 46), 'int')
    # Applying the binary operator '**' (line 547)
    result_pow_134817 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 24), '**', result_div_134815, int_134816)
    
    # Getting the type of 'dy2' (line 547)
    dy2_134818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 52), 'dy2', False)
    # Getting the type of 'dy1' (line 547)
    dy1_134819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 58), 'dy1', False)
    # Applying the binary operator '-' (line 547)
    result_sub_134820 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 52), '-', dy2_134818, dy1_134819)
    
    # Getting the type of 'ny' (line 547)
    ny_134821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 65), 'ny', False)
    # Applying the binary operator 'div' (line 547)
    result_div_134822 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 51), 'div', result_sub_134820, ny_134821)
    
    int_134823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 72), 'int')
    # Applying the binary operator '**' (line 547)
    result_pow_134824 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 50), '**', result_div_134822, int_134823)
    
    # Applying the binary operator '+' (line 547)
    result_add_134825 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 24), '+', result_pow_134817, result_pow_134824)
    
    # Processing the call keyword arguments (line 547)
    kwargs_134826 = {}
    # Getting the type of 'np' (line 547)
    np_134809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 547)
    sqrt_134810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 16), np_134809, 'sqrt')
    # Calling sqrt(args, kwargs) (line 547)
    sqrt_call_result_134827 = invoke(stypy.reporting.localization.Localization(__file__, 547, 16), sqrt_134810, *[result_add_134825], **kwargs_134826)
    
    # Assigning a type to the variable 'error' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'error', sqrt_call_result_134827)
    
    
    # Getting the type of 'error' (line 550)
    error_134828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 11), 'error')
    # Getting the type of 'maxerror' (line 550)
    maxerror_134829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'maxerror')
    # Applying the binary operator '<' (line 550)
    result_lt_134830 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 11), '<', error_134828, maxerror_134829)
    
    # Testing the type of an if condition (line 550)
    if_condition_134831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 550, 8), result_lt_134830)
    # Assigning a type to the variable 'if_condition_134831' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'if_condition_134831', if_condition_134831)
    # SSA begins for if statement (line 550)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'xi' (line 551)
    xi_134832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'xi')
    # Getting the type of 'dx2' (line 551)
    dx2_134833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 18), 'dx2')
    # Applying the binary operator '+=' (line 551)
    result_iadd_134834 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 12), '+=', xi_134832, dx2_134833)
    # Assigning a type to the variable 'xi' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'xi', result_iadd_134834)
    
    
    # Getting the type of 'yi' (line 552)
    yi_134835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'yi')
    # Getting the type of 'dy2' (line 552)
    dy2_134836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 18), 'dy2')
    # Applying the binary operator '+=' (line 552)
    result_iadd_134837 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 12), '+=', yi_134835, dy2_134836)
    # Assigning a type to the variable 'yi' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'yi', result_iadd_134837)
    
    
    
    # SSA begins for try-except statement (line 553)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to update_trajectory(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'xi' (line 554)
    xi_134840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 39), 'xi', False)
    # Getting the type of 'yi' (line 554)
    yi_134841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 43), 'yi', False)
    # Processing the call keyword arguments (line 554)
    kwargs_134842 = {}
    # Getting the type of 'dmap' (line 554)
    dmap_134838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'dmap', False)
    # Obtaining the member 'update_trajectory' of a type (line 554)
    update_trajectory_134839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 16), dmap_134838, 'update_trajectory')
    # Calling update_trajectory(args, kwargs) (line 554)
    update_trajectory_call_result_134843 = invoke(stypy.reporting.localization.Localization(__file__, 554, 16), update_trajectory_134839, *[xi_134840, yi_134841], **kwargs_134842)
    
    # SSA branch for the except part of a try statement (line 553)
    # SSA branch for the except 'InvalidIndexError' branch of a try statement (line 553)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 553)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'stotal' (line 557)
    stotal_134844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'stotal')
    # Getting the type of 'ds' (line 557)
    ds_134845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 25), 'ds')
    # Applying the binary operator '+' (line 557)
    result_add_134846 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 16), '+', stotal_134844, ds_134845)
    
    # Getting the type of 'maxlength' (line 557)
    maxlength_134847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 31), 'maxlength')
    # Applying the binary operator '>' (line 557)
    result_gt_134848 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 15), '>', result_add_134846, maxlength_134847)
    
    # Testing the type of an if condition (line 557)
    if_condition_134849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 557, 12), result_gt_134848)
    # Assigning a type to the variable 'if_condition_134849' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'if_condition_134849', if_condition_134849)
    # SSA begins for if statement (line 557)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 557)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'stotal' (line 559)
    stotal_134850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'stotal')
    # Getting the type of 'ds' (line 559)
    ds_134851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 22), 'ds')
    # Applying the binary operator '+=' (line 559)
    result_iadd_134852 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 12), '+=', stotal_134850, ds_134851)
    # Assigning a type to the variable 'stotal' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'stotal', result_iadd_134852)
    
    # SSA join for if statement (line 550)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'error' (line 562)
    error_134853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 11), 'error')
    int_134854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 20), 'int')
    # Applying the binary operator '==' (line 562)
    result_eq_134855 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 11), '==', error_134853, int_134854)
    
    # Testing the type of an if condition (line 562)
    if_condition_134856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 8), result_eq_134855)
    # Assigning a type to the variable 'if_condition_134856' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'if_condition_134856', if_condition_134856)
    # SSA begins for if statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 563):
    
    # Assigning a Name to a Name (line 563):
    # Getting the type of 'maxds' (line 563)
    maxds_134857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 17), 'maxds')
    # Assigning a type to the variable 'ds' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'ds', maxds_134857)
    # SSA branch for the else part of an if statement (line 562)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 565):
    
    # Assigning a Call to a Name (line 565):
    
    # Call to min(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'maxds' (line 565)
    maxds_134859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 21), 'maxds', False)
    float_134860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 28), 'float')
    # Getting the type of 'ds' (line 565)
    ds_134861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 35), 'ds', False)
    # Applying the binary operator '*' (line 565)
    result_mul_134862 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 28), '*', float_134860, ds_134861)
    
    # Getting the type of 'maxerror' (line 565)
    maxerror_134863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 41), 'maxerror', False)
    # Getting the type of 'error' (line 565)
    error_134864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 52), 'error', False)
    # Applying the binary operator 'div' (line 565)
    result_div_134865 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 41), 'div', maxerror_134863, error_134864)
    
    float_134866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 62), 'float')
    # Applying the binary operator '**' (line 565)
    result_pow_134867 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 40), '**', result_div_134865, float_134866)
    
    # Applying the binary operator '*' (line 565)
    result_mul_134868 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 38), '*', result_mul_134862, result_pow_134867)
    
    # Processing the call keyword arguments (line 565)
    kwargs_134869 = {}
    # Getting the type of 'min' (line 565)
    min_134858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 17), 'min', False)
    # Calling min(args, kwargs) (line 565)
    min_call_result_134870 = invoke(stypy.reporting.localization.Localization(__file__, 565, 17), min_134858, *[maxds_134859, result_mul_134868], **kwargs_134869)
    
    # Assigning a type to the variable 'ds' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'ds', min_call_result_134870)
    # SSA join for if statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 524)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 567)
    tuple_134871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 567)
    # Adding element type (line 567)
    # Getting the type of 'stotal' (line 567)
    stotal_134872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 11), 'stotal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 11), tuple_134871, stotal_134872)
    # Adding element type (line 567)
    # Getting the type of 'xf_traj' (line 567)
    xf_traj_134873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 19), 'xf_traj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 11), tuple_134871, xf_traj_134873)
    # Adding element type (line 567)
    # Getting the type of 'yf_traj' (line 567)
    yf_traj_134874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 28), 'yf_traj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 11), tuple_134871, yf_traj_134874)
    
    # Assigning a type to the variable 'stypy_return_type' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'stypy_return_type', tuple_134871)
    
    # ################# End of '_integrate_rk12(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_integrate_rk12' in the type store
    # Getting the type of 'stypy_return_type' (line 481)
    stypy_return_type_134875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_134875)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_integrate_rk12'
    return stypy_return_type_134875

# Assigning a type to the variable '_integrate_rk12' (line 481)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), '_integrate_rk12', _integrate_rk12)

@norecursion
def _euler_step(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_euler_step'
    module_type_store = module_type_store.open_function_context('_euler_step', 570, 0, False)
    
    # Passed parameters checking function
    _euler_step.stypy_localization = localization
    _euler_step.stypy_type_of_self = None
    _euler_step.stypy_type_store = module_type_store
    _euler_step.stypy_function_name = '_euler_step'
    _euler_step.stypy_param_names_list = ['xf_traj', 'yf_traj', 'dmap', 'f']
    _euler_step.stypy_varargs_param_name = None
    _euler_step.stypy_kwargs_param_name = None
    _euler_step.stypy_call_defaults = defaults
    _euler_step.stypy_call_varargs = varargs
    _euler_step.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_euler_step', ['xf_traj', 'yf_traj', 'dmap', 'f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_euler_step', localization, ['xf_traj', 'yf_traj', 'dmap', 'f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_euler_step(...)' code ##################

    unicode_134876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 4), 'unicode', u'Simple Euler integration step that extends streamline to boundary.')
    
    # Assigning a Attribute to a Tuple (line 572):
    
    # Assigning a Subscript to a Name (line 572):
    
    # Obtaining the type of the subscript
    int_134877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 4), 'int')
    # Getting the type of 'dmap' (line 572)
    dmap_134878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 13), 'dmap')
    # Obtaining the member 'grid' of a type (line 572)
    grid_134879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 13), dmap_134878, 'grid')
    # Obtaining the member 'shape' of a type (line 572)
    shape_134880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 13), grid_134879, 'shape')
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___134881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 4), shape_134880, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_134882 = invoke(stypy.reporting.localization.Localization(__file__, 572, 4), getitem___134881, int_134877)
    
    # Assigning a type to the variable 'tuple_var_assignment_133214' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'tuple_var_assignment_133214', subscript_call_result_134882)
    
    # Assigning a Subscript to a Name (line 572):
    
    # Obtaining the type of the subscript
    int_134883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 4), 'int')
    # Getting the type of 'dmap' (line 572)
    dmap_134884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 13), 'dmap')
    # Obtaining the member 'grid' of a type (line 572)
    grid_134885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 13), dmap_134884, 'grid')
    # Obtaining the member 'shape' of a type (line 572)
    shape_134886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 13), grid_134885, 'shape')
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___134887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 4), shape_134886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_134888 = invoke(stypy.reporting.localization.Localization(__file__, 572, 4), getitem___134887, int_134883)
    
    # Assigning a type to the variable 'tuple_var_assignment_133215' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'tuple_var_assignment_133215', subscript_call_result_134888)
    
    # Assigning a Name to a Name (line 572):
    # Getting the type of 'tuple_var_assignment_133214' (line 572)
    tuple_var_assignment_133214_134889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'tuple_var_assignment_133214')
    # Assigning a type to the variable 'ny' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'ny', tuple_var_assignment_133214_134889)
    
    # Assigning a Name to a Name (line 572):
    # Getting the type of 'tuple_var_assignment_133215' (line 572)
    tuple_var_assignment_133215_134890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'tuple_var_assignment_133215')
    # Assigning a type to the variable 'nx' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'nx', tuple_var_assignment_133215_134890)
    
    # Assigning a Subscript to a Name (line 573):
    
    # Assigning a Subscript to a Name (line 573):
    
    # Obtaining the type of the subscript
    int_134891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 17), 'int')
    # Getting the type of 'xf_traj' (line 573)
    xf_traj_134892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 9), 'xf_traj')
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___134893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 9), xf_traj_134892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 573)
    subscript_call_result_134894 = invoke(stypy.reporting.localization.Localization(__file__, 573, 9), getitem___134893, int_134891)
    
    # Assigning a type to the variable 'xi' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'xi', subscript_call_result_134894)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_134895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 17), 'int')
    # Getting the type of 'yf_traj' (line 574)
    yf_traj_134896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 9), 'yf_traj')
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___134897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 9), yf_traj_134896, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_134898 = invoke(stypy.reporting.localization.Localization(__file__, 574, 9), getitem___134897, int_134895)
    
    # Assigning a type to the variable 'yi' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'yi', subscript_call_result_134898)
    
    # Assigning a Call to a Tuple (line 575):
    
    # Assigning a Call to a Name:
    
    # Call to f(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'xi' (line 575)
    xi_134900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 15), 'xi', False)
    # Getting the type of 'yi' (line 575)
    yi_134901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 19), 'yi', False)
    # Processing the call keyword arguments (line 575)
    kwargs_134902 = {}
    # Getting the type of 'f' (line 575)
    f_134899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 13), 'f', False)
    # Calling f(args, kwargs) (line 575)
    f_call_result_134903 = invoke(stypy.reporting.localization.Localization(__file__, 575, 13), f_134899, *[xi_134900, yi_134901], **kwargs_134902)
    
    # Assigning a type to the variable 'call_assignment_133216' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_133216', f_call_result_134903)
    
    # Assigning a Call to a Name (line 575):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 4), 'int')
    # Processing the call keyword arguments
    kwargs_134907 = {}
    # Getting the type of 'call_assignment_133216' (line 575)
    call_assignment_133216_134904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_133216', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___134905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 4), call_assignment_133216_134904, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134908 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134905, *[int_134906], **kwargs_134907)
    
    # Assigning a type to the variable 'call_assignment_133217' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_133217', getitem___call_result_134908)
    
    # Assigning a Name to a Name (line 575):
    # Getting the type of 'call_assignment_133217' (line 575)
    call_assignment_133217_134909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_133217')
    # Assigning a type to the variable 'cx' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'cx', call_assignment_133217_134909)
    
    # Assigning a Call to a Name (line 575):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 4), 'int')
    # Processing the call keyword arguments
    kwargs_134913 = {}
    # Getting the type of 'call_assignment_133216' (line 575)
    call_assignment_133216_134910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_133216', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___134911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 4), call_assignment_133216_134910, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134914 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134911, *[int_134912], **kwargs_134913)
    
    # Assigning a type to the variable 'call_assignment_133218' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_133218', getitem___call_result_134914)
    
    # Assigning a Name to a Name (line 575):
    # Getting the type of 'call_assignment_133218' (line 575)
    call_assignment_133218_134915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_133218')
    # Assigning a type to the variable 'cy' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'cy', call_assignment_133218_134915)
    
    
    # Getting the type of 'cx' (line 576)
    cx_134916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 7), 'cx')
    int_134917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 13), 'int')
    # Applying the binary operator '==' (line 576)
    result_eq_134918 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 7), '==', cx_134916, int_134917)
    
    # Testing the type of an if condition (line 576)
    if_condition_134919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 576, 4), result_eq_134918)
    # Assigning a type to the variable 'if_condition_134919' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'if_condition_134919', if_condition_134919)
    # SSA begins for if statement (line 576)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 577):
    
    # Assigning a Attribute to a Name (line 577):
    # Getting the type of 'np' (line 577)
    np_134920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 14), 'np')
    # Obtaining the member 'inf' of a type (line 577)
    inf_134921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 14), np_134920, 'inf')
    # Assigning a type to the variable 'dsx' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'dsx', inf_134921)
    # SSA branch for the else part of an if statement (line 576)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'cx' (line 578)
    cx_134922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 9), 'cx')
    int_134923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 14), 'int')
    # Applying the binary operator '<' (line 578)
    result_lt_134924 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 9), '<', cx_134922, int_134923)
    
    # Testing the type of an if condition (line 578)
    if_condition_134925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 578, 9), result_lt_134924)
    # Assigning a type to the variable 'if_condition_134925' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 9), 'if_condition_134925', if_condition_134925)
    # SSA begins for if statement (line 578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 579):
    
    # Assigning a BinOp to a Name (line 579):
    # Getting the type of 'xi' (line 579)
    xi_134926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 14), 'xi')
    
    # Getting the type of 'cx' (line 579)
    cx_134927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 20), 'cx')
    # Applying the 'usub' unary operator (line 579)
    result___neg___134928 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 19), 'usub', cx_134927)
    
    # Applying the binary operator 'div' (line 579)
    result_div_134929 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 14), 'div', xi_134926, result___neg___134928)
    
    # Assigning a type to the variable 'dsx' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'dsx', result_div_134929)
    # SSA branch for the else part of an if statement (line 578)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 581):
    
    # Assigning a BinOp to a Name (line 581):
    # Getting the type of 'nx' (line 581)
    nx_134930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'nx')
    int_134931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 20), 'int')
    # Applying the binary operator '-' (line 581)
    result_sub_134932 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 15), '-', nx_134930, int_134931)
    
    # Getting the type of 'xi' (line 581)
    xi_134933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 24), 'xi')
    # Applying the binary operator '-' (line 581)
    result_sub_134934 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 22), '-', result_sub_134932, xi_134933)
    
    # Getting the type of 'cx' (line 581)
    cx_134935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 30), 'cx')
    # Applying the binary operator 'div' (line 581)
    result_div_134936 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 14), 'div', result_sub_134934, cx_134935)
    
    # Assigning a type to the variable 'dsx' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'dsx', result_div_134936)
    # SSA join for if statement (line 578)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 576)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cy' (line 582)
    cy_134937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 7), 'cy')
    int_134938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 13), 'int')
    # Applying the binary operator '==' (line 582)
    result_eq_134939 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 7), '==', cy_134937, int_134938)
    
    # Testing the type of an if condition (line 582)
    if_condition_134940 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 582, 4), result_eq_134939)
    # Assigning a type to the variable 'if_condition_134940' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'if_condition_134940', if_condition_134940)
    # SSA begins for if statement (line 582)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 583):
    
    # Assigning a Attribute to a Name (line 583):
    # Getting the type of 'np' (line 583)
    np_134941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 14), 'np')
    # Obtaining the member 'inf' of a type (line 583)
    inf_134942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 14), np_134941, 'inf')
    # Assigning a type to the variable 'dsy' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'dsy', inf_134942)
    # SSA branch for the else part of an if statement (line 582)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'cy' (line 584)
    cy_134943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 9), 'cy')
    int_134944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 14), 'int')
    # Applying the binary operator '<' (line 584)
    result_lt_134945 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 9), '<', cy_134943, int_134944)
    
    # Testing the type of an if condition (line 584)
    if_condition_134946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 584, 9), result_lt_134945)
    # Assigning a type to the variable 'if_condition_134946' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 9), 'if_condition_134946', if_condition_134946)
    # SSA begins for if statement (line 584)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 585):
    
    # Assigning a BinOp to a Name (line 585):
    # Getting the type of 'yi' (line 585)
    yi_134947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 14), 'yi')
    
    # Getting the type of 'cy' (line 585)
    cy_134948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'cy')
    # Applying the 'usub' unary operator (line 585)
    result___neg___134949 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 19), 'usub', cy_134948)
    
    # Applying the binary operator 'div' (line 585)
    result_div_134950 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 14), 'div', yi_134947, result___neg___134949)
    
    # Assigning a type to the variable 'dsy' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'dsy', result_div_134950)
    # SSA branch for the else part of an if statement (line 584)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 587):
    
    # Assigning a BinOp to a Name (line 587):
    # Getting the type of 'ny' (line 587)
    ny_134951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'ny')
    int_134952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 20), 'int')
    # Applying the binary operator '-' (line 587)
    result_sub_134953 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 15), '-', ny_134951, int_134952)
    
    # Getting the type of 'yi' (line 587)
    yi_134954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 24), 'yi')
    # Applying the binary operator '-' (line 587)
    result_sub_134955 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 22), '-', result_sub_134953, yi_134954)
    
    # Getting the type of 'cy' (line 587)
    cy_134956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'cy')
    # Applying the binary operator 'div' (line 587)
    result_div_134957 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 14), 'div', result_sub_134955, cy_134956)
    
    # Assigning a type to the variable 'dsy' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'dsy', result_div_134957)
    # SSA join for if statement (line 584)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 582)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 588):
    
    # Assigning a Call to a Name (line 588):
    
    # Call to min(...): (line 588)
    # Processing the call arguments (line 588)
    # Getting the type of 'dsx' (line 588)
    dsx_134959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 13), 'dsx', False)
    # Getting the type of 'dsy' (line 588)
    dsy_134960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 18), 'dsy', False)
    # Processing the call keyword arguments (line 588)
    kwargs_134961 = {}
    # Getting the type of 'min' (line 588)
    min_134958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 9), 'min', False)
    # Calling min(args, kwargs) (line 588)
    min_call_result_134962 = invoke(stypy.reporting.localization.Localization(__file__, 588, 9), min_134958, *[dsx_134959, dsy_134960], **kwargs_134961)
    
    # Assigning a type to the variable 'ds' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'ds', min_call_result_134962)
    
    # Call to append(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'xi' (line 589)
    xi_134965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 19), 'xi', False)
    # Getting the type of 'cx' (line 589)
    cx_134966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'cx', False)
    # Getting the type of 'ds' (line 589)
    ds_134967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 29), 'ds', False)
    # Applying the binary operator '*' (line 589)
    result_mul_134968 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 24), '*', cx_134966, ds_134967)
    
    # Applying the binary operator '+' (line 589)
    result_add_134969 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 19), '+', xi_134965, result_mul_134968)
    
    # Processing the call keyword arguments (line 589)
    kwargs_134970 = {}
    # Getting the type of 'xf_traj' (line 589)
    xf_traj_134963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'xf_traj', False)
    # Obtaining the member 'append' of a type (line 589)
    append_134964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 4), xf_traj_134963, 'append')
    # Calling append(args, kwargs) (line 589)
    append_call_result_134971 = invoke(stypy.reporting.localization.Localization(__file__, 589, 4), append_134964, *[result_add_134969], **kwargs_134970)
    
    
    # Call to append(...): (line 590)
    # Processing the call arguments (line 590)
    # Getting the type of 'yi' (line 590)
    yi_134974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 19), 'yi', False)
    # Getting the type of 'cy' (line 590)
    cy_134975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 24), 'cy', False)
    # Getting the type of 'ds' (line 590)
    ds_134976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 29), 'ds', False)
    # Applying the binary operator '*' (line 590)
    result_mul_134977 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 24), '*', cy_134975, ds_134976)
    
    # Applying the binary operator '+' (line 590)
    result_add_134978 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 19), '+', yi_134974, result_mul_134977)
    
    # Processing the call keyword arguments (line 590)
    kwargs_134979 = {}
    # Getting the type of 'yf_traj' (line 590)
    yf_traj_134972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'yf_traj', False)
    # Obtaining the member 'append' of a type (line 590)
    append_134973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 4), yf_traj_134972, 'append')
    # Calling append(args, kwargs) (line 590)
    append_call_result_134980 = invoke(stypy.reporting.localization.Localization(__file__, 590, 4), append_134973, *[result_add_134978], **kwargs_134979)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 591)
    tuple_134981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 591)
    # Adding element type (line 591)
    # Getting the type of 'ds' (line 591)
    ds_134982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 11), 'ds')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 11), tuple_134981, ds_134982)
    # Adding element type (line 591)
    # Getting the type of 'xf_traj' (line 591)
    xf_traj_134983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'xf_traj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 11), tuple_134981, xf_traj_134983)
    # Adding element type (line 591)
    # Getting the type of 'yf_traj' (line 591)
    yf_traj_134984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 24), 'yf_traj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 11), tuple_134981, yf_traj_134984)
    
    # Assigning a type to the variable 'stypy_return_type' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'stypy_return_type', tuple_134981)
    
    # ################# End of '_euler_step(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_euler_step' in the type store
    # Getting the type of 'stypy_return_type' (line 570)
    stypy_return_type_134985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_134985)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_euler_step'
    return stypy_return_type_134985

# Assigning a type to the variable '_euler_step' (line 570)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 0), '_euler_step', _euler_step)

@norecursion
def interpgrid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'interpgrid'
    module_type_store = module_type_store.open_function_context('interpgrid', 597, 0, False)
    
    # Passed parameters checking function
    interpgrid.stypy_localization = localization
    interpgrid.stypy_type_of_self = None
    interpgrid.stypy_type_store = module_type_store
    interpgrid.stypy_function_name = 'interpgrid'
    interpgrid.stypy_param_names_list = ['a', 'xi', 'yi']
    interpgrid.stypy_varargs_param_name = None
    interpgrid.stypy_kwargs_param_name = None
    interpgrid.stypy_call_defaults = defaults
    interpgrid.stypy_call_varargs = varargs
    interpgrid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'interpgrid', ['a', 'xi', 'yi'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'interpgrid', localization, ['a', 'xi', 'yi'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'interpgrid(...)' code ##################

    unicode_134986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 4), 'unicode', u'Fast 2D, linear interpolation on an integer grid')
    
    # Assigning a Call to a Tuple (line 600):
    
    # Assigning a Call to a Name:
    
    # Call to shape(...): (line 600)
    # Processing the call arguments (line 600)
    # Getting the type of 'a' (line 600)
    a_134989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 22), 'a', False)
    # Processing the call keyword arguments (line 600)
    kwargs_134990 = {}
    # Getting the type of 'np' (line 600)
    np_134987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 13), 'np', False)
    # Obtaining the member 'shape' of a type (line 600)
    shape_134988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 13), np_134987, 'shape')
    # Calling shape(args, kwargs) (line 600)
    shape_call_result_134991 = invoke(stypy.reporting.localization.Localization(__file__, 600, 13), shape_134988, *[a_134989], **kwargs_134990)
    
    # Assigning a type to the variable 'call_assignment_133219' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'call_assignment_133219', shape_call_result_134991)
    
    # Assigning a Call to a Name (line 600):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_134994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 4), 'int')
    # Processing the call keyword arguments
    kwargs_134995 = {}
    # Getting the type of 'call_assignment_133219' (line 600)
    call_assignment_133219_134992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'call_assignment_133219', False)
    # Obtaining the member '__getitem__' of a type (line 600)
    getitem___134993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 4), call_assignment_133219_134992, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_134996 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134993, *[int_134994], **kwargs_134995)
    
    # Assigning a type to the variable 'call_assignment_133220' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'call_assignment_133220', getitem___call_result_134996)
    
    # Assigning a Name to a Name (line 600):
    # Getting the type of 'call_assignment_133220' (line 600)
    call_assignment_133220_134997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'call_assignment_133220')
    # Assigning a type to the variable 'Ny' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'Ny', call_assignment_133220_134997)
    
    # Assigning a Call to a Name (line 600):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_135000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 4), 'int')
    # Processing the call keyword arguments
    kwargs_135001 = {}
    # Getting the type of 'call_assignment_133219' (line 600)
    call_assignment_133219_134998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'call_assignment_133219', False)
    # Obtaining the member '__getitem__' of a type (line 600)
    getitem___134999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 4), call_assignment_133219_134998, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_135002 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___134999, *[int_135000], **kwargs_135001)
    
    # Assigning a type to the variable 'call_assignment_133221' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'call_assignment_133221', getitem___call_result_135002)
    
    # Assigning a Name to a Name (line 600):
    # Getting the type of 'call_assignment_133221' (line 600)
    call_assignment_133221_135003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'call_assignment_133221')
    # Assigning a type to the variable 'Nx' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'Nx', call_assignment_133221_135003)
    
    
    # Call to isinstance(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'xi' (line 601)
    xi_135005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 18), 'xi', False)
    # Getting the type of 'np' (line 601)
    np_135006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 22), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 601)
    ndarray_135007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 22), np_135006, 'ndarray')
    # Processing the call keyword arguments (line 601)
    kwargs_135008 = {}
    # Getting the type of 'isinstance' (line 601)
    isinstance_135004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 601)
    isinstance_call_result_135009 = invoke(stypy.reporting.localization.Localization(__file__, 601, 7), isinstance_135004, *[xi_135005, ndarray_135007], **kwargs_135008)
    
    # Testing the type of an if condition (line 601)
    if_condition_135010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 601, 4), isinstance_call_result_135009)
    # Assigning a type to the variable 'if_condition_135010' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'if_condition_135010', if_condition_135010)
    # SSA begins for if statement (line 601)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 602):
    
    # Assigning a Call to a Name (line 602):
    
    # Call to astype(...): (line 602)
    # Processing the call arguments (line 602)
    # Getting the type of 'int' (line 602)
    int_135013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 22), 'int', False)
    # Processing the call keyword arguments (line 602)
    kwargs_135014 = {}
    # Getting the type of 'xi' (line 602)
    xi_135011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'xi', False)
    # Obtaining the member 'astype' of a type (line 602)
    astype_135012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 12), xi_135011, 'astype')
    # Calling astype(args, kwargs) (line 602)
    astype_call_result_135015 = invoke(stypy.reporting.localization.Localization(__file__, 602, 12), astype_135012, *[int_135013], **kwargs_135014)
    
    # Assigning a type to the variable 'x' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'x', astype_call_result_135015)
    
    # Assigning a Call to a Name (line 603):
    
    # Assigning a Call to a Name (line 603):
    
    # Call to astype(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'int' (line 603)
    int_135018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 22), 'int', False)
    # Processing the call keyword arguments (line 603)
    kwargs_135019 = {}
    # Getting the type of 'yi' (line 603)
    yi_135016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'yi', False)
    # Obtaining the member 'astype' of a type (line 603)
    astype_135017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 12), yi_135016, 'astype')
    # Calling astype(args, kwargs) (line 603)
    astype_call_result_135020 = invoke(stypy.reporting.localization.Localization(__file__, 603, 12), astype_135017, *[int_135018], **kwargs_135019)
    
    # Assigning a type to the variable 'y' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'y', astype_call_result_135020)
    
    # Assigning a Call to a Name (line 605):
    
    # Assigning a Call to a Name (line 605):
    
    # Call to clip(...): (line 605)
    # Processing the call arguments (line 605)
    # Getting the type of 'x' (line 605)
    x_135023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 21), 'x', False)
    int_135024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 25), 'int')
    # Applying the binary operator '+' (line 605)
    result_add_135025 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 21), '+', x_135023, int_135024)
    
    int_135026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 28), 'int')
    # Getting the type of 'Nx' (line 605)
    Nx_135027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 31), 'Nx', False)
    int_135028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 36), 'int')
    # Applying the binary operator '-' (line 605)
    result_sub_135029 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 31), '-', Nx_135027, int_135028)
    
    # Processing the call keyword arguments (line 605)
    kwargs_135030 = {}
    # Getting the type of 'np' (line 605)
    np_135021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 13), 'np', False)
    # Obtaining the member 'clip' of a type (line 605)
    clip_135022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 13), np_135021, 'clip')
    # Calling clip(args, kwargs) (line 605)
    clip_call_result_135031 = invoke(stypy.reporting.localization.Localization(__file__, 605, 13), clip_135022, *[result_add_135025, int_135026, result_sub_135029], **kwargs_135030)
    
    # Assigning a type to the variable 'xn' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'xn', clip_call_result_135031)
    
    # Assigning a Call to a Name (line 606):
    
    # Assigning a Call to a Name (line 606):
    
    # Call to clip(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'y' (line 606)
    y_135034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 21), 'y', False)
    int_135035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 25), 'int')
    # Applying the binary operator '+' (line 606)
    result_add_135036 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 21), '+', y_135034, int_135035)
    
    int_135037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 28), 'int')
    # Getting the type of 'Ny' (line 606)
    Ny_135038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 31), 'Ny', False)
    int_135039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 36), 'int')
    # Applying the binary operator '-' (line 606)
    result_sub_135040 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 31), '-', Ny_135038, int_135039)
    
    # Processing the call keyword arguments (line 606)
    kwargs_135041 = {}
    # Getting the type of 'np' (line 606)
    np_135032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 13), 'np', False)
    # Obtaining the member 'clip' of a type (line 606)
    clip_135033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 13), np_135032, 'clip')
    # Calling clip(args, kwargs) (line 606)
    clip_call_result_135042 = invoke(stypy.reporting.localization.Localization(__file__, 606, 13), clip_135033, *[result_add_135036, int_135037, result_sub_135040], **kwargs_135041)
    
    # Assigning a type to the variable 'yn' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'yn', clip_call_result_135042)
    # SSA branch for the else part of an if statement (line 601)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 608):
    
    # Assigning a Call to a Name (line 608):
    
    # Call to int(...): (line 608)
    # Processing the call arguments (line 608)
    # Getting the type of 'xi' (line 608)
    xi_135044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 16), 'xi', False)
    # Processing the call keyword arguments (line 608)
    kwargs_135045 = {}
    # Getting the type of 'int' (line 608)
    int_135043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'int', False)
    # Calling int(args, kwargs) (line 608)
    int_call_result_135046 = invoke(stypy.reporting.localization.Localization(__file__, 608, 12), int_135043, *[xi_135044], **kwargs_135045)
    
    # Assigning a type to the variable 'x' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'x', int_call_result_135046)
    
    # Assigning a Call to a Name (line 609):
    
    # Assigning a Call to a Name (line 609):
    
    # Call to int(...): (line 609)
    # Processing the call arguments (line 609)
    # Getting the type of 'yi' (line 609)
    yi_135048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'yi', False)
    # Processing the call keyword arguments (line 609)
    kwargs_135049 = {}
    # Getting the type of 'int' (line 609)
    int_135047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'int', False)
    # Calling int(args, kwargs) (line 609)
    int_call_result_135050 = invoke(stypy.reporting.localization.Localization(__file__, 609, 12), int_135047, *[yi_135048], **kwargs_135049)
    
    # Assigning a type to the variable 'y' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'y', int_call_result_135050)
    
    
    # Getting the type of 'x' (line 611)
    x_135051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 11), 'x')
    # Getting the type of 'Nx' (line 611)
    Nx_135052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 17), 'Nx')
    int_135053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 22), 'int')
    # Applying the binary operator '-' (line 611)
    result_sub_135054 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 17), '-', Nx_135052, int_135053)
    
    # Applying the binary operator '==' (line 611)
    result_eq_135055 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 11), '==', x_135051, result_sub_135054)
    
    # Testing the type of an if condition (line 611)
    if_condition_135056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 8), result_eq_135055)
    # Assigning a type to the variable 'if_condition_135056' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'if_condition_135056', if_condition_135056)
    # SSA begins for if statement (line 611)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 612):
    
    # Assigning a Name to a Name (line 612):
    # Getting the type of 'x' (line 612)
    x_135057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 17), 'x')
    # Assigning a type to the variable 'xn' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'xn', x_135057)
    # SSA branch for the else part of an if statement (line 611)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 614):
    
    # Assigning a BinOp to a Name (line 614):
    # Getting the type of 'x' (line 614)
    x_135058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 17), 'x')
    int_135059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 21), 'int')
    # Applying the binary operator '+' (line 614)
    result_add_135060 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 17), '+', x_135058, int_135059)
    
    # Assigning a type to the variable 'xn' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'xn', result_add_135060)
    # SSA join for if statement (line 611)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'y' (line 615)
    y_135061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 11), 'y')
    # Getting the type of 'Ny' (line 615)
    Ny_135062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 17), 'Ny')
    int_135063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 22), 'int')
    # Applying the binary operator '-' (line 615)
    result_sub_135064 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 17), '-', Ny_135062, int_135063)
    
    # Applying the binary operator '==' (line 615)
    result_eq_135065 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 11), '==', y_135061, result_sub_135064)
    
    # Testing the type of an if condition (line 615)
    if_condition_135066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 615, 8), result_eq_135065)
    # Assigning a type to the variable 'if_condition_135066' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'if_condition_135066', if_condition_135066)
    # SSA begins for if statement (line 615)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 616):
    
    # Assigning a Name to a Name (line 616):
    # Getting the type of 'y' (line 616)
    y_135067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 17), 'y')
    # Assigning a type to the variable 'yn' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'yn', y_135067)
    # SSA branch for the else part of an if statement (line 615)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 618):
    
    # Assigning a BinOp to a Name (line 618):
    # Getting the type of 'y' (line 618)
    y_135068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 17), 'y')
    int_135069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 21), 'int')
    # Applying the binary operator '+' (line 618)
    result_add_135070 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 17), '+', y_135068, int_135069)
    
    # Assigning a type to the variable 'yn' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'yn', result_add_135070)
    # SSA join for if statement (line 615)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 601)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 620):
    
    # Assigning a Subscript to a Name (line 620):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 620)
    tuple_135071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 620)
    # Adding element type (line 620)
    # Getting the type of 'y' (line 620)
    y_135072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 12), tuple_135071, y_135072)
    # Adding element type (line 620)
    # Getting the type of 'x' (line 620)
    x_135073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 12), tuple_135071, x_135073)
    
    # Getting the type of 'a' (line 620)
    a_135074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 10), 'a')
    # Obtaining the member '__getitem__' of a type (line 620)
    getitem___135075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 10), a_135074, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 620)
    subscript_call_result_135076 = invoke(stypy.reporting.localization.Localization(__file__, 620, 10), getitem___135075, tuple_135071)
    
    # Assigning a type to the variable 'a00' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'a00', subscript_call_result_135076)
    
    # Assigning a Subscript to a Name (line 621):
    
    # Assigning a Subscript to a Name (line 621):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 621)
    tuple_135077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 621)
    # Adding element type (line 621)
    # Getting the type of 'y' (line 621)
    y_135078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 12), tuple_135077, y_135078)
    # Adding element type (line 621)
    # Getting the type of 'xn' (line 621)
    xn_135079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 15), 'xn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 12), tuple_135077, xn_135079)
    
    # Getting the type of 'a' (line 621)
    a_135080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 10), 'a')
    # Obtaining the member '__getitem__' of a type (line 621)
    getitem___135081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 10), a_135080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 621)
    subscript_call_result_135082 = invoke(stypy.reporting.localization.Localization(__file__, 621, 10), getitem___135081, tuple_135077)
    
    # Assigning a type to the variable 'a01' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'a01', subscript_call_result_135082)
    
    # Assigning a Subscript to a Name (line 622):
    
    # Assigning a Subscript to a Name (line 622):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 622)
    tuple_135083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 622)
    # Adding element type (line 622)
    # Getting the type of 'yn' (line 622)
    yn_135084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'yn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 12), tuple_135083, yn_135084)
    # Adding element type (line 622)
    # Getting the type of 'x' (line 622)
    x_135085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 16), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 12), tuple_135083, x_135085)
    
    # Getting the type of 'a' (line 622)
    a_135086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 10), 'a')
    # Obtaining the member '__getitem__' of a type (line 622)
    getitem___135087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 10), a_135086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 622)
    subscript_call_result_135088 = invoke(stypy.reporting.localization.Localization(__file__, 622, 10), getitem___135087, tuple_135083)
    
    # Assigning a type to the variable 'a10' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'a10', subscript_call_result_135088)
    
    # Assigning a Subscript to a Name (line 623):
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 623)
    tuple_135089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 623)
    # Adding element type (line 623)
    # Getting the type of 'yn' (line 623)
    yn_135090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'yn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 12), tuple_135089, yn_135090)
    # Adding element type (line 623)
    # Getting the type of 'xn' (line 623)
    xn_135091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'xn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 12), tuple_135089, xn_135091)
    
    # Getting the type of 'a' (line 623)
    a_135092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 10), 'a')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___135093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 10), a_135092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_135094 = invoke(stypy.reporting.localization.Localization(__file__, 623, 10), getitem___135093, tuple_135089)
    
    # Assigning a type to the variable 'a11' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'a11', subscript_call_result_135094)
    
    # Assigning a BinOp to a Name (line 624):
    
    # Assigning a BinOp to a Name (line 624):
    # Getting the type of 'xi' (line 624)
    xi_135095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 9), 'xi')
    # Getting the type of 'x' (line 624)
    x_135096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 14), 'x')
    # Applying the binary operator '-' (line 624)
    result_sub_135097 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 9), '-', xi_135095, x_135096)
    
    # Assigning a type to the variable 'xt' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 4), 'xt', result_sub_135097)
    
    # Assigning a BinOp to a Name (line 625):
    
    # Assigning a BinOp to a Name (line 625):
    # Getting the type of 'yi' (line 625)
    yi_135098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 9), 'yi')
    # Getting the type of 'y' (line 625)
    y_135099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 14), 'y')
    # Applying the binary operator '-' (line 625)
    result_sub_135100 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 9), '-', yi_135098, y_135099)
    
    # Assigning a type to the variable 'yt' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'yt', result_sub_135100)
    
    # Assigning a BinOp to a Name (line 626):
    
    # Assigning a BinOp to a Name (line 626):
    # Getting the type of 'a00' (line 626)
    a00_135101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 9), 'a00')
    int_135102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 16), 'int')
    # Getting the type of 'xt' (line 626)
    xt_135103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 20), 'xt')
    # Applying the binary operator '-' (line 626)
    result_sub_135104 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 16), '-', int_135102, xt_135103)
    
    # Applying the binary operator '*' (line 626)
    result_mul_135105 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 9), '*', a00_135101, result_sub_135104)
    
    # Getting the type of 'a01' (line 626)
    a01_135106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 26), 'a01')
    # Getting the type of 'xt' (line 626)
    xt_135107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 32), 'xt')
    # Applying the binary operator '*' (line 626)
    result_mul_135108 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 26), '*', a01_135106, xt_135107)
    
    # Applying the binary operator '+' (line 626)
    result_add_135109 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 9), '+', result_mul_135105, result_mul_135108)
    
    # Assigning a type to the variable 'a0' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'a0', result_add_135109)
    
    # Assigning a BinOp to a Name (line 627):
    
    # Assigning a BinOp to a Name (line 627):
    # Getting the type of 'a10' (line 627)
    a10_135110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 9), 'a10')
    int_135111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 16), 'int')
    # Getting the type of 'xt' (line 627)
    xt_135112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 20), 'xt')
    # Applying the binary operator '-' (line 627)
    result_sub_135113 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 16), '-', int_135111, xt_135112)
    
    # Applying the binary operator '*' (line 627)
    result_mul_135114 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 9), '*', a10_135110, result_sub_135113)
    
    # Getting the type of 'a11' (line 627)
    a11_135115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 26), 'a11')
    # Getting the type of 'xt' (line 627)
    xt_135116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 32), 'xt')
    # Applying the binary operator '*' (line 627)
    result_mul_135117 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 26), '*', a11_135115, xt_135116)
    
    # Applying the binary operator '+' (line 627)
    result_add_135118 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 9), '+', result_mul_135114, result_mul_135117)
    
    # Assigning a type to the variable 'a1' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'a1', result_add_135118)
    
    # Assigning a BinOp to a Name (line 628):
    
    # Assigning a BinOp to a Name (line 628):
    # Getting the type of 'a0' (line 628)
    a0_135119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 9), 'a0')
    int_135120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 15), 'int')
    # Getting the type of 'yt' (line 628)
    yt_135121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 19), 'yt')
    # Applying the binary operator '-' (line 628)
    result_sub_135122 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 15), '-', int_135120, yt_135121)
    
    # Applying the binary operator '*' (line 628)
    result_mul_135123 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 9), '*', a0_135119, result_sub_135122)
    
    # Getting the type of 'a1' (line 628)
    a1_135124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 25), 'a1')
    # Getting the type of 'yt' (line 628)
    yt_135125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 30), 'yt')
    # Applying the binary operator '*' (line 628)
    result_mul_135126 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 25), '*', a1_135124, yt_135125)
    
    # Applying the binary operator '+' (line 628)
    result_add_135127 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 9), '+', result_mul_135123, result_mul_135126)
    
    # Assigning a type to the variable 'ai' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'ai', result_add_135127)
    
    
    
    # Call to isinstance(...): (line 630)
    # Processing the call arguments (line 630)
    # Getting the type of 'xi' (line 630)
    xi_135129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 22), 'xi', False)
    # Getting the type of 'np' (line 630)
    np_135130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 26), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 630)
    ndarray_135131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 26), np_135130, 'ndarray')
    # Processing the call keyword arguments (line 630)
    kwargs_135132 = {}
    # Getting the type of 'isinstance' (line 630)
    isinstance_135128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 630)
    isinstance_call_result_135133 = invoke(stypy.reporting.localization.Localization(__file__, 630, 11), isinstance_135128, *[xi_135129, ndarray_135131], **kwargs_135132)
    
    # Applying the 'not' unary operator (line 630)
    result_not__135134 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 7), 'not', isinstance_call_result_135133)
    
    # Testing the type of an if condition (line 630)
    if_condition_135135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 630, 4), result_not__135134)
    # Assigning a type to the variable 'if_condition_135135' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'if_condition_135135', if_condition_135135)
    # SSA begins for if statement (line 630)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to is_masked(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'ai' (line 631)
    ai_135139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 27), 'ai', False)
    # Processing the call keyword arguments (line 631)
    kwargs_135140 = {}
    # Getting the type of 'np' (line 631)
    np_135136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 11), 'np', False)
    # Obtaining the member 'ma' of a type (line 631)
    ma_135137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 11), np_135136, 'ma')
    # Obtaining the member 'is_masked' of a type (line 631)
    is_masked_135138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 11), ma_135137, 'is_masked')
    # Calling is_masked(args, kwargs) (line 631)
    is_masked_call_result_135141 = invoke(stypy.reporting.localization.Localization(__file__, 631, 11), is_masked_135138, *[ai_135139], **kwargs_135140)
    
    # Testing the type of an if condition (line 631)
    if_condition_135142 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 631, 8), is_masked_call_result_135141)
    # Assigning a type to the variable 'if_condition_135142' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'if_condition_135142', if_condition_135142)
    # SSA begins for if statement (line 631)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'TerminateTrajectory' (line 632)
    TerminateTrajectory_135143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 18), 'TerminateTrajectory')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 632, 12), TerminateTrajectory_135143, 'raise parameter', BaseException)
    # SSA join for if statement (line 631)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 630)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ai' (line 634)
    ai_135144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 11), 'ai')
    # Assigning a type to the variable 'stypy_return_type' (line 634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'stypy_return_type', ai_135144)
    
    # ################# End of 'interpgrid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'interpgrid' in the type store
    # Getting the type of 'stypy_return_type' (line 597)
    stypy_return_type_135145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_135145)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'interpgrid'
    return stypy_return_type_135145

# Assigning a type to the variable 'interpgrid' (line 597)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 0), 'interpgrid', interpgrid)

@norecursion
def _gen_starting_points(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_gen_starting_points'
    module_type_store = module_type_store.open_function_context('_gen_starting_points', 637, 0, False)
    
    # Passed parameters checking function
    _gen_starting_points.stypy_localization = localization
    _gen_starting_points.stypy_type_of_self = None
    _gen_starting_points.stypy_type_store = module_type_store
    _gen_starting_points.stypy_function_name = '_gen_starting_points'
    _gen_starting_points.stypy_param_names_list = ['shape']
    _gen_starting_points.stypy_varargs_param_name = None
    _gen_starting_points.stypy_kwargs_param_name = None
    _gen_starting_points.stypy_call_defaults = defaults
    _gen_starting_points.stypy_call_varargs = varargs
    _gen_starting_points.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_gen_starting_points', ['shape'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_gen_starting_points', localization, ['shape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_gen_starting_points(...)' code ##################

    unicode_135146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, (-1)), 'unicode', u'Yield starting points for streamlines.\n\n    Trying points on the boundary first gives higher quality streamlines.\n    This algorithm starts with a point on the mask corner and spirals inward.\n    This algorithm is inefficient, but fast compared to rest of streamplot.\n    ')
    
    # Assigning a Name to a Tuple (line 644):
    
    # Assigning a Subscript to a Name (line 644):
    
    # Obtaining the type of the subscript
    int_135147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 4), 'int')
    # Getting the type of 'shape' (line 644)
    shape_135148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 13), 'shape')
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___135149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 4), shape_135148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_135150 = invoke(stypy.reporting.localization.Localization(__file__, 644, 4), getitem___135149, int_135147)
    
    # Assigning a type to the variable 'tuple_var_assignment_133222' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'tuple_var_assignment_133222', subscript_call_result_135150)
    
    # Assigning a Subscript to a Name (line 644):
    
    # Obtaining the type of the subscript
    int_135151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 4), 'int')
    # Getting the type of 'shape' (line 644)
    shape_135152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 13), 'shape')
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___135153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 4), shape_135152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_135154 = invoke(stypy.reporting.localization.Localization(__file__, 644, 4), getitem___135153, int_135151)
    
    # Assigning a type to the variable 'tuple_var_assignment_133223' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'tuple_var_assignment_133223', subscript_call_result_135154)
    
    # Assigning a Name to a Name (line 644):
    # Getting the type of 'tuple_var_assignment_133222' (line 644)
    tuple_var_assignment_133222_135155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'tuple_var_assignment_133222')
    # Assigning a type to the variable 'ny' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'ny', tuple_var_assignment_133222_135155)
    
    # Assigning a Name to a Name (line 644):
    # Getting the type of 'tuple_var_assignment_133223' (line 644)
    tuple_var_assignment_133223_135156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'tuple_var_assignment_133223')
    # Assigning a type to the variable 'nx' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'nx', tuple_var_assignment_133223_135156)
    
    # Assigning a Num to a Name (line 645):
    
    # Assigning a Num to a Name (line 645):
    int_135157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 13), 'int')
    # Assigning a type to the variable 'xfirst' (line 645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 4), 'xfirst', int_135157)
    
    # Assigning a Num to a Name (line 646):
    
    # Assigning a Num to a Name (line 646):
    int_135158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 13), 'int')
    # Assigning a type to the variable 'yfirst' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 4), 'yfirst', int_135158)
    
    # Assigning a BinOp to a Name (line 647):
    
    # Assigning a BinOp to a Name (line 647):
    # Getting the type of 'nx' (line 647)
    nx_135159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'nx')
    int_135160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 17), 'int')
    # Applying the binary operator '-' (line 647)
    result_sub_135161 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 12), '-', nx_135159, int_135160)
    
    # Assigning a type to the variable 'xlast' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'xlast', result_sub_135161)
    
    # Assigning a BinOp to a Name (line 648):
    
    # Assigning a BinOp to a Name (line 648):
    # Getting the type of 'ny' (line 648)
    ny_135162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'ny')
    int_135163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 17), 'int')
    # Applying the binary operator '-' (line 648)
    result_sub_135164 = python_operator(stypy.reporting.localization.Localization(__file__, 648, 12), '-', ny_135162, int_135163)
    
    # Assigning a type to the variable 'ylast' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'ylast', result_sub_135164)
    
    # Assigning a Tuple to a Tuple (line 649):
    
    # Assigning a Num to a Name (line 649):
    int_135165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_133224' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_assignment_133224', int_135165)
    
    # Assigning a Num to a Name (line 649):
    int_135166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_133225' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_assignment_133225', int_135166)
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_assignment_133224' (line 649)
    tuple_assignment_133224_135167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_assignment_133224')
    # Assigning a type to the variable 'x' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'x', tuple_assignment_133224_135167)
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_assignment_133225' (line 649)
    tuple_assignment_133225_135168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_assignment_133225')
    # Assigning a type to the variable 'y' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 7), 'y', tuple_assignment_133225_135168)
    
    # Assigning a Num to a Name (line 650):
    
    # Assigning a Num to a Name (line 650):
    int_135169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 8), 'int')
    # Assigning a type to the variable 'i' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'i', int_135169)
    
    # Assigning a Str to a Name (line 651):
    
    # Assigning a Str to a Name (line 651):
    unicode_135170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 16), 'unicode', u'right')
    # Assigning a type to the variable 'direction' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 4), 'direction', unicode_135170)
    
    
    # Call to xrange(...): (line 652)
    # Processing the call arguments (line 652)
    # Getting the type of 'nx' (line 652)
    nx_135172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 20), 'nx', False)
    # Getting the type of 'ny' (line 652)
    ny_135173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 25), 'ny', False)
    # Applying the binary operator '*' (line 652)
    result_mul_135174 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 20), '*', nx_135172, ny_135173)
    
    # Processing the call keyword arguments (line 652)
    kwargs_135175 = {}
    # Getting the type of 'xrange' (line 652)
    xrange_135171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 652)
    xrange_call_result_135176 = invoke(stypy.reporting.localization.Localization(__file__, 652, 13), xrange_135171, *[result_mul_135174], **kwargs_135175)
    
    # Testing the type of a for loop iterable (line 652)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 652, 4), xrange_call_result_135176)
    # Getting the type of the for loop variable (line 652)
    for_loop_var_135177 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 652, 4), xrange_call_result_135176)
    # Assigning a type to the variable 'i' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'i', for_loop_var_135177)
    # SSA begins for a for statement (line 652)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Creating a generator
    
    # Obtaining an instance of the builtin type 'tuple' (line 654)
    tuple_135178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 654)
    # Adding element type (line 654)
    # Getting the type of 'x' (line 654)
    x_135179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 14), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 14), tuple_135178, x_135179)
    # Adding element type (line 654)
    # Getting the type of 'y' (line 654)
    y_135180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 17), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 14), tuple_135178, y_135180)
    
    GeneratorType_135181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 8), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 8), GeneratorType_135181, tuple_135178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'stypy_return_type', GeneratorType_135181)
    
    
    # Getting the type of 'direction' (line 656)
    direction_135182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 11), 'direction')
    unicode_135183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 24), 'unicode', u'right')
    # Applying the binary operator '==' (line 656)
    result_eq_135184 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 11), '==', direction_135182, unicode_135183)
    
    # Testing the type of an if condition (line 656)
    if_condition_135185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 656, 8), result_eq_135184)
    # Assigning a type to the variable 'if_condition_135185' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'if_condition_135185', if_condition_135185)
    # SSA begins for if statement (line 656)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'x' (line 657)
    x_135186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'x')
    int_135187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 17), 'int')
    # Applying the binary operator '+=' (line 657)
    result_iadd_135188 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 12), '+=', x_135186, int_135187)
    # Assigning a type to the variable 'x' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'x', result_iadd_135188)
    
    
    
    # Getting the type of 'x' (line 658)
    x_135189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 15), 'x')
    # Getting the type of 'xlast' (line 658)
    xlast_135190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 20), 'xlast')
    # Applying the binary operator '>=' (line 658)
    result_ge_135191 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 15), '>=', x_135189, xlast_135190)
    
    # Testing the type of an if condition (line 658)
    if_condition_135192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 658, 12), result_ge_135191)
    # Assigning a type to the variable 'if_condition_135192' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'if_condition_135192', if_condition_135192)
    # SSA begins for if statement (line 658)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'xlast' (line 659)
    xlast_135193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'xlast')
    int_135194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 25), 'int')
    # Applying the binary operator '-=' (line 659)
    result_isub_135195 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 16), '-=', xlast_135193, int_135194)
    # Assigning a type to the variable 'xlast' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'xlast', result_isub_135195)
    
    
    # Assigning a Str to a Name (line 660):
    
    # Assigning a Str to a Name (line 660):
    unicode_135196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 28), 'unicode', u'up')
    # Assigning a type to the variable 'direction' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'direction', unicode_135196)
    # SSA join for if statement (line 658)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 656)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'direction' (line 661)
    direction_135197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 13), 'direction')
    unicode_135198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 26), 'unicode', u'up')
    # Applying the binary operator '==' (line 661)
    result_eq_135199 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 13), '==', direction_135197, unicode_135198)
    
    # Testing the type of an if condition (line 661)
    if_condition_135200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 13), result_eq_135199)
    # Assigning a type to the variable 'if_condition_135200' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 13), 'if_condition_135200', if_condition_135200)
    # SSA begins for if statement (line 661)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'y' (line 662)
    y_135201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'y')
    int_135202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 17), 'int')
    # Applying the binary operator '+=' (line 662)
    result_iadd_135203 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 12), '+=', y_135201, int_135202)
    # Assigning a type to the variable 'y' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'y', result_iadd_135203)
    
    
    
    # Getting the type of 'y' (line 663)
    y_135204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 15), 'y')
    # Getting the type of 'ylast' (line 663)
    ylast_135205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 20), 'ylast')
    # Applying the binary operator '>=' (line 663)
    result_ge_135206 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 15), '>=', y_135204, ylast_135205)
    
    # Testing the type of an if condition (line 663)
    if_condition_135207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 12), result_ge_135206)
    # Assigning a type to the variable 'if_condition_135207' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'if_condition_135207', if_condition_135207)
    # SSA begins for if statement (line 663)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ylast' (line 664)
    ylast_135208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 16), 'ylast')
    int_135209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 25), 'int')
    # Applying the binary operator '-=' (line 664)
    result_isub_135210 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 16), '-=', ylast_135208, int_135209)
    # Assigning a type to the variable 'ylast' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 16), 'ylast', result_isub_135210)
    
    
    # Assigning a Str to a Name (line 665):
    
    # Assigning a Str to a Name (line 665):
    unicode_135211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 28), 'unicode', u'left')
    # Assigning a type to the variable 'direction' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'direction', unicode_135211)
    # SSA join for if statement (line 663)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 661)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'direction' (line 666)
    direction_135212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 13), 'direction')
    unicode_135213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 26), 'unicode', u'left')
    # Applying the binary operator '==' (line 666)
    result_eq_135214 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 13), '==', direction_135212, unicode_135213)
    
    # Testing the type of an if condition (line 666)
    if_condition_135215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 13), result_eq_135214)
    # Assigning a type to the variable 'if_condition_135215' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 13), 'if_condition_135215', if_condition_135215)
    # SSA begins for if statement (line 666)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'x' (line 667)
    x_135216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'x')
    int_135217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 17), 'int')
    # Applying the binary operator '-=' (line 667)
    result_isub_135218 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 12), '-=', x_135216, int_135217)
    # Assigning a type to the variable 'x' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'x', result_isub_135218)
    
    
    
    # Getting the type of 'x' (line 668)
    x_135219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'x')
    # Getting the type of 'xfirst' (line 668)
    xfirst_135220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 20), 'xfirst')
    # Applying the binary operator '<=' (line 668)
    result_le_135221 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 15), '<=', x_135219, xfirst_135220)
    
    # Testing the type of an if condition (line 668)
    if_condition_135222 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 12), result_le_135221)
    # Assigning a type to the variable 'if_condition_135222' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'if_condition_135222', if_condition_135222)
    # SSA begins for if statement (line 668)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'xfirst' (line 669)
    xfirst_135223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 16), 'xfirst')
    int_135224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 26), 'int')
    # Applying the binary operator '+=' (line 669)
    result_iadd_135225 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 16), '+=', xfirst_135223, int_135224)
    # Assigning a type to the variable 'xfirst' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 16), 'xfirst', result_iadd_135225)
    
    
    # Assigning a Str to a Name (line 670):
    
    # Assigning a Str to a Name (line 670):
    unicode_135226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 28), 'unicode', u'down')
    # Assigning a type to the variable 'direction' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 16), 'direction', unicode_135226)
    # SSA join for if statement (line 668)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 666)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'direction' (line 671)
    direction_135227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 13), 'direction')
    unicode_135228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 26), 'unicode', u'down')
    # Applying the binary operator '==' (line 671)
    result_eq_135229 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 13), '==', direction_135227, unicode_135228)
    
    # Testing the type of an if condition (line 671)
    if_condition_135230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 671, 13), result_eq_135229)
    # Assigning a type to the variable 'if_condition_135230' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 13), 'if_condition_135230', if_condition_135230)
    # SSA begins for if statement (line 671)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'y' (line 672)
    y_135231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'y')
    int_135232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 17), 'int')
    # Applying the binary operator '-=' (line 672)
    result_isub_135233 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 12), '-=', y_135231, int_135232)
    # Assigning a type to the variable 'y' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'y', result_isub_135233)
    
    
    
    # Getting the type of 'y' (line 673)
    y_135234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 15), 'y')
    # Getting the type of 'yfirst' (line 673)
    yfirst_135235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 20), 'yfirst')
    # Applying the binary operator '<=' (line 673)
    result_le_135236 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 15), '<=', y_135234, yfirst_135235)
    
    # Testing the type of an if condition (line 673)
    if_condition_135237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 673, 12), result_le_135236)
    # Assigning a type to the variable 'if_condition_135237' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'if_condition_135237', if_condition_135237)
    # SSA begins for if statement (line 673)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'yfirst' (line 674)
    yfirst_135238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'yfirst')
    int_135239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 26), 'int')
    # Applying the binary operator '+=' (line 674)
    result_iadd_135240 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 16), '+=', yfirst_135238, int_135239)
    # Assigning a type to the variable 'yfirst' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'yfirst', result_iadd_135240)
    
    
    # Assigning a Str to a Name (line 675):
    
    # Assigning a Str to a Name (line 675):
    unicode_135241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 28), 'unicode', u'right')
    # Assigning a type to the variable 'direction' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'direction', unicode_135241)
    # SSA join for if statement (line 673)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 671)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 666)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 661)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 656)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_gen_starting_points(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_gen_starting_points' in the type store
    # Getting the type of 'stypy_return_type' (line 637)
    stypy_return_type_135242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_135242)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_gen_starting_points'
    return stypy_return_type_135242

# Assigning a type to the variable '_gen_starting_points' (line 637)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), '_gen_starting_points', _gen_starting_points)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
