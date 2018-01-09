
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A module providing some utility functions regarding bezier path manipulation.
3: '''
4: 
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: 
8: import six
9: 
10: import numpy as np
11: from matplotlib.path import Path
12: 
13: from operator import xor
14: import warnings
15: 
16: 
17: class NonIntersectingPathException(ValueError):
18:     pass
19: 
20: # some functions
21: 
22: 
23: def get_intersection(cx1, cy1, cos_t1, sin_t1,
24:                      cx2, cy2, cos_t2, sin_t2):
25:     ''' return a intersecting point between a line through (cx1, cy1)
26:     and having angle t1 and a line through (cx2, cy2) and angle t2.
27:     '''
28: 
29:     # line1 => sin_t1 * (x - cx1) - cos_t1 * (y - cy1) = 0.
30:     # line1 => sin_t1 * x + cos_t1 * y = sin_t1*cx1 - cos_t1*cy1
31: 
32:     line1_rhs = sin_t1 * cx1 - cos_t1 * cy1
33:     line2_rhs = sin_t2 * cx2 - cos_t2 * cy2
34: 
35:     # rhs matrix
36:     a, b = sin_t1, -cos_t1
37:     c, d = sin_t2, -cos_t2
38: 
39:     ad_bc = a * d - b * c
40:     if ad_bc == 0.:
41:         raise ValueError("Given lines do not intersect")
42: 
43:     # rhs_inverse
44:     a_, b_ = d, -b
45:     c_, d_ = -c, a
46:     a_, b_, c_, d_ = [k / ad_bc for k in [a_, b_, c_, d_]]
47: 
48:     x = a_ * line1_rhs + b_ * line2_rhs
49:     y = c_ * line1_rhs + d_ * line2_rhs
50: 
51:     return x, y
52: 
53: 
54: def get_normal_points(cx, cy, cos_t, sin_t, length):
55:     '''
56:     For a line passing through (*cx*, *cy*) and having a angle *t*, return
57:     locations of the two points located along its perpendicular line at the
58:     distance of *length*.
59:     '''
60: 
61:     if length == 0.:
62:         return cx, cy, cx, cy
63: 
64:     cos_t1, sin_t1 = sin_t, -cos_t
65:     cos_t2, sin_t2 = -sin_t, cos_t
66: 
67:     x1, y1 = length * cos_t1 + cx, length * sin_t1 + cy
68:     x2, y2 = length * cos_t2 + cx, length * sin_t2 + cy
69: 
70:     return x1, y1, x2, y2
71: 
72: 
73: # BEZIER routines
74: 
75: # subdividing bezier curve
76: # http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html
77: 
78: 
79: def _de_casteljau1(beta, t):
80:     next_beta = beta[:-1] * (1 - t) + beta[1:] * t
81:     return next_beta
82: 
83: 
84: def split_de_casteljau(beta, t):
85:     '''split a bezier segment defined by its controlpoints *beta*
86:     into two separate segment divided at *t* and return their control points.
87: 
88:     '''
89:     beta = np.asarray(beta)
90:     beta_list = [beta]
91:     while True:
92:         beta = _de_casteljau1(beta, t)
93:         beta_list.append(beta)
94:         if len(beta) == 1:
95:             break
96:     left_beta = [beta[0] for beta in beta_list]
97:     right_beta = [beta[-1] for beta in reversed(beta_list)]
98: 
99:     return left_beta, right_beta
100: 
101: 
102: # FIXME spelling mistake in the name of the parameter ``tolerence``
103: def find_bezier_t_intersecting_with_closedpath(bezier_point_at_t,
104:                                                inside_closedpath,
105:                                                t0=0., t1=1., tolerence=0.01):
106:     ''' Find a parameter t0 and t1 of the given bezier path which
107:     bounds the intersecting points with a provided closed
108:     path(*inside_closedpath*). Search starts from *t0* and *t1* and it
109:     uses a simple bisecting algorithm therefore one of the end point
110:     must be inside the path while the orther doesn't. The search stop
111:     when |t0-t1| gets smaller than the given tolerence.
112:     value for
113: 
114:     - bezier_point_at_t : a function which returns x, y coordinates at *t*
115: 
116:     - inside_closedpath : return True if the point is inside the path
117: 
118:     '''
119:     # inside_closedpath : function
120: 
121:     start = bezier_point_at_t(t0)
122:     end = bezier_point_at_t(t1)
123: 
124:     start_inside = inside_closedpath(start)
125:     end_inside = inside_closedpath(end)
126: 
127:     if start_inside == end_inside and start != end:
128:         raise NonIntersectingPathException(
129:             "Both points are on the same side of the closed path")
130: 
131:     while True:
132: 
133:         # return if the distance is smaller than the tolerence
134:         if np.hypot(start[0] - end[0], start[1] - end[1]) < tolerence:
135:             return t0, t1
136: 
137:         # calculate the middle point
138:         middle_t = 0.5 * (t0 + t1)
139:         middle = bezier_point_at_t(middle_t)
140:         middle_inside = inside_closedpath(middle)
141: 
142:         if xor(start_inside, middle_inside):
143:             t1 = middle_t
144:             end = middle
145:             end_inside = middle_inside
146:         else:
147:             t0 = middle_t
148:             start = middle
149:             start_inside = middle_inside
150: 
151: 
152: class BezierSegment(object):
153:     '''
154:     A simple class of a 2-dimensional bezier segment
155:     '''
156: 
157:     # Higher order bezier lines can be supported by simplying adding
158:     # corresponding values.
159:     _binom_coeff = {1: np.array([1., 1.]),
160:                     2: np.array([1., 2., 1.]),
161:                     3: np.array([1., 3., 3., 1.])}
162: 
163:     def __init__(self, control_points):
164:         '''
165:         *control_points* : location of contol points. It needs have a
166:          shpae of n * 2, where n is the order of the bezier line. 1<=
167:          n <= 3 is supported.
168:         '''
169:         _o = len(control_points)
170:         self._orders = np.arange(_o)
171:         _coeff = BezierSegment._binom_coeff[_o - 1]
172: 
173:         _control_points = np.asarray(control_points)
174:         xx = _control_points[:, 0]
175:         yy = _control_points[:, 1]
176: 
177:         self._px = xx * _coeff
178:         self._py = yy * _coeff
179: 
180:     def point_at_t(self, t):
181:         "evaluate a point at t"
182:         one_minus_t_powers = np.power(1. - t, self._orders)[::-1]
183:         t_powers = np.power(t, self._orders)
184: 
185:         tt = one_minus_t_powers * t_powers
186:         _x = sum(tt * self._px)
187:         _y = sum(tt * self._py)
188: 
189:         return _x, _y
190: 
191: 
192: def split_bezier_intersecting_with_closedpath(bezier,
193:                                               inside_closedpath,
194:                                               tolerence=0.01):
195: 
196:     '''
197:     bezier : control points of the bezier segment
198:     inside_closedpath : a function which returns true if the point is inside
199:                         the path
200:     '''
201: 
202:     bz = BezierSegment(bezier)
203:     bezier_point_at_t = bz.point_at_t
204: 
205:     t0, t1 = find_bezier_t_intersecting_with_closedpath(bezier_point_at_t,
206:                                                         inside_closedpath,
207:                                                         tolerence=tolerence)
208: 
209:     _left, _right = split_de_casteljau(bezier, (t0 + t1) / 2.)
210:     return _left, _right
211: 
212: 
213: def find_r_to_boundary_of_closedpath(inside_closedpath, xy,
214:                                      cos_t, sin_t,
215:                                      rmin=0., rmax=1., tolerence=0.01):
216:     '''
217:     Find a radius r (centered at *xy*) between *rmin* and *rmax* at
218:     which it intersect with the path.
219: 
220:     inside_closedpath : function
221:     cx, cy : center
222:     cos_t, sin_t : cosine and sine for the angle
223:     rmin, rmax :
224:     '''
225: 
226:     cx, cy = xy
227: 
228:     def _f(r):
229:         return cos_t * r + cx, sin_t * r + cy
230: 
231:     find_bezier_t_intersecting_with_closedpath(_f, inside_closedpath,
232:                                                t0=rmin, t1=rmax,
233:                                                tolerence=tolerence)
234: 
235: # matplotlib specific
236: 
237: 
238: def split_path_inout(path, inside, tolerence=0.01, reorder_inout=False):
239:     ''' divide a path into two segment at the point where inside(x, y)
240:     becomes False.
241:     '''
242: 
243:     path_iter = path.iter_segments()
244: 
245:     ctl_points, command = next(path_iter)
246:     begin_inside = inside(ctl_points[-2:])  # true if begin point is inside
247: 
248:     bezier_path = None
249:     ctl_points_old = ctl_points
250: 
251:     concat = np.concatenate
252: 
253:     iold = 0
254:     i = 1
255: 
256:     for ctl_points, command in path_iter:
257:         iold = i
258:         i += len(ctl_points) // 2
259:         if inside(ctl_points[-2:]) != begin_inside:
260:             bezier_path = concat([ctl_points_old[-2:], ctl_points])
261:             break
262: 
263:         ctl_points_old = ctl_points
264: 
265:     if bezier_path is None:
266:         raise ValueError("The path does not seem to intersect with the patch")
267: 
268:     bp = list(zip(bezier_path[::2], bezier_path[1::2]))
269:     left, right = split_bezier_intersecting_with_closedpath(bp,
270:                                                             inside,
271:                                                             tolerence)
272:     if len(left) == 2:
273:         codes_left = [Path.LINETO]
274:         codes_right = [Path.MOVETO, Path.LINETO]
275:     elif len(left) == 3:
276:         codes_left = [Path.CURVE3, Path.CURVE3]
277:         codes_right = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
278:     elif len(left) == 4:
279:         codes_left = [Path.CURVE4, Path.CURVE4, Path.CURVE4]
280:         codes_right = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
281:     else:
282:         raise ValueError()
283: 
284:     verts_left = left[1:]
285:     verts_right = right[:]
286: 
287:     if path.codes is None:
288:         path_in = Path(concat([path.vertices[:i], verts_left]))
289:         path_out = Path(concat([verts_right, path.vertices[i:]]))
290: 
291:     else:
292:         path_in = Path(concat([path.vertices[:iold], verts_left]),
293:                        concat([path.codes[:iold], codes_left]))
294: 
295:         path_out = Path(concat([verts_right, path.vertices[i:]]),
296:                         concat([codes_right, path.codes[i:]]))
297: 
298:     if reorder_inout and begin_inside is False:
299:         path_in, path_out = path_out, path_in
300: 
301:     return path_in, path_out
302: 
303: 
304: def inside_circle(cx, cy, r):
305:     r2 = r ** 2
306: 
307:     def _f(xy):
308:         x, y = xy
309:         return (x - cx) ** 2 + (y - cy) ** 2 < r2
310:     return _f
311: 
312: 
313: # quadratic bezier lines
314: 
315: def get_cos_sin(x0, y0, x1, y1):
316:     dx, dy = x1 - x0, y1 - y0
317:     d = (dx * dx + dy * dy) ** .5
318:     # Account for divide by zero
319:     if d == 0:
320:         return 0.0, 0.0
321:     return dx / d, dy / d
322: 
323: 
324: def check_if_parallel(dx1, dy1, dx2, dy2, tolerence=1.e-5):
325:     ''' returns
326:        * 1 if two lines are parralel in same direction
327:        * -1 if two lines are parralel in opposite direction
328:        * 0 otherwise
329:     '''
330:     theta1 = np.arctan2(dx1, dy1)
331:     theta2 = np.arctan2(dx2, dy2)
332:     dtheta = np.abs(theta1 - theta2)
333:     if dtheta < tolerence:
334:         return 1
335:     elif np.abs(dtheta - np.pi) < tolerence:
336:         return -1
337:     else:
338:         return False
339: 
340: 
341: def get_parallels(bezier2, width):
342:     '''
343:     Given the quadratic bezier control points *bezier2*, returns
344:     control points of quadratic bezier lines roughly parallel to given
345:     one separated by *width*.
346:     '''
347: 
348:     # The parallel bezier lines are constructed by following ways.
349:     #  c1 and  c2 are contol points representing the begin and end of the
350:     #  bezier line.
351:     #  cm is the middle point
352: 
353:     c1x, c1y = bezier2[0]
354:     cmx, cmy = bezier2[1]
355:     c2x, c2y = bezier2[2]
356: 
357:     parallel_test = check_if_parallel(c1x - cmx, c1y - cmy,
358:                                       cmx - c2x, cmy - c2y)
359: 
360:     if parallel_test == -1:
361:         warnings.warn(
362:             "Lines do not intersect. A straight line is used instead.")
363:         cos_t1, sin_t1 = get_cos_sin(c1x, c1y, c2x, c2y)
364:         cos_t2, sin_t2 = cos_t1, sin_t1
365:     else:
366:         # t1 and t2 is the angle between c1 and cm, cm, c2.  They are
367:         # also a angle of the tangential line of the path at c1 and c2
368:         cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
369:         cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c2x, c2y)
370: 
371:     # find c1_left, c1_right which are located along the lines
372:     # throught c1 and perpendicular to the tangential lines of the
373:     # bezier path at a distance of width. Same thing for c2_left and
374:     # c2_right with respect to c2.
375:     c1x_left, c1y_left, c1x_right, c1y_right = (
376:         get_normal_points(c1x, c1y, cos_t1, sin_t1, width)
377:     )
378:     c2x_left, c2y_left, c2x_right, c2y_right = (
379:         get_normal_points(c2x, c2y, cos_t2, sin_t2, width)
380:     )
381: 
382:     # find cm_left which is the intersectng point of a line through
383:     # c1_left with angle t1 and a line throught c2_left with angle
384:     # t2. Same with cm_right.
385:     if parallel_test != 0:
386:         # a special case for a straight line, i.e., angle between two
387:         # lines are smaller than some (arbitrtay) value.
388:         cmx_left, cmy_left = (
389:             0.5 * (c1x_left + c2x_left), 0.5 * (c1y_left + c2y_left)
390:         )
391:         cmx_right, cmy_right = (
392:             0.5 * (c1x_right + c2x_right), 0.5 * (c1y_right + c2y_right)
393:         )
394:     else:
395:         cmx_left, cmy_left = get_intersection(c1x_left, c1y_left, cos_t1,
396:                                               sin_t1, c2x_left, c2y_left,
397:                                               cos_t2, sin_t2)
398: 
399:         cmx_right, cmy_right = get_intersection(c1x_right, c1y_right, cos_t1,
400:                                                 sin_t1, c2x_right, c2y_right,
401:                                                 cos_t2, sin_t2)
402: 
403:     # the parralel bezier lines are created with control points of
404:     # [c1_left, cm_left, c2_left] and [c1_right, cm_right, c2_right]
405:     path_left = [(c1x_left, c1y_left),
406:                  (cmx_left, cmy_left),
407:                  (c2x_left, c2y_left)]
408:     path_right = [(c1x_right, c1y_right),
409:                   (cmx_right, cmy_right),
410:                   (c2x_right, c2y_right)]
411: 
412:     return path_left, path_right
413: 
414: 
415: def find_control_points(c1x, c1y, mmx, mmy, c2x, c2y):
416:     ''' Find control points of the bezier line throught c1, mm, c2. We
417:     simply assume that c1, mm, c2 which have parametric value 0, 0.5, and 1.
418:     '''
419: 
420:     cmx = .5 * (4 * mmx - (c1x + c2x))
421:     cmy = .5 * (4 * mmy - (c1y + c2y))
422: 
423:     return [(c1x, c1y), (cmx, cmy), (c2x, c2y)]
424: 
425: 
426: def make_wedged_bezier2(bezier2, width, w1=1., wm=0.5, w2=0.):
427:     '''
428:     Being similar to get_parallels, returns control points of two quadrativ
429:     bezier lines having a width roughly parralel to given one separated by
430:     *width*.
431:     '''
432: 
433:     # c1, cm, c2
434:     c1x, c1y = bezier2[0]
435:     cmx, cmy = bezier2[1]
436:     c3x, c3y = bezier2[2]
437: 
438:     # t1 and t2 is the anlge between c1 and cm, cm, c3.
439:     # They are also a angle of the tangential line of the path at c1 and c3
440:     cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
441:     cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c3x, c3y)
442: 
443:     # find c1_left, c1_right which are located along the lines
444:     # throught c1 and perpendicular to the tangential lines of the
445:     # bezier path at a distance of width. Same thing for c3_left and
446:     # c3_right with respect to c3.
447:     c1x_left, c1y_left, c1x_right, c1y_right = (
448:         get_normal_points(c1x, c1y, cos_t1, sin_t1, width * w1)
449:     )
450:     c3x_left, c3y_left, c3x_right, c3y_right = (
451:         get_normal_points(c3x, c3y, cos_t2, sin_t2, width * w2)
452:     )
453: 
454:     # find c12, c23 and c123 which are middle points of c1-cm, cm-c3 and
455:     # c12-c23
456:     c12x, c12y = (c1x + cmx) * .5, (c1y + cmy) * .5
457:     c23x, c23y = (cmx + c3x) * .5, (cmy + c3y) * .5
458:     c123x, c123y = (c12x + c23x) * .5, (c12y + c23y) * .5
459: 
460:     # tangential angle of c123 (angle between c12 and c23)
461:     cos_t123, sin_t123 = get_cos_sin(c12x, c12y, c23x, c23y)
462: 
463:     c123x_left, c123y_left, c123x_right, c123y_right = (
464:         get_normal_points(c123x, c123y, cos_t123, sin_t123, width * wm)
465:     )
466: 
467:     path_left = find_control_points(c1x_left, c1y_left,
468:                                     c123x_left, c123y_left,
469:                                     c3x_left, c3y_left)
470:     path_right = find_control_points(c1x_right, c1y_right,
471:                                      c123x_right, c123y_right,
472:                                      c3x_right, c3y_right)
473: 
474:     return path_left, path_right
475: 
476: 
477: def make_path_regular(p):
478:     '''
479:     fill in the codes if None.
480:     '''
481:     c = p.codes
482:     if c is None:
483:         c = np.empty(p.vertices.shape[:1], "i")
484:         c.fill(Path.LINETO)
485:         c[0] = Path.MOVETO
486: 
487:         return Path(p.vertices, c)
488:     else:
489:         return p
490: 
491: 
492: def concatenate_paths(paths):
493:     '''
494:     concatenate list of paths into a single path.
495:     '''
496: 
497:     vertices = []
498:     codes = []
499:     for p in paths:
500:         p = make_path_regular(p)
501:         vertices.append(p.vertices)
502:         codes.append(p.codes)
503: 
504:     _path = Path(np.concatenate(vertices),
505:                  np.concatenate(codes))
506:     return _path
507: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_22971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nA module providing some utility functions regarding bezier path manipulation.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import six' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_22972 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_22972) is not StypyTypeError):

    if (import_22972 != 'pyd_module'):
        __import__(import_22972)
        sys_modules_22973 = sys.modules[import_22972]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_22973.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_22972)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_22974 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_22974) is not StypyTypeError):

    if (import_22974 != 'pyd_module'):
        __import__(import_22974)
        sys_modules_22975 = sys.modules[import_22974]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_22975.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_22974)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from matplotlib.path import Path' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_22976 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.path')

if (type(import_22976) is not StypyTypeError):

    if (import_22976 != 'pyd_module'):
        __import__(import_22976)
        sys_modules_22977 = sys.modules[import_22976]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.path', sys_modules_22977.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_22977, sys_modules_22977.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.path', import_22976)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from operator import xor' statement (line 13)
try:
    from operator import xor

except:
    xor = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'operator', None, module_type_store, ['xor'], [xor])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import warnings' statement (line 14)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'warnings', warnings, module_type_store)

# Declaration of the 'NonIntersectingPathException' class
# Getting the type of 'ValueError' (line 17)
ValueError_22978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 35), 'ValueError')

class NonIntersectingPathException(ValueError_22978, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 0, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonIntersectingPathException.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NonIntersectingPathException' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'NonIntersectingPathException', NonIntersectingPathException)

@norecursion
def get_intersection(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_intersection'
    module_type_store = module_type_store.open_function_context('get_intersection', 23, 0, False)
    
    # Passed parameters checking function
    get_intersection.stypy_localization = localization
    get_intersection.stypy_type_of_self = None
    get_intersection.stypy_type_store = module_type_store
    get_intersection.stypy_function_name = 'get_intersection'
    get_intersection.stypy_param_names_list = ['cx1', 'cy1', 'cos_t1', 'sin_t1', 'cx2', 'cy2', 'cos_t2', 'sin_t2']
    get_intersection.stypy_varargs_param_name = None
    get_intersection.stypy_kwargs_param_name = None
    get_intersection.stypy_call_defaults = defaults
    get_intersection.stypy_call_varargs = varargs
    get_intersection.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_intersection', ['cx1', 'cy1', 'cos_t1', 'sin_t1', 'cx2', 'cy2', 'cos_t2', 'sin_t2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_intersection', localization, ['cx1', 'cy1', 'cos_t1', 'sin_t1', 'cx2', 'cy2', 'cos_t2', 'sin_t2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_intersection(...)' code ##################

    unicode_22979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'unicode', u' return a intersecting point between a line through (cx1, cy1)\n    and having angle t1 and a line through (cx2, cy2) and angle t2.\n    ')
    
    # Assigning a BinOp to a Name (line 32):
    
    # Assigning a BinOp to a Name (line 32):
    # Getting the type of 'sin_t1' (line 32)
    sin_t1_22980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'sin_t1')
    # Getting the type of 'cx1' (line 32)
    cx1_22981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'cx1')
    # Applying the binary operator '*' (line 32)
    result_mul_22982 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 16), '*', sin_t1_22980, cx1_22981)
    
    # Getting the type of 'cos_t1' (line 32)
    cos_t1_22983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'cos_t1')
    # Getting the type of 'cy1' (line 32)
    cy1_22984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'cy1')
    # Applying the binary operator '*' (line 32)
    result_mul_22985 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 31), '*', cos_t1_22983, cy1_22984)
    
    # Applying the binary operator '-' (line 32)
    result_sub_22986 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 16), '-', result_mul_22982, result_mul_22985)
    
    # Assigning a type to the variable 'line1_rhs' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'line1_rhs', result_sub_22986)
    
    # Assigning a BinOp to a Name (line 33):
    
    # Assigning a BinOp to a Name (line 33):
    # Getting the type of 'sin_t2' (line 33)
    sin_t2_22987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'sin_t2')
    # Getting the type of 'cx2' (line 33)
    cx2_22988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'cx2')
    # Applying the binary operator '*' (line 33)
    result_mul_22989 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 16), '*', sin_t2_22987, cx2_22988)
    
    # Getting the type of 'cos_t2' (line 33)
    cos_t2_22990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'cos_t2')
    # Getting the type of 'cy2' (line 33)
    cy2_22991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'cy2')
    # Applying the binary operator '*' (line 33)
    result_mul_22992 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 31), '*', cos_t2_22990, cy2_22991)
    
    # Applying the binary operator '-' (line 33)
    result_sub_22993 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 16), '-', result_mul_22989, result_mul_22992)
    
    # Assigning a type to the variable 'line2_rhs' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'line2_rhs', result_sub_22993)
    
    # Assigning a Tuple to a Tuple (line 36):
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'sin_t1' (line 36)
    sin_t1_22994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'sin_t1')
    # Assigning a type to the variable 'tuple_assignment_22858' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_assignment_22858', sin_t1_22994)
    
    # Assigning a UnaryOp to a Name (line 36):
    
    # Getting the type of 'cos_t1' (line 36)
    cos_t1_22995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'cos_t1')
    # Applying the 'usub' unary operator (line 36)
    result___neg___22996 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 19), 'usub', cos_t1_22995)
    
    # Assigning a type to the variable 'tuple_assignment_22859' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_assignment_22859', result___neg___22996)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_assignment_22858' (line 36)
    tuple_assignment_22858_22997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_assignment_22858')
    # Assigning a type to the variable 'a' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'a', tuple_assignment_22858_22997)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_assignment_22859' (line 36)
    tuple_assignment_22859_22998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_assignment_22859')
    # Assigning a type to the variable 'b' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 7), 'b', tuple_assignment_22859_22998)
    
    # Assigning a Tuple to a Tuple (line 37):
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'sin_t2' (line 37)
    sin_t2_22999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'sin_t2')
    # Assigning a type to the variable 'tuple_assignment_22860' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_assignment_22860', sin_t2_22999)
    
    # Assigning a UnaryOp to a Name (line 37):
    
    # Getting the type of 'cos_t2' (line 37)
    cos_t2_23000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'cos_t2')
    # Applying the 'usub' unary operator (line 37)
    result___neg___23001 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 19), 'usub', cos_t2_23000)
    
    # Assigning a type to the variable 'tuple_assignment_22861' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_assignment_22861', result___neg___23001)
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'tuple_assignment_22860' (line 37)
    tuple_assignment_22860_23002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_assignment_22860')
    # Assigning a type to the variable 'c' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'c', tuple_assignment_22860_23002)
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'tuple_assignment_22861' (line 37)
    tuple_assignment_22861_23003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_assignment_22861')
    # Assigning a type to the variable 'd' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'd', tuple_assignment_22861_23003)
    
    # Assigning a BinOp to a Name (line 39):
    
    # Assigning a BinOp to a Name (line 39):
    # Getting the type of 'a' (line 39)
    a_23004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'a')
    # Getting the type of 'd' (line 39)
    d_23005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'd')
    # Applying the binary operator '*' (line 39)
    result_mul_23006 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 12), '*', a_23004, d_23005)
    
    # Getting the type of 'b' (line 39)
    b_23007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'b')
    # Getting the type of 'c' (line 39)
    c_23008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'c')
    # Applying the binary operator '*' (line 39)
    result_mul_23009 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 20), '*', b_23007, c_23008)
    
    # Applying the binary operator '-' (line 39)
    result_sub_23010 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 12), '-', result_mul_23006, result_mul_23009)
    
    # Assigning a type to the variable 'ad_bc' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'ad_bc', result_sub_23010)
    
    
    # Getting the type of 'ad_bc' (line 40)
    ad_bc_23011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 7), 'ad_bc')
    float_23012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'float')
    # Applying the binary operator '==' (line 40)
    result_eq_23013 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 7), '==', ad_bc_23011, float_23012)
    
    # Testing the type of an if condition (line 40)
    if_condition_23014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 4), result_eq_23013)
    # Assigning a type to the variable 'if_condition_23014' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'if_condition_23014', if_condition_23014)
    # SSA begins for if statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 41)
    # Processing the call arguments (line 41)
    unicode_23016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'unicode', u'Given lines do not intersect')
    # Processing the call keyword arguments (line 41)
    kwargs_23017 = {}
    # Getting the type of 'ValueError' (line 41)
    ValueError_23015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 41)
    ValueError_call_result_23018 = invoke(stypy.reporting.localization.Localization(__file__, 41, 14), ValueError_23015, *[unicode_23016], **kwargs_23017)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 41, 8), ValueError_call_result_23018, 'raise parameter', BaseException)
    # SSA join for if statement (line 40)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 44):
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'd' (line 44)
    d_23019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'd')
    # Assigning a type to the variable 'tuple_assignment_22862' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_22862', d_23019)
    
    # Assigning a UnaryOp to a Name (line 44):
    
    # Getting the type of 'b' (line 44)
    b_23020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'b')
    # Applying the 'usub' unary operator (line 44)
    result___neg___23021 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 16), 'usub', b_23020)
    
    # Assigning a type to the variable 'tuple_assignment_22863' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_22863', result___neg___23021)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_22862' (line 44)
    tuple_assignment_22862_23022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_22862')
    # Assigning a type to the variable 'a_' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'a_', tuple_assignment_22862_23022)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_22863' (line 44)
    tuple_assignment_22863_23023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_22863')
    # Assigning a type to the variable 'b_' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'b_', tuple_assignment_22863_23023)
    
    # Assigning a Tuple to a Tuple (line 45):
    
    # Assigning a UnaryOp to a Name (line 45):
    
    # Getting the type of 'c' (line 45)
    c_23024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'c')
    # Applying the 'usub' unary operator (line 45)
    result___neg___23025 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 13), 'usub', c_23024)
    
    # Assigning a type to the variable 'tuple_assignment_22864' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'tuple_assignment_22864', result___neg___23025)
    
    # Assigning a Name to a Name (line 45):
    # Getting the type of 'a' (line 45)
    a_23026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'a')
    # Assigning a type to the variable 'tuple_assignment_22865' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'tuple_assignment_22865', a_23026)
    
    # Assigning a Name to a Name (line 45):
    # Getting the type of 'tuple_assignment_22864' (line 45)
    tuple_assignment_22864_23027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'tuple_assignment_22864')
    # Assigning a type to the variable 'c_' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'c_', tuple_assignment_22864_23027)
    
    # Assigning a Name to a Name (line 45):
    # Getting the type of 'tuple_assignment_22865' (line 45)
    tuple_assignment_22865_23028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'tuple_assignment_22865')
    # Assigning a type to the variable 'd_' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'd_', tuple_assignment_22865_23028)
    
    # Assigning a ListComp to a Tuple (line 46):
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_23029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_23033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'a_' (line 46)
    a__23034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'a_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23033, a__23034)
    # Adding element type (line 46)
    # Getting the type of 'b_' (line 46)
    b__23035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'b_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23033, b__23035)
    # Adding element type (line 46)
    # Getting the type of 'c_' (line 46)
    c__23036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 50), 'c_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23033, c__23036)
    # Adding element type (line 46)
    # Getting the type of 'd_' (line 46)
    d__23037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 'd_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23033, d__23037)
    
    comprehension_23038 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_23033)
    # Assigning a type to the variable 'k' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'k', comprehension_23038)
    # Getting the type of 'k' (line 46)
    k_23030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'k')
    # Getting the type of 'ad_bc' (line 46)
    ad_bc_23031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'ad_bc')
    # Applying the binary operator 'div' (line 46)
    result_div_23032 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 22), 'div', k_23030, ad_bc_23031)
    
    list_23039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_23039, result_div_23032)
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___23040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), list_23039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_23041 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___23040, int_23029)
    
    # Assigning a type to the variable 'tuple_var_assignment_22866' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_22866', subscript_call_result_23041)
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_23042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_23046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'a_' (line 46)
    a__23047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'a_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23046, a__23047)
    # Adding element type (line 46)
    # Getting the type of 'b_' (line 46)
    b__23048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'b_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23046, b__23048)
    # Adding element type (line 46)
    # Getting the type of 'c_' (line 46)
    c__23049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 50), 'c_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23046, c__23049)
    # Adding element type (line 46)
    # Getting the type of 'd_' (line 46)
    d__23050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 'd_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23046, d__23050)
    
    comprehension_23051 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_23046)
    # Assigning a type to the variable 'k' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'k', comprehension_23051)
    # Getting the type of 'k' (line 46)
    k_23043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'k')
    # Getting the type of 'ad_bc' (line 46)
    ad_bc_23044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'ad_bc')
    # Applying the binary operator 'div' (line 46)
    result_div_23045 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 22), 'div', k_23043, ad_bc_23044)
    
    list_23052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_23052, result_div_23045)
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___23053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), list_23052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_23054 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___23053, int_23042)
    
    # Assigning a type to the variable 'tuple_var_assignment_22867' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_22867', subscript_call_result_23054)
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_23055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_23059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'a_' (line 46)
    a__23060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'a_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23059, a__23060)
    # Adding element type (line 46)
    # Getting the type of 'b_' (line 46)
    b__23061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'b_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23059, b__23061)
    # Adding element type (line 46)
    # Getting the type of 'c_' (line 46)
    c__23062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 50), 'c_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23059, c__23062)
    # Adding element type (line 46)
    # Getting the type of 'd_' (line 46)
    d__23063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 'd_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23059, d__23063)
    
    comprehension_23064 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_23059)
    # Assigning a type to the variable 'k' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'k', comprehension_23064)
    # Getting the type of 'k' (line 46)
    k_23056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'k')
    # Getting the type of 'ad_bc' (line 46)
    ad_bc_23057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'ad_bc')
    # Applying the binary operator 'div' (line 46)
    result_div_23058 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 22), 'div', k_23056, ad_bc_23057)
    
    list_23065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_23065, result_div_23058)
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___23066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), list_23065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_23067 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___23066, int_23055)
    
    # Assigning a type to the variable 'tuple_var_assignment_22868' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_22868', subscript_call_result_23067)
    
    # Assigning a Subscript to a Name (line 46):
    
    # Obtaining the type of the subscript
    int_23068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_23072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'a_' (line 46)
    a__23073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'a_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23072, a__23073)
    # Adding element type (line 46)
    # Getting the type of 'b_' (line 46)
    b__23074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'b_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23072, b__23074)
    # Adding element type (line 46)
    # Getting the type of 'c_' (line 46)
    c__23075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 50), 'c_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23072, c__23075)
    # Adding element type (line 46)
    # Getting the type of 'd_' (line 46)
    d__23076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 'd_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 41), list_23072, d__23076)
    
    comprehension_23077 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_23072)
    # Assigning a type to the variable 'k' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'k', comprehension_23077)
    # Getting the type of 'k' (line 46)
    k_23069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'k')
    # Getting the type of 'ad_bc' (line 46)
    ad_bc_23070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'ad_bc')
    # Applying the binary operator 'div' (line 46)
    result_div_23071 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 22), 'div', k_23069, ad_bc_23070)
    
    list_23078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_23078, result_div_23071)
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___23079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), list_23078, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_23080 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), getitem___23079, int_23068)
    
    # Assigning a type to the variable 'tuple_var_assignment_22869' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_22869', subscript_call_result_23080)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_22866' (line 46)
    tuple_var_assignment_22866_23081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_22866')
    # Assigning a type to the variable 'a_' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'a_', tuple_var_assignment_22866_23081)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_22867' (line 46)
    tuple_var_assignment_22867_23082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_22867')
    # Assigning a type to the variable 'b_' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'b_', tuple_var_assignment_22867_23082)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_22868' (line 46)
    tuple_var_assignment_22868_23083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_22868')
    # Assigning a type to the variable 'c_' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'c_', tuple_var_assignment_22868_23083)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_var_assignment_22869' (line 46)
    tuple_var_assignment_22869_23084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_var_assignment_22869')
    # Assigning a type to the variable 'd_' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'd_', tuple_var_assignment_22869_23084)
    
    # Assigning a BinOp to a Name (line 48):
    
    # Assigning a BinOp to a Name (line 48):
    # Getting the type of 'a_' (line 48)
    a__23085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'a_')
    # Getting the type of 'line1_rhs' (line 48)
    line1_rhs_23086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 13), 'line1_rhs')
    # Applying the binary operator '*' (line 48)
    result_mul_23087 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 8), '*', a__23085, line1_rhs_23086)
    
    # Getting the type of 'b_' (line 48)
    b__23088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'b_')
    # Getting the type of 'line2_rhs' (line 48)
    line2_rhs_23089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'line2_rhs')
    # Applying the binary operator '*' (line 48)
    result_mul_23090 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 25), '*', b__23088, line2_rhs_23089)
    
    # Applying the binary operator '+' (line 48)
    result_add_23091 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 8), '+', result_mul_23087, result_mul_23090)
    
    # Assigning a type to the variable 'x' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'x', result_add_23091)
    
    # Assigning a BinOp to a Name (line 49):
    
    # Assigning a BinOp to a Name (line 49):
    # Getting the type of 'c_' (line 49)
    c__23092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'c_')
    # Getting the type of 'line1_rhs' (line 49)
    line1_rhs_23093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'line1_rhs')
    # Applying the binary operator '*' (line 49)
    result_mul_23094 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 8), '*', c__23092, line1_rhs_23093)
    
    # Getting the type of 'd_' (line 49)
    d__23095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'd_')
    # Getting the type of 'line2_rhs' (line 49)
    line2_rhs_23096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'line2_rhs')
    # Applying the binary operator '*' (line 49)
    result_mul_23097 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 25), '*', d__23095, line2_rhs_23096)
    
    # Applying the binary operator '+' (line 49)
    result_add_23098 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 8), '+', result_mul_23094, result_mul_23097)
    
    # Assigning a type to the variable 'y' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'y', result_add_23098)
    
    # Obtaining an instance of the builtin type 'tuple' (line 51)
    tuple_23099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 51)
    # Adding element type (line 51)
    # Getting the type of 'x' (line 51)
    x_23100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), tuple_23099, x_23100)
    # Adding element type (line 51)
    # Getting the type of 'y' (line 51)
    y_23101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), tuple_23099, y_23101)
    
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type', tuple_23099)
    
    # ################# End of 'get_intersection(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_intersection' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_23102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23102)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_intersection'
    return stypy_return_type_23102

# Assigning a type to the variable 'get_intersection' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'get_intersection', get_intersection)

@norecursion
def get_normal_points(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_normal_points'
    module_type_store = module_type_store.open_function_context('get_normal_points', 54, 0, False)
    
    # Passed parameters checking function
    get_normal_points.stypy_localization = localization
    get_normal_points.stypy_type_of_self = None
    get_normal_points.stypy_type_store = module_type_store
    get_normal_points.stypy_function_name = 'get_normal_points'
    get_normal_points.stypy_param_names_list = ['cx', 'cy', 'cos_t', 'sin_t', 'length']
    get_normal_points.stypy_varargs_param_name = None
    get_normal_points.stypy_kwargs_param_name = None
    get_normal_points.stypy_call_defaults = defaults
    get_normal_points.stypy_call_varargs = varargs
    get_normal_points.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_normal_points', ['cx', 'cy', 'cos_t', 'sin_t', 'length'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_normal_points', localization, ['cx', 'cy', 'cos_t', 'sin_t', 'length'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_normal_points(...)' code ##################

    unicode_23103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'unicode', u'\n    For a line passing through (*cx*, *cy*) and having a angle *t*, return\n    locations of the two points located along its perpendicular line at the\n    distance of *length*.\n    ')
    
    
    # Getting the type of 'length' (line 61)
    length_23104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 'length')
    float_23105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 17), 'float')
    # Applying the binary operator '==' (line 61)
    result_eq_23106 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 7), '==', length_23104, float_23105)
    
    # Testing the type of an if condition (line 61)
    if_condition_23107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 4), result_eq_23106)
    # Assigning a type to the variable 'if_condition_23107' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'if_condition_23107', if_condition_23107)
    # SSA begins for if statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_23108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    # Getting the type of 'cx' (line 62)
    cx_23109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'cx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), tuple_23108, cx_23109)
    # Adding element type (line 62)
    # Getting the type of 'cy' (line 62)
    cy_23110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'cy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), tuple_23108, cy_23110)
    # Adding element type (line 62)
    # Getting the type of 'cx' (line 62)
    cx_23111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'cx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), tuple_23108, cx_23111)
    # Adding element type (line 62)
    # Getting the type of 'cy' (line 62)
    cy_23112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'cy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), tuple_23108, cy_23112)
    
    # Assigning a type to the variable 'stypy_return_type' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', tuple_23108)
    # SSA join for if statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 64):
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'sin_t' (line 64)
    sin_t_23113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'sin_t')
    # Assigning a type to the variable 'tuple_assignment_22870' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_assignment_22870', sin_t_23113)
    
    # Assigning a UnaryOp to a Name (line 64):
    
    # Getting the type of 'cos_t' (line 64)
    cos_t_23114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'cos_t')
    # Applying the 'usub' unary operator (line 64)
    result___neg___23115 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 28), 'usub', cos_t_23114)
    
    # Assigning a type to the variable 'tuple_assignment_22871' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_assignment_22871', result___neg___23115)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_assignment_22870' (line 64)
    tuple_assignment_22870_23116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_assignment_22870')
    # Assigning a type to the variable 'cos_t1' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'cos_t1', tuple_assignment_22870_23116)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_assignment_22871' (line 64)
    tuple_assignment_22871_23117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_assignment_22871')
    # Assigning a type to the variable 'sin_t1' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'sin_t1', tuple_assignment_22871_23117)
    
    # Assigning a Tuple to a Tuple (line 65):
    
    # Assigning a UnaryOp to a Name (line 65):
    
    # Getting the type of 'sin_t' (line 65)
    sin_t_23118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'sin_t')
    # Applying the 'usub' unary operator (line 65)
    result___neg___23119 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 21), 'usub', sin_t_23118)
    
    # Assigning a type to the variable 'tuple_assignment_22872' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_assignment_22872', result___neg___23119)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'cos_t' (line 65)
    cos_t_23120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'cos_t')
    # Assigning a type to the variable 'tuple_assignment_22873' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_assignment_22873', cos_t_23120)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_assignment_22872' (line 65)
    tuple_assignment_22872_23121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_assignment_22872')
    # Assigning a type to the variable 'cos_t2' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'cos_t2', tuple_assignment_22872_23121)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_assignment_22873' (line 65)
    tuple_assignment_22873_23122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_assignment_22873')
    # Assigning a type to the variable 'sin_t2' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'sin_t2', tuple_assignment_22873_23122)
    
    # Assigning a Tuple to a Tuple (line 67):
    
    # Assigning a BinOp to a Name (line 67):
    # Getting the type of 'length' (line 67)
    length_23123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'length')
    # Getting the type of 'cos_t1' (line 67)
    cos_t1_23124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'cos_t1')
    # Applying the binary operator '*' (line 67)
    result_mul_23125 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 13), '*', length_23123, cos_t1_23124)
    
    # Getting the type of 'cx' (line 67)
    cx_23126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'cx')
    # Applying the binary operator '+' (line 67)
    result_add_23127 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 13), '+', result_mul_23125, cx_23126)
    
    # Assigning a type to the variable 'tuple_assignment_22874' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'tuple_assignment_22874', result_add_23127)
    
    # Assigning a BinOp to a Name (line 67):
    # Getting the type of 'length' (line 67)
    length_23128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 35), 'length')
    # Getting the type of 'sin_t1' (line 67)
    sin_t1_23129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 44), 'sin_t1')
    # Applying the binary operator '*' (line 67)
    result_mul_23130 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 35), '*', length_23128, sin_t1_23129)
    
    # Getting the type of 'cy' (line 67)
    cy_23131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 53), 'cy')
    # Applying the binary operator '+' (line 67)
    result_add_23132 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 35), '+', result_mul_23130, cy_23131)
    
    # Assigning a type to the variable 'tuple_assignment_22875' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'tuple_assignment_22875', result_add_23132)
    
    # Assigning a Name to a Name (line 67):
    # Getting the type of 'tuple_assignment_22874' (line 67)
    tuple_assignment_22874_23133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'tuple_assignment_22874')
    # Assigning a type to the variable 'x1' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'x1', tuple_assignment_22874_23133)
    
    # Assigning a Name to a Name (line 67):
    # Getting the type of 'tuple_assignment_22875' (line 67)
    tuple_assignment_22875_23134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'tuple_assignment_22875')
    # Assigning a type to the variable 'y1' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'y1', tuple_assignment_22875_23134)
    
    # Assigning a Tuple to a Tuple (line 68):
    
    # Assigning a BinOp to a Name (line 68):
    # Getting the type of 'length' (line 68)
    length_23135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'length')
    # Getting the type of 'cos_t2' (line 68)
    cos_t2_23136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'cos_t2')
    # Applying the binary operator '*' (line 68)
    result_mul_23137 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 13), '*', length_23135, cos_t2_23136)
    
    # Getting the type of 'cx' (line 68)
    cx_23138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'cx')
    # Applying the binary operator '+' (line 68)
    result_add_23139 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 13), '+', result_mul_23137, cx_23138)
    
    # Assigning a type to the variable 'tuple_assignment_22876' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_assignment_22876', result_add_23139)
    
    # Assigning a BinOp to a Name (line 68):
    # Getting the type of 'length' (line 68)
    length_23140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'length')
    # Getting the type of 'sin_t2' (line 68)
    sin_t2_23141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 44), 'sin_t2')
    # Applying the binary operator '*' (line 68)
    result_mul_23142 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 35), '*', length_23140, sin_t2_23141)
    
    # Getting the type of 'cy' (line 68)
    cy_23143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 53), 'cy')
    # Applying the binary operator '+' (line 68)
    result_add_23144 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 35), '+', result_mul_23142, cy_23143)
    
    # Assigning a type to the variable 'tuple_assignment_22877' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_assignment_22877', result_add_23144)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_assignment_22876' (line 68)
    tuple_assignment_22876_23145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_assignment_22876')
    # Assigning a type to the variable 'x2' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'x2', tuple_assignment_22876_23145)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_assignment_22877' (line 68)
    tuple_assignment_22877_23146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_assignment_22877')
    # Assigning a type to the variable 'y2' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'y2', tuple_assignment_22877_23146)
    
    # Obtaining an instance of the builtin type 'tuple' (line 70)
    tuple_23147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 70)
    # Adding element type (line 70)
    # Getting the type of 'x1' (line 70)
    x1_23148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'x1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_23147, x1_23148)
    # Adding element type (line 70)
    # Getting the type of 'y1' (line 70)
    y1_23149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'y1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_23147, y1_23149)
    # Adding element type (line 70)
    # Getting the type of 'x2' (line 70)
    x2_23150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'x2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_23147, x2_23150)
    # Adding element type (line 70)
    # Getting the type of 'y2' (line 70)
    y2_23151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'y2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_23147, y2_23151)
    
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', tuple_23147)
    
    # ################# End of 'get_normal_points(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_normal_points' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_23152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_normal_points'
    return stypy_return_type_23152

# Assigning a type to the variable 'get_normal_points' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'get_normal_points', get_normal_points)

@norecursion
def _de_casteljau1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_de_casteljau1'
    module_type_store = module_type_store.open_function_context('_de_casteljau1', 79, 0, False)
    
    # Passed parameters checking function
    _de_casteljau1.stypy_localization = localization
    _de_casteljau1.stypy_type_of_self = None
    _de_casteljau1.stypy_type_store = module_type_store
    _de_casteljau1.stypy_function_name = '_de_casteljau1'
    _de_casteljau1.stypy_param_names_list = ['beta', 't']
    _de_casteljau1.stypy_varargs_param_name = None
    _de_casteljau1.stypy_kwargs_param_name = None
    _de_casteljau1.stypy_call_defaults = defaults
    _de_casteljau1.stypy_call_varargs = varargs
    _de_casteljau1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_de_casteljau1', ['beta', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_de_casteljau1', localization, ['beta', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_de_casteljau1(...)' code ##################

    
    # Assigning a BinOp to a Name (line 80):
    
    # Assigning a BinOp to a Name (line 80):
    
    # Obtaining the type of the subscript
    int_23153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'int')
    slice_23154 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 16), None, int_23153, None)
    # Getting the type of 'beta' (line 80)
    beta_23155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'beta')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___23156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), beta_23155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_23157 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), getitem___23156, slice_23154)
    
    int_23158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'int')
    # Getting the type of 't' (line 80)
    t_23159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 't')
    # Applying the binary operator '-' (line 80)
    result_sub_23160 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 29), '-', int_23158, t_23159)
    
    # Applying the binary operator '*' (line 80)
    result_mul_23161 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 16), '*', subscript_call_result_23157, result_sub_23160)
    
    
    # Obtaining the type of the subscript
    int_23162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 43), 'int')
    slice_23163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 38), int_23162, None, None)
    # Getting the type of 'beta' (line 80)
    beta_23164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 38), 'beta')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___23165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 38), beta_23164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_23166 = invoke(stypy.reporting.localization.Localization(__file__, 80, 38), getitem___23165, slice_23163)
    
    # Getting the type of 't' (line 80)
    t_23167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 49), 't')
    # Applying the binary operator '*' (line 80)
    result_mul_23168 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 38), '*', subscript_call_result_23166, t_23167)
    
    # Applying the binary operator '+' (line 80)
    result_add_23169 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 16), '+', result_mul_23161, result_mul_23168)
    
    # Assigning a type to the variable 'next_beta' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'next_beta', result_add_23169)
    # Getting the type of 'next_beta' (line 81)
    next_beta_23170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'next_beta')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', next_beta_23170)
    
    # ################# End of '_de_casteljau1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_de_casteljau1' in the type store
    # Getting the type of 'stypy_return_type' (line 79)
    stypy_return_type_23171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_de_casteljau1'
    return stypy_return_type_23171

# Assigning a type to the variable '_de_casteljau1' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), '_de_casteljau1', _de_casteljau1)

@norecursion
def split_de_casteljau(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'split_de_casteljau'
    module_type_store = module_type_store.open_function_context('split_de_casteljau', 84, 0, False)
    
    # Passed parameters checking function
    split_de_casteljau.stypy_localization = localization
    split_de_casteljau.stypy_type_of_self = None
    split_de_casteljau.stypy_type_store = module_type_store
    split_de_casteljau.stypy_function_name = 'split_de_casteljau'
    split_de_casteljau.stypy_param_names_list = ['beta', 't']
    split_de_casteljau.stypy_varargs_param_name = None
    split_de_casteljau.stypy_kwargs_param_name = None
    split_de_casteljau.stypy_call_defaults = defaults
    split_de_casteljau.stypy_call_varargs = varargs
    split_de_casteljau.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split_de_casteljau', ['beta', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split_de_casteljau', localization, ['beta', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split_de_casteljau(...)' code ##################

    unicode_23172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'unicode', u'split a bezier segment defined by its controlpoints *beta*\n    into two separate segment divided at *t* and return their control points.\n\n    ')
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to asarray(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'beta' (line 89)
    beta_23175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'beta', False)
    # Processing the call keyword arguments (line 89)
    kwargs_23176 = {}
    # Getting the type of 'np' (line 89)
    np_23173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'np', False)
    # Obtaining the member 'asarray' of a type (line 89)
    asarray_23174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), np_23173, 'asarray')
    # Calling asarray(args, kwargs) (line 89)
    asarray_call_result_23177 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), asarray_23174, *[beta_23175], **kwargs_23176)
    
    # Assigning a type to the variable 'beta' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'beta', asarray_call_result_23177)
    
    # Assigning a List to a Name (line 90):
    
    # Assigning a List to a Name (line 90):
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_23178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    # Getting the type of 'beta' (line 90)
    beta_23179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'beta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 16), list_23178, beta_23179)
    
    # Assigning a type to the variable 'beta_list' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'beta_list', list_23178)
    
    # Getting the type of 'True' (line 91)
    True_23180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 10), 'True')
    # Testing the type of an if condition (line 91)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 4), True_23180)
    # SSA begins for while statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to _de_casteljau1(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'beta' (line 92)
    beta_23182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'beta', False)
    # Getting the type of 't' (line 92)
    t_23183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 't', False)
    # Processing the call keyword arguments (line 92)
    kwargs_23184 = {}
    # Getting the type of '_de_casteljau1' (line 92)
    _de_casteljau1_23181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), '_de_casteljau1', False)
    # Calling _de_casteljau1(args, kwargs) (line 92)
    _de_casteljau1_call_result_23185 = invoke(stypy.reporting.localization.Localization(__file__, 92, 15), _de_casteljau1_23181, *[beta_23182, t_23183], **kwargs_23184)
    
    # Assigning a type to the variable 'beta' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'beta', _de_casteljau1_call_result_23185)
    
    # Call to append(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'beta' (line 93)
    beta_23188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'beta', False)
    # Processing the call keyword arguments (line 93)
    kwargs_23189 = {}
    # Getting the type of 'beta_list' (line 93)
    beta_list_23186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'beta_list', False)
    # Obtaining the member 'append' of a type (line 93)
    append_23187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), beta_list_23186, 'append')
    # Calling append(args, kwargs) (line 93)
    append_call_result_23190 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), append_23187, *[beta_23188], **kwargs_23189)
    
    
    
    
    # Call to len(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'beta' (line 94)
    beta_23192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'beta', False)
    # Processing the call keyword arguments (line 94)
    kwargs_23193 = {}
    # Getting the type of 'len' (line 94)
    len_23191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'len', False)
    # Calling len(args, kwargs) (line 94)
    len_call_result_23194 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), len_23191, *[beta_23192], **kwargs_23193)
    
    int_23195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 24), 'int')
    # Applying the binary operator '==' (line 94)
    result_eq_23196 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 11), '==', len_call_result_23194, int_23195)
    
    # Testing the type of an if condition (line 94)
    if_condition_23197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), result_eq_23196)
    # Assigning a type to the variable 'if_condition_23197' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'if_condition_23197', if_condition_23197)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 96):
    
    # Assigning a ListComp to a Name (line 96):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'beta_list' (line 96)
    beta_list_23202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'beta_list')
    comprehension_23203 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), beta_list_23202)
    # Assigning a type to the variable 'beta' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'beta', comprehension_23203)
    
    # Obtaining the type of the subscript
    int_23198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'int')
    # Getting the type of 'beta' (line 96)
    beta_23199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'beta')
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___23200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), beta_23199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_23201 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), getitem___23200, int_23198)
    
    list_23204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_23204, subscript_call_result_23201)
    # Assigning a type to the variable 'left_beta' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'left_beta', list_23204)
    
    # Assigning a ListComp to a Name (line 97):
    
    # Assigning a ListComp to a Name (line 97):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to reversed(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'beta_list' (line 97)
    beta_list_23210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 48), 'beta_list', False)
    # Processing the call keyword arguments (line 97)
    kwargs_23211 = {}
    # Getting the type of 'reversed' (line 97)
    reversed_23209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'reversed', False)
    # Calling reversed(args, kwargs) (line 97)
    reversed_call_result_23212 = invoke(stypy.reporting.localization.Localization(__file__, 97, 39), reversed_23209, *[beta_list_23210], **kwargs_23211)
    
    comprehension_23213 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 18), reversed_call_result_23212)
    # Assigning a type to the variable 'beta' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'beta', comprehension_23213)
    
    # Obtaining the type of the subscript
    int_23205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'int')
    # Getting the type of 'beta' (line 97)
    beta_23206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'beta')
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___23207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 18), beta_23206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_23208 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), getitem___23207, int_23205)
    
    list_23214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 18), list_23214, subscript_call_result_23208)
    # Assigning a type to the variable 'right_beta' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'right_beta', list_23214)
    
    # Obtaining an instance of the builtin type 'tuple' (line 99)
    tuple_23215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 99)
    # Adding element type (line 99)
    # Getting the type of 'left_beta' (line 99)
    left_beta_23216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'left_beta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 11), tuple_23215, left_beta_23216)
    # Adding element type (line 99)
    # Getting the type of 'right_beta' (line 99)
    right_beta_23217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'right_beta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 11), tuple_23215, right_beta_23217)
    
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type', tuple_23215)
    
    # ################# End of 'split_de_casteljau(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_de_casteljau' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_23218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23218)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_de_casteljau'
    return stypy_return_type_23218

# Assigning a type to the variable 'split_de_casteljau' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'split_de_casteljau', split_de_casteljau)

@norecursion
def find_bezier_t_intersecting_with_closedpath(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_23219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 50), 'float')
    float_23220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 57), 'float')
    float_23221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 71), 'float')
    defaults = [float_23219, float_23220, float_23221]
    # Create a new context for function 'find_bezier_t_intersecting_with_closedpath'
    module_type_store = module_type_store.open_function_context('find_bezier_t_intersecting_with_closedpath', 103, 0, False)
    
    # Passed parameters checking function
    find_bezier_t_intersecting_with_closedpath.stypy_localization = localization
    find_bezier_t_intersecting_with_closedpath.stypy_type_of_self = None
    find_bezier_t_intersecting_with_closedpath.stypy_type_store = module_type_store
    find_bezier_t_intersecting_with_closedpath.stypy_function_name = 'find_bezier_t_intersecting_with_closedpath'
    find_bezier_t_intersecting_with_closedpath.stypy_param_names_list = ['bezier_point_at_t', 'inside_closedpath', 't0', 't1', 'tolerence']
    find_bezier_t_intersecting_with_closedpath.stypy_varargs_param_name = None
    find_bezier_t_intersecting_with_closedpath.stypy_kwargs_param_name = None
    find_bezier_t_intersecting_with_closedpath.stypy_call_defaults = defaults
    find_bezier_t_intersecting_with_closedpath.stypy_call_varargs = varargs
    find_bezier_t_intersecting_with_closedpath.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_bezier_t_intersecting_with_closedpath', ['bezier_point_at_t', 'inside_closedpath', 't0', 't1', 'tolerence'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_bezier_t_intersecting_with_closedpath', localization, ['bezier_point_at_t', 'inside_closedpath', 't0', 't1', 'tolerence'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_bezier_t_intersecting_with_closedpath(...)' code ##################

    unicode_23222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'unicode', u" Find a parameter t0 and t1 of the given bezier path which\n    bounds the intersecting points with a provided closed\n    path(*inside_closedpath*). Search starts from *t0* and *t1* and it\n    uses a simple bisecting algorithm therefore one of the end point\n    must be inside the path while the orther doesn't. The search stop\n    when |t0-t1| gets smaller than the given tolerence.\n    value for\n\n    - bezier_point_at_t : a function which returns x, y coordinates at *t*\n\n    - inside_closedpath : return True if the point is inside the path\n\n    ")
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to bezier_point_at_t(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 't0' (line 121)
    t0_23224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 't0', False)
    # Processing the call keyword arguments (line 121)
    kwargs_23225 = {}
    # Getting the type of 'bezier_point_at_t' (line 121)
    bezier_point_at_t_23223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'bezier_point_at_t', False)
    # Calling bezier_point_at_t(args, kwargs) (line 121)
    bezier_point_at_t_call_result_23226 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), bezier_point_at_t_23223, *[t0_23224], **kwargs_23225)
    
    # Assigning a type to the variable 'start' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'start', bezier_point_at_t_call_result_23226)
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to bezier_point_at_t(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 't1' (line 122)
    t1_23228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 't1', False)
    # Processing the call keyword arguments (line 122)
    kwargs_23229 = {}
    # Getting the type of 'bezier_point_at_t' (line 122)
    bezier_point_at_t_23227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 10), 'bezier_point_at_t', False)
    # Calling bezier_point_at_t(args, kwargs) (line 122)
    bezier_point_at_t_call_result_23230 = invoke(stypy.reporting.localization.Localization(__file__, 122, 10), bezier_point_at_t_23227, *[t1_23228], **kwargs_23229)
    
    # Assigning a type to the variable 'end' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'end', bezier_point_at_t_call_result_23230)
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to inside_closedpath(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'start' (line 124)
    start_23232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'start', False)
    # Processing the call keyword arguments (line 124)
    kwargs_23233 = {}
    # Getting the type of 'inside_closedpath' (line 124)
    inside_closedpath_23231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'inside_closedpath', False)
    # Calling inside_closedpath(args, kwargs) (line 124)
    inside_closedpath_call_result_23234 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), inside_closedpath_23231, *[start_23232], **kwargs_23233)
    
    # Assigning a type to the variable 'start_inside' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'start_inside', inside_closedpath_call_result_23234)
    
    # Assigning a Call to a Name (line 125):
    
    # Assigning a Call to a Name (line 125):
    
    # Call to inside_closedpath(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'end' (line 125)
    end_23236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'end', False)
    # Processing the call keyword arguments (line 125)
    kwargs_23237 = {}
    # Getting the type of 'inside_closedpath' (line 125)
    inside_closedpath_23235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'inside_closedpath', False)
    # Calling inside_closedpath(args, kwargs) (line 125)
    inside_closedpath_call_result_23238 = invoke(stypy.reporting.localization.Localization(__file__, 125, 17), inside_closedpath_23235, *[end_23236], **kwargs_23237)
    
    # Assigning a type to the variable 'end_inside' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'end_inside', inside_closedpath_call_result_23238)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'start_inside' (line 127)
    start_inside_23239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 7), 'start_inside')
    # Getting the type of 'end_inside' (line 127)
    end_inside_23240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'end_inside')
    # Applying the binary operator '==' (line 127)
    result_eq_23241 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 7), '==', start_inside_23239, end_inside_23240)
    
    
    # Getting the type of 'start' (line 127)
    start_23242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 38), 'start')
    # Getting the type of 'end' (line 127)
    end_23243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'end')
    # Applying the binary operator '!=' (line 127)
    result_ne_23244 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 38), '!=', start_23242, end_23243)
    
    # Applying the binary operator 'and' (line 127)
    result_and_keyword_23245 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 7), 'and', result_eq_23241, result_ne_23244)
    
    # Testing the type of an if condition (line 127)
    if_condition_23246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 4), result_and_keyword_23245)
    # Assigning a type to the variable 'if_condition_23246' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'if_condition_23246', if_condition_23246)
    # SSA begins for if statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NonIntersectingPathException(...): (line 128)
    # Processing the call arguments (line 128)
    unicode_23248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 12), 'unicode', u'Both points are on the same side of the closed path')
    # Processing the call keyword arguments (line 128)
    kwargs_23249 = {}
    # Getting the type of 'NonIntersectingPathException' (line 128)
    NonIntersectingPathException_23247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'NonIntersectingPathException', False)
    # Calling NonIntersectingPathException(args, kwargs) (line 128)
    NonIntersectingPathException_call_result_23250 = invoke(stypy.reporting.localization.Localization(__file__, 128, 14), NonIntersectingPathException_23247, *[unicode_23248], **kwargs_23249)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 128, 8), NonIntersectingPathException_call_result_23250, 'raise parameter', BaseException)
    # SSA join for if statement (line 127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'True' (line 131)
    True_23251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 10), 'True')
    # Testing the type of an if condition (line 131)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 4), True_23251)
    # SSA begins for while statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    
    # Call to hypot(...): (line 134)
    # Processing the call arguments (line 134)
    
    # Obtaining the type of the subscript
    int_23254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 26), 'int')
    # Getting the type of 'start' (line 134)
    start_23255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'start', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___23256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 20), start_23255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_23257 = invoke(stypy.reporting.localization.Localization(__file__, 134, 20), getitem___23256, int_23254)
    
    
    # Obtaining the type of the subscript
    int_23258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 35), 'int')
    # Getting the type of 'end' (line 134)
    end_23259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'end', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___23260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 31), end_23259, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_23261 = invoke(stypy.reporting.localization.Localization(__file__, 134, 31), getitem___23260, int_23258)
    
    # Applying the binary operator '-' (line 134)
    result_sub_23262 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 20), '-', subscript_call_result_23257, subscript_call_result_23261)
    
    
    # Obtaining the type of the subscript
    int_23263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 45), 'int')
    # Getting the type of 'start' (line 134)
    start_23264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 39), 'start', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___23265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 39), start_23264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_23266 = invoke(stypy.reporting.localization.Localization(__file__, 134, 39), getitem___23265, int_23263)
    
    
    # Obtaining the type of the subscript
    int_23267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 54), 'int')
    # Getting the type of 'end' (line 134)
    end_23268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 50), 'end', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___23269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 50), end_23268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_23270 = invoke(stypy.reporting.localization.Localization(__file__, 134, 50), getitem___23269, int_23267)
    
    # Applying the binary operator '-' (line 134)
    result_sub_23271 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 39), '-', subscript_call_result_23266, subscript_call_result_23270)
    
    # Processing the call keyword arguments (line 134)
    kwargs_23272 = {}
    # Getting the type of 'np' (line 134)
    np_23252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'np', False)
    # Obtaining the member 'hypot' of a type (line 134)
    hypot_23253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), np_23252, 'hypot')
    # Calling hypot(args, kwargs) (line 134)
    hypot_call_result_23273 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), hypot_23253, *[result_sub_23262, result_sub_23271], **kwargs_23272)
    
    # Getting the type of 'tolerence' (line 134)
    tolerence_23274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 60), 'tolerence')
    # Applying the binary operator '<' (line 134)
    result_lt_23275 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), '<', hypot_call_result_23273, tolerence_23274)
    
    # Testing the type of an if condition (line 134)
    if_condition_23276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), result_lt_23275)
    # Assigning a type to the variable 'if_condition_23276' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_23276', if_condition_23276)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 135)
    tuple_23277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 135)
    # Adding element type (line 135)
    # Getting the type of 't0' (line 135)
    t0_23278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 't0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), tuple_23277, t0_23278)
    # Adding element type (line 135)
    # Getting the type of 't1' (line 135)
    t1_23279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 't1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), tuple_23277, t1_23279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'stypy_return_type', tuple_23277)
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 138):
    
    # Assigning a BinOp to a Name (line 138):
    float_23280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'float')
    # Getting the type of 't0' (line 138)
    t0_23281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 26), 't0')
    # Getting the type of 't1' (line 138)
    t1_23282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 't1')
    # Applying the binary operator '+' (line 138)
    result_add_23283 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 26), '+', t0_23281, t1_23282)
    
    # Applying the binary operator '*' (line 138)
    result_mul_23284 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 19), '*', float_23280, result_add_23283)
    
    # Assigning a type to the variable 'middle_t' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'middle_t', result_mul_23284)
    
    # Assigning a Call to a Name (line 139):
    
    # Assigning a Call to a Name (line 139):
    
    # Call to bezier_point_at_t(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'middle_t' (line 139)
    middle_t_23286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'middle_t', False)
    # Processing the call keyword arguments (line 139)
    kwargs_23287 = {}
    # Getting the type of 'bezier_point_at_t' (line 139)
    bezier_point_at_t_23285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'bezier_point_at_t', False)
    # Calling bezier_point_at_t(args, kwargs) (line 139)
    bezier_point_at_t_call_result_23288 = invoke(stypy.reporting.localization.Localization(__file__, 139, 17), bezier_point_at_t_23285, *[middle_t_23286], **kwargs_23287)
    
    # Assigning a type to the variable 'middle' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'middle', bezier_point_at_t_call_result_23288)
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to inside_closedpath(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'middle' (line 140)
    middle_23290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 42), 'middle', False)
    # Processing the call keyword arguments (line 140)
    kwargs_23291 = {}
    # Getting the type of 'inside_closedpath' (line 140)
    inside_closedpath_23289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'inside_closedpath', False)
    # Calling inside_closedpath(args, kwargs) (line 140)
    inside_closedpath_call_result_23292 = invoke(stypy.reporting.localization.Localization(__file__, 140, 24), inside_closedpath_23289, *[middle_23290], **kwargs_23291)
    
    # Assigning a type to the variable 'middle_inside' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'middle_inside', inside_closedpath_call_result_23292)
    
    
    # Call to xor(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'start_inside' (line 142)
    start_inside_23294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'start_inside', False)
    # Getting the type of 'middle_inside' (line 142)
    middle_inside_23295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'middle_inside', False)
    # Processing the call keyword arguments (line 142)
    kwargs_23296 = {}
    # Getting the type of 'xor' (line 142)
    xor_23293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'xor', False)
    # Calling xor(args, kwargs) (line 142)
    xor_call_result_23297 = invoke(stypy.reporting.localization.Localization(__file__, 142, 11), xor_23293, *[start_inside_23294, middle_inside_23295], **kwargs_23296)
    
    # Testing the type of an if condition (line 142)
    if_condition_23298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), xor_call_result_23297)
    # Assigning a type to the variable 'if_condition_23298' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_23298', if_condition_23298)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 143):
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'middle_t' (line 143)
    middle_t_23299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'middle_t')
    # Assigning a type to the variable 't1' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 't1', middle_t_23299)
    
    # Assigning a Name to a Name (line 144):
    
    # Assigning a Name to a Name (line 144):
    # Getting the type of 'middle' (line 144)
    middle_23300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'middle')
    # Assigning a type to the variable 'end' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'end', middle_23300)
    
    # Assigning a Name to a Name (line 145):
    
    # Assigning a Name to a Name (line 145):
    # Getting the type of 'middle_inside' (line 145)
    middle_inside_23301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'middle_inside')
    # Assigning a type to the variable 'end_inside' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'end_inside', middle_inside_23301)
    # SSA branch for the else part of an if statement (line 142)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 147):
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'middle_t' (line 147)
    middle_t_23302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'middle_t')
    # Assigning a type to the variable 't0' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 't0', middle_t_23302)
    
    # Assigning a Name to a Name (line 148):
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'middle' (line 148)
    middle_23303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'middle')
    # Assigning a type to the variable 'start' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'start', middle_23303)
    
    # Assigning a Name to a Name (line 149):
    
    # Assigning a Name to a Name (line 149):
    # Getting the type of 'middle_inside' (line 149)
    middle_inside_23304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'middle_inside')
    # Assigning a type to the variable 'start_inside' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'start_inside', middle_inside_23304)
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 131)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'find_bezier_t_intersecting_with_closedpath(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_bezier_t_intersecting_with_closedpath' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_23305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_bezier_t_intersecting_with_closedpath'
    return stypy_return_type_23305

# Assigning a type to the variable 'find_bezier_t_intersecting_with_closedpath' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'find_bezier_t_intersecting_with_closedpath', find_bezier_t_intersecting_with_closedpath)
# Declaration of the 'BezierSegment' class

class BezierSegment(object, ):
    unicode_23306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, (-1)), 'unicode', u'\n    A simple class of a 2-dimensional bezier segment\n    ')
    
    # Assigning a Dict to a Name (line 159):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BezierSegment.__init__', ['control_points'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['control_points'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_23307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'unicode', u'\n        *control_points* : location of contol points. It needs have a\n         shpae of n * 2, where n is the order of the bezier line. 1<=\n         n <= 3 is supported.\n        ')
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to len(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'control_points' (line 169)
        control_points_23309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'control_points', False)
        # Processing the call keyword arguments (line 169)
        kwargs_23310 = {}
        # Getting the type of 'len' (line 169)
        len_23308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'len', False)
        # Calling len(args, kwargs) (line 169)
        len_call_result_23311 = invoke(stypy.reporting.localization.Localization(__file__, 169, 13), len_23308, *[control_points_23309], **kwargs_23310)
        
        # Assigning a type to the variable '_o' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), '_o', len_call_result_23311)
        
        # Assigning a Call to a Attribute (line 170):
        
        # Assigning a Call to a Attribute (line 170):
        
        # Call to arange(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of '_o' (line 170)
        _o_23314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), '_o', False)
        # Processing the call keyword arguments (line 170)
        kwargs_23315 = {}
        # Getting the type of 'np' (line 170)
        np_23312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'np', False)
        # Obtaining the member 'arange' of a type (line 170)
        arange_23313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 23), np_23312, 'arange')
        # Calling arange(args, kwargs) (line 170)
        arange_call_result_23316 = invoke(stypy.reporting.localization.Localization(__file__, 170, 23), arange_23313, *[_o_23314], **kwargs_23315)
        
        # Getting the type of 'self' (line 170)
        self_23317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member '_orders' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_23317, '_orders', arange_call_result_23316)
        
        # Assigning a Subscript to a Name (line 171):
        
        # Assigning a Subscript to a Name (line 171):
        
        # Obtaining the type of the subscript
        # Getting the type of '_o' (line 171)
        _o_23318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), '_o')
        int_23319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 49), 'int')
        # Applying the binary operator '-' (line 171)
        result_sub_23320 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 44), '-', _o_23318, int_23319)
        
        # Getting the type of 'BezierSegment' (line 171)
        BezierSegment_23321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'BezierSegment')
        # Obtaining the member '_binom_coeff' of a type (line 171)
        _binom_coeff_23322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), BezierSegment_23321, '_binom_coeff')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___23323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), _binom_coeff_23322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_23324 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), getitem___23323, result_sub_23320)
        
        # Assigning a type to the variable '_coeff' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), '_coeff', subscript_call_result_23324)
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to asarray(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'control_points' (line 173)
        control_points_23327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 37), 'control_points', False)
        # Processing the call keyword arguments (line 173)
        kwargs_23328 = {}
        # Getting the type of 'np' (line 173)
        np_23325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 26), 'np', False)
        # Obtaining the member 'asarray' of a type (line 173)
        asarray_23326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 26), np_23325, 'asarray')
        # Calling asarray(args, kwargs) (line 173)
        asarray_call_result_23329 = invoke(stypy.reporting.localization.Localization(__file__, 173, 26), asarray_23326, *[control_points_23327], **kwargs_23328)
        
        # Assigning a type to the variable '_control_points' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), '_control_points', asarray_call_result_23329)
        
        # Assigning a Subscript to a Name (line 174):
        
        # Assigning a Subscript to a Name (line 174):
        
        # Obtaining the type of the subscript
        slice_23330 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 13), None, None, None)
        int_23331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 32), 'int')
        # Getting the type of '_control_points' (line 174)
        _control_points_23332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), '_control_points')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___23333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), _control_points_23332, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_23334 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), getitem___23333, (slice_23330, int_23331))
        
        # Assigning a type to the variable 'xx' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'xx', subscript_call_result_23334)
        
        # Assigning a Subscript to a Name (line 175):
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        slice_23335 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 175, 13), None, None, None)
        int_23336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 32), 'int')
        # Getting the type of '_control_points' (line 175)
        _control_points_23337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), '_control_points')
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___23338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 13), _control_points_23337, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_23339 = invoke(stypy.reporting.localization.Localization(__file__, 175, 13), getitem___23338, (slice_23335, int_23336))
        
        # Assigning a type to the variable 'yy' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'yy', subscript_call_result_23339)
        
        # Assigning a BinOp to a Attribute (line 177):
        
        # Assigning a BinOp to a Attribute (line 177):
        # Getting the type of 'xx' (line 177)
        xx_23340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'xx')
        # Getting the type of '_coeff' (line 177)
        _coeff_23341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), '_coeff')
        # Applying the binary operator '*' (line 177)
        result_mul_23342 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 19), '*', xx_23340, _coeff_23341)
        
        # Getting the type of 'self' (line 177)
        self_23343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member '_px' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_23343, '_px', result_mul_23342)
        
        # Assigning a BinOp to a Attribute (line 178):
        
        # Assigning a BinOp to a Attribute (line 178):
        # Getting the type of 'yy' (line 178)
        yy_23344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'yy')
        # Getting the type of '_coeff' (line 178)
        _coeff_23345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), '_coeff')
        # Applying the binary operator '*' (line 178)
        result_mul_23346 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 19), '*', yy_23344, _coeff_23345)
        
        # Getting the type of 'self' (line 178)
        self_23347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Setting the type of the member '_py' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_23347, '_py', result_mul_23346)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def point_at_t(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'point_at_t'
        module_type_store = module_type_store.open_function_context('point_at_t', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_localization', localization)
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_type_store', module_type_store)
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_function_name', 'BezierSegment.point_at_t')
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_param_names_list', ['t'])
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_varargs_param_name', None)
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_call_defaults', defaults)
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_call_varargs', varargs)
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BezierSegment.point_at_t.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BezierSegment.point_at_t', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'point_at_t', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'point_at_t(...)' code ##################

        unicode_23348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'unicode', u'evaluate a point at t')
        
        # Assigning a Subscript to a Name (line 182):
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_23349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 62), 'int')
        slice_23350 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 29), None, None, int_23349)
        
        # Call to power(...): (line 182)
        # Processing the call arguments (line 182)
        float_23353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 38), 'float')
        # Getting the type of 't' (line 182)
        t_23354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 43), 't', False)
        # Applying the binary operator '-' (line 182)
        result_sub_23355 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 38), '-', float_23353, t_23354)
        
        # Getting the type of 'self' (line 182)
        self_23356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 46), 'self', False)
        # Obtaining the member '_orders' of a type (line 182)
        _orders_23357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 46), self_23356, '_orders')
        # Processing the call keyword arguments (line 182)
        kwargs_23358 = {}
        # Getting the type of 'np' (line 182)
        np_23351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 29), 'np', False)
        # Obtaining the member 'power' of a type (line 182)
        power_23352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 29), np_23351, 'power')
        # Calling power(args, kwargs) (line 182)
        power_call_result_23359 = invoke(stypy.reporting.localization.Localization(__file__, 182, 29), power_23352, *[result_sub_23355, _orders_23357], **kwargs_23358)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___23360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 29), power_call_result_23359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_23361 = invoke(stypy.reporting.localization.Localization(__file__, 182, 29), getitem___23360, slice_23350)
        
        # Assigning a type to the variable 'one_minus_t_powers' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'one_minus_t_powers', subscript_call_result_23361)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to power(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 't' (line 183)
        t_23364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 't', False)
        # Getting the type of 'self' (line 183)
        self_23365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'self', False)
        # Obtaining the member '_orders' of a type (line 183)
        _orders_23366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 31), self_23365, '_orders')
        # Processing the call keyword arguments (line 183)
        kwargs_23367 = {}
        # Getting the type of 'np' (line 183)
        np_23362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'np', False)
        # Obtaining the member 'power' of a type (line 183)
        power_23363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 19), np_23362, 'power')
        # Calling power(args, kwargs) (line 183)
        power_call_result_23368 = invoke(stypy.reporting.localization.Localization(__file__, 183, 19), power_23363, *[t_23364, _orders_23366], **kwargs_23367)
        
        # Assigning a type to the variable 't_powers' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 't_powers', power_call_result_23368)
        
        # Assigning a BinOp to a Name (line 185):
        
        # Assigning a BinOp to a Name (line 185):
        # Getting the type of 'one_minus_t_powers' (line 185)
        one_minus_t_powers_23369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'one_minus_t_powers')
        # Getting the type of 't_powers' (line 185)
        t_powers_23370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 34), 't_powers')
        # Applying the binary operator '*' (line 185)
        result_mul_23371 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 13), '*', one_minus_t_powers_23369, t_powers_23370)
        
        # Assigning a type to the variable 'tt' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'tt', result_mul_23371)
        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to sum(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'tt' (line 186)
        tt_23373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 17), 'tt', False)
        # Getting the type of 'self' (line 186)
        self_23374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'self', False)
        # Obtaining the member '_px' of a type (line 186)
        _px_23375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 22), self_23374, '_px')
        # Applying the binary operator '*' (line 186)
        result_mul_23376 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 17), '*', tt_23373, _px_23375)
        
        # Processing the call keyword arguments (line 186)
        kwargs_23377 = {}
        # Getting the type of 'sum' (line 186)
        sum_23372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'sum', False)
        # Calling sum(args, kwargs) (line 186)
        sum_call_result_23378 = invoke(stypy.reporting.localization.Localization(__file__, 186, 13), sum_23372, *[result_mul_23376], **kwargs_23377)
        
        # Assigning a type to the variable '_x' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), '_x', sum_call_result_23378)
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to sum(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'tt' (line 187)
        tt_23380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 17), 'tt', False)
        # Getting the type of 'self' (line 187)
        self_23381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'self', False)
        # Obtaining the member '_py' of a type (line 187)
        _py_23382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 22), self_23381, '_py')
        # Applying the binary operator '*' (line 187)
        result_mul_23383 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 17), '*', tt_23380, _py_23382)
        
        # Processing the call keyword arguments (line 187)
        kwargs_23384 = {}
        # Getting the type of 'sum' (line 187)
        sum_23379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), 'sum', False)
        # Calling sum(args, kwargs) (line 187)
        sum_call_result_23385 = invoke(stypy.reporting.localization.Localization(__file__, 187, 13), sum_23379, *[result_mul_23383], **kwargs_23384)
        
        # Assigning a type to the variable '_y' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), '_y', sum_call_result_23385)
        
        # Obtaining an instance of the builtin type 'tuple' (line 189)
        tuple_23386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 189)
        # Adding element type (line 189)
        # Getting the type of '_x' (line 189)
        _x_23387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), '_x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 15), tuple_23386, _x_23387)
        # Adding element type (line 189)
        # Getting the type of '_y' (line 189)
        _y_23388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 19), '_y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 15), tuple_23386, _y_23388)
        
        # Assigning a type to the variable 'stypy_return_type' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'stypy_return_type', tuple_23386)
        
        # ################# End of 'point_at_t(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'point_at_t' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_23389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'point_at_t'
        return stypy_return_type_23389


# Assigning a type to the variable 'BezierSegment' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'BezierSegment', BezierSegment)

# Assigning a Dict to a Name (line 159):

# Obtaining an instance of the builtin type 'dict' (line 159)
dict_23390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 159)
# Adding element type (key, value) (line 159)
int_23391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'int')

# Call to array(...): (line 159)
# Processing the call arguments (line 159)

# Obtaining an instance of the builtin type 'list' (line 159)
list_23394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 159)
# Adding element type (line 159)
float_23395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 32), list_23394, float_23395)
# Adding element type (line 159)
float_23396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 32), list_23394, float_23396)

# Processing the call keyword arguments (line 159)
kwargs_23397 = {}
# Getting the type of 'np' (line 159)
np_23392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'np', False)
# Obtaining the member 'array' of a type (line 159)
array_23393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 23), np_23392, 'array')
# Calling array(args, kwargs) (line 159)
array_call_result_23398 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), array_23393, *[list_23394], **kwargs_23397)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), dict_23390, (int_23391, array_call_result_23398))
# Adding element type (key, value) (line 159)
int_23399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 20), 'int')

# Call to array(...): (line 160)
# Processing the call arguments (line 160)

# Obtaining an instance of the builtin type 'list' (line 160)
list_23402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 160)
# Adding element type (line 160)
float_23403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 32), list_23402, float_23403)
# Adding element type (line 160)
float_23404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 32), list_23402, float_23404)
# Adding element type (line 160)
float_23405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 41), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 32), list_23402, float_23405)

# Processing the call keyword arguments (line 160)
kwargs_23406 = {}
# Getting the type of 'np' (line 160)
np_23400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 23), 'np', False)
# Obtaining the member 'array' of a type (line 160)
array_23401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 23), np_23400, 'array')
# Calling array(args, kwargs) (line 160)
array_call_result_23407 = invoke(stypy.reporting.localization.Localization(__file__, 160, 23), array_23401, *[list_23402], **kwargs_23406)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), dict_23390, (int_23399, array_call_result_23407))
# Adding element type (key, value) (line 159)
int_23408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'int')

# Call to array(...): (line 161)
# Processing the call arguments (line 161)

# Obtaining an instance of the builtin type 'list' (line 161)
list_23411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 161)
# Adding element type (line 161)
float_23412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 32), list_23411, float_23412)
# Adding element type (line 161)
float_23413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 32), list_23411, float_23413)
# Adding element type (line 161)
float_23414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 41), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 32), list_23411, float_23414)
# Adding element type (line 161)
float_23415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 32), list_23411, float_23415)

# Processing the call keyword arguments (line 161)
kwargs_23416 = {}
# Getting the type of 'np' (line 161)
np_23409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'np', False)
# Obtaining the member 'array' of a type (line 161)
array_23410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 23), np_23409, 'array')
# Calling array(args, kwargs) (line 161)
array_call_result_23417 = invoke(stypy.reporting.localization.Localization(__file__, 161, 23), array_23410, *[list_23411], **kwargs_23416)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), dict_23390, (int_23408, array_call_result_23417))

# Getting the type of 'BezierSegment'
BezierSegment_23418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BezierSegment')
# Setting the type of the member '_binom_coeff' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BezierSegment_23418, '_binom_coeff', dict_23390)

@norecursion
def split_bezier_intersecting_with_closedpath(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_23419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 56), 'float')
    defaults = [float_23419]
    # Create a new context for function 'split_bezier_intersecting_with_closedpath'
    module_type_store = module_type_store.open_function_context('split_bezier_intersecting_with_closedpath', 192, 0, False)
    
    # Passed parameters checking function
    split_bezier_intersecting_with_closedpath.stypy_localization = localization
    split_bezier_intersecting_with_closedpath.stypy_type_of_self = None
    split_bezier_intersecting_with_closedpath.stypy_type_store = module_type_store
    split_bezier_intersecting_with_closedpath.stypy_function_name = 'split_bezier_intersecting_with_closedpath'
    split_bezier_intersecting_with_closedpath.stypy_param_names_list = ['bezier', 'inside_closedpath', 'tolerence']
    split_bezier_intersecting_with_closedpath.stypy_varargs_param_name = None
    split_bezier_intersecting_with_closedpath.stypy_kwargs_param_name = None
    split_bezier_intersecting_with_closedpath.stypy_call_defaults = defaults
    split_bezier_intersecting_with_closedpath.stypy_call_varargs = varargs
    split_bezier_intersecting_with_closedpath.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split_bezier_intersecting_with_closedpath', ['bezier', 'inside_closedpath', 'tolerence'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split_bezier_intersecting_with_closedpath', localization, ['bezier', 'inside_closedpath', 'tolerence'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split_bezier_intersecting_with_closedpath(...)' code ##################

    unicode_23420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, (-1)), 'unicode', u'\n    bezier : control points of the bezier segment\n    inside_closedpath : a function which returns true if the point is inside\n                        the path\n    ')
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to BezierSegment(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'bezier' (line 202)
    bezier_23422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'bezier', False)
    # Processing the call keyword arguments (line 202)
    kwargs_23423 = {}
    # Getting the type of 'BezierSegment' (line 202)
    BezierSegment_23421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 9), 'BezierSegment', False)
    # Calling BezierSegment(args, kwargs) (line 202)
    BezierSegment_call_result_23424 = invoke(stypy.reporting.localization.Localization(__file__, 202, 9), BezierSegment_23421, *[bezier_23422], **kwargs_23423)
    
    # Assigning a type to the variable 'bz' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'bz', BezierSegment_call_result_23424)
    
    # Assigning a Attribute to a Name (line 203):
    
    # Assigning a Attribute to a Name (line 203):
    # Getting the type of 'bz' (line 203)
    bz_23425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'bz')
    # Obtaining the member 'point_at_t' of a type (line 203)
    point_at_t_23426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 24), bz_23425, 'point_at_t')
    # Assigning a type to the variable 'bezier_point_at_t' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'bezier_point_at_t', point_at_t_23426)
    
    # Assigning a Call to a Tuple (line 205):
    
    # Assigning a Call to a Name:
    
    # Call to find_bezier_t_intersecting_with_closedpath(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'bezier_point_at_t' (line 205)
    bezier_point_at_t_23428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 56), 'bezier_point_at_t', False)
    # Getting the type of 'inside_closedpath' (line 206)
    inside_closedpath_23429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 56), 'inside_closedpath', False)
    # Processing the call keyword arguments (line 205)
    # Getting the type of 'tolerence' (line 207)
    tolerence_23430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 66), 'tolerence', False)
    keyword_23431 = tolerence_23430
    kwargs_23432 = {'tolerence': keyword_23431}
    # Getting the type of 'find_bezier_t_intersecting_with_closedpath' (line 205)
    find_bezier_t_intersecting_with_closedpath_23427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'find_bezier_t_intersecting_with_closedpath', False)
    # Calling find_bezier_t_intersecting_with_closedpath(args, kwargs) (line 205)
    find_bezier_t_intersecting_with_closedpath_call_result_23433 = invoke(stypy.reporting.localization.Localization(__file__, 205, 13), find_bezier_t_intersecting_with_closedpath_23427, *[bezier_point_at_t_23428, inside_closedpath_23429], **kwargs_23432)
    
    # Assigning a type to the variable 'call_assignment_22878' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'call_assignment_22878', find_bezier_t_intersecting_with_closedpath_call_result_23433)
    
    # Assigning a Call to a Name (line 205):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 4), 'int')
    # Processing the call keyword arguments
    kwargs_23437 = {}
    # Getting the type of 'call_assignment_22878' (line 205)
    call_assignment_22878_23434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'call_assignment_22878', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___23435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 4), call_assignment_22878_23434, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23438 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23435, *[int_23436], **kwargs_23437)
    
    # Assigning a type to the variable 'call_assignment_22879' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'call_assignment_22879', getitem___call_result_23438)
    
    # Assigning a Name to a Name (line 205):
    # Getting the type of 'call_assignment_22879' (line 205)
    call_assignment_22879_23439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'call_assignment_22879')
    # Assigning a type to the variable 't0' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 't0', call_assignment_22879_23439)
    
    # Assigning a Call to a Name (line 205):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 4), 'int')
    # Processing the call keyword arguments
    kwargs_23443 = {}
    # Getting the type of 'call_assignment_22878' (line 205)
    call_assignment_22878_23440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'call_assignment_22878', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___23441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 4), call_assignment_22878_23440, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23444 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23441, *[int_23442], **kwargs_23443)
    
    # Assigning a type to the variable 'call_assignment_22880' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'call_assignment_22880', getitem___call_result_23444)
    
    # Assigning a Name to a Name (line 205):
    # Getting the type of 'call_assignment_22880' (line 205)
    call_assignment_22880_23445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'call_assignment_22880')
    # Assigning a type to the variable 't1' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 't1', call_assignment_22880_23445)
    
    # Assigning a Call to a Tuple (line 209):
    
    # Assigning a Call to a Name:
    
    # Call to split_de_casteljau(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'bezier' (line 209)
    bezier_23447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 39), 'bezier', False)
    # Getting the type of 't0' (line 209)
    t0_23448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 48), 't0', False)
    # Getting the type of 't1' (line 209)
    t1_23449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 53), 't1', False)
    # Applying the binary operator '+' (line 209)
    result_add_23450 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 48), '+', t0_23448, t1_23449)
    
    float_23451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 59), 'float')
    # Applying the binary operator 'div' (line 209)
    result_div_23452 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 47), 'div', result_add_23450, float_23451)
    
    # Processing the call keyword arguments (line 209)
    kwargs_23453 = {}
    # Getting the type of 'split_de_casteljau' (line 209)
    split_de_casteljau_23446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'split_de_casteljau', False)
    # Calling split_de_casteljau(args, kwargs) (line 209)
    split_de_casteljau_call_result_23454 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), split_de_casteljau_23446, *[bezier_23447, result_div_23452], **kwargs_23453)
    
    # Assigning a type to the variable 'call_assignment_22881' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'call_assignment_22881', split_de_casteljau_call_result_23454)
    
    # Assigning a Call to a Name (line 209):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 4), 'int')
    # Processing the call keyword arguments
    kwargs_23458 = {}
    # Getting the type of 'call_assignment_22881' (line 209)
    call_assignment_22881_23455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'call_assignment_22881', False)
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___23456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 4), call_assignment_22881_23455, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23459 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23456, *[int_23457], **kwargs_23458)
    
    # Assigning a type to the variable 'call_assignment_22882' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'call_assignment_22882', getitem___call_result_23459)
    
    # Assigning a Name to a Name (line 209):
    # Getting the type of 'call_assignment_22882' (line 209)
    call_assignment_22882_23460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'call_assignment_22882')
    # Assigning a type to the variable '_left' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), '_left', call_assignment_22882_23460)
    
    # Assigning a Call to a Name (line 209):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 4), 'int')
    # Processing the call keyword arguments
    kwargs_23464 = {}
    # Getting the type of 'call_assignment_22881' (line 209)
    call_assignment_22881_23461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'call_assignment_22881', False)
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___23462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 4), call_assignment_22881_23461, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23465 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23462, *[int_23463], **kwargs_23464)
    
    # Assigning a type to the variable 'call_assignment_22883' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'call_assignment_22883', getitem___call_result_23465)
    
    # Assigning a Name to a Name (line 209):
    # Getting the type of 'call_assignment_22883' (line 209)
    call_assignment_22883_23466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'call_assignment_22883')
    # Assigning a type to the variable '_right' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), '_right', call_assignment_22883_23466)
    
    # Obtaining an instance of the builtin type 'tuple' (line 210)
    tuple_23467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of '_left' (line 210)
    _left_23468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), '_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 11), tuple_23467, _left_23468)
    # Adding element type (line 210)
    # Getting the type of '_right' (line 210)
    _right_23469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), '_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 11), tuple_23467, _right_23469)
    
    # Assigning a type to the variable 'stypy_return_type' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type', tuple_23467)
    
    # ################# End of 'split_bezier_intersecting_with_closedpath(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_bezier_intersecting_with_closedpath' in the type store
    # Getting the type of 'stypy_return_type' (line 192)
    stypy_return_type_23470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23470)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_bezier_intersecting_with_closedpath'
    return stypy_return_type_23470

# Assigning a type to the variable 'split_bezier_intersecting_with_closedpath' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'split_bezier_intersecting_with_closedpath', split_bezier_intersecting_with_closedpath)

@norecursion
def find_r_to_boundary_of_closedpath(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_23471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 42), 'float')
    float_23472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 51), 'float')
    float_23473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 65), 'float')
    defaults = [float_23471, float_23472, float_23473]
    # Create a new context for function 'find_r_to_boundary_of_closedpath'
    module_type_store = module_type_store.open_function_context('find_r_to_boundary_of_closedpath', 213, 0, False)
    
    # Passed parameters checking function
    find_r_to_boundary_of_closedpath.stypy_localization = localization
    find_r_to_boundary_of_closedpath.stypy_type_of_self = None
    find_r_to_boundary_of_closedpath.stypy_type_store = module_type_store
    find_r_to_boundary_of_closedpath.stypy_function_name = 'find_r_to_boundary_of_closedpath'
    find_r_to_boundary_of_closedpath.stypy_param_names_list = ['inside_closedpath', 'xy', 'cos_t', 'sin_t', 'rmin', 'rmax', 'tolerence']
    find_r_to_boundary_of_closedpath.stypy_varargs_param_name = None
    find_r_to_boundary_of_closedpath.stypy_kwargs_param_name = None
    find_r_to_boundary_of_closedpath.stypy_call_defaults = defaults
    find_r_to_boundary_of_closedpath.stypy_call_varargs = varargs
    find_r_to_boundary_of_closedpath.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_r_to_boundary_of_closedpath', ['inside_closedpath', 'xy', 'cos_t', 'sin_t', 'rmin', 'rmax', 'tolerence'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_r_to_boundary_of_closedpath', localization, ['inside_closedpath', 'xy', 'cos_t', 'sin_t', 'rmin', 'rmax', 'tolerence'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_r_to_boundary_of_closedpath(...)' code ##################

    unicode_23474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, (-1)), 'unicode', u'\n    Find a radius r (centered at *xy*) between *rmin* and *rmax* at\n    which it intersect with the path.\n\n    inside_closedpath : function\n    cx, cy : center\n    cos_t, sin_t : cosine and sine for the angle\n    rmin, rmax :\n    ')
    
    # Assigning a Name to a Tuple (line 226):
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_23475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    # Getting the type of 'xy' (line 226)
    xy_23476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), 'xy')
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___23477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), xy_23476, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_23478 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___23477, int_23475)
    
    # Assigning a type to the variable 'tuple_var_assignment_22884' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_22884', subscript_call_result_23478)
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_23479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    # Getting the type of 'xy' (line 226)
    xy_23480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), 'xy')
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___23481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), xy_23480, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_23482 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___23481, int_23479)
    
    # Assigning a type to the variable 'tuple_var_assignment_22885' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_22885', subscript_call_result_23482)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_22884' (line 226)
    tuple_var_assignment_22884_23483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_22884')
    # Assigning a type to the variable 'cx' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'cx', tuple_var_assignment_22884_23483)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_22885' (line 226)
    tuple_var_assignment_22885_23484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_22885')
    # Assigning a type to the variable 'cy' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'cy', tuple_var_assignment_22885_23484)

    @norecursion
    def _f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_f'
        module_type_store = module_type_store.open_function_context('_f', 228, 4, False)
        
        # Passed parameters checking function
        _f.stypy_localization = localization
        _f.stypy_type_of_self = None
        _f.stypy_type_store = module_type_store
        _f.stypy_function_name = '_f'
        _f.stypy_param_names_list = ['r']
        _f.stypy_varargs_param_name = None
        _f.stypy_kwargs_param_name = None
        _f.stypy_call_defaults = defaults
        _f.stypy_call_varargs = varargs
        _f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_f', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_f', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_f(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 229)
        tuple_23485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 229)
        # Adding element type (line 229)
        # Getting the type of 'cos_t' (line 229)
        cos_t_23486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'cos_t')
        # Getting the type of 'r' (line 229)
        r_23487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 23), 'r')
        # Applying the binary operator '*' (line 229)
        result_mul_23488 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), '*', cos_t_23486, r_23487)
        
        # Getting the type of 'cx' (line 229)
        cx_23489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'cx')
        # Applying the binary operator '+' (line 229)
        result_add_23490 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), '+', result_mul_23488, cx_23489)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 15), tuple_23485, result_add_23490)
        # Adding element type (line 229)
        # Getting the type of 'sin_t' (line 229)
        sin_t_23491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 31), 'sin_t')
        # Getting the type of 'r' (line 229)
        r_23492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 39), 'r')
        # Applying the binary operator '*' (line 229)
        result_mul_23493 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 31), '*', sin_t_23491, r_23492)
        
        # Getting the type of 'cy' (line 229)
        cy_23494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 43), 'cy')
        # Applying the binary operator '+' (line 229)
        result_add_23495 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 31), '+', result_mul_23493, cy_23494)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 15), tuple_23485, result_add_23495)
        
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', tuple_23485)
        
        # ################# End of '_f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_f' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_23496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23496)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_f'
        return stypy_return_type_23496

    # Assigning a type to the variable '_f' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), '_f', _f)
    
    # Call to find_bezier_t_intersecting_with_closedpath(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of '_f' (line 231)
    _f_23498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 47), '_f', False)
    # Getting the type of 'inside_closedpath' (line 231)
    inside_closedpath_23499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 51), 'inside_closedpath', False)
    # Processing the call keyword arguments (line 231)
    # Getting the type of 'rmin' (line 232)
    rmin_23500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 50), 'rmin', False)
    keyword_23501 = rmin_23500
    # Getting the type of 'rmax' (line 232)
    rmax_23502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 59), 'rmax', False)
    keyword_23503 = rmax_23502
    # Getting the type of 'tolerence' (line 233)
    tolerence_23504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 57), 'tolerence', False)
    keyword_23505 = tolerence_23504
    kwargs_23506 = {'tolerence': keyword_23505, 't0': keyword_23501, 't1': keyword_23503}
    # Getting the type of 'find_bezier_t_intersecting_with_closedpath' (line 231)
    find_bezier_t_intersecting_with_closedpath_23497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'find_bezier_t_intersecting_with_closedpath', False)
    # Calling find_bezier_t_intersecting_with_closedpath(args, kwargs) (line 231)
    find_bezier_t_intersecting_with_closedpath_call_result_23507 = invoke(stypy.reporting.localization.Localization(__file__, 231, 4), find_bezier_t_intersecting_with_closedpath_23497, *[_f_23498, inside_closedpath_23499], **kwargs_23506)
    
    
    # ################# End of 'find_r_to_boundary_of_closedpath(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_r_to_boundary_of_closedpath' in the type store
    # Getting the type of 'stypy_return_type' (line 213)
    stypy_return_type_23508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23508)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_r_to_boundary_of_closedpath'
    return stypy_return_type_23508

# Assigning a type to the variable 'find_r_to_boundary_of_closedpath' (line 213)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'find_r_to_boundary_of_closedpath', find_r_to_boundary_of_closedpath)

@norecursion
def split_path_inout(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_23509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 45), 'float')
    # Getting the type of 'False' (line 238)
    False_23510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 65), 'False')
    defaults = [float_23509, False_23510]
    # Create a new context for function 'split_path_inout'
    module_type_store = module_type_store.open_function_context('split_path_inout', 238, 0, False)
    
    # Passed parameters checking function
    split_path_inout.stypy_localization = localization
    split_path_inout.stypy_type_of_self = None
    split_path_inout.stypy_type_store = module_type_store
    split_path_inout.stypy_function_name = 'split_path_inout'
    split_path_inout.stypy_param_names_list = ['path', 'inside', 'tolerence', 'reorder_inout']
    split_path_inout.stypy_varargs_param_name = None
    split_path_inout.stypy_kwargs_param_name = None
    split_path_inout.stypy_call_defaults = defaults
    split_path_inout.stypy_call_varargs = varargs
    split_path_inout.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split_path_inout', ['path', 'inside', 'tolerence', 'reorder_inout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split_path_inout', localization, ['path', 'inside', 'tolerence', 'reorder_inout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split_path_inout(...)' code ##################

    unicode_23511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, (-1)), 'unicode', u' divide a path into two segment at the point where inside(x, y)\n    becomes False.\n    ')
    
    # Assigning a Call to a Name (line 243):
    
    # Assigning a Call to a Name (line 243):
    
    # Call to iter_segments(...): (line 243)
    # Processing the call keyword arguments (line 243)
    kwargs_23514 = {}
    # Getting the type of 'path' (line 243)
    path_23512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'path', False)
    # Obtaining the member 'iter_segments' of a type (line 243)
    iter_segments_23513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 16), path_23512, 'iter_segments')
    # Calling iter_segments(args, kwargs) (line 243)
    iter_segments_call_result_23515 = invoke(stypy.reporting.localization.Localization(__file__, 243, 16), iter_segments_23513, *[], **kwargs_23514)
    
    # Assigning a type to the variable 'path_iter' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'path_iter', iter_segments_call_result_23515)
    
    # Assigning a Call to a Tuple (line 245):
    
    # Assigning a Call to a Name:
    
    # Call to next(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'path_iter' (line 245)
    path_iter_23517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 31), 'path_iter', False)
    # Processing the call keyword arguments (line 245)
    kwargs_23518 = {}
    # Getting the type of 'next' (line 245)
    next_23516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), 'next', False)
    # Calling next(args, kwargs) (line 245)
    next_call_result_23519 = invoke(stypy.reporting.localization.Localization(__file__, 245, 26), next_23516, *[path_iter_23517], **kwargs_23518)
    
    # Assigning a type to the variable 'call_assignment_22886' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'call_assignment_22886', next_call_result_23519)
    
    # Assigning a Call to a Name (line 245):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 4), 'int')
    # Processing the call keyword arguments
    kwargs_23523 = {}
    # Getting the type of 'call_assignment_22886' (line 245)
    call_assignment_22886_23520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'call_assignment_22886', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___23521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 4), call_assignment_22886_23520, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23524 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23521, *[int_23522], **kwargs_23523)
    
    # Assigning a type to the variable 'call_assignment_22887' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'call_assignment_22887', getitem___call_result_23524)
    
    # Assigning a Name to a Name (line 245):
    # Getting the type of 'call_assignment_22887' (line 245)
    call_assignment_22887_23525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'call_assignment_22887')
    # Assigning a type to the variable 'ctl_points' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'ctl_points', call_assignment_22887_23525)
    
    # Assigning a Call to a Name (line 245):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 4), 'int')
    # Processing the call keyword arguments
    kwargs_23529 = {}
    # Getting the type of 'call_assignment_22886' (line 245)
    call_assignment_22886_23526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'call_assignment_22886', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___23527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 4), call_assignment_22886_23526, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23530 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23527, *[int_23528], **kwargs_23529)
    
    # Assigning a type to the variable 'call_assignment_22888' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'call_assignment_22888', getitem___call_result_23530)
    
    # Assigning a Name to a Name (line 245):
    # Getting the type of 'call_assignment_22888' (line 245)
    call_assignment_22888_23531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'call_assignment_22888')
    # Assigning a type to the variable 'command' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'command', call_assignment_22888_23531)
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to inside(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Obtaining the type of the subscript
    int_23533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 37), 'int')
    slice_23534 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 246, 26), int_23533, None, None)
    # Getting the type of 'ctl_points' (line 246)
    ctl_points_23535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'ctl_points', False)
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___23536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 26), ctl_points_23535, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_23537 = invoke(stypy.reporting.localization.Localization(__file__, 246, 26), getitem___23536, slice_23534)
    
    # Processing the call keyword arguments (line 246)
    kwargs_23538 = {}
    # Getting the type of 'inside' (line 246)
    inside_23532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'inside', False)
    # Calling inside(args, kwargs) (line 246)
    inside_call_result_23539 = invoke(stypy.reporting.localization.Localization(__file__, 246, 19), inside_23532, *[subscript_call_result_23537], **kwargs_23538)
    
    # Assigning a type to the variable 'begin_inside' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'begin_inside', inside_call_result_23539)
    
    # Assigning a Name to a Name (line 248):
    
    # Assigning a Name to a Name (line 248):
    # Getting the type of 'None' (line 248)
    None_23540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 18), 'None')
    # Assigning a type to the variable 'bezier_path' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'bezier_path', None_23540)
    
    # Assigning a Name to a Name (line 249):
    
    # Assigning a Name to a Name (line 249):
    # Getting the type of 'ctl_points' (line 249)
    ctl_points_23541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'ctl_points')
    # Assigning a type to the variable 'ctl_points_old' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'ctl_points_old', ctl_points_23541)
    
    # Assigning a Attribute to a Name (line 251):
    
    # Assigning a Attribute to a Name (line 251):
    # Getting the type of 'np' (line 251)
    np_23542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'np')
    # Obtaining the member 'concatenate' of a type (line 251)
    concatenate_23543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 13), np_23542, 'concatenate')
    # Assigning a type to the variable 'concat' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'concat', concatenate_23543)
    
    # Assigning a Num to a Name (line 253):
    
    # Assigning a Num to a Name (line 253):
    int_23544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 11), 'int')
    # Assigning a type to the variable 'iold' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'iold', int_23544)
    
    # Assigning a Num to a Name (line 254):
    
    # Assigning a Num to a Name (line 254):
    int_23545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 8), 'int')
    # Assigning a type to the variable 'i' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'i', int_23545)
    
    # Getting the type of 'path_iter' (line 256)
    path_iter_23546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 31), 'path_iter')
    # Testing the type of a for loop iterable (line 256)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 256, 4), path_iter_23546)
    # Getting the type of the for loop variable (line 256)
    for_loop_var_23547 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 256, 4), path_iter_23546)
    # Assigning a type to the variable 'ctl_points' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'ctl_points', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 4), for_loop_var_23547))
    # Assigning a type to the variable 'command' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'command', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 4), for_loop_var_23547))
    # SSA begins for a for statement (line 256)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 257):
    
    # Assigning a Name to a Name (line 257):
    # Getting the type of 'i' (line 257)
    i_23548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'i')
    # Assigning a type to the variable 'iold' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'iold', i_23548)
    
    # Getting the type of 'i' (line 258)
    i_23549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'i')
    
    # Call to len(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'ctl_points' (line 258)
    ctl_points_23551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'ctl_points', False)
    # Processing the call keyword arguments (line 258)
    kwargs_23552 = {}
    # Getting the type of 'len' (line 258)
    len_23550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'len', False)
    # Calling len(args, kwargs) (line 258)
    len_call_result_23553 = invoke(stypy.reporting.localization.Localization(__file__, 258, 13), len_23550, *[ctl_points_23551], **kwargs_23552)
    
    int_23554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 32), 'int')
    # Applying the binary operator '//' (line 258)
    result_floordiv_23555 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 13), '//', len_call_result_23553, int_23554)
    
    # Applying the binary operator '+=' (line 258)
    result_iadd_23556 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 8), '+=', i_23549, result_floordiv_23555)
    # Assigning a type to the variable 'i' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'i', result_iadd_23556)
    
    
    
    
    # Call to inside(...): (line 259)
    # Processing the call arguments (line 259)
    
    # Obtaining the type of the subscript
    int_23558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 29), 'int')
    slice_23559 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 18), int_23558, None, None)
    # Getting the type of 'ctl_points' (line 259)
    ctl_points_23560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'ctl_points', False)
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___23561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 18), ctl_points_23560, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_23562 = invoke(stypy.reporting.localization.Localization(__file__, 259, 18), getitem___23561, slice_23559)
    
    # Processing the call keyword arguments (line 259)
    kwargs_23563 = {}
    # Getting the type of 'inside' (line 259)
    inside_23557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'inside', False)
    # Calling inside(args, kwargs) (line 259)
    inside_call_result_23564 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), inside_23557, *[subscript_call_result_23562], **kwargs_23563)
    
    # Getting the type of 'begin_inside' (line 259)
    begin_inside_23565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 38), 'begin_inside')
    # Applying the binary operator '!=' (line 259)
    result_ne_23566 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 11), '!=', inside_call_result_23564, begin_inside_23565)
    
    # Testing the type of an if condition (line 259)
    if_condition_23567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 8), result_ne_23566)
    # Assigning a type to the variable 'if_condition_23567' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'if_condition_23567', if_condition_23567)
    # SSA begins for if statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 260):
    
    # Assigning a Call to a Name (line 260):
    
    # Call to concat(...): (line 260)
    # Processing the call arguments (line 260)
    
    # Obtaining an instance of the builtin type 'list' (line 260)
    list_23569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 260)
    # Adding element type (line 260)
    
    # Obtaining the type of the subscript
    int_23570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 49), 'int')
    slice_23571 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 260, 34), int_23570, None, None)
    # Getting the type of 'ctl_points_old' (line 260)
    ctl_points_old_23572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'ctl_points_old', False)
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___23573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 34), ctl_points_old_23572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_23574 = invoke(stypy.reporting.localization.Localization(__file__, 260, 34), getitem___23573, slice_23571)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 33), list_23569, subscript_call_result_23574)
    # Adding element type (line 260)
    # Getting the type of 'ctl_points' (line 260)
    ctl_points_23575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 55), 'ctl_points', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 33), list_23569, ctl_points_23575)
    
    # Processing the call keyword arguments (line 260)
    kwargs_23576 = {}
    # Getting the type of 'concat' (line 260)
    concat_23568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 26), 'concat', False)
    # Calling concat(args, kwargs) (line 260)
    concat_call_result_23577 = invoke(stypy.reporting.localization.Localization(__file__, 260, 26), concat_23568, *[list_23569], **kwargs_23576)
    
    # Assigning a type to the variable 'bezier_path' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'bezier_path', concat_call_result_23577)
    # SSA join for if statement (line 259)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 263):
    
    # Assigning a Name to a Name (line 263):
    # Getting the type of 'ctl_points' (line 263)
    ctl_points_23578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'ctl_points')
    # Assigning a type to the variable 'ctl_points_old' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'ctl_points_old', ctl_points_23578)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 265)
    # Getting the type of 'bezier_path' (line 265)
    bezier_path_23579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 7), 'bezier_path')
    # Getting the type of 'None' (line 265)
    None_23580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'None')
    
    (may_be_23581, more_types_in_union_23582) = may_be_none(bezier_path_23579, None_23580)

    if may_be_23581:

        if more_types_in_union_23582:
            # Runtime conditional SSA (line 265)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 266)
        # Processing the call arguments (line 266)
        unicode_23584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 25), 'unicode', u'The path does not seem to intersect with the patch')
        # Processing the call keyword arguments (line 266)
        kwargs_23585 = {}
        # Getting the type of 'ValueError' (line 266)
        ValueError_23583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 266)
        ValueError_call_result_23586 = invoke(stypy.reporting.localization.Localization(__file__, 266, 14), ValueError_23583, *[unicode_23584], **kwargs_23585)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 266, 8), ValueError_call_result_23586, 'raise parameter', BaseException)

        if more_types_in_union_23582:
            # SSA join for if statement (line 265)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 268):
    
    # Call to list(...): (line 268)
    # Processing the call arguments (line 268)
    
    # Call to zip(...): (line 268)
    # Processing the call arguments (line 268)
    
    # Obtaining the type of the subscript
    int_23589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 32), 'int')
    slice_23590 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 18), None, None, int_23589)
    # Getting the type of 'bezier_path' (line 268)
    bezier_path_23591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 18), 'bezier_path', False)
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___23592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 18), bezier_path_23591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_23593 = invoke(stypy.reporting.localization.Localization(__file__, 268, 18), getitem___23592, slice_23590)
    
    
    # Obtaining the type of the subscript
    int_23594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 48), 'int')
    int_23595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 51), 'int')
    slice_23596 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 36), int_23594, None, int_23595)
    # Getting the type of 'bezier_path' (line 268)
    bezier_path_23597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'bezier_path', False)
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___23598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 36), bezier_path_23597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_23599 = invoke(stypy.reporting.localization.Localization(__file__, 268, 36), getitem___23598, slice_23596)
    
    # Processing the call keyword arguments (line 268)
    kwargs_23600 = {}
    # Getting the type of 'zip' (line 268)
    zip_23588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 14), 'zip', False)
    # Calling zip(args, kwargs) (line 268)
    zip_call_result_23601 = invoke(stypy.reporting.localization.Localization(__file__, 268, 14), zip_23588, *[subscript_call_result_23593, subscript_call_result_23599], **kwargs_23600)
    
    # Processing the call keyword arguments (line 268)
    kwargs_23602 = {}
    # Getting the type of 'list' (line 268)
    list_23587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 9), 'list', False)
    # Calling list(args, kwargs) (line 268)
    list_call_result_23603 = invoke(stypy.reporting.localization.Localization(__file__, 268, 9), list_23587, *[zip_call_result_23601], **kwargs_23602)
    
    # Assigning a type to the variable 'bp' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'bp', list_call_result_23603)
    
    # Assigning a Call to a Tuple (line 269):
    
    # Assigning a Call to a Name:
    
    # Call to split_bezier_intersecting_with_closedpath(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'bp' (line 269)
    bp_23605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 60), 'bp', False)
    # Getting the type of 'inside' (line 270)
    inside_23606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 60), 'inside', False)
    # Getting the type of 'tolerence' (line 271)
    tolerence_23607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 60), 'tolerence', False)
    # Processing the call keyword arguments (line 269)
    kwargs_23608 = {}
    # Getting the type of 'split_bezier_intersecting_with_closedpath' (line 269)
    split_bezier_intersecting_with_closedpath_23604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 18), 'split_bezier_intersecting_with_closedpath', False)
    # Calling split_bezier_intersecting_with_closedpath(args, kwargs) (line 269)
    split_bezier_intersecting_with_closedpath_call_result_23609 = invoke(stypy.reporting.localization.Localization(__file__, 269, 18), split_bezier_intersecting_with_closedpath_23604, *[bp_23605, inside_23606, tolerence_23607], **kwargs_23608)
    
    # Assigning a type to the variable 'call_assignment_22889' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'call_assignment_22889', split_bezier_intersecting_with_closedpath_call_result_23609)
    
    # Assigning a Call to a Name (line 269):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 4), 'int')
    # Processing the call keyword arguments
    kwargs_23613 = {}
    # Getting the type of 'call_assignment_22889' (line 269)
    call_assignment_22889_23610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'call_assignment_22889', False)
    # Obtaining the member '__getitem__' of a type (line 269)
    getitem___23611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 4), call_assignment_22889_23610, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23614 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23611, *[int_23612], **kwargs_23613)
    
    # Assigning a type to the variable 'call_assignment_22890' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'call_assignment_22890', getitem___call_result_23614)
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'call_assignment_22890' (line 269)
    call_assignment_22890_23615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'call_assignment_22890')
    # Assigning a type to the variable 'left' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'left', call_assignment_22890_23615)
    
    # Assigning a Call to a Name (line 269):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 4), 'int')
    # Processing the call keyword arguments
    kwargs_23619 = {}
    # Getting the type of 'call_assignment_22889' (line 269)
    call_assignment_22889_23616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'call_assignment_22889', False)
    # Obtaining the member '__getitem__' of a type (line 269)
    getitem___23617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 4), call_assignment_22889_23616, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23620 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23617, *[int_23618], **kwargs_23619)
    
    # Assigning a type to the variable 'call_assignment_22891' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'call_assignment_22891', getitem___call_result_23620)
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'call_assignment_22891' (line 269)
    call_assignment_22891_23621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'call_assignment_22891')
    # Assigning a type to the variable 'right' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 10), 'right', call_assignment_22891_23621)
    
    
    
    # Call to len(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'left' (line 272)
    left_23623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'left', False)
    # Processing the call keyword arguments (line 272)
    kwargs_23624 = {}
    # Getting the type of 'len' (line 272)
    len_23622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 7), 'len', False)
    # Calling len(args, kwargs) (line 272)
    len_call_result_23625 = invoke(stypy.reporting.localization.Localization(__file__, 272, 7), len_23622, *[left_23623], **kwargs_23624)
    
    int_23626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 20), 'int')
    # Applying the binary operator '==' (line 272)
    result_eq_23627 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 7), '==', len_call_result_23625, int_23626)
    
    # Testing the type of an if condition (line 272)
    if_condition_23628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 4), result_eq_23627)
    # Assigning a type to the variable 'if_condition_23628' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'if_condition_23628', if_condition_23628)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 273):
    
    # Assigning a List to a Name (line 273):
    
    # Obtaining an instance of the builtin type 'list' (line 273)
    list_23629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 273)
    # Adding element type (line 273)
    # Getting the type of 'Path' (line 273)
    Path_23630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'Path')
    # Obtaining the member 'LINETO' of a type (line 273)
    LINETO_23631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 22), Path_23630, 'LINETO')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 21), list_23629, LINETO_23631)
    
    # Assigning a type to the variable 'codes_left' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'codes_left', list_23629)
    
    # Assigning a List to a Name (line 274):
    
    # Assigning a List to a Name (line 274):
    
    # Obtaining an instance of the builtin type 'list' (line 274)
    list_23632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 274)
    # Adding element type (line 274)
    # Getting the type of 'Path' (line 274)
    Path_23633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 23), 'Path')
    # Obtaining the member 'MOVETO' of a type (line 274)
    MOVETO_23634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 23), Path_23633, 'MOVETO')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 22), list_23632, MOVETO_23634)
    # Adding element type (line 274)
    # Getting the type of 'Path' (line 274)
    Path_23635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'Path')
    # Obtaining the member 'LINETO' of a type (line 274)
    LINETO_23636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 36), Path_23635, 'LINETO')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 22), list_23632, LINETO_23636)
    
    # Assigning a type to the variable 'codes_right' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'codes_right', list_23632)
    # SSA branch for the else part of an if statement (line 272)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'left' (line 275)
    left_23638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'left', False)
    # Processing the call keyword arguments (line 275)
    kwargs_23639 = {}
    # Getting the type of 'len' (line 275)
    len_23637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), 'len', False)
    # Calling len(args, kwargs) (line 275)
    len_call_result_23640 = invoke(stypy.reporting.localization.Localization(__file__, 275, 9), len_23637, *[left_23638], **kwargs_23639)
    
    int_23641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 22), 'int')
    # Applying the binary operator '==' (line 275)
    result_eq_23642 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 9), '==', len_call_result_23640, int_23641)
    
    # Testing the type of an if condition (line 275)
    if_condition_23643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 9), result_eq_23642)
    # Assigning a type to the variable 'if_condition_23643' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), 'if_condition_23643', if_condition_23643)
    # SSA begins for if statement (line 275)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 276):
    
    # Assigning a List to a Name (line 276):
    
    # Obtaining an instance of the builtin type 'list' (line 276)
    list_23644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 276)
    # Adding element type (line 276)
    # Getting the type of 'Path' (line 276)
    Path_23645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'Path')
    # Obtaining the member 'CURVE3' of a type (line 276)
    CURVE3_23646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 22), Path_23645, 'CURVE3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 21), list_23644, CURVE3_23646)
    # Adding element type (line 276)
    # Getting the type of 'Path' (line 276)
    Path_23647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'Path')
    # Obtaining the member 'CURVE3' of a type (line 276)
    CURVE3_23648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 35), Path_23647, 'CURVE3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 21), list_23644, CURVE3_23648)
    
    # Assigning a type to the variable 'codes_left' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'codes_left', list_23644)
    
    # Assigning a List to a Name (line 277):
    
    # Assigning a List to a Name (line 277):
    
    # Obtaining an instance of the builtin type 'list' (line 277)
    list_23649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 277)
    # Adding element type (line 277)
    # Getting the type of 'Path' (line 277)
    Path_23650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 23), 'Path')
    # Obtaining the member 'MOVETO' of a type (line 277)
    MOVETO_23651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 23), Path_23650, 'MOVETO')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 22), list_23649, MOVETO_23651)
    # Adding element type (line 277)
    # Getting the type of 'Path' (line 277)
    Path_23652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 36), 'Path')
    # Obtaining the member 'CURVE3' of a type (line 277)
    CURVE3_23653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 36), Path_23652, 'CURVE3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 22), list_23649, CURVE3_23653)
    # Adding element type (line 277)
    # Getting the type of 'Path' (line 277)
    Path_23654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 49), 'Path')
    # Obtaining the member 'CURVE3' of a type (line 277)
    CURVE3_23655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 49), Path_23654, 'CURVE3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 22), list_23649, CURVE3_23655)
    
    # Assigning a type to the variable 'codes_right' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'codes_right', list_23649)
    # SSA branch for the else part of an if statement (line 275)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'left' (line 278)
    left_23657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 13), 'left', False)
    # Processing the call keyword arguments (line 278)
    kwargs_23658 = {}
    # Getting the type of 'len' (line 278)
    len_23656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 9), 'len', False)
    # Calling len(args, kwargs) (line 278)
    len_call_result_23659 = invoke(stypy.reporting.localization.Localization(__file__, 278, 9), len_23656, *[left_23657], **kwargs_23658)
    
    int_23660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 22), 'int')
    # Applying the binary operator '==' (line 278)
    result_eq_23661 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 9), '==', len_call_result_23659, int_23660)
    
    # Testing the type of an if condition (line 278)
    if_condition_23662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 9), result_eq_23661)
    # Assigning a type to the variable 'if_condition_23662' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 9), 'if_condition_23662', if_condition_23662)
    # SSA begins for if statement (line 278)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 279):
    
    # Assigning a List to a Name (line 279):
    
    # Obtaining an instance of the builtin type 'list' (line 279)
    list_23663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 279)
    # Adding element type (line 279)
    # Getting the type of 'Path' (line 279)
    Path_23664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 22), 'Path')
    # Obtaining the member 'CURVE4' of a type (line 279)
    CURVE4_23665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 22), Path_23664, 'CURVE4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 21), list_23663, CURVE4_23665)
    # Adding element type (line 279)
    # Getting the type of 'Path' (line 279)
    Path_23666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 35), 'Path')
    # Obtaining the member 'CURVE4' of a type (line 279)
    CURVE4_23667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 35), Path_23666, 'CURVE4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 21), list_23663, CURVE4_23667)
    # Adding element type (line 279)
    # Getting the type of 'Path' (line 279)
    Path_23668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 48), 'Path')
    # Obtaining the member 'CURVE4' of a type (line 279)
    CURVE4_23669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 48), Path_23668, 'CURVE4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 21), list_23663, CURVE4_23669)
    
    # Assigning a type to the variable 'codes_left' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'codes_left', list_23663)
    
    # Assigning a List to a Name (line 280):
    
    # Assigning a List to a Name (line 280):
    
    # Obtaining an instance of the builtin type 'list' (line 280)
    list_23670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 280)
    # Adding element type (line 280)
    # Getting the type of 'Path' (line 280)
    Path_23671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'Path')
    # Obtaining the member 'MOVETO' of a type (line 280)
    MOVETO_23672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 23), Path_23671, 'MOVETO')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 22), list_23670, MOVETO_23672)
    # Adding element type (line 280)
    # Getting the type of 'Path' (line 280)
    Path_23673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 36), 'Path')
    # Obtaining the member 'CURVE4' of a type (line 280)
    CURVE4_23674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 36), Path_23673, 'CURVE4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 22), list_23670, CURVE4_23674)
    # Adding element type (line 280)
    # Getting the type of 'Path' (line 280)
    Path_23675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 49), 'Path')
    # Obtaining the member 'CURVE4' of a type (line 280)
    CURVE4_23676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 49), Path_23675, 'CURVE4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 22), list_23670, CURVE4_23676)
    # Adding element type (line 280)
    # Getting the type of 'Path' (line 280)
    Path_23677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 62), 'Path')
    # Obtaining the member 'CURVE4' of a type (line 280)
    CURVE4_23678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 62), Path_23677, 'CURVE4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 22), list_23670, CURVE4_23678)
    
    # Assigning a type to the variable 'codes_right' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'codes_right', list_23670)
    # SSA branch for the else part of an if statement (line 278)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 282)
    # Processing the call keyword arguments (line 282)
    kwargs_23680 = {}
    # Getting the type of 'ValueError' (line 282)
    ValueError_23679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 282)
    ValueError_call_result_23681 = invoke(stypy.reporting.localization.Localization(__file__, 282, 14), ValueError_23679, *[], **kwargs_23680)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 282, 8), ValueError_call_result_23681, 'raise parameter', BaseException)
    # SSA join for if statement (line 278)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 275)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 284):
    
    # Assigning a Subscript to a Name (line 284):
    
    # Obtaining the type of the subscript
    int_23682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 22), 'int')
    slice_23683 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 284, 17), int_23682, None, None)
    # Getting the type of 'left' (line 284)
    left_23684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 17), 'left')
    # Obtaining the member '__getitem__' of a type (line 284)
    getitem___23685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 17), left_23684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 284)
    subscript_call_result_23686 = invoke(stypy.reporting.localization.Localization(__file__, 284, 17), getitem___23685, slice_23683)
    
    # Assigning a type to the variable 'verts_left' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'verts_left', subscript_call_result_23686)
    
    # Assigning a Subscript to a Name (line 285):
    
    # Assigning a Subscript to a Name (line 285):
    
    # Obtaining the type of the subscript
    slice_23687 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 285, 18), None, None, None)
    # Getting the type of 'right' (line 285)
    right_23688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 18), 'right')
    # Obtaining the member '__getitem__' of a type (line 285)
    getitem___23689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 18), right_23688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 285)
    subscript_call_result_23690 = invoke(stypy.reporting.localization.Localization(__file__, 285, 18), getitem___23689, slice_23687)
    
    # Assigning a type to the variable 'verts_right' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'verts_right', subscript_call_result_23690)
    
    # Type idiom detected: calculating its left and rigth part (line 287)
    # Getting the type of 'path' (line 287)
    path_23691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 7), 'path')
    # Obtaining the member 'codes' of a type (line 287)
    codes_23692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 7), path_23691, 'codes')
    # Getting the type of 'None' (line 287)
    None_23693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), 'None')
    
    (may_be_23694, more_types_in_union_23695) = may_be_none(codes_23692, None_23693)

    if may_be_23694:

        if more_types_in_union_23695:
            # Runtime conditional SSA (line 287)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 288):
        
        # Assigning a Call to a Name (line 288):
        
        # Call to Path(...): (line 288)
        # Processing the call arguments (line 288)
        
        # Call to concat(...): (line 288)
        # Processing the call arguments (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_23698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 288)
        i_23699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 46), 'i', False)
        slice_23700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 288, 31), None, i_23699, None)
        # Getting the type of 'path' (line 288)
        path_23701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 31), 'path', False)
        # Obtaining the member 'vertices' of a type (line 288)
        vertices_23702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 31), path_23701, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___23703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 31), vertices_23702, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 288)
        subscript_call_result_23704 = invoke(stypy.reporting.localization.Localization(__file__, 288, 31), getitem___23703, slice_23700)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 30), list_23698, subscript_call_result_23704)
        # Adding element type (line 288)
        # Getting the type of 'verts_left' (line 288)
        verts_left_23705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'verts_left', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 30), list_23698, verts_left_23705)
        
        # Processing the call keyword arguments (line 288)
        kwargs_23706 = {}
        # Getting the type of 'concat' (line 288)
        concat_23697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'concat', False)
        # Calling concat(args, kwargs) (line 288)
        concat_call_result_23707 = invoke(stypy.reporting.localization.Localization(__file__, 288, 23), concat_23697, *[list_23698], **kwargs_23706)
        
        # Processing the call keyword arguments (line 288)
        kwargs_23708 = {}
        # Getting the type of 'Path' (line 288)
        Path_23696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'Path', False)
        # Calling Path(args, kwargs) (line 288)
        Path_call_result_23709 = invoke(stypy.reporting.localization.Localization(__file__, 288, 18), Path_23696, *[concat_call_result_23707], **kwargs_23708)
        
        # Assigning a type to the variable 'path_in' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'path_in', Path_call_result_23709)
        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to Path(...): (line 289)
        # Processing the call arguments (line 289)
        
        # Call to concat(...): (line 289)
        # Processing the call arguments (line 289)
        
        # Obtaining an instance of the builtin type 'list' (line 289)
        list_23712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 289)
        # Adding element type (line 289)
        # Getting the type of 'verts_right' (line 289)
        verts_right_23713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'verts_right', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 31), list_23712, verts_right_23713)
        # Adding element type (line 289)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 289)
        i_23714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 59), 'i', False)
        slice_23715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 289, 45), i_23714, None, None)
        # Getting the type of 'path' (line 289)
        path_23716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 45), 'path', False)
        # Obtaining the member 'vertices' of a type (line 289)
        vertices_23717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 45), path_23716, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___23718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 45), vertices_23717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_23719 = invoke(stypy.reporting.localization.Localization(__file__, 289, 45), getitem___23718, slice_23715)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 31), list_23712, subscript_call_result_23719)
        
        # Processing the call keyword arguments (line 289)
        kwargs_23720 = {}
        # Getting the type of 'concat' (line 289)
        concat_23711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 24), 'concat', False)
        # Calling concat(args, kwargs) (line 289)
        concat_call_result_23721 = invoke(stypy.reporting.localization.Localization(__file__, 289, 24), concat_23711, *[list_23712], **kwargs_23720)
        
        # Processing the call keyword arguments (line 289)
        kwargs_23722 = {}
        # Getting the type of 'Path' (line 289)
        Path_23710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 19), 'Path', False)
        # Calling Path(args, kwargs) (line 289)
        Path_call_result_23723 = invoke(stypy.reporting.localization.Localization(__file__, 289, 19), Path_23710, *[concat_call_result_23721], **kwargs_23722)
        
        # Assigning a type to the variable 'path_out' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'path_out', Path_call_result_23723)

        if more_types_in_union_23695:
            # Runtime conditional SSA for else branch (line 287)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_23694) or more_types_in_union_23695):
        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Call to Path(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Call to concat(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_23726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        
        # Obtaining the type of the subscript
        # Getting the type of 'iold' (line 292)
        iold_23727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 46), 'iold', False)
        slice_23728 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 292, 31), None, iold_23727, None)
        # Getting the type of 'path' (line 292)
        path_23729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'path', False)
        # Obtaining the member 'vertices' of a type (line 292)
        vertices_23730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 31), path_23729, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___23731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 31), vertices_23730, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_23732 = invoke(stypy.reporting.localization.Localization(__file__, 292, 31), getitem___23731, slice_23728)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 30), list_23726, subscript_call_result_23732)
        # Adding element type (line 292)
        # Getting the type of 'verts_left' (line 292)
        verts_left_23733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 53), 'verts_left', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 30), list_23726, verts_left_23733)
        
        # Processing the call keyword arguments (line 292)
        kwargs_23734 = {}
        # Getting the type of 'concat' (line 292)
        concat_23725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'concat', False)
        # Calling concat(args, kwargs) (line 292)
        concat_call_result_23735 = invoke(stypy.reporting.localization.Localization(__file__, 292, 23), concat_23725, *[list_23726], **kwargs_23734)
        
        
        # Call to concat(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_23737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        
        # Obtaining the type of the subscript
        # Getting the type of 'iold' (line 293)
        iold_23738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 43), 'iold', False)
        slice_23739 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 293, 31), None, iold_23738, None)
        # Getting the type of 'path' (line 293)
        path_23740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 31), 'path', False)
        # Obtaining the member 'codes' of a type (line 293)
        codes_23741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 31), path_23740, 'codes')
        # Obtaining the member '__getitem__' of a type (line 293)
        getitem___23742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 31), codes_23741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 293)
        subscript_call_result_23743 = invoke(stypy.reporting.localization.Localization(__file__, 293, 31), getitem___23742, slice_23739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 30), list_23737, subscript_call_result_23743)
        # Adding element type (line 293)
        # Getting the type of 'codes_left' (line 293)
        codes_left_23744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 50), 'codes_left', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 30), list_23737, codes_left_23744)
        
        # Processing the call keyword arguments (line 293)
        kwargs_23745 = {}
        # Getting the type of 'concat' (line 293)
        concat_23736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 23), 'concat', False)
        # Calling concat(args, kwargs) (line 293)
        concat_call_result_23746 = invoke(stypy.reporting.localization.Localization(__file__, 293, 23), concat_23736, *[list_23737], **kwargs_23745)
        
        # Processing the call keyword arguments (line 292)
        kwargs_23747 = {}
        # Getting the type of 'Path' (line 292)
        Path_23724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 18), 'Path', False)
        # Calling Path(args, kwargs) (line 292)
        Path_call_result_23748 = invoke(stypy.reporting.localization.Localization(__file__, 292, 18), Path_23724, *[concat_call_result_23735, concat_call_result_23746], **kwargs_23747)
        
        # Assigning a type to the variable 'path_in' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'path_in', Path_call_result_23748)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to Path(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Call to concat(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_23751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        # Getting the type of 'verts_right' (line 295)
        verts_right_23752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 32), 'verts_right', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 31), list_23751, verts_right_23752)
        # Adding element type (line 295)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 295)
        i_23753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 59), 'i', False)
        slice_23754 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 295, 45), i_23753, None, None)
        # Getting the type of 'path' (line 295)
        path_23755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 45), 'path', False)
        # Obtaining the member 'vertices' of a type (line 295)
        vertices_23756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 45), path_23755, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___23757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 45), vertices_23756, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_23758 = invoke(stypy.reporting.localization.Localization(__file__, 295, 45), getitem___23757, slice_23754)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 31), list_23751, subscript_call_result_23758)
        
        # Processing the call keyword arguments (line 295)
        kwargs_23759 = {}
        # Getting the type of 'concat' (line 295)
        concat_23750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'concat', False)
        # Calling concat(args, kwargs) (line 295)
        concat_call_result_23760 = invoke(stypy.reporting.localization.Localization(__file__, 295, 24), concat_23750, *[list_23751], **kwargs_23759)
        
        
        # Call to concat(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_23762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)
        # Getting the type of 'codes_right' (line 296)
        codes_right_23763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), 'codes_right', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 31), list_23762, codes_right_23763)
        # Adding element type (line 296)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 296)
        i_23764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 56), 'i', False)
        slice_23765 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 45), i_23764, None, None)
        # Getting the type of 'path' (line 296)
        path_23766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 45), 'path', False)
        # Obtaining the member 'codes' of a type (line 296)
        codes_23767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 45), path_23766, 'codes')
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___23768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 45), codes_23767, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_23769 = invoke(stypy.reporting.localization.Localization(__file__, 296, 45), getitem___23768, slice_23765)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 31), list_23762, subscript_call_result_23769)
        
        # Processing the call keyword arguments (line 296)
        kwargs_23770 = {}
        # Getting the type of 'concat' (line 296)
        concat_23761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'concat', False)
        # Calling concat(args, kwargs) (line 296)
        concat_call_result_23771 = invoke(stypy.reporting.localization.Localization(__file__, 296, 24), concat_23761, *[list_23762], **kwargs_23770)
        
        # Processing the call keyword arguments (line 295)
        kwargs_23772 = {}
        # Getting the type of 'Path' (line 295)
        Path_23749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'Path', False)
        # Calling Path(args, kwargs) (line 295)
        Path_call_result_23773 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), Path_23749, *[concat_call_result_23760, concat_call_result_23771], **kwargs_23772)
        
        # Assigning a type to the variable 'path_out' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'path_out', Path_call_result_23773)

        if (may_be_23694 and more_types_in_union_23695):
            # SSA join for if statement (line 287)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'reorder_inout' (line 298)
    reorder_inout_23774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 7), 'reorder_inout')
    
    # Getting the type of 'begin_inside' (line 298)
    begin_inside_23775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 25), 'begin_inside')
    # Getting the type of 'False' (line 298)
    False_23776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'False')
    # Applying the binary operator 'is' (line 298)
    result_is__23777 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 25), 'is', begin_inside_23775, False_23776)
    
    # Applying the binary operator 'and' (line 298)
    result_and_keyword_23778 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 7), 'and', reorder_inout_23774, result_is__23777)
    
    # Testing the type of an if condition (line 298)
    if_condition_23779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 4), result_and_keyword_23778)
    # Assigning a type to the variable 'if_condition_23779' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'if_condition_23779', if_condition_23779)
    # SSA begins for if statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 299):
    
    # Assigning a Name to a Name (line 299):
    # Getting the type of 'path_out' (line 299)
    path_out_23780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 'path_out')
    # Assigning a type to the variable 'tuple_assignment_22892' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_assignment_22892', path_out_23780)
    
    # Assigning a Name to a Name (line 299):
    # Getting the type of 'path_in' (line 299)
    path_in_23781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 38), 'path_in')
    # Assigning a type to the variable 'tuple_assignment_22893' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_assignment_22893', path_in_23781)
    
    # Assigning a Name to a Name (line 299):
    # Getting the type of 'tuple_assignment_22892' (line 299)
    tuple_assignment_22892_23782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_assignment_22892')
    # Assigning a type to the variable 'path_in' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'path_in', tuple_assignment_22892_23782)
    
    # Assigning a Name to a Name (line 299):
    # Getting the type of 'tuple_assignment_22893' (line 299)
    tuple_assignment_22893_23783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_assignment_22893')
    # Assigning a type to the variable 'path_out' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'path_out', tuple_assignment_22893_23783)
    # SSA join for if statement (line 298)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 301)
    tuple_23784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 301)
    # Adding element type (line 301)
    # Getting the type of 'path_in' (line 301)
    path_in_23785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'path_in')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 11), tuple_23784, path_in_23785)
    # Adding element type (line 301)
    # Getting the type of 'path_out' (line 301)
    path_out_23786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 20), 'path_out')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 11), tuple_23784, path_out_23786)
    
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type', tuple_23784)
    
    # ################# End of 'split_path_inout(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_path_inout' in the type store
    # Getting the type of 'stypy_return_type' (line 238)
    stypy_return_type_23787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23787)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_path_inout'
    return stypy_return_type_23787

# Assigning a type to the variable 'split_path_inout' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'split_path_inout', split_path_inout)

@norecursion
def inside_circle(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'inside_circle'
    module_type_store = module_type_store.open_function_context('inside_circle', 304, 0, False)
    
    # Passed parameters checking function
    inside_circle.stypy_localization = localization
    inside_circle.stypy_type_of_self = None
    inside_circle.stypy_type_store = module_type_store
    inside_circle.stypy_function_name = 'inside_circle'
    inside_circle.stypy_param_names_list = ['cx', 'cy', 'r']
    inside_circle.stypy_varargs_param_name = None
    inside_circle.stypy_kwargs_param_name = None
    inside_circle.stypy_call_defaults = defaults
    inside_circle.stypy_call_varargs = varargs
    inside_circle.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'inside_circle', ['cx', 'cy', 'r'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'inside_circle', localization, ['cx', 'cy', 'r'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'inside_circle(...)' code ##################

    
    # Assigning a BinOp to a Name (line 305):
    
    # Assigning a BinOp to a Name (line 305):
    # Getting the type of 'r' (line 305)
    r_23788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 9), 'r')
    int_23789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 14), 'int')
    # Applying the binary operator '**' (line 305)
    result_pow_23790 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 9), '**', r_23788, int_23789)
    
    # Assigning a type to the variable 'r2' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'r2', result_pow_23790)

    @norecursion
    def _f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_f'
        module_type_store = module_type_store.open_function_context('_f', 307, 4, False)
        
        # Passed parameters checking function
        _f.stypy_localization = localization
        _f.stypy_type_of_self = None
        _f.stypy_type_store = module_type_store
        _f.stypy_function_name = '_f'
        _f.stypy_param_names_list = ['xy']
        _f.stypy_varargs_param_name = None
        _f.stypy_kwargs_param_name = None
        _f.stypy_call_defaults = defaults
        _f.stypy_call_varargs = varargs
        _f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_f', ['xy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_f', localization, ['xy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_f(...)' code ##################

        
        # Assigning a Name to a Tuple (line 308):
        
        # Assigning a Subscript to a Name (line 308):
        
        # Obtaining the type of the subscript
        int_23791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 8), 'int')
        # Getting the type of 'xy' (line 308)
        xy_23792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 15), 'xy')
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___23793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), xy_23792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_23794 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), getitem___23793, int_23791)
        
        # Assigning a type to the variable 'tuple_var_assignment_22894' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'tuple_var_assignment_22894', subscript_call_result_23794)
        
        # Assigning a Subscript to a Name (line 308):
        
        # Obtaining the type of the subscript
        int_23795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 8), 'int')
        # Getting the type of 'xy' (line 308)
        xy_23796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 15), 'xy')
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___23797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), xy_23796, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_23798 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), getitem___23797, int_23795)
        
        # Assigning a type to the variable 'tuple_var_assignment_22895' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'tuple_var_assignment_22895', subscript_call_result_23798)
        
        # Assigning a Name to a Name (line 308):
        # Getting the type of 'tuple_var_assignment_22894' (line 308)
        tuple_var_assignment_22894_23799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'tuple_var_assignment_22894')
        # Assigning a type to the variable 'x' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'x', tuple_var_assignment_22894_23799)
        
        # Assigning a Name to a Name (line 308):
        # Getting the type of 'tuple_var_assignment_22895' (line 308)
        tuple_var_assignment_22895_23800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'tuple_var_assignment_22895')
        # Assigning a type to the variable 'y' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), 'y', tuple_var_assignment_22895_23800)
        
        # Getting the type of 'x' (line 309)
        x_23801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'x')
        # Getting the type of 'cx' (line 309)
        cx_23802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 20), 'cx')
        # Applying the binary operator '-' (line 309)
        result_sub_23803 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 16), '-', x_23801, cx_23802)
        
        int_23804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 27), 'int')
        # Applying the binary operator '**' (line 309)
        result_pow_23805 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 15), '**', result_sub_23803, int_23804)
        
        # Getting the type of 'y' (line 309)
        y_23806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 32), 'y')
        # Getting the type of 'cy' (line 309)
        cy_23807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 36), 'cy')
        # Applying the binary operator '-' (line 309)
        result_sub_23808 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 32), '-', y_23806, cy_23807)
        
        int_23809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 43), 'int')
        # Applying the binary operator '**' (line 309)
        result_pow_23810 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 31), '**', result_sub_23808, int_23809)
        
        # Applying the binary operator '+' (line 309)
        result_add_23811 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 15), '+', result_pow_23805, result_pow_23810)
        
        # Getting the type of 'r2' (line 309)
        r2_23812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 47), 'r2')
        # Applying the binary operator '<' (line 309)
        result_lt_23813 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 15), '<', result_add_23811, r2_23812)
        
        # Assigning a type to the variable 'stypy_return_type' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'stypy_return_type', result_lt_23813)
        
        # ################# End of '_f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_f' in the type store
        # Getting the type of 'stypy_return_type' (line 307)
        stypy_return_type_23814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23814)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_f'
        return stypy_return_type_23814

    # Assigning a type to the variable '_f' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), '_f', _f)
    # Getting the type of '_f' (line 310)
    _f_23815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), '_f')
    # Assigning a type to the variable 'stypy_return_type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type', _f_23815)
    
    # ################# End of 'inside_circle(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'inside_circle' in the type store
    # Getting the type of 'stypy_return_type' (line 304)
    stypy_return_type_23816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23816)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'inside_circle'
    return stypy_return_type_23816

# Assigning a type to the variable 'inside_circle' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'inside_circle', inside_circle)

@norecursion
def get_cos_sin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_cos_sin'
    module_type_store = module_type_store.open_function_context('get_cos_sin', 315, 0, False)
    
    # Passed parameters checking function
    get_cos_sin.stypy_localization = localization
    get_cos_sin.stypy_type_of_self = None
    get_cos_sin.stypy_type_store = module_type_store
    get_cos_sin.stypy_function_name = 'get_cos_sin'
    get_cos_sin.stypy_param_names_list = ['x0', 'y0', 'x1', 'y1']
    get_cos_sin.stypy_varargs_param_name = None
    get_cos_sin.stypy_kwargs_param_name = None
    get_cos_sin.stypy_call_defaults = defaults
    get_cos_sin.stypy_call_varargs = varargs
    get_cos_sin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_cos_sin', ['x0', 'y0', 'x1', 'y1'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_cos_sin', localization, ['x0', 'y0', 'x1', 'y1'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_cos_sin(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 316):
    
    # Assigning a BinOp to a Name (line 316):
    # Getting the type of 'x1' (line 316)
    x1_23817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'x1')
    # Getting the type of 'x0' (line 316)
    x0_23818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'x0')
    # Applying the binary operator '-' (line 316)
    result_sub_23819 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 13), '-', x1_23817, x0_23818)
    
    # Assigning a type to the variable 'tuple_assignment_22896' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_assignment_22896', result_sub_23819)
    
    # Assigning a BinOp to a Name (line 316):
    # Getting the type of 'y1' (line 316)
    y1_23820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'y1')
    # Getting the type of 'y0' (line 316)
    y0_23821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 27), 'y0')
    # Applying the binary operator '-' (line 316)
    result_sub_23822 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 22), '-', y1_23820, y0_23821)
    
    # Assigning a type to the variable 'tuple_assignment_22897' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_assignment_22897', result_sub_23822)
    
    # Assigning a Name to a Name (line 316):
    # Getting the type of 'tuple_assignment_22896' (line 316)
    tuple_assignment_22896_23823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_assignment_22896')
    # Assigning a type to the variable 'dx' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'dx', tuple_assignment_22896_23823)
    
    # Assigning a Name to a Name (line 316):
    # Getting the type of 'tuple_assignment_22897' (line 316)
    tuple_assignment_22897_23824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_assignment_22897')
    # Assigning a type to the variable 'dy' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'dy', tuple_assignment_22897_23824)
    
    # Assigning a BinOp to a Name (line 317):
    
    # Assigning a BinOp to a Name (line 317):
    # Getting the type of 'dx' (line 317)
    dx_23825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 9), 'dx')
    # Getting the type of 'dx' (line 317)
    dx_23826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 14), 'dx')
    # Applying the binary operator '*' (line 317)
    result_mul_23827 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 9), '*', dx_23825, dx_23826)
    
    # Getting the type of 'dy' (line 317)
    dy_23828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'dy')
    # Getting the type of 'dy' (line 317)
    dy_23829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 24), 'dy')
    # Applying the binary operator '*' (line 317)
    result_mul_23830 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 19), '*', dy_23828, dy_23829)
    
    # Applying the binary operator '+' (line 317)
    result_add_23831 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 9), '+', result_mul_23827, result_mul_23830)
    
    float_23832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 31), 'float')
    # Applying the binary operator '**' (line 317)
    result_pow_23833 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 8), '**', result_add_23831, float_23832)
    
    # Assigning a type to the variable 'd' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'd', result_pow_23833)
    
    
    # Getting the type of 'd' (line 319)
    d_23834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 7), 'd')
    int_23835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 12), 'int')
    # Applying the binary operator '==' (line 319)
    result_eq_23836 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 7), '==', d_23834, int_23835)
    
    # Testing the type of an if condition (line 319)
    if_condition_23837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 4), result_eq_23836)
    # Assigning a type to the variable 'if_condition_23837' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'if_condition_23837', if_condition_23837)
    # SSA begins for if statement (line 319)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 320)
    tuple_23838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 320)
    # Adding element type (line 320)
    float_23839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), tuple_23838, float_23839)
    # Adding element type (line 320)
    float_23840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), tuple_23838, float_23840)
    
    # Assigning a type to the variable 'stypy_return_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type', tuple_23838)
    # SSA join for if statement (line 319)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 321)
    tuple_23841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 321)
    # Adding element type (line 321)
    # Getting the type of 'dx' (line 321)
    dx_23842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 'dx')
    # Getting the type of 'd' (line 321)
    d_23843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'd')
    # Applying the binary operator 'div' (line 321)
    result_div_23844 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 11), 'div', dx_23842, d_23843)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 11), tuple_23841, result_div_23844)
    # Adding element type (line 321)
    # Getting the type of 'dy' (line 321)
    dy_23845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'dy')
    # Getting the type of 'd' (line 321)
    d_23846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'd')
    # Applying the binary operator 'div' (line 321)
    result_div_23847 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 19), 'div', dy_23845, d_23846)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 11), tuple_23841, result_div_23847)
    
    # Assigning a type to the variable 'stypy_return_type' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'stypy_return_type', tuple_23841)
    
    # ################# End of 'get_cos_sin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_cos_sin' in the type store
    # Getting the type of 'stypy_return_type' (line 315)
    stypy_return_type_23848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23848)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_cos_sin'
    return stypy_return_type_23848

# Assigning a type to the variable 'get_cos_sin' (line 315)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'get_cos_sin', get_cos_sin)

@norecursion
def check_if_parallel(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_23849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 52), 'float')
    defaults = [float_23849]
    # Create a new context for function 'check_if_parallel'
    module_type_store = module_type_store.open_function_context('check_if_parallel', 324, 0, False)
    
    # Passed parameters checking function
    check_if_parallel.stypy_localization = localization
    check_if_parallel.stypy_type_of_self = None
    check_if_parallel.stypy_type_store = module_type_store
    check_if_parallel.stypy_function_name = 'check_if_parallel'
    check_if_parallel.stypy_param_names_list = ['dx1', 'dy1', 'dx2', 'dy2', 'tolerence']
    check_if_parallel.stypy_varargs_param_name = None
    check_if_parallel.stypy_kwargs_param_name = None
    check_if_parallel.stypy_call_defaults = defaults
    check_if_parallel.stypy_call_varargs = varargs
    check_if_parallel.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_if_parallel', ['dx1', 'dy1', 'dx2', 'dy2', 'tolerence'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_if_parallel', localization, ['dx1', 'dy1', 'dx2', 'dy2', 'tolerence'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_if_parallel(...)' code ##################

    unicode_23850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'unicode', u' returns\n       * 1 if two lines are parralel in same direction\n       * -1 if two lines are parralel in opposite direction\n       * 0 otherwise\n    ')
    
    # Assigning a Call to a Name (line 330):
    
    # Assigning a Call to a Name (line 330):
    
    # Call to arctan2(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'dx1' (line 330)
    dx1_23853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'dx1', False)
    # Getting the type of 'dy1' (line 330)
    dy1_23854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 29), 'dy1', False)
    # Processing the call keyword arguments (line 330)
    kwargs_23855 = {}
    # Getting the type of 'np' (line 330)
    np_23851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 13), 'np', False)
    # Obtaining the member 'arctan2' of a type (line 330)
    arctan2_23852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 13), np_23851, 'arctan2')
    # Calling arctan2(args, kwargs) (line 330)
    arctan2_call_result_23856 = invoke(stypy.reporting.localization.Localization(__file__, 330, 13), arctan2_23852, *[dx1_23853, dy1_23854], **kwargs_23855)
    
    # Assigning a type to the variable 'theta1' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'theta1', arctan2_call_result_23856)
    
    # Assigning a Call to a Name (line 331):
    
    # Assigning a Call to a Name (line 331):
    
    # Call to arctan2(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'dx2' (line 331)
    dx2_23859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'dx2', False)
    # Getting the type of 'dy2' (line 331)
    dy2_23860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'dy2', False)
    # Processing the call keyword arguments (line 331)
    kwargs_23861 = {}
    # Getting the type of 'np' (line 331)
    np_23857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 13), 'np', False)
    # Obtaining the member 'arctan2' of a type (line 331)
    arctan2_23858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 13), np_23857, 'arctan2')
    # Calling arctan2(args, kwargs) (line 331)
    arctan2_call_result_23862 = invoke(stypy.reporting.localization.Localization(__file__, 331, 13), arctan2_23858, *[dx2_23859, dy2_23860], **kwargs_23861)
    
    # Assigning a type to the variable 'theta2' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'theta2', arctan2_call_result_23862)
    
    # Assigning a Call to a Name (line 332):
    
    # Assigning a Call to a Name (line 332):
    
    # Call to abs(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'theta1' (line 332)
    theta1_23865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 20), 'theta1', False)
    # Getting the type of 'theta2' (line 332)
    theta2_23866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 29), 'theta2', False)
    # Applying the binary operator '-' (line 332)
    result_sub_23867 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 20), '-', theta1_23865, theta2_23866)
    
    # Processing the call keyword arguments (line 332)
    kwargs_23868 = {}
    # Getting the type of 'np' (line 332)
    np_23863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 13), 'np', False)
    # Obtaining the member 'abs' of a type (line 332)
    abs_23864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 13), np_23863, 'abs')
    # Calling abs(args, kwargs) (line 332)
    abs_call_result_23869 = invoke(stypy.reporting.localization.Localization(__file__, 332, 13), abs_23864, *[result_sub_23867], **kwargs_23868)
    
    # Assigning a type to the variable 'dtheta' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'dtheta', abs_call_result_23869)
    
    
    # Getting the type of 'dtheta' (line 333)
    dtheta_23870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 7), 'dtheta')
    # Getting the type of 'tolerence' (line 333)
    tolerence_23871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'tolerence')
    # Applying the binary operator '<' (line 333)
    result_lt_23872 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 7), '<', dtheta_23870, tolerence_23871)
    
    # Testing the type of an if condition (line 333)
    if_condition_23873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 4), result_lt_23872)
    # Assigning a type to the variable 'if_condition_23873' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'if_condition_23873', if_condition_23873)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_23874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'stypy_return_type', int_23874)
    # SSA branch for the else part of an if statement (line 333)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to abs(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'dtheta' (line 335)
    dtheta_23877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'dtheta', False)
    # Getting the type of 'np' (line 335)
    np_23878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'np', False)
    # Obtaining the member 'pi' of a type (line 335)
    pi_23879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 25), np_23878, 'pi')
    # Applying the binary operator '-' (line 335)
    result_sub_23880 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 16), '-', dtheta_23877, pi_23879)
    
    # Processing the call keyword arguments (line 335)
    kwargs_23881 = {}
    # Getting the type of 'np' (line 335)
    np_23875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 9), 'np', False)
    # Obtaining the member 'abs' of a type (line 335)
    abs_23876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 9), np_23875, 'abs')
    # Calling abs(args, kwargs) (line 335)
    abs_call_result_23882 = invoke(stypy.reporting.localization.Localization(__file__, 335, 9), abs_23876, *[result_sub_23880], **kwargs_23881)
    
    # Getting the type of 'tolerence' (line 335)
    tolerence_23883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 34), 'tolerence')
    # Applying the binary operator '<' (line 335)
    result_lt_23884 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 9), '<', abs_call_result_23882, tolerence_23883)
    
    # Testing the type of an if condition (line 335)
    if_condition_23885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 9), result_lt_23884)
    # Assigning a type to the variable 'if_condition_23885' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 9), 'if_condition_23885', if_condition_23885)
    # SSA begins for if statement (line 335)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_23886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'stypy_return_type', int_23886)
    # SSA branch for the else part of an if statement (line 335)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'False' (line 338)
    False_23887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'stypy_return_type', False_23887)
    # SSA join for if statement (line 335)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_if_parallel(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_if_parallel' in the type store
    # Getting the type of 'stypy_return_type' (line 324)
    stypy_return_type_23888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23888)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_if_parallel'
    return stypy_return_type_23888

# Assigning a type to the variable 'check_if_parallel' (line 324)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 0), 'check_if_parallel', check_if_parallel)

@norecursion
def get_parallels(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_parallels'
    module_type_store = module_type_store.open_function_context('get_parallels', 341, 0, False)
    
    # Passed parameters checking function
    get_parallels.stypy_localization = localization
    get_parallels.stypy_type_of_self = None
    get_parallels.stypy_type_store = module_type_store
    get_parallels.stypy_function_name = 'get_parallels'
    get_parallels.stypy_param_names_list = ['bezier2', 'width']
    get_parallels.stypy_varargs_param_name = None
    get_parallels.stypy_kwargs_param_name = None
    get_parallels.stypy_call_defaults = defaults
    get_parallels.stypy_call_varargs = varargs
    get_parallels.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_parallels', ['bezier2', 'width'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_parallels', localization, ['bezier2', 'width'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_parallels(...)' code ##################

    unicode_23889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, (-1)), 'unicode', u'\n    Given the quadratic bezier control points *bezier2*, returns\n    control points of quadratic bezier lines roughly parallel to given\n    one separated by *width*.\n    ')
    
    # Assigning a Subscript to a Tuple (line 353):
    
    # Assigning a Subscript to a Name (line 353):
    
    # Obtaining the type of the subscript
    int_23890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 4), 'int')
    
    # Obtaining the type of the subscript
    int_23891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 23), 'int')
    # Getting the type of 'bezier2' (line 353)
    bezier2_23892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___23893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 15), bezier2_23892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_23894 = invoke(stypy.reporting.localization.Localization(__file__, 353, 15), getitem___23893, int_23891)
    
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___23895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 4), subscript_call_result_23894, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_23896 = invoke(stypy.reporting.localization.Localization(__file__, 353, 4), getitem___23895, int_23890)
    
    # Assigning a type to the variable 'tuple_var_assignment_22898' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'tuple_var_assignment_22898', subscript_call_result_23896)
    
    # Assigning a Subscript to a Name (line 353):
    
    # Obtaining the type of the subscript
    int_23897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 4), 'int')
    
    # Obtaining the type of the subscript
    int_23898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 23), 'int')
    # Getting the type of 'bezier2' (line 353)
    bezier2_23899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___23900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 15), bezier2_23899, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_23901 = invoke(stypy.reporting.localization.Localization(__file__, 353, 15), getitem___23900, int_23898)
    
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___23902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 4), subscript_call_result_23901, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_23903 = invoke(stypy.reporting.localization.Localization(__file__, 353, 4), getitem___23902, int_23897)
    
    # Assigning a type to the variable 'tuple_var_assignment_22899' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'tuple_var_assignment_22899', subscript_call_result_23903)
    
    # Assigning a Name to a Name (line 353):
    # Getting the type of 'tuple_var_assignment_22898' (line 353)
    tuple_var_assignment_22898_23904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'tuple_var_assignment_22898')
    # Assigning a type to the variable 'c1x' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'c1x', tuple_var_assignment_22898_23904)
    
    # Assigning a Name to a Name (line 353):
    # Getting the type of 'tuple_var_assignment_22899' (line 353)
    tuple_var_assignment_22899_23905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'tuple_var_assignment_22899')
    # Assigning a type to the variable 'c1y' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 9), 'c1y', tuple_var_assignment_22899_23905)
    
    # Assigning a Subscript to a Tuple (line 354):
    
    # Assigning a Subscript to a Name (line 354):
    
    # Obtaining the type of the subscript
    int_23906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 4), 'int')
    
    # Obtaining the type of the subscript
    int_23907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 23), 'int')
    # Getting the type of 'bezier2' (line 354)
    bezier2_23908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___23909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 15), bezier2_23908, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_23910 = invoke(stypy.reporting.localization.Localization(__file__, 354, 15), getitem___23909, int_23907)
    
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___23911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 4), subscript_call_result_23910, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_23912 = invoke(stypy.reporting.localization.Localization(__file__, 354, 4), getitem___23911, int_23906)
    
    # Assigning a type to the variable 'tuple_var_assignment_22900' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'tuple_var_assignment_22900', subscript_call_result_23912)
    
    # Assigning a Subscript to a Name (line 354):
    
    # Obtaining the type of the subscript
    int_23913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 4), 'int')
    
    # Obtaining the type of the subscript
    int_23914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 23), 'int')
    # Getting the type of 'bezier2' (line 354)
    bezier2_23915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___23916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 15), bezier2_23915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_23917 = invoke(stypy.reporting.localization.Localization(__file__, 354, 15), getitem___23916, int_23914)
    
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___23918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 4), subscript_call_result_23917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_23919 = invoke(stypy.reporting.localization.Localization(__file__, 354, 4), getitem___23918, int_23913)
    
    # Assigning a type to the variable 'tuple_var_assignment_22901' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'tuple_var_assignment_22901', subscript_call_result_23919)
    
    # Assigning a Name to a Name (line 354):
    # Getting the type of 'tuple_var_assignment_22900' (line 354)
    tuple_var_assignment_22900_23920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'tuple_var_assignment_22900')
    # Assigning a type to the variable 'cmx' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'cmx', tuple_var_assignment_22900_23920)
    
    # Assigning a Name to a Name (line 354):
    # Getting the type of 'tuple_var_assignment_22901' (line 354)
    tuple_var_assignment_22901_23921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'tuple_var_assignment_22901')
    # Assigning a type to the variable 'cmy' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 9), 'cmy', tuple_var_assignment_22901_23921)
    
    # Assigning a Subscript to a Tuple (line 355):
    
    # Assigning a Subscript to a Name (line 355):
    
    # Obtaining the type of the subscript
    int_23922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 4), 'int')
    
    # Obtaining the type of the subscript
    int_23923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 23), 'int')
    # Getting the type of 'bezier2' (line 355)
    bezier2_23924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___23925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 15), bezier2_23924, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_23926 = invoke(stypy.reporting.localization.Localization(__file__, 355, 15), getitem___23925, int_23923)
    
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___23927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 4), subscript_call_result_23926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_23928 = invoke(stypy.reporting.localization.Localization(__file__, 355, 4), getitem___23927, int_23922)
    
    # Assigning a type to the variable 'tuple_var_assignment_22902' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'tuple_var_assignment_22902', subscript_call_result_23928)
    
    # Assigning a Subscript to a Name (line 355):
    
    # Obtaining the type of the subscript
    int_23929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 4), 'int')
    
    # Obtaining the type of the subscript
    int_23930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 23), 'int')
    # Getting the type of 'bezier2' (line 355)
    bezier2_23931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___23932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 15), bezier2_23931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_23933 = invoke(stypy.reporting.localization.Localization(__file__, 355, 15), getitem___23932, int_23930)
    
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___23934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 4), subscript_call_result_23933, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_23935 = invoke(stypy.reporting.localization.Localization(__file__, 355, 4), getitem___23934, int_23929)
    
    # Assigning a type to the variable 'tuple_var_assignment_22903' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'tuple_var_assignment_22903', subscript_call_result_23935)
    
    # Assigning a Name to a Name (line 355):
    # Getting the type of 'tuple_var_assignment_22902' (line 355)
    tuple_var_assignment_22902_23936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'tuple_var_assignment_22902')
    # Assigning a type to the variable 'c2x' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'c2x', tuple_var_assignment_22902_23936)
    
    # Assigning a Name to a Name (line 355):
    # Getting the type of 'tuple_var_assignment_22903' (line 355)
    tuple_var_assignment_22903_23937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'tuple_var_assignment_22903')
    # Assigning a type to the variable 'c2y' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 9), 'c2y', tuple_var_assignment_22903_23937)
    
    # Assigning a Call to a Name (line 357):
    
    # Assigning a Call to a Name (line 357):
    
    # Call to check_if_parallel(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'c1x' (line 357)
    c1x_23939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 38), 'c1x', False)
    # Getting the type of 'cmx' (line 357)
    cmx_23940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 44), 'cmx', False)
    # Applying the binary operator '-' (line 357)
    result_sub_23941 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 38), '-', c1x_23939, cmx_23940)
    
    # Getting the type of 'c1y' (line 357)
    c1y_23942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 49), 'c1y', False)
    # Getting the type of 'cmy' (line 357)
    cmy_23943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 55), 'cmy', False)
    # Applying the binary operator '-' (line 357)
    result_sub_23944 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 49), '-', c1y_23942, cmy_23943)
    
    # Getting the type of 'cmx' (line 358)
    cmx_23945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 38), 'cmx', False)
    # Getting the type of 'c2x' (line 358)
    c2x_23946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 44), 'c2x', False)
    # Applying the binary operator '-' (line 358)
    result_sub_23947 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 38), '-', cmx_23945, c2x_23946)
    
    # Getting the type of 'cmy' (line 358)
    cmy_23948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 49), 'cmy', False)
    # Getting the type of 'c2y' (line 358)
    c2y_23949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 55), 'c2y', False)
    # Applying the binary operator '-' (line 358)
    result_sub_23950 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 49), '-', cmy_23948, c2y_23949)
    
    # Processing the call keyword arguments (line 357)
    kwargs_23951 = {}
    # Getting the type of 'check_if_parallel' (line 357)
    check_if_parallel_23938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'check_if_parallel', False)
    # Calling check_if_parallel(args, kwargs) (line 357)
    check_if_parallel_call_result_23952 = invoke(stypy.reporting.localization.Localization(__file__, 357, 20), check_if_parallel_23938, *[result_sub_23941, result_sub_23944, result_sub_23947, result_sub_23950], **kwargs_23951)
    
    # Assigning a type to the variable 'parallel_test' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'parallel_test', check_if_parallel_call_result_23952)
    
    
    # Getting the type of 'parallel_test' (line 360)
    parallel_test_23953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 7), 'parallel_test')
    int_23954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 24), 'int')
    # Applying the binary operator '==' (line 360)
    result_eq_23955 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 7), '==', parallel_test_23953, int_23954)
    
    # Testing the type of an if condition (line 360)
    if_condition_23956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 4), result_eq_23955)
    # Assigning a type to the variable 'if_condition_23956' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'if_condition_23956', if_condition_23956)
    # SSA begins for if statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 361)
    # Processing the call arguments (line 361)
    unicode_23959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'unicode', u'Lines do not intersect. A straight line is used instead.')
    # Processing the call keyword arguments (line 361)
    kwargs_23960 = {}
    # Getting the type of 'warnings' (line 361)
    warnings_23957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 361)
    warn_23958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), warnings_23957, 'warn')
    # Calling warn(args, kwargs) (line 361)
    warn_call_result_23961 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), warn_23958, *[unicode_23959], **kwargs_23960)
    
    
    # Assigning a Call to a Tuple (line 363):
    
    # Assigning a Call to a Name:
    
    # Call to get_cos_sin(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'c1x' (line 363)
    c1x_23963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 37), 'c1x', False)
    # Getting the type of 'c1y' (line 363)
    c1y_23964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 42), 'c1y', False)
    # Getting the type of 'c2x' (line 363)
    c2x_23965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 47), 'c2x', False)
    # Getting the type of 'c2y' (line 363)
    c2y_23966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 52), 'c2y', False)
    # Processing the call keyword arguments (line 363)
    kwargs_23967 = {}
    # Getting the type of 'get_cos_sin' (line 363)
    get_cos_sin_23962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'get_cos_sin', False)
    # Calling get_cos_sin(args, kwargs) (line 363)
    get_cos_sin_call_result_23968 = invoke(stypy.reporting.localization.Localization(__file__, 363, 25), get_cos_sin_23962, *[c1x_23963, c1y_23964, c2x_23965, c2y_23966], **kwargs_23967)
    
    # Assigning a type to the variable 'call_assignment_22904' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'call_assignment_22904', get_cos_sin_call_result_23968)
    
    # Assigning a Call to a Name (line 363):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 8), 'int')
    # Processing the call keyword arguments
    kwargs_23972 = {}
    # Getting the type of 'call_assignment_22904' (line 363)
    call_assignment_22904_23969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'call_assignment_22904', False)
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___23970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), call_assignment_22904_23969, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23973 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23970, *[int_23971], **kwargs_23972)
    
    # Assigning a type to the variable 'call_assignment_22905' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'call_assignment_22905', getitem___call_result_23973)
    
    # Assigning a Name to a Name (line 363):
    # Getting the type of 'call_assignment_22905' (line 363)
    call_assignment_22905_23974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'call_assignment_22905')
    # Assigning a type to the variable 'cos_t1' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'cos_t1', call_assignment_22905_23974)
    
    # Assigning a Call to a Name (line 363):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 8), 'int')
    # Processing the call keyword arguments
    kwargs_23978 = {}
    # Getting the type of 'call_assignment_22904' (line 363)
    call_assignment_22904_23975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'call_assignment_22904', False)
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___23976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), call_assignment_22904_23975, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23979 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23976, *[int_23977], **kwargs_23978)
    
    # Assigning a type to the variable 'call_assignment_22906' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'call_assignment_22906', getitem___call_result_23979)
    
    # Assigning a Name to a Name (line 363):
    # Getting the type of 'call_assignment_22906' (line 363)
    call_assignment_22906_23980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'call_assignment_22906')
    # Assigning a type to the variable 'sin_t1' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'sin_t1', call_assignment_22906_23980)
    
    # Assigning a Tuple to a Tuple (line 364):
    
    # Assigning a Name to a Name (line 364):
    # Getting the type of 'cos_t1' (line 364)
    cos_t1_23981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 25), 'cos_t1')
    # Assigning a type to the variable 'tuple_assignment_22907' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'tuple_assignment_22907', cos_t1_23981)
    
    # Assigning a Name to a Name (line 364):
    # Getting the type of 'sin_t1' (line 364)
    sin_t1_23982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 33), 'sin_t1')
    # Assigning a type to the variable 'tuple_assignment_22908' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'tuple_assignment_22908', sin_t1_23982)
    
    # Assigning a Name to a Name (line 364):
    # Getting the type of 'tuple_assignment_22907' (line 364)
    tuple_assignment_22907_23983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'tuple_assignment_22907')
    # Assigning a type to the variable 'cos_t2' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'cos_t2', tuple_assignment_22907_23983)
    
    # Assigning a Name to a Name (line 364):
    # Getting the type of 'tuple_assignment_22908' (line 364)
    tuple_assignment_22908_23984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'tuple_assignment_22908')
    # Assigning a type to the variable 'sin_t2' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'sin_t2', tuple_assignment_22908_23984)
    # SSA branch for the else part of an if statement (line 360)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 368):
    
    # Assigning a Call to a Name:
    
    # Call to get_cos_sin(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'c1x' (line 368)
    c1x_23986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 37), 'c1x', False)
    # Getting the type of 'c1y' (line 368)
    c1y_23987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 42), 'c1y', False)
    # Getting the type of 'cmx' (line 368)
    cmx_23988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 47), 'cmx', False)
    # Getting the type of 'cmy' (line 368)
    cmy_23989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 52), 'cmy', False)
    # Processing the call keyword arguments (line 368)
    kwargs_23990 = {}
    # Getting the type of 'get_cos_sin' (line 368)
    get_cos_sin_23985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 25), 'get_cos_sin', False)
    # Calling get_cos_sin(args, kwargs) (line 368)
    get_cos_sin_call_result_23991 = invoke(stypy.reporting.localization.Localization(__file__, 368, 25), get_cos_sin_23985, *[c1x_23986, c1y_23987, cmx_23988, cmy_23989], **kwargs_23990)
    
    # Assigning a type to the variable 'call_assignment_22909' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_22909', get_cos_sin_call_result_23991)
    
    # Assigning a Call to a Name (line 368):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_23994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 8), 'int')
    # Processing the call keyword arguments
    kwargs_23995 = {}
    # Getting the type of 'call_assignment_22909' (line 368)
    call_assignment_22909_23992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_22909', False)
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___23993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), call_assignment_22909_23992, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_23996 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23993, *[int_23994], **kwargs_23995)
    
    # Assigning a type to the variable 'call_assignment_22910' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_22910', getitem___call_result_23996)
    
    # Assigning a Name to a Name (line 368):
    # Getting the type of 'call_assignment_22910' (line 368)
    call_assignment_22910_23997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_22910')
    # Assigning a type to the variable 'cos_t1' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'cos_t1', call_assignment_22910_23997)
    
    # Assigning a Call to a Name (line 368):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 8), 'int')
    # Processing the call keyword arguments
    kwargs_24001 = {}
    # Getting the type of 'call_assignment_22909' (line 368)
    call_assignment_22909_23998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_22909', False)
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___23999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), call_assignment_22909_23998, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24002 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___23999, *[int_24000], **kwargs_24001)
    
    # Assigning a type to the variable 'call_assignment_22911' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_22911', getitem___call_result_24002)
    
    # Assigning a Name to a Name (line 368):
    # Getting the type of 'call_assignment_22911' (line 368)
    call_assignment_22911_24003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_22911')
    # Assigning a type to the variable 'sin_t1' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'sin_t1', call_assignment_22911_24003)
    
    # Assigning a Call to a Tuple (line 369):
    
    # Assigning a Call to a Name:
    
    # Call to get_cos_sin(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'cmx' (line 369)
    cmx_24005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 37), 'cmx', False)
    # Getting the type of 'cmy' (line 369)
    cmy_24006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 42), 'cmy', False)
    # Getting the type of 'c2x' (line 369)
    c2x_24007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 47), 'c2x', False)
    # Getting the type of 'c2y' (line 369)
    c2y_24008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 52), 'c2y', False)
    # Processing the call keyword arguments (line 369)
    kwargs_24009 = {}
    # Getting the type of 'get_cos_sin' (line 369)
    get_cos_sin_24004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 25), 'get_cos_sin', False)
    # Calling get_cos_sin(args, kwargs) (line 369)
    get_cos_sin_call_result_24010 = invoke(stypy.reporting.localization.Localization(__file__, 369, 25), get_cos_sin_24004, *[cmx_24005, cmy_24006, c2x_24007, c2y_24008], **kwargs_24009)
    
    # Assigning a type to the variable 'call_assignment_22912' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'call_assignment_22912', get_cos_sin_call_result_24010)
    
    # Assigning a Call to a Name (line 369):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 8), 'int')
    # Processing the call keyword arguments
    kwargs_24014 = {}
    # Getting the type of 'call_assignment_22912' (line 369)
    call_assignment_22912_24011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'call_assignment_22912', False)
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___24012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), call_assignment_22912_24011, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24015 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24012, *[int_24013], **kwargs_24014)
    
    # Assigning a type to the variable 'call_assignment_22913' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'call_assignment_22913', getitem___call_result_24015)
    
    # Assigning a Name to a Name (line 369):
    # Getting the type of 'call_assignment_22913' (line 369)
    call_assignment_22913_24016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'call_assignment_22913')
    # Assigning a type to the variable 'cos_t2' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'cos_t2', call_assignment_22913_24016)
    
    # Assigning a Call to a Name (line 369):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 8), 'int')
    # Processing the call keyword arguments
    kwargs_24020 = {}
    # Getting the type of 'call_assignment_22912' (line 369)
    call_assignment_22912_24017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'call_assignment_22912', False)
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___24018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), call_assignment_22912_24017, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24021 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24018, *[int_24019], **kwargs_24020)
    
    # Assigning a type to the variable 'call_assignment_22914' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'call_assignment_22914', getitem___call_result_24021)
    
    # Assigning a Name to a Name (line 369):
    # Getting the type of 'call_assignment_22914' (line 369)
    call_assignment_22914_24022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'call_assignment_22914')
    # Assigning a type to the variable 'sin_t2' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'sin_t2', call_assignment_22914_24022)
    # SSA join for if statement (line 360)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 375):
    
    # Assigning a Call to a Name:
    
    # Call to get_normal_points(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of 'c1x' (line 376)
    c1x_24024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 26), 'c1x', False)
    # Getting the type of 'c1y' (line 376)
    c1y_24025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 31), 'c1y', False)
    # Getting the type of 'cos_t1' (line 376)
    cos_t1_24026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 36), 'cos_t1', False)
    # Getting the type of 'sin_t1' (line 376)
    sin_t1_24027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 44), 'sin_t1', False)
    # Getting the type of 'width' (line 376)
    width_24028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 52), 'width', False)
    # Processing the call keyword arguments (line 376)
    kwargs_24029 = {}
    # Getting the type of 'get_normal_points' (line 376)
    get_normal_points_24023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'get_normal_points', False)
    # Calling get_normal_points(args, kwargs) (line 376)
    get_normal_points_call_result_24030 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), get_normal_points_24023, *[c1x_24024, c1y_24025, cos_t1_24026, sin_t1_24027, width_24028], **kwargs_24029)
    
    # Assigning a type to the variable 'call_assignment_22915' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22915', get_normal_points_call_result_24030)
    
    # Assigning a Call to a Name (line 375):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24034 = {}
    # Getting the type of 'call_assignment_22915' (line 375)
    call_assignment_22915_24031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22915', False)
    # Obtaining the member '__getitem__' of a type (line 375)
    getitem___24032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 4), call_assignment_22915_24031, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24035 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24032, *[int_24033], **kwargs_24034)
    
    # Assigning a type to the variable 'call_assignment_22916' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22916', getitem___call_result_24035)
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'call_assignment_22916' (line 375)
    call_assignment_22916_24036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22916')
    # Assigning a type to the variable 'c1x_left' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'c1x_left', call_assignment_22916_24036)
    
    # Assigning a Call to a Name (line 375):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24040 = {}
    # Getting the type of 'call_assignment_22915' (line 375)
    call_assignment_22915_24037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22915', False)
    # Obtaining the member '__getitem__' of a type (line 375)
    getitem___24038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 4), call_assignment_22915_24037, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24041 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24038, *[int_24039], **kwargs_24040)
    
    # Assigning a type to the variable 'call_assignment_22917' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22917', getitem___call_result_24041)
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'call_assignment_22917' (line 375)
    call_assignment_22917_24042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22917')
    # Assigning a type to the variable 'c1y_left' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 14), 'c1y_left', call_assignment_22917_24042)
    
    # Assigning a Call to a Name (line 375):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24046 = {}
    # Getting the type of 'call_assignment_22915' (line 375)
    call_assignment_22915_24043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22915', False)
    # Obtaining the member '__getitem__' of a type (line 375)
    getitem___24044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 4), call_assignment_22915_24043, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24047 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24044, *[int_24045], **kwargs_24046)
    
    # Assigning a type to the variable 'call_assignment_22918' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22918', getitem___call_result_24047)
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'call_assignment_22918' (line 375)
    call_assignment_22918_24048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22918')
    # Assigning a type to the variable 'c1x_right' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 24), 'c1x_right', call_assignment_22918_24048)
    
    # Assigning a Call to a Name (line 375):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24052 = {}
    # Getting the type of 'call_assignment_22915' (line 375)
    call_assignment_22915_24049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22915', False)
    # Obtaining the member '__getitem__' of a type (line 375)
    getitem___24050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 4), call_assignment_22915_24049, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24053 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24050, *[int_24051], **kwargs_24052)
    
    # Assigning a type to the variable 'call_assignment_22919' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22919', getitem___call_result_24053)
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'call_assignment_22919' (line 375)
    call_assignment_22919_24054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'call_assignment_22919')
    # Assigning a type to the variable 'c1y_right' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 35), 'c1y_right', call_assignment_22919_24054)
    
    # Assigning a Call to a Tuple (line 378):
    
    # Assigning a Call to a Name:
    
    # Call to get_normal_points(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'c2x' (line 379)
    c2x_24056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 26), 'c2x', False)
    # Getting the type of 'c2y' (line 379)
    c2y_24057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 31), 'c2y', False)
    # Getting the type of 'cos_t2' (line 379)
    cos_t2_24058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 36), 'cos_t2', False)
    # Getting the type of 'sin_t2' (line 379)
    sin_t2_24059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 44), 'sin_t2', False)
    # Getting the type of 'width' (line 379)
    width_24060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 52), 'width', False)
    # Processing the call keyword arguments (line 379)
    kwargs_24061 = {}
    # Getting the type of 'get_normal_points' (line 379)
    get_normal_points_24055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'get_normal_points', False)
    # Calling get_normal_points(args, kwargs) (line 379)
    get_normal_points_call_result_24062 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), get_normal_points_24055, *[c2x_24056, c2y_24057, cos_t2_24058, sin_t2_24059, width_24060], **kwargs_24061)
    
    # Assigning a type to the variable 'call_assignment_22920' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22920', get_normal_points_call_result_24062)
    
    # Assigning a Call to a Name (line 378):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24066 = {}
    # Getting the type of 'call_assignment_22920' (line 378)
    call_assignment_22920_24063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22920', False)
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___24064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 4), call_assignment_22920_24063, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24067 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24064, *[int_24065], **kwargs_24066)
    
    # Assigning a type to the variable 'call_assignment_22921' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22921', getitem___call_result_24067)
    
    # Assigning a Name to a Name (line 378):
    # Getting the type of 'call_assignment_22921' (line 378)
    call_assignment_22921_24068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22921')
    # Assigning a type to the variable 'c2x_left' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'c2x_left', call_assignment_22921_24068)
    
    # Assigning a Call to a Name (line 378):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24072 = {}
    # Getting the type of 'call_assignment_22920' (line 378)
    call_assignment_22920_24069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22920', False)
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___24070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 4), call_assignment_22920_24069, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24073 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24070, *[int_24071], **kwargs_24072)
    
    # Assigning a type to the variable 'call_assignment_22922' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22922', getitem___call_result_24073)
    
    # Assigning a Name to a Name (line 378):
    # Getting the type of 'call_assignment_22922' (line 378)
    call_assignment_22922_24074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22922')
    # Assigning a type to the variable 'c2y_left' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 14), 'c2y_left', call_assignment_22922_24074)
    
    # Assigning a Call to a Name (line 378):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24078 = {}
    # Getting the type of 'call_assignment_22920' (line 378)
    call_assignment_22920_24075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22920', False)
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___24076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 4), call_assignment_22920_24075, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24079 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24076, *[int_24077], **kwargs_24078)
    
    # Assigning a type to the variable 'call_assignment_22923' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22923', getitem___call_result_24079)
    
    # Assigning a Name to a Name (line 378):
    # Getting the type of 'call_assignment_22923' (line 378)
    call_assignment_22923_24080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22923')
    # Assigning a type to the variable 'c2x_right' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 24), 'c2x_right', call_assignment_22923_24080)
    
    # Assigning a Call to a Name (line 378):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24084 = {}
    # Getting the type of 'call_assignment_22920' (line 378)
    call_assignment_22920_24081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22920', False)
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___24082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 4), call_assignment_22920_24081, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24085 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24082, *[int_24083], **kwargs_24084)
    
    # Assigning a type to the variable 'call_assignment_22924' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22924', getitem___call_result_24085)
    
    # Assigning a Name to a Name (line 378):
    # Getting the type of 'call_assignment_22924' (line 378)
    call_assignment_22924_24086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'call_assignment_22924')
    # Assigning a type to the variable 'c2y_right' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 35), 'c2y_right', call_assignment_22924_24086)
    
    
    # Getting the type of 'parallel_test' (line 385)
    parallel_test_24087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 7), 'parallel_test')
    int_24088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 24), 'int')
    # Applying the binary operator '!=' (line 385)
    result_ne_24089 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 7), '!=', parallel_test_24087, int_24088)
    
    # Testing the type of an if condition (line 385)
    if_condition_24090 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 4), result_ne_24089)
    # Assigning a type to the variable 'if_condition_24090' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'if_condition_24090', if_condition_24090)
    # SSA begins for if statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 388):
    
    # Assigning a BinOp to a Name (line 388):
    float_24091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 12), 'float')
    # Getting the type of 'c1x_left' (line 389)
    c1x_left_24092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'c1x_left')
    # Getting the type of 'c2x_left' (line 389)
    c2x_left_24093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 30), 'c2x_left')
    # Applying the binary operator '+' (line 389)
    result_add_24094 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 19), '+', c1x_left_24092, c2x_left_24093)
    
    # Applying the binary operator '*' (line 389)
    result_mul_24095 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 12), '*', float_24091, result_add_24094)
    
    # Assigning a type to the variable 'tuple_assignment_22925' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_assignment_22925', result_mul_24095)
    
    # Assigning a BinOp to a Name (line 388):
    float_24096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 41), 'float')
    # Getting the type of 'c1y_left' (line 389)
    c1y_left_24097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 48), 'c1y_left')
    # Getting the type of 'c2y_left' (line 389)
    c2y_left_24098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 59), 'c2y_left')
    # Applying the binary operator '+' (line 389)
    result_add_24099 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 48), '+', c1y_left_24097, c2y_left_24098)
    
    # Applying the binary operator '*' (line 389)
    result_mul_24100 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 41), '*', float_24096, result_add_24099)
    
    # Assigning a type to the variable 'tuple_assignment_22926' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_assignment_22926', result_mul_24100)
    
    # Assigning a Name to a Name (line 388):
    # Getting the type of 'tuple_assignment_22925' (line 388)
    tuple_assignment_22925_24101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_assignment_22925')
    # Assigning a type to the variable 'cmx_left' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'cmx_left', tuple_assignment_22925_24101)
    
    # Assigning a Name to a Name (line 388):
    # Getting the type of 'tuple_assignment_22926' (line 388)
    tuple_assignment_22926_24102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_assignment_22926')
    # Assigning a type to the variable 'cmy_left' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 18), 'cmy_left', tuple_assignment_22926_24102)
    
    # Assigning a Tuple to a Tuple (line 391):
    
    # Assigning a BinOp to a Name (line 391):
    float_24103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 12), 'float')
    # Getting the type of 'c1x_right' (line 392)
    c1x_right_24104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'c1x_right')
    # Getting the type of 'c2x_right' (line 392)
    c2x_right_24105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 31), 'c2x_right')
    # Applying the binary operator '+' (line 392)
    result_add_24106 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 19), '+', c1x_right_24104, c2x_right_24105)
    
    # Applying the binary operator '*' (line 392)
    result_mul_24107 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 12), '*', float_24103, result_add_24106)
    
    # Assigning a type to the variable 'tuple_assignment_22927' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'tuple_assignment_22927', result_mul_24107)
    
    # Assigning a BinOp to a Name (line 391):
    float_24108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 43), 'float')
    # Getting the type of 'c1y_right' (line 392)
    c1y_right_24109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 50), 'c1y_right')
    # Getting the type of 'c2y_right' (line 392)
    c2y_right_24110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 62), 'c2y_right')
    # Applying the binary operator '+' (line 392)
    result_add_24111 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 50), '+', c1y_right_24109, c2y_right_24110)
    
    # Applying the binary operator '*' (line 392)
    result_mul_24112 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 43), '*', float_24108, result_add_24111)
    
    # Assigning a type to the variable 'tuple_assignment_22928' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'tuple_assignment_22928', result_mul_24112)
    
    # Assigning a Name to a Name (line 391):
    # Getting the type of 'tuple_assignment_22927' (line 391)
    tuple_assignment_22927_24113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'tuple_assignment_22927')
    # Assigning a type to the variable 'cmx_right' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'cmx_right', tuple_assignment_22927_24113)
    
    # Assigning a Name to a Name (line 391):
    # Getting the type of 'tuple_assignment_22928' (line 391)
    tuple_assignment_22928_24114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'tuple_assignment_22928')
    # Assigning a type to the variable 'cmy_right' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 19), 'cmy_right', tuple_assignment_22928_24114)
    # SSA branch for the else part of an if statement (line 385)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 395):
    
    # Assigning a Call to a Name:
    
    # Call to get_intersection(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'c1x_left' (line 395)
    c1x_left_24116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 46), 'c1x_left', False)
    # Getting the type of 'c1y_left' (line 395)
    c1y_left_24117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 56), 'c1y_left', False)
    # Getting the type of 'cos_t1' (line 395)
    cos_t1_24118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 66), 'cos_t1', False)
    # Getting the type of 'sin_t1' (line 396)
    sin_t1_24119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 46), 'sin_t1', False)
    # Getting the type of 'c2x_left' (line 396)
    c2x_left_24120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 54), 'c2x_left', False)
    # Getting the type of 'c2y_left' (line 396)
    c2y_left_24121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 64), 'c2y_left', False)
    # Getting the type of 'cos_t2' (line 397)
    cos_t2_24122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 46), 'cos_t2', False)
    # Getting the type of 'sin_t2' (line 397)
    sin_t2_24123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 54), 'sin_t2', False)
    # Processing the call keyword arguments (line 395)
    kwargs_24124 = {}
    # Getting the type of 'get_intersection' (line 395)
    get_intersection_24115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 29), 'get_intersection', False)
    # Calling get_intersection(args, kwargs) (line 395)
    get_intersection_call_result_24125 = invoke(stypy.reporting.localization.Localization(__file__, 395, 29), get_intersection_24115, *[c1x_left_24116, c1y_left_24117, cos_t1_24118, sin_t1_24119, c2x_left_24120, c2y_left_24121, cos_t2_24122, sin_t2_24123], **kwargs_24124)
    
    # Assigning a type to the variable 'call_assignment_22929' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'call_assignment_22929', get_intersection_call_result_24125)
    
    # Assigning a Call to a Name (line 395):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'int')
    # Processing the call keyword arguments
    kwargs_24129 = {}
    # Getting the type of 'call_assignment_22929' (line 395)
    call_assignment_22929_24126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'call_assignment_22929', False)
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___24127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), call_assignment_22929_24126, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24130 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24127, *[int_24128], **kwargs_24129)
    
    # Assigning a type to the variable 'call_assignment_22930' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'call_assignment_22930', getitem___call_result_24130)
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'call_assignment_22930' (line 395)
    call_assignment_22930_24131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'call_assignment_22930')
    # Assigning a type to the variable 'cmx_left' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'cmx_left', call_assignment_22930_24131)
    
    # Assigning a Call to a Name (line 395):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'int')
    # Processing the call keyword arguments
    kwargs_24135 = {}
    # Getting the type of 'call_assignment_22929' (line 395)
    call_assignment_22929_24132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'call_assignment_22929', False)
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___24133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), call_assignment_22929_24132, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24136 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24133, *[int_24134], **kwargs_24135)
    
    # Assigning a type to the variable 'call_assignment_22931' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'call_assignment_22931', getitem___call_result_24136)
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'call_assignment_22931' (line 395)
    call_assignment_22931_24137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'call_assignment_22931')
    # Assigning a type to the variable 'cmy_left' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 18), 'cmy_left', call_assignment_22931_24137)
    
    # Assigning a Call to a Tuple (line 399):
    
    # Assigning a Call to a Name:
    
    # Call to get_intersection(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'c1x_right' (line 399)
    c1x_right_24139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 48), 'c1x_right', False)
    # Getting the type of 'c1y_right' (line 399)
    c1y_right_24140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 59), 'c1y_right', False)
    # Getting the type of 'cos_t1' (line 399)
    cos_t1_24141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 70), 'cos_t1', False)
    # Getting the type of 'sin_t1' (line 400)
    sin_t1_24142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 48), 'sin_t1', False)
    # Getting the type of 'c2x_right' (line 400)
    c2x_right_24143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 56), 'c2x_right', False)
    # Getting the type of 'c2y_right' (line 400)
    c2y_right_24144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 67), 'c2y_right', False)
    # Getting the type of 'cos_t2' (line 401)
    cos_t2_24145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 48), 'cos_t2', False)
    # Getting the type of 'sin_t2' (line 401)
    sin_t2_24146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 56), 'sin_t2', False)
    # Processing the call keyword arguments (line 399)
    kwargs_24147 = {}
    # Getting the type of 'get_intersection' (line 399)
    get_intersection_24138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'get_intersection', False)
    # Calling get_intersection(args, kwargs) (line 399)
    get_intersection_call_result_24148 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), get_intersection_24138, *[c1x_right_24139, c1y_right_24140, cos_t1_24141, sin_t1_24142, c2x_right_24143, c2y_right_24144, cos_t2_24145, sin_t2_24146], **kwargs_24147)
    
    # Assigning a type to the variable 'call_assignment_22932' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'call_assignment_22932', get_intersection_call_result_24148)
    
    # Assigning a Call to a Name (line 399):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 8), 'int')
    # Processing the call keyword arguments
    kwargs_24152 = {}
    # Getting the type of 'call_assignment_22932' (line 399)
    call_assignment_22932_24149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'call_assignment_22932', False)
    # Obtaining the member '__getitem__' of a type (line 399)
    getitem___24150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), call_assignment_22932_24149, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24153 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24150, *[int_24151], **kwargs_24152)
    
    # Assigning a type to the variable 'call_assignment_22933' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'call_assignment_22933', getitem___call_result_24153)
    
    # Assigning a Name to a Name (line 399):
    # Getting the type of 'call_assignment_22933' (line 399)
    call_assignment_22933_24154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'call_assignment_22933')
    # Assigning a type to the variable 'cmx_right' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'cmx_right', call_assignment_22933_24154)
    
    # Assigning a Call to a Name (line 399):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 8), 'int')
    # Processing the call keyword arguments
    kwargs_24158 = {}
    # Getting the type of 'call_assignment_22932' (line 399)
    call_assignment_22932_24155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'call_assignment_22932', False)
    # Obtaining the member '__getitem__' of a type (line 399)
    getitem___24156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), call_assignment_22932_24155, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24159 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24156, *[int_24157], **kwargs_24158)
    
    # Assigning a type to the variable 'call_assignment_22934' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'call_assignment_22934', getitem___call_result_24159)
    
    # Assigning a Name to a Name (line 399):
    # Getting the type of 'call_assignment_22934' (line 399)
    call_assignment_22934_24160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'call_assignment_22934')
    # Assigning a type to the variable 'cmy_right' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 19), 'cmy_right', call_assignment_22934_24160)
    # SSA join for if statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 405):
    
    # Assigning a List to a Name (line 405):
    
    # Obtaining an instance of the builtin type 'list' (line 405)
    list_24161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 405)
    # Adding element type (line 405)
    
    # Obtaining an instance of the builtin type 'tuple' (line 405)
    tuple_24162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 405)
    # Adding element type (line 405)
    # Getting the type of 'c1x_left' (line 405)
    c1x_left_24163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 18), 'c1x_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 18), tuple_24162, c1x_left_24163)
    # Adding element type (line 405)
    # Getting the type of 'c1y_left' (line 405)
    c1y_left_24164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 28), 'c1y_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 18), tuple_24162, c1y_left_24164)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 16), list_24161, tuple_24162)
    # Adding element type (line 405)
    
    # Obtaining an instance of the builtin type 'tuple' (line 406)
    tuple_24165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 406)
    # Adding element type (line 406)
    # Getting the type of 'cmx_left' (line 406)
    cmx_left_24166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 18), 'cmx_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 18), tuple_24165, cmx_left_24166)
    # Adding element type (line 406)
    # Getting the type of 'cmy_left' (line 406)
    cmy_left_24167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 28), 'cmy_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 18), tuple_24165, cmy_left_24167)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 16), list_24161, tuple_24165)
    # Adding element type (line 405)
    
    # Obtaining an instance of the builtin type 'tuple' (line 407)
    tuple_24168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 407)
    # Adding element type (line 407)
    # Getting the type of 'c2x_left' (line 407)
    c2x_left_24169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'c2x_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 18), tuple_24168, c2x_left_24169)
    # Adding element type (line 407)
    # Getting the type of 'c2y_left' (line 407)
    c2y_left_24170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 28), 'c2y_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 18), tuple_24168, c2y_left_24170)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 16), list_24161, tuple_24168)
    
    # Assigning a type to the variable 'path_left' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'path_left', list_24161)
    
    # Assigning a List to a Name (line 408):
    
    # Assigning a List to a Name (line 408):
    
    # Obtaining an instance of the builtin type 'list' (line 408)
    list_24171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 408)
    # Adding element type (line 408)
    
    # Obtaining an instance of the builtin type 'tuple' (line 408)
    tuple_24172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 408)
    # Adding element type (line 408)
    # Getting the type of 'c1x_right' (line 408)
    c1x_right_24173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 19), 'c1x_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 19), tuple_24172, c1x_right_24173)
    # Adding element type (line 408)
    # Getting the type of 'c1y_right' (line 408)
    c1y_right_24174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 30), 'c1y_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 19), tuple_24172, c1y_right_24174)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 17), list_24171, tuple_24172)
    # Adding element type (line 408)
    
    # Obtaining an instance of the builtin type 'tuple' (line 409)
    tuple_24175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 409)
    # Adding element type (line 409)
    # Getting the type of 'cmx_right' (line 409)
    cmx_right_24176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 19), 'cmx_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 19), tuple_24175, cmx_right_24176)
    # Adding element type (line 409)
    # Getting the type of 'cmy_right' (line 409)
    cmy_right_24177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 30), 'cmy_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 19), tuple_24175, cmy_right_24177)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 17), list_24171, tuple_24175)
    # Adding element type (line 408)
    
    # Obtaining an instance of the builtin type 'tuple' (line 410)
    tuple_24178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 410)
    # Adding element type (line 410)
    # Getting the type of 'c2x_right' (line 410)
    c2x_right_24179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 'c2x_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 19), tuple_24178, c2x_right_24179)
    # Adding element type (line 410)
    # Getting the type of 'c2y_right' (line 410)
    c2y_right_24180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 30), 'c2y_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 19), tuple_24178, c2y_right_24180)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 17), list_24171, tuple_24178)
    
    # Assigning a type to the variable 'path_right' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'path_right', list_24171)
    
    # Obtaining an instance of the builtin type 'tuple' (line 412)
    tuple_24181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 412)
    # Adding element type (line 412)
    # Getting the type of 'path_left' (line 412)
    path_left_24182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 11), 'path_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 11), tuple_24181, path_left_24182)
    # Adding element type (line 412)
    # Getting the type of 'path_right' (line 412)
    path_right_24183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 22), 'path_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 11), tuple_24181, path_right_24183)
    
    # Assigning a type to the variable 'stypy_return_type' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type', tuple_24181)
    
    # ################# End of 'get_parallels(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_parallels' in the type store
    # Getting the type of 'stypy_return_type' (line 341)
    stypy_return_type_24184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24184)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_parallels'
    return stypy_return_type_24184

# Assigning a type to the variable 'get_parallels' (line 341)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), 'get_parallels', get_parallels)

@norecursion
def find_control_points(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_control_points'
    module_type_store = module_type_store.open_function_context('find_control_points', 415, 0, False)
    
    # Passed parameters checking function
    find_control_points.stypy_localization = localization
    find_control_points.stypy_type_of_self = None
    find_control_points.stypy_type_store = module_type_store
    find_control_points.stypy_function_name = 'find_control_points'
    find_control_points.stypy_param_names_list = ['c1x', 'c1y', 'mmx', 'mmy', 'c2x', 'c2y']
    find_control_points.stypy_varargs_param_name = None
    find_control_points.stypy_kwargs_param_name = None
    find_control_points.stypy_call_defaults = defaults
    find_control_points.stypy_call_varargs = varargs
    find_control_points.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_control_points', ['c1x', 'c1y', 'mmx', 'mmy', 'c2x', 'c2y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_control_points', localization, ['c1x', 'c1y', 'mmx', 'mmy', 'c2x', 'c2y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_control_points(...)' code ##################

    unicode_24185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, (-1)), 'unicode', u' Find control points of the bezier line throught c1, mm, c2. We\n    simply assume that c1, mm, c2 which have parametric value 0, 0.5, and 1.\n    ')
    
    # Assigning a BinOp to a Name (line 420):
    
    # Assigning a BinOp to a Name (line 420):
    float_24186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 10), 'float')
    int_24187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 16), 'int')
    # Getting the type of 'mmx' (line 420)
    mmx_24188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 20), 'mmx')
    # Applying the binary operator '*' (line 420)
    result_mul_24189 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 16), '*', int_24187, mmx_24188)
    
    # Getting the type of 'c1x' (line 420)
    c1x_24190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 27), 'c1x')
    # Getting the type of 'c2x' (line 420)
    c2x_24191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 33), 'c2x')
    # Applying the binary operator '+' (line 420)
    result_add_24192 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 27), '+', c1x_24190, c2x_24191)
    
    # Applying the binary operator '-' (line 420)
    result_sub_24193 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 16), '-', result_mul_24189, result_add_24192)
    
    # Applying the binary operator '*' (line 420)
    result_mul_24194 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 10), '*', float_24186, result_sub_24193)
    
    # Assigning a type to the variable 'cmx' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'cmx', result_mul_24194)
    
    # Assigning a BinOp to a Name (line 421):
    
    # Assigning a BinOp to a Name (line 421):
    float_24195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 10), 'float')
    int_24196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 16), 'int')
    # Getting the type of 'mmy' (line 421)
    mmy_24197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 20), 'mmy')
    # Applying the binary operator '*' (line 421)
    result_mul_24198 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 16), '*', int_24196, mmy_24197)
    
    # Getting the type of 'c1y' (line 421)
    c1y_24199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), 'c1y')
    # Getting the type of 'c2y' (line 421)
    c2y_24200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 33), 'c2y')
    # Applying the binary operator '+' (line 421)
    result_add_24201 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 27), '+', c1y_24199, c2y_24200)
    
    # Applying the binary operator '-' (line 421)
    result_sub_24202 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 16), '-', result_mul_24198, result_add_24201)
    
    # Applying the binary operator '*' (line 421)
    result_mul_24203 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 10), '*', float_24195, result_sub_24202)
    
    # Assigning a type to the variable 'cmy' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'cmy', result_mul_24203)
    
    # Obtaining an instance of the builtin type 'list' (line 423)
    list_24204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 423)
    # Adding element type (line 423)
    
    # Obtaining an instance of the builtin type 'tuple' (line 423)
    tuple_24205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 423)
    # Adding element type (line 423)
    # Getting the type of 'c1x' (line 423)
    c1x_24206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 13), 'c1x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 13), tuple_24205, c1x_24206)
    # Adding element type (line 423)
    # Getting the type of 'c1y' (line 423)
    c1y_24207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 18), 'c1y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 13), tuple_24205, c1y_24207)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 11), list_24204, tuple_24205)
    # Adding element type (line 423)
    
    # Obtaining an instance of the builtin type 'tuple' (line 423)
    tuple_24208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 423)
    # Adding element type (line 423)
    # Getting the type of 'cmx' (line 423)
    cmx_24209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 25), 'cmx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 25), tuple_24208, cmx_24209)
    # Adding element type (line 423)
    # Getting the type of 'cmy' (line 423)
    cmy_24210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 30), 'cmy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 25), tuple_24208, cmy_24210)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 11), list_24204, tuple_24208)
    # Adding element type (line 423)
    
    # Obtaining an instance of the builtin type 'tuple' (line 423)
    tuple_24211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 423)
    # Adding element type (line 423)
    # Getting the type of 'c2x' (line 423)
    c2x_24212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 37), 'c2x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 37), tuple_24211, c2x_24212)
    # Adding element type (line 423)
    # Getting the type of 'c2y' (line 423)
    c2y_24213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 42), 'c2y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 37), tuple_24211, c2y_24213)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 11), list_24204, tuple_24211)
    
    # Assigning a type to the variable 'stypy_return_type' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type', list_24204)
    
    # ################# End of 'find_control_points(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_control_points' in the type store
    # Getting the type of 'stypy_return_type' (line 415)
    stypy_return_type_24214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24214)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_control_points'
    return stypy_return_type_24214

# Assigning a type to the variable 'find_control_points' (line 415)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 0), 'find_control_points', find_control_points)

@norecursion
def make_wedged_bezier2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_24215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 43), 'float')
    float_24216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 50), 'float')
    float_24217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 58), 'float')
    defaults = [float_24215, float_24216, float_24217]
    # Create a new context for function 'make_wedged_bezier2'
    module_type_store = module_type_store.open_function_context('make_wedged_bezier2', 426, 0, False)
    
    # Passed parameters checking function
    make_wedged_bezier2.stypy_localization = localization
    make_wedged_bezier2.stypy_type_of_self = None
    make_wedged_bezier2.stypy_type_store = module_type_store
    make_wedged_bezier2.stypy_function_name = 'make_wedged_bezier2'
    make_wedged_bezier2.stypy_param_names_list = ['bezier2', 'width', 'w1', 'wm', 'w2']
    make_wedged_bezier2.stypy_varargs_param_name = None
    make_wedged_bezier2.stypy_kwargs_param_name = None
    make_wedged_bezier2.stypy_call_defaults = defaults
    make_wedged_bezier2.stypy_call_varargs = varargs
    make_wedged_bezier2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_wedged_bezier2', ['bezier2', 'width', 'w1', 'wm', 'w2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_wedged_bezier2', localization, ['bezier2', 'width', 'w1', 'wm', 'w2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_wedged_bezier2(...)' code ##################

    unicode_24218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, (-1)), 'unicode', u'\n    Being similar to get_parallels, returns control points of two quadrativ\n    bezier lines having a width roughly parralel to given one separated by\n    *width*.\n    ')
    
    # Assigning a Subscript to a Tuple (line 434):
    
    # Assigning a Subscript to a Name (line 434):
    
    # Obtaining the type of the subscript
    int_24219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 23), 'int')
    # Getting the type of 'bezier2' (line 434)
    bezier2_24221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___24222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 15), bezier2_24221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_24223 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), getitem___24222, int_24220)
    
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___24224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 4), subscript_call_result_24223, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_24225 = invoke(stypy.reporting.localization.Localization(__file__, 434, 4), getitem___24224, int_24219)
    
    # Assigning a type to the variable 'tuple_var_assignment_22935' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'tuple_var_assignment_22935', subscript_call_result_24225)
    
    # Assigning a Subscript to a Name (line 434):
    
    # Obtaining the type of the subscript
    int_24226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 23), 'int')
    # Getting the type of 'bezier2' (line 434)
    bezier2_24228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___24229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 15), bezier2_24228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_24230 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), getitem___24229, int_24227)
    
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___24231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 4), subscript_call_result_24230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_24232 = invoke(stypy.reporting.localization.Localization(__file__, 434, 4), getitem___24231, int_24226)
    
    # Assigning a type to the variable 'tuple_var_assignment_22936' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'tuple_var_assignment_22936', subscript_call_result_24232)
    
    # Assigning a Name to a Name (line 434):
    # Getting the type of 'tuple_var_assignment_22935' (line 434)
    tuple_var_assignment_22935_24233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'tuple_var_assignment_22935')
    # Assigning a type to the variable 'c1x' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'c1x', tuple_var_assignment_22935_24233)
    
    # Assigning a Name to a Name (line 434):
    # Getting the type of 'tuple_var_assignment_22936' (line 434)
    tuple_var_assignment_22936_24234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'tuple_var_assignment_22936')
    # Assigning a type to the variable 'c1y' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 9), 'c1y', tuple_var_assignment_22936_24234)
    
    # Assigning a Subscript to a Tuple (line 435):
    
    # Assigning a Subscript to a Name (line 435):
    
    # Obtaining the type of the subscript
    int_24235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 23), 'int')
    # Getting the type of 'bezier2' (line 435)
    bezier2_24237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___24238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 15), bezier2_24237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_24239 = invoke(stypy.reporting.localization.Localization(__file__, 435, 15), getitem___24238, int_24236)
    
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___24240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 4), subscript_call_result_24239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_24241 = invoke(stypy.reporting.localization.Localization(__file__, 435, 4), getitem___24240, int_24235)
    
    # Assigning a type to the variable 'tuple_var_assignment_22937' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'tuple_var_assignment_22937', subscript_call_result_24241)
    
    # Assigning a Subscript to a Name (line 435):
    
    # Obtaining the type of the subscript
    int_24242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 23), 'int')
    # Getting the type of 'bezier2' (line 435)
    bezier2_24244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___24245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 15), bezier2_24244, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_24246 = invoke(stypy.reporting.localization.Localization(__file__, 435, 15), getitem___24245, int_24243)
    
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___24247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 4), subscript_call_result_24246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_24248 = invoke(stypy.reporting.localization.Localization(__file__, 435, 4), getitem___24247, int_24242)
    
    # Assigning a type to the variable 'tuple_var_assignment_22938' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'tuple_var_assignment_22938', subscript_call_result_24248)
    
    # Assigning a Name to a Name (line 435):
    # Getting the type of 'tuple_var_assignment_22937' (line 435)
    tuple_var_assignment_22937_24249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'tuple_var_assignment_22937')
    # Assigning a type to the variable 'cmx' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'cmx', tuple_var_assignment_22937_24249)
    
    # Assigning a Name to a Name (line 435):
    # Getting the type of 'tuple_var_assignment_22938' (line 435)
    tuple_var_assignment_22938_24250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'tuple_var_assignment_22938')
    # Assigning a type to the variable 'cmy' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 9), 'cmy', tuple_var_assignment_22938_24250)
    
    # Assigning a Subscript to a Tuple (line 436):
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    int_24251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 23), 'int')
    # Getting the type of 'bezier2' (line 436)
    bezier2_24253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___24254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 15), bezier2_24253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_24255 = invoke(stypy.reporting.localization.Localization(__file__, 436, 15), getitem___24254, int_24252)
    
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___24256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 4), subscript_call_result_24255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_24257 = invoke(stypy.reporting.localization.Localization(__file__, 436, 4), getitem___24256, int_24251)
    
    # Assigning a type to the variable 'tuple_var_assignment_22939' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'tuple_var_assignment_22939', subscript_call_result_24257)
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    int_24258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 23), 'int')
    # Getting the type of 'bezier2' (line 436)
    bezier2_24260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 15), 'bezier2')
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___24261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 15), bezier2_24260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_24262 = invoke(stypy.reporting.localization.Localization(__file__, 436, 15), getitem___24261, int_24259)
    
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___24263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 4), subscript_call_result_24262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_24264 = invoke(stypy.reporting.localization.Localization(__file__, 436, 4), getitem___24263, int_24258)
    
    # Assigning a type to the variable 'tuple_var_assignment_22940' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'tuple_var_assignment_22940', subscript_call_result_24264)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'tuple_var_assignment_22939' (line 436)
    tuple_var_assignment_22939_24265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'tuple_var_assignment_22939')
    # Assigning a type to the variable 'c3x' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'c3x', tuple_var_assignment_22939_24265)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'tuple_var_assignment_22940' (line 436)
    tuple_var_assignment_22940_24266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'tuple_var_assignment_22940')
    # Assigning a type to the variable 'c3y' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 9), 'c3y', tuple_var_assignment_22940_24266)
    
    # Assigning a Call to a Tuple (line 440):
    
    # Assigning a Call to a Name:
    
    # Call to get_cos_sin(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'c1x' (line 440)
    c1x_24268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 33), 'c1x', False)
    # Getting the type of 'c1y' (line 440)
    c1y_24269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 38), 'c1y', False)
    # Getting the type of 'cmx' (line 440)
    cmx_24270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 43), 'cmx', False)
    # Getting the type of 'cmy' (line 440)
    cmy_24271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 48), 'cmy', False)
    # Processing the call keyword arguments (line 440)
    kwargs_24272 = {}
    # Getting the type of 'get_cos_sin' (line 440)
    get_cos_sin_24267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 21), 'get_cos_sin', False)
    # Calling get_cos_sin(args, kwargs) (line 440)
    get_cos_sin_call_result_24273 = invoke(stypy.reporting.localization.Localization(__file__, 440, 21), get_cos_sin_24267, *[c1x_24268, c1y_24269, cmx_24270, cmy_24271], **kwargs_24272)
    
    # Assigning a type to the variable 'call_assignment_22941' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'call_assignment_22941', get_cos_sin_call_result_24273)
    
    # Assigning a Call to a Name (line 440):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24277 = {}
    # Getting the type of 'call_assignment_22941' (line 440)
    call_assignment_22941_24274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'call_assignment_22941', False)
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___24275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 4), call_assignment_22941_24274, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24278 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24275, *[int_24276], **kwargs_24277)
    
    # Assigning a type to the variable 'call_assignment_22942' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'call_assignment_22942', getitem___call_result_24278)
    
    # Assigning a Name to a Name (line 440):
    # Getting the type of 'call_assignment_22942' (line 440)
    call_assignment_22942_24279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'call_assignment_22942')
    # Assigning a type to the variable 'cos_t1' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'cos_t1', call_assignment_22942_24279)
    
    # Assigning a Call to a Name (line 440):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24283 = {}
    # Getting the type of 'call_assignment_22941' (line 440)
    call_assignment_22941_24280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'call_assignment_22941', False)
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___24281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 4), call_assignment_22941_24280, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24284 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24281, *[int_24282], **kwargs_24283)
    
    # Assigning a type to the variable 'call_assignment_22943' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'call_assignment_22943', getitem___call_result_24284)
    
    # Assigning a Name to a Name (line 440):
    # Getting the type of 'call_assignment_22943' (line 440)
    call_assignment_22943_24285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'call_assignment_22943')
    # Assigning a type to the variable 'sin_t1' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'sin_t1', call_assignment_22943_24285)
    
    # Assigning a Call to a Tuple (line 441):
    
    # Assigning a Call to a Name:
    
    # Call to get_cos_sin(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'cmx' (line 441)
    cmx_24287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 33), 'cmx', False)
    # Getting the type of 'cmy' (line 441)
    cmy_24288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 38), 'cmy', False)
    # Getting the type of 'c3x' (line 441)
    c3x_24289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 43), 'c3x', False)
    # Getting the type of 'c3y' (line 441)
    c3y_24290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 48), 'c3y', False)
    # Processing the call keyword arguments (line 441)
    kwargs_24291 = {}
    # Getting the type of 'get_cos_sin' (line 441)
    get_cos_sin_24286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 21), 'get_cos_sin', False)
    # Calling get_cos_sin(args, kwargs) (line 441)
    get_cos_sin_call_result_24292 = invoke(stypy.reporting.localization.Localization(__file__, 441, 21), get_cos_sin_24286, *[cmx_24287, cmy_24288, c3x_24289, c3y_24290], **kwargs_24291)
    
    # Assigning a type to the variable 'call_assignment_22944' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'call_assignment_22944', get_cos_sin_call_result_24292)
    
    # Assigning a Call to a Name (line 441):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24296 = {}
    # Getting the type of 'call_assignment_22944' (line 441)
    call_assignment_22944_24293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'call_assignment_22944', False)
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___24294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 4), call_assignment_22944_24293, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24297 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24294, *[int_24295], **kwargs_24296)
    
    # Assigning a type to the variable 'call_assignment_22945' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'call_assignment_22945', getitem___call_result_24297)
    
    # Assigning a Name to a Name (line 441):
    # Getting the type of 'call_assignment_22945' (line 441)
    call_assignment_22945_24298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'call_assignment_22945')
    # Assigning a type to the variable 'cos_t2' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'cos_t2', call_assignment_22945_24298)
    
    # Assigning a Call to a Name (line 441):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24302 = {}
    # Getting the type of 'call_assignment_22944' (line 441)
    call_assignment_22944_24299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'call_assignment_22944', False)
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___24300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 4), call_assignment_22944_24299, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24303 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24300, *[int_24301], **kwargs_24302)
    
    # Assigning a type to the variable 'call_assignment_22946' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'call_assignment_22946', getitem___call_result_24303)
    
    # Assigning a Name to a Name (line 441):
    # Getting the type of 'call_assignment_22946' (line 441)
    call_assignment_22946_24304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'call_assignment_22946')
    # Assigning a type to the variable 'sin_t2' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'sin_t2', call_assignment_22946_24304)
    
    # Assigning a Call to a Tuple (line 447):
    
    # Assigning a Call to a Name:
    
    # Call to get_normal_points(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'c1x' (line 448)
    c1x_24306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 26), 'c1x', False)
    # Getting the type of 'c1y' (line 448)
    c1y_24307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 31), 'c1y', False)
    # Getting the type of 'cos_t1' (line 448)
    cos_t1_24308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 36), 'cos_t1', False)
    # Getting the type of 'sin_t1' (line 448)
    sin_t1_24309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 44), 'sin_t1', False)
    # Getting the type of 'width' (line 448)
    width_24310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 52), 'width', False)
    # Getting the type of 'w1' (line 448)
    w1_24311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 60), 'w1', False)
    # Applying the binary operator '*' (line 448)
    result_mul_24312 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 52), '*', width_24310, w1_24311)
    
    # Processing the call keyword arguments (line 448)
    kwargs_24313 = {}
    # Getting the type of 'get_normal_points' (line 448)
    get_normal_points_24305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'get_normal_points', False)
    # Calling get_normal_points(args, kwargs) (line 448)
    get_normal_points_call_result_24314 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), get_normal_points_24305, *[c1x_24306, c1y_24307, cos_t1_24308, sin_t1_24309, result_mul_24312], **kwargs_24313)
    
    # Assigning a type to the variable 'call_assignment_22947' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22947', get_normal_points_call_result_24314)
    
    # Assigning a Call to a Name (line 447):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24318 = {}
    # Getting the type of 'call_assignment_22947' (line 447)
    call_assignment_22947_24315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22947', False)
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___24316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 4), call_assignment_22947_24315, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24319 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24316, *[int_24317], **kwargs_24318)
    
    # Assigning a type to the variable 'call_assignment_22948' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22948', getitem___call_result_24319)
    
    # Assigning a Name to a Name (line 447):
    # Getting the type of 'call_assignment_22948' (line 447)
    call_assignment_22948_24320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22948')
    # Assigning a type to the variable 'c1x_left' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'c1x_left', call_assignment_22948_24320)
    
    # Assigning a Call to a Name (line 447):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24324 = {}
    # Getting the type of 'call_assignment_22947' (line 447)
    call_assignment_22947_24321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22947', False)
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___24322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 4), call_assignment_22947_24321, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24325 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24322, *[int_24323], **kwargs_24324)
    
    # Assigning a type to the variable 'call_assignment_22949' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22949', getitem___call_result_24325)
    
    # Assigning a Name to a Name (line 447):
    # Getting the type of 'call_assignment_22949' (line 447)
    call_assignment_22949_24326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22949')
    # Assigning a type to the variable 'c1y_left' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 14), 'c1y_left', call_assignment_22949_24326)
    
    # Assigning a Call to a Name (line 447):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24330 = {}
    # Getting the type of 'call_assignment_22947' (line 447)
    call_assignment_22947_24327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22947', False)
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___24328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 4), call_assignment_22947_24327, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24331 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24328, *[int_24329], **kwargs_24330)
    
    # Assigning a type to the variable 'call_assignment_22950' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22950', getitem___call_result_24331)
    
    # Assigning a Name to a Name (line 447):
    # Getting the type of 'call_assignment_22950' (line 447)
    call_assignment_22950_24332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22950')
    # Assigning a type to the variable 'c1x_right' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'c1x_right', call_assignment_22950_24332)
    
    # Assigning a Call to a Name (line 447):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24336 = {}
    # Getting the type of 'call_assignment_22947' (line 447)
    call_assignment_22947_24333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22947', False)
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___24334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 4), call_assignment_22947_24333, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24337 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24334, *[int_24335], **kwargs_24336)
    
    # Assigning a type to the variable 'call_assignment_22951' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22951', getitem___call_result_24337)
    
    # Assigning a Name to a Name (line 447):
    # Getting the type of 'call_assignment_22951' (line 447)
    call_assignment_22951_24338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'call_assignment_22951')
    # Assigning a type to the variable 'c1y_right' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 35), 'c1y_right', call_assignment_22951_24338)
    
    # Assigning a Call to a Tuple (line 450):
    
    # Assigning a Call to a Name:
    
    # Call to get_normal_points(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'c3x' (line 451)
    c3x_24340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 26), 'c3x', False)
    # Getting the type of 'c3y' (line 451)
    c3y_24341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 31), 'c3y', False)
    # Getting the type of 'cos_t2' (line 451)
    cos_t2_24342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 36), 'cos_t2', False)
    # Getting the type of 'sin_t2' (line 451)
    sin_t2_24343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 44), 'sin_t2', False)
    # Getting the type of 'width' (line 451)
    width_24344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 52), 'width', False)
    # Getting the type of 'w2' (line 451)
    w2_24345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 60), 'w2', False)
    # Applying the binary operator '*' (line 451)
    result_mul_24346 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 52), '*', width_24344, w2_24345)
    
    # Processing the call keyword arguments (line 451)
    kwargs_24347 = {}
    # Getting the type of 'get_normal_points' (line 451)
    get_normal_points_24339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'get_normal_points', False)
    # Calling get_normal_points(args, kwargs) (line 451)
    get_normal_points_call_result_24348 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), get_normal_points_24339, *[c3x_24340, c3y_24341, cos_t2_24342, sin_t2_24343, result_mul_24346], **kwargs_24347)
    
    # Assigning a type to the variable 'call_assignment_22952' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22952', get_normal_points_call_result_24348)
    
    # Assigning a Call to a Name (line 450):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24352 = {}
    # Getting the type of 'call_assignment_22952' (line 450)
    call_assignment_22952_24349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22952', False)
    # Obtaining the member '__getitem__' of a type (line 450)
    getitem___24350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 4), call_assignment_22952_24349, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24353 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24350, *[int_24351], **kwargs_24352)
    
    # Assigning a type to the variable 'call_assignment_22953' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22953', getitem___call_result_24353)
    
    # Assigning a Name to a Name (line 450):
    # Getting the type of 'call_assignment_22953' (line 450)
    call_assignment_22953_24354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22953')
    # Assigning a type to the variable 'c3x_left' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'c3x_left', call_assignment_22953_24354)
    
    # Assigning a Call to a Name (line 450):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24358 = {}
    # Getting the type of 'call_assignment_22952' (line 450)
    call_assignment_22952_24355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22952', False)
    # Obtaining the member '__getitem__' of a type (line 450)
    getitem___24356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 4), call_assignment_22952_24355, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24359 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24356, *[int_24357], **kwargs_24358)
    
    # Assigning a type to the variable 'call_assignment_22954' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22954', getitem___call_result_24359)
    
    # Assigning a Name to a Name (line 450):
    # Getting the type of 'call_assignment_22954' (line 450)
    call_assignment_22954_24360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22954')
    # Assigning a type to the variable 'c3y_left' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 14), 'c3y_left', call_assignment_22954_24360)
    
    # Assigning a Call to a Name (line 450):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24364 = {}
    # Getting the type of 'call_assignment_22952' (line 450)
    call_assignment_22952_24361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22952', False)
    # Obtaining the member '__getitem__' of a type (line 450)
    getitem___24362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 4), call_assignment_22952_24361, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24365 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24362, *[int_24363], **kwargs_24364)
    
    # Assigning a type to the variable 'call_assignment_22955' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22955', getitem___call_result_24365)
    
    # Assigning a Name to a Name (line 450):
    # Getting the type of 'call_assignment_22955' (line 450)
    call_assignment_22955_24366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22955')
    # Assigning a type to the variable 'c3x_right' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'c3x_right', call_assignment_22955_24366)
    
    # Assigning a Call to a Name (line 450):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24370 = {}
    # Getting the type of 'call_assignment_22952' (line 450)
    call_assignment_22952_24367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22952', False)
    # Obtaining the member '__getitem__' of a type (line 450)
    getitem___24368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 4), call_assignment_22952_24367, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24371 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24368, *[int_24369], **kwargs_24370)
    
    # Assigning a type to the variable 'call_assignment_22956' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22956', getitem___call_result_24371)
    
    # Assigning a Name to a Name (line 450):
    # Getting the type of 'call_assignment_22956' (line 450)
    call_assignment_22956_24372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'call_assignment_22956')
    # Assigning a type to the variable 'c3y_right' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 35), 'c3y_right', call_assignment_22956_24372)
    
    # Assigning a Tuple to a Tuple (line 456):
    
    # Assigning a BinOp to a Name (line 456):
    # Getting the type of 'c1x' (line 456)
    c1x_24373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 18), 'c1x')
    # Getting the type of 'cmx' (line 456)
    cmx_24374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 24), 'cmx')
    # Applying the binary operator '+' (line 456)
    result_add_24375 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 18), '+', c1x_24373, cmx_24374)
    
    float_24376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 31), 'float')
    # Applying the binary operator '*' (line 456)
    result_mul_24377 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 17), '*', result_add_24375, float_24376)
    
    # Assigning a type to the variable 'tuple_assignment_22957' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'tuple_assignment_22957', result_mul_24377)
    
    # Assigning a BinOp to a Name (line 456):
    # Getting the type of 'c1y' (line 456)
    c1y_24378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 36), 'c1y')
    # Getting the type of 'cmy' (line 456)
    cmy_24379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'cmy')
    # Applying the binary operator '+' (line 456)
    result_add_24380 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 36), '+', c1y_24378, cmy_24379)
    
    float_24381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 49), 'float')
    # Applying the binary operator '*' (line 456)
    result_mul_24382 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 35), '*', result_add_24380, float_24381)
    
    # Assigning a type to the variable 'tuple_assignment_22958' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'tuple_assignment_22958', result_mul_24382)
    
    # Assigning a Name to a Name (line 456):
    # Getting the type of 'tuple_assignment_22957' (line 456)
    tuple_assignment_22957_24383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'tuple_assignment_22957')
    # Assigning a type to the variable 'c12x' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'c12x', tuple_assignment_22957_24383)
    
    # Assigning a Name to a Name (line 456):
    # Getting the type of 'tuple_assignment_22958' (line 456)
    tuple_assignment_22958_24384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'tuple_assignment_22958')
    # Assigning a type to the variable 'c12y' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 10), 'c12y', tuple_assignment_22958_24384)
    
    # Assigning a Tuple to a Tuple (line 457):
    
    # Assigning a BinOp to a Name (line 457):
    # Getting the type of 'cmx' (line 457)
    cmx_24385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 18), 'cmx')
    # Getting the type of 'c3x' (line 457)
    c3x_24386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 24), 'c3x')
    # Applying the binary operator '+' (line 457)
    result_add_24387 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 18), '+', cmx_24385, c3x_24386)
    
    float_24388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 31), 'float')
    # Applying the binary operator '*' (line 457)
    result_mul_24389 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 17), '*', result_add_24387, float_24388)
    
    # Assigning a type to the variable 'tuple_assignment_22959' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'tuple_assignment_22959', result_mul_24389)
    
    # Assigning a BinOp to a Name (line 457):
    # Getting the type of 'cmy' (line 457)
    cmy_24390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 36), 'cmy')
    # Getting the type of 'c3y' (line 457)
    c3y_24391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 42), 'c3y')
    # Applying the binary operator '+' (line 457)
    result_add_24392 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 36), '+', cmy_24390, c3y_24391)
    
    float_24393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 49), 'float')
    # Applying the binary operator '*' (line 457)
    result_mul_24394 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 35), '*', result_add_24392, float_24393)
    
    # Assigning a type to the variable 'tuple_assignment_22960' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'tuple_assignment_22960', result_mul_24394)
    
    # Assigning a Name to a Name (line 457):
    # Getting the type of 'tuple_assignment_22959' (line 457)
    tuple_assignment_22959_24395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'tuple_assignment_22959')
    # Assigning a type to the variable 'c23x' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'c23x', tuple_assignment_22959_24395)
    
    # Assigning a Name to a Name (line 457):
    # Getting the type of 'tuple_assignment_22960' (line 457)
    tuple_assignment_22960_24396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'tuple_assignment_22960')
    # Assigning a type to the variable 'c23y' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 10), 'c23y', tuple_assignment_22960_24396)
    
    # Assigning a Tuple to a Tuple (line 458):
    
    # Assigning a BinOp to a Name (line 458):
    # Getting the type of 'c12x' (line 458)
    c12x_24397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 20), 'c12x')
    # Getting the type of 'c23x' (line 458)
    c23x_24398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 27), 'c23x')
    # Applying the binary operator '+' (line 458)
    result_add_24399 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 20), '+', c12x_24397, c23x_24398)
    
    float_24400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 35), 'float')
    # Applying the binary operator '*' (line 458)
    result_mul_24401 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 19), '*', result_add_24399, float_24400)
    
    # Assigning a type to the variable 'tuple_assignment_22961' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'tuple_assignment_22961', result_mul_24401)
    
    # Assigning a BinOp to a Name (line 458):
    # Getting the type of 'c12y' (line 458)
    c12y_24402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 40), 'c12y')
    # Getting the type of 'c23y' (line 458)
    c23y_24403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 47), 'c23y')
    # Applying the binary operator '+' (line 458)
    result_add_24404 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 40), '+', c12y_24402, c23y_24403)
    
    float_24405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 55), 'float')
    # Applying the binary operator '*' (line 458)
    result_mul_24406 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 39), '*', result_add_24404, float_24405)
    
    # Assigning a type to the variable 'tuple_assignment_22962' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'tuple_assignment_22962', result_mul_24406)
    
    # Assigning a Name to a Name (line 458):
    # Getting the type of 'tuple_assignment_22961' (line 458)
    tuple_assignment_22961_24407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'tuple_assignment_22961')
    # Assigning a type to the variable 'c123x' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'c123x', tuple_assignment_22961_24407)
    
    # Assigning a Name to a Name (line 458):
    # Getting the type of 'tuple_assignment_22962' (line 458)
    tuple_assignment_22962_24408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'tuple_assignment_22962')
    # Assigning a type to the variable 'c123y' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 11), 'c123y', tuple_assignment_22962_24408)
    
    # Assigning a Call to a Tuple (line 461):
    
    # Assigning a Call to a Name:
    
    # Call to get_cos_sin(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'c12x' (line 461)
    c12x_24410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 37), 'c12x', False)
    # Getting the type of 'c12y' (line 461)
    c12y_24411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 43), 'c12y', False)
    # Getting the type of 'c23x' (line 461)
    c23x_24412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 49), 'c23x', False)
    # Getting the type of 'c23y' (line 461)
    c23y_24413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 55), 'c23y', False)
    # Processing the call keyword arguments (line 461)
    kwargs_24414 = {}
    # Getting the type of 'get_cos_sin' (line 461)
    get_cos_sin_24409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 25), 'get_cos_sin', False)
    # Calling get_cos_sin(args, kwargs) (line 461)
    get_cos_sin_call_result_24415 = invoke(stypy.reporting.localization.Localization(__file__, 461, 25), get_cos_sin_24409, *[c12x_24410, c12y_24411, c23x_24412, c23y_24413], **kwargs_24414)
    
    # Assigning a type to the variable 'call_assignment_22963' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'call_assignment_22963', get_cos_sin_call_result_24415)
    
    # Assigning a Call to a Name (line 461):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24419 = {}
    # Getting the type of 'call_assignment_22963' (line 461)
    call_assignment_22963_24416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'call_assignment_22963', False)
    # Obtaining the member '__getitem__' of a type (line 461)
    getitem___24417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 4), call_assignment_22963_24416, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24420 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24417, *[int_24418], **kwargs_24419)
    
    # Assigning a type to the variable 'call_assignment_22964' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'call_assignment_22964', getitem___call_result_24420)
    
    # Assigning a Name to a Name (line 461):
    # Getting the type of 'call_assignment_22964' (line 461)
    call_assignment_22964_24421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'call_assignment_22964')
    # Assigning a type to the variable 'cos_t123' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'cos_t123', call_assignment_22964_24421)
    
    # Assigning a Call to a Name (line 461):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24425 = {}
    # Getting the type of 'call_assignment_22963' (line 461)
    call_assignment_22963_24422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'call_assignment_22963', False)
    # Obtaining the member '__getitem__' of a type (line 461)
    getitem___24423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 4), call_assignment_22963_24422, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24426 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24423, *[int_24424], **kwargs_24425)
    
    # Assigning a type to the variable 'call_assignment_22965' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'call_assignment_22965', getitem___call_result_24426)
    
    # Assigning a Name to a Name (line 461):
    # Getting the type of 'call_assignment_22965' (line 461)
    call_assignment_22965_24427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'call_assignment_22965')
    # Assigning a type to the variable 'sin_t123' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 14), 'sin_t123', call_assignment_22965_24427)
    
    # Assigning a Call to a Tuple (line 463):
    
    # Assigning a Call to a Name:
    
    # Call to get_normal_points(...): (line 464)
    # Processing the call arguments (line 464)
    # Getting the type of 'c123x' (line 464)
    c123x_24429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 26), 'c123x', False)
    # Getting the type of 'c123y' (line 464)
    c123y_24430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 33), 'c123y', False)
    # Getting the type of 'cos_t123' (line 464)
    cos_t123_24431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 40), 'cos_t123', False)
    # Getting the type of 'sin_t123' (line 464)
    sin_t123_24432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 50), 'sin_t123', False)
    # Getting the type of 'width' (line 464)
    width_24433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 60), 'width', False)
    # Getting the type of 'wm' (line 464)
    wm_24434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 68), 'wm', False)
    # Applying the binary operator '*' (line 464)
    result_mul_24435 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 60), '*', width_24433, wm_24434)
    
    # Processing the call keyword arguments (line 464)
    kwargs_24436 = {}
    # Getting the type of 'get_normal_points' (line 464)
    get_normal_points_24428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'get_normal_points', False)
    # Calling get_normal_points(args, kwargs) (line 464)
    get_normal_points_call_result_24437 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), get_normal_points_24428, *[c123x_24429, c123y_24430, cos_t123_24431, sin_t123_24432, result_mul_24435], **kwargs_24436)
    
    # Assigning a type to the variable 'call_assignment_22966' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22966', get_normal_points_call_result_24437)
    
    # Assigning a Call to a Name (line 463):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24441 = {}
    # Getting the type of 'call_assignment_22966' (line 463)
    call_assignment_22966_24438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22966', False)
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___24439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 4), call_assignment_22966_24438, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24442 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24439, *[int_24440], **kwargs_24441)
    
    # Assigning a type to the variable 'call_assignment_22967' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22967', getitem___call_result_24442)
    
    # Assigning a Name to a Name (line 463):
    # Getting the type of 'call_assignment_22967' (line 463)
    call_assignment_22967_24443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22967')
    # Assigning a type to the variable 'c123x_left' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'c123x_left', call_assignment_22967_24443)
    
    # Assigning a Call to a Name (line 463):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24447 = {}
    # Getting the type of 'call_assignment_22966' (line 463)
    call_assignment_22966_24444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22966', False)
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___24445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 4), call_assignment_22966_24444, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24448 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24445, *[int_24446], **kwargs_24447)
    
    # Assigning a type to the variable 'call_assignment_22968' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22968', getitem___call_result_24448)
    
    # Assigning a Name to a Name (line 463):
    # Getting the type of 'call_assignment_22968' (line 463)
    call_assignment_22968_24449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22968')
    # Assigning a type to the variable 'c123y_left' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'c123y_left', call_assignment_22968_24449)
    
    # Assigning a Call to a Name (line 463):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24453 = {}
    # Getting the type of 'call_assignment_22966' (line 463)
    call_assignment_22966_24450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22966', False)
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___24451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 4), call_assignment_22966_24450, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24454 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24451, *[int_24452], **kwargs_24453)
    
    # Assigning a type to the variable 'call_assignment_22969' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22969', getitem___call_result_24454)
    
    # Assigning a Name to a Name (line 463):
    # Getting the type of 'call_assignment_22969' (line 463)
    call_assignment_22969_24455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22969')
    # Assigning a type to the variable 'c123x_right' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 28), 'c123x_right', call_assignment_22969_24455)
    
    # Assigning a Call to a Name (line 463):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_24458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 4), 'int')
    # Processing the call keyword arguments
    kwargs_24459 = {}
    # Getting the type of 'call_assignment_22966' (line 463)
    call_assignment_22966_24456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22966', False)
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___24457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 4), call_assignment_22966_24456, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_24460 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___24457, *[int_24458], **kwargs_24459)
    
    # Assigning a type to the variable 'call_assignment_22970' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22970', getitem___call_result_24460)
    
    # Assigning a Name to a Name (line 463):
    # Getting the type of 'call_assignment_22970' (line 463)
    call_assignment_22970_24461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'call_assignment_22970')
    # Assigning a type to the variable 'c123y_right' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 41), 'c123y_right', call_assignment_22970_24461)
    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to find_control_points(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'c1x_left' (line 467)
    c1x_left_24463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 36), 'c1x_left', False)
    # Getting the type of 'c1y_left' (line 467)
    c1y_left_24464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 46), 'c1y_left', False)
    # Getting the type of 'c123x_left' (line 468)
    c123x_left_24465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 36), 'c123x_left', False)
    # Getting the type of 'c123y_left' (line 468)
    c123y_left_24466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 48), 'c123y_left', False)
    # Getting the type of 'c3x_left' (line 469)
    c3x_left_24467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 36), 'c3x_left', False)
    # Getting the type of 'c3y_left' (line 469)
    c3y_left_24468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 46), 'c3y_left', False)
    # Processing the call keyword arguments (line 467)
    kwargs_24469 = {}
    # Getting the type of 'find_control_points' (line 467)
    find_control_points_24462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'find_control_points', False)
    # Calling find_control_points(args, kwargs) (line 467)
    find_control_points_call_result_24470 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), find_control_points_24462, *[c1x_left_24463, c1y_left_24464, c123x_left_24465, c123y_left_24466, c3x_left_24467, c3y_left_24468], **kwargs_24469)
    
    # Assigning a type to the variable 'path_left' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'path_left', find_control_points_call_result_24470)
    
    # Assigning a Call to a Name (line 470):
    
    # Assigning a Call to a Name (line 470):
    
    # Call to find_control_points(...): (line 470)
    # Processing the call arguments (line 470)
    # Getting the type of 'c1x_right' (line 470)
    c1x_right_24472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 37), 'c1x_right', False)
    # Getting the type of 'c1y_right' (line 470)
    c1y_right_24473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 48), 'c1y_right', False)
    # Getting the type of 'c123x_right' (line 471)
    c123x_right_24474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 37), 'c123x_right', False)
    # Getting the type of 'c123y_right' (line 471)
    c123y_right_24475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 50), 'c123y_right', False)
    # Getting the type of 'c3x_right' (line 472)
    c3x_right_24476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 37), 'c3x_right', False)
    # Getting the type of 'c3y_right' (line 472)
    c3y_right_24477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 48), 'c3y_right', False)
    # Processing the call keyword arguments (line 470)
    kwargs_24478 = {}
    # Getting the type of 'find_control_points' (line 470)
    find_control_points_24471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 17), 'find_control_points', False)
    # Calling find_control_points(args, kwargs) (line 470)
    find_control_points_call_result_24479 = invoke(stypy.reporting.localization.Localization(__file__, 470, 17), find_control_points_24471, *[c1x_right_24472, c1y_right_24473, c123x_right_24474, c123y_right_24475, c3x_right_24476, c3y_right_24477], **kwargs_24478)
    
    # Assigning a type to the variable 'path_right' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'path_right', find_control_points_call_result_24479)
    
    # Obtaining an instance of the builtin type 'tuple' (line 474)
    tuple_24480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 474)
    # Adding element type (line 474)
    # Getting the type of 'path_left' (line 474)
    path_left_24481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'path_left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 11), tuple_24480, path_left_24481)
    # Adding element type (line 474)
    # Getting the type of 'path_right' (line 474)
    path_right_24482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 22), 'path_right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 11), tuple_24480, path_right_24482)
    
    # Assigning a type to the variable 'stypy_return_type' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type', tuple_24480)
    
    # ################# End of 'make_wedged_bezier2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_wedged_bezier2' in the type store
    # Getting the type of 'stypy_return_type' (line 426)
    stypy_return_type_24483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24483)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_wedged_bezier2'
    return stypy_return_type_24483

# Assigning a type to the variable 'make_wedged_bezier2' (line 426)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'make_wedged_bezier2', make_wedged_bezier2)

@norecursion
def make_path_regular(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_path_regular'
    module_type_store = module_type_store.open_function_context('make_path_regular', 477, 0, False)
    
    # Passed parameters checking function
    make_path_regular.stypy_localization = localization
    make_path_regular.stypy_type_of_self = None
    make_path_regular.stypy_type_store = module_type_store
    make_path_regular.stypy_function_name = 'make_path_regular'
    make_path_regular.stypy_param_names_list = ['p']
    make_path_regular.stypy_varargs_param_name = None
    make_path_regular.stypy_kwargs_param_name = None
    make_path_regular.stypy_call_defaults = defaults
    make_path_regular.stypy_call_varargs = varargs
    make_path_regular.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_path_regular', ['p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_path_regular', localization, ['p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_path_regular(...)' code ##################

    unicode_24484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, (-1)), 'unicode', u'\n    fill in the codes if None.\n    ')
    
    # Assigning a Attribute to a Name (line 481):
    
    # Assigning a Attribute to a Name (line 481):
    # Getting the type of 'p' (line 481)
    p_24485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'p')
    # Obtaining the member 'codes' of a type (line 481)
    codes_24486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), p_24485, 'codes')
    # Assigning a type to the variable 'c' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'c', codes_24486)
    
    # Type idiom detected: calculating its left and rigth part (line 482)
    # Getting the type of 'c' (line 482)
    c_24487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 7), 'c')
    # Getting the type of 'None' (line 482)
    None_24488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'None')
    
    (may_be_24489, more_types_in_union_24490) = may_be_none(c_24487, None_24488)

    if may_be_24489:

        if more_types_in_union_24490:
            # Runtime conditional SSA (line 482)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Call to empty(...): (line 483)
        # Processing the call arguments (line 483)
        
        # Obtaining the type of the subscript
        int_24493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 39), 'int')
        slice_24494 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 483, 21), None, int_24493, None)
        # Getting the type of 'p' (line 483)
        p_24495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 21), 'p', False)
        # Obtaining the member 'vertices' of a type (line 483)
        vertices_24496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 21), p_24495, 'vertices')
        # Obtaining the member 'shape' of a type (line 483)
        shape_24497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 21), vertices_24496, 'shape')
        # Obtaining the member '__getitem__' of a type (line 483)
        getitem___24498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 21), shape_24497, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 483)
        subscript_call_result_24499 = invoke(stypy.reporting.localization.Localization(__file__, 483, 21), getitem___24498, slice_24494)
        
        unicode_24500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 43), 'unicode', u'i')
        # Processing the call keyword arguments (line 483)
        kwargs_24501 = {}
        # Getting the type of 'np' (line 483)
        np_24491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'np', False)
        # Obtaining the member 'empty' of a type (line 483)
        empty_24492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 12), np_24491, 'empty')
        # Calling empty(args, kwargs) (line 483)
        empty_call_result_24502 = invoke(stypy.reporting.localization.Localization(__file__, 483, 12), empty_24492, *[subscript_call_result_24499, unicode_24500], **kwargs_24501)
        
        # Assigning a type to the variable 'c' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'c', empty_call_result_24502)
        
        # Call to fill(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'Path' (line 484)
        Path_24505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 15), 'Path', False)
        # Obtaining the member 'LINETO' of a type (line 484)
        LINETO_24506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 15), Path_24505, 'LINETO')
        # Processing the call keyword arguments (line 484)
        kwargs_24507 = {}
        # Getting the type of 'c' (line 484)
        c_24503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'c', False)
        # Obtaining the member 'fill' of a type (line 484)
        fill_24504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), c_24503, 'fill')
        # Calling fill(args, kwargs) (line 484)
        fill_call_result_24508 = invoke(stypy.reporting.localization.Localization(__file__, 484, 8), fill_24504, *[LINETO_24506], **kwargs_24507)
        
        
        # Assigning a Attribute to a Subscript (line 485):
        
        # Assigning a Attribute to a Subscript (line 485):
        # Getting the type of 'Path' (line 485)
        Path_24509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 15), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 485)
        MOVETO_24510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 15), Path_24509, 'MOVETO')
        # Getting the type of 'c' (line 485)
        c_24511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'c')
        int_24512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 10), 'int')
        # Storing an element on a container (line 485)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 8), c_24511, (int_24512, MOVETO_24510))
        
        # Call to Path(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'p' (line 487)
        p_24514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 20), 'p', False)
        # Obtaining the member 'vertices' of a type (line 487)
        vertices_24515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 20), p_24514, 'vertices')
        # Getting the type of 'c' (line 487)
        c_24516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 32), 'c', False)
        # Processing the call keyword arguments (line 487)
        kwargs_24517 = {}
        # Getting the type of 'Path' (line 487)
        Path_24513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 15), 'Path', False)
        # Calling Path(args, kwargs) (line 487)
        Path_call_result_24518 = invoke(stypy.reporting.localization.Localization(__file__, 487, 15), Path_24513, *[vertices_24515, c_24516], **kwargs_24517)
        
        # Assigning a type to the variable 'stypy_return_type' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'stypy_return_type', Path_call_result_24518)

        if more_types_in_union_24490:
            # Runtime conditional SSA for else branch (line 482)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_24489) or more_types_in_union_24490):
        # Getting the type of 'p' (line 489)
        p_24519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'stypy_return_type', p_24519)

        if (may_be_24489 and more_types_in_union_24490):
            # SSA join for if statement (line 482)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'make_path_regular(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_path_regular' in the type store
    # Getting the type of 'stypy_return_type' (line 477)
    stypy_return_type_24520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24520)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_path_regular'
    return stypy_return_type_24520

# Assigning a type to the variable 'make_path_regular' (line 477)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), 'make_path_regular', make_path_regular)

@norecursion
def concatenate_paths(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'concatenate_paths'
    module_type_store = module_type_store.open_function_context('concatenate_paths', 492, 0, False)
    
    # Passed parameters checking function
    concatenate_paths.stypy_localization = localization
    concatenate_paths.stypy_type_of_self = None
    concatenate_paths.stypy_type_store = module_type_store
    concatenate_paths.stypy_function_name = 'concatenate_paths'
    concatenate_paths.stypy_param_names_list = ['paths']
    concatenate_paths.stypy_varargs_param_name = None
    concatenate_paths.stypy_kwargs_param_name = None
    concatenate_paths.stypy_call_defaults = defaults
    concatenate_paths.stypy_call_varargs = varargs
    concatenate_paths.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'concatenate_paths', ['paths'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'concatenate_paths', localization, ['paths'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'concatenate_paths(...)' code ##################

    unicode_24521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, (-1)), 'unicode', u'\n    concatenate list of paths into a single path.\n    ')
    
    # Assigning a List to a Name (line 497):
    
    # Assigning a List to a Name (line 497):
    
    # Obtaining an instance of the builtin type 'list' (line 497)
    list_24522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 497)
    
    # Assigning a type to the variable 'vertices' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'vertices', list_24522)
    
    # Assigning a List to a Name (line 498):
    
    # Assigning a List to a Name (line 498):
    
    # Obtaining an instance of the builtin type 'list' (line 498)
    list_24523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 498)
    
    # Assigning a type to the variable 'codes' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'codes', list_24523)
    
    # Getting the type of 'paths' (line 499)
    paths_24524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'paths')
    # Testing the type of a for loop iterable (line 499)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 499, 4), paths_24524)
    # Getting the type of the for loop variable (line 499)
    for_loop_var_24525 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 499, 4), paths_24524)
    # Assigning a type to the variable 'p' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'p', for_loop_var_24525)
    # SSA begins for a for statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 500):
    
    # Assigning a Call to a Name (line 500):
    
    # Call to make_path_regular(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'p' (line 500)
    p_24527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 30), 'p', False)
    # Processing the call keyword arguments (line 500)
    kwargs_24528 = {}
    # Getting the type of 'make_path_regular' (line 500)
    make_path_regular_24526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'make_path_regular', False)
    # Calling make_path_regular(args, kwargs) (line 500)
    make_path_regular_call_result_24529 = invoke(stypy.reporting.localization.Localization(__file__, 500, 12), make_path_regular_24526, *[p_24527], **kwargs_24528)
    
    # Assigning a type to the variable 'p' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'p', make_path_regular_call_result_24529)
    
    # Call to append(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'p' (line 501)
    p_24532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), 'p', False)
    # Obtaining the member 'vertices' of a type (line 501)
    vertices_24533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 24), p_24532, 'vertices')
    # Processing the call keyword arguments (line 501)
    kwargs_24534 = {}
    # Getting the type of 'vertices' (line 501)
    vertices_24530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'vertices', False)
    # Obtaining the member 'append' of a type (line 501)
    append_24531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), vertices_24530, 'append')
    # Calling append(args, kwargs) (line 501)
    append_call_result_24535 = invoke(stypy.reporting.localization.Localization(__file__, 501, 8), append_24531, *[vertices_24533], **kwargs_24534)
    
    
    # Call to append(...): (line 502)
    # Processing the call arguments (line 502)
    # Getting the type of 'p' (line 502)
    p_24538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 21), 'p', False)
    # Obtaining the member 'codes' of a type (line 502)
    codes_24539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 21), p_24538, 'codes')
    # Processing the call keyword arguments (line 502)
    kwargs_24540 = {}
    # Getting the type of 'codes' (line 502)
    codes_24536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'codes', False)
    # Obtaining the member 'append' of a type (line 502)
    append_24537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), codes_24536, 'append')
    # Calling append(args, kwargs) (line 502)
    append_call_result_24541 = invoke(stypy.reporting.localization.Localization(__file__, 502, 8), append_24537, *[codes_24539], **kwargs_24540)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to Path(...): (line 504)
    # Processing the call arguments (line 504)
    
    # Call to concatenate(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'vertices' (line 504)
    vertices_24545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 32), 'vertices', False)
    # Processing the call keyword arguments (line 504)
    kwargs_24546 = {}
    # Getting the type of 'np' (line 504)
    np_24543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 17), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 504)
    concatenate_24544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 17), np_24543, 'concatenate')
    # Calling concatenate(args, kwargs) (line 504)
    concatenate_call_result_24547 = invoke(stypy.reporting.localization.Localization(__file__, 504, 17), concatenate_24544, *[vertices_24545], **kwargs_24546)
    
    
    # Call to concatenate(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'codes' (line 505)
    codes_24550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'codes', False)
    # Processing the call keyword arguments (line 505)
    kwargs_24551 = {}
    # Getting the type of 'np' (line 505)
    np_24548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 17), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 505)
    concatenate_24549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 17), np_24548, 'concatenate')
    # Calling concatenate(args, kwargs) (line 505)
    concatenate_call_result_24552 = invoke(stypy.reporting.localization.Localization(__file__, 505, 17), concatenate_24549, *[codes_24550], **kwargs_24551)
    
    # Processing the call keyword arguments (line 504)
    kwargs_24553 = {}
    # Getting the type of 'Path' (line 504)
    Path_24542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'Path', False)
    # Calling Path(args, kwargs) (line 504)
    Path_call_result_24554 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), Path_24542, *[concatenate_call_result_24547, concatenate_call_result_24552], **kwargs_24553)
    
    # Assigning a type to the variable '_path' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), '_path', Path_call_result_24554)
    # Getting the type of '_path' (line 506)
    _path_24555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 11), '_path')
    # Assigning a type to the variable 'stypy_return_type' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type', _path_24555)
    
    # ################# End of 'concatenate_paths(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'concatenate_paths' in the type store
    # Getting the type of 'stypy_return_type' (line 492)
    stypy_return_type_24556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24556)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'concatenate_paths'
    return stypy_return_type_24556

# Assigning a type to the variable 'concatenate_paths' (line 492)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 0), 'concatenate_paths', concatenate_paths)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
