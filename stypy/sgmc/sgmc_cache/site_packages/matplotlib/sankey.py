
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module for creating Sankey diagrams using matplotlib
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import six
8: from six.moves import zip
9: import numpy as np
10: 
11: from matplotlib.cbook import iterable, Bunch
12: from matplotlib.path import Path
13: from matplotlib.patches import PathPatch
14: from matplotlib.transforms import Affine2D
15: from matplotlib import verbose
16: from matplotlib import docstring
17: from matplotlib import rcParams
18: 
19: __author__ = "Kevin L. Davies"
20: __credits__ = ["Yannick Copin"]
21: __license__ = "BSD"
22: __version__ = "2011/09/16"
23: 
24: # Angles [deg/90]
25: RIGHT = 0
26: UP = 1
27: # LEFT = 2
28: DOWN = 3
29: 
30: 
31: class Sankey(object):
32:     '''
33:     Sankey diagram in matplotlib
34: 
35:       Sankey diagrams are a specific type of flow diagram, in which
36:       the width of the arrows is shown proportionally to the flow
37:       quantity.  They are typically used to visualize energy or
38:       material or cost transfers between processes.
39:       `Wikipedia (6/1/2011) <https://en.wikipedia.org/wiki/Sankey_diagram>`_
40: 
41:     '''
42: 
43:     def __init__(self, ax=None, scale=1.0, unit='', format='%G', gap=0.25,
44:                  radius=0.1, shoulder=0.03, offset=0.15, head_angle=100,
45:                  margin=0.4, tolerance=1e-6, **kwargs):
46:         '''
47:         Create a new Sankey instance.
48: 
49:         Optional keyword arguments:
50: 
51:           ===============   ===================================================
52:           Field             Description
53:           ===============   ===================================================
54:           *ax*              axes onto which the data should be plotted
55:                             If *ax* isn't provided, new axes will be created.
56:           *scale*           scaling factor for the flows
57:                             *scale* sizes the width of the paths in order to
58:                             maintain proper layout.  The same scale is applied
59:                             to all subdiagrams.  The value should be chosen
60:                             such that the product of the scale and the sum of
61:                             the inputs is approximately 1.0 (and the product of
62:                             the scale and the sum of the outputs is
63:                             approximately -1.0).
64:           *unit*            string representing the physical unit associated
65:                             with the flow quantities
66:                             If *unit* is None, then none of the quantities are
67:                             labeled.
68:           *format*          a Python number formatting string to be used in
69:                             labeling the flow as a quantity (i.e., a number
70:                             times a unit, where the unit is given)
71:           *gap*             space between paths that break in/break away
72:                             to/from the top or bottom
73:           *radius*          inner radius of the vertical paths
74:           *shoulder*        size of the shoulders of output arrowS
75:           *offset*          text offset (from the dip or tip of the arrow)
76:           *head_angle*      angle of the arrow heads (and negative of the angle
77:                             of the tails) [deg]
78:           *margin*          minimum space between Sankey outlines and the edge
79:                             of the plot area
80:           *tolerance*       acceptable maximum of the magnitude of the sum of
81:                             flows
82:                             The magnitude of the sum of connected flows cannot
83:                             be greater than *tolerance*.
84:           ===============   ===================================================
85: 
86:         The optional arguments listed above are applied to all subdiagrams so
87:         that there is consistent alignment and formatting.
88: 
89:         If :class:`Sankey` is instantiated with any keyword arguments other
90:         than those explicitly listed above (``**kwargs``), they will be passed
91:         to :meth:`add`, which will create the first subdiagram.
92: 
93:         In order to draw a complex Sankey diagram, create an instance of
94:         :class:`Sankey` by calling it without any kwargs::
95: 
96:             sankey = Sankey()
97: 
98:         Then add simple Sankey sub-diagrams::
99: 
100:             sankey.add() # 1
101:             sankey.add() # 2
102:             #...
103:             sankey.add() # n
104: 
105:         Finally, create the full diagram::
106: 
107:             sankey.finish()
108: 
109:         Or, instead, simply daisy-chain those calls::
110: 
111:             Sankey().add().add...  .add().finish()
112: 
113:         .. seealso::
114: 
115:             :meth:`add`
116:             :meth:`finish`
117: 
118: 
119:         **Examples:**
120: 
121:             .. plot:: gallery/api/sankey_basics.py
122:         '''
123:         # Check the arguments.
124:         if gap < 0:
125:             raise ValueError(
126:             "The gap is negative.\nThis isn't allowed because it "
127:             "would cause the paths to overlap.")
128:         if radius > gap:
129:             raise ValueError(
130:             "The inner radius is greater than the path spacing.\n"
131:             "This isn't allowed because it would cause the paths to overlap.")
132:         if head_angle < 0:
133:             raise ValueError(
134:             "The angle is negative.\nThis isn't allowed "
135:             "because it would cause inputs to look like "
136:             "outputs and vice versa.")
137:         if tolerance < 0:
138:             raise ValueError(
139:             "The tolerance is negative.\nIt must be a magnitude.")
140: 
141:         # Create axes if necessary.
142:         if ax is None:
143:             import matplotlib.pyplot as plt
144:             fig = plt.figure()
145:             ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
146: 
147:         self.diagrams = []
148: 
149:         # Store the inputs.
150:         self.ax = ax
151:         self.unit = unit
152:         self.format = format
153:         self.scale = scale
154:         self.gap = gap
155:         self.radius = radius
156:         self.shoulder = shoulder
157:         self.offset = offset
158:         self.margin = margin
159:         self.pitch = np.tan(np.pi * (1 - head_angle / 180.0) / 2.0)
160:         self.tolerance = tolerance
161: 
162:         # Initialize the vertices of tight box around the diagram(s).
163:         self.extent = np.array((np.inf, -np.inf, np.inf, -np.inf))
164: 
165:         # If there are any kwargs, create the first subdiagram.
166:         if len(kwargs):
167:             self.add(**kwargs)
168: 
169:     def _arc(self, quadrant=0, cw=True, radius=1, center=(0, 0)):
170:         '''
171:         Return the codes and vertices for a rotated, scaled, and translated
172:         90 degree arc.
173: 
174:         Optional keyword arguments:
175: 
176:           ===============   ==========================================
177:           Keyword           Description
178:           ===============   ==========================================
179:           *quadrant*        uses 0-based indexing (0, 1, 2, or 3)
180:           *cw*              if True, clockwise
181:           *center*          (x, y) tuple of the arc's center
182:           ===============   ==========================================
183:         '''
184:         # Note:  It would be possible to use matplotlib's transforms to rotate,
185:         # scale, and translate the arc, but since the angles are discrete,
186:         # it's just as easy and maybe more efficient to do it here.
187:         ARC_CODES = [Path.LINETO,
188:                      Path.CURVE4,
189:                      Path.CURVE4,
190:                      Path.CURVE4,
191:                      Path.CURVE4,
192:                      Path.CURVE4,
193:                      Path.CURVE4]
194:         # Vertices of a cubic Bezier curve approximating a 90 deg arc
195:         # These can be determined by Path.arc(0,90).
196:         ARC_VERTICES = np.array([[1.00000000e+00, 0.00000000e+00],
197:                                  [1.00000000e+00, 2.65114773e-01],
198:                                  [8.94571235e-01, 5.19642327e-01],
199:                                  [7.07106781e-01, 7.07106781e-01],
200:                                  [5.19642327e-01, 8.94571235e-01],
201:                                  [2.65114773e-01, 1.00000000e+00],
202:                                  # Insignificant
203:                                  # [6.12303177e-17, 1.00000000e+00]])
204:                                  [0.00000000e+00, 1.00000000e+00]])
205:         if quadrant == 0 or quadrant == 2:
206:             if cw:
207:                 vertices = ARC_VERTICES
208:             else:
209:                 vertices = ARC_VERTICES[:, ::-1]  # Swap x and y.
210:         elif quadrant == 1 or quadrant == 3:
211:             # Negate x.
212:             if cw:
213:                 # Swap x and y.
214:                 vertices = np.column_stack((-ARC_VERTICES[:, 1],
215:                                              ARC_VERTICES[:, 0]))
216:             else:
217:                 vertices = np.column_stack((-ARC_VERTICES[:, 0],
218:                                              ARC_VERTICES[:, 1]))
219:         if quadrant > 1:
220:             radius = -radius  # Rotate 180 deg.
221:         return list(zip(ARC_CODES, radius * vertices +
222:                         np.tile(center, (ARC_VERTICES.shape[0], 1))))
223: 
224:     def _add_input(self, path, angle, flow, length):
225:         '''
226:         Add an input to a path and return its tip and label locations.
227:         '''
228:         if angle is None:
229:             return [0, 0], [0, 0]
230:         else:
231:             x, y = path[-1][1]  # Use the last point as a reference.
232:             dipdepth = (flow / 2) * self.pitch
233:             if angle == RIGHT:
234:                 x -= length
235:                 dip = [x + dipdepth, y + flow / 2.0]
236:                 path.extend([(Path.LINETO, [x, y]),
237:                              (Path.LINETO, dip),
238:                              (Path.LINETO, [x, y + flow]),
239:                              (Path.LINETO, [x + self.gap, y + flow])])
240:                 label_location = [dip[0] - self.offset, dip[1]]
241:             else:  # Vertical
242:                 x -= self.gap
243:                 if angle == UP:
244:                     sign = 1
245:                 else:
246:                     sign = -1
247: 
248:                 dip = [x - flow / 2, y - sign * (length - dipdepth)]
249:                 if angle == DOWN:
250:                     quadrant = 2
251:                 else:
252:                     quadrant = 1
253: 
254:                 # Inner arc isn't needed if inner radius is zero
255:                 if self.radius:
256:                     path.extend(self._arc(quadrant=quadrant,
257:                                           cw=angle == UP,
258:                                           radius=self.radius,
259:                                           center=(x + self.radius,
260:                                                   y - sign * self.radius)))
261:                 else:
262:                     path.append((Path.LINETO, [x, y]))
263:                 path.extend([(Path.LINETO, [x, y - sign * length]),
264:                              (Path.LINETO, dip),
265:                              (Path.LINETO, [x - flow, y - sign * length])])
266:                 path.extend(self._arc(quadrant=quadrant,
267:                                       cw=angle == DOWN,
268:                                       radius=flow + self.radius,
269:                                       center=(x + self.radius,
270:                                               y - sign * self.radius)))
271:                 path.append((Path.LINETO, [x - flow, y + sign * flow]))
272:                 label_location = [dip[0], dip[1] - sign * self.offset]
273: 
274:             return dip, label_location
275: 
276:     def _add_output(self, path, angle, flow, length):
277:         '''
278:         Append an output to a path and return its tip and label locations.
279: 
280:         .. note:: *flow* is negative for an output.
281:         '''
282:         if angle is None:
283:             return [0, 0], [0, 0]
284:         else:
285:             x, y = path[-1][1]  # Use the last point as a reference.
286:             tipheight = (self.shoulder - flow / 2) * self.pitch
287:             if angle == RIGHT:
288:                 x += length
289:                 tip = [x + tipheight, y + flow / 2.0]
290:                 path.extend([(Path.LINETO, [x, y]),
291:                              (Path.LINETO, [x, y + self.shoulder]),
292:                              (Path.LINETO, tip),
293:                              (Path.LINETO, [x, y - self.shoulder + flow]),
294:                              (Path.LINETO, [x, y + flow]),
295:                              (Path.LINETO, [x - self.gap, y + flow])])
296:                 label_location = [tip[0] + self.offset, tip[1]]
297:             else:  # Vertical
298:                 x += self.gap
299:                 if angle == UP:
300:                     sign = 1
301:                 else:
302:                     sign = -1
303: 
304:                 tip = [x - flow / 2.0, y + sign * (length + tipheight)]
305:                 if angle == UP:
306:                     quadrant = 3
307:                 else:
308:                     quadrant = 0
309:                 # Inner arc isn't needed if inner radius is zero
310:                 if self.radius:
311:                     path.extend(self._arc(quadrant=quadrant,
312:                                           cw=angle == UP,
313:                                           radius=self.radius,
314:                                           center=(x - self.radius,
315:                                                   y + sign * self.radius)))
316:                 else:
317:                     path.append((Path.LINETO, [x, y]))
318:                 path.extend([(Path.LINETO, [x, y + sign * length]),
319:                              (Path.LINETO, [x - self.shoulder,
320:                                             y + sign * length]),
321:                              (Path.LINETO, tip),
322:                              (Path.LINETO, [x + self.shoulder - flow,
323:                                             y + sign * length]),
324:                              (Path.LINETO, [x - flow, y + sign * length])])
325:                 path.extend(self._arc(quadrant=quadrant,
326:                                       cw=angle == DOWN,
327:                                       radius=self.radius - flow,
328:                                       center=(x - self.radius,
329:                                               y + sign * self.radius)))
330:                 path.append((Path.LINETO, [x - flow, y + sign * flow]))
331:                 label_location = [tip[0], tip[1] + sign * self.offset]
332:             return tip, label_location
333: 
334:     def _revert(self, path, first_action=Path.LINETO):
335:         '''
336:         A path is not simply revertable by path[::-1] since the code
337:         specifies an action to take from the **previous** point.
338:         '''
339:         reverse_path = []
340:         next_code = first_action
341:         for code, position in path[::-1]:
342:             reverse_path.append((next_code, position))
343:             next_code = code
344:         return reverse_path
345:         # This might be more efficient, but it fails because 'tuple' object
346:         # doesn't support item assignment:
347:         # path[1] = path[1][-1:0:-1]
348:         # path[1][0] = first_action
349:         # path[2] = path[2][::-1]
350:         # return path
351: 
352:     @docstring.dedent_interpd
353:     def add(self, patchlabel='', flows=None, orientations=None, labels='',
354:             trunklength=1.0, pathlengths=0.25, prior=None, connect=(0, 0),
355:             rotation=0, **kwargs):
356:         '''
357:         Add a simple Sankey diagram with flows at the same hierarchical level.
358: 
359:         Return value is the instance of :class:`Sankey`.
360: 
361:         Optional keyword arguments:
362: 
363:           ===============   ===================================================
364:           Keyword           Description
365:           ===============   ===================================================
366:           *patchlabel*      label to be placed at the center of the diagram
367:                             Note: *label* (not *patchlabel*) will be passed to
368:                             the patch through ``**kwargs`` and can be used to
369:                             create an entry in the legend.
370:           *flows*           array of flow values
371:                             By convention, inputs are positive and outputs are
372:                             negative.
373:           *orientations*    list of orientations of the paths
374:                             Valid values are 1 (from/to the top), 0 (from/to
375:                             the left or right), or -1 (from/to the bottom).  If
376:                             *orientations* == 0, inputs will break in from the
377:                             left and outputs will break away to the right.
378:           *labels*          list of specifications of the labels for the flows
379:                             Each value may be *None* (no labels), '' (just
380:                             label the quantities), or a labeling string.  If a
381:                             single value is provided, it will be applied to all
382:                             flows.  If an entry is a non-empty string, then the
383:                             quantity for the corresponding flow will be shown
384:                             below the string.  However, if the *unit* of the
385:                             main diagram is None, then quantities are never
386:                             shown, regardless of the value of this argument.
387:           *trunklength*     length between the bases of the input and output
388:                             groups
389:           *pathlengths*     list of lengths of the arrows before break-in or
390:                             after break-away
391:                             If a single value is given, then it will be applied
392:                             to the first (inside) paths on the top and bottom,
393:                             and the length of all other arrows will be
394:                             justified accordingly.  The *pathlengths* are not
395:                             applied to the horizontal inputs and outputs.
396:           *prior*           index of the prior diagram to which this diagram
397:                             should be connected
398:           *connect*         a (prior, this) tuple indexing the flow of the
399:                             prior diagram and the flow of this diagram which
400:                             should be connected
401:                             If this is the first diagram or *prior* is *None*,
402:                             *connect* will be ignored.
403:           *rotation*        angle of rotation of the diagram [deg]
404:                             *rotation* is ignored if this diagram is connected
405:                             to an existing one (using *prior* and *connect*).
406:                             The interpretation of the *orientations* argument
407:                             will be rotated accordingly (e.g., if *rotation*
408:                             == 90, an *orientations* entry of 1 means to/from
409:                             the left).
410:           ===============   ===================================================
411: 
412:         Valid kwargs are :meth:`matplotlib.patches.PathPatch` arguments:
413: 
414:         %(Patch)s
415: 
416:         As examples, ``fill=False`` and ``label='A legend entry'``.
417:         By default, ``facecolor='#bfd1d4'`` (light blue) and
418:         ``linewidth=0.5``.
419: 
420:         The indexing parameters (*prior* and *connect*) are zero-based.
421: 
422:         The flows are placed along the top of the diagram from the inside out
423:         in order of their index within the *flows* list or array.  They are
424:         placed along the sides of the diagram from the top down and along the
425:         bottom from the outside in.
426: 
427:         If the sum of the inputs and outputs is nonzero, the discrepancy
428:         will appear as a cubic Bezier curve along the top and bottom edges of
429:         the trunk.
430: 
431:         .. seealso::
432: 
433:             :meth:`finish`
434:         '''
435:         # Check and preprocess the arguments.
436:         if flows is None:
437:             flows = np.array([1.0, -1.0])
438:         else:
439:             flows = np.array(flows)
440:         n = flows.shape[0]  # Number of flows
441:         if rotation is None:
442:             rotation = 0
443:         else:
444:             # In the code below, angles are expressed in deg/90.
445:             rotation /= 90.0
446:         if orientations is None:
447:             orientations = [0, 0]
448:         if len(orientations) != n:
449:             raise ValueError(
450:             "orientations and flows must have the same length.\n"
451:             "orientations has length %d, but flows has length %d."
452:             % (len(orientations), n))
453:         if labels != '' and getattr(labels, '__iter__', False):
454:             # iterable() isn't used because it would give True if labels is a
455:             # string
456:             if len(labels) != n:
457:                 raise ValueError(
458:                 "If labels is a list, then labels and flows must have the "
459:                 "same length.\nlabels has length %d, but flows has length %d."
460:                 % (len(labels), n))
461:         else:
462:             labels = [labels] * n
463:         if trunklength < 0:
464:             raise ValueError(
465:             "trunklength is negative.\nThis isn't allowed, because it would "
466:             "cause poor layout.")
467:         if np.abs(np.sum(flows)) > self.tolerance:
468:             verbose.report(
469:                 "The sum of the flows is nonzero (%f).\nIs the "
470:                 "system not at steady state?" % np.sum(flows), 'helpful')
471:         scaled_flows = self.scale * flows
472:         gain = sum(max(flow, 0) for flow in scaled_flows)
473:         loss = sum(min(flow, 0) for flow in scaled_flows)
474:         if not (0.5 <= gain <= 2.0):
475:             verbose.report(
476:                 "The scaled sum of the inputs is %f.\nThis may "
477:                 "cause poor layout.\nConsider changing the scale so"
478:                 " that the scaled sum is approximately 1.0." % gain, 'helpful')
479:         if not (-2.0 <= loss <= -0.5):
480:             verbose.report(
481:                 "The scaled sum of the outputs is %f.\nThis may "
482:                 "cause poor layout.\nConsider changing the scale so"
483:                 " that the scaled sum is approximately 1.0." % gain, 'helpful')
484:         if prior is not None:
485:             if prior < 0:
486:                 raise ValueError("The index of the prior diagram is negative.")
487:             if min(connect) < 0:
488:                 raise ValueError(
489:                 "At least one of the connection indices is negative.")
490:             if prior >= len(self.diagrams):
491:                 raise ValueError(
492:                 "The index of the prior diagram is %d, but there are "
493:                 "only %d other diagrams.\nThe index is zero-based."
494:                 % (prior, len(self.diagrams)))
495:             if connect[0] >= len(self.diagrams[prior].flows):
496:                 raise ValueError(
497:                 "The connection index to the source diagram is %d, but "
498:                 "that diagram has only %d flows.\nThe index is zero-based."
499:                 % (connect[0], len(self.diagrams[prior].flows)))
500:             if connect[1] >= n:
501:                 raise ValueError(
502:                 "The connection index to this diagram is %d, but this diagram"
503:                 "has only %d flows.\n The index is zero-based."
504:                 % (connect[1], n))
505:             if self.diagrams[prior].angles[connect[0]] is None:
506:                 raise ValueError(
507:                 "The connection cannot be made.  Check that the magnitude "
508:                 "of flow %d of diagram %d is greater than or equal to the "
509:                 "specified tolerance." % (connect[0], prior))
510:             flow_error = (self.diagrams[prior].flows[connect[0]] +
511:                           flows[connect[1]])
512:             if abs(flow_error) >= self.tolerance:
513:                 raise ValueError(
514:                 "The scaled sum of the connected flows is %f, which is not "
515:                 "within the tolerance (%f)." % (flow_error, self.tolerance))
516: 
517:         # Determine if the flows are inputs.
518:         are_inputs = [None] * n
519:         for i, flow in enumerate(flows):
520:             if flow >= self.tolerance:
521:                 are_inputs[i] = True
522:             elif flow <= -self.tolerance:
523:                 are_inputs[i] = False
524:             else:
525:                 verbose.report(
526:                     "The magnitude of flow %d (%f) is below the "
527:                     "tolerance (%f).\nIt will not be shown, and it "
528:                     "cannot be used in a connection."
529:                     % (i, flow, self.tolerance), 'helpful')
530: 
531:         # Determine the angles of the arrows (before rotation).
532:         angles = [None] * n
533:         for i, (orient, is_input) in enumerate(zip(orientations, are_inputs)):
534:             if orient == 1:
535:                 if is_input:
536:                     angles[i] = DOWN
537:                 elif not is_input:
538:                     # Be specific since is_input can be None.
539:                     angles[i] = UP
540:             elif orient == 0:
541:                 if is_input is not None:
542:                     angles[i] = RIGHT
543:             else:
544:                 if orient != -1:
545:                     raise ValueError(
546:                     "The value of orientations[%d] is %d, "
547:                     "but it must be [ -1 | 0 | 1 ]." % (i, orient))
548:                 if is_input:
549:                     angles[i] = UP
550:                 elif not is_input:
551:                     angles[i] = DOWN
552: 
553:         # Justify the lengths of the paths.
554:         if iterable(pathlengths):
555:             if len(pathlengths) != n:
556:                 raise ValueError(
557:                 "If pathlengths is a list, then pathlengths and flows must "
558:                 "have the same length.\npathlengths has length %d, but flows "
559:                 "has length %d." % (len(pathlengths), n))
560:         else:  # Make pathlengths into a list.
561:             urlength = pathlengths
562:             ullength = pathlengths
563:             lrlength = pathlengths
564:             lllength = pathlengths
565:             d = dict(RIGHT=pathlengths)
566:             pathlengths = [d.get(angle, 0) for angle in angles]
567:             # Determine the lengths of the top-side arrows
568:             # from the middle outwards.
569:             for i, (angle, is_input, flow) in enumerate(zip(angles, are_inputs,
570:                                                             scaled_flows)):
571:                 if angle == DOWN and is_input:
572:                     pathlengths[i] = ullength
573:                     ullength += flow
574:                 elif angle == UP and not is_input:
575:                     pathlengths[i] = urlength
576:                     urlength -= flow  # Flow is negative for outputs.
577:             # Determine the lengths of the bottom-side arrows
578:             # from the middle outwards.
579:             for i, (angle, is_input, flow) in enumerate(reversed(list(zip(
580:                   angles, are_inputs, scaled_flows)))):
581:                 if angle == UP and is_input:
582:                     pathlengths[n - i - 1] = lllength
583:                     lllength += flow
584:                 elif angle == DOWN and not is_input:
585:                     pathlengths[n - i - 1] = lrlength
586:                     lrlength -= flow
587:             # Determine the lengths of the left-side arrows
588:             # from the bottom upwards.
589:             has_left_input = False
590:             for i, (angle, is_input, spec) in enumerate(reversed(list(zip(
591:                   angles, are_inputs, zip(scaled_flows, pathlengths))))):
592:                 if angle == RIGHT:
593:                     if is_input:
594:                         if has_left_input:
595:                             pathlengths[n - i - 1] = 0
596:                         else:
597:                             has_left_input = True
598:             # Determine the lengths of the right-side arrows
599:             # from the top downwards.
600:             has_right_output = False
601:             for i, (angle, is_input, spec) in enumerate(zip(
602:                   angles, are_inputs, list(zip(scaled_flows, pathlengths)))):
603:                 if angle == RIGHT:
604:                     if not is_input:
605:                         if has_right_output:
606:                             pathlengths[i] = 0
607:                         else:
608:                             has_right_output = True
609: 
610:         # Begin the subpaths, and smooth the transition if the sum of the flows
611:         # is nonzero.
612:         urpath = [(Path.MOVETO, [(self.gap - trunklength / 2.0),  # Upper right
613:                                  gain / 2.0]),
614:                   (Path.LINETO, [(self.gap - trunklength / 2.0) / 2.0,
615:                                  gain / 2.0]),
616:                   (Path.CURVE4, [(self.gap - trunklength / 2.0) / 8.0,
617:                                  gain / 2.0]),
618:                   (Path.CURVE4, [(trunklength / 2.0 - self.gap) / 8.0,
619:                                  -loss / 2.0]),
620:                   (Path.LINETO, [(trunklength / 2.0 - self.gap) / 2.0,
621:                                  -loss / 2.0]),
622:                   (Path.LINETO, [(trunklength / 2.0 - self.gap),
623:                                  -loss / 2.0])]
624:         llpath = [(Path.LINETO, [(trunklength / 2.0 - self.gap),  # Lower left
625:                                  loss / 2.0]),
626:                   (Path.LINETO, [(trunklength / 2.0 - self.gap) / 2.0,
627:                                  loss / 2.0]),
628:                   (Path.CURVE4, [(trunklength / 2.0 - self.gap) / 8.0,
629:                                  loss / 2.0]),
630:                   (Path.CURVE4, [(self.gap - trunklength / 2.0) / 8.0,
631:                                  -gain / 2.0]),
632:                   (Path.LINETO, [(self.gap - trunklength / 2.0) / 2.0,
633:                                  -gain / 2.0]),
634:                   (Path.LINETO, [(self.gap - trunklength / 2.0),
635:                                  -gain / 2.0])]
636:         lrpath = [(Path.LINETO, [(trunklength / 2.0 - self.gap),  # Lower right
637:                                  loss / 2.0])]
638:         ulpath = [(Path.LINETO, [self.gap - trunklength / 2.0,  # Upper left
639:                                  gain / 2.0])]
640: 
641:         # Add the subpaths and assign the locations of the tips and labels.
642:         tips = np.zeros((n, 2))
643:         label_locations = np.zeros((n, 2))
644:         # Add the top-side inputs and outputs from the middle outwards.
645:         for i, (angle, is_input, spec) in enumerate(zip(
646:               angles, are_inputs, list(zip(scaled_flows, pathlengths)))):
647:             if angle == DOWN and is_input:
648:                 tips[i, :], label_locations[i, :] = self._add_input(
649:                     ulpath, angle, *spec)
650:             elif angle == UP and not is_input:
651:                 tips[i, :], label_locations[i, :] = self._add_output(
652:                     urpath, angle, *spec)
653:         # Add the bottom-side inputs and outputs from the middle outwards.
654:         for i, (angle, is_input, spec) in enumerate(reversed(list(zip(
655:               angles, are_inputs, list(zip(scaled_flows, pathlengths)))))):
656:             if angle == UP and is_input:
657:                 tip, label_location = self._add_input(llpath, angle, *spec)
658:                 tips[n - i - 1, :] = tip
659:                 label_locations[n - i - 1, :] = label_location
660:             elif angle == DOWN and not is_input:
661:                 tip, label_location = self._add_output(lrpath, angle, *spec)
662:                 tips[n - i - 1, :] = tip
663:                 label_locations[n - i - 1, :] = label_location
664:         # Add the left-side inputs from the bottom upwards.
665:         has_left_input = False
666:         for i, (angle, is_input, spec) in enumerate(reversed(list(zip(
667:               angles, are_inputs, list(zip(scaled_flows, pathlengths)))))):
668:             if angle == RIGHT and is_input:
669:                 if not has_left_input:
670:                     # Make sure the lower path extends
671:                     # at least as far as the upper one.
672:                     if llpath[-1][1][0] > ulpath[-1][1][0]:
673:                         llpath.append((Path.LINETO, [ulpath[-1][1][0],
674:                                                      llpath[-1][1][1]]))
675:                     has_left_input = True
676:                 tip, label_location = self._add_input(llpath, angle, *spec)
677:                 tips[n - i - 1, :] = tip
678:                 label_locations[n - i - 1, :] = label_location
679:         # Add the right-side outputs from the top downwards.
680:         has_right_output = False
681:         for i, (angle, is_input, spec) in enumerate(zip(
682:               angles, are_inputs, list(zip(scaled_flows, pathlengths)))):
683:             if angle == RIGHT and not is_input:
684:                 if not has_right_output:
685:                     # Make sure the upper path extends
686:                     # at least as far as the lower one.
687:                     if urpath[-1][1][0] < lrpath[-1][1][0]:
688:                         urpath.append((Path.LINETO, [lrpath[-1][1][0],
689:                                                      urpath[-1][1][1]]))
690:                     has_right_output = True
691:                 tips[i, :], label_locations[i, :] = self._add_output(
692:                     urpath, angle, *spec)
693:         # Trim any hanging vertices.
694:         if not has_left_input:
695:             ulpath.pop()
696:             llpath.pop()
697:         if not has_right_output:
698:             lrpath.pop()
699:             urpath.pop()
700: 
701:         # Concatenate the subpaths in the correct order (clockwise from top).
702:         path = (urpath + self._revert(lrpath) + llpath + self._revert(ulpath) +
703:                 [(Path.CLOSEPOLY, urpath[0][1])])
704: 
705:         # Create a patch with the Sankey outline.
706:         codes, vertices = list(zip(*path))
707:         vertices = np.array(vertices)
708: 
709:         def _get_angle(a, r):
710:             if a is None:
711:                 return None
712:             else:
713:                 return a + r
714: 
715:         if prior is None:
716:             if rotation != 0:  # By default, none of this is needed.
717:                 angles = [_get_angle(angle, rotation) for angle in angles]
718:                 rotate = Affine2D().rotate_deg(rotation * 90).transform_affine
719:                 tips = rotate(tips)
720:                 label_locations = rotate(label_locations)
721:                 vertices = rotate(vertices)
722:             text = self.ax.text(0, 0, s=patchlabel, ha='center', va='center')
723:         else:
724:             rotation = (self.diagrams[prior].angles[connect[0]] -
725:                         angles[connect[1]])
726:             angles = [_get_angle(angle, rotation) for angle in angles]
727:             rotate = Affine2D().rotate_deg(rotation * 90).transform_affine
728:             tips = rotate(tips)
729:             offset = self.diagrams[prior].tips[connect[0]] - tips[connect[1]]
730:             translate = Affine2D().translate(*offset).transform_affine
731:             tips = translate(tips)
732:             label_locations = translate(rotate(label_locations))
733:             vertices = translate(rotate(vertices))
734:             kwds = dict(s=patchlabel, ha='center', va='center')
735:             text = self.ax.text(*offset, **kwds)
736:         if False:  # Debug
737:             print("llpath\n", llpath)
738:             print("ulpath\n", self._revert(ulpath))
739:             print("urpath\n", urpath)
740:             print("lrpath\n", self._revert(lrpath))
741:             xs, ys = list(zip(*vertices))
742:             self.ax.plot(xs, ys, 'go-')
743:         if rcParams['_internal.classic_mode']:
744:             fc = kwargs.pop('fc', kwargs.pop('facecolor', '#bfd1d4'))
745:             lw = kwargs.pop('lw', kwargs.pop('linewidth', 0.5))
746:         else:
747:             fc = kwargs.pop('fc', kwargs.pop('facecolor', None))
748:             lw = kwargs.pop('lw', kwargs.pop('linewidth', None))
749:         if fc is None:
750:             fc = six.next(self.ax._get_patches_for_fill.prop_cycler)['color']
751:         patch = PathPatch(Path(vertices, codes), fc=fc, lw=lw, **kwargs)
752:         self.ax.add_patch(patch)
753: 
754:         # Add the path labels.
755:         texts = []
756:         for number, angle, label, location in zip(flows, angles, labels,
757:                                                   label_locations):
758:             if label is None or angle is None:
759:                 label = ''
760:             elif self.unit is not None:
761:                 quantity = self.format % abs(number) + self.unit
762:                 if label != '':
763:                     label += "\n"
764:                 label += quantity
765:             texts.append(self.ax.text(x=location[0], y=location[1],
766:                                       s=label,
767:                                       ha='center', va='center'))
768:         # Text objects are placed even they are empty (as long as the magnitude
769:         # of the corresponding flow is larger than the tolerance) in case the
770:         # user wants to provide labels later.
771: 
772:         # Expand the size of the diagram if necessary.
773:         self.extent = (min(np.min(vertices[:, 0]),
774:                            np.min(label_locations[:, 0]),
775:                            self.extent[0]),
776:                        max(np.max(vertices[:, 0]),
777:                            np.max(label_locations[:, 0]),
778:                            self.extent[1]),
779:                        min(np.min(vertices[:, 1]),
780:                            np.min(label_locations[:, 1]),
781:                            self.extent[2]),
782:                        max(np.max(vertices[:, 1]),
783:                            np.max(label_locations[:, 1]),
784:                            self.extent[3]))
785:         # Include both vertices _and_ label locations in the extents; there are
786:         # where either could determine the margins (e.g., arrow shoulders).
787: 
788:         # Add this diagram as a subdiagram.
789:         self.diagrams.append(Bunch(patch=patch, flows=flows, angles=angles,
790:                                    tips=tips, text=text, texts=texts))
791: 
792:         # Allow a daisy-chained call structure (see docstring for the class).
793:         return self
794: 
795:     def finish(self):
796:         '''
797:         Adjust the axes and return a list of information about the Sankey
798:         subdiagram(s).
799: 
800:         Return value is a list of subdiagrams represented with the following
801:         fields:
802: 
803:           ===============   ===================================================
804:           Field             Description
805:           ===============   ===================================================
806:           *patch*           Sankey outline (an instance of
807:                             :class:`~maplotlib.patches.PathPatch`)
808:           *flows*           values of the flows (positive for input, negative
809:                             for output)
810:           *angles*          list of angles of the arrows [deg/90]
811:                             For example, if the diagram has not been rotated,
812:                             an input to the top side will have an angle of 3
813:                             (DOWN), and an output from the top side will have
814:                             an angle of 1 (UP).  If a flow has been skipped
815:                             (because its magnitude is less than *tolerance*),
816:                             then its angle will be *None*.
817:           *tips*            array in which each row is an [x, y] pair
818:                             indicating the positions of the tips (or "dips") of
819:                             the flow paths
820:                             If the magnitude of a flow is less the *tolerance*
821:                             for the instance of :class:`Sankey`, the flow is
822:                             skipped and its tip will be at the center of the
823:                             diagram.
824:           *text*            :class:`~matplotlib.text.Text` instance for the
825:                             label of the diagram
826:           *texts*           list of :class:`~matplotlib.text.Text` instances
827:                             for the labels of flows
828:           ===============   ===================================================
829: 
830:         .. seealso::
831: 
832:             :meth:`add`
833:         '''
834:         self.ax.axis([self.extent[0] - self.margin,
835:                       self.extent[1] + self.margin,
836:                       self.extent[2] - self.margin,
837:                       self.extent[3] + self.margin])
838:         self.ax.set_aspect('equal', adjustable='datalim')
839:         return self.diagrams
840: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_127459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nModule for creating Sankey diagrams using matplotlib\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127460 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_127460) is not StypyTypeError):

    if (import_127460 != 'pyd_module'):
        __import__(import_127460)
        sys_modules_127461 = sys.modules[import_127460]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_127461.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_127460)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from six.moves import zip' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127462 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six.moves')

if (type(import_127462) is not StypyTypeError):

    if (import_127462 != 'pyd_module'):
        __import__(import_127462)
        sys_modules_127463 = sys.modules[import_127462]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six.moves', sys_modules_127463.module_type_store, module_type_store, ['zip'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_127463, sys_modules_127463.module_type_store, module_type_store)
    else:
        from six.moves import zip

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six.moves', None, module_type_store, ['zip'], [zip])

else:
    # Assigning a type to the variable 'six.moves' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six.moves', import_127462)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127464 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_127464) is not StypyTypeError):

    if (import_127464 != 'pyd_module'):
        __import__(import_127464)
        sys_modules_127465 = sys.modules[import_127464]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_127465.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_127464)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from matplotlib.cbook import iterable, Bunch' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127466 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.cbook')

if (type(import_127466) is not StypyTypeError):

    if (import_127466 != 'pyd_module'):
        __import__(import_127466)
        sys_modules_127467 = sys.modules[import_127466]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.cbook', sys_modules_127467.module_type_store, module_type_store, ['iterable', 'Bunch'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_127467, sys_modules_127467.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import iterable, Bunch

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.cbook', None, module_type_store, ['iterable', 'Bunch'], [iterable, Bunch])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.cbook', import_127466)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.path import Path' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127468 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.path')

if (type(import_127468) is not StypyTypeError):

    if (import_127468 != 'pyd_module'):
        __import__(import_127468)
        sys_modules_127469 = sys.modules[import_127468]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.path', sys_modules_127469.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_127469, sys_modules_127469.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.path', import_127468)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib.patches import PathPatch' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127470 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.patches')

if (type(import_127470) is not StypyTypeError):

    if (import_127470 != 'pyd_module'):
        __import__(import_127470)
        sys_modules_127471 = sys.modules[import_127470]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.patches', sys_modules_127471.module_type_store, module_type_store, ['PathPatch'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_127471, sys_modules_127471.module_type_store, module_type_store)
    else:
        from matplotlib.patches import PathPatch

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.patches', None, module_type_store, ['PathPatch'], [PathPatch])

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.patches', import_127470)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib.transforms import Affine2D' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127472 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.transforms')

if (type(import_127472) is not StypyTypeError):

    if (import_127472 != 'pyd_module'):
        __import__(import_127472)
        sys_modules_127473 = sys.modules[import_127472]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.transforms', sys_modules_127473.module_type_store, module_type_store, ['Affine2D'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_127473, sys_modules_127473.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Affine2D

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.transforms', None, module_type_store, ['Affine2D'], [Affine2D])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.transforms', import_127472)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib import verbose' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127474 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib')

if (type(import_127474) is not StypyTypeError):

    if (import_127474 != 'pyd_module'):
        __import__(import_127474)
        sys_modules_127475 = sys.modules[import_127474]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', sys_modules_127475.module_type_store, module_type_store, ['verbose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_127475, sys_modules_127475.module_type_store, module_type_store)
    else:
        from matplotlib import verbose

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', None, module_type_store, ['verbose'], [verbose])

else:
    # Assigning a type to the variable 'matplotlib' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', import_127474)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib import docstring' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127476 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib')

if (type(import_127476) is not StypyTypeError):

    if (import_127476 != 'pyd_module'):
        __import__(import_127476)
        sys_modules_127477 = sys.modules[import_127476]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib', sys_modules_127477.module_type_store, module_type_store, ['docstring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_127477, sys_modules_127477.module_type_store, module_type_store)
    else:
        from matplotlib import docstring

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib', None, module_type_store, ['docstring'], [docstring])

else:
    # Assigning a type to the variable 'matplotlib' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib', import_127476)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from matplotlib import rcParams' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_127478 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib')

if (type(import_127478) is not StypyTypeError):

    if (import_127478 != 'pyd_module'):
        __import__(import_127478)
        sys_modules_127479 = sys.modules[import_127478]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib', sys_modules_127479.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_127479, sys_modules_127479.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib', import_127478)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


# Assigning a Str to a Name (line 19):

# Assigning a Str to a Name (line 19):
unicode_127480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'unicode', u'Kevin L. Davies')
# Assigning a type to the variable '__author__' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '__author__', unicode_127480)

# Assigning a List to a Name (line 20):

# Assigning a List to a Name (line 20):

# Obtaining an instance of the builtin type 'list' (line 20)
list_127481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
unicode_127482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'unicode', u'Yannick Copin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 14), list_127481, unicode_127482)

# Assigning a type to the variable '__credits__' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '__credits__', list_127481)

# Assigning a Str to a Name (line 21):

# Assigning a Str to a Name (line 21):
unicode_127483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'unicode', u'BSD')
# Assigning a type to the variable '__license__' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '__license__', unicode_127483)

# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
unicode_127484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'unicode', u'2011/09/16')
# Assigning a type to the variable '__version__' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '__version__', unicode_127484)

# Assigning a Num to a Name (line 25):

# Assigning a Num to a Name (line 25):
int_127485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'int')
# Assigning a type to the variable 'RIGHT' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'RIGHT', int_127485)

# Assigning a Num to a Name (line 26):

# Assigning a Num to a Name (line 26):
int_127486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'int')
# Assigning a type to the variable 'UP' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'UP', int_127486)

# Assigning a Num to a Name (line 28):

# Assigning a Num to a Name (line 28):
int_127487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 7), 'int')
# Assigning a type to the variable 'DOWN' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'DOWN', int_127487)
# Declaration of the 'Sankey' class

class Sankey(object, ):
    unicode_127488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'unicode', u'\n    Sankey diagram in matplotlib\n\n      Sankey diagrams are a specific type of flow diagram, in which\n      the width of the arrows is shown proportionally to the flow\n      quantity.  They are typically used to visualize energy or\n      material or cost transfers between processes.\n      `Wikipedia (6/1/2011) <https://en.wikipedia.org/wiki/Sankey_diagram>`_\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 43)
        None_127489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'None')
        float_127490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 38), 'float')
        unicode_127491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 48), 'unicode', u'')
        unicode_127492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 59), 'unicode', u'%G')
        float_127493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 69), 'float')
        float_127494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'float')
        float_127495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'float')
        float_127496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 51), 'float')
        int_127497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 68), 'int')
        float_127498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 24), 'float')
        float_127499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 39), 'float')
        defaults = [None_127489, float_127490, unicode_127491, unicode_127492, float_127493, float_127494, float_127495, float_127496, int_127497, float_127498, float_127499]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sankey.__init__', ['ax', 'scale', 'unit', 'format', 'gap', 'radius', 'shoulder', 'offset', 'head_angle', 'margin', 'tolerance'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ax', 'scale', 'unit', 'format', 'gap', 'radius', 'shoulder', 'offset', 'head_angle', 'margin', 'tolerance'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_127500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'unicode', u"\n        Create a new Sankey instance.\n\n        Optional keyword arguments:\n\n          ===============   ===================================================\n          Field             Description\n          ===============   ===================================================\n          *ax*              axes onto which the data should be plotted\n                            If *ax* isn't provided, new axes will be created.\n          *scale*           scaling factor for the flows\n                            *scale* sizes the width of the paths in order to\n                            maintain proper layout.  The same scale is applied\n                            to all subdiagrams.  The value should be chosen\n                            such that the product of the scale and the sum of\n                            the inputs is approximately 1.0 (and the product of\n                            the scale and the sum of the outputs is\n                            approximately -1.0).\n          *unit*            string representing the physical unit associated\n                            with the flow quantities\n                            If *unit* is None, then none of the quantities are\n                            labeled.\n          *format*          a Python number formatting string to be used in\n                            labeling the flow as a quantity (i.e., a number\n                            times a unit, where the unit is given)\n          *gap*             space between paths that break in/break away\n                            to/from the top or bottom\n          *radius*          inner radius of the vertical paths\n          *shoulder*        size of the shoulders of output arrowS\n          *offset*          text offset (from the dip or tip of the arrow)\n          *head_angle*      angle of the arrow heads (and negative of the angle\n                            of the tails) [deg]\n          *margin*          minimum space between Sankey outlines and the edge\n                            of the plot area\n          *tolerance*       acceptable maximum of the magnitude of the sum of\n                            flows\n                            The magnitude of the sum of connected flows cannot\n                            be greater than *tolerance*.\n          ===============   ===================================================\n\n        The optional arguments listed above are applied to all subdiagrams so\n        that there is consistent alignment and formatting.\n\n        If :class:`Sankey` is instantiated with any keyword arguments other\n        than those explicitly listed above (``**kwargs``), they will be passed\n        to :meth:`add`, which will create the first subdiagram.\n\n        In order to draw a complex Sankey diagram, create an instance of\n        :class:`Sankey` by calling it without any kwargs::\n\n            sankey = Sankey()\n\n        Then add simple Sankey sub-diagrams::\n\n            sankey.add() # 1\n            sankey.add() # 2\n            #...\n            sankey.add() # n\n\n        Finally, create the full diagram::\n\n            sankey.finish()\n\n        Or, instead, simply daisy-chain those calls::\n\n            Sankey().add().add...  .add().finish()\n\n        .. seealso::\n\n            :meth:`add`\n            :meth:`finish`\n\n\n        **Examples:**\n\n            .. plot:: gallery/api/sankey_basics.py\n        ")
        
        
        # Getting the type of 'gap' (line 124)
        gap_127501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'gap')
        int_127502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 17), 'int')
        # Applying the binary operator '<' (line 124)
        result_lt_127503 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), '<', gap_127501, int_127502)
        
        # Testing the type of an if condition (line 124)
        if_condition_127504 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_lt_127503)
        # Assigning a type to the variable 'if_condition_127504' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_127504', if_condition_127504)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 125)
        # Processing the call arguments (line 125)
        unicode_127506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 12), 'unicode', u"The gap is negative.\nThis isn't allowed because it would cause the paths to overlap.")
        # Processing the call keyword arguments (line 125)
        kwargs_127507 = {}
        # Getting the type of 'ValueError' (line 125)
        ValueError_127505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 125)
        ValueError_call_result_127508 = invoke(stypy.reporting.localization.Localization(__file__, 125, 18), ValueError_127505, *[unicode_127506], **kwargs_127507)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 125, 12), ValueError_call_result_127508, 'raise parameter', BaseException)
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'radius' (line 128)
        radius_127509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'radius')
        # Getting the type of 'gap' (line 128)
        gap_127510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'gap')
        # Applying the binary operator '>' (line 128)
        result_gt_127511 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 11), '>', radius_127509, gap_127510)
        
        # Testing the type of an if condition (line 128)
        if_condition_127512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 8), result_gt_127511)
        # Assigning a type to the variable 'if_condition_127512' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'if_condition_127512', if_condition_127512)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 129)
        # Processing the call arguments (line 129)
        unicode_127514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 12), 'unicode', u"The inner radius is greater than the path spacing.\nThis isn't allowed because it would cause the paths to overlap.")
        # Processing the call keyword arguments (line 129)
        kwargs_127515 = {}
        # Getting the type of 'ValueError' (line 129)
        ValueError_127513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 129)
        ValueError_call_result_127516 = invoke(stypy.reporting.localization.Localization(__file__, 129, 18), ValueError_127513, *[unicode_127514], **kwargs_127515)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 129, 12), ValueError_call_result_127516, 'raise parameter', BaseException)
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'head_angle' (line 132)
        head_angle_127517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'head_angle')
        int_127518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'int')
        # Applying the binary operator '<' (line 132)
        result_lt_127519 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 11), '<', head_angle_127517, int_127518)
        
        # Testing the type of an if condition (line 132)
        if_condition_127520 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), result_lt_127519)
        # Assigning a type to the variable 'if_condition_127520' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'if_condition_127520', if_condition_127520)
        # SSA begins for if statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 133)
        # Processing the call arguments (line 133)
        unicode_127522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 12), 'unicode', u"The angle is negative.\nThis isn't allowed because it would cause inputs to look like outputs and vice versa.")
        # Processing the call keyword arguments (line 133)
        kwargs_127523 = {}
        # Getting the type of 'ValueError' (line 133)
        ValueError_127521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 133)
        ValueError_call_result_127524 = invoke(stypy.reporting.localization.Localization(__file__, 133, 18), ValueError_127521, *[unicode_127522], **kwargs_127523)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 133, 12), ValueError_call_result_127524, 'raise parameter', BaseException)
        # SSA join for if statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'tolerance' (line 137)
        tolerance_127525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'tolerance')
        int_127526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'int')
        # Applying the binary operator '<' (line 137)
        result_lt_127527 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), '<', tolerance_127525, int_127526)
        
        # Testing the type of an if condition (line 137)
        if_condition_127528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), result_lt_127527)
        # Assigning a type to the variable 'if_condition_127528' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_127528', if_condition_127528)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 138)
        # Processing the call arguments (line 138)
        unicode_127530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 12), 'unicode', u'The tolerance is negative.\nIt must be a magnitude.')
        # Processing the call keyword arguments (line 138)
        kwargs_127531 = {}
        # Getting the type of 'ValueError' (line 138)
        ValueError_127529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 138)
        ValueError_call_result_127532 = invoke(stypy.reporting.localization.Localization(__file__, 138, 18), ValueError_127529, *[unicode_127530], **kwargs_127531)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 138, 12), ValueError_call_result_127532, 'raise parameter', BaseException)
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 142)
        # Getting the type of 'ax' (line 142)
        ax_127533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'ax')
        # Getting the type of 'None' (line 142)
        None_127534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'None')
        
        (may_be_127535, more_types_in_union_127536) = may_be_none(ax_127533, None_127534)

        if may_be_127535:

            if more_types_in_union_127536:
                # Runtime conditional SSA (line 142)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 143, 12))
            
            # 'import matplotlib.pyplot' statement (line 143)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
            import_127537 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 143, 12), 'matplotlib.pyplot')

            if (type(import_127537) is not StypyTypeError):

                if (import_127537 != 'pyd_module'):
                    __import__(import_127537)
                    sys_modules_127538 = sys.modules[import_127537]
                    import_module(stypy.reporting.localization.Localization(__file__, 143, 12), 'plt', sys_modules_127538.module_type_store, module_type_store)
                else:
                    import matplotlib.pyplot as plt

                    import_module(stypy.reporting.localization.Localization(__file__, 143, 12), 'plt', matplotlib.pyplot, module_type_store)

            else:
                # Assigning a type to the variable 'matplotlib.pyplot' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'matplotlib.pyplot', import_127537)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
            
            
            # Assigning a Call to a Name (line 144):
            
            # Assigning a Call to a Name (line 144):
            
            # Call to figure(...): (line 144)
            # Processing the call keyword arguments (line 144)
            kwargs_127541 = {}
            # Getting the type of 'plt' (line 144)
            plt_127539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'plt', False)
            # Obtaining the member 'figure' of a type (line 144)
            figure_127540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 18), plt_127539, 'figure')
            # Calling figure(args, kwargs) (line 144)
            figure_call_result_127542 = invoke(stypy.reporting.localization.Localization(__file__, 144, 18), figure_127540, *[], **kwargs_127541)
            
            # Assigning a type to the variable 'fig' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'fig', figure_call_result_127542)
            
            # Assigning a Call to a Name (line 145):
            
            # Assigning a Call to a Name (line 145):
            
            # Call to add_subplot(...): (line 145)
            # Processing the call arguments (line 145)
            int_127545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 33), 'int')
            int_127546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 36), 'int')
            int_127547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 39), 'int')
            # Processing the call keyword arguments (line 145)
            
            # Obtaining an instance of the builtin type 'list' (line 145)
            list_127548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 49), 'list')
            # Adding type elements to the builtin type 'list' instance (line 145)
            
            keyword_127549 = list_127548
            
            # Obtaining an instance of the builtin type 'list' (line 145)
            list_127550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 60), 'list')
            # Adding type elements to the builtin type 'list' instance (line 145)
            
            keyword_127551 = list_127550
            kwargs_127552 = {'xticks': keyword_127549, 'yticks': keyword_127551}
            # Getting the type of 'fig' (line 145)
            fig_127543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'fig', False)
            # Obtaining the member 'add_subplot' of a type (line 145)
            add_subplot_127544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 17), fig_127543, 'add_subplot')
            # Calling add_subplot(args, kwargs) (line 145)
            add_subplot_call_result_127553 = invoke(stypy.reporting.localization.Localization(__file__, 145, 17), add_subplot_127544, *[int_127545, int_127546, int_127547], **kwargs_127552)
            
            # Assigning a type to the variable 'ax' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'ax', add_subplot_call_result_127553)

            if more_types_in_union_127536:
                # SSA join for if statement (line 142)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Attribute (line 147):
        
        # Assigning a List to a Attribute (line 147):
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_127554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        
        # Getting the type of 'self' (line 147)
        self_127555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member 'diagrams' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_127555, 'diagrams', list_127554)
        
        # Assigning a Name to a Attribute (line 150):
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'ax' (line 150)
        ax_127556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'ax')
        # Getting the type of 'self' (line 150)
        self_127557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'ax' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_127557, 'ax', ax_127556)
        
        # Assigning a Name to a Attribute (line 151):
        
        # Assigning a Name to a Attribute (line 151):
        # Getting the type of 'unit' (line 151)
        unit_127558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'unit')
        # Getting the type of 'self' (line 151)
        self_127559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'unit' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_127559, 'unit', unit_127558)
        
        # Assigning a Name to a Attribute (line 152):
        
        # Assigning a Name to a Attribute (line 152):
        # Getting the type of 'format' (line 152)
        format_127560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'format')
        # Getting the type of 'self' (line 152)
        self_127561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'self')
        # Setting the type of the member 'format' of a type (line 152)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), self_127561, 'format', format_127560)
        
        # Assigning a Name to a Attribute (line 153):
        
        # Assigning a Name to a Attribute (line 153):
        # Getting the type of 'scale' (line 153)
        scale_127562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'scale')
        # Getting the type of 'self' (line 153)
        self_127563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'self')
        # Setting the type of the member 'scale' of a type (line 153)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), self_127563, 'scale', scale_127562)
        
        # Assigning a Name to a Attribute (line 154):
        
        # Assigning a Name to a Attribute (line 154):
        # Getting the type of 'gap' (line 154)
        gap_127564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'gap')
        # Getting the type of 'self' (line 154)
        self_127565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Setting the type of the member 'gap' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_127565, 'gap', gap_127564)
        
        # Assigning a Name to a Attribute (line 155):
        
        # Assigning a Name to a Attribute (line 155):
        # Getting the type of 'radius' (line 155)
        radius_127566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'radius')
        # Getting the type of 'self' (line 155)
        self_127567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self')
        # Setting the type of the member 'radius' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_127567, 'radius', radius_127566)
        
        # Assigning a Name to a Attribute (line 156):
        
        # Assigning a Name to a Attribute (line 156):
        # Getting the type of 'shoulder' (line 156)
        shoulder_127568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'shoulder')
        # Getting the type of 'self' (line 156)
        self_127569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self')
        # Setting the type of the member 'shoulder' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_127569, 'shoulder', shoulder_127568)
        
        # Assigning a Name to a Attribute (line 157):
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'offset' (line 157)
        offset_127570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'offset')
        # Getting the type of 'self' (line 157)
        self_127571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'offset' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_127571, 'offset', offset_127570)
        
        # Assigning a Name to a Attribute (line 158):
        
        # Assigning a Name to a Attribute (line 158):
        # Getting the type of 'margin' (line 158)
        margin_127572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'margin')
        # Getting the type of 'self' (line 158)
        self_127573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self')
        # Setting the type of the member 'margin' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_127573, 'margin', margin_127572)
        
        # Assigning a Call to a Attribute (line 159):
        
        # Assigning a Call to a Attribute (line 159):
        
        # Call to tan(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'np' (line 159)
        np_127576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'np', False)
        # Obtaining the member 'pi' of a type (line 159)
        pi_127577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), np_127576, 'pi')
        int_127578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 37), 'int')
        # Getting the type of 'head_angle' (line 159)
        head_angle_127579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 'head_angle', False)
        float_127580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 54), 'float')
        # Applying the binary operator 'div' (line 159)
        result_div_127581 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 41), 'div', head_angle_127579, float_127580)
        
        # Applying the binary operator '-' (line 159)
        result_sub_127582 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 37), '-', int_127578, result_div_127581)
        
        # Applying the binary operator '*' (line 159)
        result_mul_127583 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 28), '*', pi_127577, result_sub_127582)
        
        float_127584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 63), 'float')
        # Applying the binary operator 'div' (line 159)
        result_div_127585 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 61), 'div', result_mul_127583, float_127584)
        
        # Processing the call keyword arguments (line 159)
        kwargs_127586 = {}
        # Getting the type of 'np' (line 159)
        np_127574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'np', False)
        # Obtaining the member 'tan' of a type (line 159)
        tan_127575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 21), np_127574, 'tan')
        # Calling tan(args, kwargs) (line 159)
        tan_call_result_127587 = invoke(stypy.reporting.localization.Localization(__file__, 159, 21), tan_127575, *[result_div_127585], **kwargs_127586)
        
        # Getting the type of 'self' (line 159)
        self_127588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self')
        # Setting the type of the member 'pitch' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_127588, 'pitch', tan_call_result_127587)
        
        # Assigning a Name to a Attribute (line 160):
        
        # Assigning a Name to a Attribute (line 160):
        # Getting the type of 'tolerance' (line 160)
        tolerance_127589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'tolerance')
        # Getting the type of 'self' (line 160)
        self_127590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self')
        # Setting the type of the member 'tolerance' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_127590, 'tolerance', tolerance_127589)
        
        # Assigning a Call to a Attribute (line 163):
        
        # Assigning a Call to a Attribute (line 163):
        
        # Call to array(...): (line 163)
        # Processing the call arguments (line 163)
        
        # Obtaining an instance of the builtin type 'tuple' (line 163)
        tuple_127593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 163)
        # Adding element type (line 163)
        # Getting the type of 'np' (line 163)
        np_127594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 32), 'np', False)
        # Obtaining the member 'inf' of a type (line 163)
        inf_127595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 32), np_127594, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 32), tuple_127593, inf_127595)
        # Adding element type (line 163)
        
        # Getting the type of 'np' (line 163)
        np_127596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'np', False)
        # Obtaining the member 'inf' of a type (line 163)
        inf_127597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 41), np_127596, 'inf')
        # Applying the 'usub' unary operator (line 163)
        result___neg___127598 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 40), 'usub', inf_127597)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 32), tuple_127593, result___neg___127598)
        # Adding element type (line 163)
        # Getting the type of 'np' (line 163)
        np_127599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 49), 'np', False)
        # Obtaining the member 'inf' of a type (line 163)
        inf_127600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 49), np_127599, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 32), tuple_127593, inf_127600)
        # Adding element type (line 163)
        
        # Getting the type of 'np' (line 163)
        np_127601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 58), 'np', False)
        # Obtaining the member 'inf' of a type (line 163)
        inf_127602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 58), np_127601, 'inf')
        # Applying the 'usub' unary operator (line 163)
        result___neg___127603 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 57), 'usub', inf_127602)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 32), tuple_127593, result___neg___127603)
        
        # Processing the call keyword arguments (line 163)
        kwargs_127604 = {}
        # Getting the type of 'np' (line 163)
        np_127591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 163)
        array_127592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 22), np_127591, 'array')
        # Calling array(args, kwargs) (line 163)
        array_call_result_127605 = invoke(stypy.reporting.localization.Localization(__file__, 163, 22), array_127592, *[tuple_127593], **kwargs_127604)
        
        # Getting the type of 'self' (line 163)
        self_127606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self')
        # Setting the type of the member 'extent' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_127606, 'extent', array_call_result_127605)
        
        
        # Call to len(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'kwargs' (line 166)
        kwargs_127608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'kwargs', False)
        # Processing the call keyword arguments (line 166)
        kwargs_127609 = {}
        # Getting the type of 'len' (line 166)
        len_127607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'len', False)
        # Calling len(args, kwargs) (line 166)
        len_call_result_127610 = invoke(stypy.reporting.localization.Localization(__file__, 166, 11), len_127607, *[kwargs_127608], **kwargs_127609)
        
        # Testing the type of an if condition (line 166)
        if_condition_127611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 8), len_call_result_127610)
        # Assigning a type to the variable 'if_condition_127611' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'if_condition_127611', if_condition_127611)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add(...): (line 167)
        # Processing the call keyword arguments (line 167)
        # Getting the type of 'kwargs' (line 167)
        kwargs_127614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'kwargs', False)
        kwargs_127615 = {'kwargs_127614': kwargs_127614}
        # Getting the type of 'self' (line 167)
        self_127612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'self', False)
        # Obtaining the member 'add' of a type (line 167)
        add_127613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), self_127612, 'add')
        # Calling add(args, kwargs) (line 167)
        add_call_result_127616 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), add_127613, *[], **kwargs_127615)
        
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _arc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_127617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 28), 'int')
        # Getting the type of 'True' (line 169)
        True_127618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'True')
        int_127619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 47), 'int')
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_127620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        int_127621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 58), tuple_127620, int_127621)
        # Adding element type (line 169)
        int_127622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 58), tuple_127620, int_127622)
        
        defaults = [int_127617, True_127618, int_127619, tuple_127620]
        # Create a new context for function '_arc'
        module_type_store = module_type_store.open_function_context('_arc', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Sankey._arc.__dict__.__setitem__('stypy_localization', localization)
        Sankey._arc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Sankey._arc.__dict__.__setitem__('stypy_type_store', module_type_store)
        Sankey._arc.__dict__.__setitem__('stypy_function_name', 'Sankey._arc')
        Sankey._arc.__dict__.__setitem__('stypy_param_names_list', ['quadrant', 'cw', 'radius', 'center'])
        Sankey._arc.__dict__.__setitem__('stypy_varargs_param_name', None)
        Sankey._arc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Sankey._arc.__dict__.__setitem__('stypy_call_defaults', defaults)
        Sankey._arc.__dict__.__setitem__('stypy_call_varargs', varargs)
        Sankey._arc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Sankey._arc.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sankey._arc', ['quadrant', 'cw', 'radius', 'center'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_arc', localization, ['quadrant', 'cw', 'radius', 'center'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_arc(...)' code ##################

        unicode_127623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, (-1)), 'unicode', u"\n        Return the codes and vertices for a rotated, scaled, and translated\n        90 degree arc.\n\n        Optional keyword arguments:\n\n          ===============   ==========================================\n          Keyword           Description\n          ===============   ==========================================\n          *quadrant*        uses 0-based indexing (0, 1, 2, or 3)\n          *cw*              if True, clockwise\n          *center*          (x, y) tuple of the arc's center\n          ===============   ==========================================\n        ")
        
        # Assigning a List to a Name (line 187):
        
        # Assigning a List to a Name (line 187):
        
        # Obtaining an instance of the builtin type 'list' (line 187)
        list_127624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 187)
        # Adding element type (line 187)
        # Getting the type of 'Path' (line 187)
        Path_127625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 21), 'Path')
        # Obtaining the member 'LINETO' of a type (line 187)
        LINETO_127626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 21), Path_127625, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), list_127624, LINETO_127626)
        # Adding element type (line 187)
        # Getting the type of 'Path' (line 188)
        Path_127627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 188)
        CURVE4_127628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 21), Path_127627, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), list_127624, CURVE4_127628)
        # Adding element type (line 187)
        # Getting the type of 'Path' (line 189)
        Path_127629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 189)
        CURVE4_127630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 21), Path_127629, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), list_127624, CURVE4_127630)
        # Adding element type (line 187)
        # Getting the type of 'Path' (line 190)
        Path_127631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 190)
        CURVE4_127632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 21), Path_127631, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), list_127624, CURVE4_127632)
        # Adding element type (line 187)
        # Getting the type of 'Path' (line 191)
        Path_127633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 21), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 191)
        CURVE4_127634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 21), Path_127633, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), list_127624, CURVE4_127634)
        # Adding element type (line 187)
        # Getting the type of 'Path' (line 192)
        Path_127635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 192)
        CURVE4_127636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 21), Path_127635, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), list_127624, CURVE4_127636)
        # Adding element type (line 187)
        # Getting the type of 'Path' (line 193)
        Path_127637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 21), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 193)
        CURVE4_127638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 21), Path_127637, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), list_127624, CURVE4_127638)
        
        # Assigning a type to the variable 'ARC_CODES' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'ARC_CODES', list_127624)
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to array(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_127641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_127642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        float_127643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 33), list_127642, float_127643)
        # Adding element type (line 196)
        float_127644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 33), list_127642, float_127644)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), list_127641, list_127642)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_127645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        float_127646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 33), list_127645, float_127646)
        # Adding element type (line 197)
        float_127647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 33), list_127645, float_127647)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), list_127641, list_127645)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_127648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        # Adding element type (line 198)
        float_127649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 33), list_127648, float_127649)
        # Adding element type (line 198)
        float_127650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 33), list_127648, float_127650)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), list_127641, list_127648)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_127651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        float_127652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 33), list_127651, float_127652)
        # Adding element type (line 199)
        float_127653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 33), list_127651, float_127653)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), list_127641, list_127651)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_127654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        float_127655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 33), list_127654, float_127655)
        # Adding element type (line 200)
        float_127656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 33), list_127654, float_127656)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), list_127641, list_127654)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_127657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        float_127658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 33), list_127657, float_127658)
        # Adding element type (line 201)
        float_127659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 33), list_127657, float_127659)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), list_127641, list_127657)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_127660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        float_127661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 33), list_127660, float_127661)
        # Adding element type (line 204)
        float_127662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 33), list_127660, float_127662)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), list_127641, list_127660)
        
        # Processing the call keyword arguments (line 196)
        kwargs_127663 = {}
        # Getting the type of 'np' (line 196)
        np_127639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 196)
        array_127640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 23), np_127639, 'array')
        # Calling array(args, kwargs) (line 196)
        array_call_result_127664 = invoke(stypy.reporting.localization.Localization(__file__, 196, 23), array_127640, *[list_127641], **kwargs_127663)
        
        # Assigning a type to the variable 'ARC_VERTICES' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'ARC_VERTICES', array_call_result_127664)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'quadrant' (line 205)
        quadrant_127665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'quadrant')
        int_127666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 23), 'int')
        # Applying the binary operator '==' (line 205)
        result_eq_127667 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), '==', quadrant_127665, int_127666)
        
        
        # Getting the type of 'quadrant' (line 205)
        quadrant_127668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 28), 'quadrant')
        int_127669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 40), 'int')
        # Applying the binary operator '==' (line 205)
        result_eq_127670 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 28), '==', quadrant_127668, int_127669)
        
        # Applying the binary operator 'or' (line 205)
        result_or_keyword_127671 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), 'or', result_eq_127667, result_eq_127670)
        
        # Testing the type of an if condition (line 205)
        if_condition_127672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_or_keyword_127671)
        # Assigning a type to the variable 'if_condition_127672' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_127672', if_condition_127672)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'cw' (line 206)
        cw_127673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'cw')
        # Testing the type of an if condition (line 206)
        if_condition_127674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 12), cw_127673)
        # Assigning a type to the variable 'if_condition_127674' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'if_condition_127674', if_condition_127674)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 207):
        
        # Assigning a Name to a Name (line 207):
        # Getting the type of 'ARC_VERTICES' (line 207)
        ARC_VERTICES_127675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'ARC_VERTICES')
        # Assigning a type to the variable 'vertices' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'vertices', ARC_VERTICES_127675)
        # SSA branch for the else part of an if statement (line 206)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 209):
        
        # Assigning a Subscript to a Name (line 209):
        
        # Obtaining the type of the subscript
        slice_127676 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 209, 27), None, None, None)
        int_127677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 45), 'int')
        slice_127678 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 209, 27), None, None, int_127677)
        # Getting the type of 'ARC_VERTICES' (line 209)
        ARC_VERTICES_127679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'ARC_VERTICES')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___127680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 27), ARC_VERTICES_127679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_127681 = invoke(stypy.reporting.localization.Localization(__file__, 209, 27), getitem___127680, (slice_127676, slice_127678))
        
        # Assigning a type to the variable 'vertices' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'vertices', subscript_call_result_127681)
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 205)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'quadrant' (line 210)
        quadrant_127682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'quadrant')
        int_127683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 25), 'int')
        # Applying the binary operator '==' (line 210)
        result_eq_127684 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 13), '==', quadrant_127682, int_127683)
        
        
        # Getting the type of 'quadrant' (line 210)
        quadrant_127685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'quadrant')
        int_127686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 42), 'int')
        # Applying the binary operator '==' (line 210)
        result_eq_127687 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 30), '==', quadrant_127685, int_127686)
        
        # Applying the binary operator 'or' (line 210)
        result_or_keyword_127688 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 13), 'or', result_eq_127684, result_eq_127687)
        
        # Testing the type of an if condition (line 210)
        if_condition_127689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 13), result_or_keyword_127688)
        # Assigning a type to the variable 'if_condition_127689' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'if_condition_127689', if_condition_127689)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'cw' (line 212)
        cw_127690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'cw')
        # Testing the type of an if condition (line 212)
        if_condition_127691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 12), cw_127690)
        # Assigning a type to the variable 'if_condition_127691' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'if_condition_127691', if_condition_127691)
        # SSA begins for if statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to column_stack(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Obtaining an instance of the builtin type 'tuple' (line 214)
        tuple_127694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 214)
        # Adding element type (line 214)
        
        
        # Obtaining the type of the subscript
        slice_127695 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 214, 45), None, None, None)
        int_127696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 61), 'int')
        # Getting the type of 'ARC_VERTICES' (line 214)
        ARC_VERTICES_127697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 45), 'ARC_VERTICES', False)
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___127698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 45), ARC_VERTICES_127697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_127699 = invoke(stypy.reporting.localization.Localization(__file__, 214, 45), getitem___127698, (slice_127695, int_127696))
        
        # Applying the 'usub' unary operator (line 214)
        result___neg___127700 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 44), 'usub', subscript_call_result_127699)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 44), tuple_127694, result___neg___127700)
        # Adding element type (line 214)
        
        # Obtaining the type of the subscript
        slice_127701 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 45), None, None, None)
        int_127702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 61), 'int')
        # Getting the type of 'ARC_VERTICES' (line 215)
        ARC_VERTICES_127703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 45), 'ARC_VERTICES', False)
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___127704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 45), ARC_VERTICES_127703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_127705 = invoke(stypy.reporting.localization.Localization(__file__, 215, 45), getitem___127704, (slice_127701, int_127702))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 44), tuple_127694, subscript_call_result_127705)
        
        # Processing the call keyword arguments (line 214)
        kwargs_127706 = {}
        # Getting the type of 'np' (line 214)
        np_127692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 214)
        column_stack_127693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 27), np_127692, 'column_stack')
        # Calling column_stack(args, kwargs) (line 214)
        column_stack_call_result_127707 = invoke(stypy.reporting.localization.Localization(__file__, 214, 27), column_stack_127693, *[tuple_127694], **kwargs_127706)
        
        # Assigning a type to the variable 'vertices' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'vertices', column_stack_call_result_127707)
        # SSA branch for the else part of an if statement (line 212)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to column_stack(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_127710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        
        
        # Obtaining the type of the subscript
        slice_127711 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 217, 45), None, None, None)
        int_127712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 61), 'int')
        # Getting the type of 'ARC_VERTICES' (line 217)
        ARC_VERTICES_127713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 45), 'ARC_VERTICES', False)
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___127714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 45), ARC_VERTICES_127713, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_127715 = invoke(stypy.reporting.localization.Localization(__file__, 217, 45), getitem___127714, (slice_127711, int_127712))
        
        # Applying the 'usub' unary operator (line 217)
        result___neg___127716 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 44), 'usub', subscript_call_result_127715)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 44), tuple_127710, result___neg___127716)
        # Adding element type (line 217)
        
        # Obtaining the type of the subscript
        slice_127717 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 218, 45), None, None, None)
        int_127718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 61), 'int')
        # Getting the type of 'ARC_VERTICES' (line 218)
        ARC_VERTICES_127719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 45), 'ARC_VERTICES', False)
        # Obtaining the member '__getitem__' of a type (line 218)
        getitem___127720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 45), ARC_VERTICES_127719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 218)
        subscript_call_result_127721 = invoke(stypy.reporting.localization.Localization(__file__, 218, 45), getitem___127720, (slice_127717, int_127718))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 44), tuple_127710, subscript_call_result_127721)
        
        # Processing the call keyword arguments (line 217)
        kwargs_127722 = {}
        # Getting the type of 'np' (line 217)
        np_127708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 27), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 217)
        column_stack_127709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 27), np_127708, 'column_stack')
        # Calling column_stack(args, kwargs) (line 217)
        column_stack_call_result_127723 = invoke(stypy.reporting.localization.Localization(__file__, 217, 27), column_stack_127709, *[tuple_127710], **kwargs_127722)
        
        # Assigning a type to the variable 'vertices' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'vertices', column_stack_call_result_127723)
        # SSA join for if statement (line 212)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'quadrant' (line 219)
        quadrant_127724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'quadrant')
        int_127725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 22), 'int')
        # Applying the binary operator '>' (line 219)
        result_gt_127726 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), '>', quadrant_127724, int_127725)
        
        # Testing the type of an if condition (line 219)
        if_condition_127727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 8), result_gt_127726)
        # Assigning a type to the variable 'if_condition_127727' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'if_condition_127727', if_condition_127727)
        # SSA begins for if statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Name (line 220):
        
        # Assigning a UnaryOp to a Name (line 220):
        
        # Getting the type of 'radius' (line 220)
        radius_127728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'radius')
        # Applying the 'usub' unary operator (line 220)
        result___neg___127729 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 21), 'usub', radius_127728)
        
        # Assigning a type to the variable 'radius' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'radius', result___neg___127729)
        # SSA join for if statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to list(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Call to zip(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'ARC_CODES' (line 221)
        ARC_CODES_127732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'ARC_CODES', False)
        # Getting the type of 'radius' (line 221)
        radius_127733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 'radius', False)
        # Getting the type of 'vertices' (line 221)
        vertices_127734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'vertices', False)
        # Applying the binary operator '*' (line 221)
        result_mul_127735 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 35), '*', radius_127733, vertices_127734)
        
        
        # Call to tile(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'center' (line 222)
        center_127738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 32), 'center', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 222)
        tuple_127739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 222)
        # Adding element type (line 222)
        
        # Obtaining the type of the subscript
        int_127740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 60), 'int')
        # Getting the type of 'ARC_VERTICES' (line 222)
        ARC_VERTICES_127741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 41), 'ARC_VERTICES', False)
        # Obtaining the member 'shape' of a type (line 222)
        shape_127742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 41), ARC_VERTICES_127741, 'shape')
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___127743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 41), shape_127742, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_127744 = invoke(stypy.reporting.localization.Localization(__file__, 222, 41), getitem___127743, int_127740)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 41), tuple_127739, subscript_call_result_127744)
        # Adding element type (line 222)
        int_127745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 41), tuple_127739, int_127745)
        
        # Processing the call keyword arguments (line 222)
        kwargs_127746 = {}
        # Getting the type of 'np' (line 222)
        np_127736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'np', False)
        # Obtaining the member 'tile' of a type (line 222)
        tile_127737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 24), np_127736, 'tile')
        # Calling tile(args, kwargs) (line 222)
        tile_call_result_127747 = invoke(stypy.reporting.localization.Localization(__file__, 222, 24), tile_127737, *[center_127738, tuple_127739], **kwargs_127746)
        
        # Applying the binary operator '+' (line 221)
        result_add_127748 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 35), '+', result_mul_127735, tile_call_result_127747)
        
        # Processing the call keyword arguments (line 221)
        kwargs_127749 = {}
        # Getting the type of 'zip' (line 221)
        zip_127731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 221)
        zip_call_result_127750 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), zip_127731, *[ARC_CODES_127732, result_add_127748], **kwargs_127749)
        
        # Processing the call keyword arguments (line 221)
        kwargs_127751 = {}
        # Getting the type of 'list' (line 221)
        list_127730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'list', False)
        # Calling list(args, kwargs) (line 221)
        list_call_result_127752 = invoke(stypy.reporting.localization.Localization(__file__, 221, 15), list_127730, *[zip_call_result_127750], **kwargs_127751)
        
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', list_call_result_127752)
        
        # ################# End of '_arc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_arc' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_127753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_arc'
        return stypy_return_type_127753


    @norecursion
    def _add_input(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_input'
        module_type_store = module_type_store.open_function_context('_add_input', 224, 4, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Sankey._add_input.__dict__.__setitem__('stypy_localization', localization)
        Sankey._add_input.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Sankey._add_input.__dict__.__setitem__('stypy_type_store', module_type_store)
        Sankey._add_input.__dict__.__setitem__('stypy_function_name', 'Sankey._add_input')
        Sankey._add_input.__dict__.__setitem__('stypy_param_names_list', ['path', 'angle', 'flow', 'length'])
        Sankey._add_input.__dict__.__setitem__('stypy_varargs_param_name', None)
        Sankey._add_input.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Sankey._add_input.__dict__.__setitem__('stypy_call_defaults', defaults)
        Sankey._add_input.__dict__.__setitem__('stypy_call_varargs', varargs)
        Sankey._add_input.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Sankey._add_input.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sankey._add_input', ['path', 'angle', 'flow', 'length'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_input', localization, ['path', 'angle', 'flow', 'length'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_input(...)' code ##################

        unicode_127754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, (-1)), 'unicode', u'\n        Add an input to a path and return its tip and label locations.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 228)
        # Getting the type of 'angle' (line 228)
        angle_127755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'angle')
        # Getting the type of 'None' (line 228)
        None_127756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'None')
        
        (may_be_127757, more_types_in_union_127758) = may_be_none(angle_127755, None_127756)

        if may_be_127757:

            if more_types_in_union_127758:
                # Runtime conditional SSA (line 228)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Obtaining an instance of the builtin type 'tuple' (line 229)
            tuple_127759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 229)
            # Adding element type (line 229)
            
            # Obtaining an instance of the builtin type 'list' (line 229)
            list_127760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 229)
            # Adding element type (line 229)
            int_127761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 20), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 19), list_127760, int_127761)
            # Adding element type (line 229)
            int_127762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 23), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 19), list_127760, int_127762)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 19), tuple_127759, list_127760)
            # Adding element type (line 229)
            
            # Obtaining an instance of the builtin type 'list' (line 229)
            list_127763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 229)
            # Adding element type (line 229)
            int_127764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 28), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 27), list_127763, int_127764)
            # Adding element type (line 229)
            int_127765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 31), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 27), list_127763, int_127765)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 19), tuple_127759, list_127763)
            
            # Assigning a type to the variable 'stypy_return_type' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'stypy_return_type', tuple_127759)

            if more_types_in_union_127758:
                # Runtime conditional SSA for else branch (line 228)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_127757) or more_types_in_union_127758):
            
            # Assigning a Subscript to a Tuple (line 231):
            
            # Assigning a Subscript to a Name (line 231):
            
            # Obtaining the type of the subscript
            int_127766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
            
            # Obtaining the type of the subscript
            int_127767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'int')
            
            # Obtaining the type of the subscript
            int_127768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 24), 'int')
            # Getting the type of 'path' (line 231)
            path_127769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'path')
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___127770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 19), path_127769, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_127771 = invoke(stypy.reporting.localization.Localization(__file__, 231, 19), getitem___127770, int_127768)
            
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___127772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 19), subscript_call_result_127771, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_127773 = invoke(stypy.reporting.localization.Localization(__file__, 231, 19), getitem___127772, int_127767)
            
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___127774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), subscript_call_result_127773, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_127775 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___127774, int_127766)
            
            # Assigning a type to the variable 'tuple_var_assignment_127431' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_127431', subscript_call_result_127775)
            
            # Assigning a Subscript to a Name (line 231):
            
            # Obtaining the type of the subscript
            int_127776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
            
            # Obtaining the type of the subscript
            int_127777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'int')
            
            # Obtaining the type of the subscript
            int_127778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 24), 'int')
            # Getting the type of 'path' (line 231)
            path_127779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'path')
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___127780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 19), path_127779, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_127781 = invoke(stypy.reporting.localization.Localization(__file__, 231, 19), getitem___127780, int_127778)
            
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___127782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 19), subscript_call_result_127781, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_127783 = invoke(stypy.reporting.localization.Localization(__file__, 231, 19), getitem___127782, int_127777)
            
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___127784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), subscript_call_result_127783, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_127785 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___127784, int_127776)
            
            # Assigning a type to the variable 'tuple_var_assignment_127432' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_127432', subscript_call_result_127785)
            
            # Assigning a Name to a Name (line 231):
            # Getting the type of 'tuple_var_assignment_127431' (line 231)
            tuple_var_assignment_127431_127786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_127431')
            # Assigning a type to the variable 'x' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'x', tuple_var_assignment_127431_127786)
            
            # Assigning a Name to a Name (line 231):
            # Getting the type of 'tuple_var_assignment_127432' (line 231)
            tuple_var_assignment_127432_127787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_127432')
            # Assigning a type to the variable 'y' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'y', tuple_var_assignment_127432_127787)
            
            # Assigning a BinOp to a Name (line 232):
            
            # Assigning a BinOp to a Name (line 232):
            # Getting the type of 'flow' (line 232)
            flow_127788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'flow')
            int_127789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 31), 'int')
            # Applying the binary operator 'div' (line 232)
            result_div_127790 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 24), 'div', flow_127788, int_127789)
            
            # Getting the type of 'self' (line 232)
            self_127791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 36), 'self')
            # Obtaining the member 'pitch' of a type (line 232)
            pitch_127792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 36), self_127791, 'pitch')
            # Applying the binary operator '*' (line 232)
            result_mul_127793 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 23), '*', result_div_127790, pitch_127792)
            
            # Assigning a type to the variable 'dipdepth' (line 232)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'dipdepth', result_mul_127793)
            
            
            # Getting the type of 'angle' (line 233)
            angle_127794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'angle')
            # Getting the type of 'RIGHT' (line 233)
            RIGHT_127795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'RIGHT')
            # Applying the binary operator '==' (line 233)
            result_eq_127796 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), '==', angle_127794, RIGHT_127795)
            
            # Testing the type of an if condition (line 233)
            if_condition_127797 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 12), result_eq_127796)
            # Assigning a type to the variable 'if_condition_127797' (line 233)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'if_condition_127797', if_condition_127797)
            # SSA begins for if statement (line 233)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'x' (line 234)
            x_127798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'x')
            # Getting the type of 'length' (line 234)
            length_127799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'length')
            # Applying the binary operator '-=' (line 234)
            result_isub_127800 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 16), '-=', x_127798, length_127799)
            # Assigning a type to the variable 'x' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'x', result_isub_127800)
            
            
            # Assigning a List to a Name (line 235):
            
            # Assigning a List to a Name (line 235):
            
            # Obtaining an instance of the builtin type 'list' (line 235)
            list_127801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 235)
            # Adding element type (line 235)
            # Getting the type of 'x' (line 235)
            x_127802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'x')
            # Getting the type of 'dipdepth' (line 235)
            dipdepth_127803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 27), 'dipdepth')
            # Applying the binary operator '+' (line 235)
            result_add_127804 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 23), '+', x_127802, dipdepth_127803)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 22), list_127801, result_add_127804)
            # Adding element type (line 235)
            # Getting the type of 'y' (line 235)
            y_127805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 37), 'y')
            # Getting the type of 'flow' (line 235)
            flow_127806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 41), 'flow')
            float_127807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 48), 'float')
            # Applying the binary operator 'div' (line 235)
            result_div_127808 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 41), 'div', flow_127806, float_127807)
            
            # Applying the binary operator '+' (line 235)
            result_add_127809 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 37), '+', y_127805, result_div_127808)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 22), list_127801, result_add_127809)
            
            # Assigning a type to the variable 'dip' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'dip', list_127801)
            
            # Call to extend(...): (line 236)
            # Processing the call arguments (line 236)
            
            # Obtaining an instance of the builtin type 'list' (line 236)
            list_127812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 236)
            # Adding element type (line 236)
            
            # Obtaining an instance of the builtin type 'tuple' (line 236)
            tuple_127813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 236)
            # Adding element type (line 236)
            # Getting the type of 'Path' (line 236)
            Path_127814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 236)
            LINETO_127815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 30), Path_127814, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 30), tuple_127813, LINETO_127815)
            # Adding element type (line 236)
            
            # Obtaining an instance of the builtin type 'list' (line 236)
            list_127816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 236)
            # Adding element type (line 236)
            # Getting the type of 'x' (line 236)
            x_127817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 43), list_127816, x_127817)
            # Adding element type (line 236)
            # Getting the type of 'y' (line 236)
            y_127818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 47), 'y', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 43), list_127816, y_127818)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 30), tuple_127813, list_127816)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 28), list_127812, tuple_127813)
            # Adding element type (line 236)
            
            # Obtaining an instance of the builtin type 'tuple' (line 237)
            tuple_127819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 237)
            # Adding element type (line 237)
            # Getting the type of 'Path' (line 237)
            Path_127820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 237)
            LINETO_127821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 30), Path_127820, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 30), tuple_127819, LINETO_127821)
            # Adding element type (line 237)
            # Getting the type of 'dip' (line 237)
            dip_127822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 43), 'dip', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 30), tuple_127819, dip_127822)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 28), list_127812, tuple_127819)
            # Adding element type (line 236)
            
            # Obtaining an instance of the builtin type 'tuple' (line 238)
            tuple_127823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 238)
            # Adding element type (line 238)
            # Getting the type of 'Path' (line 238)
            Path_127824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 238)
            LINETO_127825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 30), Path_127824, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 30), tuple_127823, LINETO_127825)
            # Adding element type (line 238)
            
            # Obtaining an instance of the builtin type 'list' (line 238)
            list_127826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 238)
            # Adding element type (line 238)
            # Getting the type of 'x' (line 238)
            x_127827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 43), list_127826, x_127827)
            # Adding element type (line 238)
            # Getting the type of 'y' (line 238)
            y_127828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'y', False)
            # Getting the type of 'flow' (line 238)
            flow_127829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 51), 'flow', False)
            # Applying the binary operator '+' (line 238)
            result_add_127830 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 47), '+', y_127828, flow_127829)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 43), list_127826, result_add_127830)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 30), tuple_127823, list_127826)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 28), list_127812, tuple_127823)
            # Adding element type (line 236)
            
            # Obtaining an instance of the builtin type 'tuple' (line 239)
            tuple_127831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 239)
            # Adding element type (line 239)
            # Getting the type of 'Path' (line 239)
            Path_127832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 239)
            LINETO_127833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 30), Path_127832, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 30), tuple_127831, LINETO_127833)
            # Adding element type (line 239)
            
            # Obtaining an instance of the builtin type 'list' (line 239)
            list_127834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 239)
            # Adding element type (line 239)
            # Getting the type of 'x' (line 239)
            x_127835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 44), 'x', False)
            # Getting the type of 'self' (line 239)
            self_127836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 48), 'self', False)
            # Obtaining the member 'gap' of a type (line 239)
            gap_127837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 48), self_127836, 'gap')
            # Applying the binary operator '+' (line 239)
            result_add_127838 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 44), '+', x_127835, gap_127837)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 43), list_127834, result_add_127838)
            # Adding element type (line 239)
            # Getting the type of 'y' (line 239)
            y_127839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 58), 'y', False)
            # Getting the type of 'flow' (line 239)
            flow_127840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 62), 'flow', False)
            # Applying the binary operator '+' (line 239)
            result_add_127841 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 58), '+', y_127839, flow_127840)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 43), list_127834, result_add_127841)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 30), tuple_127831, list_127834)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 28), list_127812, tuple_127831)
            
            # Processing the call keyword arguments (line 236)
            kwargs_127842 = {}
            # Getting the type of 'path' (line 236)
            path_127810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'path', False)
            # Obtaining the member 'extend' of a type (line 236)
            extend_127811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), path_127810, 'extend')
            # Calling extend(args, kwargs) (line 236)
            extend_call_result_127843 = invoke(stypy.reporting.localization.Localization(__file__, 236, 16), extend_127811, *[list_127812], **kwargs_127842)
            
            
            # Assigning a List to a Name (line 240):
            
            # Assigning a List to a Name (line 240):
            
            # Obtaining an instance of the builtin type 'list' (line 240)
            list_127844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 240)
            # Adding element type (line 240)
            
            # Obtaining the type of the subscript
            int_127845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 38), 'int')
            # Getting the type of 'dip' (line 240)
            dip_127846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 34), 'dip')
            # Obtaining the member '__getitem__' of a type (line 240)
            getitem___127847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 34), dip_127846, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 240)
            subscript_call_result_127848 = invoke(stypy.reporting.localization.Localization(__file__, 240, 34), getitem___127847, int_127845)
            
            # Getting the type of 'self' (line 240)
            self_127849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 43), 'self')
            # Obtaining the member 'offset' of a type (line 240)
            offset_127850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 43), self_127849, 'offset')
            # Applying the binary operator '-' (line 240)
            result_sub_127851 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 34), '-', subscript_call_result_127848, offset_127850)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 33), list_127844, result_sub_127851)
            # Adding element type (line 240)
            
            # Obtaining the type of the subscript
            int_127852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 60), 'int')
            # Getting the type of 'dip' (line 240)
            dip_127853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 56), 'dip')
            # Obtaining the member '__getitem__' of a type (line 240)
            getitem___127854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 56), dip_127853, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 240)
            subscript_call_result_127855 = invoke(stypy.reporting.localization.Localization(__file__, 240, 56), getitem___127854, int_127852)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 33), list_127844, subscript_call_result_127855)
            
            # Assigning a type to the variable 'label_location' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'label_location', list_127844)
            # SSA branch for the else part of an if statement (line 233)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'x' (line 242)
            x_127856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'x')
            # Getting the type of 'self' (line 242)
            self_127857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'self')
            # Obtaining the member 'gap' of a type (line 242)
            gap_127858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), self_127857, 'gap')
            # Applying the binary operator '-=' (line 242)
            result_isub_127859 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 16), '-=', x_127856, gap_127858)
            # Assigning a type to the variable 'x' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'x', result_isub_127859)
            
            
            
            # Getting the type of 'angle' (line 243)
            angle_127860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'angle')
            # Getting the type of 'UP' (line 243)
            UP_127861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'UP')
            # Applying the binary operator '==' (line 243)
            result_eq_127862 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 19), '==', angle_127860, UP_127861)
            
            # Testing the type of an if condition (line 243)
            if_condition_127863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 16), result_eq_127862)
            # Assigning a type to the variable 'if_condition_127863' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'if_condition_127863', if_condition_127863)
            # SSA begins for if statement (line 243)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 244):
            
            # Assigning a Num to a Name (line 244):
            int_127864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 27), 'int')
            # Assigning a type to the variable 'sign' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'sign', int_127864)
            # SSA branch for the else part of an if statement (line 243)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 246):
            
            # Assigning a Num to a Name (line 246):
            int_127865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 27), 'int')
            # Assigning a type to the variable 'sign' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'sign', int_127865)
            # SSA join for if statement (line 243)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a List to a Name (line 248):
            
            # Assigning a List to a Name (line 248):
            
            # Obtaining an instance of the builtin type 'list' (line 248)
            list_127866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 248)
            # Adding element type (line 248)
            # Getting the type of 'x' (line 248)
            x_127867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 23), 'x')
            # Getting the type of 'flow' (line 248)
            flow_127868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'flow')
            int_127869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 34), 'int')
            # Applying the binary operator 'div' (line 248)
            result_div_127870 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 27), 'div', flow_127868, int_127869)
            
            # Applying the binary operator '-' (line 248)
            result_sub_127871 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 23), '-', x_127867, result_div_127870)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 22), list_127866, result_sub_127871)
            # Adding element type (line 248)
            # Getting the type of 'y' (line 248)
            y_127872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 37), 'y')
            # Getting the type of 'sign' (line 248)
            sign_127873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 41), 'sign')
            # Getting the type of 'length' (line 248)
            length_127874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 49), 'length')
            # Getting the type of 'dipdepth' (line 248)
            dipdepth_127875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 58), 'dipdepth')
            # Applying the binary operator '-' (line 248)
            result_sub_127876 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 49), '-', length_127874, dipdepth_127875)
            
            # Applying the binary operator '*' (line 248)
            result_mul_127877 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 41), '*', sign_127873, result_sub_127876)
            
            # Applying the binary operator '-' (line 248)
            result_sub_127878 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 37), '-', y_127872, result_mul_127877)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 22), list_127866, result_sub_127878)
            
            # Assigning a type to the variable 'dip' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'dip', list_127866)
            
            
            # Getting the type of 'angle' (line 249)
            angle_127879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'angle')
            # Getting the type of 'DOWN' (line 249)
            DOWN_127880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 28), 'DOWN')
            # Applying the binary operator '==' (line 249)
            result_eq_127881 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 19), '==', angle_127879, DOWN_127880)
            
            # Testing the type of an if condition (line 249)
            if_condition_127882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 16), result_eq_127881)
            # Assigning a type to the variable 'if_condition_127882' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'if_condition_127882', if_condition_127882)
            # SSA begins for if statement (line 249)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 250):
            
            # Assigning a Num to a Name (line 250):
            int_127883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 31), 'int')
            # Assigning a type to the variable 'quadrant' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'quadrant', int_127883)
            # SSA branch for the else part of an if statement (line 249)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 252):
            
            # Assigning a Num to a Name (line 252):
            int_127884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 31), 'int')
            # Assigning a type to the variable 'quadrant' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 20), 'quadrant', int_127884)
            # SSA join for if statement (line 249)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'self' (line 255)
            self_127885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 19), 'self')
            # Obtaining the member 'radius' of a type (line 255)
            radius_127886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 19), self_127885, 'radius')
            # Testing the type of an if condition (line 255)
            if_condition_127887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 16), radius_127886)
            # Assigning a type to the variable 'if_condition_127887' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'if_condition_127887', if_condition_127887)
            # SSA begins for if statement (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to extend(...): (line 256)
            # Processing the call arguments (line 256)
            
            # Call to _arc(...): (line 256)
            # Processing the call keyword arguments (line 256)
            # Getting the type of 'quadrant' (line 256)
            quadrant_127892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 51), 'quadrant', False)
            keyword_127893 = quadrant_127892
            
            # Getting the type of 'angle' (line 257)
            angle_127894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 45), 'angle', False)
            # Getting the type of 'UP' (line 257)
            UP_127895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 54), 'UP', False)
            # Applying the binary operator '==' (line 257)
            result_eq_127896 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 45), '==', angle_127894, UP_127895)
            
            keyword_127897 = result_eq_127896
            # Getting the type of 'self' (line 258)
            self_127898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 49), 'self', False)
            # Obtaining the member 'radius' of a type (line 258)
            radius_127899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 49), self_127898, 'radius')
            keyword_127900 = radius_127899
            
            # Obtaining an instance of the builtin type 'tuple' (line 259)
            tuple_127901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 50), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 259)
            # Adding element type (line 259)
            # Getting the type of 'x' (line 259)
            x_127902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 50), 'x', False)
            # Getting the type of 'self' (line 259)
            self_127903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 54), 'self', False)
            # Obtaining the member 'radius' of a type (line 259)
            radius_127904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 54), self_127903, 'radius')
            # Applying the binary operator '+' (line 259)
            result_add_127905 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 50), '+', x_127902, radius_127904)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 50), tuple_127901, result_add_127905)
            # Adding element type (line 259)
            # Getting the type of 'y' (line 260)
            y_127906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 50), 'y', False)
            # Getting the type of 'sign' (line 260)
            sign_127907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 54), 'sign', False)
            # Getting the type of 'self' (line 260)
            self_127908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 61), 'self', False)
            # Obtaining the member 'radius' of a type (line 260)
            radius_127909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 61), self_127908, 'radius')
            # Applying the binary operator '*' (line 260)
            result_mul_127910 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 54), '*', sign_127907, radius_127909)
            
            # Applying the binary operator '-' (line 260)
            result_sub_127911 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 50), '-', y_127906, result_mul_127910)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 50), tuple_127901, result_sub_127911)
            
            keyword_127912 = tuple_127901
            kwargs_127913 = {'quadrant': keyword_127893, 'radius': keyword_127900, 'cw': keyword_127897, 'center': keyword_127912}
            # Getting the type of 'self' (line 256)
            self_127890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'self', False)
            # Obtaining the member '_arc' of a type (line 256)
            _arc_127891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 32), self_127890, '_arc')
            # Calling _arc(args, kwargs) (line 256)
            _arc_call_result_127914 = invoke(stypy.reporting.localization.Localization(__file__, 256, 32), _arc_127891, *[], **kwargs_127913)
            
            # Processing the call keyword arguments (line 256)
            kwargs_127915 = {}
            # Getting the type of 'path' (line 256)
            path_127888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'path', False)
            # Obtaining the member 'extend' of a type (line 256)
            extend_127889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 20), path_127888, 'extend')
            # Calling extend(args, kwargs) (line 256)
            extend_call_result_127916 = invoke(stypy.reporting.localization.Localization(__file__, 256, 20), extend_127889, *[_arc_call_result_127914], **kwargs_127915)
            
            # SSA branch for the else part of an if statement (line 255)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 262)
            # Processing the call arguments (line 262)
            
            # Obtaining an instance of the builtin type 'tuple' (line 262)
            tuple_127919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 262)
            # Adding element type (line 262)
            # Getting the type of 'Path' (line 262)
            Path_127920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 33), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 262)
            LINETO_127921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 33), Path_127920, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 33), tuple_127919, LINETO_127921)
            # Adding element type (line 262)
            
            # Obtaining an instance of the builtin type 'list' (line 262)
            list_127922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 46), 'list')
            # Adding type elements to the builtin type 'list' instance (line 262)
            # Adding element type (line 262)
            # Getting the type of 'x' (line 262)
            x_127923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 47), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 46), list_127922, x_127923)
            # Adding element type (line 262)
            # Getting the type of 'y' (line 262)
            y_127924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 50), 'y', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 46), list_127922, y_127924)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 33), tuple_127919, list_127922)
            
            # Processing the call keyword arguments (line 262)
            kwargs_127925 = {}
            # Getting the type of 'path' (line 262)
            path_127917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'path', False)
            # Obtaining the member 'append' of a type (line 262)
            append_127918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 20), path_127917, 'append')
            # Calling append(args, kwargs) (line 262)
            append_call_result_127926 = invoke(stypy.reporting.localization.Localization(__file__, 262, 20), append_127918, *[tuple_127919], **kwargs_127925)
            
            # SSA join for if statement (line 255)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to extend(...): (line 263)
            # Processing the call arguments (line 263)
            
            # Obtaining an instance of the builtin type 'list' (line 263)
            list_127929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 263)
            # Adding element type (line 263)
            
            # Obtaining an instance of the builtin type 'tuple' (line 263)
            tuple_127930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 263)
            # Adding element type (line 263)
            # Getting the type of 'Path' (line 263)
            Path_127931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 263)
            LINETO_127932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 30), Path_127931, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 30), tuple_127930, LINETO_127932)
            # Adding element type (line 263)
            
            # Obtaining an instance of the builtin type 'list' (line 263)
            list_127933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 263)
            # Adding element type (line 263)
            # Getting the type of 'x' (line 263)
            x_127934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 43), list_127933, x_127934)
            # Adding element type (line 263)
            # Getting the type of 'y' (line 263)
            y_127935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 47), 'y', False)
            # Getting the type of 'sign' (line 263)
            sign_127936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 51), 'sign', False)
            # Getting the type of 'length' (line 263)
            length_127937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 58), 'length', False)
            # Applying the binary operator '*' (line 263)
            result_mul_127938 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 51), '*', sign_127936, length_127937)
            
            # Applying the binary operator '-' (line 263)
            result_sub_127939 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 47), '-', y_127935, result_mul_127938)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 43), list_127933, result_sub_127939)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 30), tuple_127930, list_127933)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 28), list_127929, tuple_127930)
            # Adding element type (line 263)
            
            # Obtaining an instance of the builtin type 'tuple' (line 264)
            tuple_127940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 264)
            # Adding element type (line 264)
            # Getting the type of 'Path' (line 264)
            Path_127941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 264)
            LINETO_127942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 30), Path_127941, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 30), tuple_127940, LINETO_127942)
            # Adding element type (line 264)
            # Getting the type of 'dip' (line 264)
            dip_127943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 43), 'dip', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 30), tuple_127940, dip_127943)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 28), list_127929, tuple_127940)
            # Adding element type (line 263)
            
            # Obtaining an instance of the builtin type 'tuple' (line 265)
            tuple_127944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 265)
            # Adding element type (line 265)
            # Getting the type of 'Path' (line 265)
            Path_127945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 265)
            LINETO_127946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 30), Path_127945, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 30), tuple_127944, LINETO_127946)
            # Adding element type (line 265)
            
            # Obtaining an instance of the builtin type 'list' (line 265)
            list_127947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 265)
            # Adding element type (line 265)
            # Getting the type of 'x' (line 265)
            x_127948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 44), 'x', False)
            # Getting the type of 'flow' (line 265)
            flow_127949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 48), 'flow', False)
            # Applying the binary operator '-' (line 265)
            result_sub_127950 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 44), '-', x_127948, flow_127949)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 43), list_127947, result_sub_127950)
            # Adding element type (line 265)
            # Getting the type of 'y' (line 265)
            y_127951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 54), 'y', False)
            # Getting the type of 'sign' (line 265)
            sign_127952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 58), 'sign', False)
            # Getting the type of 'length' (line 265)
            length_127953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 65), 'length', False)
            # Applying the binary operator '*' (line 265)
            result_mul_127954 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 58), '*', sign_127952, length_127953)
            
            # Applying the binary operator '-' (line 265)
            result_sub_127955 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 54), '-', y_127951, result_mul_127954)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 43), list_127947, result_sub_127955)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 30), tuple_127944, list_127947)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 28), list_127929, tuple_127944)
            
            # Processing the call keyword arguments (line 263)
            kwargs_127956 = {}
            # Getting the type of 'path' (line 263)
            path_127927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'path', False)
            # Obtaining the member 'extend' of a type (line 263)
            extend_127928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 16), path_127927, 'extend')
            # Calling extend(args, kwargs) (line 263)
            extend_call_result_127957 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), extend_127928, *[list_127929], **kwargs_127956)
            
            
            # Call to extend(...): (line 266)
            # Processing the call arguments (line 266)
            
            # Call to _arc(...): (line 266)
            # Processing the call keyword arguments (line 266)
            # Getting the type of 'quadrant' (line 266)
            quadrant_127962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 47), 'quadrant', False)
            keyword_127963 = quadrant_127962
            
            # Getting the type of 'angle' (line 267)
            angle_127964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 41), 'angle', False)
            # Getting the type of 'DOWN' (line 267)
            DOWN_127965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 50), 'DOWN', False)
            # Applying the binary operator '==' (line 267)
            result_eq_127966 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 41), '==', angle_127964, DOWN_127965)
            
            keyword_127967 = result_eq_127966
            # Getting the type of 'flow' (line 268)
            flow_127968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 45), 'flow', False)
            # Getting the type of 'self' (line 268)
            self_127969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 52), 'self', False)
            # Obtaining the member 'radius' of a type (line 268)
            radius_127970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 52), self_127969, 'radius')
            # Applying the binary operator '+' (line 268)
            result_add_127971 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 45), '+', flow_127968, radius_127970)
            
            keyword_127972 = result_add_127971
            
            # Obtaining an instance of the builtin type 'tuple' (line 269)
            tuple_127973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 269)
            # Adding element type (line 269)
            # Getting the type of 'x' (line 269)
            x_127974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 46), 'x', False)
            # Getting the type of 'self' (line 269)
            self_127975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 50), 'self', False)
            # Obtaining the member 'radius' of a type (line 269)
            radius_127976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 50), self_127975, 'radius')
            # Applying the binary operator '+' (line 269)
            result_add_127977 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 46), '+', x_127974, radius_127976)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 46), tuple_127973, result_add_127977)
            # Adding element type (line 269)
            # Getting the type of 'y' (line 270)
            y_127978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 46), 'y', False)
            # Getting the type of 'sign' (line 270)
            sign_127979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'sign', False)
            # Getting the type of 'self' (line 270)
            self_127980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 57), 'self', False)
            # Obtaining the member 'radius' of a type (line 270)
            radius_127981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 57), self_127980, 'radius')
            # Applying the binary operator '*' (line 270)
            result_mul_127982 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 50), '*', sign_127979, radius_127981)
            
            # Applying the binary operator '-' (line 270)
            result_sub_127983 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 46), '-', y_127978, result_mul_127982)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 46), tuple_127973, result_sub_127983)
            
            keyword_127984 = tuple_127973
            kwargs_127985 = {'quadrant': keyword_127963, 'radius': keyword_127972, 'cw': keyword_127967, 'center': keyword_127984}
            # Getting the type of 'self' (line 266)
            self_127960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'self', False)
            # Obtaining the member '_arc' of a type (line 266)
            _arc_127961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 28), self_127960, '_arc')
            # Calling _arc(args, kwargs) (line 266)
            _arc_call_result_127986 = invoke(stypy.reporting.localization.Localization(__file__, 266, 28), _arc_127961, *[], **kwargs_127985)
            
            # Processing the call keyword arguments (line 266)
            kwargs_127987 = {}
            # Getting the type of 'path' (line 266)
            path_127958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'path', False)
            # Obtaining the member 'extend' of a type (line 266)
            extend_127959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 16), path_127958, 'extend')
            # Calling extend(args, kwargs) (line 266)
            extend_call_result_127988 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), extend_127959, *[_arc_call_result_127986], **kwargs_127987)
            
            
            # Call to append(...): (line 271)
            # Processing the call arguments (line 271)
            
            # Obtaining an instance of the builtin type 'tuple' (line 271)
            tuple_127991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 271)
            # Adding element type (line 271)
            # Getting the type of 'Path' (line 271)
            Path_127992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 271)
            LINETO_127993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 29), Path_127992, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 29), tuple_127991, LINETO_127993)
            # Adding element type (line 271)
            
            # Obtaining an instance of the builtin type 'list' (line 271)
            list_127994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 42), 'list')
            # Adding type elements to the builtin type 'list' instance (line 271)
            # Adding element type (line 271)
            # Getting the type of 'x' (line 271)
            x_127995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 43), 'x', False)
            # Getting the type of 'flow' (line 271)
            flow_127996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 47), 'flow', False)
            # Applying the binary operator '-' (line 271)
            result_sub_127997 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 43), '-', x_127995, flow_127996)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 42), list_127994, result_sub_127997)
            # Adding element type (line 271)
            # Getting the type of 'y' (line 271)
            y_127998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 53), 'y', False)
            # Getting the type of 'sign' (line 271)
            sign_127999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 57), 'sign', False)
            # Getting the type of 'flow' (line 271)
            flow_128000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 64), 'flow', False)
            # Applying the binary operator '*' (line 271)
            result_mul_128001 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 57), '*', sign_127999, flow_128000)
            
            # Applying the binary operator '+' (line 271)
            result_add_128002 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 53), '+', y_127998, result_mul_128001)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 42), list_127994, result_add_128002)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 29), tuple_127991, list_127994)
            
            # Processing the call keyword arguments (line 271)
            kwargs_128003 = {}
            # Getting the type of 'path' (line 271)
            path_127989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'path', False)
            # Obtaining the member 'append' of a type (line 271)
            append_127990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), path_127989, 'append')
            # Calling append(args, kwargs) (line 271)
            append_call_result_128004 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), append_127990, *[tuple_127991], **kwargs_128003)
            
            
            # Assigning a List to a Name (line 272):
            
            # Assigning a List to a Name (line 272):
            
            # Obtaining an instance of the builtin type 'list' (line 272)
            list_128005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 272)
            # Adding element type (line 272)
            
            # Obtaining the type of the subscript
            int_128006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 38), 'int')
            # Getting the type of 'dip' (line 272)
            dip_128007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 34), 'dip')
            # Obtaining the member '__getitem__' of a type (line 272)
            getitem___128008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 34), dip_128007, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 272)
            subscript_call_result_128009 = invoke(stypy.reporting.localization.Localization(__file__, 272, 34), getitem___128008, int_128006)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 33), list_128005, subscript_call_result_128009)
            # Adding element type (line 272)
            
            # Obtaining the type of the subscript
            int_128010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 46), 'int')
            # Getting the type of 'dip' (line 272)
            dip_128011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 42), 'dip')
            # Obtaining the member '__getitem__' of a type (line 272)
            getitem___128012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 42), dip_128011, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 272)
            subscript_call_result_128013 = invoke(stypy.reporting.localization.Localization(__file__, 272, 42), getitem___128012, int_128010)
            
            # Getting the type of 'sign' (line 272)
            sign_128014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 51), 'sign')
            # Getting the type of 'self' (line 272)
            self_128015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 58), 'self')
            # Obtaining the member 'offset' of a type (line 272)
            offset_128016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 58), self_128015, 'offset')
            # Applying the binary operator '*' (line 272)
            result_mul_128017 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 51), '*', sign_128014, offset_128016)
            
            # Applying the binary operator '-' (line 272)
            result_sub_128018 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 42), '-', subscript_call_result_128013, result_mul_128017)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 33), list_128005, result_sub_128018)
            
            # Assigning a type to the variable 'label_location' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'label_location', list_128005)
            # SSA join for if statement (line 233)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 274)
            tuple_128019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 274)
            # Adding element type (line 274)
            # Getting the type of 'dip' (line 274)
            dip_128020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'dip')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 19), tuple_128019, dip_128020)
            # Adding element type (line 274)
            # Getting the type of 'label_location' (line 274)
            label_location_128021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'label_location')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 19), tuple_128019, label_location_128021)
            
            # Assigning a type to the variable 'stypy_return_type' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'stypy_return_type', tuple_128019)

            if (may_be_127757 and more_types_in_union_127758):
                # SSA join for if statement (line 228)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_add_input(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_input' in the type store
        # Getting the type of 'stypy_return_type' (line 224)
        stypy_return_type_128022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128022)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_input'
        return stypy_return_type_128022


    @norecursion
    def _add_output(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_output'
        module_type_store = module_type_store.open_function_context('_add_output', 276, 4, False)
        # Assigning a type to the variable 'self' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Sankey._add_output.__dict__.__setitem__('stypy_localization', localization)
        Sankey._add_output.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Sankey._add_output.__dict__.__setitem__('stypy_type_store', module_type_store)
        Sankey._add_output.__dict__.__setitem__('stypy_function_name', 'Sankey._add_output')
        Sankey._add_output.__dict__.__setitem__('stypy_param_names_list', ['path', 'angle', 'flow', 'length'])
        Sankey._add_output.__dict__.__setitem__('stypy_varargs_param_name', None)
        Sankey._add_output.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Sankey._add_output.__dict__.__setitem__('stypy_call_defaults', defaults)
        Sankey._add_output.__dict__.__setitem__('stypy_call_varargs', varargs)
        Sankey._add_output.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Sankey._add_output.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sankey._add_output', ['path', 'angle', 'flow', 'length'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_output', localization, ['path', 'angle', 'flow', 'length'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_output(...)' code ##################

        unicode_128023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, (-1)), 'unicode', u'\n        Append an output to a path and return its tip and label locations.\n\n        .. note:: *flow* is negative for an output.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 282)
        # Getting the type of 'angle' (line 282)
        angle_128024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'angle')
        # Getting the type of 'None' (line 282)
        None_128025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'None')
        
        (may_be_128026, more_types_in_union_128027) = may_be_none(angle_128024, None_128025)

        if may_be_128026:

            if more_types_in_union_128027:
                # Runtime conditional SSA (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Obtaining an instance of the builtin type 'tuple' (line 283)
            tuple_128028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 283)
            # Adding element type (line 283)
            
            # Obtaining an instance of the builtin type 'list' (line 283)
            list_128029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 283)
            # Adding element type (line 283)
            int_128030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 20), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 19), list_128029, int_128030)
            # Adding element type (line 283)
            int_128031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 23), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 19), list_128029, int_128031)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 19), tuple_128028, list_128029)
            # Adding element type (line 283)
            
            # Obtaining an instance of the builtin type 'list' (line 283)
            list_128032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 283)
            # Adding element type (line 283)
            int_128033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 27), list_128032, int_128033)
            # Adding element type (line 283)
            int_128034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 31), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 27), list_128032, int_128034)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 19), tuple_128028, list_128032)
            
            # Assigning a type to the variable 'stypy_return_type' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'stypy_return_type', tuple_128028)

            if more_types_in_union_128027:
                # Runtime conditional SSA for else branch (line 282)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_128026) or more_types_in_union_128027):
            
            # Assigning a Subscript to a Tuple (line 285):
            
            # Assigning a Subscript to a Name (line 285):
            
            # Obtaining the type of the subscript
            int_128035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 12), 'int')
            
            # Obtaining the type of the subscript
            int_128036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 28), 'int')
            
            # Obtaining the type of the subscript
            int_128037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 24), 'int')
            # Getting the type of 'path' (line 285)
            path_128038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'path')
            # Obtaining the member '__getitem__' of a type (line 285)
            getitem___128039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), path_128038, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 285)
            subscript_call_result_128040 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), getitem___128039, int_128037)
            
            # Obtaining the member '__getitem__' of a type (line 285)
            getitem___128041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), subscript_call_result_128040, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 285)
            subscript_call_result_128042 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), getitem___128041, int_128036)
            
            # Obtaining the member '__getitem__' of a type (line 285)
            getitem___128043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), subscript_call_result_128042, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 285)
            subscript_call_result_128044 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), getitem___128043, int_128035)
            
            # Assigning a type to the variable 'tuple_var_assignment_127433' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'tuple_var_assignment_127433', subscript_call_result_128044)
            
            # Assigning a Subscript to a Name (line 285):
            
            # Obtaining the type of the subscript
            int_128045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 12), 'int')
            
            # Obtaining the type of the subscript
            int_128046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 28), 'int')
            
            # Obtaining the type of the subscript
            int_128047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 24), 'int')
            # Getting the type of 'path' (line 285)
            path_128048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'path')
            # Obtaining the member '__getitem__' of a type (line 285)
            getitem___128049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), path_128048, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 285)
            subscript_call_result_128050 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), getitem___128049, int_128047)
            
            # Obtaining the member '__getitem__' of a type (line 285)
            getitem___128051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), subscript_call_result_128050, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 285)
            subscript_call_result_128052 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), getitem___128051, int_128046)
            
            # Obtaining the member '__getitem__' of a type (line 285)
            getitem___128053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), subscript_call_result_128052, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 285)
            subscript_call_result_128054 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), getitem___128053, int_128045)
            
            # Assigning a type to the variable 'tuple_var_assignment_127434' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'tuple_var_assignment_127434', subscript_call_result_128054)
            
            # Assigning a Name to a Name (line 285):
            # Getting the type of 'tuple_var_assignment_127433' (line 285)
            tuple_var_assignment_127433_128055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'tuple_var_assignment_127433')
            # Assigning a type to the variable 'x' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'x', tuple_var_assignment_127433_128055)
            
            # Assigning a Name to a Name (line 285):
            # Getting the type of 'tuple_var_assignment_127434' (line 285)
            tuple_var_assignment_127434_128056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'tuple_var_assignment_127434')
            # Assigning a type to the variable 'y' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'y', tuple_var_assignment_127434_128056)
            
            # Assigning a BinOp to a Name (line 286):
            
            # Assigning a BinOp to a Name (line 286):
            # Getting the type of 'self' (line 286)
            self_128057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 'self')
            # Obtaining the member 'shoulder' of a type (line 286)
            shoulder_128058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 25), self_128057, 'shoulder')
            # Getting the type of 'flow' (line 286)
            flow_128059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 41), 'flow')
            int_128060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 48), 'int')
            # Applying the binary operator 'div' (line 286)
            result_div_128061 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 41), 'div', flow_128059, int_128060)
            
            # Applying the binary operator '-' (line 286)
            result_sub_128062 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 25), '-', shoulder_128058, result_div_128061)
            
            # Getting the type of 'self' (line 286)
            self_128063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 53), 'self')
            # Obtaining the member 'pitch' of a type (line 286)
            pitch_128064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 53), self_128063, 'pitch')
            # Applying the binary operator '*' (line 286)
            result_mul_128065 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 24), '*', result_sub_128062, pitch_128064)
            
            # Assigning a type to the variable 'tipheight' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'tipheight', result_mul_128065)
            
            
            # Getting the type of 'angle' (line 287)
            angle_128066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'angle')
            # Getting the type of 'RIGHT' (line 287)
            RIGHT_128067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 24), 'RIGHT')
            # Applying the binary operator '==' (line 287)
            result_eq_128068 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 15), '==', angle_128066, RIGHT_128067)
            
            # Testing the type of an if condition (line 287)
            if_condition_128069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 12), result_eq_128068)
            # Assigning a type to the variable 'if_condition_128069' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'if_condition_128069', if_condition_128069)
            # SSA begins for if statement (line 287)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'x' (line 288)
            x_128070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'x')
            # Getting the type of 'length' (line 288)
            length_128071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'length')
            # Applying the binary operator '+=' (line 288)
            result_iadd_128072 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 16), '+=', x_128070, length_128071)
            # Assigning a type to the variable 'x' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'x', result_iadd_128072)
            
            
            # Assigning a List to a Name (line 289):
            
            # Assigning a List to a Name (line 289):
            
            # Obtaining an instance of the builtin type 'list' (line 289)
            list_128073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 289)
            # Adding element type (line 289)
            # Getting the type of 'x' (line 289)
            x_128074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'x')
            # Getting the type of 'tipheight' (line 289)
            tipheight_128075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 27), 'tipheight')
            # Applying the binary operator '+' (line 289)
            result_add_128076 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 23), '+', x_128074, tipheight_128075)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 22), list_128073, result_add_128076)
            # Adding element type (line 289)
            # Getting the type of 'y' (line 289)
            y_128077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'y')
            # Getting the type of 'flow' (line 289)
            flow_128078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 42), 'flow')
            float_128079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 49), 'float')
            # Applying the binary operator 'div' (line 289)
            result_div_128080 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 42), 'div', flow_128078, float_128079)
            
            # Applying the binary operator '+' (line 289)
            result_add_128081 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 38), '+', y_128077, result_div_128080)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 22), list_128073, result_add_128081)
            
            # Assigning a type to the variable 'tip' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'tip', list_128073)
            
            # Call to extend(...): (line 290)
            # Processing the call arguments (line 290)
            
            # Obtaining an instance of the builtin type 'list' (line 290)
            list_128084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 290)
            # Adding element type (line 290)
            
            # Obtaining an instance of the builtin type 'tuple' (line 290)
            tuple_128085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 290)
            # Adding element type (line 290)
            # Getting the type of 'Path' (line 290)
            Path_128086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 290)
            LINETO_128087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 30), Path_128086, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), tuple_128085, LINETO_128087)
            # Adding element type (line 290)
            
            # Obtaining an instance of the builtin type 'list' (line 290)
            list_128088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 290)
            # Adding element type (line 290)
            # Getting the type of 'x' (line 290)
            x_128089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 43), list_128088, x_128089)
            # Adding element type (line 290)
            # Getting the type of 'y' (line 290)
            y_128090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 47), 'y', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 43), list_128088, y_128090)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), tuple_128085, list_128088)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 28), list_128084, tuple_128085)
            # Adding element type (line 290)
            
            # Obtaining an instance of the builtin type 'tuple' (line 291)
            tuple_128091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 291)
            # Adding element type (line 291)
            # Getting the type of 'Path' (line 291)
            Path_128092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 291)
            LINETO_128093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 30), Path_128092, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 30), tuple_128091, LINETO_128093)
            # Adding element type (line 291)
            
            # Obtaining an instance of the builtin type 'list' (line 291)
            list_128094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 291)
            # Adding element type (line 291)
            # Getting the type of 'x' (line 291)
            x_128095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 43), list_128094, x_128095)
            # Adding element type (line 291)
            # Getting the type of 'y' (line 291)
            y_128096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 47), 'y', False)
            # Getting the type of 'self' (line 291)
            self_128097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 51), 'self', False)
            # Obtaining the member 'shoulder' of a type (line 291)
            shoulder_128098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 51), self_128097, 'shoulder')
            # Applying the binary operator '+' (line 291)
            result_add_128099 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 47), '+', y_128096, shoulder_128098)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 43), list_128094, result_add_128099)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 30), tuple_128091, list_128094)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 28), list_128084, tuple_128091)
            # Adding element type (line 290)
            
            # Obtaining an instance of the builtin type 'tuple' (line 292)
            tuple_128100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 292)
            # Adding element type (line 292)
            # Getting the type of 'Path' (line 292)
            Path_128101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 292)
            LINETO_128102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 30), Path_128101, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 30), tuple_128100, LINETO_128102)
            # Adding element type (line 292)
            # Getting the type of 'tip' (line 292)
            tip_128103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 43), 'tip', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 30), tuple_128100, tip_128103)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 28), list_128084, tuple_128100)
            # Adding element type (line 290)
            
            # Obtaining an instance of the builtin type 'tuple' (line 293)
            tuple_128104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 293)
            # Adding element type (line 293)
            # Getting the type of 'Path' (line 293)
            Path_128105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 293)
            LINETO_128106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 30), Path_128105, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 30), tuple_128104, LINETO_128106)
            # Adding element type (line 293)
            
            # Obtaining an instance of the builtin type 'list' (line 293)
            list_128107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 293)
            # Adding element type (line 293)
            # Getting the type of 'x' (line 293)
            x_128108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 43), list_128107, x_128108)
            # Adding element type (line 293)
            # Getting the type of 'y' (line 293)
            y_128109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 47), 'y', False)
            # Getting the type of 'self' (line 293)
            self_128110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 51), 'self', False)
            # Obtaining the member 'shoulder' of a type (line 293)
            shoulder_128111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 51), self_128110, 'shoulder')
            # Applying the binary operator '-' (line 293)
            result_sub_128112 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 47), '-', y_128109, shoulder_128111)
            
            # Getting the type of 'flow' (line 293)
            flow_128113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 67), 'flow', False)
            # Applying the binary operator '+' (line 293)
            result_add_128114 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 65), '+', result_sub_128112, flow_128113)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 43), list_128107, result_add_128114)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 30), tuple_128104, list_128107)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 28), list_128084, tuple_128104)
            # Adding element type (line 290)
            
            # Obtaining an instance of the builtin type 'tuple' (line 294)
            tuple_128115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 294)
            # Adding element type (line 294)
            # Getting the type of 'Path' (line 294)
            Path_128116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 294)
            LINETO_128117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 30), Path_128116, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 30), tuple_128115, LINETO_128117)
            # Adding element type (line 294)
            
            # Obtaining an instance of the builtin type 'list' (line 294)
            list_128118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 294)
            # Adding element type (line 294)
            # Getting the type of 'x' (line 294)
            x_128119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 43), list_128118, x_128119)
            # Adding element type (line 294)
            # Getting the type of 'y' (line 294)
            y_128120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 47), 'y', False)
            # Getting the type of 'flow' (line 294)
            flow_128121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 51), 'flow', False)
            # Applying the binary operator '+' (line 294)
            result_add_128122 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 47), '+', y_128120, flow_128121)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 43), list_128118, result_add_128122)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 30), tuple_128115, list_128118)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 28), list_128084, tuple_128115)
            # Adding element type (line 290)
            
            # Obtaining an instance of the builtin type 'tuple' (line 295)
            tuple_128123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 295)
            # Adding element type (line 295)
            # Getting the type of 'Path' (line 295)
            Path_128124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 295)
            LINETO_128125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 30), Path_128124, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 30), tuple_128123, LINETO_128125)
            # Adding element type (line 295)
            
            # Obtaining an instance of the builtin type 'list' (line 295)
            list_128126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 295)
            # Adding element type (line 295)
            # Getting the type of 'x' (line 295)
            x_128127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 44), 'x', False)
            # Getting the type of 'self' (line 295)
            self_128128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 48), 'self', False)
            # Obtaining the member 'gap' of a type (line 295)
            gap_128129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 48), self_128128, 'gap')
            # Applying the binary operator '-' (line 295)
            result_sub_128130 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 44), '-', x_128127, gap_128129)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 43), list_128126, result_sub_128130)
            # Adding element type (line 295)
            # Getting the type of 'y' (line 295)
            y_128131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 58), 'y', False)
            # Getting the type of 'flow' (line 295)
            flow_128132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 62), 'flow', False)
            # Applying the binary operator '+' (line 295)
            result_add_128133 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 58), '+', y_128131, flow_128132)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 43), list_128126, result_add_128133)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 30), tuple_128123, list_128126)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 28), list_128084, tuple_128123)
            
            # Processing the call keyword arguments (line 290)
            kwargs_128134 = {}
            # Getting the type of 'path' (line 290)
            path_128082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'path', False)
            # Obtaining the member 'extend' of a type (line 290)
            extend_128083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 16), path_128082, 'extend')
            # Calling extend(args, kwargs) (line 290)
            extend_call_result_128135 = invoke(stypy.reporting.localization.Localization(__file__, 290, 16), extend_128083, *[list_128084], **kwargs_128134)
            
            
            # Assigning a List to a Name (line 296):
            
            # Assigning a List to a Name (line 296):
            
            # Obtaining an instance of the builtin type 'list' (line 296)
            list_128136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 296)
            # Adding element type (line 296)
            
            # Obtaining the type of the subscript
            int_128137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 38), 'int')
            # Getting the type of 'tip' (line 296)
            tip_128138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'tip')
            # Obtaining the member '__getitem__' of a type (line 296)
            getitem___128139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 34), tip_128138, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 296)
            subscript_call_result_128140 = invoke(stypy.reporting.localization.Localization(__file__, 296, 34), getitem___128139, int_128137)
            
            # Getting the type of 'self' (line 296)
            self_128141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 43), 'self')
            # Obtaining the member 'offset' of a type (line 296)
            offset_128142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 43), self_128141, 'offset')
            # Applying the binary operator '+' (line 296)
            result_add_128143 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 34), '+', subscript_call_result_128140, offset_128142)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 33), list_128136, result_add_128143)
            # Adding element type (line 296)
            
            # Obtaining the type of the subscript
            int_128144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 60), 'int')
            # Getting the type of 'tip' (line 296)
            tip_128145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 56), 'tip')
            # Obtaining the member '__getitem__' of a type (line 296)
            getitem___128146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 56), tip_128145, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 296)
            subscript_call_result_128147 = invoke(stypy.reporting.localization.Localization(__file__, 296, 56), getitem___128146, int_128144)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 33), list_128136, subscript_call_result_128147)
            
            # Assigning a type to the variable 'label_location' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'label_location', list_128136)
            # SSA branch for the else part of an if statement (line 287)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'x' (line 298)
            x_128148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'x')
            # Getting the type of 'self' (line 298)
            self_128149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'self')
            # Obtaining the member 'gap' of a type (line 298)
            gap_128150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 21), self_128149, 'gap')
            # Applying the binary operator '+=' (line 298)
            result_iadd_128151 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 16), '+=', x_128148, gap_128150)
            # Assigning a type to the variable 'x' (line 298)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'x', result_iadd_128151)
            
            
            
            # Getting the type of 'angle' (line 299)
            angle_128152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 19), 'angle')
            # Getting the type of 'UP' (line 299)
            UP_128153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 'UP')
            # Applying the binary operator '==' (line 299)
            result_eq_128154 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 19), '==', angle_128152, UP_128153)
            
            # Testing the type of an if condition (line 299)
            if_condition_128155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 16), result_eq_128154)
            # Assigning a type to the variable 'if_condition_128155' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'if_condition_128155', if_condition_128155)
            # SSA begins for if statement (line 299)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 300):
            
            # Assigning a Num to a Name (line 300):
            int_128156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 27), 'int')
            # Assigning a type to the variable 'sign' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'sign', int_128156)
            # SSA branch for the else part of an if statement (line 299)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 302):
            
            # Assigning a Num to a Name (line 302):
            int_128157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 27), 'int')
            # Assigning a type to the variable 'sign' (line 302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'sign', int_128157)
            # SSA join for if statement (line 299)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a List to a Name (line 304):
            
            # Assigning a List to a Name (line 304):
            
            # Obtaining an instance of the builtin type 'list' (line 304)
            list_128158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 304)
            # Adding element type (line 304)
            # Getting the type of 'x' (line 304)
            x_128159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'x')
            # Getting the type of 'flow' (line 304)
            flow_128160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'flow')
            float_128161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 34), 'float')
            # Applying the binary operator 'div' (line 304)
            result_div_128162 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 27), 'div', flow_128160, float_128161)
            
            # Applying the binary operator '-' (line 304)
            result_sub_128163 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 23), '-', x_128159, result_div_128162)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 22), list_128158, result_sub_128163)
            # Adding element type (line 304)
            # Getting the type of 'y' (line 304)
            y_128164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 39), 'y')
            # Getting the type of 'sign' (line 304)
            sign_128165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'sign')
            # Getting the type of 'length' (line 304)
            length_128166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 51), 'length')
            # Getting the type of 'tipheight' (line 304)
            tipheight_128167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 60), 'tipheight')
            # Applying the binary operator '+' (line 304)
            result_add_128168 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 51), '+', length_128166, tipheight_128167)
            
            # Applying the binary operator '*' (line 304)
            result_mul_128169 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 43), '*', sign_128165, result_add_128168)
            
            # Applying the binary operator '+' (line 304)
            result_add_128170 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 39), '+', y_128164, result_mul_128169)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 22), list_128158, result_add_128170)
            
            # Assigning a type to the variable 'tip' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'tip', list_128158)
            
            
            # Getting the type of 'angle' (line 305)
            angle_128171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'angle')
            # Getting the type of 'UP' (line 305)
            UP_128172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 28), 'UP')
            # Applying the binary operator '==' (line 305)
            result_eq_128173 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 19), '==', angle_128171, UP_128172)
            
            # Testing the type of an if condition (line 305)
            if_condition_128174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 16), result_eq_128173)
            # Assigning a type to the variable 'if_condition_128174' (line 305)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'if_condition_128174', if_condition_128174)
            # SSA begins for if statement (line 305)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 306):
            
            # Assigning a Num to a Name (line 306):
            int_128175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 31), 'int')
            # Assigning a type to the variable 'quadrant' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'quadrant', int_128175)
            # SSA branch for the else part of an if statement (line 305)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 308):
            
            # Assigning a Num to a Name (line 308):
            int_128176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 31), 'int')
            # Assigning a type to the variable 'quadrant' (line 308)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'quadrant', int_128176)
            # SSA join for if statement (line 305)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'self' (line 310)
            self_128177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'self')
            # Obtaining the member 'radius' of a type (line 310)
            radius_128178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 19), self_128177, 'radius')
            # Testing the type of an if condition (line 310)
            if_condition_128179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 16), radius_128178)
            # Assigning a type to the variable 'if_condition_128179' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'if_condition_128179', if_condition_128179)
            # SSA begins for if statement (line 310)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to extend(...): (line 311)
            # Processing the call arguments (line 311)
            
            # Call to _arc(...): (line 311)
            # Processing the call keyword arguments (line 311)
            # Getting the type of 'quadrant' (line 311)
            quadrant_128184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 51), 'quadrant', False)
            keyword_128185 = quadrant_128184
            
            # Getting the type of 'angle' (line 312)
            angle_128186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 45), 'angle', False)
            # Getting the type of 'UP' (line 312)
            UP_128187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 54), 'UP', False)
            # Applying the binary operator '==' (line 312)
            result_eq_128188 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 45), '==', angle_128186, UP_128187)
            
            keyword_128189 = result_eq_128188
            # Getting the type of 'self' (line 313)
            self_128190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 49), 'self', False)
            # Obtaining the member 'radius' of a type (line 313)
            radius_128191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 49), self_128190, 'radius')
            keyword_128192 = radius_128191
            
            # Obtaining an instance of the builtin type 'tuple' (line 314)
            tuple_128193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 50), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 314)
            # Adding element type (line 314)
            # Getting the type of 'x' (line 314)
            x_128194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 50), 'x', False)
            # Getting the type of 'self' (line 314)
            self_128195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 54), 'self', False)
            # Obtaining the member 'radius' of a type (line 314)
            radius_128196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 54), self_128195, 'radius')
            # Applying the binary operator '-' (line 314)
            result_sub_128197 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 50), '-', x_128194, radius_128196)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 50), tuple_128193, result_sub_128197)
            # Adding element type (line 314)
            # Getting the type of 'y' (line 315)
            y_128198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 50), 'y', False)
            # Getting the type of 'sign' (line 315)
            sign_128199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 54), 'sign', False)
            # Getting the type of 'self' (line 315)
            self_128200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 61), 'self', False)
            # Obtaining the member 'radius' of a type (line 315)
            radius_128201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 61), self_128200, 'radius')
            # Applying the binary operator '*' (line 315)
            result_mul_128202 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 54), '*', sign_128199, radius_128201)
            
            # Applying the binary operator '+' (line 315)
            result_add_128203 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 50), '+', y_128198, result_mul_128202)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 50), tuple_128193, result_add_128203)
            
            keyword_128204 = tuple_128193
            kwargs_128205 = {'quadrant': keyword_128185, 'radius': keyword_128192, 'cw': keyword_128189, 'center': keyword_128204}
            # Getting the type of 'self' (line 311)
            self_128182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 32), 'self', False)
            # Obtaining the member '_arc' of a type (line 311)
            _arc_128183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 32), self_128182, '_arc')
            # Calling _arc(args, kwargs) (line 311)
            _arc_call_result_128206 = invoke(stypy.reporting.localization.Localization(__file__, 311, 32), _arc_128183, *[], **kwargs_128205)
            
            # Processing the call keyword arguments (line 311)
            kwargs_128207 = {}
            # Getting the type of 'path' (line 311)
            path_128180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 20), 'path', False)
            # Obtaining the member 'extend' of a type (line 311)
            extend_128181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 20), path_128180, 'extend')
            # Calling extend(args, kwargs) (line 311)
            extend_call_result_128208 = invoke(stypy.reporting.localization.Localization(__file__, 311, 20), extend_128181, *[_arc_call_result_128206], **kwargs_128207)
            
            # SSA branch for the else part of an if statement (line 310)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 317)
            # Processing the call arguments (line 317)
            
            # Obtaining an instance of the builtin type 'tuple' (line 317)
            tuple_128211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 317)
            # Adding element type (line 317)
            # Getting the type of 'Path' (line 317)
            Path_128212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 317)
            LINETO_128213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 33), Path_128212, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 33), tuple_128211, LINETO_128213)
            # Adding element type (line 317)
            
            # Obtaining an instance of the builtin type 'list' (line 317)
            list_128214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 46), 'list')
            # Adding type elements to the builtin type 'list' instance (line 317)
            # Adding element type (line 317)
            # Getting the type of 'x' (line 317)
            x_128215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 47), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 46), list_128214, x_128215)
            # Adding element type (line 317)
            # Getting the type of 'y' (line 317)
            y_128216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 50), 'y', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 46), list_128214, y_128216)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 33), tuple_128211, list_128214)
            
            # Processing the call keyword arguments (line 317)
            kwargs_128217 = {}
            # Getting the type of 'path' (line 317)
            path_128209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'path', False)
            # Obtaining the member 'append' of a type (line 317)
            append_128210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 20), path_128209, 'append')
            # Calling append(args, kwargs) (line 317)
            append_call_result_128218 = invoke(stypy.reporting.localization.Localization(__file__, 317, 20), append_128210, *[tuple_128211], **kwargs_128217)
            
            # SSA join for if statement (line 310)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to extend(...): (line 318)
            # Processing the call arguments (line 318)
            
            # Obtaining an instance of the builtin type 'list' (line 318)
            list_128221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 318)
            # Adding element type (line 318)
            
            # Obtaining an instance of the builtin type 'tuple' (line 318)
            tuple_128222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 318)
            # Adding element type (line 318)
            # Getting the type of 'Path' (line 318)
            Path_128223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 318)
            LINETO_128224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 30), Path_128223, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 30), tuple_128222, LINETO_128224)
            # Adding element type (line 318)
            
            # Obtaining an instance of the builtin type 'list' (line 318)
            list_128225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 318)
            # Adding element type (line 318)
            # Getting the type of 'x' (line 318)
            x_128226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 43), list_128225, x_128226)
            # Adding element type (line 318)
            # Getting the type of 'y' (line 318)
            y_128227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 47), 'y', False)
            # Getting the type of 'sign' (line 318)
            sign_128228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 51), 'sign', False)
            # Getting the type of 'length' (line 318)
            length_128229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 58), 'length', False)
            # Applying the binary operator '*' (line 318)
            result_mul_128230 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 51), '*', sign_128228, length_128229)
            
            # Applying the binary operator '+' (line 318)
            result_add_128231 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 47), '+', y_128227, result_mul_128230)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 43), list_128225, result_add_128231)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 30), tuple_128222, list_128225)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 28), list_128221, tuple_128222)
            # Adding element type (line 318)
            
            # Obtaining an instance of the builtin type 'tuple' (line 319)
            tuple_128232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 319)
            # Adding element type (line 319)
            # Getting the type of 'Path' (line 319)
            Path_128233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 319)
            LINETO_128234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 30), Path_128233, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 30), tuple_128232, LINETO_128234)
            # Adding element type (line 319)
            
            # Obtaining an instance of the builtin type 'list' (line 319)
            list_128235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 319)
            # Adding element type (line 319)
            # Getting the type of 'x' (line 319)
            x_128236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 44), 'x', False)
            # Getting the type of 'self' (line 319)
            self_128237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 48), 'self', False)
            # Obtaining the member 'shoulder' of a type (line 319)
            shoulder_128238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 48), self_128237, 'shoulder')
            # Applying the binary operator '-' (line 319)
            result_sub_128239 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 44), '-', x_128236, shoulder_128238)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 43), list_128235, result_sub_128239)
            # Adding element type (line 319)
            # Getting the type of 'y' (line 320)
            y_128240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 44), 'y', False)
            # Getting the type of 'sign' (line 320)
            sign_128241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 48), 'sign', False)
            # Getting the type of 'length' (line 320)
            length_128242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 55), 'length', False)
            # Applying the binary operator '*' (line 320)
            result_mul_128243 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 48), '*', sign_128241, length_128242)
            
            # Applying the binary operator '+' (line 320)
            result_add_128244 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 44), '+', y_128240, result_mul_128243)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 43), list_128235, result_add_128244)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 30), tuple_128232, list_128235)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 28), list_128221, tuple_128232)
            # Adding element type (line 318)
            
            # Obtaining an instance of the builtin type 'tuple' (line 321)
            tuple_128245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 321)
            # Adding element type (line 321)
            # Getting the type of 'Path' (line 321)
            Path_128246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 321)
            LINETO_128247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 30), Path_128246, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 30), tuple_128245, LINETO_128247)
            # Adding element type (line 321)
            # Getting the type of 'tip' (line 321)
            tip_128248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 43), 'tip', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 30), tuple_128245, tip_128248)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 28), list_128221, tuple_128245)
            # Adding element type (line 318)
            
            # Obtaining an instance of the builtin type 'tuple' (line 322)
            tuple_128249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 322)
            # Adding element type (line 322)
            # Getting the type of 'Path' (line 322)
            Path_128250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 322)
            LINETO_128251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 30), Path_128250, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 30), tuple_128249, LINETO_128251)
            # Adding element type (line 322)
            
            # Obtaining an instance of the builtin type 'list' (line 322)
            list_128252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 322)
            # Adding element type (line 322)
            # Getting the type of 'x' (line 322)
            x_128253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 44), 'x', False)
            # Getting the type of 'self' (line 322)
            self_128254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 48), 'self', False)
            # Obtaining the member 'shoulder' of a type (line 322)
            shoulder_128255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 48), self_128254, 'shoulder')
            # Applying the binary operator '+' (line 322)
            result_add_128256 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 44), '+', x_128253, shoulder_128255)
            
            # Getting the type of 'flow' (line 322)
            flow_128257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 64), 'flow', False)
            # Applying the binary operator '-' (line 322)
            result_sub_128258 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 62), '-', result_add_128256, flow_128257)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 43), list_128252, result_sub_128258)
            # Adding element type (line 322)
            # Getting the type of 'y' (line 323)
            y_128259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 44), 'y', False)
            # Getting the type of 'sign' (line 323)
            sign_128260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 48), 'sign', False)
            # Getting the type of 'length' (line 323)
            length_128261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 55), 'length', False)
            # Applying the binary operator '*' (line 323)
            result_mul_128262 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 48), '*', sign_128260, length_128261)
            
            # Applying the binary operator '+' (line 323)
            result_add_128263 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 44), '+', y_128259, result_mul_128262)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 43), list_128252, result_add_128263)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 30), tuple_128249, list_128252)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 28), list_128221, tuple_128249)
            # Adding element type (line 318)
            
            # Obtaining an instance of the builtin type 'tuple' (line 324)
            tuple_128264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 324)
            # Adding element type (line 324)
            # Getting the type of 'Path' (line 324)
            Path_128265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 324)
            LINETO_128266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 30), Path_128265, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 30), tuple_128264, LINETO_128266)
            # Adding element type (line 324)
            
            # Obtaining an instance of the builtin type 'list' (line 324)
            list_128267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 324)
            # Adding element type (line 324)
            # Getting the type of 'x' (line 324)
            x_128268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 44), 'x', False)
            # Getting the type of 'flow' (line 324)
            flow_128269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 48), 'flow', False)
            # Applying the binary operator '-' (line 324)
            result_sub_128270 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 44), '-', x_128268, flow_128269)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 43), list_128267, result_sub_128270)
            # Adding element type (line 324)
            # Getting the type of 'y' (line 324)
            y_128271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 54), 'y', False)
            # Getting the type of 'sign' (line 324)
            sign_128272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 58), 'sign', False)
            # Getting the type of 'length' (line 324)
            length_128273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 65), 'length', False)
            # Applying the binary operator '*' (line 324)
            result_mul_128274 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 58), '*', sign_128272, length_128273)
            
            # Applying the binary operator '+' (line 324)
            result_add_128275 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 54), '+', y_128271, result_mul_128274)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 43), list_128267, result_add_128275)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 30), tuple_128264, list_128267)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 28), list_128221, tuple_128264)
            
            # Processing the call keyword arguments (line 318)
            kwargs_128276 = {}
            # Getting the type of 'path' (line 318)
            path_128219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'path', False)
            # Obtaining the member 'extend' of a type (line 318)
            extend_128220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 16), path_128219, 'extend')
            # Calling extend(args, kwargs) (line 318)
            extend_call_result_128277 = invoke(stypy.reporting.localization.Localization(__file__, 318, 16), extend_128220, *[list_128221], **kwargs_128276)
            
            
            # Call to extend(...): (line 325)
            # Processing the call arguments (line 325)
            
            # Call to _arc(...): (line 325)
            # Processing the call keyword arguments (line 325)
            # Getting the type of 'quadrant' (line 325)
            quadrant_128282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 47), 'quadrant', False)
            keyword_128283 = quadrant_128282
            
            # Getting the type of 'angle' (line 326)
            angle_128284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 41), 'angle', False)
            # Getting the type of 'DOWN' (line 326)
            DOWN_128285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 50), 'DOWN', False)
            # Applying the binary operator '==' (line 326)
            result_eq_128286 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 41), '==', angle_128284, DOWN_128285)
            
            keyword_128287 = result_eq_128286
            # Getting the type of 'self' (line 327)
            self_128288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 45), 'self', False)
            # Obtaining the member 'radius' of a type (line 327)
            radius_128289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 45), self_128288, 'radius')
            # Getting the type of 'flow' (line 327)
            flow_128290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 59), 'flow', False)
            # Applying the binary operator '-' (line 327)
            result_sub_128291 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 45), '-', radius_128289, flow_128290)
            
            keyword_128292 = result_sub_128291
            
            # Obtaining an instance of the builtin type 'tuple' (line 328)
            tuple_128293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 328)
            # Adding element type (line 328)
            # Getting the type of 'x' (line 328)
            x_128294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 46), 'x', False)
            # Getting the type of 'self' (line 328)
            self_128295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 50), 'self', False)
            # Obtaining the member 'radius' of a type (line 328)
            radius_128296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 50), self_128295, 'radius')
            # Applying the binary operator '-' (line 328)
            result_sub_128297 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 46), '-', x_128294, radius_128296)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 46), tuple_128293, result_sub_128297)
            # Adding element type (line 328)
            # Getting the type of 'y' (line 329)
            y_128298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 46), 'y', False)
            # Getting the type of 'sign' (line 329)
            sign_128299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 50), 'sign', False)
            # Getting the type of 'self' (line 329)
            self_128300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 57), 'self', False)
            # Obtaining the member 'radius' of a type (line 329)
            radius_128301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 57), self_128300, 'radius')
            # Applying the binary operator '*' (line 329)
            result_mul_128302 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 50), '*', sign_128299, radius_128301)
            
            # Applying the binary operator '+' (line 329)
            result_add_128303 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 46), '+', y_128298, result_mul_128302)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 46), tuple_128293, result_add_128303)
            
            keyword_128304 = tuple_128293
            kwargs_128305 = {'quadrant': keyword_128283, 'radius': keyword_128292, 'cw': keyword_128287, 'center': keyword_128304}
            # Getting the type of 'self' (line 325)
            self_128280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 28), 'self', False)
            # Obtaining the member '_arc' of a type (line 325)
            _arc_128281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 28), self_128280, '_arc')
            # Calling _arc(args, kwargs) (line 325)
            _arc_call_result_128306 = invoke(stypy.reporting.localization.Localization(__file__, 325, 28), _arc_128281, *[], **kwargs_128305)
            
            # Processing the call keyword arguments (line 325)
            kwargs_128307 = {}
            # Getting the type of 'path' (line 325)
            path_128278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'path', False)
            # Obtaining the member 'extend' of a type (line 325)
            extend_128279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 16), path_128278, 'extend')
            # Calling extend(args, kwargs) (line 325)
            extend_call_result_128308 = invoke(stypy.reporting.localization.Localization(__file__, 325, 16), extend_128279, *[_arc_call_result_128306], **kwargs_128307)
            
            
            # Call to append(...): (line 330)
            # Processing the call arguments (line 330)
            
            # Obtaining an instance of the builtin type 'tuple' (line 330)
            tuple_128311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 330)
            # Adding element type (line 330)
            # Getting the type of 'Path' (line 330)
            Path_128312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 29), 'Path', False)
            # Obtaining the member 'LINETO' of a type (line 330)
            LINETO_128313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 29), Path_128312, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 29), tuple_128311, LINETO_128313)
            # Adding element type (line 330)
            
            # Obtaining an instance of the builtin type 'list' (line 330)
            list_128314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 42), 'list')
            # Adding type elements to the builtin type 'list' instance (line 330)
            # Adding element type (line 330)
            # Getting the type of 'x' (line 330)
            x_128315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 43), 'x', False)
            # Getting the type of 'flow' (line 330)
            flow_128316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 47), 'flow', False)
            # Applying the binary operator '-' (line 330)
            result_sub_128317 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 43), '-', x_128315, flow_128316)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 42), list_128314, result_sub_128317)
            # Adding element type (line 330)
            # Getting the type of 'y' (line 330)
            y_128318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 53), 'y', False)
            # Getting the type of 'sign' (line 330)
            sign_128319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 57), 'sign', False)
            # Getting the type of 'flow' (line 330)
            flow_128320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 64), 'flow', False)
            # Applying the binary operator '*' (line 330)
            result_mul_128321 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 57), '*', sign_128319, flow_128320)
            
            # Applying the binary operator '+' (line 330)
            result_add_128322 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 53), '+', y_128318, result_mul_128321)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 42), list_128314, result_add_128322)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 29), tuple_128311, list_128314)
            
            # Processing the call keyword arguments (line 330)
            kwargs_128323 = {}
            # Getting the type of 'path' (line 330)
            path_128309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'path', False)
            # Obtaining the member 'append' of a type (line 330)
            append_128310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 16), path_128309, 'append')
            # Calling append(args, kwargs) (line 330)
            append_call_result_128324 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), append_128310, *[tuple_128311], **kwargs_128323)
            
            
            # Assigning a List to a Name (line 331):
            
            # Assigning a List to a Name (line 331):
            
            # Obtaining an instance of the builtin type 'list' (line 331)
            list_128325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 331)
            # Adding element type (line 331)
            
            # Obtaining the type of the subscript
            int_128326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 38), 'int')
            # Getting the type of 'tip' (line 331)
            tip_128327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 34), 'tip')
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___128328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 34), tip_128327, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_128329 = invoke(stypy.reporting.localization.Localization(__file__, 331, 34), getitem___128328, int_128326)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 33), list_128325, subscript_call_result_128329)
            # Adding element type (line 331)
            
            # Obtaining the type of the subscript
            int_128330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 46), 'int')
            # Getting the type of 'tip' (line 331)
            tip_128331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 42), 'tip')
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___128332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 42), tip_128331, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_128333 = invoke(stypy.reporting.localization.Localization(__file__, 331, 42), getitem___128332, int_128330)
            
            # Getting the type of 'sign' (line 331)
            sign_128334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 51), 'sign')
            # Getting the type of 'self' (line 331)
            self_128335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 58), 'self')
            # Obtaining the member 'offset' of a type (line 331)
            offset_128336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 58), self_128335, 'offset')
            # Applying the binary operator '*' (line 331)
            result_mul_128337 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 51), '*', sign_128334, offset_128336)
            
            # Applying the binary operator '+' (line 331)
            result_add_128338 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 42), '+', subscript_call_result_128333, result_mul_128337)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 33), list_128325, result_add_128338)
            
            # Assigning a type to the variable 'label_location' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'label_location', list_128325)
            # SSA join for if statement (line 287)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 332)
            tuple_128339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 332)
            # Adding element type (line 332)
            # Getting the type of 'tip' (line 332)
            tip_128340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'tip')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 19), tuple_128339, tip_128340)
            # Adding element type (line 332)
            # Getting the type of 'label_location' (line 332)
            label_location_128341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 24), 'label_location')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 19), tuple_128339, label_location_128341)
            
            # Assigning a type to the variable 'stypy_return_type' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', tuple_128339)

            if (may_be_128026 and more_types_in_union_128027):
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_add_output(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_output' in the type store
        # Getting the type of 'stypy_return_type' (line 276)
        stypy_return_type_128342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_output'
        return stypy_return_type_128342


    @norecursion
    def _revert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'Path' (line 334)
        Path_128343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 41), 'Path')
        # Obtaining the member 'LINETO' of a type (line 334)
        LINETO_128344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 41), Path_128343, 'LINETO')
        defaults = [LINETO_128344]
        # Create a new context for function '_revert'
        module_type_store = module_type_store.open_function_context('_revert', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Sankey._revert.__dict__.__setitem__('stypy_localization', localization)
        Sankey._revert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Sankey._revert.__dict__.__setitem__('stypy_type_store', module_type_store)
        Sankey._revert.__dict__.__setitem__('stypy_function_name', 'Sankey._revert')
        Sankey._revert.__dict__.__setitem__('stypy_param_names_list', ['path', 'first_action'])
        Sankey._revert.__dict__.__setitem__('stypy_varargs_param_name', None)
        Sankey._revert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Sankey._revert.__dict__.__setitem__('stypy_call_defaults', defaults)
        Sankey._revert.__dict__.__setitem__('stypy_call_varargs', varargs)
        Sankey._revert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Sankey._revert.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sankey._revert', ['path', 'first_action'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_revert', localization, ['path', 'first_action'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_revert(...)' code ##################

        unicode_128345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, (-1)), 'unicode', u'\n        A path is not simply revertable by path[::-1] since the code\n        specifies an action to take from the **previous** point.\n        ')
        
        # Assigning a List to a Name (line 339):
        
        # Assigning a List to a Name (line 339):
        
        # Obtaining an instance of the builtin type 'list' (line 339)
        list_128346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 339)
        
        # Assigning a type to the variable 'reverse_path' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'reverse_path', list_128346)
        
        # Assigning a Name to a Name (line 340):
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'first_action' (line 340)
        first_action_128347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 20), 'first_action')
        # Assigning a type to the variable 'next_code' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'next_code', first_action_128347)
        
        
        # Obtaining the type of the subscript
        int_128348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 37), 'int')
        slice_128349 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 30), None, None, int_128348)
        # Getting the type of 'path' (line 341)
        path_128350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'path')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___128351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 30), path_128350, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_128352 = invoke(stypy.reporting.localization.Localization(__file__, 341, 30), getitem___128351, slice_128349)
        
        # Testing the type of a for loop iterable (line 341)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 341, 8), subscript_call_result_128352)
        # Getting the type of the for loop variable (line 341)
        for_loop_var_128353 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 341, 8), subscript_call_result_128352)
        # Assigning a type to the variable 'code' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'code', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 8), for_loop_var_128353))
        # Assigning a type to the variable 'position' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'position', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 8), for_loop_var_128353))
        # SSA begins for a for statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 342)
        # Processing the call arguments (line 342)
        
        # Obtaining an instance of the builtin type 'tuple' (line 342)
        tuple_128356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 342)
        # Adding element type (line 342)
        # Getting the type of 'next_code' (line 342)
        next_code_128357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 33), 'next_code', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 33), tuple_128356, next_code_128357)
        # Adding element type (line 342)
        # Getting the type of 'position' (line 342)
        position_128358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 44), 'position', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 33), tuple_128356, position_128358)
        
        # Processing the call keyword arguments (line 342)
        kwargs_128359 = {}
        # Getting the type of 'reverse_path' (line 342)
        reverse_path_128354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'reverse_path', False)
        # Obtaining the member 'append' of a type (line 342)
        append_128355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), reverse_path_128354, 'append')
        # Calling append(args, kwargs) (line 342)
        append_call_result_128360 = invoke(stypy.reporting.localization.Localization(__file__, 342, 12), append_128355, *[tuple_128356], **kwargs_128359)
        
        
        # Assigning a Name to a Name (line 343):
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'code' (line 343)
        code_128361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'code')
        # Assigning a type to the variable 'next_code' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'next_code', code_128361)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'reverse_path' (line 344)
        reverse_path_128362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 15), 'reverse_path')
        # Assigning a type to the variable 'stypy_return_type' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'stypy_return_type', reverse_path_128362)
        
        # ################# End of '_revert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_revert' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_128363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128363)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_revert'
        return stypy_return_type_128363


    @norecursion
    def add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_128364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 29), 'unicode', u'')
        # Getting the type of 'None' (line 353)
        None_128365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 39), 'None')
        # Getting the type of 'None' (line 353)
        None_128366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 58), 'None')
        unicode_128367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 71), 'unicode', u'')
        float_128368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 24), 'float')
        float_128369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 41), 'float')
        # Getting the type of 'None' (line 354)
        None_128370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 53), 'None')
        
        # Obtaining an instance of the builtin type 'tuple' (line 354)
        tuple_128371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 68), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 354)
        # Adding element type (line 354)
        int_128372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 68), tuple_128371, int_128372)
        # Adding element type (line 354)
        int_128373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 71), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 68), tuple_128371, int_128373)
        
        int_128374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 21), 'int')
        defaults = [unicode_128364, None_128365, None_128366, unicode_128367, float_128368, float_128369, None_128370, tuple_128371, int_128374]
        # Create a new context for function 'add'
        module_type_store = module_type_store.open_function_context('add', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Sankey.add.__dict__.__setitem__('stypy_localization', localization)
        Sankey.add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Sankey.add.__dict__.__setitem__('stypy_type_store', module_type_store)
        Sankey.add.__dict__.__setitem__('stypy_function_name', 'Sankey.add')
        Sankey.add.__dict__.__setitem__('stypy_param_names_list', ['patchlabel', 'flows', 'orientations', 'labels', 'trunklength', 'pathlengths', 'prior', 'connect', 'rotation'])
        Sankey.add.__dict__.__setitem__('stypy_varargs_param_name', None)
        Sankey.add.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Sankey.add.__dict__.__setitem__('stypy_call_defaults', defaults)
        Sankey.add.__dict__.__setitem__('stypy_call_varargs', varargs)
        Sankey.add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Sankey.add.__dict__.__setitem__('stypy_declared_arg_number', 10)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sankey.add', ['patchlabel', 'flows', 'orientations', 'labels', 'trunklength', 'pathlengths', 'prior', 'connect', 'rotation'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add', localization, ['patchlabel', 'flows', 'orientations', 'labels', 'trunklength', 'pathlengths', 'prior', 'connect', 'rotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add(...)' code ##################

        unicode_128375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, (-1)), 'unicode', u"\n        Add a simple Sankey diagram with flows at the same hierarchical level.\n\n        Return value is the instance of :class:`Sankey`.\n\n        Optional keyword arguments:\n\n          ===============   ===================================================\n          Keyword           Description\n          ===============   ===================================================\n          *patchlabel*      label to be placed at the center of the diagram\n                            Note: *label* (not *patchlabel*) will be passed to\n                            the patch through ``**kwargs`` and can be used to\n                            create an entry in the legend.\n          *flows*           array of flow values\n                            By convention, inputs are positive and outputs are\n                            negative.\n          *orientations*    list of orientations of the paths\n                            Valid values are 1 (from/to the top), 0 (from/to\n                            the left or right), or -1 (from/to the bottom).  If\n                            *orientations* == 0, inputs will break in from the\n                            left and outputs will break away to the right.\n          *labels*          list of specifications of the labels for the flows\n                            Each value may be *None* (no labels), '' (just\n                            label the quantities), or a labeling string.  If a\n                            single value is provided, it will be applied to all\n                            flows.  If an entry is a non-empty string, then the\n                            quantity for the corresponding flow will be shown\n                            below the string.  However, if the *unit* of the\n                            main diagram is None, then quantities are never\n                            shown, regardless of the value of this argument.\n          *trunklength*     length between the bases of the input and output\n                            groups\n          *pathlengths*     list of lengths of the arrows before break-in or\n                            after break-away\n                            If a single value is given, then it will be applied\n                            to the first (inside) paths on the top and bottom,\n                            and the length of all other arrows will be\n                            justified accordingly.  The *pathlengths* are not\n                            applied to the horizontal inputs and outputs.\n          *prior*           index of the prior diagram to which this diagram\n                            should be connected\n          *connect*         a (prior, this) tuple indexing the flow of the\n                            prior diagram and the flow of this diagram which\n                            should be connected\n                            If this is the first diagram or *prior* is *None*,\n                            *connect* will be ignored.\n          *rotation*        angle of rotation of the diagram [deg]\n                            *rotation* is ignored if this diagram is connected\n                            to an existing one (using *prior* and *connect*).\n                            The interpretation of the *orientations* argument\n                            will be rotated accordingly (e.g., if *rotation*\n                            == 90, an *orientations* entry of 1 means to/from\n                            the left).\n          ===============   ===================================================\n\n        Valid kwargs are :meth:`matplotlib.patches.PathPatch` arguments:\n\n        %(Patch)s\n\n        As examples, ``fill=False`` and ``label='A legend entry'``.\n        By default, ``facecolor='#bfd1d4'`` (light blue) and\n        ``linewidth=0.5``.\n\n        The indexing parameters (*prior* and *connect*) are zero-based.\n\n        The flows are placed along the top of the diagram from the inside out\n        in order of their index within the *flows* list or array.  They are\n        placed along the sides of the diagram from the top down and along the\n        bottom from the outside in.\n\n        If the sum of the inputs and outputs is nonzero, the discrepancy\n        will appear as a cubic Bezier curve along the top and bottom edges of\n        the trunk.\n\n        .. seealso::\n\n            :meth:`finish`\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 436)
        # Getting the type of 'flows' (line 436)
        flows_128376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'flows')
        # Getting the type of 'None' (line 436)
        None_128377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'None')
        
        (may_be_128378, more_types_in_union_128379) = may_be_none(flows_128376, None_128377)

        if may_be_128378:

            if more_types_in_union_128379:
                # Runtime conditional SSA (line 436)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 437):
            
            # Assigning a Call to a Name (line 437):
            
            # Call to array(...): (line 437)
            # Processing the call arguments (line 437)
            
            # Obtaining an instance of the builtin type 'list' (line 437)
            list_128382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 437)
            # Adding element type (line 437)
            float_128383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 30), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 29), list_128382, float_128383)
            # Adding element type (line 437)
            float_128384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 35), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 29), list_128382, float_128384)
            
            # Processing the call keyword arguments (line 437)
            kwargs_128385 = {}
            # Getting the type of 'np' (line 437)
            np_128380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'np', False)
            # Obtaining the member 'array' of a type (line 437)
            array_128381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 20), np_128380, 'array')
            # Calling array(args, kwargs) (line 437)
            array_call_result_128386 = invoke(stypy.reporting.localization.Localization(__file__, 437, 20), array_128381, *[list_128382], **kwargs_128385)
            
            # Assigning a type to the variable 'flows' (line 437)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'flows', array_call_result_128386)

            if more_types_in_union_128379:
                # Runtime conditional SSA for else branch (line 436)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_128378) or more_types_in_union_128379):
            
            # Assigning a Call to a Name (line 439):
            
            # Assigning a Call to a Name (line 439):
            
            # Call to array(...): (line 439)
            # Processing the call arguments (line 439)
            # Getting the type of 'flows' (line 439)
            flows_128389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 29), 'flows', False)
            # Processing the call keyword arguments (line 439)
            kwargs_128390 = {}
            # Getting the type of 'np' (line 439)
            np_128387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'np', False)
            # Obtaining the member 'array' of a type (line 439)
            array_128388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 20), np_128387, 'array')
            # Calling array(args, kwargs) (line 439)
            array_call_result_128391 = invoke(stypy.reporting.localization.Localization(__file__, 439, 20), array_128388, *[flows_128389], **kwargs_128390)
            
            # Assigning a type to the variable 'flows' (line 439)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'flows', array_call_result_128391)

            if (may_be_128378 and more_types_in_union_128379):
                # SSA join for if statement (line 436)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Subscript to a Name (line 440):
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        int_128392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 24), 'int')
        # Getting the type of 'flows' (line 440)
        flows_128393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'flows')
        # Obtaining the member 'shape' of a type (line 440)
        shape_128394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 12), flows_128393, 'shape')
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___128395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 12), shape_128394, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_128396 = invoke(stypy.reporting.localization.Localization(__file__, 440, 12), getitem___128395, int_128392)
        
        # Assigning a type to the variable 'n' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'n', subscript_call_result_128396)
        
        # Type idiom detected: calculating its left and rigth part (line 441)
        # Getting the type of 'rotation' (line 441)
        rotation_128397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'rotation')
        # Getting the type of 'None' (line 441)
        None_128398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'None')
        
        (may_be_128399, more_types_in_union_128400) = may_be_none(rotation_128397, None_128398)

        if may_be_128399:

            if more_types_in_union_128400:
                # Runtime conditional SSA (line 441)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 442):
            
            # Assigning a Num to a Name (line 442):
            int_128401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 23), 'int')
            # Assigning a type to the variable 'rotation' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'rotation', int_128401)

            if more_types_in_union_128400:
                # Runtime conditional SSA for else branch (line 441)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_128399) or more_types_in_union_128400):
            
            # Getting the type of 'rotation' (line 445)
            rotation_128402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'rotation')
            float_128403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 24), 'float')
            # Applying the binary operator 'div=' (line 445)
            result_div_128404 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 12), 'div=', rotation_128402, float_128403)
            # Assigning a type to the variable 'rotation' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'rotation', result_div_128404)
            

            if (may_be_128399 and more_types_in_union_128400):
                # SSA join for if statement (line 441)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 446)
        # Getting the type of 'orientations' (line 446)
        orientations_128405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 11), 'orientations')
        # Getting the type of 'None' (line 446)
        None_128406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 27), 'None')
        
        (may_be_128407, more_types_in_union_128408) = may_be_none(orientations_128405, None_128406)

        if may_be_128407:

            if more_types_in_union_128408:
                # Runtime conditional SSA (line 446)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 447):
            
            # Assigning a List to a Name (line 447):
            
            # Obtaining an instance of the builtin type 'list' (line 447)
            list_128409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 447)
            # Adding element type (line 447)
            int_128410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 28), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 27), list_128409, int_128410)
            # Adding element type (line 447)
            int_128411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 31), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 27), list_128409, int_128411)
            
            # Assigning a type to the variable 'orientations' (line 447)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'orientations', list_128409)

            if more_types_in_union_128408:
                # SSA join for if statement (line 446)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to len(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'orientations' (line 448)
        orientations_128413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'orientations', False)
        # Processing the call keyword arguments (line 448)
        kwargs_128414 = {}
        # Getting the type of 'len' (line 448)
        len_128412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'len', False)
        # Calling len(args, kwargs) (line 448)
        len_call_result_128415 = invoke(stypy.reporting.localization.Localization(__file__, 448, 11), len_128412, *[orientations_128413], **kwargs_128414)
        
        # Getting the type of 'n' (line 448)
        n_128416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 32), 'n')
        # Applying the binary operator '!=' (line 448)
        result_ne_128417 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 11), '!=', len_call_result_128415, n_128416)
        
        # Testing the type of an if condition (line 448)
        if_condition_128418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 8), result_ne_128417)
        # Assigning a type to the variable 'if_condition_128418' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'if_condition_128418', if_condition_128418)
        # SSA begins for if statement (line 448)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 449)
        # Processing the call arguments (line 449)
        unicode_128420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 12), 'unicode', u'orientations and flows must have the same length.\norientations has length %d, but flows has length %d.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 452)
        tuple_128421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 452)
        # Adding element type (line 452)
        
        # Call to len(...): (line 452)
        # Processing the call arguments (line 452)
        # Getting the type of 'orientations' (line 452)
        orientations_128423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), 'orientations', False)
        # Processing the call keyword arguments (line 452)
        kwargs_128424 = {}
        # Getting the type of 'len' (line 452)
        len_128422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 15), 'len', False)
        # Calling len(args, kwargs) (line 452)
        len_call_result_128425 = invoke(stypy.reporting.localization.Localization(__file__, 452, 15), len_128422, *[orientations_128423], **kwargs_128424)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 15), tuple_128421, len_call_result_128425)
        # Adding element type (line 452)
        # Getting the type of 'n' (line 452)
        n_128426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 34), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 15), tuple_128421, n_128426)
        
        # Applying the binary operator '%' (line 450)
        result_mod_128427 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 12), '%', unicode_128420, tuple_128421)
        
        # Processing the call keyword arguments (line 449)
        kwargs_128428 = {}
        # Getting the type of 'ValueError' (line 449)
        ValueError_128419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 449)
        ValueError_call_result_128429 = invoke(stypy.reporting.localization.Localization(__file__, 449, 18), ValueError_128419, *[result_mod_128427], **kwargs_128428)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 449, 12), ValueError_call_result_128429, 'raise parameter', BaseException)
        # SSA join for if statement (line 448)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'labels' (line 453)
        labels_128430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'labels')
        unicode_128431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 21), 'unicode', u'')
        # Applying the binary operator '!=' (line 453)
        result_ne_128432 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 11), '!=', labels_128430, unicode_128431)
        
        
        # Call to getattr(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'labels' (line 453)
        labels_128434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 36), 'labels', False)
        unicode_128435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 44), 'unicode', u'__iter__')
        # Getting the type of 'False' (line 453)
        False_128436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 56), 'False', False)
        # Processing the call keyword arguments (line 453)
        kwargs_128437 = {}
        # Getting the type of 'getattr' (line 453)
        getattr_128433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 453)
        getattr_call_result_128438 = invoke(stypy.reporting.localization.Localization(__file__, 453, 28), getattr_128433, *[labels_128434, unicode_128435, False_128436], **kwargs_128437)
        
        # Applying the binary operator 'and' (line 453)
        result_and_keyword_128439 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 11), 'and', result_ne_128432, getattr_call_result_128438)
        
        # Testing the type of an if condition (line 453)
        if_condition_128440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), result_and_keyword_128439)
        # Assigning a type to the variable 'if_condition_128440' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_128440', if_condition_128440)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'labels' (line 456)
        labels_128442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 19), 'labels', False)
        # Processing the call keyword arguments (line 456)
        kwargs_128443 = {}
        # Getting the type of 'len' (line 456)
        len_128441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'len', False)
        # Calling len(args, kwargs) (line 456)
        len_call_result_128444 = invoke(stypy.reporting.localization.Localization(__file__, 456, 15), len_128441, *[labels_128442], **kwargs_128443)
        
        # Getting the type of 'n' (line 456)
        n_128445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'n')
        # Applying the binary operator '!=' (line 456)
        result_ne_128446 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 15), '!=', len_call_result_128444, n_128445)
        
        # Testing the type of an if condition (line 456)
        if_condition_128447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 12), result_ne_128446)
        # Assigning a type to the variable 'if_condition_128447' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'if_condition_128447', if_condition_128447)
        # SSA begins for if statement (line 456)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 457)
        # Processing the call arguments (line 457)
        unicode_128449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 16), 'unicode', u'If labels is a list, then labels and flows must have the same length.\nlabels has length %d, but flows has length %d.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 460)
        tuple_128450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 460)
        # Adding element type (line 460)
        
        # Call to len(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'labels' (line 460)
        labels_128452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'labels', False)
        # Processing the call keyword arguments (line 460)
        kwargs_128453 = {}
        # Getting the type of 'len' (line 460)
        len_128451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 19), 'len', False)
        # Calling len(args, kwargs) (line 460)
        len_call_result_128454 = invoke(stypy.reporting.localization.Localization(__file__, 460, 19), len_128451, *[labels_128452], **kwargs_128453)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 19), tuple_128450, len_call_result_128454)
        # Adding element type (line 460)
        # Getting the type of 'n' (line 460)
        n_128455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 32), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 19), tuple_128450, n_128455)
        
        # Applying the binary operator '%' (line 458)
        result_mod_128456 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 16), '%', unicode_128449, tuple_128450)
        
        # Processing the call keyword arguments (line 457)
        kwargs_128457 = {}
        # Getting the type of 'ValueError' (line 457)
        ValueError_128448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 457)
        ValueError_call_result_128458 = invoke(stypy.reporting.localization.Localization(__file__, 457, 22), ValueError_128448, *[result_mod_128456], **kwargs_128457)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 457, 16), ValueError_call_result_128458, 'raise parameter', BaseException)
        # SSA join for if statement (line 456)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 453)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 462):
        
        # Assigning a BinOp to a Name (line 462):
        
        # Obtaining an instance of the builtin type 'list' (line 462)
        list_128459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 462)
        # Adding element type (line 462)
        # Getting the type of 'labels' (line 462)
        labels_128460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 22), 'labels')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 21), list_128459, labels_128460)
        
        # Getting the type of 'n' (line 462)
        n_128461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 32), 'n')
        # Applying the binary operator '*' (line 462)
        result_mul_128462 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 21), '*', list_128459, n_128461)
        
        # Assigning a type to the variable 'labels' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'labels', result_mul_128462)
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'trunklength' (line 463)
        trunklength_128463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'trunklength')
        int_128464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 25), 'int')
        # Applying the binary operator '<' (line 463)
        result_lt_128465 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 11), '<', trunklength_128463, int_128464)
        
        # Testing the type of an if condition (line 463)
        if_condition_128466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 8), result_lt_128465)
        # Assigning a type to the variable 'if_condition_128466' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'if_condition_128466', if_condition_128466)
        # SSA begins for if statement (line 463)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 464)
        # Processing the call arguments (line 464)
        unicode_128468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 12), 'unicode', u"trunklength is negative.\nThis isn't allowed, because it would cause poor layout.")
        # Processing the call keyword arguments (line 464)
        kwargs_128469 = {}
        # Getting the type of 'ValueError' (line 464)
        ValueError_128467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 464)
        ValueError_call_result_128470 = invoke(stypy.reporting.localization.Localization(__file__, 464, 18), ValueError_128467, *[unicode_128468], **kwargs_128469)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 464, 12), ValueError_call_result_128470, 'raise parameter', BaseException)
        # SSA join for if statement (line 463)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 467)
        # Processing the call arguments (line 467)
        
        # Call to sum(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'flows' (line 467)
        flows_128475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 25), 'flows', False)
        # Processing the call keyword arguments (line 467)
        kwargs_128476 = {}
        # Getting the type of 'np' (line 467)
        np_128473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 18), 'np', False)
        # Obtaining the member 'sum' of a type (line 467)
        sum_128474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 18), np_128473, 'sum')
        # Calling sum(args, kwargs) (line 467)
        sum_call_result_128477 = invoke(stypy.reporting.localization.Localization(__file__, 467, 18), sum_128474, *[flows_128475], **kwargs_128476)
        
        # Processing the call keyword arguments (line 467)
        kwargs_128478 = {}
        # Getting the type of 'np' (line 467)
        np_128471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 11), 'np', False)
        # Obtaining the member 'abs' of a type (line 467)
        abs_128472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 11), np_128471, 'abs')
        # Calling abs(args, kwargs) (line 467)
        abs_call_result_128479 = invoke(stypy.reporting.localization.Localization(__file__, 467, 11), abs_128472, *[sum_call_result_128477], **kwargs_128478)
        
        # Getting the type of 'self' (line 467)
        self_128480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 35), 'self')
        # Obtaining the member 'tolerance' of a type (line 467)
        tolerance_128481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 35), self_128480, 'tolerance')
        # Applying the binary operator '>' (line 467)
        result_gt_128482 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 11), '>', abs_call_result_128479, tolerance_128481)
        
        # Testing the type of an if condition (line 467)
        if_condition_128483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 8), result_gt_128482)
        # Assigning a type to the variable 'if_condition_128483' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'if_condition_128483', if_condition_128483)
        # SSA begins for if statement (line 467)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to report(...): (line 468)
        # Processing the call arguments (line 468)
        unicode_128486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 16), 'unicode', u'The sum of the flows is nonzero (%f).\nIs the system not at steady state?')
        
        # Call to sum(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'flows' (line 470)
        flows_128489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 55), 'flows', False)
        # Processing the call keyword arguments (line 470)
        kwargs_128490 = {}
        # Getting the type of 'np' (line 470)
        np_128487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 48), 'np', False)
        # Obtaining the member 'sum' of a type (line 470)
        sum_128488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 48), np_128487, 'sum')
        # Calling sum(args, kwargs) (line 470)
        sum_call_result_128491 = invoke(stypy.reporting.localization.Localization(__file__, 470, 48), sum_128488, *[flows_128489], **kwargs_128490)
        
        # Applying the binary operator '%' (line 469)
        result_mod_128492 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 16), '%', unicode_128486, sum_call_result_128491)
        
        unicode_128493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 63), 'unicode', u'helpful')
        # Processing the call keyword arguments (line 468)
        kwargs_128494 = {}
        # Getting the type of 'verbose' (line 468)
        verbose_128484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'verbose', False)
        # Obtaining the member 'report' of a type (line 468)
        report_128485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 12), verbose_128484, 'report')
        # Calling report(args, kwargs) (line 468)
        report_call_result_128495 = invoke(stypy.reporting.localization.Localization(__file__, 468, 12), report_128485, *[result_mod_128492, unicode_128493], **kwargs_128494)
        
        # SSA join for if statement (line 467)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 471):
        
        # Assigning a BinOp to a Name (line 471):
        # Getting the type of 'self' (line 471)
        self_128496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 23), 'self')
        # Obtaining the member 'scale' of a type (line 471)
        scale_128497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 23), self_128496, 'scale')
        # Getting the type of 'flows' (line 471)
        flows_128498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 36), 'flows')
        # Applying the binary operator '*' (line 471)
        result_mul_128499 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 23), '*', scale_128497, flows_128498)
        
        # Assigning a type to the variable 'scaled_flows' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'scaled_flows', result_mul_128499)
        
        # Assigning a Call to a Name (line 472):
        
        # Assigning a Call to a Name (line 472):
        
        # Call to sum(...): (line 472)
        # Processing the call arguments (line 472)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 472, 19, True)
        # Calculating comprehension expression
        # Getting the type of 'scaled_flows' (line 472)
        scaled_flows_128506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 44), 'scaled_flows', False)
        comprehension_128507 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 19), scaled_flows_128506)
        # Assigning a type to the variable 'flow' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 19), 'flow', comprehension_128507)
        
        # Call to max(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'flow' (line 472)
        flow_128502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 23), 'flow', False)
        int_128503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 29), 'int')
        # Processing the call keyword arguments (line 472)
        kwargs_128504 = {}
        # Getting the type of 'max' (line 472)
        max_128501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 19), 'max', False)
        # Calling max(args, kwargs) (line 472)
        max_call_result_128505 = invoke(stypy.reporting.localization.Localization(__file__, 472, 19), max_128501, *[flow_128502, int_128503], **kwargs_128504)
        
        list_128508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 19), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 19), list_128508, max_call_result_128505)
        # Processing the call keyword arguments (line 472)
        kwargs_128509 = {}
        # Getting the type of 'sum' (line 472)
        sum_128500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'sum', False)
        # Calling sum(args, kwargs) (line 472)
        sum_call_result_128510 = invoke(stypy.reporting.localization.Localization(__file__, 472, 15), sum_128500, *[list_128508], **kwargs_128509)
        
        # Assigning a type to the variable 'gain' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'gain', sum_call_result_128510)
        
        # Assigning a Call to a Name (line 473):
        
        # Assigning a Call to a Name (line 473):
        
        # Call to sum(...): (line 473)
        # Processing the call arguments (line 473)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 473, 19, True)
        # Calculating comprehension expression
        # Getting the type of 'scaled_flows' (line 473)
        scaled_flows_128517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 44), 'scaled_flows', False)
        comprehension_128518 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 19), scaled_flows_128517)
        # Assigning a type to the variable 'flow' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'flow', comprehension_128518)
        
        # Call to min(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'flow' (line 473)
        flow_128513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 23), 'flow', False)
        int_128514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 29), 'int')
        # Processing the call keyword arguments (line 473)
        kwargs_128515 = {}
        # Getting the type of 'min' (line 473)
        min_128512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'min', False)
        # Calling min(args, kwargs) (line 473)
        min_call_result_128516 = invoke(stypy.reporting.localization.Localization(__file__, 473, 19), min_128512, *[flow_128513, int_128514], **kwargs_128515)
        
        list_128519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 19), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 19), list_128519, min_call_result_128516)
        # Processing the call keyword arguments (line 473)
        kwargs_128520 = {}
        # Getting the type of 'sum' (line 473)
        sum_128511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 15), 'sum', False)
        # Calling sum(args, kwargs) (line 473)
        sum_call_result_128521 = invoke(stypy.reporting.localization.Localization(__file__, 473, 15), sum_128511, *[list_128519], **kwargs_128520)
        
        # Assigning a type to the variable 'loss' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'loss', sum_call_result_128521)
        
        
        
        float_128522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 16), 'float')
        # Getting the type of 'gain' (line 474)
        gain_128523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), 'gain')
        # Applying the binary operator '<=' (line 474)
        result_le_128524 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 16), '<=', float_128522, gain_128523)
        float_128525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 31), 'float')
        # Applying the binary operator '<=' (line 474)
        result_le_128526 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 16), '<=', gain_128523, float_128525)
        # Applying the binary operator '&' (line 474)
        result_and__128527 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 16), '&', result_le_128524, result_le_128526)
        
        # Applying the 'not' unary operator (line 474)
        result_not__128528 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 11), 'not', result_and__128527)
        
        # Testing the type of an if condition (line 474)
        if_condition_128529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 8), result_not__128528)
        # Assigning a type to the variable 'if_condition_128529' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'if_condition_128529', if_condition_128529)
        # SSA begins for if statement (line 474)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to report(...): (line 475)
        # Processing the call arguments (line 475)
        unicode_128532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 16), 'unicode', u'The scaled sum of the inputs is %f.\nThis may cause poor layout.\nConsider changing the scale so that the scaled sum is approximately 1.0.')
        # Getting the type of 'gain' (line 478)
        gain_128533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 63), 'gain', False)
        # Applying the binary operator '%' (line 476)
        result_mod_128534 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 16), '%', unicode_128532, gain_128533)
        
        unicode_128535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 69), 'unicode', u'helpful')
        # Processing the call keyword arguments (line 475)
        kwargs_128536 = {}
        # Getting the type of 'verbose' (line 475)
        verbose_128530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'verbose', False)
        # Obtaining the member 'report' of a type (line 475)
        report_128531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), verbose_128530, 'report')
        # Calling report(args, kwargs) (line 475)
        report_call_result_128537 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), report_128531, *[result_mod_128534, unicode_128535], **kwargs_128536)
        
        # SSA join for if statement (line 474)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        float_128538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 16), 'float')
        # Getting the type of 'loss' (line 479)
        loss_128539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 24), 'loss')
        # Applying the binary operator '<=' (line 479)
        result_le_128540 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 16), '<=', float_128538, loss_128539)
        float_128541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 32), 'float')
        # Applying the binary operator '<=' (line 479)
        result_le_128542 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 16), '<=', loss_128539, float_128541)
        # Applying the binary operator '&' (line 479)
        result_and__128543 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 16), '&', result_le_128540, result_le_128542)
        
        # Applying the 'not' unary operator (line 479)
        result_not__128544 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 11), 'not', result_and__128543)
        
        # Testing the type of an if condition (line 479)
        if_condition_128545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 8), result_not__128544)
        # Assigning a type to the variable 'if_condition_128545' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'if_condition_128545', if_condition_128545)
        # SSA begins for if statement (line 479)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to report(...): (line 480)
        # Processing the call arguments (line 480)
        unicode_128548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 16), 'unicode', u'The scaled sum of the outputs is %f.\nThis may cause poor layout.\nConsider changing the scale so that the scaled sum is approximately 1.0.')
        # Getting the type of 'gain' (line 483)
        gain_128549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 63), 'gain', False)
        # Applying the binary operator '%' (line 481)
        result_mod_128550 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 16), '%', unicode_128548, gain_128549)
        
        unicode_128551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 69), 'unicode', u'helpful')
        # Processing the call keyword arguments (line 480)
        kwargs_128552 = {}
        # Getting the type of 'verbose' (line 480)
        verbose_128546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'verbose', False)
        # Obtaining the member 'report' of a type (line 480)
        report_128547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 12), verbose_128546, 'report')
        # Calling report(args, kwargs) (line 480)
        report_call_result_128553 = invoke(stypy.reporting.localization.Localization(__file__, 480, 12), report_128547, *[result_mod_128550, unicode_128551], **kwargs_128552)
        
        # SSA join for if statement (line 479)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 484)
        # Getting the type of 'prior' (line 484)
        prior_128554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'prior')
        # Getting the type of 'None' (line 484)
        None_128555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 24), 'None')
        
        (may_be_128556, more_types_in_union_128557) = may_not_be_none(prior_128554, None_128555)

        if may_be_128556:

            if more_types_in_union_128557:
                # Runtime conditional SSA (line 484)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'prior' (line 485)
            prior_128558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 15), 'prior')
            int_128559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 23), 'int')
            # Applying the binary operator '<' (line 485)
            result_lt_128560 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 15), '<', prior_128558, int_128559)
            
            # Testing the type of an if condition (line 485)
            if_condition_128561 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 12), result_lt_128560)
            # Assigning a type to the variable 'if_condition_128561' (line 485)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'if_condition_128561', if_condition_128561)
            # SSA begins for if statement (line 485)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 486)
            # Processing the call arguments (line 486)
            unicode_128563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 33), 'unicode', u'The index of the prior diagram is negative.')
            # Processing the call keyword arguments (line 486)
            kwargs_128564 = {}
            # Getting the type of 'ValueError' (line 486)
            ValueError_128562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 486)
            ValueError_call_result_128565 = invoke(stypy.reporting.localization.Localization(__file__, 486, 22), ValueError_128562, *[unicode_128563], **kwargs_128564)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 486, 16), ValueError_call_result_128565, 'raise parameter', BaseException)
            # SSA join for if statement (line 485)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Call to min(...): (line 487)
            # Processing the call arguments (line 487)
            # Getting the type of 'connect' (line 487)
            connect_128567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 19), 'connect', False)
            # Processing the call keyword arguments (line 487)
            kwargs_128568 = {}
            # Getting the type of 'min' (line 487)
            min_128566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 15), 'min', False)
            # Calling min(args, kwargs) (line 487)
            min_call_result_128569 = invoke(stypy.reporting.localization.Localization(__file__, 487, 15), min_128566, *[connect_128567], **kwargs_128568)
            
            int_128570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 30), 'int')
            # Applying the binary operator '<' (line 487)
            result_lt_128571 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 15), '<', min_call_result_128569, int_128570)
            
            # Testing the type of an if condition (line 487)
            if_condition_128572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 487, 12), result_lt_128571)
            # Assigning a type to the variable 'if_condition_128572' (line 487)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'if_condition_128572', if_condition_128572)
            # SSA begins for if statement (line 487)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 488)
            # Processing the call arguments (line 488)
            unicode_128574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 16), 'unicode', u'At least one of the connection indices is negative.')
            # Processing the call keyword arguments (line 488)
            kwargs_128575 = {}
            # Getting the type of 'ValueError' (line 488)
            ValueError_128573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 488)
            ValueError_call_result_128576 = invoke(stypy.reporting.localization.Localization(__file__, 488, 22), ValueError_128573, *[unicode_128574], **kwargs_128575)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 488, 16), ValueError_call_result_128576, 'raise parameter', BaseException)
            # SSA join for if statement (line 487)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'prior' (line 490)
            prior_128577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'prior')
            
            # Call to len(...): (line 490)
            # Processing the call arguments (line 490)
            # Getting the type of 'self' (line 490)
            self_128579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 28), 'self', False)
            # Obtaining the member 'diagrams' of a type (line 490)
            diagrams_128580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 28), self_128579, 'diagrams')
            # Processing the call keyword arguments (line 490)
            kwargs_128581 = {}
            # Getting the type of 'len' (line 490)
            len_128578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 24), 'len', False)
            # Calling len(args, kwargs) (line 490)
            len_call_result_128582 = invoke(stypy.reporting.localization.Localization(__file__, 490, 24), len_128578, *[diagrams_128580], **kwargs_128581)
            
            # Applying the binary operator '>=' (line 490)
            result_ge_128583 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 15), '>=', prior_128577, len_call_result_128582)
            
            # Testing the type of an if condition (line 490)
            if_condition_128584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 12), result_ge_128583)
            # Assigning a type to the variable 'if_condition_128584' (line 490)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'if_condition_128584', if_condition_128584)
            # SSA begins for if statement (line 490)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 491)
            # Processing the call arguments (line 491)
            unicode_128586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 16), 'unicode', u'The index of the prior diagram is %d, but there are only %d other diagrams.\nThe index is zero-based.')
            
            # Obtaining an instance of the builtin type 'tuple' (line 494)
            tuple_128587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 494)
            # Adding element type (line 494)
            # Getting the type of 'prior' (line 494)
            prior_128588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'prior', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 19), tuple_128587, prior_128588)
            # Adding element type (line 494)
            
            # Call to len(...): (line 494)
            # Processing the call arguments (line 494)
            # Getting the type of 'self' (line 494)
            self_128590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 30), 'self', False)
            # Obtaining the member 'diagrams' of a type (line 494)
            diagrams_128591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 30), self_128590, 'diagrams')
            # Processing the call keyword arguments (line 494)
            kwargs_128592 = {}
            # Getting the type of 'len' (line 494)
            len_128589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 26), 'len', False)
            # Calling len(args, kwargs) (line 494)
            len_call_result_128593 = invoke(stypy.reporting.localization.Localization(__file__, 494, 26), len_128589, *[diagrams_128591], **kwargs_128592)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 19), tuple_128587, len_call_result_128593)
            
            # Applying the binary operator '%' (line 492)
            result_mod_128594 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 16), '%', unicode_128586, tuple_128587)
            
            # Processing the call keyword arguments (line 491)
            kwargs_128595 = {}
            # Getting the type of 'ValueError' (line 491)
            ValueError_128585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 491)
            ValueError_call_result_128596 = invoke(stypy.reporting.localization.Localization(__file__, 491, 22), ValueError_128585, *[result_mod_128594], **kwargs_128595)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 491, 16), ValueError_call_result_128596, 'raise parameter', BaseException)
            # SSA join for if statement (line 490)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Obtaining the type of the subscript
            int_128597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 23), 'int')
            # Getting the type of 'connect' (line 495)
            connect_128598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 15), 'connect')
            # Obtaining the member '__getitem__' of a type (line 495)
            getitem___128599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 15), connect_128598, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 495)
            subscript_call_result_128600 = invoke(stypy.reporting.localization.Localization(__file__, 495, 15), getitem___128599, int_128597)
            
            
            # Call to len(...): (line 495)
            # Processing the call arguments (line 495)
            
            # Obtaining the type of the subscript
            # Getting the type of 'prior' (line 495)
            prior_128602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 47), 'prior', False)
            # Getting the type of 'self' (line 495)
            self_128603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 33), 'self', False)
            # Obtaining the member 'diagrams' of a type (line 495)
            diagrams_128604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 33), self_128603, 'diagrams')
            # Obtaining the member '__getitem__' of a type (line 495)
            getitem___128605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 33), diagrams_128604, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 495)
            subscript_call_result_128606 = invoke(stypy.reporting.localization.Localization(__file__, 495, 33), getitem___128605, prior_128602)
            
            # Obtaining the member 'flows' of a type (line 495)
            flows_128607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 33), subscript_call_result_128606, 'flows')
            # Processing the call keyword arguments (line 495)
            kwargs_128608 = {}
            # Getting the type of 'len' (line 495)
            len_128601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 29), 'len', False)
            # Calling len(args, kwargs) (line 495)
            len_call_result_128609 = invoke(stypy.reporting.localization.Localization(__file__, 495, 29), len_128601, *[flows_128607], **kwargs_128608)
            
            # Applying the binary operator '>=' (line 495)
            result_ge_128610 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 15), '>=', subscript_call_result_128600, len_call_result_128609)
            
            # Testing the type of an if condition (line 495)
            if_condition_128611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 495, 12), result_ge_128610)
            # Assigning a type to the variable 'if_condition_128611' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'if_condition_128611', if_condition_128611)
            # SSA begins for if statement (line 495)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 496)
            # Processing the call arguments (line 496)
            unicode_128613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 16), 'unicode', u'The connection index to the source diagram is %d, but that diagram has only %d flows.\nThe index is zero-based.')
            
            # Obtaining an instance of the builtin type 'tuple' (line 499)
            tuple_128614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 499)
            # Adding element type (line 499)
            
            # Obtaining the type of the subscript
            int_128615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 27), 'int')
            # Getting the type of 'connect' (line 499)
            connect_128616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 19), 'connect', False)
            # Obtaining the member '__getitem__' of a type (line 499)
            getitem___128617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 19), connect_128616, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 499)
            subscript_call_result_128618 = invoke(stypy.reporting.localization.Localization(__file__, 499, 19), getitem___128617, int_128615)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 19), tuple_128614, subscript_call_result_128618)
            # Adding element type (line 499)
            
            # Call to len(...): (line 499)
            # Processing the call arguments (line 499)
            
            # Obtaining the type of the subscript
            # Getting the type of 'prior' (line 499)
            prior_128620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 49), 'prior', False)
            # Getting the type of 'self' (line 499)
            self_128621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 35), 'self', False)
            # Obtaining the member 'diagrams' of a type (line 499)
            diagrams_128622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 35), self_128621, 'diagrams')
            # Obtaining the member '__getitem__' of a type (line 499)
            getitem___128623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 35), diagrams_128622, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 499)
            subscript_call_result_128624 = invoke(stypy.reporting.localization.Localization(__file__, 499, 35), getitem___128623, prior_128620)
            
            # Obtaining the member 'flows' of a type (line 499)
            flows_128625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 35), subscript_call_result_128624, 'flows')
            # Processing the call keyword arguments (line 499)
            kwargs_128626 = {}
            # Getting the type of 'len' (line 499)
            len_128619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 31), 'len', False)
            # Calling len(args, kwargs) (line 499)
            len_call_result_128627 = invoke(stypy.reporting.localization.Localization(__file__, 499, 31), len_128619, *[flows_128625], **kwargs_128626)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 19), tuple_128614, len_call_result_128627)
            
            # Applying the binary operator '%' (line 497)
            result_mod_128628 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 16), '%', unicode_128613, tuple_128614)
            
            # Processing the call keyword arguments (line 496)
            kwargs_128629 = {}
            # Getting the type of 'ValueError' (line 496)
            ValueError_128612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 496)
            ValueError_call_result_128630 = invoke(stypy.reporting.localization.Localization(__file__, 496, 22), ValueError_128612, *[result_mod_128628], **kwargs_128629)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 496, 16), ValueError_call_result_128630, 'raise parameter', BaseException)
            # SSA join for if statement (line 495)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Obtaining the type of the subscript
            int_128631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 23), 'int')
            # Getting the type of 'connect' (line 500)
            connect_128632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'connect')
            # Obtaining the member '__getitem__' of a type (line 500)
            getitem___128633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 15), connect_128632, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 500)
            subscript_call_result_128634 = invoke(stypy.reporting.localization.Localization(__file__, 500, 15), getitem___128633, int_128631)
            
            # Getting the type of 'n' (line 500)
            n_128635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), 'n')
            # Applying the binary operator '>=' (line 500)
            result_ge_128636 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 15), '>=', subscript_call_result_128634, n_128635)
            
            # Testing the type of an if condition (line 500)
            if_condition_128637 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 12), result_ge_128636)
            # Assigning a type to the variable 'if_condition_128637' (line 500)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'if_condition_128637', if_condition_128637)
            # SSA begins for if statement (line 500)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 501)
            # Processing the call arguments (line 501)
            unicode_128639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 16), 'unicode', u'The connection index to this diagram is %d, but this diagramhas only %d flows.\n The index is zero-based.')
            
            # Obtaining an instance of the builtin type 'tuple' (line 504)
            tuple_128640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 504)
            # Adding element type (line 504)
            
            # Obtaining the type of the subscript
            int_128641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 27), 'int')
            # Getting the type of 'connect' (line 504)
            connect_128642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 19), 'connect', False)
            # Obtaining the member '__getitem__' of a type (line 504)
            getitem___128643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 19), connect_128642, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 504)
            subscript_call_result_128644 = invoke(stypy.reporting.localization.Localization(__file__, 504, 19), getitem___128643, int_128641)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 19), tuple_128640, subscript_call_result_128644)
            # Adding element type (line 504)
            # Getting the type of 'n' (line 504)
            n_128645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 31), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 19), tuple_128640, n_128645)
            
            # Applying the binary operator '%' (line 502)
            result_mod_128646 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 16), '%', unicode_128639, tuple_128640)
            
            # Processing the call keyword arguments (line 501)
            kwargs_128647 = {}
            # Getting the type of 'ValueError' (line 501)
            ValueError_128638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 501)
            ValueError_call_result_128648 = invoke(stypy.reporting.localization.Localization(__file__, 501, 22), ValueError_128638, *[result_mod_128646], **kwargs_128647)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 501, 16), ValueError_call_result_128648, 'raise parameter', BaseException)
            # SSA join for if statement (line 500)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Type idiom detected: calculating its left and rigth part (line 505)
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_128649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 51), 'int')
            # Getting the type of 'connect' (line 505)
            connect_128650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 43), 'connect')
            # Obtaining the member '__getitem__' of a type (line 505)
            getitem___128651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 43), connect_128650, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 505)
            subscript_call_result_128652 = invoke(stypy.reporting.localization.Localization(__file__, 505, 43), getitem___128651, int_128649)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'prior' (line 505)
            prior_128653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 29), 'prior')
            # Getting the type of 'self' (line 505)
            self_128654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 15), 'self')
            # Obtaining the member 'diagrams' of a type (line 505)
            diagrams_128655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 15), self_128654, 'diagrams')
            # Obtaining the member '__getitem__' of a type (line 505)
            getitem___128656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 15), diagrams_128655, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 505)
            subscript_call_result_128657 = invoke(stypy.reporting.localization.Localization(__file__, 505, 15), getitem___128656, prior_128653)
            
            # Obtaining the member 'angles' of a type (line 505)
            angles_128658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 15), subscript_call_result_128657, 'angles')
            # Obtaining the member '__getitem__' of a type (line 505)
            getitem___128659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 15), angles_128658, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 505)
            subscript_call_result_128660 = invoke(stypy.reporting.localization.Localization(__file__, 505, 15), getitem___128659, subscript_call_result_128652)
            
            # Getting the type of 'None' (line 505)
            None_128661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 58), 'None')
            
            (may_be_128662, more_types_in_union_128663) = may_be_none(subscript_call_result_128660, None_128661)

            if may_be_128662:

                if more_types_in_union_128663:
                    # Runtime conditional SSA (line 505)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to ValueError(...): (line 506)
                # Processing the call arguments (line 506)
                unicode_128665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 16), 'unicode', u'The connection cannot be made.  Check that the magnitude of flow %d of diagram %d is greater than or equal to the specified tolerance.')
                
                # Obtaining an instance of the builtin type 'tuple' (line 509)
                tuple_128666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 42), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 509)
                # Adding element type (line 509)
                
                # Obtaining the type of the subscript
                int_128667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 50), 'int')
                # Getting the type of 'connect' (line 509)
                connect_128668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 42), 'connect', False)
                # Obtaining the member '__getitem__' of a type (line 509)
                getitem___128669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 42), connect_128668, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 509)
                subscript_call_result_128670 = invoke(stypy.reporting.localization.Localization(__file__, 509, 42), getitem___128669, int_128667)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 42), tuple_128666, subscript_call_result_128670)
                # Adding element type (line 509)
                # Getting the type of 'prior' (line 509)
                prior_128671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 54), 'prior', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 42), tuple_128666, prior_128671)
                
                # Applying the binary operator '%' (line 507)
                result_mod_128672 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 16), '%', unicode_128665, tuple_128666)
                
                # Processing the call keyword arguments (line 506)
                kwargs_128673 = {}
                # Getting the type of 'ValueError' (line 506)
                ValueError_128664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 22), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 506)
                ValueError_call_result_128674 = invoke(stypy.reporting.localization.Localization(__file__, 506, 22), ValueError_128664, *[result_mod_128672], **kwargs_128673)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 506, 16), ValueError_call_result_128674, 'raise parameter', BaseException)

                if more_types_in_union_128663:
                    # SSA join for if statement (line 505)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a BinOp to a Name (line 510):
            
            # Assigning a BinOp to a Name (line 510):
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_128675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 61), 'int')
            # Getting the type of 'connect' (line 510)
            connect_128676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 53), 'connect')
            # Obtaining the member '__getitem__' of a type (line 510)
            getitem___128677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 53), connect_128676, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 510)
            subscript_call_result_128678 = invoke(stypy.reporting.localization.Localization(__file__, 510, 53), getitem___128677, int_128675)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'prior' (line 510)
            prior_128679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 40), 'prior')
            # Getting the type of 'self' (line 510)
            self_128680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'self')
            # Obtaining the member 'diagrams' of a type (line 510)
            diagrams_128681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 26), self_128680, 'diagrams')
            # Obtaining the member '__getitem__' of a type (line 510)
            getitem___128682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 26), diagrams_128681, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 510)
            subscript_call_result_128683 = invoke(stypy.reporting.localization.Localization(__file__, 510, 26), getitem___128682, prior_128679)
            
            # Obtaining the member 'flows' of a type (line 510)
            flows_128684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 26), subscript_call_result_128683, 'flows')
            # Obtaining the member '__getitem__' of a type (line 510)
            getitem___128685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 26), flows_128684, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 510)
            subscript_call_result_128686 = invoke(stypy.reporting.localization.Localization(__file__, 510, 26), getitem___128685, subscript_call_result_128678)
            
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_128687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 40), 'int')
            # Getting the type of 'connect' (line 511)
            connect_128688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 32), 'connect')
            # Obtaining the member '__getitem__' of a type (line 511)
            getitem___128689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 32), connect_128688, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 511)
            subscript_call_result_128690 = invoke(stypy.reporting.localization.Localization(__file__, 511, 32), getitem___128689, int_128687)
            
            # Getting the type of 'flows' (line 511)
            flows_128691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 26), 'flows')
            # Obtaining the member '__getitem__' of a type (line 511)
            getitem___128692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 26), flows_128691, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 511)
            subscript_call_result_128693 = invoke(stypy.reporting.localization.Localization(__file__, 511, 26), getitem___128692, subscript_call_result_128690)
            
            # Applying the binary operator '+' (line 510)
            result_add_128694 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 26), '+', subscript_call_result_128686, subscript_call_result_128693)
            
            # Assigning a type to the variable 'flow_error' (line 510)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'flow_error', result_add_128694)
            
            
            
            # Call to abs(...): (line 512)
            # Processing the call arguments (line 512)
            # Getting the type of 'flow_error' (line 512)
            flow_error_128696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 19), 'flow_error', False)
            # Processing the call keyword arguments (line 512)
            kwargs_128697 = {}
            # Getting the type of 'abs' (line 512)
            abs_128695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 15), 'abs', False)
            # Calling abs(args, kwargs) (line 512)
            abs_call_result_128698 = invoke(stypy.reporting.localization.Localization(__file__, 512, 15), abs_128695, *[flow_error_128696], **kwargs_128697)
            
            # Getting the type of 'self' (line 512)
            self_128699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 34), 'self')
            # Obtaining the member 'tolerance' of a type (line 512)
            tolerance_128700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 34), self_128699, 'tolerance')
            # Applying the binary operator '>=' (line 512)
            result_ge_128701 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 15), '>=', abs_call_result_128698, tolerance_128700)
            
            # Testing the type of an if condition (line 512)
            if_condition_128702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 12), result_ge_128701)
            # Assigning a type to the variable 'if_condition_128702' (line 512)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'if_condition_128702', if_condition_128702)
            # SSA begins for if statement (line 512)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 513)
            # Processing the call arguments (line 513)
            unicode_128704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 16), 'unicode', u'The scaled sum of the connected flows is %f, which is not within the tolerance (%f).')
            
            # Obtaining an instance of the builtin type 'tuple' (line 515)
            tuple_128705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 48), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 515)
            # Adding element type (line 515)
            # Getting the type of 'flow_error' (line 515)
            flow_error_128706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 48), 'flow_error', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 48), tuple_128705, flow_error_128706)
            # Adding element type (line 515)
            # Getting the type of 'self' (line 515)
            self_128707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 60), 'self', False)
            # Obtaining the member 'tolerance' of a type (line 515)
            tolerance_128708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 60), self_128707, 'tolerance')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 48), tuple_128705, tolerance_128708)
            
            # Applying the binary operator '%' (line 514)
            result_mod_128709 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 16), '%', unicode_128704, tuple_128705)
            
            # Processing the call keyword arguments (line 513)
            kwargs_128710 = {}
            # Getting the type of 'ValueError' (line 513)
            ValueError_128703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 513)
            ValueError_call_result_128711 = invoke(stypy.reporting.localization.Localization(__file__, 513, 22), ValueError_128703, *[result_mod_128709], **kwargs_128710)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 513, 16), ValueError_call_result_128711, 'raise parameter', BaseException)
            # SSA join for if statement (line 512)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_128557:
                # SSA join for if statement (line 484)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 518):
        
        # Assigning a BinOp to a Name (line 518):
        
        # Obtaining an instance of the builtin type 'list' (line 518)
        list_128712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 518)
        # Adding element type (line 518)
        # Getting the type of 'None' (line 518)
        None_128713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 22), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 21), list_128712, None_128713)
        
        # Getting the type of 'n' (line 518)
        n_128714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 30), 'n')
        # Applying the binary operator '*' (line 518)
        result_mul_128715 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 21), '*', list_128712, n_128714)
        
        # Assigning a type to the variable 'are_inputs' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'are_inputs', result_mul_128715)
        
        
        # Call to enumerate(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'flows' (line 519)
        flows_128717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 33), 'flows', False)
        # Processing the call keyword arguments (line 519)
        kwargs_128718 = {}
        # Getting the type of 'enumerate' (line 519)
        enumerate_128716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 23), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 519)
        enumerate_call_result_128719 = invoke(stypy.reporting.localization.Localization(__file__, 519, 23), enumerate_128716, *[flows_128717], **kwargs_128718)
        
        # Testing the type of a for loop iterable (line 519)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 519, 8), enumerate_call_result_128719)
        # Getting the type of the for loop variable (line 519)
        for_loop_var_128720 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 519, 8), enumerate_call_result_128719)
        # Assigning a type to the variable 'i' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 8), for_loop_var_128720))
        # Assigning a type to the variable 'flow' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'flow', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 8), for_loop_var_128720))
        # SSA begins for a for statement (line 519)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'flow' (line 520)
        flow_128721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 15), 'flow')
        # Getting the type of 'self' (line 520)
        self_128722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 23), 'self')
        # Obtaining the member 'tolerance' of a type (line 520)
        tolerance_128723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 23), self_128722, 'tolerance')
        # Applying the binary operator '>=' (line 520)
        result_ge_128724 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 15), '>=', flow_128721, tolerance_128723)
        
        # Testing the type of an if condition (line 520)
        if_condition_128725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 520, 12), result_ge_128724)
        # Assigning a type to the variable 'if_condition_128725' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'if_condition_128725', if_condition_128725)
        # SSA begins for if statement (line 520)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 521):
        
        # Assigning a Name to a Subscript (line 521):
        # Getting the type of 'True' (line 521)
        True_128726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 32), 'True')
        # Getting the type of 'are_inputs' (line 521)
        are_inputs_128727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 16), 'are_inputs')
        # Getting the type of 'i' (line 521)
        i_128728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 27), 'i')
        # Storing an element on a container (line 521)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 16), are_inputs_128727, (i_128728, True_128726))
        # SSA branch for the else part of an if statement (line 520)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'flow' (line 522)
        flow_128729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 17), 'flow')
        
        # Getting the type of 'self' (line 522)
        self_128730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 26), 'self')
        # Obtaining the member 'tolerance' of a type (line 522)
        tolerance_128731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 26), self_128730, 'tolerance')
        # Applying the 'usub' unary operator (line 522)
        result___neg___128732 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 25), 'usub', tolerance_128731)
        
        # Applying the binary operator '<=' (line 522)
        result_le_128733 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 17), '<=', flow_128729, result___neg___128732)
        
        # Testing the type of an if condition (line 522)
        if_condition_128734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 17), result_le_128733)
        # Assigning a type to the variable 'if_condition_128734' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 17), 'if_condition_128734', if_condition_128734)
        # SSA begins for if statement (line 522)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 523):
        
        # Assigning a Name to a Subscript (line 523):
        # Getting the type of 'False' (line 523)
        False_128735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 32), 'False')
        # Getting the type of 'are_inputs' (line 523)
        are_inputs_128736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'are_inputs')
        # Getting the type of 'i' (line 523)
        i_128737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 27), 'i')
        # Storing an element on a container (line 523)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 16), are_inputs_128736, (i_128737, False_128735))
        # SSA branch for the else part of an if statement (line 522)
        module_type_store.open_ssa_branch('else')
        
        # Call to report(...): (line 525)
        # Processing the call arguments (line 525)
        unicode_128740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 20), 'unicode', u'The magnitude of flow %d (%f) is below the tolerance (%f).\nIt will not be shown, and it cannot be used in a connection.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 529)
        tuple_128741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 529)
        # Adding element type (line 529)
        # Getting the type of 'i' (line 529)
        i_128742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 23), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 23), tuple_128741, i_128742)
        # Adding element type (line 529)
        # Getting the type of 'flow' (line 529)
        flow_128743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 26), 'flow', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 23), tuple_128741, flow_128743)
        # Adding element type (line 529)
        # Getting the type of 'self' (line 529)
        self_128744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 32), 'self', False)
        # Obtaining the member 'tolerance' of a type (line 529)
        tolerance_128745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 32), self_128744, 'tolerance')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 23), tuple_128741, tolerance_128745)
        
        # Applying the binary operator '%' (line 526)
        result_mod_128746 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 20), '%', unicode_128740, tuple_128741)
        
        unicode_128747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 49), 'unicode', u'helpful')
        # Processing the call keyword arguments (line 525)
        kwargs_128748 = {}
        # Getting the type of 'verbose' (line 525)
        verbose_128738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'verbose', False)
        # Obtaining the member 'report' of a type (line 525)
        report_128739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 16), verbose_128738, 'report')
        # Calling report(args, kwargs) (line 525)
        report_call_result_128749 = invoke(stypy.reporting.localization.Localization(__file__, 525, 16), report_128739, *[result_mod_128746, unicode_128747], **kwargs_128748)
        
        # SSA join for if statement (line 522)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 520)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 532):
        
        # Assigning a BinOp to a Name (line 532):
        
        # Obtaining an instance of the builtin type 'list' (line 532)
        list_128750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 532)
        # Adding element type (line 532)
        # Getting the type of 'None' (line 532)
        None_128751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 18), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 17), list_128750, None_128751)
        
        # Getting the type of 'n' (line 532)
        n_128752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 26), 'n')
        # Applying the binary operator '*' (line 532)
        result_mul_128753 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 17), '*', list_128750, n_128752)
        
        # Assigning a type to the variable 'angles' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'angles', result_mul_128753)
        
        
        # Call to enumerate(...): (line 533)
        # Processing the call arguments (line 533)
        
        # Call to zip(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'orientations' (line 533)
        orientations_128756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 51), 'orientations', False)
        # Getting the type of 'are_inputs' (line 533)
        are_inputs_128757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 65), 'are_inputs', False)
        # Processing the call keyword arguments (line 533)
        kwargs_128758 = {}
        # Getting the type of 'zip' (line 533)
        zip_128755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 47), 'zip', False)
        # Calling zip(args, kwargs) (line 533)
        zip_call_result_128759 = invoke(stypy.reporting.localization.Localization(__file__, 533, 47), zip_128755, *[orientations_128756, are_inputs_128757], **kwargs_128758)
        
        # Processing the call keyword arguments (line 533)
        kwargs_128760 = {}
        # Getting the type of 'enumerate' (line 533)
        enumerate_128754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 37), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 533)
        enumerate_call_result_128761 = invoke(stypy.reporting.localization.Localization(__file__, 533, 37), enumerate_128754, *[zip_call_result_128759], **kwargs_128760)
        
        # Testing the type of a for loop iterable (line 533)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 533, 8), enumerate_call_result_128761)
        # Getting the type of the for loop variable (line 533)
        for_loop_var_128762 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 533, 8), enumerate_call_result_128761)
        # Assigning a type to the variable 'i' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 8), for_loop_var_128762))
        # Assigning a type to the variable 'orient' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'orient', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 8), for_loop_var_128762))
        # Assigning a type to the variable 'is_input' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 8), for_loop_var_128762))
        # SSA begins for a for statement (line 533)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'orient' (line 534)
        orient_128763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 15), 'orient')
        int_128764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 25), 'int')
        # Applying the binary operator '==' (line 534)
        result_eq_128765 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 15), '==', orient_128763, int_128764)
        
        # Testing the type of an if condition (line 534)
        if_condition_128766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 12), result_eq_128765)
        # Assigning a type to the variable 'if_condition_128766' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'if_condition_128766', if_condition_128766)
        # SSA begins for if statement (line 534)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'is_input' (line 535)
        is_input_128767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 19), 'is_input')
        # Testing the type of an if condition (line 535)
        if_condition_128768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 16), is_input_128767)
        # Assigning a type to the variable 'if_condition_128768' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 16), 'if_condition_128768', if_condition_128768)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 536):
        
        # Assigning a Name to a Subscript (line 536):
        # Getting the type of 'DOWN' (line 536)
        DOWN_128769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 32), 'DOWN')
        # Getting the type of 'angles' (line 536)
        angles_128770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 20), 'angles')
        # Getting the type of 'i' (line 536)
        i_128771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 27), 'i')
        # Storing an element on a container (line 536)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 20), angles_128770, (i_128771, DOWN_128769))
        # SSA branch for the else part of an if statement (line 535)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'is_input' (line 537)
        is_input_128772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 25), 'is_input')
        # Applying the 'not' unary operator (line 537)
        result_not__128773 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 21), 'not', is_input_128772)
        
        # Testing the type of an if condition (line 537)
        if_condition_128774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 21), result_not__128773)
        # Assigning a type to the variable 'if_condition_128774' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 21), 'if_condition_128774', if_condition_128774)
        # SSA begins for if statement (line 537)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 539):
        
        # Assigning a Name to a Subscript (line 539):
        # Getting the type of 'UP' (line 539)
        UP_128775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 32), 'UP')
        # Getting the type of 'angles' (line 539)
        angles_128776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'angles')
        # Getting the type of 'i' (line 539)
        i_128777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 27), 'i')
        # Storing an element on a container (line 539)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 20), angles_128776, (i_128777, UP_128775))
        # SSA join for if statement (line 537)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 534)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'orient' (line 540)
        orient_128778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 17), 'orient')
        int_128779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 27), 'int')
        # Applying the binary operator '==' (line 540)
        result_eq_128780 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 17), '==', orient_128778, int_128779)
        
        # Testing the type of an if condition (line 540)
        if_condition_128781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 17), result_eq_128780)
        # Assigning a type to the variable 'if_condition_128781' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 17), 'if_condition_128781', if_condition_128781)
        # SSA begins for if statement (line 540)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 541)
        # Getting the type of 'is_input' (line 541)
        is_input_128782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'is_input')
        # Getting the type of 'None' (line 541)
        None_128783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 35), 'None')
        
        (may_be_128784, more_types_in_union_128785) = may_not_be_none(is_input_128782, None_128783)

        if may_be_128784:

            if more_types_in_union_128785:
                # Runtime conditional SSA (line 541)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 542):
            
            # Assigning a Name to a Subscript (line 542):
            # Getting the type of 'RIGHT' (line 542)
            RIGHT_128786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'RIGHT')
            # Getting the type of 'angles' (line 542)
            angles_128787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 20), 'angles')
            # Getting the type of 'i' (line 542)
            i_128788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 27), 'i')
            # Storing an element on a container (line 542)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 20), angles_128787, (i_128788, RIGHT_128786))

            if more_types_in_union_128785:
                # SSA join for if statement (line 541)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 540)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'orient' (line 544)
        orient_128789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 19), 'orient')
        int_128790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 29), 'int')
        # Applying the binary operator '!=' (line 544)
        result_ne_128791 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 19), '!=', orient_128789, int_128790)
        
        # Testing the type of an if condition (line 544)
        if_condition_128792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 16), result_ne_128791)
        # Assigning a type to the variable 'if_condition_128792' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'if_condition_128792', if_condition_128792)
        # SSA begins for if statement (line 544)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 545)
        # Processing the call arguments (line 545)
        unicode_128794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 20), 'unicode', u'The value of orientations[%d] is %d, but it must be [ -1 | 0 | 1 ].')
        
        # Obtaining an instance of the builtin type 'tuple' (line 547)
        tuple_128795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 547)
        # Adding element type (line 547)
        # Getting the type of 'i' (line 547)
        i_128796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 56), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 56), tuple_128795, i_128796)
        # Adding element type (line 547)
        # Getting the type of 'orient' (line 547)
        orient_128797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 59), 'orient', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 56), tuple_128795, orient_128797)
        
        # Applying the binary operator '%' (line 546)
        result_mod_128798 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 20), '%', unicode_128794, tuple_128795)
        
        # Processing the call keyword arguments (line 545)
        kwargs_128799 = {}
        # Getting the type of 'ValueError' (line 545)
        ValueError_128793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 545)
        ValueError_call_result_128800 = invoke(stypy.reporting.localization.Localization(__file__, 545, 26), ValueError_128793, *[result_mod_128798], **kwargs_128799)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 545, 20), ValueError_call_result_128800, 'raise parameter', BaseException)
        # SSA join for if statement (line 544)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'is_input' (line 548)
        is_input_128801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'is_input')
        # Testing the type of an if condition (line 548)
        if_condition_128802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 16), is_input_128801)
        # Assigning a type to the variable 'if_condition_128802' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'if_condition_128802', if_condition_128802)
        # SSA begins for if statement (line 548)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 549):
        
        # Assigning a Name to a Subscript (line 549):
        # Getting the type of 'UP' (line 549)
        UP_128803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 32), 'UP')
        # Getting the type of 'angles' (line 549)
        angles_128804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'angles')
        # Getting the type of 'i' (line 549)
        i_128805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 27), 'i')
        # Storing an element on a container (line 549)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 20), angles_128804, (i_128805, UP_128803))
        # SSA branch for the else part of an if statement (line 548)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'is_input' (line 550)
        is_input_128806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 25), 'is_input')
        # Applying the 'not' unary operator (line 550)
        result_not__128807 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 21), 'not', is_input_128806)
        
        # Testing the type of an if condition (line 550)
        if_condition_128808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 550, 21), result_not__128807)
        # Assigning a type to the variable 'if_condition_128808' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 21), 'if_condition_128808', if_condition_128808)
        # SSA begins for if statement (line 550)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 551):
        
        # Assigning a Name to a Subscript (line 551):
        # Getting the type of 'DOWN' (line 551)
        DOWN_128809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 32), 'DOWN')
        # Getting the type of 'angles' (line 551)
        angles_128810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'angles')
        # Getting the type of 'i' (line 551)
        i_128811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 27), 'i')
        # Storing an element on a container (line 551)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 20), angles_128810, (i_128811, DOWN_128809))
        # SSA join for if statement (line 550)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 548)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 540)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 534)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to iterable(...): (line 554)
        # Processing the call arguments (line 554)
        # Getting the type of 'pathlengths' (line 554)
        pathlengths_128813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'pathlengths', False)
        # Processing the call keyword arguments (line 554)
        kwargs_128814 = {}
        # Getting the type of 'iterable' (line 554)
        iterable_128812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'iterable', False)
        # Calling iterable(args, kwargs) (line 554)
        iterable_call_result_128815 = invoke(stypy.reporting.localization.Localization(__file__, 554, 11), iterable_128812, *[pathlengths_128813], **kwargs_128814)
        
        # Testing the type of an if condition (line 554)
        if_condition_128816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 8), iterable_call_result_128815)
        # Assigning a type to the variable 'if_condition_128816' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'if_condition_128816', if_condition_128816)
        # SSA begins for if statement (line 554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 555)
        # Processing the call arguments (line 555)
        # Getting the type of 'pathlengths' (line 555)
        pathlengths_128818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 'pathlengths', False)
        # Processing the call keyword arguments (line 555)
        kwargs_128819 = {}
        # Getting the type of 'len' (line 555)
        len_128817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'len', False)
        # Calling len(args, kwargs) (line 555)
        len_call_result_128820 = invoke(stypy.reporting.localization.Localization(__file__, 555, 15), len_128817, *[pathlengths_128818], **kwargs_128819)
        
        # Getting the type of 'n' (line 555)
        n_128821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 35), 'n')
        # Applying the binary operator '!=' (line 555)
        result_ne_128822 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 15), '!=', len_call_result_128820, n_128821)
        
        # Testing the type of an if condition (line 555)
        if_condition_128823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 12), result_ne_128822)
        # Assigning a type to the variable 'if_condition_128823' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'if_condition_128823', if_condition_128823)
        # SSA begins for if statement (line 555)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 556)
        # Processing the call arguments (line 556)
        unicode_128825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 16), 'unicode', u'If pathlengths is a list, then pathlengths and flows must have the same length.\npathlengths has length %d, but flows has length %d.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 559)
        tuple_128826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 559)
        # Adding element type (line 559)
        
        # Call to len(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'pathlengths' (line 559)
        pathlengths_128828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 40), 'pathlengths', False)
        # Processing the call keyword arguments (line 559)
        kwargs_128829 = {}
        # Getting the type of 'len' (line 559)
        len_128827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 36), 'len', False)
        # Calling len(args, kwargs) (line 559)
        len_call_result_128830 = invoke(stypy.reporting.localization.Localization(__file__, 559, 36), len_128827, *[pathlengths_128828], **kwargs_128829)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 36), tuple_128826, len_call_result_128830)
        # Adding element type (line 559)
        # Getting the type of 'n' (line 559)
        n_128831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 54), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 36), tuple_128826, n_128831)
        
        # Applying the binary operator '%' (line 557)
        result_mod_128832 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 16), '%', unicode_128825, tuple_128826)
        
        # Processing the call keyword arguments (line 556)
        kwargs_128833 = {}
        # Getting the type of 'ValueError' (line 556)
        ValueError_128824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 556)
        ValueError_call_result_128834 = invoke(stypy.reporting.localization.Localization(__file__, 556, 22), ValueError_128824, *[result_mod_128832], **kwargs_128833)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 556, 16), ValueError_call_result_128834, 'raise parameter', BaseException)
        # SSA join for if statement (line 555)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 554)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 561):
        
        # Assigning a Name to a Name (line 561):
        # Getting the type of 'pathlengths' (line 561)
        pathlengths_128835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 23), 'pathlengths')
        # Assigning a type to the variable 'urlength' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'urlength', pathlengths_128835)
        
        # Assigning a Name to a Name (line 562):
        
        # Assigning a Name to a Name (line 562):
        # Getting the type of 'pathlengths' (line 562)
        pathlengths_128836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 23), 'pathlengths')
        # Assigning a type to the variable 'ullength' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'ullength', pathlengths_128836)
        
        # Assigning a Name to a Name (line 563):
        
        # Assigning a Name to a Name (line 563):
        # Getting the type of 'pathlengths' (line 563)
        pathlengths_128837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 23), 'pathlengths')
        # Assigning a type to the variable 'lrlength' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'lrlength', pathlengths_128837)
        
        # Assigning a Name to a Name (line 564):
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'pathlengths' (line 564)
        pathlengths_128838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 23), 'pathlengths')
        # Assigning a type to the variable 'lllength' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'lllength', pathlengths_128838)
        
        # Assigning a Call to a Name (line 565):
        
        # Assigning a Call to a Name (line 565):
        
        # Call to dict(...): (line 565)
        # Processing the call keyword arguments (line 565)
        # Getting the type of 'pathlengths' (line 565)
        pathlengths_128840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 27), 'pathlengths', False)
        keyword_128841 = pathlengths_128840
        kwargs_128842 = {'RIGHT': keyword_128841}
        # Getting the type of 'dict' (line 565)
        dict_128839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'dict', False)
        # Calling dict(args, kwargs) (line 565)
        dict_call_result_128843 = invoke(stypy.reporting.localization.Localization(__file__, 565, 16), dict_128839, *[], **kwargs_128842)
        
        # Assigning a type to the variable 'd' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'd', dict_call_result_128843)
        
        # Assigning a ListComp to a Name (line 566):
        
        # Assigning a ListComp to a Name (line 566):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'angles' (line 566)
        angles_128850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 56), 'angles')
        comprehension_128851 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 27), angles_128850)
        # Assigning a type to the variable 'angle' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 27), 'angle', comprehension_128851)
        
        # Call to get(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'angle' (line 566)
        angle_128846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 33), 'angle', False)
        int_128847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 40), 'int')
        # Processing the call keyword arguments (line 566)
        kwargs_128848 = {}
        # Getting the type of 'd' (line 566)
        d_128844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 27), 'd', False)
        # Obtaining the member 'get' of a type (line 566)
        get_128845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 27), d_128844, 'get')
        # Calling get(args, kwargs) (line 566)
        get_call_result_128849 = invoke(stypy.reporting.localization.Localization(__file__, 566, 27), get_128845, *[angle_128846, int_128847], **kwargs_128848)
        
        list_128852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 27), list_128852, get_call_result_128849)
        # Assigning a type to the variable 'pathlengths' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'pathlengths', list_128852)
        
        
        # Call to enumerate(...): (line 569)
        # Processing the call arguments (line 569)
        
        # Call to zip(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'angles' (line 569)
        angles_128855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 60), 'angles', False)
        # Getting the type of 'are_inputs' (line 569)
        are_inputs_128856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 68), 'are_inputs', False)
        # Getting the type of 'scaled_flows' (line 570)
        scaled_flows_128857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 60), 'scaled_flows', False)
        # Processing the call keyword arguments (line 569)
        kwargs_128858 = {}
        # Getting the type of 'zip' (line 569)
        zip_128854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 56), 'zip', False)
        # Calling zip(args, kwargs) (line 569)
        zip_call_result_128859 = invoke(stypy.reporting.localization.Localization(__file__, 569, 56), zip_128854, *[angles_128855, are_inputs_128856, scaled_flows_128857], **kwargs_128858)
        
        # Processing the call keyword arguments (line 569)
        kwargs_128860 = {}
        # Getting the type of 'enumerate' (line 569)
        enumerate_128853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 46), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 569)
        enumerate_call_result_128861 = invoke(stypy.reporting.localization.Localization(__file__, 569, 46), enumerate_128853, *[zip_call_result_128859], **kwargs_128860)
        
        # Testing the type of a for loop iterable (line 569)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 569, 12), enumerate_call_result_128861)
        # Getting the type of the for loop variable (line 569)
        for_loop_var_128862 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 569, 12), enumerate_call_result_128861)
        # Assigning a type to the variable 'i' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 12), for_loop_var_128862))
        # Assigning a type to the variable 'angle' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 12), for_loop_var_128862))
        # Assigning a type to the variable 'is_input' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 12), for_loop_var_128862))
        # Assigning a type to the variable 'flow' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'flow', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 12), for_loop_var_128862))
        # SSA begins for a for statement (line 569)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 571)
        angle_128863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 19), 'angle')
        # Getting the type of 'DOWN' (line 571)
        DOWN_128864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 28), 'DOWN')
        # Applying the binary operator '==' (line 571)
        result_eq_128865 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 19), '==', angle_128863, DOWN_128864)
        
        # Getting the type of 'is_input' (line 571)
        is_input_128866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 37), 'is_input')
        # Applying the binary operator 'and' (line 571)
        result_and_keyword_128867 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 19), 'and', result_eq_128865, is_input_128866)
        
        # Testing the type of an if condition (line 571)
        if_condition_128868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 16), result_and_keyword_128867)
        # Assigning a type to the variable 'if_condition_128868' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'if_condition_128868', if_condition_128868)
        # SSA begins for if statement (line 571)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 572):
        
        # Assigning a Name to a Subscript (line 572):
        # Getting the type of 'ullength' (line 572)
        ullength_128869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 37), 'ullength')
        # Getting the type of 'pathlengths' (line 572)
        pathlengths_128870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 20), 'pathlengths')
        # Getting the type of 'i' (line 572)
        i_128871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 32), 'i')
        # Storing an element on a container (line 572)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 20), pathlengths_128870, (i_128871, ullength_128869))
        
        # Getting the type of 'ullength' (line 573)
        ullength_128872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 20), 'ullength')
        # Getting the type of 'flow' (line 573)
        flow_128873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 32), 'flow')
        # Applying the binary operator '+=' (line 573)
        result_iadd_128874 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 20), '+=', ullength_128872, flow_128873)
        # Assigning a type to the variable 'ullength' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 20), 'ullength', result_iadd_128874)
        
        # SSA branch for the else part of an if statement (line 571)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 574)
        angle_128875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 21), 'angle')
        # Getting the type of 'UP' (line 574)
        UP_128876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 30), 'UP')
        # Applying the binary operator '==' (line 574)
        result_eq_128877 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 21), '==', angle_128875, UP_128876)
        
        
        # Getting the type of 'is_input' (line 574)
        is_input_128878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 41), 'is_input')
        # Applying the 'not' unary operator (line 574)
        result_not__128879 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 37), 'not', is_input_128878)
        
        # Applying the binary operator 'and' (line 574)
        result_and_keyword_128880 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 21), 'and', result_eq_128877, result_not__128879)
        
        # Testing the type of an if condition (line 574)
        if_condition_128881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 21), result_and_keyword_128880)
        # Assigning a type to the variable 'if_condition_128881' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 21), 'if_condition_128881', if_condition_128881)
        # SSA begins for if statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 575):
        
        # Assigning a Name to a Subscript (line 575):
        # Getting the type of 'urlength' (line 575)
        urlength_128882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 37), 'urlength')
        # Getting the type of 'pathlengths' (line 575)
        pathlengths_128883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 20), 'pathlengths')
        # Getting the type of 'i' (line 575)
        i_128884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 32), 'i')
        # Storing an element on a container (line 575)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 20), pathlengths_128883, (i_128884, urlength_128882))
        
        # Getting the type of 'urlength' (line 576)
        urlength_128885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 20), 'urlength')
        # Getting the type of 'flow' (line 576)
        flow_128886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 32), 'flow')
        # Applying the binary operator '-=' (line 576)
        result_isub_128887 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 20), '-=', urlength_128885, flow_128886)
        # Assigning a type to the variable 'urlength' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 20), 'urlength', result_isub_128887)
        
        # SSA join for if statement (line 574)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 571)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 579)
        # Processing the call arguments (line 579)
        
        # Call to reversed(...): (line 579)
        # Processing the call arguments (line 579)
        
        # Call to list(...): (line 579)
        # Processing the call arguments (line 579)
        
        # Call to zip(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 'angles' (line 580)
        angles_128892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 18), 'angles', False)
        # Getting the type of 'are_inputs' (line 580)
        are_inputs_128893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 26), 'are_inputs', False)
        # Getting the type of 'scaled_flows' (line 580)
        scaled_flows_128894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 38), 'scaled_flows', False)
        # Processing the call keyword arguments (line 579)
        kwargs_128895 = {}
        # Getting the type of 'zip' (line 579)
        zip_128891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 70), 'zip', False)
        # Calling zip(args, kwargs) (line 579)
        zip_call_result_128896 = invoke(stypy.reporting.localization.Localization(__file__, 579, 70), zip_128891, *[angles_128892, are_inputs_128893, scaled_flows_128894], **kwargs_128895)
        
        # Processing the call keyword arguments (line 579)
        kwargs_128897 = {}
        # Getting the type of 'list' (line 579)
        list_128890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 65), 'list', False)
        # Calling list(args, kwargs) (line 579)
        list_call_result_128898 = invoke(stypy.reporting.localization.Localization(__file__, 579, 65), list_128890, *[zip_call_result_128896], **kwargs_128897)
        
        # Processing the call keyword arguments (line 579)
        kwargs_128899 = {}
        # Getting the type of 'reversed' (line 579)
        reversed_128889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 56), 'reversed', False)
        # Calling reversed(args, kwargs) (line 579)
        reversed_call_result_128900 = invoke(stypy.reporting.localization.Localization(__file__, 579, 56), reversed_128889, *[list_call_result_128898], **kwargs_128899)
        
        # Processing the call keyword arguments (line 579)
        kwargs_128901 = {}
        # Getting the type of 'enumerate' (line 579)
        enumerate_128888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 46), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 579)
        enumerate_call_result_128902 = invoke(stypy.reporting.localization.Localization(__file__, 579, 46), enumerate_128888, *[reversed_call_result_128900], **kwargs_128901)
        
        # Testing the type of a for loop iterable (line 579)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 579, 12), enumerate_call_result_128902)
        # Getting the type of the for loop variable (line 579)
        for_loop_var_128903 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 579, 12), enumerate_call_result_128902)
        # Assigning a type to the variable 'i' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 12), for_loop_var_128903))
        # Assigning a type to the variable 'angle' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 12), for_loop_var_128903))
        # Assigning a type to the variable 'is_input' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 12), for_loop_var_128903))
        # Assigning a type to the variable 'flow' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'flow', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 12), for_loop_var_128903))
        # SSA begins for a for statement (line 579)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 581)
        angle_128904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 19), 'angle')
        # Getting the type of 'UP' (line 581)
        UP_128905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 28), 'UP')
        # Applying the binary operator '==' (line 581)
        result_eq_128906 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 19), '==', angle_128904, UP_128905)
        
        # Getting the type of 'is_input' (line 581)
        is_input_128907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 35), 'is_input')
        # Applying the binary operator 'and' (line 581)
        result_and_keyword_128908 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 19), 'and', result_eq_128906, is_input_128907)
        
        # Testing the type of an if condition (line 581)
        if_condition_128909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 16), result_and_keyword_128908)
        # Assigning a type to the variable 'if_condition_128909' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'if_condition_128909', if_condition_128909)
        # SSA begins for if statement (line 581)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 582):
        
        # Assigning a Name to a Subscript (line 582):
        # Getting the type of 'lllength' (line 582)
        lllength_128910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 45), 'lllength')
        # Getting the type of 'pathlengths' (line 582)
        pathlengths_128911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 20), 'pathlengths')
        # Getting the type of 'n' (line 582)
        n_128912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 32), 'n')
        # Getting the type of 'i' (line 582)
        i_128913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 36), 'i')
        # Applying the binary operator '-' (line 582)
        result_sub_128914 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 32), '-', n_128912, i_128913)
        
        int_128915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 40), 'int')
        # Applying the binary operator '-' (line 582)
        result_sub_128916 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 38), '-', result_sub_128914, int_128915)
        
        # Storing an element on a container (line 582)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 20), pathlengths_128911, (result_sub_128916, lllength_128910))
        
        # Getting the type of 'lllength' (line 583)
        lllength_128917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'lllength')
        # Getting the type of 'flow' (line 583)
        flow_128918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 32), 'flow')
        # Applying the binary operator '+=' (line 583)
        result_iadd_128919 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 20), '+=', lllength_128917, flow_128918)
        # Assigning a type to the variable 'lllength' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'lllength', result_iadd_128919)
        
        # SSA branch for the else part of an if statement (line 581)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 584)
        angle_128920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 21), 'angle')
        # Getting the type of 'DOWN' (line 584)
        DOWN_128921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 30), 'DOWN')
        # Applying the binary operator '==' (line 584)
        result_eq_128922 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 21), '==', angle_128920, DOWN_128921)
        
        
        # Getting the type of 'is_input' (line 584)
        is_input_128923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 43), 'is_input')
        # Applying the 'not' unary operator (line 584)
        result_not__128924 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 39), 'not', is_input_128923)
        
        # Applying the binary operator 'and' (line 584)
        result_and_keyword_128925 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 21), 'and', result_eq_128922, result_not__128924)
        
        # Testing the type of an if condition (line 584)
        if_condition_128926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 584, 21), result_and_keyword_128925)
        # Assigning a type to the variable 'if_condition_128926' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 21), 'if_condition_128926', if_condition_128926)
        # SSA begins for if statement (line 584)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 585):
        
        # Assigning a Name to a Subscript (line 585):
        # Getting the type of 'lrlength' (line 585)
        lrlength_128927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 45), 'lrlength')
        # Getting the type of 'pathlengths' (line 585)
        pathlengths_128928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'pathlengths')
        # Getting the type of 'n' (line 585)
        n_128929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 32), 'n')
        # Getting the type of 'i' (line 585)
        i_128930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 36), 'i')
        # Applying the binary operator '-' (line 585)
        result_sub_128931 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 32), '-', n_128929, i_128930)
        
        int_128932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 40), 'int')
        # Applying the binary operator '-' (line 585)
        result_sub_128933 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 38), '-', result_sub_128931, int_128932)
        
        # Storing an element on a container (line 585)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 20), pathlengths_128928, (result_sub_128933, lrlength_128927))
        
        # Getting the type of 'lrlength' (line 586)
        lrlength_128934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'lrlength')
        # Getting the type of 'flow' (line 586)
        flow_128935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 32), 'flow')
        # Applying the binary operator '-=' (line 586)
        result_isub_128936 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 20), '-=', lrlength_128934, flow_128935)
        # Assigning a type to the variable 'lrlength' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'lrlength', result_isub_128936)
        
        # SSA join for if statement (line 584)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 581)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 589):
        
        # Assigning a Name to a Name (line 589):
        # Getting the type of 'False' (line 589)
        False_128937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 29), 'False')
        # Assigning a type to the variable 'has_left_input' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'has_left_input', False_128937)
        
        
        # Call to enumerate(...): (line 590)
        # Processing the call arguments (line 590)
        
        # Call to reversed(...): (line 590)
        # Processing the call arguments (line 590)
        
        # Call to list(...): (line 590)
        # Processing the call arguments (line 590)
        
        # Call to zip(...): (line 590)
        # Processing the call arguments (line 590)
        # Getting the type of 'angles' (line 591)
        angles_128942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 18), 'angles', False)
        # Getting the type of 'are_inputs' (line 591)
        are_inputs_128943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 26), 'are_inputs', False)
        
        # Call to zip(...): (line 591)
        # Processing the call arguments (line 591)
        # Getting the type of 'scaled_flows' (line 591)
        scaled_flows_128945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 42), 'scaled_flows', False)
        # Getting the type of 'pathlengths' (line 591)
        pathlengths_128946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 56), 'pathlengths', False)
        # Processing the call keyword arguments (line 591)
        kwargs_128947 = {}
        # Getting the type of 'zip' (line 591)
        zip_128944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 38), 'zip', False)
        # Calling zip(args, kwargs) (line 591)
        zip_call_result_128948 = invoke(stypy.reporting.localization.Localization(__file__, 591, 38), zip_128944, *[scaled_flows_128945, pathlengths_128946], **kwargs_128947)
        
        # Processing the call keyword arguments (line 590)
        kwargs_128949 = {}
        # Getting the type of 'zip' (line 590)
        zip_128941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 70), 'zip', False)
        # Calling zip(args, kwargs) (line 590)
        zip_call_result_128950 = invoke(stypy.reporting.localization.Localization(__file__, 590, 70), zip_128941, *[angles_128942, are_inputs_128943, zip_call_result_128948], **kwargs_128949)
        
        # Processing the call keyword arguments (line 590)
        kwargs_128951 = {}
        # Getting the type of 'list' (line 590)
        list_128940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 65), 'list', False)
        # Calling list(args, kwargs) (line 590)
        list_call_result_128952 = invoke(stypy.reporting.localization.Localization(__file__, 590, 65), list_128940, *[zip_call_result_128950], **kwargs_128951)
        
        # Processing the call keyword arguments (line 590)
        kwargs_128953 = {}
        # Getting the type of 'reversed' (line 590)
        reversed_128939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 56), 'reversed', False)
        # Calling reversed(args, kwargs) (line 590)
        reversed_call_result_128954 = invoke(stypy.reporting.localization.Localization(__file__, 590, 56), reversed_128939, *[list_call_result_128952], **kwargs_128953)
        
        # Processing the call keyword arguments (line 590)
        kwargs_128955 = {}
        # Getting the type of 'enumerate' (line 590)
        enumerate_128938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 46), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 590)
        enumerate_call_result_128956 = invoke(stypy.reporting.localization.Localization(__file__, 590, 46), enumerate_128938, *[reversed_call_result_128954], **kwargs_128955)
        
        # Testing the type of a for loop iterable (line 590)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 590, 12), enumerate_call_result_128956)
        # Getting the type of the for loop variable (line 590)
        for_loop_var_128957 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 590, 12), enumerate_call_result_128956)
        # Assigning a type to the variable 'i' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), for_loop_var_128957))
        # Assigning a type to the variable 'angle' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), for_loop_var_128957))
        # Assigning a type to the variable 'is_input' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), for_loop_var_128957))
        # Assigning a type to the variable 'spec' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), for_loop_var_128957))
        # SSA begins for a for statement (line 590)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'angle' (line 592)
        angle_128958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), 'angle')
        # Getting the type of 'RIGHT' (line 592)
        RIGHT_128959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 28), 'RIGHT')
        # Applying the binary operator '==' (line 592)
        result_eq_128960 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 19), '==', angle_128958, RIGHT_128959)
        
        # Testing the type of an if condition (line 592)
        if_condition_128961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 16), result_eq_128960)
        # Assigning a type to the variable 'if_condition_128961' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'if_condition_128961', if_condition_128961)
        # SSA begins for if statement (line 592)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'is_input' (line 593)
        is_input_128962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), 'is_input')
        # Testing the type of an if condition (line 593)
        if_condition_128963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 20), is_input_128962)
        # Assigning a type to the variable 'if_condition_128963' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 20), 'if_condition_128963', if_condition_128963)
        # SSA begins for if statement (line 593)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'has_left_input' (line 594)
        has_left_input_128964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 27), 'has_left_input')
        # Testing the type of an if condition (line 594)
        if_condition_128965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 594, 24), has_left_input_128964)
        # Assigning a type to the variable 'if_condition_128965' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 24), 'if_condition_128965', if_condition_128965)
        # SSA begins for if statement (line 594)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 595):
        
        # Assigning a Num to a Subscript (line 595):
        int_128966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 53), 'int')
        # Getting the type of 'pathlengths' (line 595)
        pathlengths_128967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 28), 'pathlengths')
        # Getting the type of 'n' (line 595)
        n_128968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 40), 'n')
        # Getting the type of 'i' (line 595)
        i_128969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 44), 'i')
        # Applying the binary operator '-' (line 595)
        result_sub_128970 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 40), '-', n_128968, i_128969)
        
        int_128971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 48), 'int')
        # Applying the binary operator '-' (line 595)
        result_sub_128972 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 46), '-', result_sub_128970, int_128971)
        
        # Storing an element on a container (line 595)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 28), pathlengths_128967, (result_sub_128972, int_128966))
        # SSA branch for the else part of an if statement (line 594)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 597):
        
        # Assigning a Name to a Name (line 597):
        # Getting the type of 'True' (line 597)
        True_128973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 45), 'True')
        # Assigning a type to the variable 'has_left_input' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 28), 'has_left_input', True_128973)
        # SSA join for if statement (line 594)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 593)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 592)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 600):
        
        # Assigning a Name to a Name (line 600):
        # Getting the type of 'False' (line 600)
        False_128974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 31), 'False')
        # Assigning a type to the variable 'has_right_output' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'has_right_output', False_128974)
        
        
        # Call to enumerate(...): (line 601)
        # Processing the call arguments (line 601)
        
        # Call to zip(...): (line 601)
        # Processing the call arguments (line 601)
        # Getting the type of 'angles' (line 602)
        angles_128977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 18), 'angles', False)
        # Getting the type of 'are_inputs' (line 602)
        are_inputs_128978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 26), 'are_inputs', False)
        
        # Call to list(...): (line 602)
        # Processing the call arguments (line 602)
        
        # Call to zip(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'scaled_flows' (line 602)
        scaled_flows_128981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 47), 'scaled_flows', False)
        # Getting the type of 'pathlengths' (line 602)
        pathlengths_128982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 61), 'pathlengths', False)
        # Processing the call keyword arguments (line 602)
        kwargs_128983 = {}
        # Getting the type of 'zip' (line 602)
        zip_128980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 43), 'zip', False)
        # Calling zip(args, kwargs) (line 602)
        zip_call_result_128984 = invoke(stypy.reporting.localization.Localization(__file__, 602, 43), zip_128980, *[scaled_flows_128981, pathlengths_128982], **kwargs_128983)
        
        # Processing the call keyword arguments (line 602)
        kwargs_128985 = {}
        # Getting the type of 'list' (line 602)
        list_128979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 38), 'list', False)
        # Calling list(args, kwargs) (line 602)
        list_call_result_128986 = invoke(stypy.reporting.localization.Localization(__file__, 602, 38), list_128979, *[zip_call_result_128984], **kwargs_128985)
        
        # Processing the call keyword arguments (line 601)
        kwargs_128987 = {}
        # Getting the type of 'zip' (line 601)
        zip_128976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 56), 'zip', False)
        # Calling zip(args, kwargs) (line 601)
        zip_call_result_128988 = invoke(stypy.reporting.localization.Localization(__file__, 601, 56), zip_128976, *[angles_128977, are_inputs_128978, list_call_result_128986], **kwargs_128987)
        
        # Processing the call keyword arguments (line 601)
        kwargs_128989 = {}
        # Getting the type of 'enumerate' (line 601)
        enumerate_128975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 46), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 601)
        enumerate_call_result_128990 = invoke(stypy.reporting.localization.Localization(__file__, 601, 46), enumerate_128975, *[zip_call_result_128988], **kwargs_128989)
        
        # Testing the type of a for loop iterable (line 601)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 601, 12), enumerate_call_result_128990)
        # Getting the type of the for loop variable (line 601)
        for_loop_var_128991 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 601, 12), enumerate_call_result_128990)
        # Assigning a type to the variable 'i' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 12), for_loop_var_128991))
        # Assigning a type to the variable 'angle' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 12), for_loop_var_128991))
        # Assigning a type to the variable 'is_input' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 12), for_loop_var_128991))
        # Assigning a type to the variable 'spec' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 12), for_loop_var_128991))
        # SSA begins for a for statement (line 601)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'angle' (line 603)
        angle_128992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 19), 'angle')
        # Getting the type of 'RIGHT' (line 603)
        RIGHT_128993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 28), 'RIGHT')
        # Applying the binary operator '==' (line 603)
        result_eq_128994 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 19), '==', angle_128992, RIGHT_128993)
        
        # Testing the type of an if condition (line 603)
        if_condition_128995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 16), result_eq_128994)
        # Assigning a type to the variable 'if_condition_128995' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 16), 'if_condition_128995', if_condition_128995)
        # SSA begins for if statement (line 603)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'is_input' (line 604)
        is_input_128996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 27), 'is_input')
        # Applying the 'not' unary operator (line 604)
        result_not__128997 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 23), 'not', is_input_128996)
        
        # Testing the type of an if condition (line 604)
        if_condition_128998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 20), result_not__128997)
        # Assigning a type to the variable 'if_condition_128998' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 20), 'if_condition_128998', if_condition_128998)
        # SSA begins for if statement (line 604)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'has_right_output' (line 605)
        has_right_output_128999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 27), 'has_right_output')
        # Testing the type of an if condition (line 605)
        if_condition_129000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 24), has_right_output_128999)
        # Assigning a type to the variable 'if_condition_129000' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 24), 'if_condition_129000', if_condition_129000)
        # SSA begins for if statement (line 605)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 606):
        
        # Assigning a Num to a Subscript (line 606):
        int_129001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 45), 'int')
        # Getting the type of 'pathlengths' (line 606)
        pathlengths_129002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 28), 'pathlengths')
        # Getting the type of 'i' (line 606)
        i_129003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 40), 'i')
        # Storing an element on a container (line 606)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 28), pathlengths_129002, (i_129003, int_129001))
        # SSA branch for the else part of an if statement (line 605)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 608):
        
        # Assigning a Name to a Name (line 608):
        # Getting the type of 'True' (line 608)
        True_129004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 47), 'True')
        # Assigning a type to the variable 'has_right_output' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 28), 'has_right_output', True_129004)
        # SSA join for if statement (line 605)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 604)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 603)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 554)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 612):
        
        # Assigning a List to a Name (line 612):
        
        # Obtaining an instance of the builtin type 'list' (line 612)
        list_129005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 612)
        # Adding element type (line 612)
        
        # Obtaining an instance of the builtin type 'tuple' (line 612)
        tuple_129006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 612)
        # Adding element type (line 612)
        # Getting the type of 'Path' (line 612)
        Path_129007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 19), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 612)
        MOVETO_129008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 19), Path_129007, 'MOVETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 19), tuple_129006, MOVETO_129008)
        # Adding element type (line 612)
        
        # Obtaining an instance of the builtin type 'list' (line 612)
        list_129009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 612)
        # Adding element type (line 612)
        # Getting the type of 'self' (line 612)
        self_129010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 34), 'self')
        # Obtaining the member 'gap' of a type (line 612)
        gap_129011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 34), self_129010, 'gap')
        # Getting the type of 'trunklength' (line 612)
        trunklength_129012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 45), 'trunklength')
        float_129013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 59), 'float')
        # Applying the binary operator 'div' (line 612)
        result_div_129014 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 45), 'div', trunklength_129012, float_129013)
        
        # Applying the binary operator '-' (line 612)
        result_sub_129015 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 34), '-', gap_129011, result_div_129014)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 32), list_129009, result_sub_129015)
        # Adding element type (line 612)
        # Getting the type of 'gain' (line 613)
        gain_129016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 33), 'gain')
        float_129017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 40), 'float')
        # Applying the binary operator 'div' (line 613)
        result_div_129018 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 33), 'div', gain_129016, float_129017)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 32), list_129009, result_div_129018)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 19), tuple_129006, list_129009)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 17), list_129005, tuple_129006)
        # Adding element type (line 612)
        
        # Obtaining an instance of the builtin type 'tuple' (line 614)
        tuple_129019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 614)
        # Adding element type (line 614)
        # Getting the type of 'Path' (line 614)
        Path_129020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 614)
        LINETO_129021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 19), Path_129020, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 19), tuple_129019, LINETO_129021)
        # Adding element type (line 614)
        
        # Obtaining an instance of the builtin type 'list' (line 614)
        list_129022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 614)
        # Adding element type (line 614)
        # Getting the type of 'self' (line 614)
        self_129023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 34), 'self')
        # Obtaining the member 'gap' of a type (line 614)
        gap_129024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 34), self_129023, 'gap')
        # Getting the type of 'trunklength' (line 614)
        trunklength_129025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 45), 'trunklength')
        float_129026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 59), 'float')
        # Applying the binary operator 'div' (line 614)
        result_div_129027 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 45), 'div', trunklength_129025, float_129026)
        
        # Applying the binary operator '-' (line 614)
        result_sub_129028 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 34), '-', gap_129024, result_div_129027)
        
        float_129029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 66), 'float')
        # Applying the binary operator 'div' (line 614)
        result_div_129030 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 33), 'div', result_sub_129028, float_129029)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 32), list_129022, result_div_129030)
        # Adding element type (line 614)
        # Getting the type of 'gain' (line 615)
        gain_129031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 33), 'gain')
        float_129032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 40), 'float')
        # Applying the binary operator 'div' (line 615)
        result_div_129033 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 33), 'div', gain_129031, float_129032)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 32), list_129022, result_div_129033)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 19), tuple_129019, list_129022)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 17), list_129005, tuple_129019)
        # Adding element type (line 612)
        
        # Obtaining an instance of the builtin type 'tuple' (line 616)
        tuple_129034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 616)
        # Adding element type (line 616)
        # Getting the type of 'Path' (line 616)
        Path_129035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 19), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 616)
        CURVE4_129036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 19), Path_129035, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 19), tuple_129034, CURVE4_129036)
        # Adding element type (line 616)
        
        # Obtaining an instance of the builtin type 'list' (line 616)
        list_129037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 616)
        # Adding element type (line 616)
        # Getting the type of 'self' (line 616)
        self_129038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 34), 'self')
        # Obtaining the member 'gap' of a type (line 616)
        gap_129039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 34), self_129038, 'gap')
        # Getting the type of 'trunklength' (line 616)
        trunklength_129040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 45), 'trunklength')
        float_129041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 59), 'float')
        # Applying the binary operator 'div' (line 616)
        result_div_129042 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 45), 'div', trunklength_129040, float_129041)
        
        # Applying the binary operator '-' (line 616)
        result_sub_129043 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 34), '-', gap_129039, result_div_129042)
        
        float_129044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 66), 'float')
        # Applying the binary operator 'div' (line 616)
        result_div_129045 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 33), 'div', result_sub_129043, float_129044)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 32), list_129037, result_div_129045)
        # Adding element type (line 616)
        # Getting the type of 'gain' (line 617)
        gain_129046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 33), 'gain')
        float_129047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 40), 'float')
        # Applying the binary operator 'div' (line 617)
        result_div_129048 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 33), 'div', gain_129046, float_129047)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 32), list_129037, result_div_129048)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 19), tuple_129034, list_129037)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 17), list_129005, tuple_129034)
        # Adding element type (line 612)
        
        # Obtaining an instance of the builtin type 'tuple' (line 618)
        tuple_129049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 618)
        # Adding element type (line 618)
        # Getting the type of 'Path' (line 618)
        Path_129050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 19), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 618)
        CURVE4_129051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 19), Path_129050, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 19), tuple_129049, CURVE4_129051)
        # Adding element type (line 618)
        
        # Obtaining an instance of the builtin type 'list' (line 618)
        list_129052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 618)
        # Adding element type (line 618)
        # Getting the type of 'trunklength' (line 618)
        trunklength_129053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 34), 'trunklength')
        float_129054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 48), 'float')
        # Applying the binary operator 'div' (line 618)
        result_div_129055 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 34), 'div', trunklength_129053, float_129054)
        
        # Getting the type of 'self' (line 618)
        self_129056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 54), 'self')
        # Obtaining the member 'gap' of a type (line 618)
        gap_129057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 54), self_129056, 'gap')
        # Applying the binary operator '-' (line 618)
        result_sub_129058 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 34), '-', result_div_129055, gap_129057)
        
        float_129059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 66), 'float')
        # Applying the binary operator 'div' (line 618)
        result_div_129060 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 33), 'div', result_sub_129058, float_129059)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 32), list_129052, result_div_129060)
        # Adding element type (line 618)
        
        # Getting the type of 'loss' (line 619)
        loss_129061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 34), 'loss')
        # Applying the 'usub' unary operator (line 619)
        result___neg___129062 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 33), 'usub', loss_129061)
        
        float_129063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 41), 'float')
        # Applying the binary operator 'div' (line 619)
        result_div_129064 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 33), 'div', result___neg___129062, float_129063)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 32), list_129052, result_div_129064)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 19), tuple_129049, list_129052)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 17), list_129005, tuple_129049)
        # Adding element type (line 612)
        
        # Obtaining an instance of the builtin type 'tuple' (line 620)
        tuple_129065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 620)
        # Adding element type (line 620)
        # Getting the type of 'Path' (line 620)
        Path_129066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 620)
        LINETO_129067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 19), Path_129066, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 19), tuple_129065, LINETO_129067)
        # Adding element type (line 620)
        
        # Obtaining an instance of the builtin type 'list' (line 620)
        list_129068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 620)
        # Adding element type (line 620)
        # Getting the type of 'trunklength' (line 620)
        trunklength_129069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 34), 'trunklength')
        float_129070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 48), 'float')
        # Applying the binary operator 'div' (line 620)
        result_div_129071 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 34), 'div', trunklength_129069, float_129070)
        
        # Getting the type of 'self' (line 620)
        self_129072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 54), 'self')
        # Obtaining the member 'gap' of a type (line 620)
        gap_129073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 54), self_129072, 'gap')
        # Applying the binary operator '-' (line 620)
        result_sub_129074 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 34), '-', result_div_129071, gap_129073)
        
        float_129075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 66), 'float')
        # Applying the binary operator 'div' (line 620)
        result_div_129076 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 33), 'div', result_sub_129074, float_129075)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 32), list_129068, result_div_129076)
        # Adding element type (line 620)
        
        # Getting the type of 'loss' (line 621)
        loss_129077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 34), 'loss')
        # Applying the 'usub' unary operator (line 621)
        result___neg___129078 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 33), 'usub', loss_129077)
        
        float_129079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 41), 'float')
        # Applying the binary operator 'div' (line 621)
        result_div_129080 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 33), 'div', result___neg___129078, float_129079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 32), list_129068, result_div_129080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 19), tuple_129065, list_129068)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 17), list_129005, tuple_129065)
        # Adding element type (line 612)
        
        # Obtaining an instance of the builtin type 'tuple' (line 622)
        tuple_129081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 622)
        # Adding element type (line 622)
        # Getting the type of 'Path' (line 622)
        Path_129082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 622)
        LINETO_129083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 19), Path_129082, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 19), tuple_129081, LINETO_129083)
        # Adding element type (line 622)
        
        # Obtaining an instance of the builtin type 'list' (line 622)
        list_129084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 622)
        # Adding element type (line 622)
        # Getting the type of 'trunklength' (line 622)
        trunklength_129085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 34), 'trunklength')
        float_129086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 48), 'float')
        # Applying the binary operator 'div' (line 622)
        result_div_129087 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 34), 'div', trunklength_129085, float_129086)
        
        # Getting the type of 'self' (line 622)
        self_129088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 54), 'self')
        # Obtaining the member 'gap' of a type (line 622)
        gap_129089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 54), self_129088, 'gap')
        # Applying the binary operator '-' (line 622)
        result_sub_129090 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 34), '-', result_div_129087, gap_129089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 32), list_129084, result_sub_129090)
        # Adding element type (line 622)
        
        # Getting the type of 'loss' (line 623)
        loss_129091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 34), 'loss')
        # Applying the 'usub' unary operator (line 623)
        result___neg___129092 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 33), 'usub', loss_129091)
        
        float_129093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 41), 'float')
        # Applying the binary operator 'div' (line 623)
        result_div_129094 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 33), 'div', result___neg___129092, float_129093)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 32), list_129084, result_div_129094)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 19), tuple_129081, list_129084)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 17), list_129005, tuple_129081)
        
        # Assigning a type to the variable 'urpath' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'urpath', list_129005)
        
        # Assigning a List to a Name (line 624):
        
        # Assigning a List to a Name (line 624):
        
        # Obtaining an instance of the builtin type 'list' (line 624)
        list_129095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 624)
        # Adding element type (line 624)
        
        # Obtaining an instance of the builtin type 'tuple' (line 624)
        tuple_129096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 624)
        # Adding element type (line 624)
        # Getting the type of 'Path' (line 624)
        Path_129097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 624)
        LINETO_129098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 19), Path_129097, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 19), tuple_129096, LINETO_129098)
        # Adding element type (line 624)
        
        # Obtaining an instance of the builtin type 'list' (line 624)
        list_129099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 624)
        # Adding element type (line 624)
        # Getting the type of 'trunklength' (line 624)
        trunklength_129100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 34), 'trunklength')
        float_129101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 48), 'float')
        # Applying the binary operator 'div' (line 624)
        result_div_129102 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 34), 'div', trunklength_129100, float_129101)
        
        # Getting the type of 'self' (line 624)
        self_129103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 54), 'self')
        # Obtaining the member 'gap' of a type (line 624)
        gap_129104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 54), self_129103, 'gap')
        # Applying the binary operator '-' (line 624)
        result_sub_129105 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 34), '-', result_div_129102, gap_129104)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 32), list_129099, result_sub_129105)
        # Adding element type (line 624)
        # Getting the type of 'loss' (line 625)
        loss_129106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 33), 'loss')
        float_129107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 40), 'float')
        # Applying the binary operator 'div' (line 625)
        result_div_129108 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 33), 'div', loss_129106, float_129107)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 32), list_129099, result_div_129108)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 19), tuple_129096, list_129099)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 17), list_129095, tuple_129096)
        # Adding element type (line 624)
        
        # Obtaining an instance of the builtin type 'tuple' (line 626)
        tuple_129109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 626)
        # Adding element type (line 626)
        # Getting the type of 'Path' (line 626)
        Path_129110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 626)
        LINETO_129111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 19), Path_129110, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 626, 19), tuple_129109, LINETO_129111)
        # Adding element type (line 626)
        
        # Obtaining an instance of the builtin type 'list' (line 626)
        list_129112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 626)
        # Adding element type (line 626)
        # Getting the type of 'trunklength' (line 626)
        trunklength_129113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 34), 'trunklength')
        float_129114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 48), 'float')
        # Applying the binary operator 'div' (line 626)
        result_div_129115 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 34), 'div', trunklength_129113, float_129114)
        
        # Getting the type of 'self' (line 626)
        self_129116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 54), 'self')
        # Obtaining the member 'gap' of a type (line 626)
        gap_129117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 54), self_129116, 'gap')
        # Applying the binary operator '-' (line 626)
        result_sub_129118 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 34), '-', result_div_129115, gap_129117)
        
        float_129119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 66), 'float')
        # Applying the binary operator 'div' (line 626)
        result_div_129120 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 33), 'div', result_sub_129118, float_129119)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 626, 32), list_129112, result_div_129120)
        # Adding element type (line 626)
        # Getting the type of 'loss' (line 627)
        loss_129121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 33), 'loss')
        float_129122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 40), 'float')
        # Applying the binary operator 'div' (line 627)
        result_div_129123 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 33), 'div', loss_129121, float_129122)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 626, 32), list_129112, result_div_129123)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 626, 19), tuple_129109, list_129112)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 17), list_129095, tuple_129109)
        # Adding element type (line 624)
        
        # Obtaining an instance of the builtin type 'tuple' (line 628)
        tuple_129124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 628)
        # Adding element type (line 628)
        # Getting the type of 'Path' (line 628)
        Path_129125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 19), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 628)
        CURVE4_129126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 19), Path_129125, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 19), tuple_129124, CURVE4_129126)
        # Adding element type (line 628)
        
        # Obtaining an instance of the builtin type 'list' (line 628)
        list_129127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 628)
        # Adding element type (line 628)
        # Getting the type of 'trunklength' (line 628)
        trunklength_129128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 34), 'trunklength')
        float_129129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 48), 'float')
        # Applying the binary operator 'div' (line 628)
        result_div_129130 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 34), 'div', trunklength_129128, float_129129)
        
        # Getting the type of 'self' (line 628)
        self_129131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 54), 'self')
        # Obtaining the member 'gap' of a type (line 628)
        gap_129132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 54), self_129131, 'gap')
        # Applying the binary operator '-' (line 628)
        result_sub_129133 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 34), '-', result_div_129130, gap_129132)
        
        float_129134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 66), 'float')
        # Applying the binary operator 'div' (line 628)
        result_div_129135 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 33), 'div', result_sub_129133, float_129134)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 32), list_129127, result_div_129135)
        # Adding element type (line 628)
        # Getting the type of 'loss' (line 629)
        loss_129136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 33), 'loss')
        float_129137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 40), 'float')
        # Applying the binary operator 'div' (line 629)
        result_div_129138 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 33), 'div', loss_129136, float_129137)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 32), list_129127, result_div_129138)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 19), tuple_129124, list_129127)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 17), list_129095, tuple_129124)
        # Adding element type (line 624)
        
        # Obtaining an instance of the builtin type 'tuple' (line 630)
        tuple_129139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 630)
        # Adding element type (line 630)
        # Getting the type of 'Path' (line 630)
        Path_129140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 19), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 630)
        CURVE4_129141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 19), Path_129140, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 19), tuple_129139, CURVE4_129141)
        # Adding element type (line 630)
        
        # Obtaining an instance of the builtin type 'list' (line 630)
        list_129142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 630)
        # Adding element type (line 630)
        # Getting the type of 'self' (line 630)
        self_129143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 34), 'self')
        # Obtaining the member 'gap' of a type (line 630)
        gap_129144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 34), self_129143, 'gap')
        # Getting the type of 'trunklength' (line 630)
        trunklength_129145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 45), 'trunklength')
        float_129146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 59), 'float')
        # Applying the binary operator 'div' (line 630)
        result_div_129147 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 45), 'div', trunklength_129145, float_129146)
        
        # Applying the binary operator '-' (line 630)
        result_sub_129148 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 34), '-', gap_129144, result_div_129147)
        
        float_129149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 66), 'float')
        # Applying the binary operator 'div' (line 630)
        result_div_129150 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 33), 'div', result_sub_129148, float_129149)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 32), list_129142, result_div_129150)
        # Adding element type (line 630)
        
        # Getting the type of 'gain' (line 631)
        gain_129151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 34), 'gain')
        # Applying the 'usub' unary operator (line 631)
        result___neg___129152 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 33), 'usub', gain_129151)
        
        float_129153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 41), 'float')
        # Applying the binary operator 'div' (line 631)
        result_div_129154 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 33), 'div', result___neg___129152, float_129153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 32), list_129142, result_div_129154)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 19), tuple_129139, list_129142)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 17), list_129095, tuple_129139)
        # Adding element type (line 624)
        
        # Obtaining an instance of the builtin type 'tuple' (line 632)
        tuple_129155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 632)
        # Adding element type (line 632)
        # Getting the type of 'Path' (line 632)
        Path_129156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 632)
        LINETO_129157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 19), Path_129156, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 19), tuple_129155, LINETO_129157)
        # Adding element type (line 632)
        
        # Obtaining an instance of the builtin type 'list' (line 632)
        list_129158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 632)
        # Adding element type (line 632)
        # Getting the type of 'self' (line 632)
        self_129159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 34), 'self')
        # Obtaining the member 'gap' of a type (line 632)
        gap_129160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 34), self_129159, 'gap')
        # Getting the type of 'trunklength' (line 632)
        trunklength_129161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 45), 'trunklength')
        float_129162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 59), 'float')
        # Applying the binary operator 'div' (line 632)
        result_div_129163 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 45), 'div', trunklength_129161, float_129162)
        
        # Applying the binary operator '-' (line 632)
        result_sub_129164 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 34), '-', gap_129160, result_div_129163)
        
        float_129165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 66), 'float')
        # Applying the binary operator 'div' (line 632)
        result_div_129166 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 33), 'div', result_sub_129164, float_129165)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 32), list_129158, result_div_129166)
        # Adding element type (line 632)
        
        # Getting the type of 'gain' (line 633)
        gain_129167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 34), 'gain')
        # Applying the 'usub' unary operator (line 633)
        result___neg___129168 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 33), 'usub', gain_129167)
        
        float_129169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 41), 'float')
        # Applying the binary operator 'div' (line 633)
        result_div_129170 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 33), 'div', result___neg___129168, float_129169)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 32), list_129158, result_div_129170)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 19), tuple_129155, list_129158)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 17), list_129095, tuple_129155)
        # Adding element type (line 624)
        
        # Obtaining an instance of the builtin type 'tuple' (line 634)
        tuple_129171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 634)
        # Adding element type (line 634)
        # Getting the type of 'Path' (line 634)
        Path_129172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 634)
        LINETO_129173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 19), Path_129172, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 19), tuple_129171, LINETO_129173)
        # Adding element type (line 634)
        
        # Obtaining an instance of the builtin type 'list' (line 634)
        list_129174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 634)
        # Adding element type (line 634)
        # Getting the type of 'self' (line 634)
        self_129175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 34), 'self')
        # Obtaining the member 'gap' of a type (line 634)
        gap_129176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 34), self_129175, 'gap')
        # Getting the type of 'trunklength' (line 634)
        trunklength_129177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 45), 'trunklength')
        float_129178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 59), 'float')
        # Applying the binary operator 'div' (line 634)
        result_div_129179 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 45), 'div', trunklength_129177, float_129178)
        
        # Applying the binary operator '-' (line 634)
        result_sub_129180 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 34), '-', gap_129176, result_div_129179)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 32), list_129174, result_sub_129180)
        # Adding element type (line 634)
        
        # Getting the type of 'gain' (line 635)
        gain_129181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 34), 'gain')
        # Applying the 'usub' unary operator (line 635)
        result___neg___129182 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 33), 'usub', gain_129181)
        
        float_129183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 41), 'float')
        # Applying the binary operator 'div' (line 635)
        result_div_129184 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 33), 'div', result___neg___129182, float_129183)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 32), list_129174, result_div_129184)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 19), tuple_129171, list_129174)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 17), list_129095, tuple_129171)
        
        # Assigning a type to the variable 'llpath' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'llpath', list_129095)
        
        # Assigning a List to a Name (line 636):
        
        # Assigning a List to a Name (line 636):
        
        # Obtaining an instance of the builtin type 'list' (line 636)
        list_129185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 636)
        # Adding element type (line 636)
        
        # Obtaining an instance of the builtin type 'tuple' (line 636)
        tuple_129186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 636)
        # Adding element type (line 636)
        # Getting the type of 'Path' (line 636)
        Path_129187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 636)
        LINETO_129188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 19), Path_129187, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 19), tuple_129186, LINETO_129188)
        # Adding element type (line 636)
        
        # Obtaining an instance of the builtin type 'list' (line 636)
        list_129189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 636)
        # Adding element type (line 636)
        # Getting the type of 'trunklength' (line 636)
        trunklength_129190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 34), 'trunklength')
        float_129191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 48), 'float')
        # Applying the binary operator 'div' (line 636)
        result_div_129192 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 34), 'div', trunklength_129190, float_129191)
        
        # Getting the type of 'self' (line 636)
        self_129193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 54), 'self')
        # Obtaining the member 'gap' of a type (line 636)
        gap_129194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 54), self_129193, 'gap')
        # Applying the binary operator '-' (line 636)
        result_sub_129195 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 34), '-', result_div_129192, gap_129194)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 32), list_129189, result_sub_129195)
        # Adding element type (line 636)
        # Getting the type of 'loss' (line 637)
        loss_129196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 33), 'loss')
        float_129197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 40), 'float')
        # Applying the binary operator 'div' (line 637)
        result_div_129198 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 33), 'div', loss_129196, float_129197)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 32), list_129189, result_div_129198)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 19), tuple_129186, list_129189)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 17), list_129185, tuple_129186)
        
        # Assigning a type to the variable 'lrpath' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'lrpath', list_129185)
        
        # Assigning a List to a Name (line 638):
        
        # Assigning a List to a Name (line 638):
        
        # Obtaining an instance of the builtin type 'list' (line 638)
        list_129199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 638)
        # Adding element type (line 638)
        
        # Obtaining an instance of the builtin type 'tuple' (line 638)
        tuple_129200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 638)
        # Adding element type (line 638)
        # Getting the type of 'Path' (line 638)
        Path_129201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 19), 'Path')
        # Obtaining the member 'LINETO' of a type (line 638)
        LINETO_129202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 19), Path_129201, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 19), tuple_129200, LINETO_129202)
        # Adding element type (line 638)
        
        # Obtaining an instance of the builtin type 'list' (line 638)
        list_129203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 638)
        # Adding element type (line 638)
        # Getting the type of 'self' (line 638)
        self_129204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 33), 'self')
        # Obtaining the member 'gap' of a type (line 638)
        gap_129205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 33), self_129204, 'gap')
        # Getting the type of 'trunklength' (line 638)
        trunklength_129206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 44), 'trunklength')
        float_129207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 58), 'float')
        # Applying the binary operator 'div' (line 638)
        result_div_129208 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 44), 'div', trunklength_129206, float_129207)
        
        # Applying the binary operator '-' (line 638)
        result_sub_129209 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 33), '-', gap_129205, result_div_129208)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 32), list_129203, result_sub_129209)
        # Adding element type (line 638)
        # Getting the type of 'gain' (line 639)
        gain_129210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 33), 'gain')
        float_129211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 40), 'float')
        # Applying the binary operator 'div' (line 639)
        result_div_129212 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 33), 'div', gain_129210, float_129211)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 32), list_129203, result_div_129212)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 19), tuple_129200, list_129203)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 17), list_129199, tuple_129200)
        
        # Assigning a type to the variable 'ulpath' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'ulpath', list_129199)
        
        # Assigning a Call to a Name (line 642):
        
        # Assigning a Call to a Name (line 642):
        
        # Call to zeros(...): (line 642)
        # Processing the call arguments (line 642)
        
        # Obtaining an instance of the builtin type 'tuple' (line 642)
        tuple_129215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 642)
        # Adding element type (line 642)
        # Getting the type of 'n' (line 642)
        n_129216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 25), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 25), tuple_129215, n_129216)
        # Adding element type (line 642)
        int_129217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 25), tuple_129215, int_129217)
        
        # Processing the call keyword arguments (line 642)
        kwargs_129218 = {}
        # Getting the type of 'np' (line 642)
        np_129213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 15), 'np', False)
        # Obtaining the member 'zeros' of a type (line 642)
        zeros_129214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 15), np_129213, 'zeros')
        # Calling zeros(args, kwargs) (line 642)
        zeros_call_result_129219 = invoke(stypy.reporting.localization.Localization(__file__, 642, 15), zeros_129214, *[tuple_129215], **kwargs_129218)
        
        # Assigning a type to the variable 'tips' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'tips', zeros_call_result_129219)
        
        # Assigning a Call to a Name (line 643):
        
        # Assigning a Call to a Name (line 643):
        
        # Call to zeros(...): (line 643)
        # Processing the call arguments (line 643)
        
        # Obtaining an instance of the builtin type 'tuple' (line 643)
        tuple_129222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 643)
        # Adding element type (line 643)
        # Getting the type of 'n' (line 643)
        n_129223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 36), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 36), tuple_129222, n_129223)
        # Adding element type (line 643)
        int_129224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 36), tuple_129222, int_129224)
        
        # Processing the call keyword arguments (line 643)
        kwargs_129225 = {}
        # Getting the type of 'np' (line 643)
        np_129220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 26), 'np', False)
        # Obtaining the member 'zeros' of a type (line 643)
        zeros_129221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 26), np_129220, 'zeros')
        # Calling zeros(args, kwargs) (line 643)
        zeros_call_result_129226 = invoke(stypy.reporting.localization.Localization(__file__, 643, 26), zeros_129221, *[tuple_129222], **kwargs_129225)
        
        # Assigning a type to the variable 'label_locations' (line 643)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'label_locations', zeros_call_result_129226)
        
        
        # Call to enumerate(...): (line 645)
        # Processing the call arguments (line 645)
        
        # Call to zip(...): (line 645)
        # Processing the call arguments (line 645)
        # Getting the type of 'angles' (line 646)
        angles_129229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 14), 'angles', False)
        # Getting the type of 'are_inputs' (line 646)
        are_inputs_129230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 22), 'are_inputs', False)
        
        # Call to list(...): (line 646)
        # Processing the call arguments (line 646)
        
        # Call to zip(...): (line 646)
        # Processing the call arguments (line 646)
        # Getting the type of 'scaled_flows' (line 646)
        scaled_flows_129233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 43), 'scaled_flows', False)
        # Getting the type of 'pathlengths' (line 646)
        pathlengths_129234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 57), 'pathlengths', False)
        # Processing the call keyword arguments (line 646)
        kwargs_129235 = {}
        # Getting the type of 'zip' (line 646)
        zip_129232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 39), 'zip', False)
        # Calling zip(args, kwargs) (line 646)
        zip_call_result_129236 = invoke(stypy.reporting.localization.Localization(__file__, 646, 39), zip_129232, *[scaled_flows_129233, pathlengths_129234], **kwargs_129235)
        
        # Processing the call keyword arguments (line 646)
        kwargs_129237 = {}
        # Getting the type of 'list' (line 646)
        list_129231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 34), 'list', False)
        # Calling list(args, kwargs) (line 646)
        list_call_result_129238 = invoke(stypy.reporting.localization.Localization(__file__, 646, 34), list_129231, *[zip_call_result_129236], **kwargs_129237)
        
        # Processing the call keyword arguments (line 645)
        kwargs_129239 = {}
        # Getting the type of 'zip' (line 645)
        zip_129228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 52), 'zip', False)
        # Calling zip(args, kwargs) (line 645)
        zip_call_result_129240 = invoke(stypy.reporting.localization.Localization(__file__, 645, 52), zip_129228, *[angles_129229, are_inputs_129230, list_call_result_129238], **kwargs_129239)
        
        # Processing the call keyword arguments (line 645)
        kwargs_129241 = {}
        # Getting the type of 'enumerate' (line 645)
        enumerate_129227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 42), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 645)
        enumerate_call_result_129242 = invoke(stypy.reporting.localization.Localization(__file__, 645, 42), enumerate_129227, *[zip_call_result_129240], **kwargs_129241)
        
        # Testing the type of a for loop iterable (line 645)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 645, 8), enumerate_call_result_129242)
        # Getting the type of the for loop variable (line 645)
        for_loop_var_129243 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 645, 8), enumerate_call_result_129242)
        # Assigning a type to the variable 'i' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 8), for_loop_var_129243))
        # Assigning a type to the variable 'angle' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 8), for_loop_var_129243))
        # Assigning a type to the variable 'is_input' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 8), for_loop_var_129243))
        # Assigning a type to the variable 'spec' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 8), for_loop_var_129243))
        # SSA begins for a for statement (line 645)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 647)
        angle_129244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'angle')
        # Getting the type of 'DOWN' (line 647)
        DOWN_129245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 24), 'DOWN')
        # Applying the binary operator '==' (line 647)
        result_eq_129246 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 15), '==', angle_129244, DOWN_129245)
        
        # Getting the type of 'is_input' (line 647)
        is_input_129247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 33), 'is_input')
        # Applying the binary operator 'and' (line 647)
        result_and_keyword_129248 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 15), 'and', result_eq_129246, is_input_129247)
        
        # Testing the type of an if condition (line 647)
        if_condition_129249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 647, 12), result_and_keyword_129248)
        # Assigning a type to the variable 'if_condition_129249' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'if_condition_129249', if_condition_129249)
        # SSA begins for if statement (line 647)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 648):
        
        # Assigning a Call to a Name:
        
        # Call to _add_input(...): (line 648)
        # Processing the call arguments (line 648)
        # Getting the type of 'ulpath' (line 649)
        ulpath_129252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 20), 'ulpath', False)
        # Getting the type of 'angle' (line 649)
        angle_129253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 28), 'angle', False)
        # Getting the type of 'spec' (line 649)
        spec_129254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 36), 'spec', False)
        # Processing the call keyword arguments (line 648)
        kwargs_129255 = {}
        # Getting the type of 'self' (line 648)
        self_129250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 52), 'self', False)
        # Obtaining the member '_add_input' of a type (line 648)
        _add_input_129251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 52), self_129250, '_add_input')
        # Calling _add_input(args, kwargs) (line 648)
        _add_input_call_result_129256 = invoke(stypy.reporting.localization.Localization(__file__, 648, 52), _add_input_129251, *[ulpath_129252, angle_129253, spec_129254], **kwargs_129255)
        
        # Assigning a type to the variable 'call_assignment_127435' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'call_assignment_127435', _add_input_call_result_129256)
        
        # Assigning a Call to a Name (line 648):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129260 = {}
        # Getting the type of 'call_assignment_127435' (line 648)
        call_assignment_127435_129257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'call_assignment_127435', False)
        # Obtaining the member '__getitem__' of a type (line 648)
        getitem___129258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 16), call_assignment_127435_129257, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129261 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129258, *[int_129259], **kwargs_129260)
        
        # Assigning a type to the variable 'call_assignment_127436' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'call_assignment_127436', getitem___call_result_129261)
        
        # Assigning a Name to a Subscript (line 648):
        # Getting the type of 'call_assignment_127436' (line 648)
        call_assignment_127436_129262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'call_assignment_127436')
        # Getting the type of 'tips' (line 648)
        tips_129263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'tips')
        # Getting the type of 'i' (line 648)
        i_129264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 21), 'i')
        slice_129265 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 648, 16), None, None, None)
        # Storing an element on a container (line 648)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 648, 16), tips_129263, ((i_129264, slice_129265), call_assignment_127436_129262))
        
        # Assigning a Call to a Name (line 648):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129269 = {}
        # Getting the type of 'call_assignment_127435' (line 648)
        call_assignment_127435_129266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'call_assignment_127435', False)
        # Obtaining the member '__getitem__' of a type (line 648)
        getitem___129267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 16), call_assignment_127435_129266, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129270 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129267, *[int_129268], **kwargs_129269)
        
        # Assigning a type to the variable 'call_assignment_127437' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'call_assignment_127437', getitem___call_result_129270)
        
        # Assigning a Name to a Subscript (line 648):
        # Getting the type of 'call_assignment_127437' (line 648)
        call_assignment_127437_129271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'call_assignment_127437')
        # Getting the type of 'label_locations' (line 648)
        label_locations_129272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 28), 'label_locations')
        # Getting the type of 'i' (line 648)
        i_129273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 44), 'i')
        slice_129274 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 648, 28), None, None, None)
        # Storing an element on a container (line 648)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 648, 28), label_locations_129272, ((i_129273, slice_129274), call_assignment_127437_129271))
        # SSA branch for the else part of an if statement (line 647)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 650)
        angle_129275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 17), 'angle')
        # Getting the type of 'UP' (line 650)
        UP_129276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 26), 'UP')
        # Applying the binary operator '==' (line 650)
        result_eq_129277 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 17), '==', angle_129275, UP_129276)
        
        
        # Getting the type of 'is_input' (line 650)
        is_input_129278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 37), 'is_input')
        # Applying the 'not' unary operator (line 650)
        result_not__129279 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 33), 'not', is_input_129278)
        
        # Applying the binary operator 'and' (line 650)
        result_and_keyword_129280 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 17), 'and', result_eq_129277, result_not__129279)
        
        # Testing the type of an if condition (line 650)
        if_condition_129281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 17), result_and_keyword_129280)
        # Assigning a type to the variable 'if_condition_129281' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 17), 'if_condition_129281', if_condition_129281)
        # SSA begins for if statement (line 650)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 651):
        
        # Assigning a Call to a Name:
        
        # Call to _add_output(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'urpath' (line 652)
        urpath_129284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 20), 'urpath', False)
        # Getting the type of 'angle' (line 652)
        angle_129285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 28), 'angle', False)
        # Getting the type of 'spec' (line 652)
        spec_129286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 36), 'spec', False)
        # Processing the call keyword arguments (line 651)
        kwargs_129287 = {}
        # Getting the type of 'self' (line 651)
        self_129282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 52), 'self', False)
        # Obtaining the member '_add_output' of a type (line 651)
        _add_output_129283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 52), self_129282, '_add_output')
        # Calling _add_output(args, kwargs) (line 651)
        _add_output_call_result_129288 = invoke(stypy.reporting.localization.Localization(__file__, 651, 52), _add_output_129283, *[urpath_129284, angle_129285, spec_129286], **kwargs_129287)
        
        # Assigning a type to the variable 'call_assignment_127438' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'call_assignment_127438', _add_output_call_result_129288)
        
        # Assigning a Call to a Name (line 651):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129292 = {}
        # Getting the type of 'call_assignment_127438' (line 651)
        call_assignment_127438_129289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'call_assignment_127438', False)
        # Obtaining the member '__getitem__' of a type (line 651)
        getitem___129290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 16), call_assignment_127438_129289, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129293 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129290, *[int_129291], **kwargs_129292)
        
        # Assigning a type to the variable 'call_assignment_127439' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'call_assignment_127439', getitem___call_result_129293)
        
        # Assigning a Name to a Subscript (line 651):
        # Getting the type of 'call_assignment_127439' (line 651)
        call_assignment_127439_129294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'call_assignment_127439')
        # Getting the type of 'tips' (line 651)
        tips_129295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'tips')
        # Getting the type of 'i' (line 651)
        i_129296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 21), 'i')
        slice_129297 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 651, 16), None, None, None)
        # Storing an element on a container (line 651)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 651, 16), tips_129295, ((i_129296, slice_129297), call_assignment_127439_129294))
        
        # Assigning a Call to a Name (line 651):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129301 = {}
        # Getting the type of 'call_assignment_127438' (line 651)
        call_assignment_127438_129298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'call_assignment_127438', False)
        # Obtaining the member '__getitem__' of a type (line 651)
        getitem___129299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 16), call_assignment_127438_129298, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129302 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129299, *[int_129300], **kwargs_129301)
        
        # Assigning a type to the variable 'call_assignment_127440' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'call_assignment_127440', getitem___call_result_129302)
        
        # Assigning a Name to a Subscript (line 651):
        # Getting the type of 'call_assignment_127440' (line 651)
        call_assignment_127440_129303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'call_assignment_127440')
        # Getting the type of 'label_locations' (line 651)
        label_locations_129304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 28), 'label_locations')
        # Getting the type of 'i' (line 651)
        i_129305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 44), 'i')
        slice_129306 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 651, 28), None, None, None)
        # Storing an element on a container (line 651)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 651, 28), label_locations_129304, ((i_129305, slice_129306), call_assignment_127440_129303))
        # SSA join for if statement (line 650)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 647)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 654)
        # Processing the call arguments (line 654)
        
        # Call to reversed(...): (line 654)
        # Processing the call arguments (line 654)
        
        # Call to list(...): (line 654)
        # Processing the call arguments (line 654)
        
        # Call to zip(...): (line 654)
        # Processing the call arguments (line 654)
        # Getting the type of 'angles' (line 655)
        angles_129311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 14), 'angles', False)
        # Getting the type of 'are_inputs' (line 655)
        are_inputs_129312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 22), 'are_inputs', False)
        
        # Call to list(...): (line 655)
        # Processing the call arguments (line 655)
        
        # Call to zip(...): (line 655)
        # Processing the call arguments (line 655)
        # Getting the type of 'scaled_flows' (line 655)
        scaled_flows_129315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 43), 'scaled_flows', False)
        # Getting the type of 'pathlengths' (line 655)
        pathlengths_129316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 57), 'pathlengths', False)
        # Processing the call keyword arguments (line 655)
        kwargs_129317 = {}
        # Getting the type of 'zip' (line 655)
        zip_129314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 39), 'zip', False)
        # Calling zip(args, kwargs) (line 655)
        zip_call_result_129318 = invoke(stypy.reporting.localization.Localization(__file__, 655, 39), zip_129314, *[scaled_flows_129315, pathlengths_129316], **kwargs_129317)
        
        # Processing the call keyword arguments (line 655)
        kwargs_129319 = {}
        # Getting the type of 'list' (line 655)
        list_129313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 34), 'list', False)
        # Calling list(args, kwargs) (line 655)
        list_call_result_129320 = invoke(stypy.reporting.localization.Localization(__file__, 655, 34), list_129313, *[zip_call_result_129318], **kwargs_129319)
        
        # Processing the call keyword arguments (line 654)
        kwargs_129321 = {}
        # Getting the type of 'zip' (line 654)
        zip_129310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 66), 'zip', False)
        # Calling zip(args, kwargs) (line 654)
        zip_call_result_129322 = invoke(stypy.reporting.localization.Localization(__file__, 654, 66), zip_129310, *[angles_129311, are_inputs_129312, list_call_result_129320], **kwargs_129321)
        
        # Processing the call keyword arguments (line 654)
        kwargs_129323 = {}
        # Getting the type of 'list' (line 654)
        list_129309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 61), 'list', False)
        # Calling list(args, kwargs) (line 654)
        list_call_result_129324 = invoke(stypy.reporting.localization.Localization(__file__, 654, 61), list_129309, *[zip_call_result_129322], **kwargs_129323)
        
        # Processing the call keyword arguments (line 654)
        kwargs_129325 = {}
        # Getting the type of 'reversed' (line 654)
        reversed_129308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 52), 'reversed', False)
        # Calling reversed(args, kwargs) (line 654)
        reversed_call_result_129326 = invoke(stypy.reporting.localization.Localization(__file__, 654, 52), reversed_129308, *[list_call_result_129324], **kwargs_129325)
        
        # Processing the call keyword arguments (line 654)
        kwargs_129327 = {}
        # Getting the type of 'enumerate' (line 654)
        enumerate_129307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 42), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 654)
        enumerate_call_result_129328 = invoke(stypy.reporting.localization.Localization(__file__, 654, 42), enumerate_129307, *[reversed_call_result_129326], **kwargs_129327)
        
        # Testing the type of a for loop iterable (line 654)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 654, 8), enumerate_call_result_129328)
        # Getting the type of the for loop variable (line 654)
        for_loop_var_129329 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 654, 8), enumerate_call_result_129328)
        # Assigning a type to the variable 'i' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 8), for_loop_var_129329))
        # Assigning a type to the variable 'angle' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 8), for_loop_var_129329))
        # Assigning a type to the variable 'is_input' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 8), for_loop_var_129329))
        # Assigning a type to the variable 'spec' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 8), for_loop_var_129329))
        # SSA begins for a for statement (line 654)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 656)
        angle_129330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 15), 'angle')
        # Getting the type of 'UP' (line 656)
        UP_129331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 24), 'UP')
        # Applying the binary operator '==' (line 656)
        result_eq_129332 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 15), '==', angle_129330, UP_129331)
        
        # Getting the type of 'is_input' (line 656)
        is_input_129333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 31), 'is_input')
        # Applying the binary operator 'and' (line 656)
        result_and_keyword_129334 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 15), 'and', result_eq_129332, is_input_129333)
        
        # Testing the type of an if condition (line 656)
        if_condition_129335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 656, 12), result_and_keyword_129334)
        # Assigning a type to the variable 'if_condition_129335' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'if_condition_129335', if_condition_129335)
        # SSA begins for if statement (line 656)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 657):
        
        # Assigning a Call to a Name:
        
        # Call to _add_input(...): (line 657)
        # Processing the call arguments (line 657)
        # Getting the type of 'llpath' (line 657)
        llpath_129338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 54), 'llpath', False)
        # Getting the type of 'angle' (line 657)
        angle_129339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 62), 'angle', False)
        # Getting the type of 'spec' (line 657)
        spec_129340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 70), 'spec', False)
        # Processing the call keyword arguments (line 657)
        kwargs_129341 = {}
        # Getting the type of 'self' (line 657)
        self_129336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 38), 'self', False)
        # Obtaining the member '_add_input' of a type (line 657)
        _add_input_129337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 38), self_129336, '_add_input')
        # Calling _add_input(args, kwargs) (line 657)
        _add_input_call_result_129342 = invoke(stypy.reporting.localization.Localization(__file__, 657, 38), _add_input_129337, *[llpath_129338, angle_129339, spec_129340], **kwargs_129341)
        
        # Assigning a type to the variable 'call_assignment_127441' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'call_assignment_127441', _add_input_call_result_129342)
        
        # Assigning a Call to a Name (line 657):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129346 = {}
        # Getting the type of 'call_assignment_127441' (line 657)
        call_assignment_127441_129343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'call_assignment_127441', False)
        # Obtaining the member '__getitem__' of a type (line 657)
        getitem___129344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 16), call_assignment_127441_129343, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129347 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129344, *[int_129345], **kwargs_129346)
        
        # Assigning a type to the variable 'call_assignment_127442' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'call_assignment_127442', getitem___call_result_129347)
        
        # Assigning a Name to a Name (line 657):
        # Getting the type of 'call_assignment_127442' (line 657)
        call_assignment_127442_129348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'call_assignment_127442')
        # Assigning a type to the variable 'tip' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'tip', call_assignment_127442_129348)
        
        # Assigning a Call to a Name (line 657):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129352 = {}
        # Getting the type of 'call_assignment_127441' (line 657)
        call_assignment_127441_129349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'call_assignment_127441', False)
        # Obtaining the member '__getitem__' of a type (line 657)
        getitem___129350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 16), call_assignment_127441_129349, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129353 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129350, *[int_129351], **kwargs_129352)
        
        # Assigning a type to the variable 'call_assignment_127443' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'call_assignment_127443', getitem___call_result_129353)
        
        # Assigning a Name to a Name (line 657):
        # Getting the type of 'call_assignment_127443' (line 657)
        call_assignment_127443_129354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'call_assignment_127443')
        # Assigning a type to the variable 'label_location' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 21), 'label_location', call_assignment_127443_129354)
        
        # Assigning a Name to a Subscript (line 658):
        
        # Assigning a Name to a Subscript (line 658):
        # Getting the type of 'tip' (line 658)
        tip_129355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 37), 'tip')
        # Getting the type of 'tips' (line 658)
        tips_129356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 'tips')
        # Getting the type of 'n' (line 658)
        n_129357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 21), 'n')
        # Getting the type of 'i' (line 658)
        i_129358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 25), 'i')
        # Applying the binary operator '-' (line 658)
        result_sub_129359 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 21), '-', n_129357, i_129358)
        
        int_129360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 29), 'int')
        # Applying the binary operator '-' (line 658)
        result_sub_129361 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 27), '-', result_sub_129359, int_129360)
        
        slice_129362 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 658, 16), None, None, None)
        # Storing an element on a container (line 658)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 16), tips_129356, ((result_sub_129361, slice_129362), tip_129355))
        
        # Assigning a Name to a Subscript (line 659):
        
        # Assigning a Name to a Subscript (line 659):
        # Getting the type of 'label_location' (line 659)
        label_location_129363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 48), 'label_location')
        # Getting the type of 'label_locations' (line 659)
        label_locations_129364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'label_locations')
        # Getting the type of 'n' (line 659)
        n_129365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 32), 'n')
        # Getting the type of 'i' (line 659)
        i_129366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 36), 'i')
        # Applying the binary operator '-' (line 659)
        result_sub_129367 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 32), '-', n_129365, i_129366)
        
        int_129368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 40), 'int')
        # Applying the binary operator '-' (line 659)
        result_sub_129369 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 38), '-', result_sub_129367, int_129368)
        
        slice_129370 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 659, 16), None, None, None)
        # Storing an element on a container (line 659)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 16), label_locations_129364, ((result_sub_129369, slice_129370), label_location_129363))
        # SSA branch for the else part of an if statement (line 656)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 660)
        angle_129371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 17), 'angle')
        # Getting the type of 'DOWN' (line 660)
        DOWN_129372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 26), 'DOWN')
        # Applying the binary operator '==' (line 660)
        result_eq_129373 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 17), '==', angle_129371, DOWN_129372)
        
        
        # Getting the type of 'is_input' (line 660)
        is_input_129374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 39), 'is_input')
        # Applying the 'not' unary operator (line 660)
        result_not__129375 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 35), 'not', is_input_129374)
        
        # Applying the binary operator 'and' (line 660)
        result_and_keyword_129376 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 17), 'and', result_eq_129373, result_not__129375)
        
        # Testing the type of an if condition (line 660)
        if_condition_129377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 660, 17), result_and_keyword_129376)
        # Assigning a type to the variable 'if_condition_129377' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 17), 'if_condition_129377', if_condition_129377)
        # SSA begins for if statement (line 660)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 661):
        
        # Assigning a Call to a Name:
        
        # Call to _add_output(...): (line 661)
        # Processing the call arguments (line 661)
        # Getting the type of 'lrpath' (line 661)
        lrpath_129380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 55), 'lrpath', False)
        # Getting the type of 'angle' (line 661)
        angle_129381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 63), 'angle', False)
        # Getting the type of 'spec' (line 661)
        spec_129382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 71), 'spec', False)
        # Processing the call keyword arguments (line 661)
        kwargs_129383 = {}
        # Getting the type of 'self' (line 661)
        self_129378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 38), 'self', False)
        # Obtaining the member '_add_output' of a type (line 661)
        _add_output_129379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 38), self_129378, '_add_output')
        # Calling _add_output(args, kwargs) (line 661)
        _add_output_call_result_129384 = invoke(stypy.reporting.localization.Localization(__file__, 661, 38), _add_output_129379, *[lrpath_129380, angle_129381, spec_129382], **kwargs_129383)
        
        # Assigning a type to the variable 'call_assignment_127444' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'call_assignment_127444', _add_output_call_result_129384)
        
        # Assigning a Call to a Name (line 661):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129388 = {}
        # Getting the type of 'call_assignment_127444' (line 661)
        call_assignment_127444_129385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'call_assignment_127444', False)
        # Obtaining the member '__getitem__' of a type (line 661)
        getitem___129386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 16), call_assignment_127444_129385, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129389 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129386, *[int_129387], **kwargs_129388)
        
        # Assigning a type to the variable 'call_assignment_127445' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'call_assignment_127445', getitem___call_result_129389)
        
        # Assigning a Name to a Name (line 661):
        # Getting the type of 'call_assignment_127445' (line 661)
        call_assignment_127445_129390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'call_assignment_127445')
        # Assigning a type to the variable 'tip' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'tip', call_assignment_127445_129390)
        
        # Assigning a Call to a Name (line 661):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129394 = {}
        # Getting the type of 'call_assignment_127444' (line 661)
        call_assignment_127444_129391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'call_assignment_127444', False)
        # Obtaining the member '__getitem__' of a type (line 661)
        getitem___129392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 16), call_assignment_127444_129391, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129395 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129392, *[int_129393], **kwargs_129394)
        
        # Assigning a type to the variable 'call_assignment_127446' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'call_assignment_127446', getitem___call_result_129395)
        
        # Assigning a Name to a Name (line 661):
        # Getting the type of 'call_assignment_127446' (line 661)
        call_assignment_127446_129396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'call_assignment_127446')
        # Assigning a type to the variable 'label_location' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 21), 'label_location', call_assignment_127446_129396)
        
        # Assigning a Name to a Subscript (line 662):
        
        # Assigning a Name to a Subscript (line 662):
        # Getting the type of 'tip' (line 662)
        tip_129397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 37), 'tip')
        # Getting the type of 'tips' (line 662)
        tips_129398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'tips')
        # Getting the type of 'n' (line 662)
        n_129399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 21), 'n')
        # Getting the type of 'i' (line 662)
        i_129400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 25), 'i')
        # Applying the binary operator '-' (line 662)
        result_sub_129401 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 21), '-', n_129399, i_129400)
        
        int_129402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 29), 'int')
        # Applying the binary operator '-' (line 662)
        result_sub_129403 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 27), '-', result_sub_129401, int_129402)
        
        slice_129404 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 662, 16), None, None, None)
        # Storing an element on a container (line 662)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 16), tips_129398, ((result_sub_129403, slice_129404), tip_129397))
        
        # Assigning a Name to a Subscript (line 663):
        
        # Assigning a Name to a Subscript (line 663):
        # Getting the type of 'label_location' (line 663)
        label_location_129405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 48), 'label_location')
        # Getting the type of 'label_locations' (line 663)
        label_locations_129406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'label_locations')
        # Getting the type of 'n' (line 663)
        n_129407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 32), 'n')
        # Getting the type of 'i' (line 663)
        i_129408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 36), 'i')
        # Applying the binary operator '-' (line 663)
        result_sub_129409 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 32), '-', n_129407, i_129408)
        
        int_129410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 40), 'int')
        # Applying the binary operator '-' (line 663)
        result_sub_129411 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 38), '-', result_sub_129409, int_129410)
        
        slice_129412 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 663, 16), None, None, None)
        # Storing an element on a container (line 663)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 16), label_locations_129406, ((result_sub_129411, slice_129412), label_location_129405))
        # SSA join for if statement (line 660)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 656)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 665):
        
        # Assigning a Name to a Name (line 665):
        # Getting the type of 'False' (line 665)
        False_129413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 25), 'False')
        # Assigning a type to the variable 'has_left_input' (line 665)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 8), 'has_left_input', False_129413)
        
        
        # Call to enumerate(...): (line 666)
        # Processing the call arguments (line 666)
        
        # Call to reversed(...): (line 666)
        # Processing the call arguments (line 666)
        
        # Call to list(...): (line 666)
        # Processing the call arguments (line 666)
        
        # Call to zip(...): (line 666)
        # Processing the call arguments (line 666)
        # Getting the type of 'angles' (line 667)
        angles_129418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 14), 'angles', False)
        # Getting the type of 'are_inputs' (line 667)
        are_inputs_129419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 22), 'are_inputs', False)
        
        # Call to list(...): (line 667)
        # Processing the call arguments (line 667)
        
        # Call to zip(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'scaled_flows' (line 667)
        scaled_flows_129422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 43), 'scaled_flows', False)
        # Getting the type of 'pathlengths' (line 667)
        pathlengths_129423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 57), 'pathlengths', False)
        # Processing the call keyword arguments (line 667)
        kwargs_129424 = {}
        # Getting the type of 'zip' (line 667)
        zip_129421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 39), 'zip', False)
        # Calling zip(args, kwargs) (line 667)
        zip_call_result_129425 = invoke(stypy.reporting.localization.Localization(__file__, 667, 39), zip_129421, *[scaled_flows_129422, pathlengths_129423], **kwargs_129424)
        
        # Processing the call keyword arguments (line 667)
        kwargs_129426 = {}
        # Getting the type of 'list' (line 667)
        list_129420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 34), 'list', False)
        # Calling list(args, kwargs) (line 667)
        list_call_result_129427 = invoke(stypy.reporting.localization.Localization(__file__, 667, 34), list_129420, *[zip_call_result_129425], **kwargs_129426)
        
        # Processing the call keyword arguments (line 666)
        kwargs_129428 = {}
        # Getting the type of 'zip' (line 666)
        zip_129417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 66), 'zip', False)
        # Calling zip(args, kwargs) (line 666)
        zip_call_result_129429 = invoke(stypy.reporting.localization.Localization(__file__, 666, 66), zip_129417, *[angles_129418, are_inputs_129419, list_call_result_129427], **kwargs_129428)
        
        # Processing the call keyword arguments (line 666)
        kwargs_129430 = {}
        # Getting the type of 'list' (line 666)
        list_129416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 61), 'list', False)
        # Calling list(args, kwargs) (line 666)
        list_call_result_129431 = invoke(stypy.reporting.localization.Localization(__file__, 666, 61), list_129416, *[zip_call_result_129429], **kwargs_129430)
        
        # Processing the call keyword arguments (line 666)
        kwargs_129432 = {}
        # Getting the type of 'reversed' (line 666)
        reversed_129415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 52), 'reversed', False)
        # Calling reversed(args, kwargs) (line 666)
        reversed_call_result_129433 = invoke(stypy.reporting.localization.Localization(__file__, 666, 52), reversed_129415, *[list_call_result_129431], **kwargs_129432)
        
        # Processing the call keyword arguments (line 666)
        kwargs_129434 = {}
        # Getting the type of 'enumerate' (line 666)
        enumerate_129414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 42), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 666)
        enumerate_call_result_129435 = invoke(stypy.reporting.localization.Localization(__file__, 666, 42), enumerate_129414, *[reversed_call_result_129433], **kwargs_129434)
        
        # Testing the type of a for loop iterable (line 666)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 666, 8), enumerate_call_result_129435)
        # Getting the type of the for loop variable (line 666)
        for_loop_var_129436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 666, 8), enumerate_call_result_129435)
        # Assigning a type to the variable 'i' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 8), for_loop_var_129436))
        # Assigning a type to the variable 'angle' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 8), for_loop_var_129436))
        # Assigning a type to the variable 'is_input' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 8), for_loop_var_129436))
        # Assigning a type to the variable 'spec' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 8), for_loop_var_129436))
        # SSA begins for a for statement (line 666)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 668)
        angle_129437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'angle')
        # Getting the type of 'RIGHT' (line 668)
        RIGHT_129438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 24), 'RIGHT')
        # Applying the binary operator '==' (line 668)
        result_eq_129439 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 15), '==', angle_129437, RIGHT_129438)
        
        # Getting the type of 'is_input' (line 668)
        is_input_129440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 34), 'is_input')
        # Applying the binary operator 'and' (line 668)
        result_and_keyword_129441 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 15), 'and', result_eq_129439, is_input_129440)
        
        # Testing the type of an if condition (line 668)
        if_condition_129442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 12), result_and_keyword_129441)
        # Assigning a type to the variable 'if_condition_129442' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'if_condition_129442', if_condition_129442)
        # SSA begins for if statement (line 668)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'has_left_input' (line 669)
        has_left_input_129443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 23), 'has_left_input')
        # Applying the 'not' unary operator (line 669)
        result_not__129444 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 19), 'not', has_left_input_129443)
        
        # Testing the type of an if condition (line 669)
        if_condition_129445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 16), result_not__129444)
        # Assigning a type to the variable 'if_condition_129445' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 16), 'if_condition_129445', if_condition_129445)
        # SSA begins for if statement (line 669)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_129446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 37), 'int')
        
        # Obtaining the type of the subscript
        int_129447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 34), 'int')
        
        # Obtaining the type of the subscript
        int_129448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 30), 'int')
        # Getting the type of 'llpath' (line 672)
        llpath_129449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 23), 'llpath')
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___129450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 23), llpath_129449, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_129451 = invoke(stypy.reporting.localization.Localization(__file__, 672, 23), getitem___129450, int_129448)
        
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___129452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 23), subscript_call_result_129451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_129453 = invoke(stypy.reporting.localization.Localization(__file__, 672, 23), getitem___129452, int_129447)
        
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___129454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 23), subscript_call_result_129453, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_129455 = invoke(stypy.reporting.localization.Localization(__file__, 672, 23), getitem___129454, int_129446)
        
        
        # Obtaining the type of the subscript
        int_129456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 56), 'int')
        
        # Obtaining the type of the subscript
        int_129457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 53), 'int')
        
        # Obtaining the type of the subscript
        int_129458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 49), 'int')
        # Getting the type of 'ulpath' (line 672)
        ulpath_129459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 42), 'ulpath')
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___129460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 42), ulpath_129459, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_129461 = invoke(stypy.reporting.localization.Localization(__file__, 672, 42), getitem___129460, int_129458)
        
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___129462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 42), subscript_call_result_129461, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_129463 = invoke(stypy.reporting.localization.Localization(__file__, 672, 42), getitem___129462, int_129457)
        
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___129464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 42), subscript_call_result_129463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_129465 = invoke(stypy.reporting.localization.Localization(__file__, 672, 42), getitem___129464, int_129456)
        
        # Applying the binary operator '>' (line 672)
        result_gt_129466 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 23), '>', subscript_call_result_129455, subscript_call_result_129465)
        
        # Testing the type of an if condition (line 672)
        if_condition_129467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 672, 20), result_gt_129466)
        # Assigning a type to the variable 'if_condition_129467' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 20), 'if_condition_129467', if_condition_129467)
        # SSA begins for if statement (line 672)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 673)
        # Processing the call arguments (line 673)
        
        # Obtaining an instance of the builtin type 'tuple' (line 673)
        tuple_129470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 673)
        # Adding element type (line 673)
        # Getting the type of 'Path' (line 673)
        Path_129471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 39), 'Path', False)
        # Obtaining the member 'LINETO' of a type (line 673)
        LINETO_129472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 39), Path_129471, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 39), tuple_129470, LINETO_129472)
        # Adding element type (line 673)
        
        # Obtaining an instance of the builtin type 'list' (line 673)
        list_129473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 673)
        # Adding element type (line 673)
        
        # Obtaining the type of the subscript
        int_129474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 67), 'int')
        
        # Obtaining the type of the subscript
        int_129475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 64), 'int')
        
        # Obtaining the type of the subscript
        int_129476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 60), 'int')
        # Getting the type of 'ulpath' (line 673)
        ulpath_129477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 53), 'ulpath', False)
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___129478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 53), ulpath_129477, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 673)
        subscript_call_result_129479 = invoke(stypy.reporting.localization.Localization(__file__, 673, 53), getitem___129478, int_129476)
        
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___129480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 53), subscript_call_result_129479, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 673)
        subscript_call_result_129481 = invoke(stypy.reporting.localization.Localization(__file__, 673, 53), getitem___129480, int_129475)
        
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___129482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 53), subscript_call_result_129481, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 673)
        subscript_call_result_129483 = invoke(stypy.reporting.localization.Localization(__file__, 673, 53), getitem___129482, int_129474)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 52), list_129473, subscript_call_result_129483)
        # Adding element type (line 673)
        
        # Obtaining the type of the subscript
        int_129484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 67), 'int')
        
        # Obtaining the type of the subscript
        int_129485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 64), 'int')
        
        # Obtaining the type of the subscript
        int_129486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 60), 'int')
        # Getting the type of 'llpath' (line 674)
        llpath_129487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 53), 'llpath', False)
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___129488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 53), llpath_129487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_129489 = invoke(stypy.reporting.localization.Localization(__file__, 674, 53), getitem___129488, int_129486)
        
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___129490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 53), subscript_call_result_129489, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_129491 = invoke(stypy.reporting.localization.Localization(__file__, 674, 53), getitem___129490, int_129485)
        
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___129492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 53), subscript_call_result_129491, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_129493 = invoke(stypy.reporting.localization.Localization(__file__, 674, 53), getitem___129492, int_129484)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 52), list_129473, subscript_call_result_129493)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 39), tuple_129470, list_129473)
        
        # Processing the call keyword arguments (line 673)
        kwargs_129494 = {}
        # Getting the type of 'llpath' (line 673)
        llpath_129468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 24), 'llpath', False)
        # Obtaining the member 'append' of a type (line 673)
        append_129469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 24), llpath_129468, 'append')
        # Calling append(args, kwargs) (line 673)
        append_call_result_129495 = invoke(stypy.reporting.localization.Localization(__file__, 673, 24), append_129469, *[tuple_129470], **kwargs_129494)
        
        # SSA join for if statement (line 672)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 675):
        
        # Assigning a Name to a Name (line 675):
        # Getting the type of 'True' (line 675)
        True_129496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 37), 'True')
        # Assigning a type to the variable 'has_left_input' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 20), 'has_left_input', True_129496)
        # SSA join for if statement (line 669)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 676):
        
        # Assigning a Call to a Name:
        
        # Call to _add_input(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'llpath' (line 676)
        llpath_129499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 54), 'llpath', False)
        # Getting the type of 'angle' (line 676)
        angle_129500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 62), 'angle', False)
        # Getting the type of 'spec' (line 676)
        spec_129501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 70), 'spec', False)
        # Processing the call keyword arguments (line 676)
        kwargs_129502 = {}
        # Getting the type of 'self' (line 676)
        self_129497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 38), 'self', False)
        # Obtaining the member '_add_input' of a type (line 676)
        _add_input_129498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 38), self_129497, '_add_input')
        # Calling _add_input(args, kwargs) (line 676)
        _add_input_call_result_129503 = invoke(stypy.reporting.localization.Localization(__file__, 676, 38), _add_input_129498, *[llpath_129499, angle_129500, spec_129501], **kwargs_129502)
        
        # Assigning a type to the variable 'call_assignment_127447' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'call_assignment_127447', _add_input_call_result_129503)
        
        # Assigning a Call to a Name (line 676):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129507 = {}
        # Getting the type of 'call_assignment_127447' (line 676)
        call_assignment_127447_129504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'call_assignment_127447', False)
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___129505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), call_assignment_127447_129504, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129508 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129505, *[int_129506], **kwargs_129507)
        
        # Assigning a type to the variable 'call_assignment_127448' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'call_assignment_127448', getitem___call_result_129508)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'call_assignment_127448' (line 676)
        call_assignment_127448_129509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'call_assignment_127448')
        # Assigning a type to the variable 'tip' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tip', call_assignment_127448_129509)
        
        # Assigning a Call to a Name (line 676):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129513 = {}
        # Getting the type of 'call_assignment_127447' (line 676)
        call_assignment_127447_129510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'call_assignment_127447', False)
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___129511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), call_assignment_127447_129510, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129514 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129511, *[int_129512], **kwargs_129513)
        
        # Assigning a type to the variable 'call_assignment_127449' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'call_assignment_127449', getitem___call_result_129514)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'call_assignment_127449' (line 676)
        call_assignment_127449_129515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'call_assignment_127449')
        # Assigning a type to the variable 'label_location' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 21), 'label_location', call_assignment_127449_129515)
        
        # Assigning a Name to a Subscript (line 677):
        
        # Assigning a Name to a Subscript (line 677):
        # Getting the type of 'tip' (line 677)
        tip_129516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 37), 'tip')
        # Getting the type of 'tips' (line 677)
        tips_129517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'tips')
        # Getting the type of 'n' (line 677)
        n_129518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 21), 'n')
        # Getting the type of 'i' (line 677)
        i_129519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 25), 'i')
        # Applying the binary operator '-' (line 677)
        result_sub_129520 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 21), '-', n_129518, i_129519)
        
        int_129521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 29), 'int')
        # Applying the binary operator '-' (line 677)
        result_sub_129522 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 27), '-', result_sub_129520, int_129521)
        
        slice_129523 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 677, 16), None, None, None)
        # Storing an element on a container (line 677)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 16), tips_129517, ((result_sub_129522, slice_129523), tip_129516))
        
        # Assigning a Name to a Subscript (line 678):
        
        # Assigning a Name to a Subscript (line 678):
        # Getting the type of 'label_location' (line 678)
        label_location_129524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 48), 'label_location')
        # Getting the type of 'label_locations' (line 678)
        label_locations_129525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 16), 'label_locations')
        # Getting the type of 'n' (line 678)
        n_129526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 32), 'n')
        # Getting the type of 'i' (line 678)
        i_129527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 36), 'i')
        # Applying the binary operator '-' (line 678)
        result_sub_129528 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 32), '-', n_129526, i_129527)
        
        int_129529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 40), 'int')
        # Applying the binary operator '-' (line 678)
        result_sub_129530 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 38), '-', result_sub_129528, int_129529)
        
        slice_129531 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 678, 16), None, None, None)
        # Storing an element on a container (line 678)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 16), label_locations_129525, ((result_sub_129530, slice_129531), label_location_129524))
        # SSA join for if statement (line 668)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 680):
        
        # Assigning a Name to a Name (line 680):
        # Getting the type of 'False' (line 680)
        False_129532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 27), 'False')
        # Assigning a type to the variable 'has_right_output' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'has_right_output', False_129532)
        
        
        # Call to enumerate(...): (line 681)
        # Processing the call arguments (line 681)
        
        # Call to zip(...): (line 681)
        # Processing the call arguments (line 681)
        # Getting the type of 'angles' (line 682)
        angles_129535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 14), 'angles', False)
        # Getting the type of 'are_inputs' (line 682)
        are_inputs_129536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 22), 'are_inputs', False)
        
        # Call to list(...): (line 682)
        # Processing the call arguments (line 682)
        
        # Call to zip(...): (line 682)
        # Processing the call arguments (line 682)
        # Getting the type of 'scaled_flows' (line 682)
        scaled_flows_129539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 43), 'scaled_flows', False)
        # Getting the type of 'pathlengths' (line 682)
        pathlengths_129540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 57), 'pathlengths', False)
        # Processing the call keyword arguments (line 682)
        kwargs_129541 = {}
        # Getting the type of 'zip' (line 682)
        zip_129538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 39), 'zip', False)
        # Calling zip(args, kwargs) (line 682)
        zip_call_result_129542 = invoke(stypy.reporting.localization.Localization(__file__, 682, 39), zip_129538, *[scaled_flows_129539, pathlengths_129540], **kwargs_129541)
        
        # Processing the call keyword arguments (line 682)
        kwargs_129543 = {}
        # Getting the type of 'list' (line 682)
        list_129537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 34), 'list', False)
        # Calling list(args, kwargs) (line 682)
        list_call_result_129544 = invoke(stypy.reporting.localization.Localization(__file__, 682, 34), list_129537, *[zip_call_result_129542], **kwargs_129543)
        
        # Processing the call keyword arguments (line 681)
        kwargs_129545 = {}
        # Getting the type of 'zip' (line 681)
        zip_129534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 52), 'zip', False)
        # Calling zip(args, kwargs) (line 681)
        zip_call_result_129546 = invoke(stypy.reporting.localization.Localization(__file__, 681, 52), zip_129534, *[angles_129535, are_inputs_129536, list_call_result_129544], **kwargs_129545)
        
        # Processing the call keyword arguments (line 681)
        kwargs_129547 = {}
        # Getting the type of 'enumerate' (line 681)
        enumerate_129533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 42), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 681)
        enumerate_call_result_129548 = invoke(stypy.reporting.localization.Localization(__file__, 681, 42), enumerate_129533, *[zip_call_result_129546], **kwargs_129547)
        
        # Testing the type of a for loop iterable (line 681)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 681, 8), enumerate_call_result_129548)
        # Getting the type of the for loop variable (line 681)
        for_loop_var_129549 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 681, 8), enumerate_call_result_129548)
        # Assigning a type to the variable 'i' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 8), for_loop_var_129549))
        # Assigning a type to the variable 'angle' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 8), for_loop_var_129549))
        # Assigning a type to the variable 'is_input' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'is_input', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 8), for_loop_var_129549))
        # Assigning a type to the variable 'spec' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 8), for_loop_var_129549))
        # SSA begins for a for statement (line 681)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'angle' (line 683)
        angle_129550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 15), 'angle')
        # Getting the type of 'RIGHT' (line 683)
        RIGHT_129551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 24), 'RIGHT')
        # Applying the binary operator '==' (line 683)
        result_eq_129552 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 15), '==', angle_129550, RIGHT_129551)
        
        
        # Getting the type of 'is_input' (line 683)
        is_input_129553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 38), 'is_input')
        # Applying the 'not' unary operator (line 683)
        result_not__129554 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 34), 'not', is_input_129553)
        
        # Applying the binary operator 'and' (line 683)
        result_and_keyword_129555 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 15), 'and', result_eq_129552, result_not__129554)
        
        # Testing the type of an if condition (line 683)
        if_condition_129556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 12), result_and_keyword_129555)
        # Assigning a type to the variable 'if_condition_129556' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'if_condition_129556', if_condition_129556)
        # SSA begins for if statement (line 683)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'has_right_output' (line 684)
        has_right_output_129557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 23), 'has_right_output')
        # Applying the 'not' unary operator (line 684)
        result_not__129558 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 19), 'not', has_right_output_129557)
        
        # Testing the type of an if condition (line 684)
        if_condition_129559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 684, 16), result_not__129558)
        # Assigning a type to the variable 'if_condition_129559' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 16), 'if_condition_129559', if_condition_129559)
        # SSA begins for if statement (line 684)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_129560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 37), 'int')
        
        # Obtaining the type of the subscript
        int_129561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 34), 'int')
        
        # Obtaining the type of the subscript
        int_129562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 30), 'int')
        # Getting the type of 'urpath' (line 687)
        urpath_129563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 23), 'urpath')
        # Obtaining the member '__getitem__' of a type (line 687)
        getitem___129564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 23), urpath_129563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 687)
        subscript_call_result_129565 = invoke(stypy.reporting.localization.Localization(__file__, 687, 23), getitem___129564, int_129562)
        
        # Obtaining the member '__getitem__' of a type (line 687)
        getitem___129566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 23), subscript_call_result_129565, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 687)
        subscript_call_result_129567 = invoke(stypy.reporting.localization.Localization(__file__, 687, 23), getitem___129566, int_129561)
        
        # Obtaining the member '__getitem__' of a type (line 687)
        getitem___129568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 23), subscript_call_result_129567, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 687)
        subscript_call_result_129569 = invoke(stypy.reporting.localization.Localization(__file__, 687, 23), getitem___129568, int_129560)
        
        
        # Obtaining the type of the subscript
        int_129570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 56), 'int')
        
        # Obtaining the type of the subscript
        int_129571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 53), 'int')
        
        # Obtaining the type of the subscript
        int_129572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 49), 'int')
        # Getting the type of 'lrpath' (line 687)
        lrpath_129573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 42), 'lrpath')
        # Obtaining the member '__getitem__' of a type (line 687)
        getitem___129574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 42), lrpath_129573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 687)
        subscript_call_result_129575 = invoke(stypy.reporting.localization.Localization(__file__, 687, 42), getitem___129574, int_129572)
        
        # Obtaining the member '__getitem__' of a type (line 687)
        getitem___129576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 42), subscript_call_result_129575, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 687)
        subscript_call_result_129577 = invoke(stypy.reporting.localization.Localization(__file__, 687, 42), getitem___129576, int_129571)
        
        # Obtaining the member '__getitem__' of a type (line 687)
        getitem___129578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 42), subscript_call_result_129577, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 687)
        subscript_call_result_129579 = invoke(stypy.reporting.localization.Localization(__file__, 687, 42), getitem___129578, int_129570)
        
        # Applying the binary operator '<' (line 687)
        result_lt_129580 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 23), '<', subscript_call_result_129569, subscript_call_result_129579)
        
        # Testing the type of an if condition (line 687)
        if_condition_129581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 687, 20), result_lt_129580)
        # Assigning a type to the variable 'if_condition_129581' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 20), 'if_condition_129581', if_condition_129581)
        # SSA begins for if statement (line 687)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 688)
        # Processing the call arguments (line 688)
        
        # Obtaining an instance of the builtin type 'tuple' (line 688)
        tuple_129584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 688)
        # Adding element type (line 688)
        # Getting the type of 'Path' (line 688)
        Path_129585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 39), 'Path', False)
        # Obtaining the member 'LINETO' of a type (line 688)
        LINETO_129586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 39), Path_129585, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 39), tuple_129584, LINETO_129586)
        # Adding element type (line 688)
        
        # Obtaining an instance of the builtin type 'list' (line 688)
        list_129587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 688)
        # Adding element type (line 688)
        
        # Obtaining the type of the subscript
        int_129588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 67), 'int')
        
        # Obtaining the type of the subscript
        int_129589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 64), 'int')
        
        # Obtaining the type of the subscript
        int_129590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 60), 'int')
        # Getting the type of 'lrpath' (line 688)
        lrpath_129591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 53), 'lrpath', False)
        # Obtaining the member '__getitem__' of a type (line 688)
        getitem___129592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 53), lrpath_129591, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 688)
        subscript_call_result_129593 = invoke(stypy.reporting.localization.Localization(__file__, 688, 53), getitem___129592, int_129590)
        
        # Obtaining the member '__getitem__' of a type (line 688)
        getitem___129594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 53), subscript_call_result_129593, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 688)
        subscript_call_result_129595 = invoke(stypy.reporting.localization.Localization(__file__, 688, 53), getitem___129594, int_129589)
        
        # Obtaining the member '__getitem__' of a type (line 688)
        getitem___129596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 53), subscript_call_result_129595, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 688)
        subscript_call_result_129597 = invoke(stypy.reporting.localization.Localization(__file__, 688, 53), getitem___129596, int_129588)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 52), list_129587, subscript_call_result_129597)
        # Adding element type (line 688)
        
        # Obtaining the type of the subscript
        int_129598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 67), 'int')
        
        # Obtaining the type of the subscript
        int_129599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 64), 'int')
        
        # Obtaining the type of the subscript
        int_129600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 60), 'int')
        # Getting the type of 'urpath' (line 689)
        urpath_129601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 53), 'urpath', False)
        # Obtaining the member '__getitem__' of a type (line 689)
        getitem___129602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 53), urpath_129601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 689)
        subscript_call_result_129603 = invoke(stypy.reporting.localization.Localization(__file__, 689, 53), getitem___129602, int_129600)
        
        # Obtaining the member '__getitem__' of a type (line 689)
        getitem___129604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 53), subscript_call_result_129603, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 689)
        subscript_call_result_129605 = invoke(stypy.reporting.localization.Localization(__file__, 689, 53), getitem___129604, int_129599)
        
        # Obtaining the member '__getitem__' of a type (line 689)
        getitem___129606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 53), subscript_call_result_129605, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 689)
        subscript_call_result_129607 = invoke(stypy.reporting.localization.Localization(__file__, 689, 53), getitem___129606, int_129598)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 52), list_129587, subscript_call_result_129607)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 39), tuple_129584, list_129587)
        
        # Processing the call keyword arguments (line 688)
        kwargs_129608 = {}
        # Getting the type of 'urpath' (line 688)
        urpath_129582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 24), 'urpath', False)
        # Obtaining the member 'append' of a type (line 688)
        append_129583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 24), urpath_129582, 'append')
        # Calling append(args, kwargs) (line 688)
        append_call_result_129609 = invoke(stypy.reporting.localization.Localization(__file__, 688, 24), append_129583, *[tuple_129584], **kwargs_129608)
        
        # SSA join for if statement (line 687)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 690):
        
        # Assigning a Name to a Name (line 690):
        # Getting the type of 'True' (line 690)
        True_129610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 39), 'True')
        # Assigning a type to the variable 'has_right_output' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 20), 'has_right_output', True_129610)
        # SSA join for if statement (line 684)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 691):
        
        # Assigning a Call to a Name:
        
        # Call to _add_output(...): (line 691)
        # Processing the call arguments (line 691)
        # Getting the type of 'urpath' (line 692)
        urpath_129613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 20), 'urpath', False)
        # Getting the type of 'angle' (line 692)
        angle_129614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 28), 'angle', False)
        # Getting the type of 'spec' (line 692)
        spec_129615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 36), 'spec', False)
        # Processing the call keyword arguments (line 691)
        kwargs_129616 = {}
        # Getting the type of 'self' (line 691)
        self_129611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 52), 'self', False)
        # Obtaining the member '_add_output' of a type (line 691)
        _add_output_129612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 52), self_129611, '_add_output')
        # Calling _add_output(args, kwargs) (line 691)
        _add_output_call_result_129617 = invoke(stypy.reporting.localization.Localization(__file__, 691, 52), _add_output_129612, *[urpath_129613, angle_129614, spec_129615], **kwargs_129616)
        
        # Assigning a type to the variable 'call_assignment_127450' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'call_assignment_127450', _add_output_call_result_129617)
        
        # Assigning a Call to a Name (line 691):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129621 = {}
        # Getting the type of 'call_assignment_127450' (line 691)
        call_assignment_127450_129618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'call_assignment_127450', False)
        # Obtaining the member '__getitem__' of a type (line 691)
        getitem___129619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 16), call_assignment_127450_129618, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129622 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129619, *[int_129620], **kwargs_129621)
        
        # Assigning a type to the variable 'call_assignment_127451' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'call_assignment_127451', getitem___call_result_129622)
        
        # Assigning a Name to a Subscript (line 691):
        # Getting the type of 'call_assignment_127451' (line 691)
        call_assignment_127451_129623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'call_assignment_127451')
        # Getting the type of 'tips' (line 691)
        tips_129624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'tips')
        # Getting the type of 'i' (line 691)
        i_129625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 21), 'i')
        slice_129626 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 691, 16), None, None, None)
        # Storing an element on a container (line 691)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 691, 16), tips_129624, ((i_129625, slice_129626), call_assignment_127451_129623))
        
        # Assigning a Call to a Name (line 691):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 16), 'int')
        # Processing the call keyword arguments
        kwargs_129630 = {}
        # Getting the type of 'call_assignment_127450' (line 691)
        call_assignment_127450_129627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'call_assignment_127450', False)
        # Obtaining the member '__getitem__' of a type (line 691)
        getitem___129628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 16), call_assignment_127450_129627, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129631 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129628, *[int_129629], **kwargs_129630)
        
        # Assigning a type to the variable 'call_assignment_127452' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'call_assignment_127452', getitem___call_result_129631)
        
        # Assigning a Name to a Subscript (line 691):
        # Getting the type of 'call_assignment_127452' (line 691)
        call_assignment_127452_129632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'call_assignment_127452')
        # Getting the type of 'label_locations' (line 691)
        label_locations_129633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 28), 'label_locations')
        # Getting the type of 'i' (line 691)
        i_129634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 44), 'i')
        slice_129635 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 691, 28), None, None, None)
        # Storing an element on a container (line 691)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 691, 28), label_locations_129633, ((i_129634, slice_129635), call_assignment_127452_129632))
        # SSA join for if statement (line 683)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'has_left_input' (line 694)
        has_left_input_129636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 15), 'has_left_input')
        # Applying the 'not' unary operator (line 694)
        result_not__129637 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 11), 'not', has_left_input_129636)
        
        # Testing the type of an if condition (line 694)
        if_condition_129638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 8), result_not__129637)
        # Assigning a type to the variable 'if_condition_129638' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'if_condition_129638', if_condition_129638)
        # SSA begins for if statement (line 694)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to pop(...): (line 695)
        # Processing the call keyword arguments (line 695)
        kwargs_129641 = {}
        # Getting the type of 'ulpath' (line 695)
        ulpath_129639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 12), 'ulpath', False)
        # Obtaining the member 'pop' of a type (line 695)
        pop_129640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 12), ulpath_129639, 'pop')
        # Calling pop(args, kwargs) (line 695)
        pop_call_result_129642 = invoke(stypy.reporting.localization.Localization(__file__, 695, 12), pop_129640, *[], **kwargs_129641)
        
        
        # Call to pop(...): (line 696)
        # Processing the call keyword arguments (line 696)
        kwargs_129645 = {}
        # Getting the type of 'llpath' (line 696)
        llpath_129643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 12), 'llpath', False)
        # Obtaining the member 'pop' of a type (line 696)
        pop_129644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 12), llpath_129643, 'pop')
        # Calling pop(args, kwargs) (line 696)
        pop_call_result_129646 = invoke(stypy.reporting.localization.Localization(__file__, 696, 12), pop_129644, *[], **kwargs_129645)
        
        # SSA join for if statement (line 694)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'has_right_output' (line 697)
        has_right_output_129647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 15), 'has_right_output')
        # Applying the 'not' unary operator (line 697)
        result_not__129648 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 11), 'not', has_right_output_129647)
        
        # Testing the type of an if condition (line 697)
        if_condition_129649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 8), result_not__129648)
        # Assigning a type to the variable 'if_condition_129649' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'if_condition_129649', if_condition_129649)
        # SSA begins for if statement (line 697)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to pop(...): (line 698)
        # Processing the call keyword arguments (line 698)
        kwargs_129652 = {}
        # Getting the type of 'lrpath' (line 698)
        lrpath_129650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 12), 'lrpath', False)
        # Obtaining the member 'pop' of a type (line 698)
        pop_129651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 12), lrpath_129650, 'pop')
        # Calling pop(args, kwargs) (line 698)
        pop_call_result_129653 = invoke(stypy.reporting.localization.Localization(__file__, 698, 12), pop_129651, *[], **kwargs_129652)
        
        
        # Call to pop(...): (line 699)
        # Processing the call keyword arguments (line 699)
        kwargs_129656 = {}
        # Getting the type of 'urpath' (line 699)
        urpath_129654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'urpath', False)
        # Obtaining the member 'pop' of a type (line 699)
        pop_129655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 12), urpath_129654, 'pop')
        # Calling pop(args, kwargs) (line 699)
        pop_call_result_129657 = invoke(stypy.reporting.localization.Localization(__file__, 699, 12), pop_129655, *[], **kwargs_129656)
        
        # SSA join for if statement (line 697)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 702):
        
        # Assigning a BinOp to a Name (line 702):
        # Getting the type of 'urpath' (line 702)
        urpath_129658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 16), 'urpath')
        
        # Call to _revert(...): (line 702)
        # Processing the call arguments (line 702)
        # Getting the type of 'lrpath' (line 702)
        lrpath_129661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 38), 'lrpath', False)
        # Processing the call keyword arguments (line 702)
        kwargs_129662 = {}
        # Getting the type of 'self' (line 702)
        self_129659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 25), 'self', False)
        # Obtaining the member '_revert' of a type (line 702)
        _revert_129660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 25), self_129659, '_revert')
        # Calling _revert(args, kwargs) (line 702)
        _revert_call_result_129663 = invoke(stypy.reporting.localization.Localization(__file__, 702, 25), _revert_129660, *[lrpath_129661], **kwargs_129662)
        
        # Applying the binary operator '+' (line 702)
        result_add_129664 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 16), '+', urpath_129658, _revert_call_result_129663)
        
        # Getting the type of 'llpath' (line 702)
        llpath_129665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 48), 'llpath')
        # Applying the binary operator '+' (line 702)
        result_add_129666 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 46), '+', result_add_129664, llpath_129665)
        
        
        # Call to _revert(...): (line 702)
        # Processing the call arguments (line 702)
        # Getting the type of 'ulpath' (line 702)
        ulpath_129669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 70), 'ulpath', False)
        # Processing the call keyword arguments (line 702)
        kwargs_129670 = {}
        # Getting the type of 'self' (line 702)
        self_129667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 57), 'self', False)
        # Obtaining the member '_revert' of a type (line 702)
        _revert_129668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 57), self_129667, '_revert')
        # Calling _revert(args, kwargs) (line 702)
        _revert_call_result_129671 = invoke(stypy.reporting.localization.Localization(__file__, 702, 57), _revert_129668, *[ulpath_129669], **kwargs_129670)
        
        # Applying the binary operator '+' (line 702)
        result_add_129672 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 55), '+', result_add_129666, _revert_call_result_129671)
        
        
        # Obtaining an instance of the builtin type 'list' (line 703)
        list_129673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 703)
        # Adding element type (line 703)
        
        # Obtaining an instance of the builtin type 'tuple' (line 703)
        tuple_129674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 703)
        # Adding element type (line 703)
        # Getting the type of 'Path' (line 703)
        Path_129675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 18), 'Path')
        # Obtaining the member 'CLOSEPOLY' of a type (line 703)
        CLOSEPOLY_129676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 18), Path_129675, 'CLOSEPOLY')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 18), tuple_129674, CLOSEPOLY_129676)
        # Adding element type (line 703)
        
        # Obtaining the type of the subscript
        int_129677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 44), 'int')
        
        # Obtaining the type of the subscript
        int_129678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 41), 'int')
        # Getting the type of 'urpath' (line 703)
        urpath_129679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 34), 'urpath')
        # Obtaining the member '__getitem__' of a type (line 703)
        getitem___129680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 34), urpath_129679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 703)
        subscript_call_result_129681 = invoke(stypy.reporting.localization.Localization(__file__, 703, 34), getitem___129680, int_129678)
        
        # Obtaining the member '__getitem__' of a type (line 703)
        getitem___129682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 34), subscript_call_result_129681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 703)
        subscript_call_result_129683 = invoke(stypy.reporting.localization.Localization(__file__, 703, 34), getitem___129682, int_129677)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 18), tuple_129674, subscript_call_result_129683)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 16), list_129673, tuple_129674)
        
        # Applying the binary operator '+' (line 702)
        result_add_129684 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 78), '+', result_add_129672, list_129673)
        
        # Assigning a type to the variable 'path' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'path', result_add_129684)
        
        # Assigning a Call to a Tuple (line 706):
        
        # Assigning a Call to a Name:
        
        # Call to list(...): (line 706)
        # Processing the call arguments (line 706)
        
        # Call to zip(...): (line 706)
        # Getting the type of 'path' (line 706)
        path_129687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 36), 'path', False)
        # Processing the call keyword arguments (line 706)
        kwargs_129688 = {}
        # Getting the type of 'zip' (line 706)
        zip_129686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 31), 'zip', False)
        # Calling zip(args, kwargs) (line 706)
        zip_call_result_129689 = invoke(stypy.reporting.localization.Localization(__file__, 706, 31), zip_129686, *[path_129687], **kwargs_129688)
        
        # Processing the call keyword arguments (line 706)
        kwargs_129690 = {}
        # Getting the type of 'list' (line 706)
        list_129685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 26), 'list', False)
        # Calling list(args, kwargs) (line 706)
        list_call_result_129691 = invoke(stypy.reporting.localization.Localization(__file__, 706, 26), list_129685, *[zip_call_result_129689], **kwargs_129690)
        
        # Assigning a type to the variable 'call_assignment_127453' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_127453', list_call_result_129691)
        
        # Assigning a Call to a Name (line 706):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 8), 'int')
        # Processing the call keyword arguments
        kwargs_129695 = {}
        # Getting the type of 'call_assignment_127453' (line 706)
        call_assignment_127453_129692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_127453', False)
        # Obtaining the member '__getitem__' of a type (line 706)
        getitem___129693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 8), call_assignment_127453_129692, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129696 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129693, *[int_129694], **kwargs_129695)
        
        # Assigning a type to the variable 'call_assignment_127454' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_127454', getitem___call_result_129696)
        
        # Assigning a Name to a Name (line 706):
        # Getting the type of 'call_assignment_127454' (line 706)
        call_assignment_127454_129697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_127454')
        # Assigning a type to the variable 'codes' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'codes', call_assignment_127454_129697)
        
        # Assigning a Call to a Name (line 706):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 8), 'int')
        # Processing the call keyword arguments
        kwargs_129701 = {}
        # Getting the type of 'call_assignment_127453' (line 706)
        call_assignment_127453_129698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_127453', False)
        # Obtaining the member '__getitem__' of a type (line 706)
        getitem___129699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 8), call_assignment_127453_129698, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129702 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129699, *[int_129700], **kwargs_129701)
        
        # Assigning a type to the variable 'call_assignment_127455' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_127455', getitem___call_result_129702)
        
        # Assigning a Name to a Name (line 706):
        # Getting the type of 'call_assignment_127455' (line 706)
        call_assignment_127455_129703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_127455')
        # Assigning a type to the variable 'vertices' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'vertices', call_assignment_127455_129703)
        
        # Assigning a Call to a Name (line 707):
        
        # Assigning a Call to a Name (line 707):
        
        # Call to array(...): (line 707)
        # Processing the call arguments (line 707)
        # Getting the type of 'vertices' (line 707)
        vertices_129706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 28), 'vertices', False)
        # Processing the call keyword arguments (line 707)
        kwargs_129707 = {}
        # Getting the type of 'np' (line 707)
        np_129704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 707)
        array_129705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 19), np_129704, 'array')
        # Calling array(args, kwargs) (line 707)
        array_call_result_129708 = invoke(stypy.reporting.localization.Localization(__file__, 707, 19), array_129705, *[vertices_129706], **kwargs_129707)
        
        # Assigning a type to the variable 'vertices' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'vertices', array_call_result_129708)

        @norecursion
        def _get_angle(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_get_angle'
            module_type_store = module_type_store.open_function_context('_get_angle', 709, 8, False)
            
            # Passed parameters checking function
            _get_angle.stypy_localization = localization
            _get_angle.stypy_type_of_self = None
            _get_angle.stypy_type_store = module_type_store
            _get_angle.stypy_function_name = '_get_angle'
            _get_angle.stypy_param_names_list = ['a', 'r']
            _get_angle.stypy_varargs_param_name = None
            _get_angle.stypy_kwargs_param_name = None
            _get_angle.stypy_call_defaults = defaults
            _get_angle.stypy_call_varargs = varargs
            _get_angle.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_get_angle', ['a', 'r'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_get_angle', localization, ['a', 'r'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_get_angle(...)' code ##################

            
            # Type idiom detected: calculating its left and rigth part (line 710)
            # Getting the type of 'a' (line 710)
            a_129709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 15), 'a')
            # Getting the type of 'None' (line 710)
            None_129710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 20), 'None')
            
            (may_be_129711, more_types_in_union_129712) = may_be_none(a_129709, None_129710)

            if may_be_129711:

                if more_types_in_union_129712:
                    # Runtime conditional SSA (line 710)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'None' (line 711)
                None_129713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 23), 'None')
                # Assigning a type to the variable 'stypy_return_type' (line 711)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 16), 'stypy_return_type', None_129713)

                if more_types_in_union_129712:
                    # Runtime conditional SSA for else branch (line 710)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_129711) or more_types_in_union_129712):
                # Getting the type of 'a' (line 713)
                a_129714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 23), 'a')
                # Getting the type of 'r' (line 713)
                r_129715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 27), 'r')
                # Applying the binary operator '+' (line 713)
                result_add_129716 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 23), '+', a_129714, r_129715)
                
                # Assigning a type to the variable 'stypy_return_type' (line 713)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 16), 'stypy_return_type', result_add_129716)

                if (may_be_129711 and more_types_in_union_129712):
                    # SSA join for if statement (line 710)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # ################# End of '_get_angle(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_get_angle' in the type store
            # Getting the type of 'stypy_return_type' (line 709)
            stypy_return_type_129717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_129717)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_get_angle'
            return stypy_return_type_129717

        # Assigning a type to the variable '_get_angle' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), '_get_angle', _get_angle)
        
        # Type idiom detected: calculating its left and rigth part (line 715)
        # Getting the type of 'prior' (line 715)
        prior_129718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 11), 'prior')
        # Getting the type of 'None' (line 715)
        None_129719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 20), 'None')
        
        (may_be_129720, more_types_in_union_129721) = may_be_none(prior_129718, None_129719)

        if may_be_129720:

            if more_types_in_union_129721:
                # Runtime conditional SSA (line 715)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'rotation' (line 716)
            rotation_129722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 15), 'rotation')
            int_129723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 27), 'int')
            # Applying the binary operator '!=' (line 716)
            result_ne_129724 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 15), '!=', rotation_129722, int_129723)
            
            # Testing the type of an if condition (line 716)
            if_condition_129725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 716, 12), result_ne_129724)
            # Assigning a type to the variable 'if_condition_129725' (line 716)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'if_condition_129725', if_condition_129725)
            # SSA begins for if statement (line 716)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a ListComp to a Name (line 717):
            
            # Assigning a ListComp to a Name (line 717):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'angles' (line 717)
            angles_129731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 67), 'angles')
            comprehension_129732 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 717, 26), angles_129731)
            # Assigning a type to the variable 'angle' (line 717)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 26), 'angle', comprehension_129732)
            
            # Call to _get_angle(...): (line 717)
            # Processing the call arguments (line 717)
            # Getting the type of 'angle' (line 717)
            angle_129727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 37), 'angle', False)
            # Getting the type of 'rotation' (line 717)
            rotation_129728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 44), 'rotation', False)
            # Processing the call keyword arguments (line 717)
            kwargs_129729 = {}
            # Getting the type of '_get_angle' (line 717)
            _get_angle_129726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 26), '_get_angle', False)
            # Calling _get_angle(args, kwargs) (line 717)
            _get_angle_call_result_129730 = invoke(stypy.reporting.localization.Localization(__file__, 717, 26), _get_angle_129726, *[angle_129727, rotation_129728], **kwargs_129729)
            
            list_129733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 26), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 717, 26), list_129733, _get_angle_call_result_129730)
            # Assigning a type to the variable 'angles' (line 717)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 16), 'angles', list_129733)
            
            # Assigning a Attribute to a Name (line 718):
            
            # Assigning a Attribute to a Name (line 718):
            
            # Call to rotate_deg(...): (line 718)
            # Processing the call arguments (line 718)
            # Getting the type of 'rotation' (line 718)
            rotation_129738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 47), 'rotation', False)
            int_129739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 58), 'int')
            # Applying the binary operator '*' (line 718)
            result_mul_129740 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 47), '*', rotation_129738, int_129739)
            
            # Processing the call keyword arguments (line 718)
            kwargs_129741 = {}
            
            # Call to Affine2D(...): (line 718)
            # Processing the call keyword arguments (line 718)
            kwargs_129735 = {}
            # Getting the type of 'Affine2D' (line 718)
            Affine2D_129734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 25), 'Affine2D', False)
            # Calling Affine2D(args, kwargs) (line 718)
            Affine2D_call_result_129736 = invoke(stypy.reporting.localization.Localization(__file__, 718, 25), Affine2D_129734, *[], **kwargs_129735)
            
            # Obtaining the member 'rotate_deg' of a type (line 718)
            rotate_deg_129737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 25), Affine2D_call_result_129736, 'rotate_deg')
            # Calling rotate_deg(args, kwargs) (line 718)
            rotate_deg_call_result_129742 = invoke(stypy.reporting.localization.Localization(__file__, 718, 25), rotate_deg_129737, *[result_mul_129740], **kwargs_129741)
            
            # Obtaining the member 'transform_affine' of a type (line 718)
            transform_affine_129743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 25), rotate_deg_call_result_129742, 'transform_affine')
            # Assigning a type to the variable 'rotate' (line 718)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 16), 'rotate', transform_affine_129743)
            
            # Assigning a Call to a Name (line 719):
            
            # Assigning a Call to a Name (line 719):
            
            # Call to rotate(...): (line 719)
            # Processing the call arguments (line 719)
            # Getting the type of 'tips' (line 719)
            tips_129745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 30), 'tips', False)
            # Processing the call keyword arguments (line 719)
            kwargs_129746 = {}
            # Getting the type of 'rotate' (line 719)
            rotate_129744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 23), 'rotate', False)
            # Calling rotate(args, kwargs) (line 719)
            rotate_call_result_129747 = invoke(stypy.reporting.localization.Localization(__file__, 719, 23), rotate_129744, *[tips_129745], **kwargs_129746)
            
            # Assigning a type to the variable 'tips' (line 719)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'tips', rotate_call_result_129747)
            
            # Assigning a Call to a Name (line 720):
            
            # Assigning a Call to a Name (line 720):
            
            # Call to rotate(...): (line 720)
            # Processing the call arguments (line 720)
            # Getting the type of 'label_locations' (line 720)
            label_locations_129749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 41), 'label_locations', False)
            # Processing the call keyword arguments (line 720)
            kwargs_129750 = {}
            # Getting the type of 'rotate' (line 720)
            rotate_129748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 34), 'rotate', False)
            # Calling rotate(args, kwargs) (line 720)
            rotate_call_result_129751 = invoke(stypy.reporting.localization.Localization(__file__, 720, 34), rotate_129748, *[label_locations_129749], **kwargs_129750)
            
            # Assigning a type to the variable 'label_locations' (line 720)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 16), 'label_locations', rotate_call_result_129751)
            
            # Assigning a Call to a Name (line 721):
            
            # Assigning a Call to a Name (line 721):
            
            # Call to rotate(...): (line 721)
            # Processing the call arguments (line 721)
            # Getting the type of 'vertices' (line 721)
            vertices_129753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 34), 'vertices', False)
            # Processing the call keyword arguments (line 721)
            kwargs_129754 = {}
            # Getting the type of 'rotate' (line 721)
            rotate_129752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 27), 'rotate', False)
            # Calling rotate(args, kwargs) (line 721)
            rotate_call_result_129755 = invoke(stypy.reporting.localization.Localization(__file__, 721, 27), rotate_129752, *[vertices_129753], **kwargs_129754)
            
            # Assigning a type to the variable 'vertices' (line 721)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 16), 'vertices', rotate_call_result_129755)
            # SSA join for if statement (line 716)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 722):
            
            # Assigning a Call to a Name (line 722):
            
            # Call to text(...): (line 722)
            # Processing the call arguments (line 722)
            int_129759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 32), 'int')
            int_129760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 35), 'int')
            # Processing the call keyword arguments (line 722)
            # Getting the type of 'patchlabel' (line 722)
            patchlabel_129761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 40), 'patchlabel', False)
            keyword_129762 = patchlabel_129761
            unicode_129763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 55), 'unicode', u'center')
            keyword_129764 = unicode_129763
            unicode_129765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 68), 'unicode', u'center')
            keyword_129766 = unicode_129765
            kwargs_129767 = {'va': keyword_129766, 's': keyword_129762, 'ha': keyword_129764}
            # Getting the type of 'self' (line 722)
            self_129756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 19), 'self', False)
            # Obtaining the member 'ax' of a type (line 722)
            ax_129757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 19), self_129756, 'ax')
            # Obtaining the member 'text' of a type (line 722)
            text_129758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 19), ax_129757, 'text')
            # Calling text(args, kwargs) (line 722)
            text_call_result_129768 = invoke(stypy.reporting.localization.Localization(__file__, 722, 19), text_129758, *[int_129759, int_129760], **kwargs_129767)
            
            # Assigning a type to the variable 'text' (line 722)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 12), 'text', text_call_result_129768)

            if more_types_in_union_129721:
                # Runtime conditional SSA for else branch (line 715)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_129720) or more_types_in_union_129721):
            
            # Assigning a BinOp to a Name (line 724):
            
            # Assigning a BinOp to a Name (line 724):
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_129769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 60), 'int')
            # Getting the type of 'connect' (line 724)
            connect_129770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 52), 'connect')
            # Obtaining the member '__getitem__' of a type (line 724)
            getitem___129771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 52), connect_129770, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 724)
            subscript_call_result_129772 = invoke(stypy.reporting.localization.Localization(__file__, 724, 52), getitem___129771, int_129769)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'prior' (line 724)
            prior_129773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 38), 'prior')
            # Getting the type of 'self' (line 724)
            self_129774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 24), 'self')
            # Obtaining the member 'diagrams' of a type (line 724)
            diagrams_129775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 24), self_129774, 'diagrams')
            # Obtaining the member '__getitem__' of a type (line 724)
            getitem___129776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 24), diagrams_129775, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 724)
            subscript_call_result_129777 = invoke(stypy.reporting.localization.Localization(__file__, 724, 24), getitem___129776, prior_129773)
            
            # Obtaining the member 'angles' of a type (line 724)
            angles_129778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 24), subscript_call_result_129777, 'angles')
            # Obtaining the member '__getitem__' of a type (line 724)
            getitem___129779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 24), angles_129778, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 724)
            subscript_call_result_129780 = invoke(stypy.reporting.localization.Localization(__file__, 724, 24), getitem___129779, subscript_call_result_129772)
            
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_129781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 39), 'int')
            # Getting the type of 'connect' (line 725)
            connect_129782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 31), 'connect')
            # Obtaining the member '__getitem__' of a type (line 725)
            getitem___129783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 31), connect_129782, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 725)
            subscript_call_result_129784 = invoke(stypy.reporting.localization.Localization(__file__, 725, 31), getitem___129783, int_129781)
            
            # Getting the type of 'angles' (line 725)
            angles_129785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 24), 'angles')
            # Obtaining the member '__getitem__' of a type (line 725)
            getitem___129786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 24), angles_129785, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 725)
            subscript_call_result_129787 = invoke(stypy.reporting.localization.Localization(__file__, 725, 24), getitem___129786, subscript_call_result_129784)
            
            # Applying the binary operator '-' (line 724)
            result_sub_129788 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 24), '-', subscript_call_result_129780, subscript_call_result_129787)
            
            # Assigning a type to the variable 'rotation' (line 724)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'rotation', result_sub_129788)
            
            # Assigning a ListComp to a Name (line 726):
            
            # Assigning a ListComp to a Name (line 726):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'angles' (line 726)
            angles_129794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 63), 'angles')
            comprehension_129795 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 22), angles_129794)
            # Assigning a type to the variable 'angle' (line 726)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 22), 'angle', comprehension_129795)
            
            # Call to _get_angle(...): (line 726)
            # Processing the call arguments (line 726)
            # Getting the type of 'angle' (line 726)
            angle_129790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 33), 'angle', False)
            # Getting the type of 'rotation' (line 726)
            rotation_129791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 40), 'rotation', False)
            # Processing the call keyword arguments (line 726)
            kwargs_129792 = {}
            # Getting the type of '_get_angle' (line 726)
            _get_angle_129789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 22), '_get_angle', False)
            # Calling _get_angle(args, kwargs) (line 726)
            _get_angle_call_result_129793 = invoke(stypy.reporting.localization.Localization(__file__, 726, 22), _get_angle_129789, *[angle_129790, rotation_129791], **kwargs_129792)
            
            list_129796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 22), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 22), list_129796, _get_angle_call_result_129793)
            # Assigning a type to the variable 'angles' (line 726)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 12), 'angles', list_129796)
            
            # Assigning a Attribute to a Name (line 727):
            
            # Assigning a Attribute to a Name (line 727):
            
            # Call to rotate_deg(...): (line 727)
            # Processing the call arguments (line 727)
            # Getting the type of 'rotation' (line 727)
            rotation_129801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 43), 'rotation', False)
            int_129802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 54), 'int')
            # Applying the binary operator '*' (line 727)
            result_mul_129803 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 43), '*', rotation_129801, int_129802)
            
            # Processing the call keyword arguments (line 727)
            kwargs_129804 = {}
            
            # Call to Affine2D(...): (line 727)
            # Processing the call keyword arguments (line 727)
            kwargs_129798 = {}
            # Getting the type of 'Affine2D' (line 727)
            Affine2D_129797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 21), 'Affine2D', False)
            # Calling Affine2D(args, kwargs) (line 727)
            Affine2D_call_result_129799 = invoke(stypy.reporting.localization.Localization(__file__, 727, 21), Affine2D_129797, *[], **kwargs_129798)
            
            # Obtaining the member 'rotate_deg' of a type (line 727)
            rotate_deg_129800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 21), Affine2D_call_result_129799, 'rotate_deg')
            # Calling rotate_deg(args, kwargs) (line 727)
            rotate_deg_call_result_129805 = invoke(stypy.reporting.localization.Localization(__file__, 727, 21), rotate_deg_129800, *[result_mul_129803], **kwargs_129804)
            
            # Obtaining the member 'transform_affine' of a type (line 727)
            transform_affine_129806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 21), rotate_deg_call_result_129805, 'transform_affine')
            # Assigning a type to the variable 'rotate' (line 727)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 12), 'rotate', transform_affine_129806)
            
            # Assigning a Call to a Name (line 728):
            
            # Assigning a Call to a Name (line 728):
            
            # Call to rotate(...): (line 728)
            # Processing the call arguments (line 728)
            # Getting the type of 'tips' (line 728)
            tips_129808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 26), 'tips', False)
            # Processing the call keyword arguments (line 728)
            kwargs_129809 = {}
            # Getting the type of 'rotate' (line 728)
            rotate_129807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 19), 'rotate', False)
            # Calling rotate(args, kwargs) (line 728)
            rotate_call_result_129810 = invoke(stypy.reporting.localization.Localization(__file__, 728, 19), rotate_129807, *[tips_129808], **kwargs_129809)
            
            # Assigning a type to the variable 'tips' (line 728)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 12), 'tips', rotate_call_result_129810)
            
            # Assigning a BinOp to a Name (line 729):
            
            # Assigning a BinOp to a Name (line 729):
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_129811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 55), 'int')
            # Getting the type of 'connect' (line 729)
            connect_129812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 47), 'connect')
            # Obtaining the member '__getitem__' of a type (line 729)
            getitem___129813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 47), connect_129812, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 729)
            subscript_call_result_129814 = invoke(stypy.reporting.localization.Localization(__file__, 729, 47), getitem___129813, int_129811)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'prior' (line 729)
            prior_129815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 35), 'prior')
            # Getting the type of 'self' (line 729)
            self_129816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 21), 'self')
            # Obtaining the member 'diagrams' of a type (line 729)
            diagrams_129817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 21), self_129816, 'diagrams')
            # Obtaining the member '__getitem__' of a type (line 729)
            getitem___129818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 21), diagrams_129817, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 729)
            subscript_call_result_129819 = invoke(stypy.reporting.localization.Localization(__file__, 729, 21), getitem___129818, prior_129815)
            
            # Obtaining the member 'tips' of a type (line 729)
            tips_129820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 21), subscript_call_result_129819, 'tips')
            # Obtaining the member '__getitem__' of a type (line 729)
            getitem___129821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 21), tips_129820, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 729)
            subscript_call_result_129822 = invoke(stypy.reporting.localization.Localization(__file__, 729, 21), getitem___129821, subscript_call_result_129814)
            
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_129823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 74), 'int')
            # Getting the type of 'connect' (line 729)
            connect_129824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 66), 'connect')
            # Obtaining the member '__getitem__' of a type (line 729)
            getitem___129825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 66), connect_129824, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 729)
            subscript_call_result_129826 = invoke(stypy.reporting.localization.Localization(__file__, 729, 66), getitem___129825, int_129823)
            
            # Getting the type of 'tips' (line 729)
            tips_129827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 61), 'tips')
            # Obtaining the member '__getitem__' of a type (line 729)
            getitem___129828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 61), tips_129827, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 729)
            subscript_call_result_129829 = invoke(stypy.reporting.localization.Localization(__file__, 729, 61), getitem___129828, subscript_call_result_129826)
            
            # Applying the binary operator '-' (line 729)
            result_sub_129830 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 21), '-', subscript_call_result_129822, subscript_call_result_129829)
            
            # Assigning a type to the variable 'offset' (line 729)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 12), 'offset', result_sub_129830)
            
            # Assigning a Attribute to a Name (line 730):
            
            # Assigning a Attribute to a Name (line 730):
            
            # Call to translate(...): (line 730)
            # Getting the type of 'offset' (line 730)
            offset_129835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 46), 'offset', False)
            # Processing the call keyword arguments (line 730)
            kwargs_129836 = {}
            
            # Call to Affine2D(...): (line 730)
            # Processing the call keyword arguments (line 730)
            kwargs_129832 = {}
            # Getting the type of 'Affine2D' (line 730)
            Affine2D_129831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 24), 'Affine2D', False)
            # Calling Affine2D(args, kwargs) (line 730)
            Affine2D_call_result_129833 = invoke(stypy.reporting.localization.Localization(__file__, 730, 24), Affine2D_129831, *[], **kwargs_129832)
            
            # Obtaining the member 'translate' of a type (line 730)
            translate_129834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 24), Affine2D_call_result_129833, 'translate')
            # Calling translate(args, kwargs) (line 730)
            translate_call_result_129837 = invoke(stypy.reporting.localization.Localization(__file__, 730, 24), translate_129834, *[offset_129835], **kwargs_129836)
            
            # Obtaining the member 'transform_affine' of a type (line 730)
            transform_affine_129838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 24), translate_call_result_129837, 'transform_affine')
            # Assigning a type to the variable 'translate' (line 730)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'translate', transform_affine_129838)
            
            # Assigning a Call to a Name (line 731):
            
            # Assigning a Call to a Name (line 731):
            
            # Call to translate(...): (line 731)
            # Processing the call arguments (line 731)
            # Getting the type of 'tips' (line 731)
            tips_129840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 29), 'tips', False)
            # Processing the call keyword arguments (line 731)
            kwargs_129841 = {}
            # Getting the type of 'translate' (line 731)
            translate_129839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 19), 'translate', False)
            # Calling translate(args, kwargs) (line 731)
            translate_call_result_129842 = invoke(stypy.reporting.localization.Localization(__file__, 731, 19), translate_129839, *[tips_129840], **kwargs_129841)
            
            # Assigning a type to the variable 'tips' (line 731)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 12), 'tips', translate_call_result_129842)
            
            # Assigning a Call to a Name (line 732):
            
            # Assigning a Call to a Name (line 732):
            
            # Call to translate(...): (line 732)
            # Processing the call arguments (line 732)
            
            # Call to rotate(...): (line 732)
            # Processing the call arguments (line 732)
            # Getting the type of 'label_locations' (line 732)
            label_locations_129845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 47), 'label_locations', False)
            # Processing the call keyword arguments (line 732)
            kwargs_129846 = {}
            # Getting the type of 'rotate' (line 732)
            rotate_129844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 40), 'rotate', False)
            # Calling rotate(args, kwargs) (line 732)
            rotate_call_result_129847 = invoke(stypy.reporting.localization.Localization(__file__, 732, 40), rotate_129844, *[label_locations_129845], **kwargs_129846)
            
            # Processing the call keyword arguments (line 732)
            kwargs_129848 = {}
            # Getting the type of 'translate' (line 732)
            translate_129843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 30), 'translate', False)
            # Calling translate(args, kwargs) (line 732)
            translate_call_result_129849 = invoke(stypy.reporting.localization.Localization(__file__, 732, 30), translate_129843, *[rotate_call_result_129847], **kwargs_129848)
            
            # Assigning a type to the variable 'label_locations' (line 732)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'label_locations', translate_call_result_129849)
            
            # Assigning a Call to a Name (line 733):
            
            # Assigning a Call to a Name (line 733):
            
            # Call to translate(...): (line 733)
            # Processing the call arguments (line 733)
            
            # Call to rotate(...): (line 733)
            # Processing the call arguments (line 733)
            # Getting the type of 'vertices' (line 733)
            vertices_129852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 40), 'vertices', False)
            # Processing the call keyword arguments (line 733)
            kwargs_129853 = {}
            # Getting the type of 'rotate' (line 733)
            rotate_129851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 33), 'rotate', False)
            # Calling rotate(args, kwargs) (line 733)
            rotate_call_result_129854 = invoke(stypy.reporting.localization.Localization(__file__, 733, 33), rotate_129851, *[vertices_129852], **kwargs_129853)
            
            # Processing the call keyword arguments (line 733)
            kwargs_129855 = {}
            # Getting the type of 'translate' (line 733)
            translate_129850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 23), 'translate', False)
            # Calling translate(args, kwargs) (line 733)
            translate_call_result_129856 = invoke(stypy.reporting.localization.Localization(__file__, 733, 23), translate_129850, *[rotate_call_result_129854], **kwargs_129855)
            
            # Assigning a type to the variable 'vertices' (line 733)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'vertices', translate_call_result_129856)
            
            # Assigning a Call to a Name (line 734):
            
            # Assigning a Call to a Name (line 734):
            
            # Call to dict(...): (line 734)
            # Processing the call keyword arguments (line 734)
            # Getting the type of 'patchlabel' (line 734)
            patchlabel_129858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 26), 'patchlabel', False)
            keyword_129859 = patchlabel_129858
            unicode_129860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 41), 'unicode', u'center')
            keyword_129861 = unicode_129860
            unicode_129862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 54), 'unicode', u'center')
            keyword_129863 = unicode_129862
            kwargs_129864 = {'va': keyword_129863, 's': keyword_129859, 'ha': keyword_129861}
            # Getting the type of 'dict' (line 734)
            dict_129857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 19), 'dict', False)
            # Calling dict(args, kwargs) (line 734)
            dict_call_result_129865 = invoke(stypy.reporting.localization.Localization(__file__, 734, 19), dict_129857, *[], **kwargs_129864)
            
            # Assigning a type to the variable 'kwds' (line 734)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 12), 'kwds', dict_call_result_129865)
            
            # Assigning a Call to a Name (line 735):
            
            # Assigning a Call to a Name (line 735):
            
            # Call to text(...): (line 735)
            # Getting the type of 'offset' (line 735)
            offset_129869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 33), 'offset', False)
            # Processing the call keyword arguments (line 735)
            # Getting the type of 'kwds' (line 735)
            kwds_129870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 43), 'kwds', False)
            kwargs_129871 = {'kwds_129870': kwds_129870}
            # Getting the type of 'self' (line 735)
            self_129866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), 'self', False)
            # Obtaining the member 'ax' of a type (line 735)
            ax_129867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 19), self_129866, 'ax')
            # Obtaining the member 'text' of a type (line 735)
            text_129868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 19), ax_129867, 'text')
            # Calling text(args, kwargs) (line 735)
            text_call_result_129872 = invoke(stypy.reporting.localization.Localization(__file__, 735, 19), text_129868, *[offset_129869], **kwargs_129871)
            
            # Assigning a type to the variable 'text' (line 735)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'text', text_call_result_129872)

            if (may_be_129720 and more_types_in_union_129721):
                # SSA join for if statement (line 715)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'False' (line 736)
        False_129873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 11), 'False')
        # Testing the type of an if condition (line 736)
        if_condition_129874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 8), False_129873)
        # Assigning a type to the variable 'if_condition_129874' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'if_condition_129874', if_condition_129874)
        # SSA begins for if statement (line 736)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 737)
        # Processing the call arguments (line 737)
        unicode_129876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 18), 'unicode', u'llpath\n')
        # Getting the type of 'llpath' (line 737)
        llpath_129877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 30), 'llpath', False)
        # Processing the call keyword arguments (line 737)
        kwargs_129878 = {}
        # Getting the type of 'print' (line 737)
        print_129875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'print', False)
        # Calling print(args, kwargs) (line 737)
        print_call_result_129879 = invoke(stypy.reporting.localization.Localization(__file__, 737, 12), print_129875, *[unicode_129876, llpath_129877], **kwargs_129878)
        
        
        # Call to print(...): (line 738)
        # Processing the call arguments (line 738)
        unicode_129881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 18), 'unicode', u'ulpath\n')
        
        # Call to _revert(...): (line 738)
        # Processing the call arguments (line 738)
        # Getting the type of 'ulpath' (line 738)
        ulpath_129884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 43), 'ulpath', False)
        # Processing the call keyword arguments (line 738)
        kwargs_129885 = {}
        # Getting the type of 'self' (line 738)
        self_129882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 30), 'self', False)
        # Obtaining the member '_revert' of a type (line 738)
        _revert_129883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 30), self_129882, '_revert')
        # Calling _revert(args, kwargs) (line 738)
        _revert_call_result_129886 = invoke(stypy.reporting.localization.Localization(__file__, 738, 30), _revert_129883, *[ulpath_129884], **kwargs_129885)
        
        # Processing the call keyword arguments (line 738)
        kwargs_129887 = {}
        # Getting the type of 'print' (line 738)
        print_129880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'print', False)
        # Calling print(args, kwargs) (line 738)
        print_call_result_129888 = invoke(stypy.reporting.localization.Localization(__file__, 738, 12), print_129880, *[unicode_129881, _revert_call_result_129886], **kwargs_129887)
        
        
        # Call to print(...): (line 739)
        # Processing the call arguments (line 739)
        unicode_129890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 18), 'unicode', u'urpath\n')
        # Getting the type of 'urpath' (line 739)
        urpath_129891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 30), 'urpath', False)
        # Processing the call keyword arguments (line 739)
        kwargs_129892 = {}
        # Getting the type of 'print' (line 739)
        print_129889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'print', False)
        # Calling print(args, kwargs) (line 739)
        print_call_result_129893 = invoke(stypy.reporting.localization.Localization(__file__, 739, 12), print_129889, *[unicode_129890, urpath_129891], **kwargs_129892)
        
        
        # Call to print(...): (line 740)
        # Processing the call arguments (line 740)
        unicode_129895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 18), 'unicode', u'lrpath\n')
        
        # Call to _revert(...): (line 740)
        # Processing the call arguments (line 740)
        # Getting the type of 'lrpath' (line 740)
        lrpath_129898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 43), 'lrpath', False)
        # Processing the call keyword arguments (line 740)
        kwargs_129899 = {}
        # Getting the type of 'self' (line 740)
        self_129896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 30), 'self', False)
        # Obtaining the member '_revert' of a type (line 740)
        _revert_129897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 30), self_129896, '_revert')
        # Calling _revert(args, kwargs) (line 740)
        _revert_call_result_129900 = invoke(stypy.reporting.localization.Localization(__file__, 740, 30), _revert_129897, *[lrpath_129898], **kwargs_129899)
        
        # Processing the call keyword arguments (line 740)
        kwargs_129901 = {}
        # Getting the type of 'print' (line 740)
        print_129894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'print', False)
        # Calling print(args, kwargs) (line 740)
        print_call_result_129902 = invoke(stypy.reporting.localization.Localization(__file__, 740, 12), print_129894, *[unicode_129895, _revert_call_result_129900], **kwargs_129901)
        
        
        # Assigning a Call to a Tuple (line 741):
        
        # Assigning a Call to a Name:
        
        # Call to list(...): (line 741)
        # Processing the call arguments (line 741)
        
        # Call to zip(...): (line 741)
        # Getting the type of 'vertices' (line 741)
        vertices_129905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 31), 'vertices', False)
        # Processing the call keyword arguments (line 741)
        kwargs_129906 = {}
        # Getting the type of 'zip' (line 741)
        zip_129904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 26), 'zip', False)
        # Calling zip(args, kwargs) (line 741)
        zip_call_result_129907 = invoke(stypy.reporting.localization.Localization(__file__, 741, 26), zip_129904, *[vertices_129905], **kwargs_129906)
        
        # Processing the call keyword arguments (line 741)
        kwargs_129908 = {}
        # Getting the type of 'list' (line 741)
        list_129903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 21), 'list', False)
        # Calling list(args, kwargs) (line 741)
        list_call_result_129909 = invoke(stypy.reporting.localization.Localization(__file__, 741, 21), list_129903, *[zip_call_result_129907], **kwargs_129908)
        
        # Assigning a type to the variable 'call_assignment_127456' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'call_assignment_127456', list_call_result_129909)
        
        # Assigning a Call to a Name (line 741):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 12), 'int')
        # Processing the call keyword arguments
        kwargs_129913 = {}
        # Getting the type of 'call_assignment_127456' (line 741)
        call_assignment_127456_129910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'call_assignment_127456', False)
        # Obtaining the member '__getitem__' of a type (line 741)
        getitem___129911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 12), call_assignment_127456_129910, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129914 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129911, *[int_129912], **kwargs_129913)
        
        # Assigning a type to the variable 'call_assignment_127457' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'call_assignment_127457', getitem___call_result_129914)
        
        # Assigning a Name to a Name (line 741):
        # Getting the type of 'call_assignment_127457' (line 741)
        call_assignment_127457_129915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'call_assignment_127457')
        # Assigning a type to the variable 'xs' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'xs', call_assignment_127457_129915)
        
        # Assigning a Call to a Name (line 741):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_129918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 12), 'int')
        # Processing the call keyword arguments
        kwargs_129919 = {}
        # Getting the type of 'call_assignment_127456' (line 741)
        call_assignment_127456_129916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'call_assignment_127456', False)
        # Obtaining the member '__getitem__' of a type (line 741)
        getitem___129917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 12), call_assignment_127456_129916, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_129920 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129917, *[int_129918], **kwargs_129919)
        
        # Assigning a type to the variable 'call_assignment_127458' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'call_assignment_127458', getitem___call_result_129920)
        
        # Assigning a Name to a Name (line 741):
        # Getting the type of 'call_assignment_127458' (line 741)
        call_assignment_127458_129921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'call_assignment_127458')
        # Assigning a type to the variable 'ys' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'ys', call_assignment_127458_129921)
        
        # Call to plot(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'xs' (line 742)
        xs_129925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 25), 'xs', False)
        # Getting the type of 'ys' (line 742)
        ys_129926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 29), 'ys', False)
        unicode_129927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 33), 'unicode', u'go-')
        # Processing the call keyword arguments (line 742)
        kwargs_129928 = {}
        # Getting the type of 'self' (line 742)
        self_129922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 12), 'self', False)
        # Obtaining the member 'ax' of a type (line 742)
        ax_129923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 12), self_129922, 'ax')
        # Obtaining the member 'plot' of a type (line 742)
        plot_129924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 12), ax_129923, 'plot')
        # Calling plot(args, kwargs) (line 742)
        plot_call_result_129929 = invoke(stypy.reporting.localization.Localization(__file__, 742, 12), plot_129924, *[xs_129925, ys_129926, unicode_129927], **kwargs_129928)
        
        # SSA join for if statement (line 736)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining the type of the subscript
        unicode_129930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 20), 'unicode', u'_internal.classic_mode')
        # Getting the type of 'rcParams' (line 743)
        rcParams_129931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 743)
        getitem___129932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 11), rcParams_129931, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 743)
        subscript_call_result_129933 = invoke(stypy.reporting.localization.Localization(__file__, 743, 11), getitem___129932, unicode_129930)
        
        # Testing the type of an if condition (line 743)
        if_condition_129934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 743, 8), subscript_call_result_129933)
        # Assigning a type to the variable 'if_condition_129934' (line 743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'if_condition_129934', if_condition_129934)
        # SSA begins for if statement (line 743)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 744):
        
        # Assigning a Call to a Name (line 744):
        
        # Call to pop(...): (line 744)
        # Processing the call arguments (line 744)
        unicode_129937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 28), 'unicode', u'fc')
        
        # Call to pop(...): (line 744)
        # Processing the call arguments (line 744)
        unicode_129940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 45), 'unicode', u'facecolor')
        unicode_129941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 58), 'unicode', u'#bfd1d4')
        # Processing the call keyword arguments (line 744)
        kwargs_129942 = {}
        # Getting the type of 'kwargs' (line 744)
        kwargs_129938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 34), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 744)
        pop_129939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 34), kwargs_129938, 'pop')
        # Calling pop(args, kwargs) (line 744)
        pop_call_result_129943 = invoke(stypy.reporting.localization.Localization(__file__, 744, 34), pop_129939, *[unicode_129940, unicode_129941], **kwargs_129942)
        
        # Processing the call keyword arguments (line 744)
        kwargs_129944 = {}
        # Getting the type of 'kwargs' (line 744)
        kwargs_129935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 17), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 744)
        pop_129936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 17), kwargs_129935, 'pop')
        # Calling pop(args, kwargs) (line 744)
        pop_call_result_129945 = invoke(stypy.reporting.localization.Localization(__file__, 744, 17), pop_129936, *[unicode_129937, pop_call_result_129943], **kwargs_129944)
        
        # Assigning a type to the variable 'fc' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'fc', pop_call_result_129945)
        
        # Assigning a Call to a Name (line 745):
        
        # Assigning a Call to a Name (line 745):
        
        # Call to pop(...): (line 745)
        # Processing the call arguments (line 745)
        unicode_129948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 28), 'unicode', u'lw')
        
        # Call to pop(...): (line 745)
        # Processing the call arguments (line 745)
        unicode_129951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 45), 'unicode', u'linewidth')
        float_129952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 58), 'float')
        # Processing the call keyword arguments (line 745)
        kwargs_129953 = {}
        # Getting the type of 'kwargs' (line 745)
        kwargs_129949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 34), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 745)
        pop_129950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 34), kwargs_129949, 'pop')
        # Calling pop(args, kwargs) (line 745)
        pop_call_result_129954 = invoke(stypy.reporting.localization.Localization(__file__, 745, 34), pop_129950, *[unicode_129951, float_129952], **kwargs_129953)
        
        # Processing the call keyword arguments (line 745)
        kwargs_129955 = {}
        # Getting the type of 'kwargs' (line 745)
        kwargs_129946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 17), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 745)
        pop_129947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 17), kwargs_129946, 'pop')
        # Calling pop(args, kwargs) (line 745)
        pop_call_result_129956 = invoke(stypy.reporting.localization.Localization(__file__, 745, 17), pop_129947, *[unicode_129948, pop_call_result_129954], **kwargs_129955)
        
        # Assigning a type to the variable 'lw' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'lw', pop_call_result_129956)
        # SSA branch for the else part of an if statement (line 743)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 747):
        
        # Assigning a Call to a Name (line 747):
        
        # Call to pop(...): (line 747)
        # Processing the call arguments (line 747)
        unicode_129959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 28), 'unicode', u'fc')
        
        # Call to pop(...): (line 747)
        # Processing the call arguments (line 747)
        unicode_129962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 45), 'unicode', u'facecolor')
        # Getting the type of 'None' (line 747)
        None_129963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 58), 'None', False)
        # Processing the call keyword arguments (line 747)
        kwargs_129964 = {}
        # Getting the type of 'kwargs' (line 747)
        kwargs_129960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 34), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 747)
        pop_129961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 34), kwargs_129960, 'pop')
        # Calling pop(args, kwargs) (line 747)
        pop_call_result_129965 = invoke(stypy.reporting.localization.Localization(__file__, 747, 34), pop_129961, *[unicode_129962, None_129963], **kwargs_129964)
        
        # Processing the call keyword arguments (line 747)
        kwargs_129966 = {}
        # Getting the type of 'kwargs' (line 747)
        kwargs_129957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 17), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 747)
        pop_129958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 17), kwargs_129957, 'pop')
        # Calling pop(args, kwargs) (line 747)
        pop_call_result_129967 = invoke(stypy.reporting.localization.Localization(__file__, 747, 17), pop_129958, *[unicode_129959, pop_call_result_129965], **kwargs_129966)
        
        # Assigning a type to the variable 'fc' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'fc', pop_call_result_129967)
        
        # Assigning a Call to a Name (line 748):
        
        # Assigning a Call to a Name (line 748):
        
        # Call to pop(...): (line 748)
        # Processing the call arguments (line 748)
        unicode_129970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 28), 'unicode', u'lw')
        
        # Call to pop(...): (line 748)
        # Processing the call arguments (line 748)
        unicode_129973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 45), 'unicode', u'linewidth')
        # Getting the type of 'None' (line 748)
        None_129974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 58), 'None', False)
        # Processing the call keyword arguments (line 748)
        kwargs_129975 = {}
        # Getting the type of 'kwargs' (line 748)
        kwargs_129971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 34), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 748)
        pop_129972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 34), kwargs_129971, 'pop')
        # Calling pop(args, kwargs) (line 748)
        pop_call_result_129976 = invoke(stypy.reporting.localization.Localization(__file__, 748, 34), pop_129972, *[unicode_129973, None_129974], **kwargs_129975)
        
        # Processing the call keyword arguments (line 748)
        kwargs_129977 = {}
        # Getting the type of 'kwargs' (line 748)
        kwargs_129968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 17), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 748)
        pop_129969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 17), kwargs_129968, 'pop')
        # Calling pop(args, kwargs) (line 748)
        pop_call_result_129978 = invoke(stypy.reporting.localization.Localization(__file__, 748, 17), pop_129969, *[unicode_129970, pop_call_result_129976], **kwargs_129977)
        
        # Assigning a type to the variable 'lw' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 12), 'lw', pop_call_result_129978)
        # SSA join for if statement (line 743)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 749)
        # Getting the type of 'fc' (line 749)
        fc_129979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 11), 'fc')
        # Getting the type of 'None' (line 749)
        None_129980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 17), 'None')
        
        (may_be_129981, more_types_in_union_129982) = may_be_none(fc_129979, None_129980)

        if may_be_129981:

            if more_types_in_union_129982:
                # Runtime conditional SSA (line 749)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 750):
            
            # Assigning a Subscript to a Name (line 750):
            
            # Obtaining the type of the subscript
            unicode_129983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 69), 'unicode', u'color')
            
            # Call to next(...): (line 750)
            # Processing the call arguments (line 750)
            # Getting the type of 'self' (line 750)
            self_129986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 26), 'self', False)
            # Obtaining the member 'ax' of a type (line 750)
            ax_129987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 26), self_129986, 'ax')
            # Obtaining the member '_get_patches_for_fill' of a type (line 750)
            _get_patches_for_fill_129988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 26), ax_129987, '_get_patches_for_fill')
            # Obtaining the member 'prop_cycler' of a type (line 750)
            prop_cycler_129989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 26), _get_patches_for_fill_129988, 'prop_cycler')
            # Processing the call keyword arguments (line 750)
            kwargs_129990 = {}
            # Getting the type of 'six' (line 750)
            six_129984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 17), 'six', False)
            # Obtaining the member 'next' of a type (line 750)
            next_129985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 17), six_129984, 'next')
            # Calling next(args, kwargs) (line 750)
            next_call_result_129991 = invoke(stypy.reporting.localization.Localization(__file__, 750, 17), next_129985, *[prop_cycler_129989], **kwargs_129990)
            
            # Obtaining the member '__getitem__' of a type (line 750)
            getitem___129992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 17), next_call_result_129991, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 750)
            subscript_call_result_129993 = invoke(stypy.reporting.localization.Localization(__file__, 750, 17), getitem___129992, unicode_129983)
            
            # Assigning a type to the variable 'fc' (line 750)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'fc', subscript_call_result_129993)

            if more_types_in_union_129982:
                # SSA join for if statement (line 749)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 751):
        
        # Assigning a Call to a Name (line 751):
        
        # Call to PathPatch(...): (line 751)
        # Processing the call arguments (line 751)
        
        # Call to Path(...): (line 751)
        # Processing the call arguments (line 751)
        # Getting the type of 'vertices' (line 751)
        vertices_129996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 31), 'vertices', False)
        # Getting the type of 'codes' (line 751)
        codes_129997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 41), 'codes', False)
        # Processing the call keyword arguments (line 751)
        kwargs_129998 = {}
        # Getting the type of 'Path' (line 751)
        Path_129995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 26), 'Path', False)
        # Calling Path(args, kwargs) (line 751)
        Path_call_result_129999 = invoke(stypy.reporting.localization.Localization(__file__, 751, 26), Path_129995, *[vertices_129996, codes_129997], **kwargs_129998)
        
        # Processing the call keyword arguments (line 751)
        # Getting the type of 'fc' (line 751)
        fc_130000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 52), 'fc', False)
        keyword_130001 = fc_130000
        # Getting the type of 'lw' (line 751)
        lw_130002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 59), 'lw', False)
        keyword_130003 = lw_130002
        # Getting the type of 'kwargs' (line 751)
        kwargs_130004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 65), 'kwargs', False)
        kwargs_130005 = {'lw': keyword_130003, 'fc': keyword_130001, 'kwargs_130004': kwargs_130004}
        # Getting the type of 'PathPatch' (line 751)
        PathPatch_129994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'PathPatch', False)
        # Calling PathPatch(args, kwargs) (line 751)
        PathPatch_call_result_130006 = invoke(stypy.reporting.localization.Localization(__file__, 751, 16), PathPatch_129994, *[Path_call_result_129999], **kwargs_130005)
        
        # Assigning a type to the variable 'patch' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'patch', PathPatch_call_result_130006)
        
        # Call to add_patch(...): (line 752)
        # Processing the call arguments (line 752)
        # Getting the type of 'patch' (line 752)
        patch_130010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 26), 'patch', False)
        # Processing the call keyword arguments (line 752)
        kwargs_130011 = {}
        # Getting the type of 'self' (line 752)
        self_130007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'self', False)
        # Obtaining the member 'ax' of a type (line 752)
        ax_130008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 8), self_130007, 'ax')
        # Obtaining the member 'add_patch' of a type (line 752)
        add_patch_130009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 8), ax_130008, 'add_patch')
        # Calling add_patch(args, kwargs) (line 752)
        add_patch_call_result_130012 = invoke(stypy.reporting.localization.Localization(__file__, 752, 8), add_patch_130009, *[patch_130010], **kwargs_130011)
        
        
        # Assigning a List to a Name (line 755):
        
        # Assigning a List to a Name (line 755):
        
        # Obtaining an instance of the builtin type 'list' (line 755)
        list_130013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 755)
        
        # Assigning a type to the variable 'texts' (line 755)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'texts', list_130013)
        
        
        # Call to zip(...): (line 756)
        # Processing the call arguments (line 756)
        # Getting the type of 'flows' (line 756)
        flows_130015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 50), 'flows', False)
        # Getting the type of 'angles' (line 756)
        angles_130016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 57), 'angles', False)
        # Getting the type of 'labels' (line 756)
        labels_130017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 65), 'labels', False)
        # Getting the type of 'label_locations' (line 757)
        label_locations_130018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 50), 'label_locations', False)
        # Processing the call keyword arguments (line 756)
        kwargs_130019 = {}
        # Getting the type of 'zip' (line 756)
        zip_130014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 46), 'zip', False)
        # Calling zip(args, kwargs) (line 756)
        zip_call_result_130020 = invoke(stypy.reporting.localization.Localization(__file__, 756, 46), zip_130014, *[flows_130015, angles_130016, labels_130017, label_locations_130018], **kwargs_130019)
        
        # Testing the type of a for loop iterable (line 756)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 756, 8), zip_call_result_130020)
        # Getting the type of the for loop variable (line 756)
        for_loop_var_130021 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 756, 8), zip_call_result_130020)
        # Assigning a type to the variable 'number' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'number', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 8), for_loop_var_130021))
        # Assigning a type to the variable 'angle' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 8), for_loop_var_130021))
        # Assigning a type to the variable 'label' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'label', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 8), for_loop_var_130021))
        # Assigning a type to the variable 'location' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'location', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 8), for_loop_var_130021))
        # SSA begins for a for statement (line 756)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'label' (line 758)
        label_130022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 15), 'label')
        # Getting the type of 'None' (line 758)
        None_130023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 24), 'None')
        # Applying the binary operator 'is' (line 758)
        result_is__130024 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 15), 'is', label_130022, None_130023)
        
        
        # Getting the type of 'angle' (line 758)
        angle_130025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 32), 'angle')
        # Getting the type of 'None' (line 758)
        None_130026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 41), 'None')
        # Applying the binary operator 'is' (line 758)
        result_is__130027 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 32), 'is', angle_130025, None_130026)
        
        # Applying the binary operator 'or' (line 758)
        result_or_keyword_130028 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 15), 'or', result_is__130024, result_is__130027)
        
        # Testing the type of an if condition (line 758)
        if_condition_130029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 758, 12), result_or_keyword_130028)
        # Assigning a type to the variable 'if_condition_130029' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 12), 'if_condition_130029', if_condition_130029)
        # SSA begins for if statement (line 758)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 759):
        
        # Assigning a Str to a Name (line 759):
        unicode_130030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 24), 'unicode', u'')
        # Assigning a type to the variable 'label' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 16), 'label', unicode_130030)
        # SSA branch for the else part of an if statement (line 758)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 760)
        self_130031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 17), 'self')
        # Obtaining the member 'unit' of a type (line 760)
        unit_130032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 17), self_130031, 'unit')
        # Getting the type of 'None' (line 760)
        None_130033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 34), 'None')
        # Applying the binary operator 'isnot' (line 760)
        result_is_not_130034 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 17), 'isnot', unit_130032, None_130033)
        
        # Testing the type of an if condition (line 760)
        if_condition_130035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 760, 17), result_is_not_130034)
        # Assigning a type to the variable 'if_condition_130035' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 17), 'if_condition_130035', if_condition_130035)
        # SSA begins for if statement (line 760)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 761):
        
        # Assigning a BinOp to a Name (line 761):
        # Getting the type of 'self' (line 761)
        self_130036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 27), 'self')
        # Obtaining the member 'format' of a type (line 761)
        format_130037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 27), self_130036, 'format')
        
        # Call to abs(...): (line 761)
        # Processing the call arguments (line 761)
        # Getting the type of 'number' (line 761)
        number_130039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 45), 'number', False)
        # Processing the call keyword arguments (line 761)
        kwargs_130040 = {}
        # Getting the type of 'abs' (line 761)
        abs_130038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 41), 'abs', False)
        # Calling abs(args, kwargs) (line 761)
        abs_call_result_130041 = invoke(stypy.reporting.localization.Localization(__file__, 761, 41), abs_130038, *[number_130039], **kwargs_130040)
        
        # Applying the binary operator '%' (line 761)
        result_mod_130042 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 27), '%', format_130037, abs_call_result_130041)
        
        # Getting the type of 'self' (line 761)
        self_130043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 55), 'self')
        # Obtaining the member 'unit' of a type (line 761)
        unit_130044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 55), self_130043, 'unit')
        # Applying the binary operator '+' (line 761)
        result_add_130045 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 27), '+', result_mod_130042, unit_130044)
        
        # Assigning a type to the variable 'quantity' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 16), 'quantity', result_add_130045)
        
        
        # Getting the type of 'label' (line 762)
        label_130046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 19), 'label')
        unicode_130047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 28), 'unicode', u'')
        # Applying the binary operator '!=' (line 762)
        result_ne_130048 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 19), '!=', label_130046, unicode_130047)
        
        # Testing the type of an if condition (line 762)
        if_condition_130049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 16), result_ne_130048)
        # Assigning a type to the variable 'if_condition_130049' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 16), 'if_condition_130049', if_condition_130049)
        # SSA begins for if statement (line 762)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'label' (line 763)
        label_130050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 20), 'label')
        unicode_130051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 29), 'unicode', u'\n')
        # Applying the binary operator '+=' (line 763)
        result_iadd_130052 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 20), '+=', label_130050, unicode_130051)
        # Assigning a type to the variable 'label' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 20), 'label', result_iadd_130052)
        
        # SSA join for if statement (line 762)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'label' (line 764)
        label_130053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 16), 'label')
        # Getting the type of 'quantity' (line 764)
        quantity_130054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 25), 'quantity')
        # Applying the binary operator '+=' (line 764)
        result_iadd_130055 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 16), '+=', label_130053, quantity_130054)
        # Assigning a type to the variable 'label' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 16), 'label', result_iadd_130055)
        
        # SSA join for if statement (line 760)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 758)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 765)
        # Processing the call arguments (line 765)
        
        # Call to text(...): (line 765)
        # Processing the call keyword arguments (line 765)
        
        # Obtaining the type of the subscript
        int_130061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 49), 'int')
        # Getting the type of 'location' (line 765)
        location_130062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 40), 'location', False)
        # Obtaining the member '__getitem__' of a type (line 765)
        getitem___130063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 40), location_130062, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 765)
        subscript_call_result_130064 = invoke(stypy.reporting.localization.Localization(__file__, 765, 40), getitem___130063, int_130061)
        
        keyword_130065 = subscript_call_result_130064
        
        # Obtaining the type of the subscript
        int_130066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 64), 'int')
        # Getting the type of 'location' (line 765)
        location_130067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 55), 'location', False)
        # Obtaining the member '__getitem__' of a type (line 765)
        getitem___130068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 55), location_130067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 765)
        subscript_call_result_130069 = invoke(stypy.reporting.localization.Localization(__file__, 765, 55), getitem___130068, int_130066)
        
        keyword_130070 = subscript_call_result_130069
        # Getting the type of 'label' (line 766)
        label_130071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 40), 'label', False)
        keyword_130072 = label_130071
        unicode_130073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 41), 'unicode', u'center')
        keyword_130074 = unicode_130073
        unicode_130075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 54), 'unicode', u'center')
        keyword_130076 = unicode_130075
        kwargs_130077 = {'y': keyword_130070, 'x': keyword_130065, 's': keyword_130072, 'ha': keyword_130074, 'va': keyword_130076}
        # Getting the type of 'self' (line 765)
        self_130058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 25), 'self', False)
        # Obtaining the member 'ax' of a type (line 765)
        ax_130059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 25), self_130058, 'ax')
        # Obtaining the member 'text' of a type (line 765)
        text_130060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 25), ax_130059, 'text')
        # Calling text(args, kwargs) (line 765)
        text_call_result_130078 = invoke(stypy.reporting.localization.Localization(__file__, 765, 25), text_130060, *[], **kwargs_130077)
        
        # Processing the call keyword arguments (line 765)
        kwargs_130079 = {}
        # Getting the type of 'texts' (line 765)
        texts_130056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'texts', False)
        # Obtaining the member 'append' of a type (line 765)
        append_130057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 12), texts_130056, 'append')
        # Calling append(args, kwargs) (line 765)
        append_call_result_130080 = invoke(stypy.reporting.localization.Localization(__file__, 765, 12), append_130057, *[text_call_result_130078], **kwargs_130079)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Attribute (line 773):
        
        # Assigning a Tuple to a Attribute (line 773):
        
        # Obtaining an instance of the builtin type 'tuple' (line 773)
        tuple_130081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 773)
        # Adding element type (line 773)
        
        # Call to min(...): (line 773)
        # Processing the call arguments (line 773)
        
        # Call to min(...): (line 773)
        # Processing the call arguments (line 773)
        
        # Obtaining the type of the subscript
        slice_130085 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 773, 34), None, None, None)
        int_130086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 46), 'int')
        # Getting the type of 'vertices' (line 773)
        vertices_130087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 34), 'vertices', False)
        # Obtaining the member '__getitem__' of a type (line 773)
        getitem___130088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 34), vertices_130087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 773)
        subscript_call_result_130089 = invoke(stypy.reporting.localization.Localization(__file__, 773, 34), getitem___130088, (slice_130085, int_130086))
        
        # Processing the call keyword arguments (line 773)
        kwargs_130090 = {}
        # Getting the type of 'np' (line 773)
        np_130083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 27), 'np', False)
        # Obtaining the member 'min' of a type (line 773)
        min_130084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 27), np_130083, 'min')
        # Calling min(args, kwargs) (line 773)
        min_call_result_130091 = invoke(stypy.reporting.localization.Localization(__file__, 773, 27), min_130084, *[subscript_call_result_130089], **kwargs_130090)
        
        
        # Call to min(...): (line 774)
        # Processing the call arguments (line 774)
        
        # Obtaining the type of the subscript
        slice_130094 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 774, 34), None, None, None)
        int_130095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 53), 'int')
        # Getting the type of 'label_locations' (line 774)
        label_locations_130096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 34), 'label_locations', False)
        # Obtaining the member '__getitem__' of a type (line 774)
        getitem___130097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 34), label_locations_130096, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 774)
        subscript_call_result_130098 = invoke(stypy.reporting.localization.Localization(__file__, 774, 34), getitem___130097, (slice_130094, int_130095))
        
        # Processing the call keyword arguments (line 774)
        kwargs_130099 = {}
        # Getting the type of 'np' (line 774)
        np_130092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 27), 'np', False)
        # Obtaining the member 'min' of a type (line 774)
        min_130093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 27), np_130092, 'min')
        # Calling min(args, kwargs) (line 774)
        min_call_result_130100 = invoke(stypy.reporting.localization.Localization(__file__, 774, 27), min_130093, *[subscript_call_result_130098], **kwargs_130099)
        
        
        # Obtaining the type of the subscript
        int_130101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 39), 'int')
        # Getting the type of 'self' (line 775)
        self_130102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 27), 'self', False)
        # Obtaining the member 'extent' of a type (line 775)
        extent_130103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 27), self_130102, 'extent')
        # Obtaining the member '__getitem__' of a type (line 775)
        getitem___130104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 27), extent_130103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 775)
        subscript_call_result_130105 = invoke(stypy.reporting.localization.Localization(__file__, 775, 27), getitem___130104, int_130101)
        
        # Processing the call keyword arguments (line 773)
        kwargs_130106 = {}
        # Getting the type of 'min' (line 773)
        min_130082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 23), 'min', False)
        # Calling min(args, kwargs) (line 773)
        min_call_result_130107 = invoke(stypy.reporting.localization.Localization(__file__, 773, 23), min_130082, *[min_call_result_130091, min_call_result_130100, subscript_call_result_130105], **kwargs_130106)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 23), tuple_130081, min_call_result_130107)
        # Adding element type (line 773)
        
        # Call to max(...): (line 776)
        # Processing the call arguments (line 776)
        
        # Call to max(...): (line 776)
        # Processing the call arguments (line 776)
        
        # Obtaining the type of the subscript
        slice_130111 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 776, 34), None, None, None)
        int_130112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 46), 'int')
        # Getting the type of 'vertices' (line 776)
        vertices_130113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 34), 'vertices', False)
        # Obtaining the member '__getitem__' of a type (line 776)
        getitem___130114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 34), vertices_130113, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 776)
        subscript_call_result_130115 = invoke(stypy.reporting.localization.Localization(__file__, 776, 34), getitem___130114, (slice_130111, int_130112))
        
        # Processing the call keyword arguments (line 776)
        kwargs_130116 = {}
        # Getting the type of 'np' (line 776)
        np_130109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 27), 'np', False)
        # Obtaining the member 'max' of a type (line 776)
        max_130110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 27), np_130109, 'max')
        # Calling max(args, kwargs) (line 776)
        max_call_result_130117 = invoke(stypy.reporting.localization.Localization(__file__, 776, 27), max_130110, *[subscript_call_result_130115], **kwargs_130116)
        
        
        # Call to max(...): (line 777)
        # Processing the call arguments (line 777)
        
        # Obtaining the type of the subscript
        slice_130120 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 777, 34), None, None, None)
        int_130121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 53), 'int')
        # Getting the type of 'label_locations' (line 777)
        label_locations_130122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 34), 'label_locations', False)
        # Obtaining the member '__getitem__' of a type (line 777)
        getitem___130123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 34), label_locations_130122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 777)
        subscript_call_result_130124 = invoke(stypy.reporting.localization.Localization(__file__, 777, 34), getitem___130123, (slice_130120, int_130121))
        
        # Processing the call keyword arguments (line 777)
        kwargs_130125 = {}
        # Getting the type of 'np' (line 777)
        np_130118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 27), 'np', False)
        # Obtaining the member 'max' of a type (line 777)
        max_130119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 27), np_130118, 'max')
        # Calling max(args, kwargs) (line 777)
        max_call_result_130126 = invoke(stypy.reporting.localization.Localization(__file__, 777, 27), max_130119, *[subscript_call_result_130124], **kwargs_130125)
        
        
        # Obtaining the type of the subscript
        int_130127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 39), 'int')
        # Getting the type of 'self' (line 778)
        self_130128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 27), 'self', False)
        # Obtaining the member 'extent' of a type (line 778)
        extent_130129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 27), self_130128, 'extent')
        # Obtaining the member '__getitem__' of a type (line 778)
        getitem___130130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 27), extent_130129, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 778)
        subscript_call_result_130131 = invoke(stypy.reporting.localization.Localization(__file__, 778, 27), getitem___130130, int_130127)
        
        # Processing the call keyword arguments (line 776)
        kwargs_130132 = {}
        # Getting the type of 'max' (line 776)
        max_130108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 23), 'max', False)
        # Calling max(args, kwargs) (line 776)
        max_call_result_130133 = invoke(stypy.reporting.localization.Localization(__file__, 776, 23), max_130108, *[max_call_result_130117, max_call_result_130126, subscript_call_result_130131], **kwargs_130132)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 23), tuple_130081, max_call_result_130133)
        # Adding element type (line 773)
        
        # Call to min(...): (line 779)
        # Processing the call arguments (line 779)
        
        # Call to min(...): (line 779)
        # Processing the call arguments (line 779)
        
        # Obtaining the type of the subscript
        slice_130137 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 779, 34), None, None, None)
        int_130138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 46), 'int')
        # Getting the type of 'vertices' (line 779)
        vertices_130139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 34), 'vertices', False)
        # Obtaining the member '__getitem__' of a type (line 779)
        getitem___130140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 34), vertices_130139, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 779)
        subscript_call_result_130141 = invoke(stypy.reporting.localization.Localization(__file__, 779, 34), getitem___130140, (slice_130137, int_130138))
        
        # Processing the call keyword arguments (line 779)
        kwargs_130142 = {}
        # Getting the type of 'np' (line 779)
        np_130135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 27), 'np', False)
        # Obtaining the member 'min' of a type (line 779)
        min_130136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 27), np_130135, 'min')
        # Calling min(args, kwargs) (line 779)
        min_call_result_130143 = invoke(stypy.reporting.localization.Localization(__file__, 779, 27), min_130136, *[subscript_call_result_130141], **kwargs_130142)
        
        
        # Call to min(...): (line 780)
        # Processing the call arguments (line 780)
        
        # Obtaining the type of the subscript
        slice_130146 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 780, 34), None, None, None)
        int_130147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 53), 'int')
        # Getting the type of 'label_locations' (line 780)
        label_locations_130148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 34), 'label_locations', False)
        # Obtaining the member '__getitem__' of a type (line 780)
        getitem___130149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 34), label_locations_130148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 780)
        subscript_call_result_130150 = invoke(stypy.reporting.localization.Localization(__file__, 780, 34), getitem___130149, (slice_130146, int_130147))
        
        # Processing the call keyword arguments (line 780)
        kwargs_130151 = {}
        # Getting the type of 'np' (line 780)
        np_130144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 27), 'np', False)
        # Obtaining the member 'min' of a type (line 780)
        min_130145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 27), np_130144, 'min')
        # Calling min(args, kwargs) (line 780)
        min_call_result_130152 = invoke(stypy.reporting.localization.Localization(__file__, 780, 27), min_130145, *[subscript_call_result_130150], **kwargs_130151)
        
        
        # Obtaining the type of the subscript
        int_130153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 39), 'int')
        # Getting the type of 'self' (line 781)
        self_130154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 27), 'self', False)
        # Obtaining the member 'extent' of a type (line 781)
        extent_130155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 27), self_130154, 'extent')
        # Obtaining the member '__getitem__' of a type (line 781)
        getitem___130156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 27), extent_130155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 781)
        subscript_call_result_130157 = invoke(stypy.reporting.localization.Localization(__file__, 781, 27), getitem___130156, int_130153)
        
        # Processing the call keyword arguments (line 779)
        kwargs_130158 = {}
        # Getting the type of 'min' (line 779)
        min_130134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 23), 'min', False)
        # Calling min(args, kwargs) (line 779)
        min_call_result_130159 = invoke(stypy.reporting.localization.Localization(__file__, 779, 23), min_130134, *[min_call_result_130143, min_call_result_130152, subscript_call_result_130157], **kwargs_130158)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 23), tuple_130081, min_call_result_130159)
        # Adding element type (line 773)
        
        # Call to max(...): (line 782)
        # Processing the call arguments (line 782)
        
        # Call to max(...): (line 782)
        # Processing the call arguments (line 782)
        
        # Obtaining the type of the subscript
        slice_130163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 782, 34), None, None, None)
        int_130164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 46), 'int')
        # Getting the type of 'vertices' (line 782)
        vertices_130165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 34), 'vertices', False)
        # Obtaining the member '__getitem__' of a type (line 782)
        getitem___130166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 34), vertices_130165, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 782)
        subscript_call_result_130167 = invoke(stypy.reporting.localization.Localization(__file__, 782, 34), getitem___130166, (slice_130163, int_130164))
        
        # Processing the call keyword arguments (line 782)
        kwargs_130168 = {}
        # Getting the type of 'np' (line 782)
        np_130161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 27), 'np', False)
        # Obtaining the member 'max' of a type (line 782)
        max_130162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 27), np_130161, 'max')
        # Calling max(args, kwargs) (line 782)
        max_call_result_130169 = invoke(stypy.reporting.localization.Localization(__file__, 782, 27), max_130162, *[subscript_call_result_130167], **kwargs_130168)
        
        
        # Call to max(...): (line 783)
        # Processing the call arguments (line 783)
        
        # Obtaining the type of the subscript
        slice_130172 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 783, 34), None, None, None)
        int_130173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 53), 'int')
        # Getting the type of 'label_locations' (line 783)
        label_locations_130174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 34), 'label_locations', False)
        # Obtaining the member '__getitem__' of a type (line 783)
        getitem___130175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 34), label_locations_130174, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 783)
        subscript_call_result_130176 = invoke(stypy.reporting.localization.Localization(__file__, 783, 34), getitem___130175, (slice_130172, int_130173))
        
        # Processing the call keyword arguments (line 783)
        kwargs_130177 = {}
        # Getting the type of 'np' (line 783)
        np_130170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 27), 'np', False)
        # Obtaining the member 'max' of a type (line 783)
        max_130171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 27), np_130170, 'max')
        # Calling max(args, kwargs) (line 783)
        max_call_result_130178 = invoke(stypy.reporting.localization.Localization(__file__, 783, 27), max_130171, *[subscript_call_result_130176], **kwargs_130177)
        
        
        # Obtaining the type of the subscript
        int_130179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 39), 'int')
        # Getting the type of 'self' (line 784)
        self_130180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 27), 'self', False)
        # Obtaining the member 'extent' of a type (line 784)
        extent_130181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 27), self_130180, 'extent')
        # Obtaining the member '__getitem__' of a type (line 784)
        getitem___130182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 27), extent_130181, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 784)
        subscript_call_result_130183 = invoke(stypy.reporting.localization.Localization(__file__, 784, 27), getitem___130182, int_130179)
        
        # Processing the call keyword arguments (line 782)
        kwargs_130184 = {}
        # Getting the type of 'max' (line 782)
        max_130160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 23), 'max', False)
        # Calling max(args, kwargs) (line 782)
        max_call_result_130185 = invoke(stypy.reporting.localization.Localization(__file__, 782, 23), max_130160, *[max_call_result_130169, max_call_result_130178, subscript_call_result_130183], **kwargs_130184)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 773, 23), tuple_130081, max_call_result_130185)
        
        # Getting the type of 'self' (line 773)
        self_130186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'self')
        # Setting the type of the member 'extent' of a type (line 773)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 8), self_130186, 'extent', tuple_130081)
        
        # Call to append(...): (line 789)
        # Processing the call arguments (line 789)
        
        # Call to Bunch(...): (line 789)
        # Processing the call keyword arguments (line 789)
        # Getting the type of 'patch' (line 789)
        patch_130191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 41), 'patch', False)
        keyword_130192 = patch_130191
        # Getting the type of 'flows' (line 789)
        flows_130193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 54), 'flows', False)
        keyword_130194 = flows_130193
        # Getting the type of 'angles' (line 789)
        angles_130195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 68), 'angles', False)
        keyword_130196 = angles_130195
        # Getting the type of 'tips' (line 790)
        tips_130197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 40), 'tips', False)
        keyword_130198 = tips_130197
        # Getting the type of 'text' (line 790)
        text_130199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 51), 'text', False)
        keyword_130200 = text_130199
        # Getting the type of 'texts' (line 790)
        texts_130201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 63), 'texts', False)
        keyword_130202 = texts_130201
        kwargs_130203 = {'text': keyword_130200, 'flows': keyword_130194, 'patch': keyword_130192, 'texts': keyword_130202, 'angles': keyword_130196, 'tips': keyword_130198}
        # Getting the type of 'Bunch' (line 789)
        Bunch_130190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 29), 'Bunch', False)
        # Calling Bunch(args, kwargs) (line 789)
        Bunch_call_result_130204 = invoke(stypy.reporting.localization.Localization(__file__, 789, 29), Bunch_130190, *[], **kwargs_130203)
        
        # Processing the call keyword arguments (line 789)
        kwargs_130205 = {}
        # Getting the type of 'self' (line 789)
        self_130187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'self', False)
        # Obtaining the member 'diagrams' of a type (line 789)
        diagrams_130188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 8), self_130187, 'diagrams')
        # Obtaining the member 'append' of a type (line 789)
        append_130189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 8), diagrams_130188, 'append')
        # Calling append(args, kwargs) (line 789)
        append_call_result_130206 = invoke(stypy.reporting.localization.Localization(__file__, 789, 8), append_130189, *[Bunch_call_result_130204], **kwargs_130205)
        
        # Getting the type of 'self' (line 793)
        self_130207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 793)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'stypy_return_type', self_130207)
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_130208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_130208


    @norecursion
    def finish(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finish'
        module_type_store = module_type_store.open_function_context('finish', 795, 4, False)
        # Assigning a type to the variable 'self' (line 796)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Sankey.finish.__dict__.__setitem__('stypy_localization', localization)
        Sankey.finish.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Sankey.finish.__dict__.__setitem__('stypy_type_store', module_type_store)
        Sankey.finish.__dict__.__setitem__('stypy_function_name', 'Sankey.finish')
        Sankey.finish.__dict__.__setitem__('stypy_param_names_list', [])
        Sankey.finish.__dict__.__setitem__('stypy_varargs_param_name', None)
        Sankey.finish.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Sankey.finish.__dict__.__setitem__('stypy_call_defaults', defaults)
        Sankey.finish.__dict__.__setitem__('stypy_call_varargs', varargs)
        Sankey.finish.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Sankey.finish.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sankey.finish', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finish', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finish(...)' code ##################

        unicode_130209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, (-1)), 'unicode', u'\n        Adjust the axes and return a list of information about the Sankey\n        subdiagram(s).\n\n        Return value is a list of subdiagrams represented with the following\n        fields:\n\n          ===============   ===================================================\n          Field             Description\n          ===============   ===================================================\n          *patch*           Sankey outline (an instance of\n                            :class:`~maplotlib.patches.PathPatch`)\n          *flows*           values of the flows (positive for input, negative\n                            for output)\n          *angles*          list of angles of the arrows [deg/90]\n                            For example, if the diagram has not been rotated,\n                            an input to the top side will have an angle of 3\n                            (DOWN), and an output from the top side will have\n                            an angle of 1 (UP).  If a flow has been skipped\n                            (because its magnitude is less than *tolerance*),\n                            then its angle will be *None*.\n          *tips*            array in which each row is an [x, y] pair\n                            indicating the positions of the tips (or "dips") of\n                            the flow paths\n                            If the magnitude of a flow is less the *tolerance*\n                            for the instance of :class:`Sankey`, the flow is\n                            skipped and its tip will be at the center of the\n                            diagram.\n          *text*            :class:`~matplotlib.text.Text` instance for the\n                            label of the diagram\n          *texts*           list of :class:`~matplotlib.text.Text` instances\n                            for the labels of flows\n          ===============   ===================================================\n\n        .. seealso::\n\n            :meth:`add`\n        ')
        
        # Call to axis(...): (line 834)
        # Processing the call arguments (line 834)
        
        # Obtaining an instance of the builtin type 'list' (line 834)
        list_130213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 834)
        # Adding element type (line 834)
        
        # Obtaining the type of the subscript
        int_130214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 34), 'int')
        # Getting the type of 'self' (line 834)
        self_130215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 22), 'self', False)
        # Obtaining the member 'extent' of a type (line 834)
        extent_130216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 22), self_130215, 'extent')
        # Obtaining the member '__getitem__' of a type (line 834)
        getitem___130217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 22), extent_130216, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 834)
        subscript_call_result_130218 = invoke(stypy.reporting.localization.Localization(__file__, 834, 22), getitem___130217, int_130214)
        
        # Getting the type of 'self' (line 834)
        self_130219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 39), 'self', False)
        # Obtaining the member 'margin' of a type (line 834)
        margin_130220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 39), self_130219, 'margin')
        # Applying the binary operator '-' (line 834)
        result_sub_130221 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 22), '-', subscript_call_result_130218, margin_130220)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 21), list_130213, result_sub_130221)
        # Adding element type (line 834)
        
        # Obtaining the type of the subscript
        int_130222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 34), 'int')
        # Getting the type of 'self' (line 835)
        self_130223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 22), 'self', False)
        # Obtaining the member 'extent' of a type (line 835)
        extent_130224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 22), self_130223, 'extent')
        # Obtaining the member '__getitem__' of a type (line 835)
        getitem___130225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 22), extent_130224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 835)
        subscript_call_result_130226 = invoke(stypy.reporting.localization.Localization(__file__, 835, 22), getitem___130225, int_130222)
        
        # Getting the type of 'self' (line 835)
        self_130227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 39), 'self', False)
        # Obtaining the member 'margin' of a type (line 835)
        margin_130228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 39), self_130227, 'margin')
        # Applying the binary operator '+' (line 835)
        result_add_130229 = python_operator(stypy.reporting.localization.Localization(__file__, 835, 22), '+', subscript_call_result_130226, margin_130228)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 21), list_130213, result_add_130229)
        # Adding element type (line 834)
        
        # Obtaining the type of the subscript
        int_130230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 34), 'int')
        # Getting the type of 'self' (line 836)
        self_130231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 22), 'self', False)
        # Obtaining the member 'extent' of a type (line 836)
        extent_130232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 22), self_130231, 'extent')
        # Obtaining the member '__getitem__' of a type (line 836)
        getitem___130233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 22), extent_130232, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 836)
        subscript_call_result_130234 = invoke(stypy.reporting.localization.Localization(__file__, 836, 22), getitem___130233, int_130230)
        
        # Getting the type of 'self' (line 836)
        self_130235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 39), 'self', False)
        # Obtaining the member 'margin' of a type (line 836)
        margin_130236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 39), self_130235, 'margin')
        # Applying the binary operator '-' (line 836)
        result_sub_130237 = python_operator(stypy.reporting.localization.Localization(__file__, 836, 22), '-', subscript_call_result_130234, margin_130236)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 21), list_130213, result_sub_130237)
        # Adding element type (line 834)
        
        # Obtaining the type of the subscript
        int_130238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 34), 'int')
        # Getting the type of 'self' (line 837)
        self_130239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 22), 'self', False)
        # Obtaining the member 'extent' of a type (line 837)
        extent_130240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 22), self_130239, 'extent')
        # Obtaining the member '__getitem__' of a type (line 837)
        getitem___130241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 22), extent_130240, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 837)
        subscript_call_result_130242 = invoke(stypy.reporting.localization.Localization(__file__, 837, 22), getitem___130241, int_130238)
        
        # Getting the type of 'self' (line 837)
        self_130243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 39), 'self', False)
        # Obtaining the member 'margin' of a type (line 837)
        margin_130244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 39), self_130243, 'margin')
        # Applying the binary operator '+' (line 837)
        result_add_130245 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 22), '+', subscript_call_result_130242, margin_130244)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 21), list_130213, result_add_130245)
        
        # Processing the call keyword arguments (line 834)
        kwargs_130246 = {}
        # Getting the type of 'self' (line 834)
        self_130210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'self', False)
        # Obtaining the member 'ax' of a type (line 834)
        ax_130211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 8), self_130210, 'ax')
        # Obtaining the member 'axis' of a type (line 834)
        axis_130212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 8), ax_130211, 'axis')
        # Calling axis(args, kwargs) (line 834)
        axis_call_result_130247 = invoke(stypy.reporting.localization.Localization(__file__, 834, 8), axis_130212, *[list_130213], **kwargs_130246)
        
        
        # Call to set_aspect(...): (line 838)
        # Processing the call arguments (line 838)
        unicode_130251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 27), 'unicode', u'equal')
        # Processing the call keyword arguments (line 838)
        unicode_130252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 47), 'unicode', u'datalim')
        keyword_130253 = unicode_130252
        kwargs_130254 = {'adjustable': keyword_130253}
        # Getting the type of 'self' (line 838)
        self_130248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'self', False)
        # Obtaining the member 'ax' of a type (line 838)
        ax_130249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 8), self_130248, 'ax')
        # Obtaining the member 'set_aspect' of a type (line 838)
        set_aspect_130250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 8), ax_130249, 'set_aspect')
        # Calling set_aspect(args, kwargs) (line 838)
        set_aspect_call_result_130255 = invoke(stypy.reporting.localization.Localization(__file__, 838, 8), set_aspect_130250, *[unicode_130251], **kwargs_130254)
        
        # Getting the type of 'self' (line 839)
        self_130256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 15), 'self')
        # Obtaining the member 'diagrams' of a type (line 839)
        diagrams_130257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 15), self_130256, 'diagrams')
        # Assigning a type to the variable 'stypy_return_type' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'stypy_return_type', diagrams_130257)
        
        # ################# End of 'finish(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finish' in the type store
        # Getting the type of 'stypy_return_type' (line 795)
        stypy_return_type_130258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130258)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finish'
        return stypy_return_type_130258


# Assigning a type to the variable 'Sankey' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'Sankey', Sankey)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
