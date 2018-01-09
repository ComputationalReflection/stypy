
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2003-2005 Peter J. Verveer
2: #
3: # Redistribution and use in source and binary forms, with or without
4: # modification, are permitted provided that the following conditions
5: # are met:
6: #
7: # 1. Redistributions of source code must retain the above copyright
8: #    notice, this list of conditions and the following disclaimer.
9: #
10: # 2. Redistributions in binary form must reproduce the above
11: #    copyright notice, this list of conditions and the following
12: #    disclaimer in the documentation and/or other materials provided
13: #    with the distribution.
14: #
15: # 3. The name of the author may not be used to endorse or promote
16: #    products derived from this software without specific prior
17: #    written permission.
18: #
19: # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
20: # OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
21: # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
22: # ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
23: # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
24: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
25: # GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
26: # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
27: # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
28: # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
29: # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
30: 
31: from __future__ import division, print_function, absolute_import
32: 
33: import numpy
34: from . import _ni_support
35: from . import _nd_image
36: from . import filters
37: 
38: __all__ = ['iterate_structure', 'generate_binary_structure', 'binary_erosion',
39:            'binary_dilation', 'binary_opening', 'binary_closing',
40:            'binary_hit_or_miss', 'binary_propagation', 'binary_fill_holes',
41:            'grey_erosion', 'grey_dilation', 'grey_opening', 'grey_closing',
42:            'morphological_gradient', 'morphological_laplace', 'white_tophat',
43:            'black_tophat', 'distance_transform_bf', 'distance_transform_cdt',
44:            'distance_transform_edt']
45: 
46: 
47: def _center_is_true(structure, origin):
48:     structure = numpy.array(structure)
49:     coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape,
50:                                                  origin)])
51:     return bool(structure[coor])
52: 
53: 
54: def iterate_structure(structure, iterations, origin=None):
55:     '''
56:     Iterate a structure by dilating it with itself.
57: 
58:     Parameters
59:     ----------
60:     structure : array_like
61:        Structuring element (an array of bools, for example), to be dilated with
62:        itself.
63:     iterations : int
64:        number of dilations performed on the structure with itself
65:     origin : optional
66:         If origin is None, only the iterated structure is returned. If
67:         not, a tuple of the iterated structure and the modified origin is
68:         returned.
69: 
70:     Returns
71:     -------
72:     iterate_structure : ndarray of bools
73:         A new structuring element obtained by dilating `structure`
74:         (`iterations` - 1) times with itself.
75: 
76:     See also
77:     --------
78:     generate_binary_structure
79: 
80:     Examples
81:     --------
82:     >>> from scipy import ndimage
83:     >>> struct = ndimage.generate_binary_structure(2, 1)
84:     >>> struct.astype(int)
85:     array([[0, 1, 0],
86:            [1, 1, 1],
87:            [0, 1, 0]])
88:     >>> ndimage.iterate_structure(struct, 2).astype(int)
89:     array([[0, 0, 1, 0, 0],
90:            [0, 1, 1, 1, 0],
91:            [1, 1, 1, 1, 1],
92:            [0, 1, 1, 1, 0],
93:            [0, 0, 1, 0, 0]])
94:     >>> ndimage.iterate_structure(struct, 3).astype(int)
95:     array([[0, 0, 0, 1, 0, 0, 0],
96:            [0, 0, 1, 1, 1, 0, 0],
97:            [0, 1, 1, 1, 1, 1, 0],
98:            [1, 1, 1, 1, 1, 1, 1],
99:            [0, 1, 1, 1, 1, 1, 0],
100:            [0, 0, 1, 1, 1, 0, 0],
101:            [0, 0, 0, 1, 0, 0, 0]])
102: 
103:     '''
104:     structure = numpy.asarray(structure)
105:     if iterations < 2:
106:         return structure.copy()
107:     ni = iterations - 1
108:     shape = [ii + ni * (ii - 1) for ii in structure.shape]
109:     pos = [ni * (structure.shape[ii] // 2) for ii in range(len(shape))]
110:     slc = [slice(pos[ii], pos[ii] + structure.shape[ii], None)
111:            for ii in range(len(shape))]
112:     out = numpy.zeros(shape, bool)
113:     out[slc] = structure != 0
114:     out = binary_dilation(out, structure, iterations=ni)
115:     if origin is None:
116:         return out
117:     else:
118:         origin = _ni_support._normalize_sequence(origin, structure.ndim)
119:         origin = [iterations * o for o in origin]
120:         return out, origin
121: 
122: 
123: def generate_binary_structure(rank, connectivity):
124:     '''
125:     Generate a binary structure for binary morphological operations.
126: 
127:     Parameters
128:     ----------
129:     rank : int
130:          Number of dimensions of the array to which the structuring element
131:          will be applied, as returned by `np.ndim`.
132:     connectivity : int
133:          `connectivity` determines which elements of the output array belong
134:          to the structure, i.e. are considered as neighbors of the central
135:          element. Elements up to a squared distance of `connectivity` from
136:          the center are considered neighbors. `connectivity` may range from 1
137:          (no diagonal elements are neighbors) to `rank` (all elements are
138:          neighbors).
139: 
140:     Returns
141:     -------
142:     output : ndarray of bools
143:          Structuring element which may be used for binary morphological
144:          operations, with `rank` dimensions and all dimensions equal to 3.
145: 
146:     See also
147:     --------
148:     iterate_structure, binary_dilation, binary_erosion
149: 
150:     Notes
151:     -----
152:     `generate_binary_structure` can only create structuring elements with
153:     dimensions equal to 3, i.e. minimal dimensions. For larger structuring
154:     elements, that are useful e.g. for eroding large objects, one may either
155:     use   `iterate_structure`, or create directly custom arrays with
156:     numpy functions such as `numpy.ones`.
157: 
158:     Examples
159:     --------
160:     >>> from scipy import ndimage
161:     >>> struct = ndimage.generate_binary_structure(2, 1)
162:     >>> struct
163:     array([[False,  True, False],
164:            [ True,  True,  True],
165:            [False,  True, False]], dtype=bool)
166:     >>> a = np.zeros((5,5))
167:     >>> a[2, 2] = 1
168:     >>> a
169:     array([[ 0.,  0.,  0.,  0.,  0.],
170:            [ 0.,  0.,  0.,  0.,  0.],
171:            [ 0.,  0.,  1.,  0.,  0.],
172:            [ 0.,  0.,  0.,  0.,  0.],
173:            [ 0.,  0.,  0.,  0.,  0.]])
174:     >>> b = ndimage.binary_dilation(a, structure=struct).astype(a.dtype)
175:     >>> b
176:     array([[ 0.,  0.,  0.,  0.,  0.],
177:            [ 0.,  0.,  1.,  0.,  0.],
178:            [ 0.,  1.,  1.,  1.,  0.],
179:            [ 0.,  0.,  1.,  0.,  0.],
180:            [ 0.,  0.,  0.,  0.,  0.]])
181:     >>> ndimage.binary_dilation(b, structure=struct).astype(a.dtype)
182:     array([[ 0.,  0.,  1.,  0.,  0.],
183:            [ 0.,  1.,  1.,  1.,  0.],
184:            [ 1.,  1.,  1.,  1.,  1.],
185:            [ 0.,  1.,  1.,  1.,  0.],
186:            [ 0.,  0.,  1.,  0.,  0.]])
187:     >>> struct = ndimage.generate_binary_structure(2, 2)
188:     >>> struct
189:     array([[ True,  True,  True],
190:            [ True,  True,  True],
191:            [ True,  True,  True]], dtype=bool)
192:     >>> struct = ndimage.generate_binary_structure(3, 1)
193:     >>> struct # no diagonal elements
194:     array([[[False, False, False],
195:             [False,  True, False],
196:             [False, False, False]],
197:            [[False,  True, False],
198:             [ True,  True,  True],
199:             [False,  True, False]],
200:            [[False, False, False],
201:             [False,  True, False],
202:             [False, False, False]]], dtype=bool)
203: 
204:     '''
205:     if connectivity < 1:
206:         connectivity = 1
207:     if rank < 1:
208:         return numpy.array(True, dtype=bool)
209:     output = numpy.fabs(numpy.indices([3] * rank) - 1)
210:     output = numpy.add.reduce(output, 0)
211:     return output <= connectivity
212: 
213: 
214: def _binary_erosion(input, structure, iterations, mask, output,
215:                     border_value, origin, invert, brute_force):
216:     input = numpy.asarray(input)
217:     if numpy.iscomplexobj(input):
218:         raise TypeError('Complex type not supported')
219:     if structure is None:
220:         structure = generate_binary_structure(input.ndim, 1)
221:     else:
222:         structure = numpy.asarray(structure, dtype=bool)
223:     if structure.ndim != input.ndim:
224:         raise RuntimeError('structure and input must have same dimensionality')
225:     if not structure.flags.contiguous:
226:         structure = structure.copy()
227:     if numpy.product(structure.shape,axis=0) < 1:
228:         raise RuntimeError('structure must not be empty')
229:     if mask is not None:
230:         mask = numpy.asarray(mask)
231:         if mask.shape != input.shape:
232:             raise RuntimeError('mask and input must have equal sizes')
233:     origin = _ni_support._normalize_sequence(origin, input.ndim)
234:     cit = _center_is_true(structure, origin)
235:     if isinstance(output, numpy.ndarray):
236:         if numpy.iscomplexobj(output):
237:             raise TypeError('Complex output type not supported')
238:     else:
239:         output = bool
240:     output, return_value = _ni_support._get_output(output, input)
241: 
242:     if iterations == 1:
243:         _nd_image.binary_erosion(input, structure, mask, output,
244:                                      border_value, origin, invert, cit, 0)
245:         return return_value
246:     elif cit and not brute_force:
247:         changed, coordinate_list = _nd_image.binary_erosion(input,
248:              structure, mask, output, border_value, origin, invert, cit, 1)
249:         structure = structure[tuple([slice(None, None, -1)] *
250:                                     structure.ndim)]
251:         for ii in range(len(origin)):
252:             origin[ii] = -origin[ii]
253:             if not structure.shape[ii] & 1:
254:                 origin[ii] -= 1
255:         if mask is not None:
256:             mask = numpy.asarray(mask, dtype=numpy.int8)
257:         if not structure.flags.contiguous:
258:             structure = structure.copy()
259:         _nd_image.binary_erosion2(output, structure, mask, iterations - 1,
260:                                   origin, invert, coordinate_list)
261:         return return_value
262:     else:
263:         tmp_in = numpy.zeros(input.shape, bool)
264:         if return_value is None:
265:             tmp_out = output
266:         else:
267:             tmp_out = numpy.zeros(input.shape, bool)
268:         if not iterations & 1:
269:             tmp_in, tmp_out = tmp_out, tmp_in
270:         changed = _nd_image.binary_erosion(input, structure, mask,
271:                             tmp_out, border_value, origin, invert, cit, 0)
272:         ii = 1
273:         while (ii < iterations) or (iterations < 1) and changed:
274:             tmp_in, tmp_out = tmp_out, tmp_in
275:             changed = _nd_image.binary_erosion(tmp_in, structure, mask,
276:                             tmp_out, border_value, origin, invert, cit, 0)
277:             ii += 1
278:         if return_value is not None:
279:             return tmp_out
280: 
281: 
282: def binary_erosion(input, structure=None, iterations=1, mask=None,
283:         output=None, border_value=0, origin=0, brute_force=False):
284:     '''
285:     Multi-dimensional binary erosion with a given structuring element.
286: 
287:     Binary erosion is a mathematical morphology operation used for image
288:     processing.
289: 
290:     Parameters
291:     ----------
292:     input : array_like
293:         Binary image to be eroded. Non-zero (True) elements form
294:         the subset to be eroded.
295:     structure : array_like, optional
296:         Structuring element used for the erosion. Non-zero elements are
297:         considered True. If no structuring element is provided, an element
298:         is generated with a square connectivity equal to one.
299:     iterations : {int, float}, optional
300:         The erosion is repeated `iterations` times (one, by default).
301:         If iterations is less than 1, the erosion is repeated until the
302:         result does not change anymore.
303:     mask : array_like, optional
304:         If a mask is given, only those elements with a True value at
305:         the corresponding mask element are modified at each iteration.
306:     output : ndarray, optional
307:         Array of the same shape as input, into which the output is placed.
308:         By default, a new array is created.
309:     origin : int or tuple of ints, optional
310:         Placement of the filter, by default 0.
311:     border_value : int (cast to 0 or 1), optional
312:         Value at the border in the output array.
313: 
314:     Returns
315:     -------
316:     binary_erosion : ndarray of bools
317:         Erosion of the input by the structuring element.
318: 
319:     See also
320:     --------
321:     grey_erosion, binary_dilation, binary_closing, binary_opening,
322:     generate_binary_structure
323: 
324:     Notes
325:     -----
326:     Erosion [1]_ is a mathematical morphology operation [2]_ that uses a
327:     structuring element for shrinking the shapes in an image. The binary
328:     erosion of an image by a structuring element is the locus of the points
329:     where a superimposition of the structuring element centered on the point
330:     is entirely contained in the set of non-zero elements of the image.
331: 
332:     References
333:     ----------
334:     .. [1] http://en.wikipedia.org/wiki/Erosion_%28morphology%29
335:     .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology
336: 
337:     Examples
338:     --------
339:     >>> from scipy import ndimage
340:     >>> a = np.zeros((7,7), dtype=int)
341:     >>> a[1:6, 2:5] = 1
342:     >>> a
343:     array([[0, 0, 0, 0, 0, 0, 0],
344:            [0, 0, 1, 1, 1, 0, 0],
345:            [0, 0, 1, 1, 1, 0, 0],
346:            [0, 0, 1, 1, 1, 0, 0],
347:            [0, 0, 1, 1, 1, 0, 0],
348:            [0, 0, 1, 1, 1, 0, 0],
349:            [0, 0, 0, 0, 0, 0, 0]])
350:     >>> ndimage.binary_erosion(a).astype(a.dtype)
351:     array([[0, 0, 0, 0, 0, 0, 0],
352:            [0, 0, 0, 0, 0, 0, 0],
353:            [0, 0, 0, 1, 0, 0, 0],
354:            [0, 0, 0, 1, 0, 0, 0],
355:            [0, 0, 0, 1, 0, 0, 0],
356:            [0, 0, 0, 0, 0, 0, 0],
357:            [0, 0, 0, 0, 0, 0, 0]])
358:     >>> #Erosion removes objects smaller than the structure
359:     >>> ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
360:     array([[0, 0, 0, 0, 0, 0, 0],
361:            [0, 0, 0, 0, 0, 0, 0],
362:            [0, 0, 0, 0, 0, 0, 0],
363:            [0, 0, 0, 0, 0, 0, 0],
364:            [0, 0, 0, 0, 0, 0, 0],
365:            [0, 0, 0, 0, 0, 0, 0],
366:            [0, 0, 0, 0, 0, 0, 0]])
367: 
368:     '''
369:     return _binary_erosion(input, structure, iterations, mask,
370:                            output, border_value, origin, 0, brute_force)
371: 
372: 
373: def binary_dilation(input, structure=None, iterations=1, mask=None,
374:         output=None, border_value=0, origin=0, brute_force=False):
375:     '''
376:     Multi-dimensional binary dilation with the given structuring element.
377: 
378:     Parameters
379:     ----------
380:     input : array_like
381:         Binary array_like to be dilated. Non-zero (True) elements form
382:         the subset to be dilated.
383:     structure : array_like, optional
384:         Structuring element used for the dilation. Non-zero elements are
385:         considered True. If no structuring element is provided an element
386:         is generated with a square connectivity equal to one.
387:     iterations : {int, float}, optional
388:         The dilation is repeated `iterations` times (one, by default).
389:         If iterations is less than 1, the dilation is repeated until the
390:         result does not change anymore.
391:     mask : array_like, optional
392:         If a mask is given, only those elements with a True value at
393:         the corresponding mask element are modified at each iteration.
394:     output : ndarray, optional
395:         Array of the same shape as input, into which the output is placed.
396:         By default, a new array is created.
397:     origin : int or tuple of ints, optional
398:         Placement of the filter, by default 0.
399:     border_value : int (cast to 0 or 1), optional
400:         Value at the border in the output array.
401: 
402:     Returns
403:     -------
404:     binary_dilation : ndarray of bools
405:         Dilation of the input by the structuring element.
406: 
407:     See also
408:     --------
409:     grey_dilation, binary_erosion, binary_closing, binary_opening,
410:     generate_binary_structure
411: 
412:     Notes
413:     -----
414:     Dilation [1]_ is a mathematical morphology operation [2]_ that uses a
415:     structuring element for expanding the shapes in an image. The binary
416:     dilation of an image by a structuring element is the locus of the points
417:     covered by the structuring element, when its center lies within the
418:     non-zero points of the image.
419: 
420:     References
421:     ----------
422:     .. [1] http://en.wikipedia.org/wiki/Dilation_%28morphology%29
423:     .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology
424: 
425:     Examples
426:     --------
427:     >>> from scipy import ndimage
428:     >>> a = np.zeros((5, 5))
429:     >>> a[2, 2] = 1
430:     >>> a
431:     array([[ 0.,  0.,  0.,  0.,  0.],
432:            [ 0.,  0.,  0.,  0.,  0.],
433:            [ 0.,  0.,  1.,  0.,  0.],
434:            [ 0.,  0.,  0.,  0.,  0.],
435:            [ 0.,  0.,  0.,  0.,  0.]])
436:     >>> ndimage.binary_dilation(a)
437:     array([[False, False, False, False, False],
438:            [False, False,  True, False, False],
439:            [False,  True,  True,  True, False],
440:            [False, False,  True, False, False],
441:            [False, False, False, False, False]], dtype=bool)
442:     >>> ndimage.binary_dilation(a).astype(a.dtype)
443:     array([[ 0.,  0.,  0.,  0.,  0.],
444:            [ 0.,  0.,  1.,  0.,  0.],
445:            [ 0.,  1.,  1.,  1.,  0.],
446:            [ 0.,  0.,  1.,  0.,  0.],
447:            [ 0.,  0.,  0.,  0.,  0.]])
448:     >>> # 3x3 structuring element with connectivity 1, used by default
449:     >>> struct1 = ndimage.generate_binary_structure(2, 1)
450:     >>> struct1
451:     array([[False,  True, False],
452:            [ True,  True,  True],
453:            [False,  True, False]], dtype=bool)
454:     >>> # 3x3 structuring element with connectivity 2
455:     >>> struct2 = ndimage.generate_binary_structure(2, 2)
456:     >>> struct2
457:     array([[ True,  True,  True],
458:            [ True,  True,  True],
459:            [ True,  True,  True]], dtype=bool)
460:     >>> ndimage.binary_dilation(a, structure=struct1).astype(a.dtype)
461:     array([[ 0.,  0.,  0.,  0.,  0.],
462:            [ 0.,  0.,  1.,  0.,  0.],
463:            [ 0.,  1.,  1.,  1.,  0.],
464:            [ 0.,  0.,  1.,  0.,  0.],
465:            [ 0.,  0.,  0.,  0.,  0.]])
466:     >>> ndimage.binary_dilation(a, structure=struct2).astype(a.dtype)
467:     array([[ 0.,  0.,  0.,  0.,  0.],
468:            [ 0.,  1.,  1.,  1.,  0.],
469:            [ 0.,  1.,  1.,  1.,  0.],
470:            [ 0.,  1.,  1.,  1.,  0.],
471:            [ 0.,  0.,  0.,  0.,  0.]])
472:     >>> ndimage.binary_dilation(a, structure=struct1,\\
473:     ... iterations=2).astype(a.dtype)
474:     array([[ 0.,  0.,  1.,  0.,  0.],
475:            [ 0.,  1.,  1.,  1.,  0.],
476:            [ 1.,  1.,  1.,  1.,  1.],
477:            [ 0.,  1.,  1.,  1.,  0.],
478:            [ 0.,  0.,  1.,  0.,  0.]])
479: 
480:     '''
481:     input = numpy.asarray(input)
482:     if structure is None:
483:         structure = generate_binary_structure(input.ndim, 1)
484:     origin = _ni_support._normalize_sequence(origin, input.ndim)
485:     structure = numpy.asarray(structure)
486:     structure = structure[tuple([slice(None, None, -1)] *
487:                                 structure.ndim)]
488:     for ii in range(len(origin)):
489:         origin[ii] = -origin[ii]
490:         if not structure.shape[ii] & 1:
491:             origin[ii] -= 1
492: 
493:     return _binary_erosion(input, structure, iterations, mask,
494:                            output, border_value, origin, 1, brute_force)
495: 
496: 
497: def binary_opening(input, structure=None, iterations=1, output=None,
498:                    origin=0):
499:     '''
500:     Multi-dimensional binary opening with the given structuring element.
501: 
502:     The *opening* of an input image by a structuring element is the
503:     *dilation* of the *erosion* of the image by the structuring element.
504: 
505:     Parameters
506:     ----------
507:     input : array_like
508:         Binary array_like to be opened. Non-zero (True) elements form
509:         the subset to be opened.
510:     structure : array_like, optional
511:         Structuring element used for the opening. Non-zero elements are
512:         considered True. If no structuring element is provided an element
513:         is generated with a square connectivity equal to one (i.e., only
514:         nearest neighbors are connected to the center, diagonally-connected
515:         elements are not considered neighbors).
516:     iterations : {int, float}, optional
517:         The erosion step of the opening, then the dilation step are each
518:         repeated `iterations` times (one, by default). If `iterations` is
519:         less than 1, each operation is repeated until the result does
520:         not change anymore.
521:     output : ndarray, optional
522:         Array of the same shape as input, into which the output is placed.
523:         By default, a new array is created.
524:     origin : int or tuple of ints, optional
525:         Placement of the filter, by default 0.
526: 
527:     Returns
528:     -------
529:     binary_opening : ndarray of bools
530:         Opening of the input by the structuring element.
531: 
532:     See also
533:     --------
534:     grey_opening, binary_closing, binary_erosion, binary_dilation,
535:     generate_binary_structure
536: 
537:     Notes
538:     -----
539:     *Opening* [1]_ is a mathematical morphology operation [2]_ that
540:     consists in the succession of an erosion and a dilation of the
541:     input with the same structuring element. Opening therefore removes
542:     objects smaller than the structuring element.
543: 
544:     Together with *closing* (`binary_closing`), opening can be used for
545:     noise removal.
546: 
547:     References
548:     ----------
549:     .. [1] http://en.wikipedia.org/wiki/Opening_%28morphology%29
550:     .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology
551: 
552:     Examples
553:     --------
554:     >>> from scipy import ndimage
555:     >>> a = np.zeros((5,5), dtype=int)
556:     >>> a[1:4, 1:4] = 1; a[4, 4] = 1
557:     >>> a
558:     array([[0, 0, 0, 0, 0],
559:            [0, 1, 1, 1, 0],
560:            [0, 1, 1, 1, 0],
561:            [0, 1, 1, 1, 0],
562:            [0, 0, 0, 0, 1]])
563:     >>> # Opening removes small objects
564:     >>> ndimage.binary_opening(a, structure=np.ones((3,3))).astype(int)
565:     array([[0, 0, 0, 0, 0],
566:            [0, 1, 1, 1, 0],
567:            [0, 1, 1, 1, 0],
568:            [0, 1, 1, 1, 0],
569:            [0, 0, 0, 0, 0]])
570:     >>> # Opening can also smooth corners
571:     >>> ndimage.binary_opening(a).astype(int)
572:     array([[0, 0, 0, 0, 0],
573:            [0, 0, 1, 0, 0],
574:            [0, 1, 1, 1, 0],
575:            [0, 0, 1, 0, 0],
576:            [0, 0, 0, 0, 0]])
577:     >>> # Opening is the dilation of the erosion of the input
578:     >>> ndimage.binary_erosion(a).astype(int)
579:     array([[0, 0, 0, 0, 0],
580:            [0, 0, 0, 0, 0],
581:            [0, 0, 1, 0, 0],
582:            [0, 0, 0, 0, 0],
583:            [0, 0, 0, 0, 0]])
584:     >>> ndimage.binary_dilation(ndimage.binary_erosion(a)).astype(int)
585:     array([[0, 0, 0, 0, 0],
586:            [0, 0, 1, 0, 0],
587:            [0, 1, 1, 1, 0],
588:            [0, 0, 1, 0, 0],
589:            [0, 0, 0, 0, 0]])
590: 
591:     '''
592:     input = numpy.asarray(input)
593:     if structure is None:
594:         rank = input.ndim
595:         structure = generate_binary_structure(rank, 1)
596: 
597:     tmp = binary_erosion(input, structure, iterations, None, None, 0,
598:                          origin)
599:     return binary_dilation(tmp, structure, iterations, None, output, 0,
600:                            origin)
601: 
602: 
603: def binary_closing(input, structure=None, iterations=1, output=None,
604:                    origin=0):
605:     '''
606:     Multi-dimensional binary closing with the given structuring element.
607: 
608:     The *closing* of an input image by a structuring element is the
609:     *erosion* of the *dilation* of the image by the structuring element.
610: 
611:     Parameters
612:     ----------
613:     input : array_like
614:         Binary array_like to be closed. Non-zero (True) elements form
615:         the subset to be closed.
616:     structure : array_like, optional
617:         Structuring element used for the closing. Non-zero elements are
618:         considered True. If no structuring element is provided an element
619:         is generated with a square connectivity equal to one (i.e., only
620:         nearest neighbors are connected to the center, diagonally-connected
621:         elements are not considered neighbors).
622:     iterations : {int, float}, optional
623:         The dilation step of the closing, then the erosion step are each
624:         repeated `iterations` times (one, by default). If iterations is
625:         less than 1, each operations is repeated until the result does
626:         not change anymore.
627:     output : ndarray, optional
628:         Array of the same shape as input, into which the output is placed.
629:         By default, a new array is created.
630:     origin : int or tuple of ints, optional
631:         Placement of the filter, by default 0.
632: 
633:     Returns
634:     -------
635:     binary_closing : ndarray of bools
636:         Closing of the input by the structuring element.
637: 
638:     See also
639:     --------
640:     grey_closing, binary_opening, binary_dilation, binary_erosion,
641:     generate_binary_structure
642: 
643:     Notes
644:     -----
645:     *Closing* [1]_ is a mathematical morphology operation [2]_ that
646:     consists in the succession of a dilation and an erosion of the
647:     input with the same structuring element. Closing therefore fills
648:     holes smaller than the structuring element.
649: 
650:     Together with *opening* (`binary_opening`), closing can be used for
651:     noise removal.
652: 
653:     References
654:     ----------
655:     .. [1] http://en.wikipedia.org/wiki/Closing_%28morphology%29
656:     .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology
657: 
658:     Examples
659:     --------
660:     >>> from scipy import ndimage
661:     >>> a = np.zeros((5,5), dtype=int)
662:     >>> a[1:-1, 1:-1] = 1; a[2,2] = 0
663:     >>> a
664:     array([[0, 0, 0, 0, 0],
665:            [0, 1, 1, 1, 0],
666:            [0, 1, 0, 1, 0],
667:            [0, 1, 1, 1, 0],
668:            [0, 0, 0, 0, 0]])
669:     >>> # Closing removes small holes
670:     >>> ndimage.binary_closing(a).astype(int)
671:     array([[0, 0, 0, 0, 0],
672:            [0, 1, 1, 1, 0],
673:            [0, 1, 1, 1, 0],
674:            [0, 1, 1, 1, 0],
675:            [0, 0, 0, 0, 0]])
676:     >>> # Closing is the erosion of the dilation of the input
677:     >>> ndimage.binary_dilation(a).astype(int)
678:     array([[0, 1, 1, 1, 0],
679:            [1, 1, 1, 1, 1],
680:            [1, 1, 1, 1, 1],
681:            [1, 1, 1, 1, 1],
682:            [0, 1, 1, 1, 0]])
683:     >>> ndimage.binary_erosion(ndimage.binary_dilation(a)).astype(int)
684:     array([[0, 0, 0, 0, 0],
685:            [0, 1, 1, 1, 0],
686:            [0, 1, 1, 1, 0],
687:            [0, 1, 1, 1, 0],
688:            [0, 0, 0, 0, 0]])
689: 
690: 
691:     >>> a = np.zeros((7,7), dtype=int)
692:     >>> a[1:6, 2:5] = 1; a[1:3,3] = 0
693:     >>> a
694:     array([[0, 0, 0, 0, 0, 0, 0],
695:            [0, 0, 1, 0, 1, 0, 0],
696:            [0, 0, 1, 0, 1, 0, 0],
697:            [0, 0, 1, 1, 1, 0, 0],
698:            [0, 0, 1, 1, 1, 0, 0],
699:            [0, 0, 1, 1, 1, 0, 0],
700:            [0, 0, 0, 0, 0, 0, 0]])
701:     >>> # In addition to removing holes, closing can also
702:     >>> # coarsen boundaries with fine hollows.
703:     >>> ndimage.binary_closing(a).astype(int)
704:     array([[0, 0, 0, 0, 0, 0, 0],
705:            [0, 0, 1, 0, 1, 0, 0],
706:            [0, 0, 1, 1, 1, 0, 0],
707:            [0, 0, 1, 1, 1, 0, 0],
708:            [0, 0, 1, 1, 1, 0, 0],
709:            [0, 0, 1, 1, 1, 0, 0],
710:            [0, 0, 0, 0, 0, 0, 0]])
711:     >>> ndimage.binary_closing(a, structure=np.ones((2,2))).astype(int)
712:     array([[0, 0, 0, 0, 0, 0, 0],
713:            [0, 0, 1, 1, 1, 0, 0],
714:            [0, 0, 1, 1, 1, 0, 0],
715:            [0, 0, 1, 1, 1, 0, 0],
716:            [0, 0, 1, 1, 1, 0, 0],
717:            [0, 0, 1, 1, 1, 0, 0],
718:            [0, 0, 0, 0, 0, 0, 0]])
719: 
720:     '''
721:     input = numpy.asarray(input)
722:     if structure is None:
723:         rank = input.ndim
724:         structure = generate_binary_structure(rank, 1)
725: 
726:     tmp = binary_dilation(input, structure, iterations, None, None, 0,
727:                           origin)
728:     return binary_erosion(tmp, structure, iterations, None, output, 0,
729:                           origin)
730: 
731: 
732: def binary_hit_or_miss(input, structure1=None, structure2=None,
733:                        output=None, origin1=0, origin2=None):
734:     '''
735:     Multi-dimensional binary hit-or-miss transform.
736: 
737:     The hit-or-miss transform finds the locations of a given pattern
738:     inside the input image.
739: 
740:     Parameters
741:     ----------
742:     input : array_like (cast to booleans)
743:         Binary image where a pattern is to be detected.
744:     structure1 : array_like (cast to booleans), optional
745:         Part of the structuring element to be fitted to the foreground
746:         (non-zero elements) of `input`. If no value is provided, a
747:         structure of square connectivity 1 is chosen.
748:     structure2 : array_like (cast to booleans), optional
749:         Second part of the structuring element that has to miss completely
750:         the foreground. If no value is provided, the complementary of
751:         `structure1` is taken.
752:     output : ndarray, optional
753:         Array of the same shape as input, into which the output is placed.
754:         By default, a new array is created.
755:     origin1 : int or tuple of ints, optional
756:         Placement of the first part of the structuring element `structure1`,
757:         by default 0 for a centered structure.
758:     origin2 : int or tuple of ints, optional
759:         Placement of the second part of the structuring element `structure2`,
760:         by default 0 for a centered structure. If a value is provided for
761:         `origin1` and not for `origin2`, then `origin2` is set to `origin1`.
762: 
763:     Returns
764:     -------
765:     binary_hit_or_miss : ndarray
766:         Hit-or-miss transform of `input` with the given structuring
767:         element (`structure1`, `structure2`).
768: 
769:     See also
770:     --------
771:     ndimage.morphology, binary_erosion
772: 
773:     References
774:     ----------
775:     .. [1] http://en.wikipedia.org/wiki/Hit-or-miss_transform
776: 
777:     Examples
778:     --------
779:     >>> from scipy import ndimage
780:     >>> a = np.zeros((7,7), dtype=int)
781:     >>> a[1, 1] = 1; a[2:4, 2:4] = 1; a[4:6, 4:6] = 1
782:     >>> a
783:     array([[0, 0, 0, 0, 0, 0, 0],
784:            [0, 1, 0, 0, 0, 0, 0],
785:            [0, 0, 1, 1, 0, 0, 0],
786:            [0, 0, 1, 1, 0, 0, 0],
787:            [0, 0, 0, 0, 1, 1, 0],
788:            [0, 0, 0, 0, 1, 1, 0],
789:            [0, 0, 0, 0, 0, 0, 0]])
790:     >>> structure1 = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
791:     >>> structure1
792:     array([[1, 0, 0],
793:            [0, 1, 1],
794:            [0, 1, 1]])
795:     >>> # Find the matches of structure1 in the array a
796:     >>> ndimage.binary_hit_or_miss(a, structure1=structure1).astype(int)
797:     array([[0, 0, 0, 0, 0, 0, 0],
798:            [0, 0, 0, 0, 0, 0, 0],
799:            [0, 0, 1, 0, 0, 0, 0],
800:            [0, 0, 0, 0, 0, 0, 0],
801:            [0, 0, 0, 0, 1, 0, 0],
802:            [0, 0, 0, 0, 0, 0, 0],
803:            [0, 0, 0, 0, 0, 0, 0]])
804:     >>> # Change the origin of the filter
805:     >>> # origin1=1 is equivalent to origin1=(1,1) here
806:     >>> ndimage.binary_hit_or_miss(a, structure1=structure1,\\
807:     ... origin1=1).astype(int)
808:     array([[0, 0, 0, 0, 0, 0, 0],
809:            [0, 0, 0, 0, 0, 0, 0],
810:            [0, 0, 0, 0, 0, 0, 0],
811:            [0, 0, 0, 1, 0, 0, 0],
812:            [0, 0, 0, 0, 0, 0, 0],
813:            [0, 0, 0, 0, 0, 1, 0],
814:            [0, 0, 0, 0, 0, 0, 0]])
815: 
816:     '''
817:     input = numpy.asarray(input)
818:     if structure1 is None:
819:         structure1 = generate_binary_structure(input.ndim, 1)
820:     if structure2 is None:
821:         structure2 = numpy.logical_not(structure1)
822:     origin1 = _ni_support._normalize_sequence(origin1, input.ndim)
823:     if origin2 is None:
824:         origin2 = origin1
825:     else:
826:         origin2 = _ni_support._normalize_sequence(origin2, input.ndim)
827: 
828:     tmp1 = _binary_erosion(input, structure1, 1, None, None, 0, origin1,
829:                            0, False)
830:     inplace = isinstance(output, numpy.ndarray)
831:     result = _binary_erosion(input, structure2, 1, None, output, 0,
832:                              origin2, 1, False)
833:     if inplace:
834:         numpy.logical_not(output, output)
835:         numpy.logical_and(tmp1, output, output)
836:     else:
837:         numpy.logical_not(result, result)
838:         return numpy.logical_and(tmp1, result)
839: 
840: 
841: def binary_propagation(input, structure=None, mask=None,
842:                        output=None, border_value=0, origin=0):
843:     '''
844:     Multi-dimensional binary propagation with the given structuring element.
845: 
846:     Parameters
847:     ----------
848:     input : array_like
849:         Binary image to be propagated inside `mask`.
850:     structure : array_like, optional
851:         Structuring element used in the successive dilations. The output
852:         may depend on the structuring element, especially if `mask` has
853:         several connex components. If no structuring element is
854:         provided, an element is generated with a squared connectivity equal
855:         to one.
856:     mask : array_like, optional
857:         Binary mask defining the region into which `input` is allowed to
858:         propagate.
859:     output : ndarray, optional
860:         Array of the same shape as input, into which the output is placed.
861:         By default, a new array is created.
862:     border_value : int (cast to 0 or 1), optional
863:         Value at the border in the output array.
864:     origin : int or tuple of ints, optional
865:         Placement of the filter, by default 0.
866: 
867:     Returns
868:     -------
869:     binary_propagation : ndarray
870:         Binary propagation of `input` inside `mask`.
871: 
872:     Notes
873:     -----
874:     This function is functionally equivalent to calling binary_dilation
875:     with the number of iterations less than one: iterative dilation until
876:     the result does not change anymore.
877: 
878:     The succession of an erosion and propagation inside the original image
879:     can be used instead of an *opening* for deleting small objects while
880:     keeping the contours of larger objects untouched.
881: 
882:     References
883:     ----------
884:     .. [1] http://cmm.ensmp.fr/~serra/cours/pdf/en/ch6en.pdf, slide 15.
885:     .. [2] I.T. Young, J.J. Gerbrands, and L.J. van Vliet, "Fundamentals of
886:         image processing", 1998
887:         ftp://qiftp.tudelft.nl/DIPimage/docs/FIP2.3.pdf
888: 
889:     Examples
890:     --------
891:     >>> from scipy import ndimage
892:     >>> input = np.zeros((8, 8), dtype=int)
893:     >>> input[2, 2] = 1
894:     >>> mask = np.zeros((8, 8), dtype=int)
895:     >>> mask[1:4, 1:4] = mask[4, 4]  = mask[6:8, 6:8] = 1
896:     >>> input
897:     array([[0, 0, 0, 0, 0, 0, 0, 0],
898:            [0, 0, 0, 0, 0, 0, 0, 0],
899:            [0, 0, 1, 0, 0, 0, 0, 0],
900:            [0, 0, 0, 0, 0, 0, 0, 0],
901:            [0, 0, 0, 0, 0, 0, 0, 0],
902:            [0, 0, 0, 0, 0, 0, 0, 0],
903:            [0, 0, 0, 0, 0, 0, 0, 0],
904:            [0, 0, 0, 0, 0, 0, 0, 0]])
905:     >>> mask
906:     array([[0, 0, 0, 0, 0, 0, 0, 0],
907:            [0, 1, 1, 1, 0, 0, 0, 0],
908:            [0, 1, 1, 1, 0, 0, 0, 0],
909:            [0, 1, 1, 1, 0, 0, 0, 0],
910:            [0, 0, 0, 0, 1, 0, 0, 0],
911:            [0, 0, 0, 0, 0, 0, 0, 0],
912:            [0, 0, 0, 0, 0, 0, 1, 1],
913:            [0, 0, 0, 0, 0, 0, 1, 1]])
914:     >>> ndimage.binary_propagation(input, mask=mask).astype(int)
915:     array([[0, 0, 0, 0, 0, 0, 0, 0],
916:            [0, 1, 1, 1, 0, 0, 0, 0],
917:            [0, 1, 1, 1, 0, 0, 0, 0],
918:            [0, 1, 1, 1, 0, 0, 0, 0],
919:            [0, 0, 0, 0, 0, 0, 0, 0],
920:            [0, 0, 0, 0, 0, 0, 0, 0],
921:            [0, 0, 0, 0, 0, 0, 0, 0],
922:            [0, 0, 0, 0, 0, 0, 0, 0]])
923:     >>> ndimage.binary_propagation(input, mask=mask,\\
924:     ... structure=np.ones((3,3))).astype(int)
925:     array([[0, 0, 0, 0, 0, 0, 0, 0],
926:            [0, 1, 1, 1, 0, 0, 0, 0],
927:            [0, 1, 1, 1, 0, 0, 0, 0],
928:            [0, 1, 1, 1, 0, 0, 0, 0],
929:            [0, 0, 0, 0, 1, 0, 0, 0],
930:            [0, 0, 0, 0, 0, 0, 0, 0],
931:            [0, 0, 0, 0, 0, 0, 0, 0],
932:            [0, 0, 0, 0, 0, 0, 0, 0]])
933: 
934:     >>> # Comparison between opening and erosion+propagation
935:     >>> a = np.zeros((6,6), dtype=int)
936:     >>> a[2:5, 2:5] = 1; a[0, 0] = 1; a[5, 5] = 1
937:     >>> a
938:     array([[1, 0, 0, 0, 0, 0],
939:            [0, 0, 0, 0, 0, 0],
940:            [0, 0, 1, 1, 1, 0],
941:            [0, 0, 1, 1, 1, 0],
942:            [0, 0, 1, 1, 1, 0],
943:            [0, 0, 0, 0, 0, 1]])
944:     >>> ndimage.binary_opening(a).astype(int)
945:     array([[0, 0, 0, 0, 0, 0],
946:            [0, 0, 0, 0, 0, 0],
947:            [0, 0, 0, 1, 0, 0],
948:            [0, 0, 1, 1, 1, 0],
949:            [0, 0, 0, 1, 0, 0],
950:            [0, 0, 0, 0, 0, 0]])
951:     >>> b = ndimage.binary_erosion(a)
952:     >>> b.astype(int)
953:     array([[0, 0, 0, 0, 0, 0],
954:            [0, 0, 0, 0, 0, 0],
955:            [0, 0, 0, 0, 0, 0],
956:            [0, 0, 0, 1, 0, 0],
957:            [0, 0, 0, 0, 0, 0],
958:            [0, 0, 0, 0, 0, 0]])
959:     >>> ndimage.binary_propagation(b, mask=a).astype(int)
960:     array([[0, 0, 0, 0, 0, 0],
961:            [0, 0, 0, 0, 0, 0],
962:            [0, 0, 1, 1, 1, 0],
963:            [0, 0, 1, 1, 1, 0],
964:            [0, 0, 1, 1, 1, 0],
965:            [0, 0, 0, 0, 0, 0]])
966: 
967:     '''
968:     return binary_dilation(input, structure, -1, mask, output,
969:                            border_value, origin)
970: 
971: 
972: def binary_fill_holes(input, structure=None, output=None, origin=0):
973:     '''
974:     Fill the holes in binary objects.
975: 
976: 
977:     Parameters
978:     ----------
979:     input : array_like
980:         n-dimensional binary array with holes to be filled
981:     structure : array_like, optional
982:         Structuring element used in the computation; large-size elements
983:         make computations faster but may miss holes separated from the
984:         background by thin regions. The default element (with a square
985:         connectivity equal to one) yields the intuitive result where all
986:         holes in the input have been filled.
987:     output : ndarray, optional
988:         Array of the same shape as input, into which the output is placed.
989:         By default, a new array is created.
990:     origin : int, tuple of ints, optional
991:         Position of the structuring element.
992: 
993:     Returns
994:     -------
995:     out : ndarray
996:         Transformation of the initial image `input` where holes have been
997:         filled.
998: 
999:     See also
1000:     --------
1001:     binary_dilation, binary_propagation, label
1002: 
1003:     Notes
1004:     -----
1005:     The algorithm used in this function consists in invading the complementary
1006:     of the shapes in `input` from the outer boundary of the image,
1007:     using binary dilations. Holes are not connected to the boundary and are
1008:     therefore not invaded. The result is the complementary subset of the
1009:     invaded region.
1010: 
1011:     References
1012:     ----------
1013:     .. [1] http://en.wikipedia.org/wiki/Mathematical_morphology
1014: 
1015: 
1016:     Examples
1017:     --------
1018:     >>> from scipy import ndimage
1019:     >>> a = np.zeros((5, 5), dtype=int)
1020:     >>> a[1:4, 1:4] = 1
1021:     >>> a[2,2] = 0
1022:     >>> a
1023:     array([[0, 0, 0, 0, 0],
1024:            [0, 1, 1, 1, 0],
1025:            [0, 1, 0, 1, 0],
1026:            [0, 1, 1, 1, 0],
1027:            [0, 0, 0, 0, 0]])
1028:     >>> ndimage.binary_fill_holes(a).astype(int)
1029:     array([[0, 0, 0, 0, 0],
1030:            [0, 1, 1, 1, 0],
1031:            [0, 1, 1, 1, 0],
1032:            [0, 1, 1, 1, 0],
1033:            [0, 0, 0, 0, 0]])
1034:     >>> # Too big structuring element
1035:     >>> ndimage.binary_fill_holes(a, structure=np.ones((5,5))).astype(int)
1036:     array([[0, 0, 0, 0, 0],
1037:            [0, 1, 1, 1, 0],
1038:            [0, 1, 0, 1, 0],
1039:            [0, 1, 1, 1, 0],
1040:            [0, 0, 0, 0, 0]])
1041: 
1042:     '''
1043:     mask = numpy.logical_not(input)
1044:     tmp = numpy.zeros(mask.shape, bool)
1045:     inplace = isinstance(output, numpy.ndarray)
1046:     if inplace:
1047:         binary_dilation(tmp, structure, -1, mask, output, 1, origin)
1048:         numpy.logical_not(output, output)
1049:     else:
1050:         output = binary_dilation(tmp, structure, -1, mask, None, 1,
1051:                                  origin)
1052:         numpy.logical_not(output, output)
1053:         return output
1054: 
1055: 
1056: def grey_erosion(input, size=None, footprint=None, structure=None,
1057:                  output=None, mode="reflect", cval=0.0, origin=0):
1058:     '''
1059:     Calculate a greyscale erosion, using either a structuring element,
1060:     or a footprint corresponding to a flat structuring element.
1061: 
1062:     Grayscale erosion is a mathematical morphology operation. For the
1063:     simple case of a full and flat structuring element, it can be viewed
1064:     as a minimum filter over a sliding window.
1065: 
1066:     Parameters
1067:     ----------
1068:     input : array_like
1069:         Array over which the grayscale erosion is to be computed.
1070:     size : tuple of ints
1071:         Shape of a flat and full structuring element used for the grayscale
1072:         erosion. Optional if `footprint` or `structure` is provided.
1073:     footprint : array of ints, optional
1074:         Positions of non-infinite elements of a flat structuring element
1075:         used for the grayscale erosion. Non-zero values give the set of
1076:         neighbors of the center over which the minimum is chosen.
1077:     structure : array of ints, optional
1078:         Structuring element used for the grayscale erosion. `structure`
1079:         may be a non-flat structuring element.
1080:     output : array, optional
1081:         An array used for storing the ouput of the erosion may be provided.
1082:     mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
1083:         The `mode` parameter determines how the array borders are
1084:         handled, where `cval` is the value when mode is equal to
1085:         'constant'. Default is 'reflect'
1086:     cval : scalar, optional
1087:         Value to fill past edges of input if `mode` is 'constant'. Default
1088:         is 0.0.
1089:     origin : scalar, optional
1090:         The `origin` parameter controls the placement of the filter.
1091:         Default 0
1092: 
1093:     Returns
1094:     -------
1095:     output : ndarray
1096:         Grayscale erosion of `input`.
1097: 
1098:     See also
1099:     --------
1100:     binary_erosion, grey_dilation, grey_opening, grey_closing
1101:     generate_binary_structure, ndimage.minimum_filter
1102: 
1103:     Notes
1104:     -----
1105:     The grayscale erosion of an image input by a structuring element s defined
1106:     over a domain E is given by:
1107: 
1108:     (input+s)(x) = min {input(y) - s(x-y), for y in E}
1109: 
1110:     In particular, for structuring elements defined as
1111:     s(y) = 0 for y in E, the grayscale erosion computes the minimum of the
1112:     input image inside a sliding window defined by E.
1113: 
1114:     Grayscale erosion [1]_ is a *mathematical morphology* operation [2]_.
1115: 
1116:     References
1117:     ----------
1118:     .. [1] http://en.wikipedia.org/wiki/Erosion_%28morphology%29
1119:     .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology
1120: 
1121:     Examples
1122:     --------
1123:     >>> from scipy import ndimage
1124:     >>> a = np.zeros((7,7), dtype=int)
1125:     >>> a[1:6, 1:6] = 3
1126:     >>> a[4,4] = 2; a[2,3] = 1
1127:     >>> a
1128:     array([[0, 0, 0, 0, 0, 0, 0],
1129:            [0, 3, 3, 3, 3, 3, 0],
1130:            [0, 3, 3, 1, 3, 3, 0],
1131:            [0, 3, 3, 3, 3, 3, 0],
1132:            [0, 3, 3, 3, 2, 3, 0],
1133:            [0, 3, 3, 3, 3, 3, 0],
1134:            [0, 0, 0, 0, 0, 0, 0]])
1135:     >>> ndimage.grey_erosion(a, size=(3,3))
1136:     array([[0, 0, 0, 0, 0, 0, 0],
1137:            [0, 0, 0, 0, 0, 0, 0],
1138:            [0, 0, 1, 1, 1, 0, 0],
1139:            [0, 0, 1, 1, 1, 0, 0],
1140:            [0, 0, 3, 2, 2, 0, 0],
1141:            [0, 0, 0, 0, 0, 0, 0],
1142:            [0, 0, 0, 0, 0, 0, 0]])
1143:     >>> footprint = ndimage.generate_binary_structure(2, 1)
1144:     >>> footprint
1145:     array([[False,  True, False],
1146:            [ True,  True,  True],
1147:            [False,  True, False]], dtype=bool)
1148:     >>> # Diagonally-connected elements are not considered neighbors
1149:     >>> ndimage.grey_erosion(a, size=(3,3), footprint=footprint)
1150:     array([[0, 0, 0, 0, 0, 0, 0],
1151:            [0, 0, 0, 0, 0, 0, 0],
1152:            [0, 0, 1, 1, 1, 0, 0],
1153:            [0, 0, 3, 1, 2, 0, 0],
1154:            [0, 0, 3, 2, 2, 0, 0],
1155:            [0, 0, 0, 0, 0, 0, 0],
1156:            [0, 0, 0, 0, 0, 0, 0]])
1157: 
1158:     '''
1159:     if size is None and footprint is None and structure is None:
1160:         raise ValueError("size, footprint or structure must be specified")
1161: 
1162:     return filters._min_or_max_filter(input, size, footprint, structure,
1163:                                       output, mode, cval, origin, 1)
1164: 
1165: 
1166: def grey_dilation(input, size=None, footprint=None, structure=None,
1167:                  output=None, mode="reflect", cval=0.0, origin=0):
1168:     '''
1169:     Calculate a greyscale dilation, using either a structuring element,
1170:     or a footprint corresponding to a flat structuring element.
1171: 
1172:     Grayscale dilation is a mathematical morphology operation. For the
1173:     simple case of a full and flat structuring element, it can be viewed
1174:     as a maximum filter over a sliding window.
1175: 
1176:     Parameters
1177:     ----------
1178:     input : array_like
1179:         Array over which the grayscale dilation is to be computed.
1180:     size : tuple of ints
1181:         Shape of a flat and full structuring element used for the grayscale
1182:         dilation. Optional if `footprint` or `structure` is provided.
1183:     footprint : array of ints, optional
1184:         Positions of non-infinite elements of a flat structuring element
1185:         used for the grayscale dilation. Non-zero values give the set of
1186:         neighbors of the center over which the maximum is chosen.
1187:     structure : array of ints, optional
1188:         Structuring element used for the grayscale dilation. `structure`
1189:         may be a non-flat structuring element.
1190:     output : array, optional
1191:         An array used for storing the ouput of the dilation may be provided.
1192:     mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
1193:         The `mode` parameter determines how the array borders are
1194:         handled, where `cval` is the value when mode is equal to
1195:         'constant'. Default is 'reflect'
1196:     cval : scalar, optional
1197:         Value to fill past edges of input if `mode` is 'constant'. Default
1198:         is 0.0.
1199:     origin : scalar, optional
1200:         The `origin` parameter controls the placement of the filter.
1201:         Default 0
1202: 
1203:     Returns
1204:     -------
1205:     grey_dilation : ndarray
1206:         Grayscale dilation of `input`.
1207: 
1208:     See also
1209:     --------
1210:     binary_dilation, grey_erosion, grey_closing, grey_opening
1211:     generate_binary_structure, ndimage.maximum_filter
1212: 
1213:     Notes
1214:     -----
1215:     The grayscale dilation of an image input by a structuring element s defined
1216:     over a domain E is given by:
1217: 
1218:     (input+s)(x) = max {input(y) + s(x-y), for y in E}
1219: 
1220:     In particular, for structuring elements defined as
1221:     s(y) = 0 for y in E, the grayscale dilation computes the maximum of the
1222:     input image inside a sliding window defined by E.
1223: 
1224:     Grayscale dilation [1]_ is a *mathematical morphology* operation [2]_.
1225: 
1226:     References
1227:     ----------
1228:     .. [1] http://en.wikipedia.org/wiki/Dilation_%28morphology%29
1229:     .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology
1230: 
1231:     Examples
1232:     --------
1233:     >>> from scipy import ndimage
1234:     >>> a = np.zeros((7,7), dtype=int)
1235:     >>> a[2:5, 2:5] = 1
1236:     >>> a[4,4] = 2; a[2,3] = 3
1237:     >>> a
1238:     array([[0, 0, 0, 0, 0, 0, 0],
1239:            [0, 0, 0, 0, 0, 0, 0],
1240:            [0, 0, 1, 3, 1, 0, 0],
1241:            [0, 0, 1, 1, 1, 0, 0],
1242:            [0, 0, 1, 1, 2, 0, 0],
1243:            [0, 0, 0, 0, 0, 0, 0],
1244:            [0, 0, 0, 0, 0, 0, 0]])
1245:     >>> ndimage.grey_dilation(a, size=(3,3))
1246:     array([[0, 0, 0, 0, 0, 0, 0],
1247:            [0, 1, 3, 3, 3, 1, 0],
1248:            [0, 1, 3, 3, 3, 1, 0],
1249:            [0, 1, 3, 3, 3, 2, 0],
1250:            [0, 1, 1, 2, 2, 2, 0],
1251:            [0, 1, 1, 2, 2, 2, 0],
1252:            [0, 0, 0, 0, 0, 0, 0]])
1253:     >>> ndimage.grey_dilation(a, footprint=np.ones((3,3)))
1254:     array([[0, 0, 0, 0, 0, 0, 0],
1255:            [0, 1, 3, 3, 3, 1, 0],
1256:            [0, 1, 3, 3, 3, 1, 0],
1257:            [0, 1, 3, 3, 3, 2, 0],
1258:            [0, 1, 1, 2, 2, 2, 0],
1259:            [0, 1, 1, 2, 2, 2, 0],
1260:            [0, 0, 0, 0, 0, 0, 0]])
1261:     >>> s = ndimage.generate_binary_structure(2,1)
1262:     >>> s
1263:     array([[False,  True, False],
1264:            [ True,  True,  True],
1265:            [False,  True, False]], dtype=bool)
1266:     >>> ndimage.grey_dilation(a, footprint=s)
1267:     array([[0, 0, 0, 0, 0, 0, 0],
1268:            [0, 0, 1, 3, 1, 0, 0],
1269:            [0, 1, 3, 3, 3, 1, 0],
1270:            [0, 1, 1, 3, 2, 1, 0],
1271:            [0, 1, 1, 2, 2, 2, 0],
1272:            [0, 0, 1, 1, 2, 0, 0],
1273:            [0, 0, 0, 0, 0, 0, 0]])
1274:     >>> ndimage.grey_dilation(a, size=(3,3), structure=np.ones((3,3)))
1275:     array([[1, 1, 1, 1, 1, 1, 1],
1276:            [1, 2, 4, 4, 4, 2, 1],
1277:            [1, 2, 4, 4, 4, 2, 1],
1278:            [1, 2, 4, 4, 4, 3, 1],
1279:            [1, 2, 2, 3, 3, 3, 1],
1280:            [1, 2, 2, 3, 3, 3, 1],
1281:            [1, 1, 1, 1, 1, 1, 1]])
1282: 
1283:     '''
1284:     if size is None and footprint is None and structure is None:
1285:         raise ValueError("size, footprint or structure must be specified")
1286:     if structure is not None:
1287:         structure = numpy.asarray(structure)
1288:         structure = structure[tuple([slice(None, None, -1)] *
1289:                                     structure.ndim)]
1290:     if footprint is not None:
1291:         footprint = numpy.asarray(footprint)
1292:         footprint = footprint[tuple([slice(None, None, -1)] *
1293:                                     footprint.ndim)]
1294: 
1295:     input = numpy.asarray(input)
1296:     origin = _ni_support._normalize_sequence(origin, input.ndim)
1297:     for ii in range(len(origin)):
1298:         origin[ii] = -origin[ii]
1299:         if footprint is not None:
1300:             sz = footprint.shape[ii]
1301:         elif structure is not None:
1302:             sz = structure.shape[ii]
1303:         elif numpy.isscalar(size):
1304:             sz = size
1305:         else:
1306:             sz = size[ii]
1307:         if not sz & 1:
1308:             origin[ii] -= 1
1309: 
1310:     return filters._min_or_max_filter(input, size, footprint, structure,
1311:                                       output, mode, cval, origin, 0)
1312: 
1313: 
1314: def grey_opening(input, size=None, footprint=None, structure=None,
1315:                  output=None, mode="reflect", cval=0.0, origin=0):
1316:     '''
1317:     Multi-dimensional greyscale opening.
1318: 
1319:     A greyscale opening consists in the succession of a greyscale erosion,
1320:     and a greyscale dilation.
1321: 
1322:     Parameters
1323:     ----------
1324:     input : array_like
1325:         Array over which the grayscale opening is to be computed.
1326:     size : tuple of ints
1327:         Shape of a flat and full structuring element used for the grayscale
1328:         opening. Optional if `footprint` or `structure` is provided.
1329:     footprint : array of ints, optional
1330:         Positions of non-infinite elements of a flat structuring element
1331:         used for the grayscale opening.
1332:     structure : array of ints, optional
1333:         Structuring element used for the grayscale opening. `structure`
1334:         may be a non-flat structuring element.
1335:     output : array, optional
1336:         An array used for storing the ouput of the opening may be provided.
1337:     mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
1338:         The `mode` parameter determines how the array borders are
1339:         handled, where `cval` is the value when mode is equal to
1340:         'constant'. Default is 'reflect'
1341:     cval : scalar, optional
1342:         Value to fill past edges of input if `mode` is 'constant'. Default
1343:         is 0.0.
1344:     origin : scalar, optional
1345:         The `origin` parameter controls the placement of the filter.
1346:         Default 0
1347: 
1348:     Returns
1349:     -------
1350:     grey_opening : ndarray
1351:         Result of the grayscale opening of `input` with `structure`.
1352: 
1353:     See also
1354:     --------
1355:     binary_opening, grey_dilation, grey_erosion, grey_closing
1356:     generate_binary_structure
1357: 
1358:     Notes
1359:     -----
1360:     The action of a grayscale opening with a flat structuring element amounts
1361:     to smoothen high local maxima, whereas binary opening erases small objects.
1362: 
1363:     References
1364:     ----------
1365:     .. [1] http://en.wikipedia.org/wiki/Mathematical_morphology
1366: 
1367:     Examples
1368:     --------
1369:     >>> from scipy import ndimage
1370:     >>> a = np.arange(36).reshape((6,6))
1371:     >>> a[3, 3] = 50
1372:     >>> a
1373:     array([[ 0,  1,  2,  3,  4,  5],
1374:            [ 6,  7,  8,  9, 10, 11],
1375:            [12, 13, 14, 15, 16, 17],
1376:            [18, 19, 20, 50, 22, 23],
1377:            [24, 25, 26, 27, 28, 29],
1378:            [30, 31, 32, 33, 34, 35]])
1379:     >>> ndimage.grey_opening(a, size=(3,3))
1380:     array([[ 0,  1,  2,  3,  4,  4],
1381:            [ 6,  7,  8,  9, 10, 10],
1382:            [12, 13, 14, 15, 16, 16],
1383:            [18, 19, 20, 22, 22, 22],
1384:            [24, 25, 26, 27, 28, 28],
1385:            [24, 25, 26, 27, 28, 28]])
1386:     >>> # Note that the local maximum a[3,3] has disappeared
1387: 
1388:     '''
1389:     tmp = grey_erosion(input, size, footprint, structure, None, mode,
1390:                        cval, origin)
1391:     return grey_dilation(tmp, size, footprint, structure, output, mode,
1392:                          cval, origin)
1393: 
1394: 
1395: def grey_closing(input, size=None, footprint=None, structure=None,
1396:                  output=None, mode="reflect", cval=0.0, origin=0):
1397:     '''
1398:     Multi-dimensional greyscale closing.
1399: 
1400:     A greyscale closing consists in the succession of a greyscale dilation,
1401:     and a greyscale erosion.
1402: 
1403:     Parameters
1404:     ----------
1405:     input : array_like
1406:         Array over which the grayscale closing is to be computed.
1407:     size : tuple of ints
1408:         Shape of a flat and full structuring element used for the grayscale
1409:         closing. Optional if `footprint` or `structure` is provided.
1410:     footprint : array of ints, optional
1411:         Positions of non-infinite elements of a flat structuring element
1412:         used for the grayscale closing.
1413:     structure : array of ints, optional
1414:         Structuring element used for the grayscale closing. `structure`
1415:         may be a non-flat structuring element.
1416:     output : array, optional
1417:         An array used for storing the ouput of the closing may be provided.
1418:     mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
1419:         The `mode` parameter determines how the array borders are
1420:         handled, where `cval` is the value when mode is equal to
1421:         'constant'. Default is 'reflect'
1422:     cval : scalar, optional
1423:         Value to fill past edges of input if `mode` is 'constant'. Default
1424:         is 0.0.
1425:     origin : scalar, optional
1426:         The `origin` parameter controls the placement of the filter.
1427:         Default 0
1428: 
1429:     Returns
1430:     -------
1431:     grey_closing : ndarray
1432:         Result of the grayscale closing of `input` with `structure`.
1433: 
1434:     See also
1435:     --------
1436:     binary_closing, grey_dilation, grey_erosion, grey_opening,
1437:     generate_binary_structure
1438: 
1439:     Notes
1440:     -----
1441:     The action of a grayscale closing with a flat structuring element amounts
1442:     to smoothen deep local minima, whereas binary closing fills small holes.
1443: 
1444:     References
1445:     ----------
1446:     .. [1] http://en.wikipedia.org/wiki/Mathematical_morphology
1447: 
1448:     Examples
1449:     --------
1450:     >>> from scipy import ndimage
1451:     >>> a = np.arange(36).reshape((6,6))
1452:     >>> a[3,3] = 0
1453:     >>> a
1454:     array([[ 0,  1,  2,  3,  4,  5],
1455:            [ 6,  7,  8,  9, 10, 11],
1456:            [12, 13, 14, 15, 16, 17],
1457:            [18, 19, 20,  0, 22, 23],
1458:            [24, 25, 26, 27, 28, 29],
1459:            [30, 31, 32, 33, 34, 35]])
1460:     >>> ndimage.grey_closing(a, size=(3,3))
1461:     array([[ 7,  7,  8,  9, 10, 11],
1462:            [ 7,  7,  8,  9, 10, 11],
1463:            [13, 13, 14, 15, 16, 17],
1464:            [19, 19, 20, 20, 22, 23],
1465:            [25, 25, 26, 27, 28, 29],
1466:            [31, 31, 32, 33, 34, 35]])
1467:     >>> # Note that the local minimum a[3,3] has disappeared
1468: 
1469:     '''
1470:     tmp = grey_dilation(input, size, footprint, structure, None, mode,
1471:                         cval, origin)
1472:     return grey_erosion(tmp, size, footprint, structure, output, mode,
1473:                         cval, origin)
1474: 
1475: 
1476: def morphological_gradient(input, size=None, footprint=None,
1477:                         structure=None, output=None, mode="reflect",
1478:                         cval=0.0, origin=0):
1479:     '''
1480:     Multi-dimensional morphological gradient.
1481: 
1482:     The morphological gradient is calculated as the difference between a
1483:     dilation and an erosion of the input with a given structuring element.
1484: 
1485:     Parameters
1486:     ----------
1487:     input : array_like
1488:         Array over which to compute the morphlogical gradient.
1489:     size : tuple of ints
1490:         Shape of a flat and full structuring element used for the mathematical
1491:         morphology operations. Optional if `footprint` or `structure` is
1492:         provided. A larger `size` yields a more blurred gradient.
1493:     footprint : array of ints, optional
1494:         Positions of non-infinite elements of a flat structuring element
1495:         used for the morphology operations. Larger footprints
1496:         give a more blurred morphological gradient.
1497:     structure : array of ints, optional
1498:         Structuring element used for the morphology operations.
1499:         `structure` may be a non-flat structuring element.
1500:     output : array, optional
1501:         An array used for storing the ouput of the morphological gradient
1502:         may be provided.
1503:     mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
1504:         The `mode` parameter determines how the array borders are
1505:         handled, where `cval` is the value when mode is equal to
1506:         'constant'. Default is 'reflect'
1507:     cval : scalar, optional
1508:         Value to fill past edges of input if `mode` is 'constant'. Default
1509:         is 0.0.
1510:     origin : scalar, optional
1511:         The `origin` parameter controls the placement of the filter.
1512:         Default 0
1513: 
1514:     Returns
1515:     -------
1516:     morphological_gradient : ndarray
1517:         Morphological gradient of `input`.
1518: 
1519:     See also
1520:     --------
1521:     grey_dilation, grey_erosion, ndimage.gaussian_gradient_magnitude
1522: 
1523:     Notes
1524:     -----
1525:     For a flat structuring element, the morphological gradient
1526:     computed at a given point corresponds to the maximal difference
1527:     between elements of the input among the elements covered by the
1528:     structuring element centered on the point.
1529: 
1530:     References
1531:     ----------
1532:     .. [1] http://en.wikipedia.org/wiki/Mathematical_morphology
1533: 
1534:     Examples
1535:     --------
1536:     >>> from scipy import ndimage
1537:     >>> a = np.zeros((7,7), dtype=int)
1538:     >>> a[2:5, 2:5] = 1
1539:     >>> ndimage.morphological_gradient(a, size=(3,3))
1540:     array([[0, 0, 0, 0, 0, 0, 0],
1541:            [0, 1, 1, 1, 1, 1, 0],
1542:            [0, 1, 1, 1, 1, 1, 0],
1543:            [0, 1, 1, 0, 1, 1, 0],
1544:            [0, 1, 1, 1, 1, 1, 0],
1545:            [0, 1, 1, 1, 1, 1, 0],
1546:            [0, 0, 0, 0, 0, 0, 0]])
1547:     >>> # The morphological gradient is computed as the difference
1548:     >>> # between a dilation and an erosion
1549:     >>> ndimage.grey_dilation(a, size=(3,3)) -\\
1550:     ...  ndimage.grey_erosion(a, size=(3,3))
1551:     array([[0, 0, 0, 0, 0, 0, 0],
1552:            [0, 1, 1, 1, 1, 1, 0],
1553:            [0, 1, 1, 1, 1, 1, 0],
1554:            [0, 1, 1, 0, 1, 1, 0],
1555:            [0, 1, 1, 1, 1, 1, 0],
1556:            [0, 1, 1, 1, 1, 1, 0],
1557:            [0, 0, 0, 0, 0, 0, 0]])
1558:     >>> a = np.zeros((7,7), dtype=int)
1559:     >>> a[2:5, 2:5] = 1
1560:     >>> a[4,4] = 2; a[2,3] = 3
1561:     >>> a
1562:     array([[0, 0, 0, 0, 0, 0, 0],
1563:            [0, 0, 0, 0, 0, 0, 0],
1564:            [0, 0, 1, 3, 1, 0, 0],
1565:            [0, 0, 1, 1, 1, 0, 0],
1566:            [0, 0, 1, 1, 2, 0, 0],
1567:            [0, 0, 0, 0, 0, 0, 0],
1568:            [0, 0, 0, 0, 0, 0, 0]])
1569:     >>> ndimage.morphological_gradient(a, size=(3,3))
1570:     array([[0, 0, 0, 0, 0, 0, 0],
1571:            [0, 1, 3, 3, 3, 1, 0],
1572:            [0, 1, 3, 3, 3, 1, 0],
1573:            [0, 1, 3, 2, 3, 2, 0],
1574:            [0, 1, 1, 2, 2, 2, 0],
1575:            [0, 1, 1, 2, 2, 2, 0],
1576:            [0, 0, 0, 0, 0, 0, 0]])
1577: 
1578:     '''
1579:     tmp = grey_dilation(input, size, footprint, structure, None, mode,
1580:                         cval, origin)
1581:     if isinstance(output, numpy.ndarray):
1582:         grey_erosion(input, size, footprint, structure, output, mode,
1583:                      cval, origin)
1584:         return numpy.subtract(tmp, output, output)
1585:     else:
1586:         return (tmp - grey_erosion(input, size, footprint, structure,
1587:                                    None, mode, cval, origin))
1588: 
1589: 
1590: def morphological_laplace(input, size=None, footprint=None,
1591:                           structure=None, output=None,
1592:                           mode="reflect", cval=0.0, origin=0):
1593:     '''
1594:     Multi-dimensional morphological laplace.
1595: 
1596:     Parameters
1597:     ----------
1598:     input : array_like
1599:         Input.
1600:     size : int or sequence of ints, optional
1601:         See `structure`.
1602:     footprint : bool or ndarray, optional
1603:         See `structure`.
1604:     structure : structure, optional
1605:         Either `size`, `footprint`, or the `structure` must be provided.
1606:     output : ndarray, optional
1607:         An output array can optionally be provided.
1608:     mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
1609:         The mode parameter determines how the array borders are handled.
1610:         For 'constant' mode, values beyond borders are set to be `cval`.
1611:         Default is 'reflect'.
1612:     cval : scalar, optional
1613:         Value to fill past edges of input if mode is 'constant'.
1614:         Default is 0.0
1615:     origin : origin, optional
1616:         The origin parameter controls the placement of the filter.
1617: 
1618:     Returns
1619:     -------
1620:     morphological_laplace : ndarray
1621:         Output
1622: 
1623:     '''
1624:     tmp1 = grey_dilation(input, size, footprint, structure, None, mode,
1625:                          cval, origin)
1626:     if isinstance(output, numpy.ndarray):
1627:         grey_erosion(input, size, footprint, structure, output, mode,
1628:                      cval, origin)
1629:         numpy.add(tmp1, output, output)
1630:         numpy.subtract(output, input, output)
1631:         return numpy.subtract(output, input, output)
1632:     else:
1633:         tmp2 = grey_erosion(input, size, footprint, structure, None, mode,
1634:                             cval, origin)
1635:         numpy.add(tmp1, tmp2, tmp2)
1636:         numpy.subtract(tmp2, input, tmp2)
1637:         numpy.subtract(tmp2, input, tmp2)
1638:         return tmp2
1639: 
1640: 
1641: def white_tophat(input, size=None, footprint=None, structure=None,
1642:                  output=None, mode="reflect", cval=0.0, origin=0):
1643:     '''
1644:     Multi-dimensional white tophat filter.
1645: 
1646:     Parameters
1647:     ----------
1648:     input : array_like
1649:         Input.
1650:     size : tuple of ints
1651:         Shape of a flat and full structuring element used for the filter.
1652:         Optional if `footprint` or `structure` is provided.
1653:     footprint : array of ints, optional
1654:         Positions of elements of a flat structuring element
1655:         used for the white tophat filter.
1656:     structure : array of ints, optional
1657:         Structuring element used for the filter. `structure`
1658:         may be a non-flat structuring element.
1659:     output : array, optional
1660:         An array used for storing the output of the filter may be provided.
1661:     mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
1662:         The `mode` parameter determines how the array borders are
1663:         handled, where `cval` is the value when mode is equal to
1664:         'constant'. Default is 'reflect'
1665:     cval : scalar, optional
1666:         Value to fill past edges of input if `mode` is 'constant'.
1667:         Default is 0.0.
1668:     origin : scalar, optional
1669:         The `origin` parameter controls the placement of the filter.
1670:         Default is 0.
1671: 
1672:     Returns
1673:     -------
1674:     output : ndarray
1675:         Result of the filter of `input` with `structure`.
1676: 
1677:     See also
1678:     --------
1679:     black_tophat
1680: 
1681:     '''
1682:     tmp = grey_erosion(input, size, footprint, structure, None, mode,
1683:                        cval, origin)
1684:     if isinstance(output, numpy.ndarray):
1685:         grey_dilation(tmp, size, footprint, structure, output, mode, cval,
1686:                       origin)
1687:         return numpy.subtract(input, output, output)
1688:     else:
1689:         tmp = grey_dilation(tmp, size, footprint, structure, None, mode,
1690:                             cval, origin)
1691:         return input - tmp
1692: 
1693: 
1694: def black_tophat(input, size=None, footprint=None,
1695:                  structure=None, output=None, mode="reflect",
1696:                  cval=0.0, origin=0):
1697:     '''
1698:     Multi-dimensional black tophat filter.
1699: 
1700:     Parameters
1701:     ----------
1702:     input : array_like
1703:         Input.
1704:     size : tuple of ints, optional
1705:         Shape of a flat and full structuring element used for the filter.
1706:         Optional if `footprint` or `structure` is provided.
1707:     footprint : array of ints, optional
1708:         Positions of non-infinite elements of a flat structuring element
1709:         used for the black tophat filter.
1710:     structure : array of ints, optional
1711:         Structuring element used for the filter. `structure`
1712:         may be a non-flat structuring element.
1713:     output : array, optional
1714:         An array used for storing the output of the filter may be provided.
1715:     mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
1716:         The `mode` parameter determines how the array borders are
1717:         handled, where `cval` is the value when mode is equal to
1718:         'constant'. Default is 'reflect'
1719:     cval : scalar, optional
1720:         Value to fill past edges of input if `mode` is 'constant'. Default
1721:         is 0.0.
1722:     origin : scalar, optional
1723:         The `origin` parameter controls the placement of the filter.
1724:         Default 0
1725: 
1726:     Returns
1727:     -------
1728:     black_tophat : ndarray
1729:         Result of the filter of `input` with `structure`.
1730: 
1731:     See also
1732:     --------
1733:     white_tophat, grey_opening, grey_closing
1734: 
1735:     '''
1736:     tmp = grey_dilation(input, size, footprint, structure, None, mode,
1737:                         cval, origin)
1738:     if isinstance(output, numpy.ndarray):
1739:         grey_erosion(tmp, size, footprint, structure, output, mode, cval,
1740:                      origin)
1741:         return numpy.subtract(output, input, output)
1742:     else:
1743:         tmp = grey_erosion(tmp, size, footprint, structure, None, mode,
1744:                            cval, origin)
1745:         return tmp - input
1746: 
1747: 
1748: def distance_transform_bf(input, metric="euclidean", sampling=None,
1749:                           return_distances=True, return_indices=False,
1750:                           distances=None, indices=None):
1751:     '''
1752:     Distance transform function by a brute force algorithm.
1753: 
1754:     This function calculates the distance transform of the `input`, by
1755:     replacing each foreground (non-zero) element, with its
1756:     shortest distance to the background (any zero-valued element).
1757: 
1758:     In addition to the distance transform, the feature transform can
1759:     be calculated. In this case the index of the closest background
1760:     element is returned along the first axis of the result.
1761: 
1762:     Parameters
1763:     ----------
1764:     input : array_like
1765:         Input
1766:     metric : str, optional
1767:         Three types of distance metric are supported: 'euclidean', 'taxicab'
1768:         and 'chessboard'.
1769:     sampling : {int, sequence of ints}, optional
1770:         This parameter is only used in the case of the euclidean `metric`
1771:         distance transform.
1772: 
1773:         The sampling along each axis can be given by the `sampling` parameter
1774:         which should be a sequence of length equal to the input rank, or a
1775:         single number in which the `sampling` is assumed to be equal along all
1776:         axes.
1777:     return_distances : bool, optional
1778:         The `return_distances` flag can be used to indicate if the distance
1779:         transform is returned.
1780: 
1781:         The default is True.
1782:     return_indices : bool, optional
1783:         The `return_indices` flags can be used to indicate if the feature
1784:         transform is returned.
1785: 
1786:         The default is False.
1787:     distances : float64 ndarray, optional
1788:         Optional output array to hold distances (if `return_distances` is
1789:         True).
1790:     indices : int64 ndarray, optional
1791:         Optional output array to hold indices (if `return_indices` is True).
1792: 
1793:     Returns
1794:     -------
1795:     distances : ndarray
1796:         Distance array if `return_distances` is True.
1797:     indices : ndarray
1798:         Indices array if `return_indices` is True.
1799: 
1800:     Notes
1801:     -----
1802:     This function employs a slow brute force algorithm, see also the
1803:     function distance_transform_cdt for more efficient taxicab and
1804:     chessboard algorithms.
1805: 
1806:     '''
1807:     if (not return_distances) and (not return_indices):
1808:         msg = 'at least one of distances/indices must be specified'
1809:         raise RuntimeError(msg)
1810: 
1811:     tmp1 = numpy.asarray(input) != 0
1812:     struct = generate_binary_structure(tmp1.ndim, tmp1.ndim)
1813:     tmp2 = binary_dilation(tmp1, struct)
1814:     tmp2 = numpy.logical_xor(tmp1, tmp2)
1815:     tmp1 = tmp1.astype(numpy.int8) - tmp2.astype(numpy.int8)
1816:     metric = metric.lower()
1817:     if metric == 'euclidean':
1818:         metric = 1
1819:     elif metric in ['taxicab', 'cityblock', 'manhattan']:
1820:         metric = 2
1821:     elif metric == 'chessboard':
1822:         metric = 3
1823:     else:
1824:         raise RuntimeError('distance metric not supported')
1825:     if sampling is not None:
1826:         sampling = _ni_support._normalize_sequence(sampling, tmp1.ndim)
1827:         sampling = numpy.asarray(sampling, dtype=numpy.float64)
1828:         if not sampling.flags.contiguous:
1829:             sampling = sampling.copy()
1830:     if return_indices:
1831:         ft = numpy.zeros(tmp1.shape, dtype=numpy.int32)
1832:     else:
1833:         ft = None
1834:     if return_distances:
1835:         if distances is None:
1836:             if metric == 1:
1837:                 dt = numpy.zeros(tmp1.shape, dtype=numpy.float64)
1838:             else:
1839:                 dt = numpy.zeros(tmp1.shape, dtype=numpy.uint32)
1840:         else:
1841:             if distances.shape != tmp1.shape:
1842:                 raise RuntimeError('distances array has wrong shape')
1843:             if metric == 1:
1844:                 if distances.dtype.type != numpy.float64:
1845:                     raise RuntimeError('distances array must be float64')
1846:             else:
1847:                 if distances.dtype.type != numpy.uint32:
1848:                     raise RuntimeError('distances array must be uint32')
1849:             dt = distances
1850:     else:
1851:         dt = None
1852: 
1853:     _nd_image.distance_transform_bf(tmp1, metric, sampling, dt, ft)
1854:     if return_indices:
1855:         if isinstance(indices, numpy.ndarray):
1856:             if indices.dtype.type != numpy.int32:
1857:                 raise RuntimeError('indices must of int32 type')
1858:             if indices.shape != (tmp1.ndim,) + tmp1.shape:
1859:                 raise RuntimeError('indices has wrong shape')
1860:             tmp2 = indices
1861:         else:
1862:             tmp2 = numpy.indices(tmp1.shape, dtype=numpy.int32)
1863:         ft = numpy.ravel(ft)
1864:         for ii in range(tmp2.shape[0]):
1865:             rtmp = numpy.ravel(tmp2[ii, ...])[ft]
1866:             rtmp.shape = tmp1.shape
1867:             tmp2[ii, ...] = rtmp
1868:         ft = tmp2
1869: 
1870:     # construct and return the result
1871:     result = []
1872:     if return_distances and not isinstance(distances, numpy.ndarray):
1873:         result.append(dt)
1874:     if return_indices and not isinstance(indices, numpy.ndarray):
1875:         result.append(ft)
1876: 
1877:     if len(result) == 2:
1878:         return tuple(result)
1879:     elif len(result) == 1:
1880:         return result[0]
1881:     else:
1882:         return None
1883: 
1884: 
1885: def distance_transform_cdt(input, metric='chessboard',
1886:                         return_distances=True, return_indices=False,
1887:                         distances=None, indices=None):
1888:     '''
1889:     Distance transform for chamfer type of transforms.
1890: 
1891:     Parameters
1892:     ----------
1893:     input : array_like
1894:         Input
1895:     metric : {'chessboard', 'taxicab'}, optional
1896:         The `metric` determines the type of chamfering that is done. If the
1897:         `metric` is equal to 'taxicab' a structure is generated using
1898:         generate_binary_structure with a squared distance equal to 1. If
1899:         the `metric` is equal to 'chessboard', a `metric` is generated
1900:         using generate_binary_structure with a squared distance equal to
1901:         the dimensionality of the array. These choices correspond to the
1902:         common interpretations of the 'taxicab' and the 'chessboard'
1903:         distance metrics in two dimensions.
1904: 
1905:         The default for `metric` is 'chessboard'.
1906:     return_distances, return_indices : bool, optional
1907:         The `return_distances`, and `return_indices` flags can be used to
1908:         indicate if the distance transform, the feature transform, or both
1909:         must be returned.
1910: 
1911:         If the feature transform is returned (``return_indices=True``),
1912:         the index of the closest background element is returned along
1913:         the first axis of the result.
1914: 
1915:         The `return_distances` default is True, and the
1916:         `return_indices` default is False.
1917:     distances, indices : ndarrays of int32, optional
1918:         The `distances` and `indices` arguments can be used to give optional
1919:         output arrays that must be the same shape as `input`.
1920: 
1921:     '''
1922:     if (not return_distances) and (not return_indices):
1923:         msg = 'at least one of distances/indices must be specified'
1924:         raise RuntimeError(msg)
1925: 
1926:     ft_inplace = isinstance(indices, numpy.ndarray)
1927:     dt_inplace = isinstance(distances, numpy.ndarray)
1928:     input = numpy.asarray(input)
1929:     if metric in ['taxicab', 'cityblock', 'manhattan']:
1930:         rank = input.ndim
1931:         metric = generate_binary_structure(rank, 1)
1932:     elif metric == 'chessboard':
1933:         rank = input.ndim
1934:         metric = generate_binary_structure(rank, rank)
1935:     else:
1936:         try:
1937:             metric = numpy.asarray(metric)
1938:         except:
1939:             raise RuntimeError('invalid metric provided')
1940:         for s in metric.shape:
1941:             if s != 3:
1942:                 raise RuntimeError('metric sizes must be equal to 3')
1943: 
1944:     if not metric.flags.contiguous:
1945:         metric = metric.copy()
1946:     if dt_inplace:
1947:         if distances.dtype.type != numpy.int32:
1948:             raise RuntimeError('distances must be of int32 type')
1949:         if distances.shape != input.shape:
1950:             raise RuntimeError('distances has wrong shape')
1951:         dt = distances
1952:         dt[...] = numpy.where(input, -1, 0).astype(numpy.int32)
1953:     else:
1954:         dt = numpy.where(input, -1, 0).astype(numpy.int32)
1955: 
1956:     rank = dt.ndim
1957:     if return_indices:
1958:         sz = numpy.product(dt.shape,axis=0)
1959:         ft = numpy.arange(sz, dtype=numpy.int32)
1960:         ft.shape = dt.shape
1961:     else:
1962:         ft = None
1963: 
1964:     _nd_image.distance_transform_op(metric, dt, ft)
1965:     dt = dt[tuple([slice(None, None, -1)] * rank)]
1966:     if return_indices:
1967:         ft = ft[tuple([slice(None, None, -1)] * rank)]
1968:     _nd_image.distance_transform_op(metric, dt, ft)
1969:     dt = dt[tuple([slice(None, None, -1)] * rank)]
1970:     if return_indices:
1971:         ft = ft[tuple([slice(None, None, -1)] * rank)]
1972:         ft = numpy.ravel(ft)
1973:         if ft_inplace:
1974:             if indices.dtype.type != numpy.int32:
1975:                 raise RuntimeError('indices must of int32 type')
1976:             if indices.shape != (dt.ndim,) + dt.shape:
1977:                 raise RuntimeError('indices has wrong shape')
1978:             tmp = indices
1979:         else:
1980:             tmp = numpy.indices(dt.shape, dtype=numpy.int32)
1981:         for ii in range(tmp.shape[0]):
1982:             rtmp = numpy.ravel(tmp[ii, ...])[ft]
1983:             rtmp.shape = dt.shape
1984:             tmp[ii, ...] = rtmp
1985:         ft = tmp
1986: 
1987:     # construct and return the result
1988:     result = []
1989:     if return_distances and not dt_inplace:
1990:         result.append(dt)
1991:     if return_indices and not ft_inplace:
1992:         result.append(ft)
1993: 
1994:     if len(result) == 2:
1995:         return tuple(result)
1996:     elif len(result) == 1:
1997:         return result[0]
1998:     else:
1999:         return None
2000: 
2001: 
2002: def distance_transform_edt(input, sampling=None,
2003:                         return_distances=True, return_indices=False,
2004:                         distances=None, indices=None):
2005:     '''
2006:     Exact euclidean distance transform.
2007: 
2008:     In addition to the distance transform, the feature transform can
2009:     be calculated. In this case the index of the closest background
2010:     element is returned along the first axis of the result.
2011: 
2012:     Parameters
2013:     ----------
2014:     input : array_like
2015:         Input data to transform. Can be any type but will be converted
2016:         into binary: 1 wherever input equates to True, 0 elsewhere.
2017:     sampling : float or int, or sequence of same, optional
2018:         Spacing of elements along each dimension. If a sequence, must be of
2019:         length equal to the input rank; if a single number, this is used for
2020:         all axes. If not specified, a grid spacing of unity is implied.
2021:     return_distances : bool, optional
2022:         Whether to return distance matrix. At least one of
2023:         return_distances/return_indices must be True. Default is True.
2024:     return_indices : bool, optional
2025:         Whether to return indices matrix. Default is False.
2026:     distances : ndarray, optional
2027:         Used for output of distance array, must be of type float64.
2028:     indices : ndarray, optional
2029:         Used for output of indices, must be of type int32.
2030: 
2031:     Returns
2032:     -------
2033:     distance_transform_edt : ndarray or list of ndarrays
2034:         Either distance matrix, index matrix, or a list of the two,
2035:         depending on `return_x` flags and `distance` and `indices`
2036:         input parameters.
2037: 
2038:     Notes
2039:     -----
2040:     The euclidean distance transform gives values of the euclidean
2041:     distance::
2042: 
2043:                     n
2044:       y_i = sqrt(sum (x[i]-b[i])**2)
2045:                     i
2046: 
2047:     where b[i] is the background point (value 0) with the smallest
2048:     Euclidean distance to input points x[i], and n is the
2049:     number of dimensions.
2050: 
2051:     Examples
2052:     --------
2053:     >>> from scipy import ndimage
2054:     >>> a = np.array(([0,1,1,1,1],
2055:     ...               [0,0,1,1,1],
2056:     ...               [0,1,1,1,1],
2057:     ...               [0,1,1,1,0],
2058:     ...               [0,1,1,0,0]))
2059:     >>> ndimage.distance_transform_edt(a)
2060:     array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
2061:            [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
2062:            [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
2063:            [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
2064:            [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
2065: 
2066:     With a sampling of 2 units along x, 1 along y:
2067: 
2068:     >>> ndimage.distance_transform_edt(a, sampling=[2,1])
2069:     array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],
2070:            [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],
2071:            [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],
2072:            [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],
2073:            [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
2074: 
2075:     Asking for indices as well:
2076: 
2077:     >>> edt, inds = ndimage.distance_transform_edt(a, return_indices=True)
2078:     >>> inds
2079:     array([[[0, 0, 1, 1, 3],
2080:             [1, 1, 1, 1, 3],
2081:             [2, 2, 1, 3, 3],
2082:             [3, 3, 4, 4, 3],
2083:             [4, 4, 4, 4, 4]],
2084:            [[0, 0, 1, 1, 4],
2085:             [0, 1, 1, 1, 4],
2086:             [0, 0, 1, 4, 4],
2087:             [0, 0, 3, 3, 4],
2088:             [0, 0, 3, 3, 4]]])
2089: 
2090:     With arrays provided for inplace outputs:
2091: 
2092:     >>> indices = np.zeros(((np.ndim(a),) + a.shape), dtype=np.int32)
2093:     >>> ndimage.distance_transform_edt(a, return_indices=True, indices=indices)
2094:     array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
2095:            [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
2096:            [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
2097:            [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
2098:            [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
2099:     >>> indices
2100:     array([[[0, 0, 1, 1, 3],
2101:             [1, 1, 1, 1, 3],
2102:             [2, 2, 1, 3, 3],
2103:             [3, 3, 4, 4, 3],
2104:             [4, 4, 4, 4, 4]],
2105:            [[0, 0, 1, 1, 4],
2106:             [0, 1, 1, 1, 4],
2107:             [0, 0, 1, 4, 4],
2108:             [0, 0, 3, 3, 4],
2109:             [0, 0, 3, 3, 4]]])
2110: 
2111:     '''
2112:     if (not return_distances) and (not return_indices):
2113:         msg = 'at least one of distances/indices must be specified'
2114:         raise RuntimeError(msg)
2115: 
2116:     ft_inplace = isinstance(indices, numpy.ndarray)
2117:     dt_inplace = isinstance(distances, numpy.ndarray)
2118:     # calculate the feature transform
2119:     input = numpy.atleast_1d(numpy.where(input, 1, 0).astype(numpy.int8))
2120:     if sampling is not None:
2121:         sampling = _ni_support._normalize_sequence(sampling, input.ndim)
2122:         sampling = numpy.asarray(sampling, dtype=numpy.float64)
2123:         if not sampling.flags.contiguous:
2124:             sampling = sampling.copy()
2125: 
2126:     if ft_inplace:
2127:         ft = indices
2128:         if ft.shape != (input.ndim,) + input.shape:
2129:             raise RuntimeError('indices has wrong shape')
2130:         if ft.dtype.type != numpy.int32:
2131:             raise RuntimeError('indices must be of int32 type')
2132:     else:
2133:         ft = numpy.zeros((input.ndim,) + input.shape,
2134:                             dtype=numpy.int32)
2135: 
2136:     _nd_image.euclidean_feature_transform(input, sampling, ft)
2137:     # if requested, calculate the distance transform
2138:     if return_distances:
2139:         dt = ft - numpy.indices(input.shape, dtype=ft.dtype)
2140:         dt = dt.astype(numpy.float64)
2141:         if sampling is not None:
2142:             for ii in range(len(sampling)):
2143:                 dt[ii, ...] *= sampling[ii]
2144:         numpy.multiply(dt, dt, dt)
2145:         if dt_inplace:
2146:             dt = numpy.add.reduce(dt, axis=0)
2147:             if distances.shape != dt.shape:
2148:                 raise RuntimeError('indices has wrong shape')
2149:             if distances.dtype.type != numpy.float64:
2150:                 raise RuntimeError('indices must be of float64 type')
2151:             numpy.sqrt(dt, distances)
2152:         else:
2153:             dt = numpy.add.reduce(dt, axis=0)
2154:             dt = numpy.sqrt(dt)
2155: 
2156:     # construct and return the result
2157:     result = []
2158:     if return_distances and not dt_inplace:
2159:         result.append(dt)
2160:     if return_indices and not ft_inplace:
2161:         result.append(ft)
2162: 
2163:     if len(result) == 2:
2164:         return tuple(result)
2165:     elif len(result) == 1:
2166:         return result[0]
2167:     else:
2168:         return None
2169: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import numpy' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_124066 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy')

if (type(import_124066) is not StypyTypeError):

    if (import_124066 != 'pyd_module'):
        __import__(import_124066)
        sys_modules_124067 = sys.modules[import_124066]
        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', sys_modules_124067.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', import_124066)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from scipy.ndimage import _ni_support' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_124068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.ndimage')

if (type(import_124068) is not StypyTypeError):

    if (import_124068 != 'pyd_module'):
        __import__(import_124068)
        sys_modules_124069 = sys.modules[import_124068]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.ndimage', sys_modules_124069.module_type_store, module_type_store, ['_ni_support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_124069, sys_modules_124069.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _ni_support

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.ndimage', None, module_type_store, ['_ni_support'], [_ni_support])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.ndimage', import_124068)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from scipy.ndimage import _nd_image' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_124070 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage')

if (type(import_124070) is not StypyTypeError):

    if (import_124070 != 'pyd_module'):
        __import__(import_124070)
        sys_modules_124071 = sys.modules[import_124070]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', sys_modules_124071.module_type_store, module_type_store, ['_nd_image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_124071, sys_modules_124071.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _nd_image

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', None, module_type_store, ['_nd_image'], [_nd_image])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', import_124070)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from scipy.ndimage import filters' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_124072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage')

if (type(import_124072) is not StypyTypeError):

    if (import_124072 != 'pyd_module'):
        __import__(import_124072)
        sys_modules_124073 = sys.modules[import_124072]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', sys_modules_124073.module_type_store, module_type_store, ['filters'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_124073, sys_modules_124073.module_type_store, module_type_store)
    else:
        from scipy.ndimage import filters

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', None, module_type_store, ['filters'], [filters])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', import_124072)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


# Assigning a List to a Name (line 38):

# Assigning a List to a Name (line 38):
__all__ = ['iterate_structure', 'generate_binary_structure', 'binary_erosion', 'binary_dilation', 'binary_opening', 'binary_closing', 'binary_hit_or_miss', 'binary_propagation', 'binary_fill_holes', 'grey_erosion', 'grey_dilation', 'grey_opening', 'grey_closing', 'morphological_gradient', 'morphological_laplace', 'white_tophat', 'black_tophat', 'distance_transform_bf', 'distance_transform_cdt', 'distance_transform_edt']
module_type_store.set_exportable_members(['iterate_structure', 'generate_binary_structure', 'binary_erosion', 'binary_dilation', 'binary_opening', 'binary_closing', 'binary_hit_or_miss', 'binary_propagation', 'binary_fill_holes', 'grey_erosion', 'grey_dilation', 'grey_opening', 'grey_closing', 'morphological_gradient', 'morphological_laplace', 'white_tophat', 'black_tophat', 'distance_transform_bf', 'distance_transform_cdt', 'distance_transform_edt'])

# Obtaining an instance of the builtin type 'list' (line 38)
list_124074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
str_124075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 11), 'str', 'iterate_structure')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124075)
# Adding element type (line 38)
str_124076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 32), 'str', 'generate_binary_structure')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124076)
# Adding element type (line 38)
str_124077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 61), 'str', 'binary_erosion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124077)
# Adding element type (line 38)
str_124078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 11), 'str', 'binary_dilation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124078)
# Adding element type (line 38)
str_124079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'str', 'binary_opening')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124079)
# Adding element type (line 38)
str_124080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 48), 'str', 'binary_closing')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124080)
# Adding element type (line 38)
str_124081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 11), 'str', 'binary_hit_or_miss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124081)
# Adding element type (line 38)
str_124082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'str', 'binary_propagation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124082)
# Adding element type (line 38)
str_124083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 55), 'str', 'binary_fill_holes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124083)
# Adding element type (line 38)
str_124084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'str', 'grey_erosion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124084)
# Adding element type (line 38)
str_124085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'str', 'grey_dilation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124085)
# Adding element type (line 38)
str_124086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'str', 'grey_opening')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124086)
# Adding element type (line 38)
str_124087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 60), 'str', 'grey_closing')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124087)
# Adding element type (line 38)
str_124088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'str', 'morphological_gradient')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124088)
# Adding element type (line 38)
str_124089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'str', 'morphological_laplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124089)
# Adding element type (line 38)
str_124090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 62), 'str', 'white_tophat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124090)
# Adding element type (line 38)
str_124091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'str', 'black_tophat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124091)
# Adding element type (line 38)
str_124092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'str', 'distance_transform_bf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124092)
# Adding element type (line 38)
str_124093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 52), 'str', 'distance_transform_cdt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124093)
# Adding element type (line 38)
str_124094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', 'distance_transform_edt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_124074, str_124094)

# Assigning a type to the variable '__all__' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), '__all__', list_124074)

@norecursion
def _center_is_true(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_center_is_true'
    module_type_store = module_type_store.open_function_context('_center_is_true', 47, 0, False)
    
    # Passed parameters checking function
    _center_is_true.stypy_localization = localization
    _center_is_true.stypy_type_of_self = None
    _center_is_true.stypy_type_store = module_type_store
    _center_is_true.stypy_function_name = '_center_is_true'
    _center_is_true.stypy_param_names_list = ['structure', 'origin']
    _center_is_true.stypy_varargs_param_name = None
    _center_is_true.stypy_kwargs_param_name = None
    _center_is_true.stypy_call_defaults = defaults
    _center_is_true.stypy_call_varargs = varargs
    _center_is_true.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_center_is_true', ['structure', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_center_is_true', localization, ['structure', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_center_is_true(...)' code ##################

    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to array(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'structure' (line 48)
    structure_124097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'structure', False)
    # Processing the call keyword arguments (line 48)
    kwargs_124098 = {}
    # Getting the type of 'numpy' (line 48)
    numpy_124095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'numpy', False)
    # Obtaining the member 'array' of a type (line 48)
    array_124096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), numpy_124095, 'array')
    # Calling array(args, kwargs) (line 48)
    array_call_result_124099 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), array_124096, *[structure_124097], **kwargs_124098)
    
    # Assigning a type to the variable 'structure' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'structure', array_call_result_124099)
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to tuple(...): (line 49)
    # Processing the call arguments (line 49)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'structure' (line 49)
    structure_124107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'structure', False)
    # Obtaining the member 'shape' of a type (line 49)
    shape_124108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 49), structure_124107, 'shape')
    # Getting the type of 'origin' (line 50)
    origin_124109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 49), 'origin', False)
    # Processing the call keyword arguments (line 49)
    kwargs_124110 = {}
    # Getting the type of 'zip' (line 49)
    zip_124106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 45), 'zip', False)
    # Calling zip(args, kwargs) (line 49)
    zip_call_result_124111 = invoke(stypy.reporting.localization.Localization(__file__, 49, 45), zip_124106, *[shape_124108, origin_124109], **kwargs_124110)
    
    comprehension_124112 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 18), zip_call_result_124111)
    # Assigning a type to the variable 'ss' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'ss', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 18), comprehension_124112))
    # Assigning a type to the variable 'oo' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'oo', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 18), comprehension_124112))
    # Getting the type of 'oo' (line 49)
    oo_124101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'oo', False)
    # Getting the type of 'ss' (line 49)
    ss_124102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'ss', False)
    int_124103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'int')
    # Applying the binary operator '//' (line 49)
    result_floordiv_124104 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 23), '//', ss_124102, int_124103)
    
    # Applying the binary operator '+' (line 49)
    result_add_124105 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 18), '+', oo_124101, result_floordiv_124104)
    
    list_124113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 18), list_124113, result_add_124105)
    # Processing the call keyword arguments (line 49)
    kwargs_124114 = {}
    # Getting the type of 'tuple' (line 49)
    tuple_124100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 49)
    tuple_call_result_124115 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), tuple_124100, *[list_124113], **kwargs_124114)
    
    # Assigning a type to the variable 'coor' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'coor', tuple_call_result_124115)
    
    # Call to bool(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Obtaining the type of the subscript
    # Getting the type of 'coor' (line 51)
    coor_124117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'coor', False)
    # Getting the type of 'structure' (line 51)
    structure_124118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___124119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), structure_124118, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_124120 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), getitem___124119, coor_124117)
    
    # Processing the call keyword arguments (line 51)
    kwargs_124121 = {}
    # Getting the type of 'bool' (line 51)
    bool_124116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'bool', False)
    # Calling bool(args, kwargs) (line 51)
    bool_call_result_124122 = invoke(stypy.reporting.localization.Localization(__file__, 51, 11), bool_124116, *[subscript_call_result_124120], **kwargs_124121)
    
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type', bool_call_result_124122)
    
    # ################# End of '_center_is_true(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_center_is_true' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_124123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124123)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_center_is_true'
    return stypy_return_type_124123

# Assigning a type to the variable '_center_is_true' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), '_center_is_true', _center_is_true)

@norecursion
def iterate_structure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 54)
    None_124124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 52), 'None')
    defaults = [None_124124]
    # Create a new context for function 'iterate_structure'
    module_type_store = module_type_store.open_function_context('iterate_structure', 54, 0, False)
    
    # Passed parameters checking function
    iterate_structure.stypy_localization = localization
    iterate_structure.stypy_type_of_self = None
    iterate_structure.stypy_type_store = module_type_store
    iterate_structure.stypy_function_name = 'iterate_structure'
    iterate_structure.stypy_param_names_list = ['structure', 'iterations', 'origin']
    iterate_structure.stypy_varargs_param_name = None
    iterate_structure.stypy_kwargs_param_name = None
    iterate_structure.stypy_call_defaults = defaults
    iterate_structure.stypy_call_varargs = varargs
    iterate_structure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iterate_structure', ['structure', 'iterations', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iterate_structure', localization, ['structure', 'iterations', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iterate_structure(...)' code ##################

    str_124125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', '\n    Iterate a structure by dilating it with itself.\n\n    Parameters\n    ----------\n    structure : array_like\n       Structuring element (an array of bools, for example), to be dilated with\n       itself.\n    iterations : int\n       number of dilations performed on the structure with itself\n    origin : optional\n        If origin is None, only the iterated structure is returned. If\n        not, a tuple of the iterated structure and the modified origin is\n        returned.\n\n    Returns\n    -------\n    iterate_structure : ndarray of bools\n        A new structuring element obtained by dilating `structure`\n        (`iterations` - 1) times with itself.\n\n    See also\n    --------\n    generate_binary_structure\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> struct = ndimage.generate_binary_structure(2, 1)\n    >>> struct.astype(int)\n    array([[0, 1, 0],\n           [1, 1, 1],\n           [0, 1, 0]])\n    >>> ndimage.iterate_structure(struct, 2).astype(int)\n    array([[0, 0, 1, 0, 0],\n           [0, 1, 1, 1, 0],\n           [1, 1, 1, 1, 1],\n           [0, 1, 1, 1, 0],\n           [0, 0, 1, 0, 0]])\n    >>> ndimage.iterate_structure(struct, 3).astype(int)\n    array([[0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [1, 1, 1, 1, 1, 1, 1],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0]])\n\n    ')
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to asarray(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'structure' (line 104)
    structure_124128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'structure', False)
    # Processing the call keyword arguments (line 104)
    kwargs_124129 = {}
    # Getting the type of 'numpy' (line 104)
    numpy_124126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 104)
    asarray_124127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), numpy_124126, 'asarray')
    # Calling asarray(args, kwargs) (line 104)
    asarray_call_result_124130 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), asarray_124127, *[structure_124128], **kwargs_124129)
    
    # Assigning a type to the variable 'structure' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'structure', asarray_call_result_124130)
    
    
    # Getting the type of 'iterations' (line 105)
    iterations_124131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 7), 'iterations')
    int_124132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'int')
    # Applying the binary operator '<' (line 105)
    result_lt_124133 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), '<', iterations_124131, int_124132)
    
    # Testing the type of an if condition (line 105)
    if_condition_124134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 4), result_lt_124133)
    # Assigning a type to the variable 'if_condition_124134' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'if_condition_124134', if_condition_124134)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to copy(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_124137 = {}
    # Getting the type of 'structure' (line 106)
    structure_124135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'structure', False)
    # Obtaining the member 'copy' of a type (line 106)
    copy_124136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 15), structure_124135, 'copy')
    # Calling copy(args, kwargs) (line 106)
    copy_call_result_124138 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), copy_124136, *[], **kwargs_124137)
    
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'stypy_return_type', copy_call_result_124138)
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 107):
    
    # Assigning a BinOp to a Name (line 107):
    # Getting the type of 'iterations' (line 107)
    iterations_124139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 9), 'iterations')
    int_124140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'int')
    # Applying the binary operator '-' (line 107)
    result_sub_124141 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 9), '-', iterations_124139, int_124140)
    
    # Assigning a type to the variable 'ni' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'ni', result_sub_124141)
    
    # Assigning a ListComp to a Name (line 108):
    
    # Assigning a ListComp to a Name (line 108):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'structure' (line 108)
    structure_124149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 42), 'structure')
    # Obtaining the member 'shape' of a type (line 108)
    shape_124150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 42), structure_124149, 'shape')
    comprehension_124151 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 13), shape_124150)
    # Assigning a type to the variable 'ii' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'ii', comprehension_124151)
    # Getting the type of 'ii' (line 108)
    ii_124142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'ii')
    # Getting the type of 'ni' (line 108)
    ni_124143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'ni')
    # Getting the type of 'ii' (line 108)
    ii_124144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'ii')
    int_124145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'int')
    # Applying the binary operator '-' (line 108)
    result_sub_124146 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 24), '-', ii_124144, int_124145)
    
    # Applying the binary operator '*' (line 108)
    result_mul_124147 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 18), '*', ni_124143, result_sub_124146)
    
    # Applying the binary operator '+' (line 108)
    result_add_124148 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 13), '+', ii_124142, result_mul_124147)
    
    list_124152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 13), list_124152, result_add_124148)
    # Assigning a type to the variable 'shape' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'shape', list_124152)
    
    # Assigning a ListComp to a Name (line 109):
    
    # Assigning a ListComp to a Name (line 109):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Call to len(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'shape' (line 109)
    shape_124164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 63), 'shape', False)
    # Processing the call keyword arguments (line 109)
    kwargs_124165 = {}
    # Getting the type of 'len' (line 109)
    len_124163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 59), 'len', False)
    # Calling len(args, kwargs) (line 109)
    len_call_result_124166 = invoke(stypy.reporting.localization.Localization(__file__, 109, 59), len_124163, *[shape_124164], **kwargs_124165)
    
    # Processing the call keyword arguments (line 109)
    kwargs_124167 = {}
    # Getting the type of 'range' (line 109)
    range_124162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 53), 'range', False)
    # Calling range(args, kwargs) (line 109)
    range_call_result_124168 = invoke(stypy.reporting.localization.Localization(__file__, 109, 53), range_124162, *[len_call_result_124166], **kwargs_124167)
    
    comprehension_124169 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 11), range_call_result_124168)
    # Assigning a type to the variable 'ii' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'ii', comprehension_124169)
    # Getting the type of 'ni' (line 109)
    ni_124153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'ni')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 109)
    ii_124154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'ii')
    # Getting the type of 'structure' (line 109)
    structure_124155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'structure')
    # Obtaining the member 'shape' of a type (line 109)
    shape_124156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 17), structure_124155, 'shape')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___124157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 17), shape_124156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_124158 = invoke(stypy.reporting.localization.Localization(__file__, 109, 17), getitem___124157, ii_124154)
    
    int_124159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 40), 'int')
    # Applying the binary operator '//' (line 109)
    result_floordiv_124160 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 17), '//', subscript_call_result_124158, int_124159)
    
    # Applying the binary operator '*' (line 109)
    result_mul_124161 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), '*', ni_124153, result_floordiv_124160)
    
    list_124170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 11), list_124170, result_mul_124161)
    # Assigning a type to the variable 'pos' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'pos', list_124170)
    
    # Assigning a ListComp to a Name (line 110):
    
    # Assigning a ListComp to a Name (line 110):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Call to len(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'shape' (line 111)
    shape_124191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 31), 'shape', False)
    # Processing the call keyword arguments (line 111)
    kwargs_124192 = {}
    # Getting the type of 'len' (line 111)
    len_124190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'len', False)
    # Calling len(args, kwargs) (line 111)
    len_call_result_124193 = invoke(stypy.reporting.localization.Localization(__file__, 111, 27), len_124190, *[shape_124191], **kwargs_124192)
    
    # Processing the call keyword arguments (line 111)
    kwargs_124194 = {}
    # Getting the type of 'range' (line 111)
    range_124189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'range', False)
    # Calling range(args, kwargs) (line 111)
    range_call_result_124195 = invoke(stypy.reporting.localization.Localization(__file__, 111, 21), range_124189, *[len_call_result_124193], **kwargs_124194)
    
    comprehension_124196 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 11), range_call_result_124195)
    # Assigning a type to the variable 'ii' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'ii', comprehension_124196)
    
    # Call to slice(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 110)
    ii_124172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'ii', False)
    # Getting the type of 'pos' (line 110)
    pos_124173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'pos', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___124174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), pos_124173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_124175 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), getitem___124174, ii_124172)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 110)
    ii_124176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 30), 'ii', False)
    # Getting the type of 'pos' (line 110)
    pos_124177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'pos', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___124178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 26), pos_124177, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_124179 = invoke(stypy.reporting.localization.Localization(__file__, 110, 26), getitem___124178, ii_124176)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 110)
    ii_124180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 52), 'ii', False)
    # Getting the type of 'structure' (line 110)
    structure_124181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 36), 'structure', False)
    # Obtaining the member 'shape' of a type (line 110)
    shape_124182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 36), structure_124181, 'shape')
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___124183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 36), shape_124182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_124184 = invoke(stypy.reporting.localization.Localization(__file__, 110, 36), getitem___124183, ii_124180)
    
    # Applying the binary operator '+' (line 110)
    result_add_124185 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 26), '+', subscript_call_result_124179, subscript_call_result_124184)
    
    # Getting the type of 'None' (line 110)
    None_124186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 57), 'None', False)
    # Processing the call keyword arguments (line 110)
    kwargs_124187 = {}
    # Getting the type of 'slice' (line 110)
    slice_124171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'slice', False)
    # Calling slice(args, kwargs) (line 110)
    slice_call_result_124188 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), slice_124171, *[subscript_call_result_124175, result_add_124185, None_124186], **kwargs_124187)
    
    list_124197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 11), list_124197, slice_call_result_124188)
    # Assigning a type to the variable 'slc' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'slc', list_124197)
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to zeros(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'shape' (line 112)
    shape_124200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'shape', False)
    # Getting the type of 'bool' (line 112)
    bool_124201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'bool', False)
    # Processing the call keyword arguments (line 112)
    kwargs_124202 = {}
    # Getting the type of 'numpy' (line 112)
    numpy_124198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 10), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 112)
    zeros_124199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 10), numpy_124198, 'zeros')
    # Calling zeros(args, kwargs) (line 112)
    zeros_call_result_124203 = invoke(stypy.reporting.localization.Localization(__file__, 112, 10), zeros_124199, *[shape_124200, bool_124201], **kwargs_124202)
    
    # Assigning a type to the variable 'out' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'out', zeros_call_result_124203)
    
    # Assigning a Compare to a Subscript (line 113):
    
    # Assigning a Compare to a Subscript (line 113):
    
    # Getting the type of 'structure' (line 113)
    structure_124204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'structure')
    int_124205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'int')
    # Applying the binary operator '!=' (line 113)
    result_ne_124206 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), '!=', structure_124204, int_124205)
    
    # Getting the type of 'out' (line 113)
    out_124207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'out')
    # Getting the type of 'slc' (line 113)
    slc_124208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'slc')
    # Storing an element on a container (line 113)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 4), out_124207, (slc_124208, result_ne_124206))
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to binary_dilation(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'out' (line 114)
    out_124210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'out', False)
    # Getting the type of 'structure' (line 114)
    structure_124211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 31), 'structure', False)
    # Processing the call keyword arguments (line 114)
    # Getting the type of 'ni' (line 114)
    ni_124212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 53), 'ni', False)
    keyword_124213 = ni_124212
    kwargs_124214 = {'iterations': keyword_124213}
    # Getting the type of 'binary_dilation' (line 114)
    binary_dilation_124209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 10), 'binary_dilation', False)
    # Calling binary_dilation(args, kwargs) (line 114)
    binary_dilation_call_result_124215 = invoke(stypy.reporting.localization.Localization(__file__, 114, 10), binary_dilation_124209, *[out_124210, structure_124211], **kwargs_124214)
    
    # Assigning a type to the variable 'out' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'out', binary_dilation_call_result_124215)
    
    # Type idiom detected: calculating its left and rigth part (line 115)
    # Getting the type of 'origin' (line 115)
    origin_124216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 7), 'origin')
    # Getting the type of 'None' (line 115)
    None_124217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'None')
    
    (may_be_124218, more_types_in_union_124219) = may_be_none(origin_124216, None_124217)

    if may_be_124218:

        if more_types_in_union_124219:
            # Runtime conditional SSA (line 115)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'out' (line 116)
        out_124220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', out_124220)

        if more_types_in_union_124219:
            # Runtime conditional SSA for else branch (line 115)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_124218) or more_types_in_union_124219):
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to _normalize_sequence(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'origin' (line 118)
        origin_124223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 49), 'origin', False)
        # Getting the type of 'structure' (line 118)
        structure_124224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 57), 'structure', False)
        # Obtaining the member 'ndim' of a type (line 118)
        ndim_124225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 57), structure_124224, 'ndim')
        # Processing the call keyword arguments (line 118)
        kwargs_124226 = {}
        # Getting the type of '_ni_support' (line 118)
        _ni_support_124221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), '_ni_support', False)
        # Obtaining the member '_normalize_sequence' of a type (line 118)
        _normalize_sequence_124222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 17), _ni_support_124221, '_normalize_sequence')
        # Calling _normalize_sequence(args, kwargs) (line 118)
        _normalize_sequence_call_result_124227 = invoke(stypy.reporting.localization.Localization(__file__, 118, 17), _normalize_sequence_124222, *[origin_124223, ndim_124225], **kwargs_124226)
        
        # Assigning a type to the variable 'origin' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'origin', _normalize_sequence_call_result_124227)
        
        # Assigning a ListComp to a Name (line 119):
        
        # Assigning a ListComp to a Name (line 119):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'origin' (line 119)
        origin_124231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'origin')
        comprehension_124232 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), origin_124231)
        # Assigning a type to the variable 'o' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'o', comprehension_124232)
        # Getting the type of 'iterations' (line 119)
        iterations_124228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'iterations')
        # Getting the type of 'o' (line 119)
        o_124229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'o')
        # Applying the binary operator '*' (line 119)
        result_mul_124230 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 18), '*', iterations_124228, o_124229)
        
        list_124233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), list_124233, result_mul_124230)
        # Assigning a type to the variable 'origin' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'origin', list_124233)
        
        # Obtaining an instance of the builtin type 'tuple' (line 120)
        tuple_124234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 120)
        # Adding element type (line 120)
        # Getting the type of 'out' (line 120)
        out_124235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'out')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), tuple_124234, out_124235)
        # Adding element type (line 120)
        # Getting the type of 'origin' (line 120)
        origin_124236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'origin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), tuple_124234, origin_124236)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', tuple_124234)

        if (may_be_124218 and more_types_in_union_124219):
            # SSA join for if statement (line 115)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'iterate_structure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterate_structure' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_124237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124237)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterate_structure'
    return stypy_return_type_124237

# Assigning a type to the variable 'iterate_structure' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'iterate_structure', iterate_structure)

@norecursion
def generate_binary_structure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_binary_structure'
    module_type_store = module_type_store.open_function_context('generate_binary_structure', 123, 0, False)
    
    # Passed parameters checking function
    generate_binary_structure.stypy_localization = localization
    generate_binary_structure.stypy_type_of_self = None
    generate_binary_structure.stypy_type_store = module_type_store
    generate_binary_structure.stypy_function_name = 'generate_binary_structure'
    generate_binary_structure.stypy_param_names_list = ['rank', 'connectivity']
    generate_binary_structure.stypy_varargs_param_name = None
    generate_binary_structure.stypy_kwargs_param_name = None
    generate_binary_structure.stypy_call_defaults = defaults
    generate_binary_structure.stypy_call_varargs = varargs
    generate_binary_structure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_binary_structure', ['rank', 'connectivity'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_binary_structure', localization, ['rank', 'connectivity'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_binary_structure(...)' code ##################

    str_124238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'str', '\n    Generate a binary structure for binary morphological operations.\n\n    Parameters\n    ----------\n    rank : int\n         Number of dimensions of the array to which the structuring element\n         will be applied, as returned by `np.ndim`.\n    connectivity : int\n         `connectivity` determines which elements of the output array belong\n         to the structure, i.e. are considered as neighbors of the central\n         element. Elements up to a squared distance of `connectivity` from\n         the center are considered neighbors. `connectivity` may range from 1\n         (no diagonal elements are neighbors) to `rank` (all elements are\n         neighbors).\n\n    Returns\n    -------\n    output : ndarray of bools\n         Structuring element which may be used for binary morphological\n         operations, with `rank` dimensions and all dimensions equal to 3.\n\n    See also\n    --------\n    iterate_structure, binary_dilation, binary_erosion\n\n    Notes\n    -----\n    `generate_binary_structure` can only create structuring elements with\n    dimensions equal to 3, i.e. minimal dimensions. For larger structuring\n    elements, that are useful e.g. for eroding large objects, one may either\n    use   `iterate_structure`, or create directly custom arrays with\n    numpy functions such as `numpy.ones`.\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> struct = ndimage.generate_binary_structure(2, 1)\n    >>> struct\n    array([[False,  True, False],\n           [ True,  True,  True],\n           [False,  True, False]], dtype=bool)\n    >>> a = np.zeros((5,5))\n    >>> a[2, 2] = 1\n    >>> a\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.]])\n    >>> b = ndimage.binary_dilation(a, structure=struct).astype(a.dtype)\n    >>> b\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.]])\n    >>> ndimage.binary_dilation(b, structure=struct).astype(a.dtype)\n    array([[ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 1.,  1.,  1.,  1.,  1.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.]])\n    >>> struct = ndimage.generate_binary_structure(2, 2)\n    >>> struct\n    array([[ True,  True,  True],\n           [ True,  True,  True],\n           [ True,  True,  True]], dtype=bool)\n    >>> struct = ndimage.generate_binary_structure(3, 1)\n    >>> struct # no diagonal elements\n    array([[[False, False, False],\n            [False,  True, False],\n            [False, False, False]],\n           [[False,  True, False],\n            [ True,  True,  True],\n            [False,  True, False]],\n           [[False, False, False],\n            [False,  True, False],\n            [False, False, False]]], dtype=bool)\n\n    ')
    
    
    # Getting the type of 'connectivity' (line 205)
    connectivity_124239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 7), 'connectivity')
    int_124240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 22), 'int')
    # Applying the binary operator '<' (line 205)
    result_lt_124241 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 7), '<', connectivity_124239, int_124240)
    
    # Testing the type of an if condition (line 205)
    if_condition_124242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 4), result_lt_124241)
    # Assigning a type to the variable 'if_condition_124242' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'if_condition_124242', if_condition_124242)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 206):
    
    # Assigning a Num to a Name (line 206):
    int_124243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 23), 'int')
    # Assigning a type to the variable 'connectivity' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'connectivity', int_124243)
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rank' (line 207)
    rank_124244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 7), 'rank')
    int_124245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 14), 'int')
    # Applying the binary operator '<' (line 207)
    result_lt_124246 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 7), '<', rank_124244, int_124245)
    
    # Testing the type of an if condition (line 207)
    if_condition_124247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 4), result_lt_124246)
    # Assigning a type to the variable 'if_condition_124247' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'if_condition_124247', if_condition_124247)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'True' (line 208)
    True_124250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'True', False)
    # Processing the call keyword arguments (line 208)
    # Getting the type of 'bool' (line 208)
    bool_124251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 39), 'bool', False)
    keyword_124252 = bool_124251
    kwargs_124253 = {'dtype': keyword_124252}
    # Getting the type of 'numpy' (line 208)
    numpy_124248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'numpy', False)
    # Obtaining the member 'array' of a type (line 208)
    array_124249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 15), numpy_124248, 'array')
    # Calling array(args, kwargs) (line 208)
    array_call_result_124254 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), array_124249, *[True_124250], **kwargs_124253)
    
    # Assigning a type to the variable 'stypy_return_type' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', array_call_result_124254)
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to fabs(...): (line 209)
    # Processing the call arguments (line 209)
    
    # Call to indices(...): (line 209)
    # Processing the call arguments (line 209)
    
    # Obtaining an instance of the builtin type 'list' (line 209)
    list_124259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 209)
    # Adding element type (line 209)
    int_124260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 38), list_124259, int_124260)
    
    # Getting the type of 'rank' (line 209)
    rank_124261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 44), 'rank', False)
    # Applying the binary operator '*' (line 209)
    result_mul_124262 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 38), '*', list_124259, rank_124261)
    
    # Processing the call keyword arguments (line 209)
    kwargs_124263 = {}
    # Getting the type of 'numpy' (line 209)
    numpy_124257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'numpy', False)
    # Obtaining the member 'indices' of a type (line 209)
    indices_124258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 24), numpy_124257, 'indices')
    # Calling indices(args, kwargs) (line 209)
    indices_call_result_124264 = invoke(stypy.reporting.localization.Localization(__file__, 209, 24), indices_124258, *[result_mul_124262], **kwargs_124263)
    
    int_124265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 52), 'int')
    # Applying the binary operator '-' (line 209)
    result_sub_124266 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 24), '-', indices_call_result_124264, int_124265)
    
    # Processing the call keyword arguments (line 209)
    kwargs_124267 = {}
    # Getting the type of 'numpy' (line 209)
    numpy_124255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'numpy', False)
    # Obtaining the member 'fabs' of a type (line 209)
    fabs_124256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 13), numpy_124255, 'fabs')
    # Calling fabs(args, kwargs) (line 209)
    fabs_call_result_124268 = invoke(stypy.reporting.localization.Localization(__file__, 209, 13), fabs_124256, *[result_sub_124266], **kwargs_124267)
    
    # Assigning a type to the variable 'output' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'output', fabs_call_result_124268)
    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to reduce(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'output' (line 210)
    output_124272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'output', False)
    int_124273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 38), 'int')
    # Processing the call keyword arguments (line 210)
    kwargs_124274 = {}
    # Getting the type of 'numpy' (line 210)
    numpy_124269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'numpy', False)
    # Obtaining the member 'add' of a type (line 210)
    add_124270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 13), numpy_124269, 'add')
    # Obtaining the member 'reduce' of a type (line 210)
    reduce_124271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 13), add_124270, 'reduce')
    # Calling reduce(args, kwargs) (line 210)
    reduce_call_result_124275 = invoke(stypy.reporting.localization.Localization(__file__, 210, 13), reduce_124271, *[output_124272, int_124273], **kwargs_124274)
    
    # Assigning a type to the variable 'output' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'output', reduce_call_result_124275)
    
    # Getting the type of 'output' (line 211)
    output_124276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'output')
    # Getting the type of 'connectivity' (line 211)
    connectivity_124277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'connectivity')
    # Applying the binary operator '<=' (line 211)
    result_le_124278 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), '<=', output_124276, connectivity_124277)
    
    # Assigning a type to the variable 'stypy_return_type' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type', result_le_124278)
    
    # ################# End of 'generate_binary_structure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_binary_structure' in the type store
    # Getting the type of 'stypy_return_type' (line 123)
    stypy_return_type_124279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124279)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_binary_structure'
    return stypy_return_type_124279

# Assigning a type to the variable 'generate_binary_structure' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'generate_binary_structure', generate_binary_structure)

@norecursion
def _binary_erosion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_binary_erosion'
    module_type_store = module_type_store.open_function_context('_binary_erosion', 214, 0, False)
    
    # Passed parameters checking function
    _binary_erosion.stypy_localization = localization
    _binary_erosion.stypy_type_of_self = None
    _binary_erosion.stypy_type_store = module_type_store
    _binary_erosion.stypy_function_name = '_binary_erosion'
    _binary_erosion.stypy_param_names_list = ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'invert', 'brute_force']
    _binary_erosion.stypy_varargs_param_name = None
    _binary_erosion.stypy_kwargs_param_name = None
    _binary_erosion.stypy_call_defaults = defaults
    _binary_erosion.stypy_call_varargs = varargs
    _binary_erosion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_binary_erosion', ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'invert', 'brute_force'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_binary_erosion', localization, ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'invert', 'brute_force'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_binary_erosion(...)' code ##################

    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to asarray(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'input' (line 216)
    input_124282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 26), 'input', False)
    # Processing the call keyword arguments (line 216)
    kwargs_124283 = {}
    # Getting the type of 'numpy' (line 216)
    numpy_124280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 216)
    asarray_124281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), numpy_124280, 'asarray')
    # Calling asarray(args, kwargs) (line 216)
    asarray_call_result_124284 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), asarray_124281, *[input_124282], **kwargs_124283)
    
    # Assigning a type to the variable 'input' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'input', asarray_call_result_124284)
    
    
    # Call to iscomplexobj(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'input' (line 217)
    input_124287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'input', False)
    # Processing the call keyword arguments (line 217)
    kwargs_124288 = {}
    # Getting the type of 'numpy' (line 217)
    numpy_124285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 217)
    iscomplexobj_124286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 7), numpy_124285, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 217)
    iscomplexobj_call_result_124289 = invoke(stypy.reporting.localization.Localization(__file__, 217, 7), iscomplexobj_124286, *[input_124287], **kwargs_124288)
    
    # Testing the type of an if condition (line 217)
    if_condition_124290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 4), iscomplexobj_call_result_124289)
    # Assigning a type to the variable 'if_condition_124290' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'if_condition_124290', if_condition_124290)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 218)
    # Processing the call arguments (line 218)
    str_124292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 218)
    kwargs_124293 = {}
    # Getting the type of 'TypeError' (line 218)
    TypeError_124291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 218)
    TypeError_call_result_124294 = invoke(stypy.reporting.localization.Localization(__file__, 218, 14), TypeError_124291, *[str_124292], **kwargs_124293)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 218, 8), TypeError_call_result_124294, 'raise parameter', BaseException)
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 219)
    # Getting the type of 'structure' (line 219)
    structure_124295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 7), 'structure')
    # Getting the type of 'None' (line 219)
    None_124296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'None')
    
    (may_be_124297, more_types_in_union_124298) = may_be_none(structure_124295, None_124296)

    if may_be_124297:

        if more_types_in_union_124298:
            # Runtime conditional SSA (line 219)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to generate_binary_structure(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'input' (line 220)
        input_124300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 46), 'input', False)
        # Obtaining the member 'ndim' of a type (line 220)
        ndim_124301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 46), input_124300, 'ndim')
        int_124302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 58), 'int')
        # Processing the call keyword arguments (line 220)
        kwargs_124303 = {}
        # Getting the type of 'generate_binary_structure' (line 220)
        generate_binary_structure_124299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'generate_binary_structure', False)
        # Calling generate_binary_structure(args, kwargs) (line 220)
        generate_binary_structure_call_result_124304 = invoke(stypy.reporting.localization.Localization(__file__, 220, 20), generate_binary_structure_124299, *[ndim_124301, int_124302], **kwargs_124303)
        
        # Assigning a type to the variable 'structure' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'structure', generate_binary_structure_call_result_124304)

        if more_types_in_union_124298:
            # Runtime conditional SSA for else branch (line 219)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_124297) or more_types_in_union_124298):
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to asarray(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'structure' (line 222)
        structure_124307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 34), 'structure', False)
        # Processing the call keyword arguments (line 222)
        # Getting the type of 'bool' (line 222)
        bool_124308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 51), 'bool', False)
        keyword_124309 = bool_124308
        kwargs_124310 = {'dtype': keyword_124309}
        # Getting the type of 'numpy' (line 222)
        numpy_124305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 222)
        asarray_124306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 20), numpy_124305, 'asarray')
        # Calling asarray(args, kwargs) (line 222)
        asarray_call_result_124311 = invoke(stypy.reporting.localization.Localization(__file__, 222, 20), asarray_124306, *[structure_124307], **kwargs_124310)
        
        # Assigning a type to the variable 'structure' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'structure', asarray_call_result_124311)

        if (may_be_124297 and more_types_in_union_124298):
            # SSA join for if statement (line 219)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'structure' (line 223)
    structure_124312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 7), 'structure')
    # Obtaining the member 'ndim' of a type (line 223)
    ndim_124313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 7), structure_124312, 'ndim')
    # Getting the type of 'input' (line 223)
    input_124314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'input')
    # Obtaining the member 'ndim' of a type (line 223)
    ndim_124315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), input_124314, 'ndim')
    # Applying the binary operator '!=' (line 223)
    result_ne_124316 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 7), '!=', ndim_124313, ndim_124315)
    
    # Testing the type of an if condition (line 223)
    if_condition_124317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 4), result_ne_124316)
    # Assigning a type to the variable 'if_condition_124317' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'if_condition_124317', if_condition_124317)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 224)
    # Processing the call arguments (line 224)
    str_124319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 27), 'str', 'structure and input must have same dimensionality')
    # Processing the call keyword arguments (line 224)
    kwargs_124320 = {}
    # Getting the type of 'RuntimeError' (line 224)
    RuntimeError_124318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 224)
    RuntimeError_call_result_124321 = invoke(stypy.reporting.localization.Localization(__file__, 224, 14), RuntimeError_124318, *[str_124319], **kwargs_124320)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 224, 8), RuntimeError_call_result_124321, 'raise parameter', BaseException)
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'structure' (line 225)
    structure_124322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'structure')
    # Obtaining the member 'flags' of a type (line 225)
    flags_124323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 11), structure_124322, 'flags')
    # Obtaining the member 'contiguous' of a type (line 225)
    contiguous_124324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 11), flags_124323, 'contiguous')
    # Applying the 'not' unary operator (line 225)
    result_not__124325 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 7), 'not', contiguous_124324)
    
    # Testing the type of an if condition (line 225)
    if_condition_124326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 4), result_not__124325)
    # Assigning a type to the variable 'if_condition_124326' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'if_condition_124326', if_condition_124326)
    # SSA begins for if statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to copy(...): (line 226)
    # Processing the call keyword arguments (line 226)
    kwargs_124329 = {}
    # Getting the type of 'structure' (line 226)
    structure_124327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'structure', False)
    # Obtaining the member 'copy' of a type (line 226)
    copy_124328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 20), structure_124327, 'copy')
    # Calling copy(args, kwargs) (line 226)
    copy_call_result_124330 = invoke(stypy.reporting.localization.Localization(__file__, 226, 20), copy_124328, *[], **kwargs_124329)
    
    # Assigning a type to the variable 'structure' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'structure', copy_call_result_124330)
    # SSA join for if statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to product(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'structure' (line 227)
    structure_124333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'structure', False)
    # Obtaining the member 'shape' of a type (line 227)
    shape_124334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 21), structure_124333, 'shape')
    # Processing the call keyword arguments (line 227)
    int_124335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 42), 'int')
    keyword_124336 = int_124335
    kwargs_124337 = {'axis': keyword_124336}
    # Getting the type of 'numpy' (line 227)
    numpy_124331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'numpy', False)
    # Obtaining the member 'product' of a type (line 227)
    product_124332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 7), numpy_124331, 'product')
    # Calling product(args, kwargs) (line 227)
    product_call_result_124338 = invoke(stypy.reporting.localization.Localization(__file__, 227, 7), product_124332, *[shape_124334], **kwargs_124337)
    
    int_124339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 47), 'int')
    # Applying the binary operator '<' (line 227)
    result_lt_124340 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 7), '<', product_call_result_124338, int_124339)
    
    # Testing the type of an if condition (line 227)
    if_condition_124341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 4), result_lt_124340)
    # Assigning a type to the variable 'if_condition_124341' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'if_condition_124341', if_condition_124341)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 228)
    # Processing the call arguments (line 228)
    str_124343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 27), 'str', 'structure must not be empty')
    # Processing the call keyword arguments (line 228)
    kwargs_124344 = {}
    # Getting the type of 'RuntimeError' (line 228)
    RuntimeError_124342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 228)
    RuntimeError_call_result_124345 = invoke(stypy.reporting.localization.Localization(__file__, 228, 14), RuntimeError_124342, *[str_124343], **kwargs_124344)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 228, 8), RuntimeError_call_result_124345, 'raise parameter', BaseException)
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 229)
    # Getting the type of 'mask' (line 229)
    mask_124346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'mask')
    # Getting the type of 'None' (line 229)
    None_124347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'None')
    
    (may_be_124348, more_types_in_union_124349) = may_not_be_none(mask_124346, None_124347)

    if may_be_124348:

        if more_types_in_union_124349:
            # Runtime conditional SSA (line 229)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to asarray(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'mask' (line 230)
        mask_124352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 29), 'mask', False)
        # Processing the call keyword arguments (line 230)
        kwargs_124353 = {}
        # Getting the type of 'numpy' (line 230)
        numpy_124350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 230)
        asarray_124351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 15), numpy_124350, 'asarray')
        # Calling asarray(args, kwargs) (line 230)
        asarray_call_result_124354 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), asarray_124351, *[mask_124352], **kwargs_124353)
        
        # Assigning a type to the variable 'mask' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'mask', asarray_call_result_124354)
        
        
        # Getting the type of 'mask' (line 231)
        mask_124355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'mask')
        # Obtaining the member 'shape' of a type (line 231)
        shape_124356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 11), mask_124355, 'shape')
        # Getting the type of 'input' (line 231)
        input_124357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 25), 'input')
        # Obtaining the member 'shape' of a type (line 231)
        shape_124358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 25), input_124357, 'shape')
        # Applying the binary operator '!=' (line 231)
        result_ne_124359 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 11), '!=', shape_124356, shape_124358)
        
        # Testing the type of an if condition (line 231)
        if_condition_124360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 8), result_ne_124359)
        # Assigning a type to the variable 'if_condition_124360' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'if_condition_124360', if_condition_124360)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 232)
        # Processing the call arguments (line 232)
        str_124362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 31), 'str', 'mask and input must have equal sizes')
        # Processing the call keyword arguments (line 232)
        kwargs_124363 = {}
        # Getting the type of 'RuntimeError' (line 232)
        RuntimeError_124361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 232)
        RuntimeError_call_result_124364 = invoke(stypy.reporting.localization.Localization(__file__, 232, 18), RuntimeError_124361, *[str_124362], **kwargs_124363)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 232, 12), RuntimeError_call_result_124364, 'raise parameter', BaseException)
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_124349:
            # SSA join for if statement (line 229)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 233):
    
    # Assigning a Call to a Name (line 233):
    
    # Call to _normalize_sequence(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'origin' (line 233)
    origin_124367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 45), 'origin', False)
    # Getting the type of 'input' (line 233)
    input_124368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 53), 'input', False)
    # Obtaining the member 'ndim' of a type (line 233)
    ndim_124369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 53), input_124368, 'ndim')
    # Processing the call keyword arguments (line 233)
    kwargs_124370 = {}
    # Getting the type of '_ni_support' (line 233)
    _ni_support_124365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 233)
    _normalize_sequence_124366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 13), _ni_support_124365, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 233)
    _normalize_sequence_call_result_124371 = invoke(stypy.reporting.localization.Localization(__file__, 233, 13), _normalize_sequence_124366, *[origin_124367, ndim_124369], **kwargs_124370)
    
    # Assigning a type to the variable 'origin' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'origin', _normalize_sequence_call_result_124371)
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to _center_is_true(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'structure' (line 234)
    structure_124373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 26), 'structure', False)
    # Getting the type of 'origin' (line 234)
    origin_124374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 37), 'origin', False)
    # Processing the call keyword arguments (line 234)
    kwargs_124375 = {}
    # Getting the type of '_center_is_true' (line 234)
    _center_is_true_124372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 10), '_center_is_true', False)
    # Calling _center_is_true(args, kwargs) (line 234)
    _center_is_true_call_result_124376 = invoke(stypy.reporting.localization.Localization(__file__, 234, 10), _center_is_true_124372, *[structure_124373, origin_124374], **kwargs_124375)
    
    # Assigning a type to the variable 'cit' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'cit', _center_is_true_call_result_124376)
    
    
    # Call to isinstance(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'output' (line 235)
    output_124378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 18), 'output', False)
    # Getting the type of 'numpy' (line 235)
    numpy_124379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 26), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 235)
    ndarray_124380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 26), numpy_124379, 'ndarray')
    # Processing the call keyword arguments (line 235)
    kwargs_124381 = {}
    # Getting the type of 'isinstance' (line 235)
    isinstance_124377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 235)
    isinstance_call_result_124382 = invoke(stypy.reporting.localization.Localization(__file__, 235, 7), isinstance_124377, *[output_124378, ndarray_124380], **kwargs_124381)
    
    # Testing the type of an if condition (line 235)
    if_condition_124383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 4), isinstance_call_result_124382)
    # Assigning a type to the variable 'if_condition_124383' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'if_condition_124383', if_condition_124383)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to iscomplexobj(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'output' (line 236)
    output_124386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), 'output', False)
    # Processing the call keyword arguments (line 236)
    kwargs_124387 = {}
    # Getting the type of 'numpy' (line 236)
    numpy_124384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 236)
    iscomplexobj_124385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 11), numpy_124384, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 236)
    iscomplexobj_call_result_124388 = invoke(stypy.reporting.localization.Localization(__file__, 236, 11), iscomplexobj_124385, *[output_124386], **kwargs_124387)
    
    # Testing the type of an if condition (line 236)
    if_condition_124389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), iscomplexobj_call_result_124388)
    # Assigning a type to the variable 'if_condition_124389' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_124389', if_condition_124389)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 237)
    # Processing the call arguments (line 237)
    str_124391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 28), 'str', 'Complex output type not supported')
    # Processing the call keyword arguments (line 237)
    kwargs_124392 = {}
    # Getting the type of 'TypeError' (line 237)
    TypeError_124390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 237)
    TypeError_call_result_124393 = invoke(stypy.reporting.localization.Localization(__file__, 237, 18), TypeError_124390, *[str_124391], **kwargs_124392)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 237, 12), TypeError_call_result_124393, 'raise parameter', BaseException)
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 235)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 239):
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'bool' (line 239)
    bool_124394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 17), 'bool')
    # Assigning a type to the variable 'output' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'output', bool_124394)
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 240):
    
    # Assigning a Subscript to a Name (line 240):
    
    # Obtaining the type of the subscript
    int_124395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'int')
    
    # Call to _get_output(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'output' (line 240)
    output_124398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 51), 'output', False)
    # Getting the type of 'input' (line 240)
    input_124399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 59), 'input', False)
    # Processing the call keyword arguments (line 240)
    kwargs_124400 = {}
    # Getting the type of '_ni_support' (line 240)
    _ni_support_124396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 240)
    _get_output_124397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 27), _ni_support_124396, '_get_output')
    # Calling _get_output(args, kwargs) (line 240)
    _get_output_call_result_124401 = invoke(stypy.reporting.localization.Localization(__file__, 240, 27), _get_output_124397, *[output_124398, input_124399], **kwargs_124400)
    
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___124402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), _get_output_call_result_124401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_124403 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), getitem___124402, int_124395)
    
    # Assigning a type to the variable 'tuple_var_assignment_124058' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_124058', subscript_call_result_124403)
    
    # Assigning a Subscript to a Name (line 240):
    
    # Obtaining the type of the subscript
    int_124404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'int')
    
    # Call to _get_output(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'output' (line 240)
    output_124407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 51), 'output', False)
    # Getting the type of 'input' (line 240)
    input_124408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 59), 'input', False)
    # Processing the call keyword arguments (line 240)
    kwargs_124409 = {}
    # Getting the type of '_ni_support' (line 240)
    _ni_support_124405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 240)
    _get_output_124406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 27), _ni_support_124405, '_get_output')
    # Calling _get_output(args, kwargs) (line 240)
    _get_output_call_result_124410 = invoke(stypy.reporting.localization.Localization(__file__, 240, 27), _get_output_124406, *[output_124407, input_124408], **kwargs_124409)
    
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___124411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), _get_output_call_result_124410, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_124412 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), getitem___124411, int_124404)
    
    # Assigning a type to the variable 'tuple_var_assignment_124059' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_124059', subscript_call_result_124412)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_var_assignment_124058' (line 240)
    tuple_var_assignment_124058_124413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_124058')
    # Assigning a type to the variable 'output' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'output', tuple_var_assignment_124058_124413)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_var_assignment_124059' (line 240)
    tuple_var_assignment_124059_124414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_124059')
    # Assigning a type to the variable 'return_value' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'return_value', tuple_var_assignment_124059_124414)
    
    
    # Getting the type of 'iterations' (line 242)
    iterations_124415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 7), 'iterations')
    int_124416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 21), 'int')
    # Applying the binary operator '==' (line 242)
    result_eq_124417 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 7), '==', iterations_124415, int_124416)
    
    # Testing the type of an if condition (line 242)
    if_condition_124418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 4), result_eq_124417)
    # Assigning a type to the variable 'if_condition_124418' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'if_condition_124418', if_condition_124418)
    # SSA begins for if statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to binary_erosion(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'input' (line 243)
    input_124421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 33), 'input', False)
    # Getting the type of 'structure' (line 243)
    structure_124422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 40), 'structure', False)
    # Getting the type of 'mask' (line 243)
    mask_124423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 51), 'mask', False)
    # Getting the type of 'output' (line 243)
    output_124424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 57), 'output', False)
    # Getting the type of 'border_value' (line 244)
    border_value_124425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 37), 'border_value', False)
    # Getting the type of 'origin' (line 244)
    origin_124426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 51), 'origin', False)
    # Getting the type of 'invert' (line 244)
    invert_124427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 59), 'invert', False)
    # Getting the type of 'cit' (line 244)
    cit_124428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 67), 'cit', False)
    int_124429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 72), 'int')
    # Processing the call keyword arguments (line 243)
    kwargs_124430 = {}
    # Getting the type of '_nd_image' (line 243)
    _nd_image_124419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), '_nd_image', False)
    # Obtaining the member 'binary_erosion' of a type (line 243)
    binary_erosion_124420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), _nd_image_124419, 'binary_erosion')
    # Calling binary_erosion(args, kwargs) (line 243)
    binary_erosion_call_result_124431 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), binary_erosion_124420, *[input_124421, structure_124422, mask_124423, output_124424, border_value_124425, origin_124426, invert_124427, cit_124428, int_124429], **kwargs_124430)
    
    # Getting the type of 'return_value' (line 245)
    return_value_124432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', return_value_124432)
    # SSA branch for the else part of an if statement (line 242)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'cit' (line 246)
    cit_124433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'cit')
    
    # Getting the type of 'brute_force' (line 246)
    brute_force_124434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'brute_force')
    # Applying the 'not' unary operator (line 246)
    result_not__124435 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 17), 'not', brute_force_124434)
    
    # Applying the binary operator 'and' (line 246)
    result_and_keyword_124436 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 9), 'and', cit_124433, result_not__124435)
    
    # Testing the type of an if condition (line 246)
    if_condition_124437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 9), result_and_keyword_124436)
    # Assigning a type to the variable 'if_condition_124437' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'if_condition_124437', if_condition_124437)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 247):
    
    # Assigning a Subscript to a Name (line 247):
    
    # Obtaining the type of the subscript
    int_124438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'int')
    
    # Call to binary_erosion(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'input' (line 247)
    input_124441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 60), 'input', False)
    # Getting the type of 'structure' (line 248)
    structure_124442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 13), 'structure', False)
    # Getting the type of 'mask' (line 248)
    mask_124443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'mask', False)
    # Getting the type of 'output' (line 248)
    output_124444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 30), 'output', False)
    # Getting the type of 'border_value' (line 248)
    border_value_124445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 38), 'border_value', False)
    # Getting the type of 'origin' (line 248)
    origin_124446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 52), 'origin', False)
    # Getting the type of 'invert' (line 248)
    invert_124447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 60), 'invert', False)
    # Getting the type of 'cit' (line 248)
    cit_124448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 68), 'cit', False)
    int_124449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 73), 'int')
    # Processing the call keyword arguments (line 247)
    kwargs_124450 = {}
    # Getting the type of '_nd_image' (line 247)
    _nd_image_124439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 35), '_nd_image', False)
    # Obtaining the member 'binary_erosion' of a type (line 247)
    binary_erosion_124440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 35), _nd_image_124439, 'binary_erosion')
    # Calling binary_erosion(args, kwargs) (line 247)
    binary_erosion_call_result_124451 = invoke(stypy.reporting.localization.Localization(__file__, 247, 35), binary_erosion_124440, *[input_124441, structure_124442, mask_124443, output_124444, border_value_124445, origin_124446, invert_124447, cit_124448, int_124449], **kwargs_124450)
    
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___124452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), binary_erosion_call_result_124451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_124453 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), getitem___124452, int_124438)
    
    # Assigning a type to the variable 'tuple_var_assignment_124060' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tuple_var_assignment_124060', subscript_call_result_124453)
    
    # Assigning a Subscript to a Name (line 247):
    
    # Obtaining the type of the subscript
    int_124454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'int')
    
    # Call to binary_erosion(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'input' (line 247)
    input_124457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 60), 'input', False)
    # Getting the type of 'structure' (line 248)
    structure_124458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 13), 'structure', False)
    # Getting the type of 'mask' (line 248)
    mask_124459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'mask', False)
    # Getting the type of 'output' (line 248)
    output_124460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 30), 'output', False)
    # Getting the type of 'border_value' (line 248)
    border_value_124461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 38), 'border_value', False)
    # Getting the type of 'origin' (line 248)
    origin_124462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 52), 'origin', False)
    # Getting the type of 'invert' (line 248)
    invert_124463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 60), 'invert', False)
    # Getting the type of 'cit' (line 248)
    cit_124464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 68), 'cit', False)
    int_124465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 73), 'int')
    # Processing the call keyword arguments (line 247)
    kwargs_124466 = {}
    # Getting the type of '_nd_image' (line 247)
    _nd_image_124455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 35), '_nd_image', False)
    # Obtaining the member 'binary_erosion' of a type (line 247)
    binary_erosion_124456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 35), _nd_image_124455, 'binary_erosion')
    # Calling binary_erosion(args, kwargs) (line 247)
    binary_erosion_call_result_124467 = invoke(stypy.reporting.localization.Localization(__file__, 247, 35), binary_erosion_124456, *[input_124457, structure_124458, mask_124459, output_124460, border_value_124461, origin_124462, invert_124463, cit_124464, int_124465], **kwargs_124466)
    
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___124468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), binary_erosion_call_result_124467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_124469 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), getitem___124468, int_124454)
    
    # Assigning a type to the variable 'tuple_var_assignment_124061' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tuple_var_assignment_124061', subscript_call_result_124469)
    
    # Assigning a Name to a Name (line 247):
    # Getting the type of 'tuple_var_assignment_124060' (line 247)
    tuple_var_assignment_124060_124470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tuple_var_assignment_124060')
    # Assigning a type to the variable 'changed' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'changed', tuple_var_assignment_124060_124470)
    
    # Assigning a Name to a Name (line 247):
    # Getting the type of 'tuple_var_assignment_124061' (line 247)
    tuple_var_assignment_124061_124471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tuple_var_assignment_124061')
    # Assigning a type to the variable 'coordinate_list' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'coordinate_list', tuple_var_assignment_124061_124471)
    
    # Assigning a Subscript to a Name (line 249):
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Obtaining an instance of the builtin type 'list' (line 249)
    list_124473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 249)
    # Adding element type (line 249)
    
    # Call to slice(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'None' (line 249)
    None_124475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 43), 'None', False)
    # Getting the type of 'None' (line 249)
    None_124476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 49), 'None', False)
    int_124477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 55), 'int')
    # Processing the call keyword arguments (line 249)
    kwargs_124478 = {}
    # Getting the type of 'slice' (line 249)
    slice_124474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 37), 'slice', False)
    # Calling slice(args, kwargs) (line 249)
    slice_call_result_124479 = invoke(stypy.reporting.localization.Localization(__file__, 249, 37), slice_124474, *[None_124475, None_124476, int_124477], **kwargs_124478)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 36), list_124473, slice_call_result_124479)
    
    # Getting the type of 'structure' (line 250)
    structure_124480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 36), 'structure', False)
    # Obtaining the member 'ndim' of a type (line 250)
    ndim_124481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 36), structure_124480, 'ndim')
    # Applying the binary operator '*' (line 249)
    result_mul_124482 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 36), '*', list_124473, ndim_124481)
    
    # Processing the call keyword arguments (line 249)
    kwargs_124483 = {}
    # Getting the type of 'tuple' (line 249)
    tuple_124472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 30), 'tuple', False)
    # Calling tuple(args, kwargs) (line 249)
    tuple_call_result_124484 = invoke(stypy.reporting.localization.Localization(__file__, 249, 30), tuple_124472, *[result_mul_124482], **kwargs_124483)
    
    # Getting the type of 'structure' (line 249)
    structure_124485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'structure')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___124486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 20), structure_124485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_124487 = invoke(stypy.reporting.localization.Localization(__file__, 249, 20), getitem___124486, tuple_call_result_124484)
    
    # Assigning a type to the variable 'structure' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'structure', subscript_call_result_124487)
    
    
    # Call to range(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Call to len(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'origin' (line 251)
    origin_124490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'origin', False)
    # Processing the call keyword arguments (line 251)
    kwargs_124491 = {}
    # Getting the type of 'len' (line 251)
    len_124489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'len', False)
    # Calling len(args, kwargs) (line 251)
    len_call_result_124492 = invoke(stypy.reporting.localization.Localization(__file__, 251, 24), len_124489, *[origin_124490], **kwargs_124491)
    
    # Processing the call keyword arguments (line 251)
    kwargs_124493 = {}
    # Getting the type of 'range' (line 251)
    range_124488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'range', False)
    # Calling range(args, kwargs) (line 251)
    range_call_result_124494 = invoke(stypy.reporting.localization.Localization(__file__, 251, 18), range_124488, *[len_call_result_124492], **kwargs_124493)
    
    # Testing the type of a for loop iterable (line 251)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 251, 8), range_call_result_124494)
    # Getting the type of the for loop variable (line 251)
    for_loop_var_124495 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 251, 8), range_call_result_124494)
    # Assigning a type to the variable 'ii' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'ii', for_loop_var_124495)
    # SSA begins for a for statement (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a UnaryOp to a Subscript (line 252):
    
    # Assigning a UnaryOp to a Subscript (line 252):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 252)
    ii_124496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 33), 'ii')
    # Getting the type of 'origin' (line 252)
    origin_124497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'origin')
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___124498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 26), origin_124497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_124499 = invoke(stypy.reporting.localization.Localization(__file__, 252, 26), getitem___124498, ii_124496)
    
    # Applying the 'usub' unary operator (line 252)
    result___neg___124500 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 25), 'usub', subscript_call_result_124499)
    
    # Getting the type of 'origin' (line 252)
    origin_124501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'origin')
    # Getting the type of 'ii' (line 252)
    ii_124502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'ii')
    # Storing an element on a container (line 252)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 12), origin_124501, (ii_124502, result___neg___124500))
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 253)
    ii_124503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'ii')
    # Getting the type of 'structure' (line 253)
    structure_124504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'structure')
    # Obtaining the member 'shape' of a type (line 253)
    shape_124505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 19), structure_124504, 'shape')
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___124506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 19), shape_124505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_124507 = invoke(stypy.reporting.localization.Localization(__file__, 253, 19), getitem___124506, ii_124503)
    
    int_124508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 41), 'int')
    # Applying the binary operator '&' (line 253)
    result_and__124509 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 19), '&', subscript_call_result_124507, int_124508)
    
    # Applying the 'not' unary operator (line 253)
    result_not__124510 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 15), 'not', result_and__124509)
    
    # Testing the type of an if condition (line 253)
    if_condition_124511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 12), result_not__124510)
    # Assigning a type to the variable 'if_condition_124511' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'if_condition_124511', if_condition_124511)
    # SSA begins for if statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'origin' (line 254)
    origin_124512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'origin')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 254)
    ii_124513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'ii')
    # Getting the type of 'origin' (line 254)
    origin_124514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'origin')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___124515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 16), origin_124514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_124516 = invoke(stypy.reporting.localization.Localization(__file__, 254, 16), getitem___124515, ii_124513)
    
    int_124517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 30), 'int')
    # Applying the binary operator '-=' (line 254)
    result_isub_124518 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 16), '-=', subscript_call_result_124516, int_124517)
    # Getting the type of 'origin' (line 254)
    origin_124519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'origin')
    # Getting the type of 'ii' (line 254)
    ii_124520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'ii')
    # Storing an element on a container (line 254)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 16), origin_124519, (ii_124520, result_isub_124518))
    
    # SSA join for if statement (line 253)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 255)
    # Getting the type of 'mask' (line 255)
    mask_124521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'mask')
    # Getting the type of 'None' (line 255)
    None_124522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 'None')
    
    (may_be_124523, more_types_in_union_124524) = may_not_be_none(mask_124521, None_124522)

    if may_be_124523:

        if more_types_in_union_124524:
            # Runtime conditional SSA (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to asarray(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'mask' (line 256)
        mask_124527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 33), 'mask', False)
        # Processing the call keyword arguments (line 256)
        # Getting the type of 'numpy' (line 256)
        numpy_124528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 45), 'numpy', False)
        # Obtaining the member 'int8' of a type (line 256)
        int8_124529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 45), numpy_124528, 'int8')
        keyword_124530 = int8_124529
        kwargs_124531 = {'dtype': keyword_124530}
        # Getting the type of 'numpy' (line 256)
        numpy_124525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 256)
        asarray_124526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 19), numpy_124525, 'asarray')
        # Calling asarray(args, kwargs) (line 256)
        asarray_call_result_124532 = invoke(stypy.reporting.localization.Localization(__file__, 256, 19), asarray_124526, *[mask_124527], **kwargs_124531)
        
        # Assigning a type to the variable 'mask' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'mask', asarray_call_result_124532)

        if more_types_in_union_124524:
            # SSA join for if statement (line 255)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'structure' (line 257)
    structure_124533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'structure')
    # Obtaining the member 'flags' of a type (line 257)
    flags_124534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 15), structure_124533, 'flags')
    # Obtaining the member 'contiguous' of a type (line 257)
    contiguous_124535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 15), flags_124534, 'contiguous')
    # Applying the 'not' unary operator (line 257)
    result_not__124536 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), 'not', contiguous_124535)
    
    # Testing the type of an if condition (line 257)
    if_condition_124537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), result_not__124536)
    # Assigning a type to the variable 'if_condition_124537' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_124537', if_condition_124537)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to copy(...): (line 258)
    # Processing the call keyword arguments (line 258)
    kwargs_124540 = {}
    # Getting the type of 'structure' (line 258)
    structure_124538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'structure', False)
    # Obtaining the member 'copy' of a type (line 258)
    copy_124539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 24), structure_124538, 'copy')
    # Calling copy(args, kwargs) (line 258)
    copy_call_result_124541 = invoke(stypy.reporting.localization.Localization(__file__, 258, 24), copy_124539, *[], **kwargs_124540)
    
    # Assigning a type to the variable 'structure' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'structure', copy_call_result_124541)
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to binary_erosion2(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'output' (line 259)
    output_124544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'output', False)
    # Getting the type of 'structure' (line 259)
    structure_124545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 42), 'structure', False)
    # Getting the type of 'mask' (line 259)
    mask_124546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 53), 'mask', False)
    # Getting the type of 'iterations' (line 259)
    iterations_124547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 59), 'iterations', False)
    int_124548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 72), 'int')
    # Applying the binary operator '-' (line 259)
    result_sub_124549 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 59), '-', iterations_124547, int_124548)
    
    # Getting the type of 'origin' (line 260)
    origin_124550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'origin', False)
    # Getting the type of 'invert' (line 260)
    invert_124551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 42), 'invert', False)
    # Getting the type of 'coordinate_list' (line 260)
    coordinate_list_124552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 50), 'coordinate_list', False)
    # Processing the call keyword arguments (line 259)
    kwargs_124553 = {}
    # Getting the type of '_nd_image' (line 259)
    _nd_image_124542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), '_nd_image', False)
    # Obtaining the member 'binary_erosion2' of a type (line 259)
    binary_erosion2_124543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), _nd_image_124542, 'binary_erosion2')
    # Calling binary_erosion2(args, kwargs) (line 259)
    binary_erosion2_call_result_124554 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), binary_erosion2_124543, *[output_124544, structure_124545, mask_124546, result_sub_124549, origin_124550, invert_124551, coordinate_list_124552], **kwargs_124553)
    
    # Getting the type of 'return_value' (line 261)
    return_value_124555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'stypy_return_type', return_value_124555)
    # SSA branch for the else part of an if statement (line 246)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 263):
    
    # Assigning a Call to a Name (line 263):
    
    # Call to zeros(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'input' (line 263)
    input_124558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'input', False)
    # Obtaining the member 'shape' of a type (line 263)
    shape_124559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 29), input_124558, 'shape')
    # Getting the type of 'bool' (line 263)
    bool_124560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 42), 'bool', False)
    # Processing the call keyword arguments (line 263)
    kwargs_124561 = {}
    # Getting the type of 'numpy' (line 263)
    numpy_124556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 263)
    zeros_124557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 17), numpy_124556, 'zeros')
    # Calling zeros(args, kwargs) (line 263)
    zeros_call_result_124562 = invoke(stypy.reporting.localization.Localization(__file__, 263, 17), zeros_124557, *[shape_124559, bool_124560], **kwargs_124561)
    
    # Assigning a type to the variable 'tmp_in' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tmp_in', zeros_call_result_124562)
    
    # Type idiom detected: calculating its left and rigth part (line 264)
    # Getting the type of 'return_value' (line 264)
    return_value_124563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'return_value')
    # Getting the type of 'None' (line 264)
    None_124564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 27), 'None')
    
    (may_be_124565, more_types_in_union_124566) = may_be_none(return_value_124563, None_124564)

    if may_be_124565:

        if more_types_in_union_124566:
            # Runtime conditional SSA (line 264)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 265):
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'output' (line 265)
        output_124567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'output')
        # Assigning a type to the variable 'tmp_out' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'tmp_out', output_124567)

        if more_types_in_union_124566:
            # Runtime conditional SSA for else branch (line 264)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_124565) or more_types_in_union_124566):
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to zeros(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'input' (line 267)
        input_124570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 34), 'input', False)
        # Obtaining the member 'shape' of a type (line 267)
        shape_124571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 34), input_124570, 'shape')
        # Getting the type of 'bool' (line 267)
        bool_124572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 47), 'bool', False)
        # Processing the call keyword arguments (line 267)
        kwargs_124573 = {}
        # Getting the type of 'numpy' (line 267)
        numpy_124568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 22), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 267)
        zeros_124569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 22), numpy_124568, 'zeros')
        # Calling zeros(args, kwargs) (line 267)
        zeros_call_result_124574 = invoke(stypy.reporting.localization.Localization(__file__, 267, 22), zeros_124569, *[shape_124571, bool_124572], **kwargs_124573)
        
        # Assigning a type to the variable 'tmp_out' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'tmp_out', zeros_call_result_124574)

        if (may_be_124565 and more_types_in_union_124566):
            # SSA join for if statement (line 264)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'iterations' (line 268)
    iterations_124575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'iterations')
    int_124576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 28), 'int')
    # Applying the binary operator '&' (line 268)
    result_and__124577 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 15), '&', iterations_124575, int_124576)
    
    # Applying the 'not' unary operator (line 268)
    result_not__124578 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 11), 'not', result_and__124577)
    
    # Testing the type of an if condition (line 268)
    if_condition_124579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), result_not__124578)
    # Assigning a type to the variable 'if_condition_124579' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_124579', if_condition_124579)
    # SSA begins for if statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 269):
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'tmp_out' (line 269)
    tmp_out_124580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 30), 'tmp_out')
    # Assigning a type to the variable 'tuple_assignment_124062' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_assignment_124062', tmp_out_124580)
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'tmp_in' (line 269)
    tmp_in_124581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 39), 'tmp_in')
    # Assigning a type to the variable 'tuple_assignment_124063' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_assignment_124063', tmp_in_124581)
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'tuple_assignment_124062' (line 269)
    tuple_assignment_124062_124582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_assignment_124062')
    # Assigning a type to the variable 'tmp_in' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tmp_in', tuple_assignment_124062_124582)
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'tuple_assignment_124063' (line 269)
    tuple_assignment_124063_124583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_assignment_124063')
    # Assigning a type to the variable 'tmp_out' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'tmp_out', tuple_assignment_124063_124583)
    # SSA join for if statement (line 268)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to binary_erosion(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'input' (line 270)
    input_124586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 43), 'input', False)
    # Getting the type of 'structure' (line 270)
    structure_124587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'structure', False)
    # Getting the type of 'mask' (line 270)
    mask_124588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 61), 'mask', False)
    # Getting the type of 'tmp_out' (line 271)
    tmp_out_124589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 28), 'tmp_out', False)
    # Getting the type of 'border_value' (line 271)
    border_value_124590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 37), 'border_value', False)
    # Getting the type of 'origin' (line 271)
    origin_124591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 51), 'origin', False)
    # Getting the type of 'invert' (line 271)
    invert_124592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 59), 'invert', False)
    # Getting the type of 'cit' (line 271)
    cit_124593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 67), 'cit', False)
    int_124594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 72), 'int')
    # Processing the call keyword arguments (line 270)
    kwargs_124595 = {}
    # Getting the type of '_nd_image' (line 270)
    _nd_image_124584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 18), '_nd_image', False)
    # Obtaining the member 'binary_erosion' of a type (line 270)
    binary_erosion_124585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 18), _nd_image_124584, 'binary_erosion')
    # Calling binary_erosion(args, kwargs) (line 270)
    binary_erosion_call_result_124596 = invoke(stypy.reporting.localization.Localization(__file__, 270, 18), binary_erosion_124585, *[input_124586, structure_124587, mask_124588, tmp_out_124589, border_value_124590, origin_124591, invert_124592, cit_124593, int_124594], **kwargs_124595)
    
    # Assigning a type to the variable 'changed' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'changed', binary_erosion_call_result_124596)
    
    # Assigning a Num to a Name (line 272):
    
    # Assigning a Num to a Name (line 272):
    int_124597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 13), 'int')
    # Assigning a type to the variable 'ii' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'ii', int_124597)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ii' (line 273)
    ii_124598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'ii')
    # Getting the type of 'iterations' (line 273)
    iterations_124599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'iterations')
    # Applying the binary operator '<' (line 273)
    result_lt_124600 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 15), '<', ii_124598, iterations_124599)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'iterations' (line 273)
    iterations_124601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 36), 'iterations')
    int_124602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 49), 'int')
    # Applying the binary operator '<' (line 273)
    result_lt_124603 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 36), '<', iterations_124601, int_124602)
    
    # Getting the type of 'changed' (line 273)
    changed_124604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 56), 'changed')
    # Applying the binary operator 'and' (line 273)
    result_and_keyword_124605 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 35), 'and', result_lt_124603, changed_124604)
    
    # Applying the binary operator 'or' (line 273)
    result_or_keyword_124606 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 14), 'or', result_lt_124600, result_and_keyword_124605)
    
    # Testing the type of an if condition (line 273)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_or_keyword_124606)
    # SSA begins for while statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Tuple to a Tuple (line 274):
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'tmp_out' (line 274)
    tmp_out_124607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 30), 'tmp_out')
    # Assigning a type to the variable 'tuple_assignment_124064' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'tuple_assignment_124064', tmp_out_124607)
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'tmp_in' (line 274)
    tmp_in_124608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 39), 'tmp_in')
    # Assigning a type to the variable 'tuple_assignment_124065' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'tuple_assignment_124065', tmp_in_124608)
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'tuple_assignment_124064' (line 274)
    tuple_assignment_124064_124609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'tuple_assignment_124064')
    # Assigning a type to the variable 'tmp_in' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'tmp_in', tuple_assignment_124064_124609)
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'tuple_assignment_124065' (line 274)
    tuple_assignment_124065_124610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'tuple_assignment_124065')
    # Assigning a type to the variable 'tmp_out' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'tmp_out', tuple_assignment_124065_124610)
    
    # Assigning a Call to a Name (line 275):
    
    # Assigning a Call to a Name (line 275):
    
    # Call to binary_erosion(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'tmp_in' (line 275)
    tmp_in_124613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 47), 'tmp_in', False)
    # Getting the type of 'structure' (line 275)
    structure_124614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 55), 'structure', False)
    # Getting the type of 'mask' (line 275)
    mask_124615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 66), 'mask', False)
    # Getting the type of 'tmp_out' (line 276)
    tmp_out_124616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'tmp_out', False)
    # Getting the type of 'border_value' (line 276)
    border_value_124617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 37), 'border_value', False)
    # Getting the type of 'origin' (line 276)
    origin_124618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 51), 'origin', False)
    # Getting the type of 'invert' (line 276)
    invert_124619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 59), 'invert', False)
    # Getting the type of 'cit' (line 276)
    cit_124620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 67), 'cit', False)
    int_124621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 72), 'int')
    # Processing the call keyword arguments (line 275)
    kwargs_124622 = {}
    # Getting the type of '_nd_image' (line 275)
    _nd_image_124611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), '_nd_image', False)
    # Obtaining the member 'binary_erosion' of a type (line 275)
    binary_erosion_124612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 22), _nd_image_124611, 'binary_erosion')
    # Calling binary_erosion(args, kwargs) (line 275)
    binary_erosion_call_result_124623 = invoke(stypy.reporting.localization.Localization(__file__, 275, 22), binary_erosion_124612, *[tmp_in_124613, structure_124614, mask_124615, tmp_out_124616, border_value_124617, origin_124618, invert_124619, cit_124620, int_124621], **kwargs_124622)
    
    # Assigning a type to the variable 'changed' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'changed', binary_erosion_call_result_124623)
    
    # Getting the type of 'ii' (line 277)
    ii_124624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'ii')
    int_124625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 18), 'int')
    # Applying the binary operator '+=' (line 277)
    result_iadd_124626 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 12), '+=', ii_124624, int_124625)
    # Assigning a type to the variable 'ii' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'ii', result_iadd_124626)
    
    # SSA join for while statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 278)
    # Getting the type of 'return_value' (line 278)
    return_value_124627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'return_value')
    # Getting the type of 'None' (line 278)
    None_124628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 31), 'None')
    
    (may_be_124629, more_types_in_union_124630) = may_not_be_none(return_value_124627, None_124628)

    if may_be_124629:

        if more_types_in_union_124630:
            # Runtime conditional SSA (line 278)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'tmp_out' (line 279)
        tmp_out_124631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 'tmp_out')
        # Assigning a type to the variable 'stypy_return_type' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'stypy_return_type', tmp_out_124631)

        if more_types_in_union_124630:
            # SSA join for if statement (line 278)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 242)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_binary_erosion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_binary_erosion' in the type store
    # Getting the type of 'stypy_return_type' (line 214)
    stypy_return_type_124632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124632)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_binary_erosion'
    return stypy_return_type_124632

# Assigning a type to the variable '_binary_erosion' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), '_binary_erosion', _binary_erosion)

@norecursion
def binary_erosion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 282)
    None_124633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 36), 'None')
    int_124634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 53), 'int')
    # Getting the type of 'None' (line 282)
    None_124635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 61), 'None')
    # Getting the type of 'None' (line 283)
    None_124636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'None')
    int_124637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 34), 'int')
    int_124638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 44), 'int')
    # Getting the type of 'False' (line 283)
    False_124639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 59), 'False')
    defaults = [None_124633, int_124634, None_124635, None_124636, int_124637, int_124638, False_124639]
    # Create a new context for function 'binary_erosion'
    module_type_store = module_type_store.open_function_context('binary_erosion', 282, 0, False)
    
    # Passed parameters checking function
    binary_erosion.stypy_localization = localization
    binary_erosion.stypy_type_of_self = None
    binary_erosion.stypy_type_store = module_type_store
    binary_erosion.stypy_function_name = 'binary_erosion'
    binary_erosion.stypy_param_names_list = ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'brute_force']
    binary_erosion.stypy_varargs_param_name = None
    binary_erosion.stypy_kwargs_param_name = None
    binary_erosion.stypy_call_defaults = defaults
    binary_erosion.stypy_call_varargs = varargs
    binary_erosion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binary_erosion', ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'brute_force'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binary_erosion', localization, ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'brute_force'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binary_erosion(...)' code ##################

    str_124640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, (-1)), 'str', '\n    Multi-dimensional binary erosion with a given structuring element.\n\n    Binary erosion is a mathematical morphology operation used for image\n    processing.\n\n    Parameters\n    ----------\n    input : array_like\n        Binary image to be eroded. Non-zero (True) elements form\n        the subset to be eroded.\n    structure : array_like, optional\n        Structuring element used for the erosion. Non-zero elements are\n        considered True. If no structuring element is provided, an element\n        is generated with a square connectivity equal to one.\n    iterations : {int, float}, optional\n        The erosion is repeated `iterations` times (one, by default).\n        If iterations is less than 1, the erosion is repeated until the\n        result does not change anymore.\n    mask : array_like, optional\n        If a mask is given, only those elements with a True value at\n        the corresponding mask element are modified at each iteration.\n    output : ndarray, optional\n        Array of the same shape as input, into which the output is placed.\n        By default, a new array is created.\n    origin : int or tuple of ints, optional\n        Placement of the filter, by default 0.\n    border_value : int (cast to 0 or 1), optional\n        Value at the border in the output array.\n\n    Returns\n    -------\n    binary_erosion : ndarray of bools\n        Erosion of the input by the structuring element.\n\n    See also\n    --------\n    grey_erosion, binary_dilation, binary_closing, binary_opening,\n    generate_binary_structure\n\n    Notes\n    -----\n    Erosion [1]_ is a mathematical morphology operation [2]_ that uses a\n    structuring element for shrinking the shapes in an image. The binary\n    erosion of an image by a structuring element is the locus of the points\n    where a superimposition of the structuring element centered on the point\n    is entirely contained in the set of non-zero elements of the image.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Erosion_%28morphology%29\n    .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((7,7), dtype=int)\n    >>> a[1:6, 2:5] = 1\n    >>> a\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> ndimage.binary_erosion(a).astype(a.dtype)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> #Erosion removes objects smaller than the structure\n    >>> ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n\n    ')
    
    # Call to _binary_erosion(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'input' (line 369)
    input_124642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 27), 'input', False)
    # Getting the type of 'structure' (line 369)
    structure_124643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 34), 'structure', False)
    # Getting the type of 'iterations' (line 369)
    iterations_124644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 45), 'iterations', False)
    # Getting the type of 'mask' (line 369)
    mask_124645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 57), 'mask', False)
    # Getting the type of 'output' (line 370)
    output_124646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'output', False)
    # Getting the type of 'border_value' (line 370)
    border_value_124647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 35), 'border_value', False)
    # Getting the type of 'origin' (line 370)
    origin_124648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 49), 'origin', False)
    int_124649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 57), 'int')
    # Getting the type of 'brute_force' (line 370)
    brute_force_124650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 60), 'brute_force', False)
    # Processing the call keyword arguments (line 369)
    kwargs_124651 = {}
    # Getting the type of '_binary_erosion' (line 369)
    _binary_erosion_124641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), '_binary_erosion', False)
    # Calling _binary_erosion(args, kwargs) (line 369)
    _binary_erosion_call_result_124652 = invoke(stypy.reporting.localization.Localization(__file__, 369, 11), _binary_erosion_124641, *[input_124642, structure_124643, iterations_124644, mask_124645, output_124646, border_value_124647, origin_124648, int_124649, brute_force_124650], **kwargs_124651)
    
    # Assigning a type to the variable 'stypy_return_type' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type', _binary_erosion_call_result_124652)
    
    # ################# End of 'binary_erosion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binary_erosion' in the type store
    # Getting the type of 'stypy_return_type' (line 282)
    stypy_return_type_124653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124653)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binary_erosion'
    return stypy_return_type_124653

# Assigning a type to the variable 'binary_erosion' (line 282)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'binary_erosion', binary_erosion)

@norecursion
def binary_dilation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 373)
    None_124654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 37), 'None')
    int_124655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 54), 'int')
    # Getting the type of 'None' (line 373)
    None_124656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 62), 'None')
    # Getting the type of 'None' (line 374)
    None_124657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'None')
    int_124658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 34), 'int')
    int_124659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 44), 'int')
    # Getting the type of 'False' (line 374)
    False_124660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 59), 'False')
    defaults = [None_124654, int_124655, None_124656, None_124657, int_124658, int_124659, False_124660]
    # Create a new context for function 'binary_dilation'
    module_type_store = module_type_store.open_function_context('binary_dilation', 373, 0, False)
    
    # Passed parameters checking function
    binary_dilation.stypy_localization = localization
    binary_dilation.stypy_type_of_self = None
    binary_dilation.stypy_type_store = module_type_store
    binary_dilation.stypy_function_name = 'binary_dilation'
    binary_dilation.stypy_param_names_list = ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'brute_force']
    binary_dilation.stypy_varargs_param_name = None
    binary_dilation.stypy_kwargs_param_name = None
    binary_dilation.stypy_call_defaults = defaults
    binary_dilation.stypy_call_varargs = varargs
    binary_dilation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binary_dilation', ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'brute_force'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binary_dilation', localization, ['input', 'structure', 'iterations', 'mask', 'output', 'border_value', 'origin', 'brute_force'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binary_dilation(...)' code ##################

    str_124661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, (-1)), 'str', '\n    Multi-dimensional binary dilation with the given structuring element.\n\n    Parameters\n    ----------\n    input : array_like\n        Binary array_like to be dilated. Non-zero (True) elements form\n        the subset to be dilated.\n    structure : array_like, optional\n        Structuring element used for the dilation. Non-zero elements are\n        considered True. If no structuring element is provided an element\n        is generated with a square connectivity equal to one.\n    iterations : {int, float}, optional\n        The dilation is repeated `iterations` times (one, by default).\n        If iterations is less than 1, the dilation is repeated until the\n        result does not change anymore.\n    mask : array_like, optional\n        If a mask is given, only those elements with a True value at\n        the corresponding mask element are modified at each iteration.\n    output : ndarray, optional\n        Array of the same shape as input, into which the output is placed.\n        By default, a new array is created.\n    origin : int or tuple of ints, optional\n        Placement of the filter, by default 0.\n    border_value : int (cast to 0 or 1), optional\n        Value at the border in the output array.\n\n    Returns\n    -------\n    binary_dilation : ndarray of bools\n        Dilation of the input by the structuring element.\n\n    See also\n    --------\n    grey_dilation, binary_erosion, binary_closing, binary_opening,\n    generate_binary_structure\n\n    Notes\n    -----\n    Dilation [1]_ is a mathematical morphology operation [2]_ that uses a\n    structuring element for expanding the shapes in an image. The binary\n    dilation of an image by a structuring element is the locus of the points\n    covered by the structuring element, when its center lies within the\n    non-zero points of the image.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Dilation_%28morphology%29\n    .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((5, 5))\n    >>> a[2, 2] = 1\n    >>> a\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.]])\n    >>> ndimage.binary_dilation(a)\n    array([[False, False, False, False, False],\n           [False, False,  True, False, False],\n           [False,  True,  True,  True, False],\n           [False, False,  True, False, False],\n           [False, False, False, False, False]], dtype=bool)\n    >>> ndimage.binary_dilation(a).astype(a.dtype)\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.]])\n    >>> # 3x3 structuring element with connectivity 1, used by default\n    >>> struct1 = ndimage.generate_binary_structure(2, 1)\n    >>> struct1\n    array([[False,  True, False],\n           [ True,  True,  True],\n           [False,  True, False]], dtype=bool)\n    >>> # 3x3 structuring element with connectivity 2\n    >>> struct2 = ndimage.generate_binary_structure(2, 2)\n    >>> struct2\n    array([[ True,  True,  True],\n           [ True,  True,  True],\n           [ True,  True,  True]], dtype=bool)\n    >>> ndimage.binary_dilation(a, structure=struct1).astype(a.dtype)\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.]])\n    >>> ndimage.binary_dilation(a, structure=struct2).astype(a.dtype)\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.]])\n    >>> ndimage.binary_dilation(a, structure=struct1,\\\n    ... iterations=2).astype(a.dtype)\n    array([[ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 1.,  1.,  1.,  1.,  1.],\n           [ 0.,  1.,  1.,  1.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.]])\n\n    ')
    
    # Assigning a Call to a Name (line 481):
    
    # Assigning a Call to a Name (line 481):
    
    # Call to asarray(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'input' (line 481)
    input_124664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 26), 'input', False)
    # Processing the call keyword arguments (line 481)
    kwargs_124665 = {}
    # Getting the type of 'numpy' (line 481)
    numpy_124662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 481)
    asarray_124663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), numpy_124662, 'asarray')
    # Calling asarray(args, kwargs) (line 481)
    asarray_call_result_124666 = invoke(stypy.reporting.localization.Localization(__file__, 481, 12), asarray_124663, *[input_124664], **kwargs_124665)
    
    # Assigning a type to the variable 'input' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'input', asarray_call_result_124666)
    
    # Type idiom detected: calculating its left and rigth part (line 482)
    # Getting the type of 'structure' (line 482)
    structure_124667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 7), 'structure')
    # Getting the type of 'None' (line 482)
    None_124668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 20), 'None')
    
    (may_be_124669, more_types_in_union_124670) = may_be_none(structure_124667, None_124668)

    if may_be_124669:

        if more_types_in_union_124670:
            # Runtime conditional SSA (line 482)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Call to generate_binary_structure(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of 'input' (line 483)
        input_124672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 46), 'input', False)
        # Obtaining the member 'ndim' of a type (line 483)
        ndim_124673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 46), input_124672, 'ndim')
        int_124674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 58), 'int')
        # Processing the call keyword arguments (line 483)
        kwargs_124675 = {}
        # Getting the type of 'generate_binary_structure' (line 483)
        generate_binary_structure_124671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 20), 'generate_binary_structure', False)
        # Calling generate_binary_structure(args, kwargs) (line 483)
        generate_binary_structure_call_result_124676 = invoke(stypy.reporting.localization.Localization(__file__, 483, 20), generate_binary_structure_124671, *[ndim_124673, int_124674], **kwargs_124675)
        
        # Assigning a type to the variable 'structure' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'structure', generate_binary_structure_call_result_124676)

        if more_types_in_union_124670:
            # SSA join for if statement (line 482)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 484):
    
    # Assigning a Call to a Name (line 484):
    
    # Call to _normalize_sequence(...): (line 484)
    # Processing the call arguments (line 484)
    # Getting the type of 'origin' (line 484)
    origin_124679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 45), 'origin', False)
    # Getting the type of 'input' (line 484)
    input_124680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 53), 'input', False)
    # Obtaining the member 'ndim' of a type (line 484)
    ndim_124681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 53), input_124680, 'ndim')
    # Processing the call keyword arguments (line 484)
    kwargs_124682 = {}
    # Getting the type of '_ni_support' (line 484)
    _ni_support_124677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 13), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 484)
    _normalize_sequence_124678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 13), _ni_support_124677, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 484)
    _normalize_sequence_call_result_124683 = invoke(stypy.reporting.localization.Localization(__file__, 484, 13), _normalize_sequence_124678, *[origin_124679, ndim_124681], **kwargs_124682)
    
    # Assigning a type to the variable 'origin' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'origin', _normalize_sequence_call_result_124683)
    
    # Assigning a Call to a Name (line 485):
    
    # Assigning a Call to a Name (line 485):
    
    # Call to asarray(...): (line 485)
    # Processing the call arguments (line 485)
    # Getting the type of 'structure' (line 485)
    structure_124686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 30), 'structure', False)
    # Processing the call keyword arguments (line 485)
    kwargs_124687 = {}
    # Getting the type of 'numpy' (line 485)
    numpy_124684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 485)
    asarray_124685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 16), numpy_124684, 'asarray')
    # Calling asarray(args, kwargs) (line 485)
    asarray_call_result_124688 = invoke(stypy.reporting.localization.Localization(__file__, 485, 16), asarray_124685, *[structure_124686], **kwargs_124687)
    
    # Assigning a type to the variable 'structure' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'structure', asarray_call_result_124688)
    
    # Assigning a Subscript to a Name (line 486):
    
    # Assigning a Subscript to a Name (line 486):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 486)
    # Processing the call arguments (line 486)
    
    # Obtaining an instance of the builtin type 'list' (line 486)
    list_124690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 486)
    # Adding element type (line 486)
    
    # Call to slice(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'None' (line 486)
    None_124692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 39), 'None', False)
    # Getting the type of 'None' (line 486)
    None_124693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 45), 'None', False)
    int_124694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 51), 'int')
    # Processing the call keyword arguments (line 486)
    kwargs_124695 = {}
    # Getting the type of 'slice' (line 486)
    slice_124691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 33), 'slice', False)
    # Calling slice(args, kwargs) (line 486)
    slice_call_result_124696 = invoke(stypy.reporting.localization.Localization(__file__, 486, 33), slice_124691, *[None_124692, None_124693, int_124694], **kwargs_124695)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 32), list_124690, slice_call_result_124696)
    
    # Getting the type of 'structure' (line 487)
    structure_124697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 32), 'structure', False)
    # Obtaining the member 'ndim' of a type (line 487)
    ndim_124698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 32), structure_124697, 'ndim')
    # Applying the binary operator '*' (line 486)
    result_mul_124699 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 32), '*', list_124690, ndim_124698)
    
    # Processing the call keyword arguments (line 486)
    kwargs_124700 = {}
    # Getting the type of 'tuple' (line 486)
    tuple_124689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 26), 'tuple', False)
    # Calling tuple(args, kwargs) (line 486)
    tuple_call_result_124701 = invoke(stypy.reporting.localization.Localization(__file__, 486, 26), tuple_124689, *[result_mul_124699], **kwargs_124700)
    
    # Getting the type of 'structure' (line 486)
    structure_124702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 16), 'structure')
    # Obtaining the member '__getitem__' of a type (line 486)
    getitem___124703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 16), structure_124702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 486)
    subscript_call_result_124704 = invoke(stypy.reporting.localization.Localization(__file__, 486, 16), getitem___124703, tuple_call_result_124701)
    
    # Assigning a type to the variable 'structure' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'structure', subscript_call_result_124704)
    
    
    # Call to range(...): (line 488)
    # Processing the call arguments (line 488)
    
    # Call to len(...): (line 488)
    # Processing the call arguments (line 488)
    # Getting the type of 'origin' (line 488)
    origin_124707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 24), 'origin', False)
    # Processing the call keyword arguments (line 488)
    kwargs_124708 = {}
    # Getting the type of 'len' (line 488)
    len_124706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 20), 'len', False)
    # Calling len(args, kwargs) (line 488)
    len_call_result_124709 = invoke(stypy.reporting.localization.Localization(__file__, 488, 20), len_124706, *[origin_124707], **kwargs_124708)
    
    # Processing the call keyword arguments (line 488)
    kwargs_124710 = {}
    # Getting the type of 'range' (line 488)
    range_124705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 14), 'range', False)
    # Calling range(args, kwargs) (line 488)
    range_call_result_124711 = invoke(stypy.reporting.localization.Localization(__file__, 488, 14), range_124705, *[len_call_result_124709], **kwargs_124710)
    
    # Testing the type of a for loop iterable (line 488)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 488, 4), range_call_result_124711)
    # Getting the type of the for loop variable (line 488)
    for_loop_var_124712 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 488, 4), range_call_result_124711)
    # Assigning a type to the variable 'ii' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'ii', for_loop_var_124712)
    # SSA begins for a for statement (line 488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a UnaryOp to a Subscript (line 489):
    
    # Assigning a UnaryOp to a Subscript (line 489):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 489)
    ii_124713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 29), 'ii')
    # Getting the type of 'origin' (line 489)
    origin_124714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 22), 'origin')
    # Obtaining the member '__getitem__' of a type (line 489)
    getitem___124715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 22), origin_124714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 489)
    subscript_call_result_124716 = invoke(stypy.reporting.localization.Localization(__file__, 489, 22), getitem___124715, ii_124713)
    
    # Applying the 'usub' unary operator (line 489)
    result___neg___124717 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 21), 'usub', subscript_call_result_124716)
    
    # Getting the type of 'origin' (line 489)
    origin_124718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'origin')
    # Getting the type of 'ii' (line 489)
    ii_124719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'ii')
    # Storing an element on a container (line 489)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 8), origin_124718, (ii_124719, result___neg___124717))
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 490)
    ii_124720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 31), 'ii')
    # Getting the type of 'structure' (line 490)
    structure_124721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'structure')
    # Obtaining the member 'shape' of a type (line 490)
    shape_124722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 15), structure_124721, 'shape')
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___124723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 15), shape_124722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_124724 = invoke(stypy.reporting.localization.Localization(__file__, 490, 15), getitem___124723, ii_124720)
    
    int_124725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 37), 'int')
    # Applying the binary operator '&' (line 490)
    result_and__124726 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 15), '&', subscript_call_result_124724, int_124725)
    
    # Applying the 'not' unary operator (line 490)
    result_not__124727 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), 'not', result_and__124726)
    
    # Testing the type of an if condition (line 490)
    if_condition_124728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_not__124727)
    # Assigning a type to the variable 'if_condition_124728' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_124728', if_condition_124728)
    # SSA begins for if statement (line 490)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'origin' (line 491)
    origin_124729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'origin')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 491)
    ii_124730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 19), 'ii')
    # Getting the type of 'origin' (line 491)
    origin_124731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'origin')
    # Obtaining the member '__getitem__' of a type (line 491)
    getitem___124732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 12), origin_124731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 491)
    subscript_call_result_124733 = invoke(stypy.reporting.localization.Localization(__file__, 491, 12), getitem___124732, ii_124730)
    
    int_124734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 26), 'int')
    # Applying the binary operator '-=' (line 491)
    result_isub_124735 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 12), '-=', subscript_call_result_124733, int_124734)
    # Getting the type of 'origin' (line 491)
    origin_124736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'origin')
    # Getting the type of 'ii' (line 491)
    ii_124737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 19), 'ii')
    # Storing an element on a container (line 491)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 12), origin_124736, (ii_124737, result_isub_124735))
    
    # SSA join for if statement (line 490)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _binary_erosion(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'input' (line 493)
    input_124739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 27), 'input', False)
    # Getting the type of 'structure' (line 493)
    structure_124740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 34), 'structure', False)
    # Getting the type of 'iterations' (line 493)
    iterations_124741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 45), 'iterations', False)
    # Getting the type of 'mask' (line 493)
    mask_124742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 57), 'mask', False)
    # Getting the type of 'output' (line 494)
    output_124743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 27), 'output', False)
    # Getting the type of 'border_value' (line 494)
    border_value_124744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 35), 'border_value', False)
    # Getting the type of 'origin' (line 494)
    origin_124745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 49), 'origin', False)
    int_124746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 57), 'int')
    # Getting the type of 'brute_force' (line 494)
    brute_force_124747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 60), 'brute_force', False)
    # Processing the call keyword arguments (line 493)
    kwargs_124748 = {}
    # Getting the type of '_binary_erosion' (line 493)
    _binary_erosion_124738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), '_binary_erosion', False)
    # Calling _binary_erosion(args, kwargs) (line 493)
    _binary_erosion_call_result_124749 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), _binary_erosion_124738, *[input_124739, structure_124740, iterations_124741, mask_124742, output_124743, border_value_124744, origin_124745, int_124746, brute_force_124747], **kwargs_124748)
    
    # Assigning a type to the variable 'stypy_return_type' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'stypy_return_type', _binary_erosion_call_result_124749)
    
    # ################# End of 'binary_dilation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binary_dilation' in the type store
    # Getting the type of 'stypy_return_type' (line 373)
    stypy_return_type_124750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124750)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binary_dilation'
    return stypy_return_type_124750

# Assigning a type to the variable 'binary_dilation' (line 373)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), 'binary_dilation', binary_dilation)

@norecursion
def binary_opening(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 497)
    None_124751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 36), 'None')
    int_124752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 53), 'int')
    # Getting the type of 'None' (line 497)
    None_124753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 63), 'None')
    int_124754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 26), 'int')
    defaults = [None_124751, int_124752, None_124753, int_124754]
    # Create a new context for function 'binary_opening'
    module_type_store = module_type_store.open_function_context('binary_opening', 497, 0, False)
    
    # Passed parameters checking function
    binary_opening.stypy_localization = localization
    binary_opening.stypy_type_of_self = None
    binary_opening.stypy_type_store = module_type_store
    binary_opening.stypy_function_name = 'binary_opening'
    binary_opening.stypy_param_names_list = ['input', 'structure', 'iterations', 'output', 'origin']
    binary_opening.stypy_varargs_param_name = None
    binary_opening.stypy_kwargs_param_name = None
    binary_opening.stypy_call_defaults = defaults
    binary_opening.stypy_call_varargs = varargs
    binary_opening.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binary_opening', ['input', 'structure', 'iterations', 'output', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binary_opening', localization, ['input', 'structure', 'iterations', 'output', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binary_opening(...)' code ##################

    str_124755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, (-1)), 'str', '\n    Multi-dimensional binary opening with the given structuring element.\n\n    The *opening* of an input image by a structuring element is the\n    *dilation* of the *erosion* of the image by the structuring element.\n\n    Parameters\n    ----------\n    input : array_like\n        Binary array_like to be opened. Non-zero (True) elements form\n        the subset to be opened.\n    structure : array_like, optional\n        Structuring element used for the opening. Non-zero elements are\n        considered True. If no structuring element is provided an element\n        is generated with a square connectivity equal to one (i.e., only\n        nearest neighbors are connected to the center, diagonally-connected\n        elements are not considered neighbors).\n    iterations : {int, float}, optional\n        The erosion step of the opening, then the dilation step are each\n        repeated `iterations` times (one, by default). If `iterations` is\n        less than 1, each operation is repeated until the result does\n        not change anymore.\n    output : ndarray, optional\n        Array of the same shape as input, into which the output is placed.\n        By default, a new array is created.\n    origin : int or tuple of ints, optional\n        Placement of the filter, by default 0.\n\n    Returns\n    -------\n    binary_opening : ndarray of bools\n        Opening of the input by the structuring element.\n\n    See also\n    --------\n    grey_opening, binary_closing, binary_erosion, binary_dilation,\n    generate_binary_structure\n\n    Notes\n    -----\n    *Opening* [1]_ is a mathematical morphology operation [2]_ that\n    consists in the succession of an erosion and a dilation of the\n    input with the same structuring element. Opening therefore removes\n    objects smaller than the structuring element.\n\n    Together with *closing* (`binary_closing`), opening can be used for\n    noise removal.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Opening_%28morphology%29\n    .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((5,5), dtype=int)\n    >>> a[1:4, 1:4] = 1; a[4, 4] = 1\n    >>> a\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 1]])\n    >>> # Opening removes small objects\n    >>> ndimage.binary_opening(a, structure=np.ones((3,3))).astype(int)\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0]])\n    >>> # Opening can also smooth corners\n    >>> ndimage.binary_opening(a).astype(int)\n    array([[0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0]])\n    >>> # Opening is the dilation of the erosion of the input\n    >>> ndimage.binary_erosion(a).astype(int)\n    array([[0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0]])\n    >>> ndimage.binary_dilation(ndimage.binary_erosion(a)).astype(int)\n    array([[0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0]])\n\n    ')
    
    # Assigning a Call to a Name (line 592):
    
    # Assigning a Call to a Name (line 592):
    
    # Call to asarray(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'input' (line 592)
    input_124758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 26), 'input', False)
    # Processing the call keyword arguments (line 592)
    kwargs_124759 = {}
    # Getting the type of 'numpy' (line 592)
    numpy_124756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 592)
    asarray_124757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 12), numpy_124756, 'asarray')
    # Calling asarray(args, kwargs) (line 592)
    asarray_call_result_124760 = invoke(stypy.reporting.localization.Localization(__file__, 592, 12), asarray_124757, *[input_124758], **kwargs_124759)
    
    # Assigning a type to the variable 'input' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'input', asarray_call_result_124760)
    
    # Type idiom detected: calculating its left and rigth part (line 593)
    # Getting the type of 'structure' (line 593)
    structure_124761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 7), 'structure')
    # Getting the type of 'None' (line 593)
    None_124762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 20), 'None')
    
    (may_be_124763, more_types_in_union_124764) = may_be_none(structure_124761, None_124762)

    if may_be_124763:

        if more_types_in_union_124764:
            # Runtime conditional SSA (line 593)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 594):
        
        # Assigning a Attribute to a Name (line 594):
        # Getting the type of 'input' (line 594)
        input_124765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 15), 'input')
        # Obtaining the member 'ndim' of a type (line 594)
        ndim_124766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 15), input_124765, 'ndim')
        # Assigning a type to the variable 'rank' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'rank', ndim_124766)
        
        # Assigning a Call to a Name (line 595):
        
        # Assigning a Call to a Name (line 595):
        
        # Call to generate_binary_structure(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'rank' (line 595)
        rank_124768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 46), 'rank', False)
        int_124769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 52), 'int')
        # Processing the call keyword arguments (line 595)
        kwargs_124770 = {}
        # Getting the type of 'generate_binary_structure' (line 595)
        generate_binary_structure_124767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 20), 'generate_binary_structure', False)
        # Calling generate_binary_structure(args, kwargs) (line 595)
        generate_binary_structure_call_result_124771 = invoke(stypy.reporting.localization.Localization(__file__, 595, 20), generate_binary_structure_124767, *[rank_124768, int_124769], **kwargs_124770)
        
        # Assigning a type to the variable 'structure' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'structure', generate_binary_structure_call_result_124771)

        if more_types_in_union_124764:
            # SSA join for if statement (line 593)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 597):
    
    # Assigning a Call to a Name (line 597):
    
    # Call to binary_erosion(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'input' (line 597)
    input_124773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 25), 'input', False)
    # Getting the type of 'structure' (line 597)
    structure_124774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 32), 'structure', False)
    # Getting the type of 'iterations' (line 597)
    iterations_124775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 43), 'iterations', False)
    # Getting the type of 'None' (line 597)
    None_124776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 55), 'None', False)
    # Getting the type of 'None' (line 597)
    None_124777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 61), 'None', False)
    int_124778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 67), 'int')
    # Getting the type of 'origin' (line 598)
    origin_124779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 25), 'origin', False)
    # Processing the call keyword arguments (line 597)
    kwargs_124780 = {}
    # Getting the type of 'binary_erosion' (line 597)
    binary_erosion_124772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 10), 'binary_erosion', False)
    # Calling binary_erosion(args, kwargs) (line 597)
    binary_erosion_call_result_124781 = invoke(stypy.reporting.localization.Localization(__file__, 597, 10), binary_erosion_124772, *[input_124773, structure_124774, iterations_124775, None_124776, None_124777, int_124778, origin_124779], **kwargs_124780)
    
    # Assigning a type to the variable 'tmp' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'tmp', binary_erosion_call_result_124781)
    
    # Call to binary_dilation(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 'tmp' (line 599)
    tmp_124783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 27), 'tmp', False)
    # Getting the type of 'structure' (line 599)
    structure_124784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 32), 'structure', False)
    # Getting the type of 'iterations' (line 599)
    iterations_124785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 43), 'iterations', False)
    # Getting the type of 'None' (line 599)
    None_124786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 55), 'None', False)
    # Getting the type of 'output' (line 599)
    output_124787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 61), 'output', False)
    int_124788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 69), 'int')
    # Getting the type of 'origin' (line 600)
    origin_124789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 27), 'origin', False)
    # Processing the call keyword arguments (line 599)
    kwargs_124790 = {}
    # Getting the type of 'binary_dilation' (line 599)
    binary_dilation_124782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 11), 'binary_dilation', False)
    # Calling binary_dilation(args, kwargs) (line 599)
    binary_dilation_call_result_124791 = invoke(stypy.reporting.localization.Localization(__file__, 599, 11), binary_dilation_124782, *[tmp_124783, structure_124784, iterations_124785, None_124786, output_124787, int_124788, origin_124789], **kwargs_124790)
    
    # Assigning a type to the variable 'stypy_return_type' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'stypy_return_type', binary_dilation_call_result_124791)
    
    # ################# End of 'binary_opening(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binary_opening' in the type store
    # Getting the type of 'stypy_return_type' (line 497)
    stypy_return_type_124792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124792)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binary_opening'
    return stypy_return_type_124792

# Assigning a type to the variable 'binary_opening' (line 497)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 0), 'binary_opening', binary_opening)

@norecursion
def binary_closing(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 603)
    None_124793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 36), 'None')
    int_124794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 53), 'int')
    # Getting the type of 'None' (line 603)
    None_124795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 63), 'None')
    int_124796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 26), 'int')
    defaults = [None_124793, int_124794, None_124795, int_124796]
    # Create a new context for function 'binary_closing'
    module_type_store = module_type_store.open_function_context('binary_closing', 603, 0, False)
    
    # Passed parameters checking function
    binary_closing.stypy_localization = localization
    binary_closing.stypy_type_of_self = None
    binary_closing.stypy_type_store = module_type_store
    binary_closing.stypy_function_name = 'binary_closing'
    binary_closing.stypy_param_names_list = ['input', 'structure', 'iterations', 'output', 'origin']
    binary_closing.stypy_varargs_param_name = None
    binary_closing.stypy_kwargs_param_name = None
    binary_closing.stypy_call_defaults = defaults
    binary_closing.stypy_call_varargs = varargs
    binary_closing.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binary_closing', ['input', 'structure', 'iterations', 'output', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binary_closing', localization, ['input', 'structure', 'iterations', 'output', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binary_closing(...)' code ##################

    str_124797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, (-1)), 'str', '\n    Multi-dimensional binary closing with the given structuring element.\n\n    The *closing* of an input image by a structuring element is the\n    *erosion* of the *dilation* of the image by the structuring element.\n\n    Parameters\n    ----------\n    input : array_like\n        Binary array_like to be closed. Non-zero (True) elements form\n        the subset to be closed.\n    structure : array_like, optional\n        Structuring element used for the closing. Non-zero elements are\n        considered True. If no structuring element is provided an element\n        is generated with a square connectivity equal to one (i.e., only\n        nearest neighbors are connected to the center, diagonally-connected\n        elements are not considered neighbors).\n    iterations : {int, float}, optional\n        The dilation step of the closing, then the erosion step are each\n        repeated `iterations` times (one, by default). If iterations is\n        less than 1, each operations is repeated until the result does\n        not change anymore.\n    output : ndarray, optional\n        Array of the same shape as input, into which the output is placed.\n        By default, a new array is created.\n    origin : int or tuple of ints, optional\n        Placement of the filter, by default 0.\n\n    Returns\n    -------\n    binary_closing : ndarray of bools\n        Closing of the input by the structuring element.\n\n    See also\n    --------\n    grey_closing, binary_opening, binary_dilation, binary_erosion,\n    generate_binary_structure\n\n    Notes\n    -----\n    *Closing* [1]_ is a mathematical morphology operation [2]_ that\n    consists in the succession of a dilation and an erosion of the\n    input with the same structuring element. Closing therefore fills\n    holes smaller than the structuring element.\n\n    Together with *opening* (`binary_opening`), closing can be used for\n    noise removal.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Closing_%28morphology%29\n    .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((5,5), dtype=int)\n    >>> a[1:-1, 1:-1] = 1; a[2,2] = 0\n    >>> a\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 0, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0]])\n    >>> # Closing removes small holes\n    >>> ndimage.binary_closing(a).astype(int)\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0]])\n    >>> # Closing is the erosion of the dilation of the input\n    >>> ndimage.binary_dilation(a).astype(int)\n    array([[0, 1, 1, 1, 0],\n           [1, 1, 1, 1, 1],\n           [1, 1, 1, 1, 1],\n           [1, 1, 1, 1, 1],\n           [0, 1, 1, 1, 0]])\n    >>> ndimage.binary_erosion(ndimage.binary_dilation(a)).astype(int)\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0]])\n\n\n    >>> a = np.zeros((7,7), dtype=int)\n    >>> a[1:6, 2:5] = 1; a[1:3,3] = 0\n    >>> a\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 1, 0, 0],\n           [0, 0, 1, 0, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> # In addition to removing holes, closing can also\n    >>> # coarsen boundaries with fine hollows.\n    >>> ndimage.binary_closing(a).astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> ndimage.binary_closing(a, structure=np.ones((2,2))).astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n\n    ')
    
    # Assigning a Call to a Name (line 721):
    
    # Assigning a Call to a Name (line 721):
    
    # Call to asarray(...): (line 721)
    # Processing the call arguments (line 721)
    # Getting the type of 'input' (line 721)
    input_124800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 26), 'input', False)
    # Processing the call keyword arguments (line 721)
    kwargs_124801 = {}
    # Getting the type of 'numpy' (line 721)
    numpy_124798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 721)
    asarray_124799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 12), numpy_124798, 'asarray')
    # Calling asarray(args, kwargs) (line 721)
    asarray_call_result_124802 = invoke(stypy.reporting.localization.Localization(__file__, 721, 12), asarray_124799, *[input_124800], **kwargs_124801)
    
    # Assigning a type to the variable 'input' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'input', asarray_call_result_124802)
    
    # Type idiom detected: calculating its left and rigth part (line 722)
    # Getting the type of 'structure' (line 722)
    structure_124803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 7), 'structure')
    # Getting the type of 'None' (line 722)
    None_124804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 20), 'None')
    
    (may_be_124805, more_types_in_union_124806) = may_be_none(structure_124803, None_124804)

    if may_be_124805:

        if more_types_in_union_124806:
            # Runtime conditional SSA (line 722)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 723):
        
        # Assigning a Attribute to a Name (line 723):
        # Getting the type of 'input' (line 723)
        input_124807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 15), 'input')
        # Obtaining the member 'ndim' of a type (line 723)
        ndim_124808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 15), input_124807, 'ndim')
        # Assigning a type to the variable 'rank' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'rank', ndim_124808)
        
        # Assigning a Call to a Name (line 724):
        
        # Assigning a Call to a Name (line 724):
        
        # Call to generate_binary_structure(...): (line 724)
        # Processing the call arguments (line 724)
        # Getting the type of 'rank' (line 724)
        rank_124810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 46), 'rank', False)
        int_124811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 52), 'int')
        # Processing the call keyword arguments (line 724)
        kwargs_124812 = {}
        # Getting the type of 'generate_binary_structure' (line 724)
        generate_binary_structure_124809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 20), 'generate_binary_structure', False)
        # Calling generate_binary_structure(args, kwargs) (line 724)
        generate_binary_structure_call_result_124813 = invoke(stypy.reporting.localization.Localization(__file__, 724, 20), generate_binary_structure_124809, *[rank_124810, int_124811], **kwargs_124812)
        
        # Assigning a type to the variable 'structure' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'structure', generate_binary_structure_call_result_124813)

        if more_types_in_union_124806:
            # SSA join for if statement (line 722)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 726):
    
    # Assigning a Call to a Name (line 726):
    
    # Call to binary_dilation(...): (line 726)
    # Processing the call arguments (line 726)
    # Getting the type of 'input' (line 726)
    input_124815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 26), 'input', False)
    # Getting the type of 'structure' (line 726)
    structure_124816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 33), 'structure', False)
    # Getting the type of 'iterations' (line 726)
    iterations_124817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 44), 'iterations', False)
    # Getting the type of 'None' (line 726)
    None_124818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 56), 'None', False)
    # Getting the type of 'None' (line 726)
    None_124819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 62), 'None', False)
    int_124820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 68), 'int')
    # Getting the type of 'origin' (line 727)
    origin_124821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 26), 'origin', False)
    # Processing the call keyword arguments (line 726)
    kwargs_124822 = {}
    # Getting the type of 'binary_dilation' (line 726)
    binary_dilation_124814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 10), 'binary_dilation', False)
    # Calling binary_dilation(args, kwargs) (line 726)
    binary_dilation_call_result_124823 = invoke(stypy.reporting.localization.Localization(__file__, 726, 10), binary_dilation_124814, *[input_124815, structure_124816, iterations_124817, None_124818, None_124819, int_124820, origin_124821], **kwargs_124822)
    
    # Assigning a type to the variable 'tmp' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'tmp', binary_dilation_call_result_124823)
    
    # Call to binary_erosion(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'tmp' (line 728)
    tmp_124825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 26), 'tmp', False)
    # Getting the type of 'structure' (line 728)
    structure_124826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 31), 'structure', False)
    # Getting the type of 'iterations' (line 728)
    iterations_124827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 42), 'iterations', False)
    # Getting the type of 'None' (line 728)
    None_124828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 54), 'None', False)
    # Getting the type of 'output' (line 728)
    output_124829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 60), 'output', False)
    int_124830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 68), 'int')
    # Getting the type of 'origin' (line 729)
    origin_124831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 26), 'origin', False)
    # Processing the call keyword arguments (line 728)
    kwargs_124832 = {}
    # Getting the type of 'binary_erosion' (line 728)
    binary_erosion_124824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 11), 'binary_erosion', False)
    # Calling binary_erosion(args, kwargs) (line 728)
    binary_erosion_call_result_124833 = invoke(stypy.reporting.localization.Localization(__file__, 728, 11), binary_erosion_124824, *[tmp_124825, structure_124826, iterations_124827, None_124828, output_124829, int_124830, origin_124831], **kwargs_124832)
    
    # Assigning a type to the variable 'stypy_return_type' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'stypy_return_type', binary_erosion_call_result_124833)
    
    # ################# End of 'binary_closing(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binary_closing' in the type store
    # Getting the type of 'stypy_return_type' (line 603)
    stypy_return_type_124834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124834)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binary_closing'
    return stypy_return_type_124834

# Assigning a type to the variable 'binary_closing' (line 603)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 0), 'binary_closing', binary_closing)

@norecursion
def binary_hit_or_miss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 732)
    None_124835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 41), 'None')
    # Getting the type of 'None' (line 732)
    None_124836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 58), 'None')
    # Getting the type of 'None' (line 733)
    None_124837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 30), 'None')
    int_124838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 44), 'int')
    # Getting the type of 'None' (line 733)
    None_124839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 55), 'None')
    defaults = [None_124835, None_124836, None_124837, int_124838, None_124839]
    # Create a new context for function 'binary_hit_or_miss'
    module_type_store = module_type_store.open_function_context('binary_hit_or_miss', 732, 0, False)
    
    # Passed parameters checking function
    binary_hit_or_miss.stypy_localization = localization
    binary_hit_or_miss.stypy_type_of_self = None
    binary_hit_or_miss.stypy_type_store = module_type_store
    binary_hit_or_miss.stypy_function_name = 'binary_hit_or_miss'
    binary_hit_or_miss.stypy_param_names_list = ['input', 'structure1', 'structure2', 'output', 'origin1', 'origin2']
    binary_hit_or_miss.stypy_varargs_param_name = None
    binary_hit_or_miss.stypy_kwargs_param_name = None
    binary_hit_or_miss.stypy_call_defaults = defaults
    binary_hit_or_miss.stypy_call_varargs = varargs
    binary_hit_or_miss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binary_hit_or_miss', ['input', 'structure1', 'structure2', 'output', 'origin1', 'origin2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binary_hit_or_miss', localization, ['input', 'structure1', 'structure2', 'output', 'origin1', 'origin2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binary_hit_or_miss(...)' code ##################

    str_124840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, (-1)), 'str', '\n    Multi-dimensional binary hit-or-miss transform.\n\n    The hit-or-miss transform finds the locations of a given pattern\n    inside the input image.\n\n    Parameters\n    ----------\n    input : array_like (cast to booleans)\n        Binary image where a pattern is to be detected.\n    structure1 : array_like (cast to booleans), optional\n        Part of the structuring element to be fitted to the foreground\n        (non-zero elements) of `input`. If no value is provided, a\n        structure of square connectivity 1 is chosen.\n    structure2 : array_like (cast to booleans), optional\n        Second part of the structuring element that has to miss completely\n        the foreground. If no value is provided, the complementary of\n        `structure1` is taken.\n    output : ndarray, optional\n        Array of the same shape as input, into which the output is placed.\n        By default, a new array is created.\n    origin1 : int or tuple of ints, optional\n        Placement of the first part of the structuring element `structure1`,\n        by default 0 for a centered structure.\n    origin2 : int or tuple of ints, optional\n        Placement of the second part of the structuring element `structure2`,\n        by default 0 for a centered structure. If a value is provided for\n        `origin1` and not for `origin2`, then `origin2` is set to `origin1`.\n\n    Returns\n    -------\n    binary_hit_or_miss : ndarray\n        Hit-or-miss transform of `input` with the given structuring\n        element (`structure1`, `structure2`).\n\n    See also\n    --------\n    ndimage.morphology, binary_erosion\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Hit-or-miss_transform\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((7,7), dtype=int)\n    >>> a[1, 1] = 1; a[2:4, 2:4] = 1; a[4:6, 4:6] = 1\n    >>> a\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 0, 0, 0],\n           [0, 0, 1, 1, 0, 0, 0],\n           [0, 0, 0, 0, 1, 1, 0],\n           [0, 0, 0, 0, 1, 1, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> structure1 = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])\n    >>> structure1\n    array([[1, 0, 0],\n           [0, 1, 1],\n           [0, 1, 1]])\n    >>> # Find the matches of structure1 in the array a\n    >>> ndimage.binary_hit_or_miss(a, structure1=structure1).astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> # Change the origin of the filter\n    >>> # origin1=1 is equivalent to origin1=(1,1) here\n    >>> ndimage.binary_hit_or_miss(a, structure1=structure1,\\\n    ... origin1=1).astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 1, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n\n    ')
    
    # Assigning a Call to a Name (line 817):
    
    # Assigning a Call to a Name (line 817):
    
    # Call to asarray(...): (line 817)
    # Processing the call arguments (line 817)
    # Getting the type of 'input' (line 817)
    input_124843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 26), 'input', False)
    # Processing the call keyword arguments (line 817)
    kwargs_124844 = {}
    # Getting the type of 'numpy' (line 817)
    numpy_124841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 817)
    asarray_124842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 12), numpy_124841, 'asarray')
    # Calling asarray(args, kwargs) (line 817)
    asarray_call_result_124845 = invoke(stypy.reporting.localization.Localization(__file__, 817, 12), asarray_124842, *[input_124843], **kwargs_124844)
    
    # Assigning a type to the variable 'input' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'input', asarray_call_result_124845)
    
    # Type idiom detected: calculating its left and rigth part (line 818)
    # Getting the type of 'structure1' (line 818)
    structure1_124846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 7), 'structure1')
    # Getting the type of 'None' (line 818)
    None_124847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 21), 'None')
    
    (may_be_124848, more_types_in_union_124849) = may_be_none(structure1_124846, None_124847)

    if may_be_124848:

        if more_types_in_union_124849:
            # Runtime conditional SSA (line 818)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 819):
        
        # Assigning a Call to a Name (line 819):
        
        # Call to generate_binary_structure(...): (line 819)
        # Processing the call arguments (line 819)
        # Getting the type of 'input' (line 819)
        input_124851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 47), 'input', False)
        # Obtaining the member 'ndim' of a type (line 819)
        ndim_124852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 47), input_124851, 'ndim')
        int_124853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 59), 'int')
        # Processing the call keyword arguments (line 819)
        kwargs_124854 = {}
        # Getting the type of 'generate_binary_structure' (line 819)
        generate_binary_structure_124850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 21), 'generate_binary_structure', False)
        # Calling generate_binary_structure(args, kwargs) (line 819)
        generate_binary_structure_call_result_124855 = invoke(stypy.reporting.localization.Localization(__file__, 819, 21), generate_binary_structure_124850, *[ndim_124852, int_124853], **kwargs_124854)
        
        # Assigning a type to the variable 'structure1' (line 819)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 8), 'structure1', generate_binary_structure_call_result_124855)

        if more_types_in_union_124849:
            # SSA join for if statement (line 818)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 820)
    # Getting the type of 'structure2' (line 820)
    structure2_124856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 7), 'structure2')
    # Getting the type of 'None' (line 820)
    None_124857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 21), 'None')
    
    (may_be_124858, more_types_in_union_124859) = may_be_none(structure2_124856, None_124857)

    if may_be_124858:

        if more_types_in_union_124859:
            # Runtime conditional SSA (line 820)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 821):
        
        # Assigning a Call to a Name (line 821):
        
        # Call to logical_not(...): (line 821)
        # Processing the call arguments (line 821)
        # Getting the type of 'structure1' (line 821)
        structure1_124862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 39), 'structure1', False)
        # Processing the call keyword arguments (line 821)
        kwargs_124863 = {}
        # Getting the type of 'numpy' (line 821)
        numpy_124860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 21), 'numpy', False)
        # Obtaining the member 'logical_not' of a type (line 821)
        logical_not_124861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 21), numpy_124860, 'logical_not')
        # Calling logical_not(args, kwargs) (line 821)
        logical_not_call_result_124864 = invoke(stypy.reporting.localization.Localization(__file__, 821, 21), logical_not_124861, *[structure1_124862], **kwargs_124863)
        
        # Assigning a type to the variable 'structure2' (line 821)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'structure2', logical_not_call_result_124864)

        if more_types_in_union_124859:
            # SSA join for if statement (line 820)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 822):
    
    # Assigning a Call to a Name (line 822):
    
    # Call to _normalize_sequence(...): (line 822)
    # Processing the call arguments (line 822)
    # Getting the type of 'origin1' (line 822)
    origin1_124867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 46), 'origin1', False)
    # Getting the type of 'input' (line 822)
    input_124868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 55), 'input', False)
    # Obtaining the member 'ndim' of a type (line 822)
    ndim_124869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 55), input_124868, 'ndim')
    # Processing the call keyword arguments (line 822)
    kwargs_124870 = {}
    # Getting the type of '_ni_support' (line 822)
    _ni_support_124865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 14), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 822)
    _normalize_sequence_124866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 14), _ni_support_124865, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 822)
    _normalize_sequence_call_result_124871 = invoke(stypy.reporting.localization.Localization(__file__, 822, 14), _normalize_sequence_124866, *[origin1_124867, ndim_124869], **kwargs_124870)
    
    # Assigning a type to the variable 'origin1' (line 822)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 4), 'origin1', _normalize_sequence_call_result_124871)
    
    # Type idiom detected: calculating its left and rigth part (line 823)
    # Getting the type of 'origin2' (line 823)
    origin2_124872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 7), 'origin2')
    # Getting the type of 'None' (line 823)
    None_124873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 18), 'None')
    
    (may_be_124874, more_types_in_union_124875) = may_be_none(origin2_124872, None_124873)

    if may_be_124874:

        if more_types_in_union_124875:
            # Runtime conditional SSA (line 823)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 824):
        
        # Assigning a Name to a Name (line 824):
        # Getting the type of 'origin1' (line 824)
        origin1_124876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 18), 'origin1')
        # Assigning a type to the variable 'origin2' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 8), 'origin2', origin1_124876)

        if more_types_in_union_124875:
            # Runtime conditional SSA for else branch (line 823)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_124874) or more_types_in_union_124875):
        
        # Assigning a Call to a Name (line 826):
        
        # Assigning a Call to a Name (line 826):
        
        # Call to _normalize_sequence(...): (line 826)
        # Processing the call arguments (line 826)
        # Getting the type of 'origin2' (line 826)
        origin2_124879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 50), 'origin2', False)
        # Getting the type of 'input' (line 826)
        input_124880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 59), 'input', False)
        # Obtaining the member 'ndim' of a type (line 826)
        ndim_124881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 59), input_124880, 'ndim')
        # Processing the call keyword arguments (line 826)
        kwargs_124882 = {}
        # Getting the type of '_ni_support' (line 826)
        _ni_support_124877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 18), '_ni_support', False)
        # Obtaining the member '_normalize_sequence' of a type (line 826)
        _normalize_sequence_124878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 18), _ni_support_124877, '_normalize_sequence')
        # Calling _normalize_sequence(args, kwargs) (line 826)
        _normalize_sequence_call_result_124883 = invoke(stypy.reporting.localization.Localization(__file__, 826, 18), _normalize_sequence_124878, *[origin2_124879, ndim_124881], **kwargs_124882)
        
        # Assigning a type to the variable 'origin2' (line 826)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 8), 'origin2', _normalize_sequence_call_result_124883)

        if (may_be_124874 and more_types_in_union_124875):
            # SSA join for if statement (line 823)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 828):
    
    # Assigning a Call to a Name (line 828):
    
    # Call to _binary_erosion(...): (line 828)
    # Processing the call arguments (line 828)
    # Getting the type of 'input' (line 828)
    input_124885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 27), 'input', False)
    # Getting the type of 'structure1' (line 828)
    structure1_124886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 34), 'structure1', False)
    int_124887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 46), 'int')
    # Getting the type of 'None' (line 828)
    None_124888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 49), 'None', False)
    # Getting the type of 'None' (line 828)
    None_124889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 55), 'None', False)
    int_124890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 61), 'int')
    # Getting the type of 'origin1' (line 828)
    origin1_124891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 64), 'origin1', False)
    int_124892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 27), 'int')
    # Getting the type of 'False' (line 829)
    False_124893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 30), 'False', False)
    # Processing the call keyword arguments (line 828)
    kwargs_124894 = {}
    # Getting the type of '_binary_erosion' (line 828)
    _binary_erosion_124884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 11), '_binary_erosion', False)
    # Calling _binary_erosion(args, kwargs) (line 828)
    _binary_erosion_call_result_124895 = invoke(stypy.reporting.localization.Localization(__file__, 828, 11), _binary_erosion_124884, *[input_124885, structure1_124886, int_124887, None_124888, None_124889, int_124890, origin1_124891, int_124892, False_124893], **kwargs_124894)
    
    # Assigning a type to the variable 'tmp1' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 4), 'tmp1', _binary_erosion_call_result_124895)
    
    # Assigning a Call to a Name (line 830):
    
    # Assigning a Call to a Name (line 830):
    
    # Call to isinstance(...): (line 830)
    # Processing the call arguments (line 830)
    # Getting the type of 'output' (line 830)
    output_124897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 25), 'output', False)
    # Getting the type of 'numpy' (line 830)
    numpy_124898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 33), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 830)
    ndarray_124899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 33), numpy_124898, 'ndarray')
    # Processing the call keyword arguments (line 830)
    kwargs_124900 = {}
    # Getting the type of 'isinstance' (line 830)
    isinstance_124896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 14), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 830)
    isinstance_call_result_124901 = invoke(stypy.reporting.localization.Localization(__file__, 830, 14), isinstance_124896, *[output_124897, ndarray_124899], **kwargs_124900)
    
    # Assigning a type to the variable 'inplace' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'inplace', isinstance_call_result_124901)
    
    # Assigning a Call to a Name (line 831):
    
    # Assigning a Call to a Name (line 831):
    
    # Call to _binary_erosion(...): (line 831)
    # Processing the call arguments (line 831)
    # Getting the type of 'input' (line 831)
    input_124903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 29), 'input', False)
    # Getting the type of 'structure2' (line 831)
    structure2_124904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 36), 'structure2', False)
    int_124905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 48), 'int')
    # Getting the type of 'None' (line 831)
    None_124906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 51), 'None', False)
    # Getting the type of 'output' (line 831)
    output_124907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 57), 'output', False)
    int_124908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 65), 'int')
    # Getting the type of 'origin2' (line 832)
    origin2_124909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 29), 'origin2', False)
    int_124910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 38), 'int')
    # Getting the type of 'False' (line 832)
    False_124911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 41), 'False', False)
    # Processing the call keyword arguments (line 831)
    kwargs_124912 = {}
    # Getting the type of '_binary_erosion' (line 831)
    _binary_erosion_124902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 13), '_binary_erosion', False)
    # Calling _binary_erosion(args, kwargs) (line 831)
    _binary_erosion_call_result_124913 = invoke(stypy.reporting.localization.Localization(__file__, 831, 13), _binary_erosion_124902, *[input_124903, structure2_124904, int_124905, None_124906, output_124907, int_124908, origin2_124909, int_124910, False_124911], **kwargs_124912)
    
    # Assigning a type to the variable 'result' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 4), 'result', _binary_erosion_call_result_124913)
    
    # Getting the type of 'inplace' (line 833)
    inplace_124914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 7), 'inplace')
    # Testing the type of an if condition (line 833)
    if_condition_124915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 833, 4), inplace_124914)
    # Assigning a type to the variable 'if_condition_124915' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'if_condition_124915', if_condition_124915)
    # SSA begins for if statement (line 833)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to logical_not(...): (line 834)
    # Processing the call arguments (line 834)
    # Getting the type of 'output' (line 834)
    output_124918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 26), 'output', False)
    # Getting the type of 'output' (line 834)
    output_124919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 34), 'output', False)
    # Processing the call keyword arguments (line 834)
    kwargs_124920 = {}
    # Getting the type of 'numpy' (line 834)
    numpy_124916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'numpy', False)
    # Obtaining the member 'logical_not' of a type (line 834)
    logical_not_124917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 8), numpy_124916, 'logical_not')
    # Calling logical_not(args, kwargs) (line 834)
    logical_not_call_result_124921 = invoke(stypy.reporting.localization.Localization(__file__, 834, 8), logical_not_124917, *[output_124918, output_124919], **kwargs_124920)
    
    
    # Call to logical_and(...): (line 835)
    # Processing the call arguments (line 835)
    # Getting the type of 'tmp1' (line 835)
    tmp1_124924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 26), 'tmp1', False)
    # Getting the type of 'output' (line 835)
    output_124925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 32), 'output', False)
    # Getting the type of 'output' (line 835)
    output_124926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 40), 'output', False)
    # Processing the call keyword arguments (line 835)
    kwargs_124927 = {}
    # Getting the type of 'numpy' (line 835)
    numpy_124922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'numpy', False)
    # Obtaining the member 'logical_and' of a type (line 835)
    logical_and_124923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 8), numpy_124922, 'logical_and')
    # Calling logical_and(args, kwargs) (line 835)
    logical_and_call_result_124928 = invoke(stypy.reporting.localization.Localization(__file__, 835, 8), logical_and_124923, *[tmp1_124924, output_124925, output_124926], **kwargs_124927)
    
    # SSA branch for the else part of an if statement (line 833)
    module_type_store.open_ssa_branch('else')
    
    # Call to logical_not(...): (line 837)
    # Processing the call arguments (line 837)
    # Getting the type of 'result' (line 837)
    result_124931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 26), 'result', False)
    # Getting the type of 'result' (line 837)
    result_124932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 34), 'result', False)
    # Processing the call keyword arguments (line 837)
    kwargs_124933 = {}
    # Getting the type of 'numpy' (line 837)
    numpy_124929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'numpy', False)
    # Obtaining the member 'logical_not' of a type (line 837)
    logical_not_124930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 8), numpy_124929, 'logical_not')
    # Calling logical_not(args, kwargs) (line 837)
    logical_not_call_result_124934 = invoke(stypy.reporting.localization.Localization(__file__, 837, 8), logical_not_124930, *[result_124931, result_124932], **kwargs_124933)
    
    
    # Call to logical_and(...): (line 838)
    # Processing the call arguments (line 838)
    # Getting the type of 'tmp1' (line 838)
    tmp1_124937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 33), 'tmp1', False)
    # Getting the type of 'result' (line 838)
    result_124938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 39), 'result', False)
    # Processing the call keyword arguments (line 838)
    kwargs_124939 = {}
    # Getting the type of 'numpy' (line 838)
    numpy_124935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 15), 'numpy', False)
    # Obtaining the member 'logical_and' of a type (line 838)
    logical_and_124936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 15), numpy_124935, 'logical_and')
    # Calling logical_and(args, kwargs) (line 838)
    logical_and_call_result_124940 = invoke(stypy.reporting.localization.Localization(__file__, 838, 15), logical_and_124936, *[tmp1_124937, result_124938], **kwargs_124939)
    
    # Assigning a type to the variable 'stypy_return_type' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'stypy_return_type', logical_and_call_result_124940)
    # SSA join for if statement (line 833)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'binary_hit_or_miss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binary_hit_or_miss' in the type store
    # Getting the type of 'stypy_return_type' (line 732)
    stypy_return_type_124941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124941)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binary_hit_or_miss'
    return stypy_return_type_124941

# Assigning a type to the variable 'binary_hit_or_miss' (line 732)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 0), 'binary_hit_or_miss', binary_hit_or_miss)

@norecursion
def binary_propagation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 841)
    None_124942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 40), 'None')
    # Getting the type of 'None' (line 841)
    None_124943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 51), 'None')
    # Getting the type of 'None' (line 842)
    None_124944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 30), 'None')
    int_124945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 49), 'int')
    int_124946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 59), 'int')
    defaults = [None_124942, None_124943, None_124944, int_124945, int_124946]
    # Create a new context for function 'binary_propagation'
    module_type_store = module_type_store.open_function_context('binary_propagation', 841, 0, False)
    
    # Passed parameters checking function
    binary_propagation.stypy_localization = localization
    binary_propagation.stypy_type_of_self = None
    binary_propagation.stypy_type_store = module_type_store
    binary_propagation.stypy_function_name = 'binary_propagation'
    binary_propagation.stypy_param_names_list = ['input', 'structure', 'mask', 'output', 'border_value', 'origin']
    binary_propagation.stypy_varargs_param_name = None
    binary_propagation.stypy_kwargs_param_name = None
    binary_propagation.stypy_call_defaults = defaults
    binary_propagation.stypy_call_varargs = varargs
    binary_propagation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binary_propagation', ['input', 'structure', 'mask', 'output', 'border_value', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binary_propagation', localization, ['input', 'structure', 'mask', 'output', 'border_value', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binary_propagation(...)' code ##################

    str_124947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, (-1)), 'str', '\n    Multi-dimensional binary propagation with the given structuring element.\n\n    Parameters\n    ----------\n    input : array_like\n        Binary image to be propagated inside `mask`.\n    structure : array_like, optional\n        Structuring element used in the successive dilations. The output\n        may depend on the structuring element, especially if `mask` has\n        several connex components. If no structuring element is\n        provided, an element is generated with a squared connectivity equal\n        to one.\n    mask : array_like, optional\n        Binary mask defining the region into which `input` is allowed to\n        propagate.\n    output : ndarray, optional\n        Array of the same shape as input, into which the output is placed.\n        By default, a new array is created.\n    border_value : int (cast to 0 or 1), optional\n        Value at the border in the output array.\n    origin : int or tuple of ints, optional\n        Placement of the filter, by default 0.\n\n    Returns\n    -------\n    binary_propagation : ndarray\n        Binary propagation of `input` inside `mask`.\n\n    Notes\n    -----\n    This function is functionally equivalent to calling binary_dilation\n    with the number of iterations less than one: iterative dilation until\n    the result does not change anymore.\n\n    The succession of an erosion and propagation inside the original image\n    can be used instead of an *opening* for deleting small objects while\n    keeping the contours of larger objects untouched.\n\n    References\n    ----------\n    .. [1] http://cmm.ensmp.fr/~serra/cours/pdf/en/ch6en.pdf, slide 15.\n    .. [2] I.T. Young, J.J. Gerbrands, and L.J. van Vliet, "Fundamentals of\n        image processing", 1998\n        ftp://qiftp.tudelft.nl/DIPimage/docs/FIP2.3.pdf\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> input = np.zeros((8, 8), dtype=int)\n    >>> input[2, 2] = 1\n    >>> mask = np.zeros((8, 8), dtype=int)\n    >>> mask[1:4, 1:4] = mask[4, 4]  = mask[6:8, 6:8] = 1\n    >>> input\n    array([[0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> mask\n    array([[0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 1, 1],\n           [0, 0, 0, 0, 0, 0, 1, 1]])\n    >>> ndimage.binary_propagation(input, mask=mask).astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> ndimage.binary_propagation(input, mask=mask,\\\n    ... structure=np.ones((3,3))).astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0]])\n\n    >>> # Comparison between opening and erosion+propagation\n    >>> a = np.zeros((6,6), dtype=int)\n    >>> a[2:5, 2:5] = 1; a[0, 0] = 1; a[5, 5] = 1\n    >>> a\n    array([[1, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0],\n           [0, 0, 1, 1, 1, 0],\n           [0, 0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0, 1]])\n    >>> ndimage.binary_opening(a).astype(int)\n    array([[0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0],\n           [0, 0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0]])\n    >>> b = ndimage.binary_erosion(a)\n    >>> b.astype(int)\n    array([[0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0]])\n    >>> ndimage.binary_propagation(b, mask=a).astype(int)\n    array([[0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0],\n           [0, 0, 1, 1, 1, 0],\n           [0, 0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0, 0]])\n\n    ')
    
    # Call to binary_dilation(...): (line 968)
    # Processing the call arguments (line 968)
    # Getting the type of 'input' (line 968)
    input_124949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 27), 'input', False)
    # Getting the type of 'structure' (line 968)
    structure_124950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 34), 'structure', False)
    int_124951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 45), 'int')
    # Getting the type of 'mask' (line 968)
    mask_124952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 49), 'mask', False)
    # Getting the type of 'output' (line 968)
    output_124953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 55), 'output', False)
    # Getting the type of 'border_value' (line 969)
    border_value_124954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 27), 'border_value', False)
    # Getting the type of 'origin' (line 969)
    origin_124955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 41), 'origin', False)
    # Processing the call keyword arguments (line 968)
    kwargs_124956 = {}
    # Getting the type of 'binary_dilation' (line 968)
    binary_dilation_124948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 11), 'binary_dilation', False)
    # Calling binary_dilation(args, kwargs) (line 968)
    binary_dilation_call_result_124957 = invoke(stypy.reporting.localization.Localization(__file__, 968, 11), binary_dilation_124948, *[input_124949, structure_124950, int_124951, mask_124952, output_124953, border_value_124954, origin_124955], **kwargs_124956)
    
    # Assigning a type to the variable 'stypy_return_type' (line 968)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'stypy_return_type', binary_dilation_call_result_124957)
    
    # ################# End of 'binary_propagation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binary_propagation' in the type store
    # Getting the type of 'stypy_return_type' (line 841)
    stypy_return_type_124958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124958)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binary_propagation'
    return stypy_return_type_124958

# Assigning a type to the variable 'binary_propagation' (line 841)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 0), 'binary_propagation', binary_propagation)

@norecursion
def binary_fill_holes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 972)
    None_124959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 39), 'None')
    # Getting the type of 'None' (line 972)
    None_124960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 52), 'None')
    int_124961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 65), 'int')
    defaults = [None_124959, None_124960, int_124961]
    # Create a new context for function 'binary_fill_holes'
    module_type_store = module_type_store.open_function_context('binary_fill_holes', 972, 0, False)
    
    # Passed parameters checking function
    binary_fill_holes.stypy_localization = localization
    binary_fill_holes.stypy_type_of_self = None
    binary_fill_holes.stypy_type_store = module_type_store
    binary_fill_holes.stypy_function_name = 'binary_fill_holes'
    binary_fill_holes.stypy_param_names_list = ['input', 'structure', 'output', 'origin']
    binary_fill_holes.stypy_varargs_param_name = None
    binary_fill_holes.stypy_kwargs_param_name = None
    binary_fill_holes.stypy_call_defaults = defaults
    binary_fill_holes.stypy_call_varargs = varargs
    binary_fill_holes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binary_fill_holes', ['input', 'structure', 'output', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binary_fill_holes', localization, ['input', 'structure', 'output', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binary_fill_holes(...)' code ##################

    str_124962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, (-1)), 'str', '\n    Fill the holes in binary objects.\n\n\n    Parameters\n    ----------\n    input : array_like\n        n-dimensional binary array with holes to be filled\n    structure : array_like, optional\n        Structuring element used in the computation; large-size elements\n        make computations faster but may miss holes separated from the\n        background by thin regions. The default element (with a square\n        connectivity equal to one) yields the intuitive result where all\n        holes in the input have been filled.\n    output : ndarray, optional\n        Array of the same shape as input, into which the output is placed.\n        By default, a new array is created.\n    origin : int, tuple of ints, optional\n        Position of the structuring element.\n\n    Returns\n    -------\n    out : ndarray\n        Transformation of the initial image `input` where holes have been\n        filled.\n\n    See also\n    --------\n    binary_dilation, binary_propagation, label\n\n    Notes\n    -----\n    The algorithm used in this function consists in invading the complementary\n    of the shapes in `input` from the outer boundary of the image,\n    using binary dilations. Holes are not connected to the boundary and are\n    therefore not invaded. The result is the complementary subset of the\n    invaded region.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((5, 5), dtype=int)\n    >>> a[1:4, 1:4] = 1\n    >>> a[2,2] = 0\n    >>> a\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 0, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0]])\n    >>> ndimage.binary_fill_holes(a).astype(int)\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0]])\n    >>> # Too big structuring element\n    >>> ndimage.binary_fill_holes(a, structure=np.ones((5,5))).astype(int)\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 0, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0]])\n\n    ')
    
    # Assigning a Call to a Name (line 1043):
    
    # Assigning a Call to a Name (line 1043):
    
    # Call to logical_not(...): (line 1043)
    # Processing the call arguments (line 1043)
    # Getting the type of 'input' (line 1043)
    input_124965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 29), 'input', False)
    # Processing the call keyword arguments (line 1043)
    kwargs_124966 = {}
    # Getting the type of 'numpy' (line 1043)
    numpy_124963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 11), 'numpy', False)
    # Obtaining the member 'logical_not' of a type (line 1043)
    logical_not_124964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 11), numpy_124963, 'logical_not')
    # Calling logical_not(args, kwargs) (line 1043)
    logical_not_call_result_124967 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 11), logical_not_124964, *[input_124965], **kwargs_124966)
    
    # Assigning a type to the variable 'mask' (line 1043)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 4), 'mask', logical_not_call_result_124967)
    
    # Assigning a Call to a Name (line 1044):
    
    # Assigning a Call to a Name (line 1044):
    
    # Call to zeros(...): (line 1044)
    # Processing the call arguments (line 1044)
    # Getting the type of 'mask' (line 1044)
    mask_124970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 22), 'mask', False)
    # Obtaining the member 'shape' of a type (line 1044)
    shape_124971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 22), mask_124970, 'shape')
    # Getting the type of 'bool' (line 1044)
    bool_124972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 34), 'bool', False)
    # Processing the call keyword arguments (line 1044)
    kwargs_124973 = {}
    # Getting the type of 'numpy' (line 1044)
    numpy_124968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 10), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 1044)
    zeros_124969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 10), numpy_124968, 'zeros')
    # Calling zeros(args, kwargs) (line 1044)
    zeros_call_result_124974 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 10), zeros_124969, *[shape_124971, bool_124972], **kwargs_124973)
    
    # Assigning a type to the variable 'tmp' (line 1044)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1044, 4), 'tmp', zeros_call_result_124974)
    
    # Assigning a Call to a Name (line 1045):
    
    # Assigning a Call to a Name (line 1045):
    
    # Call to isinstance(...): (line 1045)
    # Processing the call arguments (line 1045)
    # Getting the type of 'output' (line 1045)
    output_124976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 25), 'output', False)
    # Getting the type of 'numpy' (line 1045)
    numpy_124977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 33), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1045)
    ndarray_124978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1045, 33), numpy_124977, 'ndarray')
    # Processing the call keyword arguments (line 1045)
    kwargs_124979 = {}
    # Getting the type of 'isinstance' (line 1045)
    isinstance_124975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 14), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1045)
    isinstance_call_result_124980 = invoke(stypy.reporting.localization.Localization(__file__, 1045, 14), isinstance_124975, *[output_124976, ndarray_124978], **kwargs_124979)
    
    # Assigning a type to the variable 'inplace' (line 1045)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'inplace', isinstance_call_result_124980)
    
    # Getting the type of 'inplace' (line 1046)
    inplace_124981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 7), 'inplace')
    # Testing the type of an if condition (line 1046)
    if_condition_124982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1046, 4), inplace_124981)
    # Assigning a type to the variable 'if_condition_124982' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 4), 'if_condition_124982', if_condition_124982)
    # SSA begins for if statement (line 1046)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to binary_dilation(...): (line 1047)
    # Processing the call arguments (line 1047)
    # Getting the type of 'tmp' (line 1047)
    tmp_124984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 24), 'tmp', False)
    # Getting the type of 'structure' (line 1047)
    structure_124985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 29), 'structure', False)
    int_124986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 40), 'int')
    # Getting the type of 'mask' (line 1047)
    mask_124987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 44), 'mask', False)
    # Getting the type of 'output' (line 1047)
    output_124988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 50), 'output', False)
    int_124989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 58), 'int')
    # Getting the type of 'origin' (line 1047)
    origin_124990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 61), 'origin', False)
    # Processing the call keyword arguments (line 1047)
    kwargs_124991 = {}
    # Getting the type of 'binary_dilation' (line 1047)
    binary_dilation_124983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 8), 'binary_dilation', False)
    # Calling binary_dilation(args, kwargs) (line 1047)
    binary_dilation_call_result_124992 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 8), binary_dilation_124983, *[tmp_124984, structure_124985, int_124986, mask_124987, output_124988, int_124989, origin_124990], **kwargs_124991)
    
    
    # Call to logical_not(...): (line 1048)
    # Processing the call arguments (line 1048)
    # Getting the type of 'output' (line 1048)
    output_124995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 26), 'output', False)
    # Getting the type of 'output' (line 1048)
    output_124996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 34), 'output', False)
    # Processing the call keyword arguments (line 1048)
    kwargs_124997 = {}
    # Getting the type of 'numpy' (line 1048)
    numpy_124993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 8), 'numpy', False)
    # Obtaining the member 'logical_not' of a type (line 1048)
    logical_not_124994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1048, 8), numpy_124993, 'logical_not')
    # Calling logical_not(args, kwargs) (line 1048)
    logical_not_call_result_124998 = invoke(stypy.reporting.localization.Localization(__file__, 1048, 8), logical_not_124994, *[output_124995, output_124996], **kwargs_124997)
    
    # SSA branch for the else part of an if statement (line 1046)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1050):
    
    # Assigning a Call to a Name (line 1050):
    
    # Call to binary_dilation(...): (line 1050)
    # Processing the call arguments (line 1050)
    # Getting the type of 'tmp' (line 1050)
    tmp_125000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 33), 'tmp', False)
    # Getting the type of 'structure' (line 1050)
    structure_125001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 38), 'structure', False)
    int_125002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 49), 'int')
    # Getting the type of 'mask' (line 1050)
    mask_125003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 53), 'mask', False)
    # Getting the type of 'None' (line 1050)
    None_125004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 59), 'None', False)
    int_125005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 65), 'int')
    # Getting the type of 'origin' (line 1051)
    origin_125006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 33), 'origin', False)
    # Processing the call keyword arguments (line 1050)
    kwargs_125007 = {}
    # Getting the type of 'binary_dilation' (line 1050)
    binary_dilation_124999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 17), 'binary_dilation', False)
    # Calling binary_dilation(args, kwargs) (line 1050)
    binary_dilation_call_result_125008 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 17), binary_dilation_124999, *[tmp_125000, structure_125001, int_125002, mask_125003, None_125004, int_125005, origin_125006], **kwargs_125007)
    
    # Assigning a type to the variable 'output' (line 1050)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 8), 'output', binary_dilation_call_result_125008)
    
    # Call to logical_not(...): (line 1052)
    # Processing the call arguments (line 1052)
    # Getting the type of 'output' (line 1052)
    output_125011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 26), 'output', False)
    # Getting the type of 'output' (line 1052)
    output_125012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 34), 'output', False)
    # Processing the call keyword arguments (line 1052)
    kwargs_125013 = {}
    # Getting the type of 'numpy' (line 1052)
    numpy_125009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'numpy', False)
    # Obtaining the member 'logical_not' of a type (line 1052)
    logical_not_125010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 8), numpy_125009, 'logical_not')
    # Calling logical_not(args, kwargs) (line 1052)
    logical_not_call_result_125014 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 8), logical_not_125010, *[output_125011, output_125012], **kwargs_125013)
    
    # Getting the type of 'output' (line 1053)
    output_125015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 15), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 1053)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'stypy_return_type', output_125015)
    # SSA join for if statement (line 1046)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'binary_fill_holes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binary_fill_holes' in the type store
    # Getting the type of 'stypy_return_type' (line 972)
    stypy_return_type_125016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125016)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binary_fill_holes'
    return stypy_return_type_125016

# Assigning a type to the variable 'binary_fill_holes' (line 972)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 0), 'binary_fill_holes', binary_fill_holes)

@norecursion
def grey_erosion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1056)
    None_125017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 29), 'None')
    # Getting the type of 'None' (line 1056)
    None_125018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 45), 'None')
    # Getting the type of 'None' (line 1056)
    None_125019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 61), 'None')
    # Getting the type of 'None' (line 1057)
    None_125020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 24), 'None')
    str_125021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 35), 'str', 'reflect')
    float_125022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 51), 'float')
    int_125023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 63), 'int')
    defaults = [None_125017, None_125018, None_125019, None_125020, str_125021, float_125022, int_125023]
    # Create a new context for function 'grey_erosion'
    module_type_store = module_type_store.open_function_context('grey_erosion', 1056, 0, False)
    
    # Passed parameters checking function
    grey_erosion.stypy_localization = localization
    grey_erosion.stypy_type_of_self = None
    grey_erosion.stypy_type_store = module_type_store
    grey_erosion.stypy_function_name = 'grey_erosion'
    grey_erosion.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin']
    grey_erosion.stypy_varargs_param_name = None
    grey_erosion.stypy_kwargs_param_name = None
    grey_erosion.stypy_call_defaults = defaults
    grey_erosion.stypy_call_varargs = varargs
    grey_erosion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'grey_erosion', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'grey_erosion', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'grey_erosion(...)' code ##################

    str_125024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, (-1)), 'str', "\n    Calculate a greyscale erosion, using either a structuring element,\n    or a footprint corresponding to a flat structuring element.\n\n    Grayscale erosion is a mathematical morphology operation. For the\n    simple case of a full and flat structuring element, it can be viewed\n    as a minimum filter over a sliding window.\n\n    Parameters\n    ----------\n    input : array_like\n        Array over which the grayscale erosion is to be computed.\n    size : tuple of ints\n        Shape of a flat and full structuring element used for the grayscale\n        erosion. Optional if `footprint` or `structure` is provided.\n    footprint : array of ints, optional\n        Positions of non-infinite elements of a flat structuring element\n        used for the grayscale erosion. Non-zero values give the set of\n        neighbors of the center over which the minimum is chosen.\n    structure : array of ints, optional\n        Structuring element used for the grayscale erosion. `structure`\n        may be a non-flat structuring element.\n    output : array, optional\n        An array used for storing the ouput of the erosion may be provided.\n    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional\n        The `mode` parameter determines how the array borders are\n        handled, where `cval` is the value when mode is equal to\n        'constant'. Default is 'reflect'\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n    origin : scalar, optional\n        The `origin` parameter controls the placement of the filter.\n        Default 0\n\n    Returns\n    -------\n    output : ndarray\n        Grayscale erosion of `input`.\n\n    See also\n    --------\n    binary_erosion, grey_dilation, grey_opening, grey_closing\n    generate_binary_structure, ndimage.minimum_filter\n\n    Notes\n    -----\n    The grayscale erosion of an image input by a structuring element s defined\n    over a domain E is given by:\n\n    (input+s)(x) = min {input(y) - s(x-y), for y in E}\n\n    In particular, for structuring elements defined as\n    s(y) = 0 for y in E, the grayscale erosion computes the minimum of the\n    input image inside a sliding window defined by E.\n\n    Grayscale erosion [1]_ is a *mathematical morphology* operation [2]_.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Erosion_%28morphology%29\n    .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((7,7), dtype=int)\n    >>> a[1:6, 1:6] = 3\n    >>> a[4,4] = 2; a[2,3] = 1\n    >>> a\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 3, 3, 3, 3, 3, 0],\n           [0, 3, 3, 1, 3, 3, 0],\n           [0, 3, 3, 3, 3, 3, 0],\n           [0, 3, 3, 3, 2, 3, 0],\n           [0, 3, 3, 3, 3, 3, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> ndimage.grey_erosion(a, size=(3,3))\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 3, 2, 2, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> footprint = ndimage.generate_binary_structure(2, 1)\n    >>> footprint\n    array([[False,  True, False],\n           [ True,  True,  True],\n           [False,  True, False]], dtype=bool)\n    >>> # Diagonally-connected elements are not considered neighbors\n    >>> ndimage.grey_erosion(a, size=(3,3), footprint=footprint)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 3, 1, 2, 0, 0],\n           [0, 0, 3, 2, 2, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 1159)
    size_125025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 7), 'size')
    # Getting the type of 'None' (line 1159)
    None_125026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 15), 'None')
    # Applying the binary operator 'is' (line 1159)
    result_is__125027 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 7), 'is', size_125025, None_125026)
    
    
    # Getting the type of 'footprint' (line 1159)
    footprint_125028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 24), 'footprint')
    # Getting the type of 'None' (line 1159)
    None_125029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 37), 'None')
    # Applying the binary operator 'is' (line 1159)
    result_is__125030 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 24), 'is', footprint_125028, None_125029)
    
    # Applying the binary operator 'and' (line 1159)
    result_and_keyword_125031 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 7), 'and', result_is__125027, result_is__125030)
    
    # Getting the type of 'structure' (line 1159)
    structure_125032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 46), 'structure')
    # Getting the type of 'None' (line 1159)
    None_125033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 59), 'None')
    # Applying the binary operator 'is' (line 1159)
    result_is__125034 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 46), 'is', structure_125032, None_125033)
    
    # Applying the binary operator 'and' (line 1159)
    result_and_keyword_125035 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 7), 'and', result_and_keyword_125031, result_is__125034)
    
    # Testing the type of an if condition (line 1159)
    if_condition_125036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1159, 4), result_and_keyword_125035)
    # Assigning a type to the variable 'if_condition_125036' (line 1159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 4), 'if_condition_125036', if_condition_125036)
    # SSA begins for if statement (line 1159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1160)
    # Processing the call arguments (line 1160)
    str_125038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 25), 'str', 'size, footprint or structure must be specified')
    # Processing the call keyword arguments (line 1160)
    kwargs_125039 = {}
    # Getting the type of 'ValueError' (line 1160)
    ValueError_125037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1160)
    ValueError_call_result_125040 = invoke(stypy.reporting.localization.Localization(__file__, 1160, 14), ValueError_125037, *[str_125038], **kwargs_125039)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1160, 8), ValueError_call_result_125040, 'raise parameter', BaseException)
    # SSA join for if statement (line 1159)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _min_or_max_filter(...): (line 1162)
    # Processing the call arguments (line 1162)
    # Getting the type of 'input' (line 1162)
    input_125043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 38), 'input', False)
    # Getting the type of 'size' (line 1162)
    size_125044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 45), 'size', False)
    # Getting the type of 'footprint' (line 1162)
    footprint_125045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 51), 'footprint', False)
    # Getting the type of 'structure' (line 1162)
    structure_125046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 62), 'structure', False)
    # Getting the type of 'output' (line 1163)
    output_125047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 38), 'output', False)
    # Getting the type of 'mode' (line 1163)
    mode_125048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 46), 'mode', False)
    # Getting the type of 'cval' (line 1163)
    cval_125049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 52), 'cval', False)
    # Getting the type of 'origin' (line 1163)
    origin_125050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 58), 'origin', False)
    int_125051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, 66), 'int')
    # Processing the call keyword arguments (line 1162)
    kwargs_125052 = {}
    # Getting the type of 'filters' (line 1162)
    filters_125041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 11), 'filters', False)
    # Obtaining the member '_min_or_max_filter' of a type (line 1162)
    _min_or_max_filter_125042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1162, 11), filters_125041, '_min_or_max_filter')
    # Calling _min_or_max_filter(args, kwargs) (line 1162)
    _min_or_max_filter_call_result_125053 = invoke(stypy.reporting.localization.Localization(__file__, 1162, 11), _min_or_max_filter_125042, *[input_125043, size_125044, footprint_125045, structure_125046, output_125047, mode_125048, cval_125049, origin_125050, int_125051], **kwargs_125052)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 4), 'stypy_return_type', _min_or_max_filter_call_result_125053)
    
    # ################# End of 'grey_erosion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'grey_erosion' in the type store
    # Getting the type of 'stypy_return_type' (line 1056)
    stypy_return_type_125054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125054)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'grey_erosion'
    return stypy_return_type_125054

# Assigning a type to the variable 'grey_erosion' (line 1056)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 0), 'grey_erosion', grey_erosion)

@norecursion
def grey_dilation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1166)
    None_125055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 30), 'None')
    # Getting the type of 'None' (line 1166)
    None_125056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 46), 'None')
    # Getting the type of 'None' (line 1166)
    None_125057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 62), 'None')
    # Getting the type of 'None' (line 1167)
    None_125058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 24), 'None')
    str_125059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 35), 'str', 'reflect')
    float_125060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 51), 'float')
    int_125061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 63), 'int')
    defaults = [None_125055, None_125056, None_125057, None_125058, str_125059, float_125060, int_125061]
    # Create a new context for function 'grey_dilation'
    module_type_store = module_type_store.open_function_context('grey_dilation', 1166, 0, False)
    
    # Passed parameters checking function
    grey_dilation.stypy_localization = localization
    grey_dilation.stypy_type_of_self = None
    grey_dilation.stypy_type_store = module_type_store
    grey_dilation.stypy_function_name = 'grey_dilation'
    grey_dilation.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin']
    grey_dilation.stypy_varargs_param_name = None
    grey_dilation.stypy_kwargs_param_name = None
    grey_dilation.stypy_call_defaults = defaults
    grey_dilation.stypy_call_varargs = varargs
    grey_dilation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'grey_dilation', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'grey_dilation', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'grey_dilation(...)' code ##################

    str_125062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1283, (-1)), 'str', "\n    Calculate a greyscale dilation, using either a structuring element,\n    or a footprint corresponding to a flat structuring element.\n\n    Grayscale dilation is a mathematical morphology operation. For the\n    simple case of a full and flat structuring element, it can be viewed\n    as a maximum filter over a sliding window.\n\n    Parameters\n    ----------\n    input : array_like\n        Array over which the grayscale dilation is to be computed.\n    size : tuple of ints\n        Shape of a flat and full structuring element used for the grayscale\n        dilation. Optional if `footprint` or `structure` is provided.\n    footprint : array of ints, optional\n        Positions of non-infinite elements of a flat structuring element\n        used for the grayscale dilation. Non-zero values give the set of\n        neighbors of the center over which the maximum is chosen.\n    structure : array of ints, optional\n        Structuring element used for the grayscale dilation. `structure`\n        may be a non-flat structuring element.\n    output : array, optional\n        An array used for storing the ouput of the dilation may be provided.\n    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional\n        The `mode` parameter determines how the array borders are\n        handled, where `cval` is the value when mode is equal to\n        'constant'. Default is 'reflect'\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n    origin : scalar, optional\n        The `origin` parameter controls the placement of the filter.\n        Default 0\n\n    Returns\n    -------\n    grey_dilation : ndarray\n        Grayscale dilation of `input`.\n\n    See also\n    --------\n    binary_dilation, grey_erosion, grey_closing, grey_opening\n    generate_binary_structure, ndimage.maximum_filter\n\n    Notes\n    -----\n    The grayscale dilation of an image input by a structuring element s defined\n    over a domain E is given by:\n\n    (input+s)(x) = max {input(y) + s(x-y), for y in E}\n\n    In particular, for structuring elements defined as\n    s(y) = 0 for y in E, the grayscale dilation computes the maximum of the\n    input image inside a sliding window defined by E.\n\n    Grayscale dilation [1]_ is a *mathematical morphology* operation [2]_.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Dilation_%28morphology%29\n    .. [2] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((7,7), dtype=int)\n    >>> a[2:5, 2:5] = 1\n    >>> a[4,4] = 2; a[2,3] = 3\n    >>> a\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 3, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 2, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> ndimage.grey_dilation(a, size=(3,3))\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 3, 3, 3, 1, 0],\n           [0, 1, 3, 3, 3, 1, 0],\n           [0, 1, 3, 3, 3, 2, 0],\n           [0, 1, 1, 2, 2, 2, 0],\n           [0, 1, 1, 2, 2, 2, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> ndimage.grey_dilation(a, footprint=np.ones((3,3)))\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 3, 3, 3, 1, 0],\n           [0, 1, 3, 3, 3, 1, 0],\n           [0, 1, 3, 3, 3, 2, 0],\n           [0, 1, 1, 2, 2, 2, 0],\n           [0, 1, 1, 2, 2, 2, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> s = ndimage.generate_binary_structure(2,1)\n    >>> s\n    array([[False,  True, False],\n           [ True,  True,  True],\n           [False,  True, False]], dtype=bool)\n    >>> ndimage.grey_dilation(a, footprint=s)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 3, 1, 0, 0],\n           [0, 1, 3, 3, 3, 1, 0],\n           [0, 1, 1, 3, 2, 1, 0],\n           [0, 1, 1, 2, 2, 2, 0],\n           [0, 0, 1, 1, 2, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> ndimage.grey_dilation(a, size=(3,3), structure=np.ones((3,3)))\n    array([[1, 1, 1, 1, 1, 1, 1],\n           [1, 2, 4, 4, 4, 2, 1],\n           [1, 2, 4, 4, 4, 2, 1],\n           [1, 2, 4, 4, 4, 3, 1],\n           [1, 2, 2, 3, 3, 3, 1],\n           [1, 2, 2, 3, 3, 3, 1],\n           [1, 1, 1, 1, 1, 1, 1]])\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 1284)
    size_125063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 7), 'size')
    # Getting the type of 'None' (line 1284)
    None_125064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 15), 'None')
    # Applying the binary operator 'is' (line 1284)
    result_is__125065 = python_operator(stypy.reporting.localization.Localization(__file__, 1284, 7), 'is', size_125063, None_125064)
    
    
    # Getting the type of 'footprint' (line 1284)
    footprint_125066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 24), 'footprint')
    # Getting the type of 'None' (line 1284)
    None_125067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 37), 'None')
    # Applying the binary operator 'is' (line 1284)
    result_is__125068 = python_operator(stypy.reporting.localization.Localization(__file__, 1284, 24), 'is', footprint_125066, None_125067)
    
    # Applying the binary operator 'and' (line 1284)
    result_and_keyword_125069 = python_operator(stypy.reporting.localization.Localization(__file__, 1284, 7), 'and', result_is__125065, result_is__125068)
    
    # Getting the type of 'structure' (line 1284)
    structure_125070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 46), 'structure')
    # Getting the type of 'None' (line 1284)
    None_125071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 59), 'None')
    # Applying the binary operator 'is' (line 1284)
    result_is__125072 = python_operator(stypy.reporting.localization.Localization(__file__, 1284, 46), 'is', structure_125070, None_125071)
    
    # Applying the binary operator 'and' (line 1284)
    result_and_keyword_125073 = python_operator(stypy.reporting.localization.Localization(__file__, 1284, 7), 'and', result_and_keyword_125069, result_is__125072)
    
    # Testing the type of an if condition (line 1284)
    if_condition_125074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1284, 4), result_and_keyword_125073)
    # Assigning a type to the variable 'if_condition_125074' (line 1284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1284, 4), 'if_condition_125074', if_condition_125074)
    # SSA begins for if statement (line 1284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1285)
    # Processing the call arguments (line 1285)
    str_125076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, 25), 'str', 'size, footprint or structure must be specified')
    # Processing the call keyword arguments (line 1285)
    kwargs_125077 = {}
    # Getting the type of 'ValueError' (line 1285)
    ValueError_125075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1285)
    ValueError_call_result_125078 = invoke(stypy.reporting.localization.Localization(__file__, 1285, 14), ValueError_125075, *[str_125076], **kwargs_125077)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1285, 8), ValueError_call_result_125078, 'raise parameter', BaseException)
    # SSA join for if statement (line 1284)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1286)
    # Getting the type of 'structure' (line 1286)
    structure_125079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 4), 'structure')
    # Getting the type of 'None' (line 1286)
    None_125080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 24), 'None')
    
    (may_be_125081, more_types_in_union_125082) = may_not_be_none(structure_125079, None_125080)

    if may_be_125081:

        if more_types_in_union_125082:
            # Runtime conditional SSA (line 1286)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1287):
        
        # Assigning a Call to a Name (line 1287):
        
        # Call to asarray(...): (line 1287)
        # Processing the call arguments (line 1287)
        # Getting the type of 'structure' (line 1287)
        structure_125085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 34), 'structure', False)
        # Processing the call keyword arguments (line 1287)
        kwargs_125086 = {}
        # Getting the type of 'numpy' (line 1287)
        numpy_125083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 20), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 1287)
        asarray_125084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 20), numpy_125083, 'asarray')
        # Calling asarray(args, kwargs) (line 1287)
        asarray_call_result_125087 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 20), asarray_125084, *[structure_125085], **kwargs_125086)
        
        # Assigning a type to the variable 'structure' (line 1287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 8), 'structure', asarray_call_result_125087)
        
        # Assigning a Subscript to a Name (line 1288):
        
        # Assigning a Subscript to a Name (line 1288):
        
        # Obtaining the type of the subscript
        
        # Call to tuple(...): (line 1288)
        # Processing the call arguments (line 1288)
        
        # Obtaining an instance of the builtin type 'list' (line 1288)
        list_125089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1288, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1288)
        # Adding element type (line 1288)
        
        # Call to slice(...): (line 1288)
        # Processing the call arguments (line 1288)
        # Getting the type of 'None' (line 1288)
        None_125091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 43), 'None', False)
        # Getting the type of 'None' (line 1288)
        None_125092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 49), 'None', False)
        int_125093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1288, 55), 'int')
        # Processing the call keyword arguments (line 1288)
        kwargs_125094 = {}
        # Getting the type of 'slice' (line 1288)
        slice_125090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 37), 'slice', False)
        # Calling slice(args, kwargs) (line 1288)
        slice_call_result_125095 = invoke(stypy.reporting.localization.Localization(__file__, 1288, 37), slice_125090, *[None_125091, None_125092, int_125093], **kwargs_125094)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1288, 36), list_125089, slice_call_result_125095)
        
        # Getting the type of 'structure' (line 1289)
        structure_125096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 36), 'structure', False)
        # Obtaining the member 'ndim' of a type (line 1289)
        ndim_125097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 36), structure_125096, 'ndim')
        # Applying the binary operator '*' (line 1288)
        result_mul_125098 = python_operator(stypy.reporting.localization.Localization(__file__, 1288, 36), '*', list_125089, ndim_125097)
        
        # Processing the call keyword arguments (line 1288)
        kwargs_125099 = {}
        # Getting the type of 'tuple' (line 1288)
        tuple_125088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 30), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1288)
        tuple_call_result_125100 = invoke(stypy.reporting.localization.Localization(__file__, 1288, 30), tuple_125088, *[result_mul_125098], **kwargs_125099)
        
        # Getting the type of 'structure' (line 1288)
        structure_125101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 20), 'structure')
        # Obtaining the member '__getitem__' of a type (line 1288)
        getitem___125102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1288, 20), structure_125101, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1288)
        subscript_call_result_125103 = invoke(stypy.reporting.localization.Localization(__file__, 1288, 20), getitem___125102, tuple_call_result_125100)
        
        # Assigning a type to the variable 'structure' (line 1288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1288, 8), 'structure', subscript_call_result_125103)

        if more_types_in_union_125082:
            # SSA join for if statement (line 1286)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1290)
    # Getting the type of 'footprint' (line 1290)
    footprint_125104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 4), 'footprint')
    # Getting the type of 'None' (line 1290)
    None_125105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 24), 'None')
    
    (may_be_125106, more_types_in_union_125107) = may_not_be_none(footprint_125104, None_125105)

    if may_be_125106:

        if more_types_in_union_125107:
            # Runtime conditional SSA (line 1290)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1291):
        
        # Assigning a Call to a Name (line 1291):
        
        # Call to asarray(...): (line 1291)
        # Processing the call arguments (line 1291)
        # Getting the type of 'footprint' (line 1291)
        footprint_125110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 34), 'footprint', False)
        # Processing the call keyword arguments (line 1291)
        kwargs_125111 = {}
        # Getting the type of 'numpy' (line 1291)
        numpy_125108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 20), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 1291)
        asarray_125109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1291, 20), numpy_125108, 'asarray')
        # Calling asarray(args, kwargs) (line 1291)
        asarray_call_result_125112 = invoke(stypy.reporting.localization.Localization(__file__, 1291, 20), asarray_125109, *[footprint_125110], **kwargs_125111)
        
        # Assigning a type to the variable 'footprint' (line 1291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 8), 'footprint', asarray_call_result_125112)
        
        # Assigning a Subscript to a Name (line 1292):
        
        # Assigning a Subscript to a Name (line 1292):
        
        # Obtaining the type of the subscript
        
        # Call to tuple(...): (line 1292)
        # Processing the call arguments (line 1292)
        
        # Obtaining an instance of the builtin type 'list' (line 1292)
        list_125114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1292)
        # Adding element type (line 1292)
        
        # Call to slice(...): (line 1292)
        # Processing the call arguments (line 1292)
        # Getting the type of 'None' (line 1292)
        None_125116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 43), 'None', False)
        # Getting the type of 'None' (line 1292)
        None_125117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 49), 'None', False)
        int_125118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 55), 'int')
        # Processing the call keyword arguments (line 1292)
        kwargs_125119 = {}
        # Getting the type of 'slice' (line 1292)
        slice_125115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 37), 'slice', False)
        # Calling slice(args, kwargs) (line 1292)
        slice_call_result_125120 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 37), slice_125115, *[None_125116, None_125117, int_125118], **kwargs_125119)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 36), list_125114, slice_call_result_125120)
        
        # Getting the type of 'footprint' (line 1293)
        footprint_125121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 36), 'footprint', False)
        # Obtaining the member 'ndim' of a type (line 1293)
        ndim_125122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1293, 36), footprint_125121, 'ndim')
        # Applying the binary operator '*' (line 1292)
        result_mul_125123 = python_operator(stypy.reporting.localization.Localization(__file__, 1292, 36), '*', list_125114, ndim_125122)
        
        # Processing the call keyword arguments (line 1292)
        kwargs_125124 = {}
        # Getting the type of 'tuple' (line 1292)
        tuple_125113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 30), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1292)
        tuple_call_result_125125 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 30), tuple_125113, *[result_mul_125123], **kwargs_125124)
        
        # Getting the type of 'footprint' (line 1292)
        footprint_125126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 20), 'footprint')
        # Obtaining the member '__getitem__' of a type (line 1292)
        getitem___125127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1292, 20), footprint_125126, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1292)
        subscript_call_result_125128 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 20), getitem___125127, tuple_call_result_125125)
        
        # Assigning a type to the variable 'footprint' (line 1292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 8), 'footprint', subscript_call_result_125128)

        if more_types_in_union_125107:
            # SSA join for if statement (line 1290)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1295):
    
    # Assigning a Call to a Name (line 1295):
    
    # Call to asarray(...): (line 1295)
    # Processing the call arguments (line 1295)
    # Getting the type of 'input' (line 1295)
    input_125131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 26), 'input', False)
    # Processing the call keyword arguments (line 1295)
    kwargs_125132 = {}
    # Getting the type of 'numpy' (line 1295)
    numpy_125129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1295)
    asarray_125130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1295, 12), numpy_125129, 'asarray')
    # Calling asarray(args, kwargs) (line 1295)
    asarray_call_result_125133 = invoke(stypy.reporting.localization.Localization(__file__, 1295, 12), asarray_125130, *[input_125131], **kwargs_125132)
    
    # Assigning a type to the variable 'input' (line 1295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1295, 4), 'input', asarray_call_result_125133)
    
    # Assigning a Call to a Name (line 1296):
    
    # Assigning a Call to a Name (line 1296):
    
    # Call to _normalize_sequence(...): (line 1296)
    # Processing the call arguments (line 1296)
    # Getting the type of 'origin' (line 1296)
    origin_125136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 45), 'origin', False)
    # Getting the type of 'input' (line 1296)
    input_125137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 53), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1296)
    ndim_125138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1296, 53), input_125137, 'ndim')
    # Processing the call keyword arguments (line 1296)
    kwargs_125139 = {}
    # Getting the type of '_ni_support' (line 1296)
    _ni_support_125134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 13), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 1296)
    _normalize_sequence_125135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1296, 13), _ni_support_125134, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 1296)
    _normalize_sequence_call_result_125140 = invoke(stypy.reporting.localization.Localization(__file__, 1296, 13), _normalize_sequence_125135, *[origin_125136, ndim_125138], **kwargs_125139)
    
    # Assigning a type to the variable 'origin' (line 1296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1296, 4), 'origin', _normalize_sequence_call_result_125140)
    
    
    # Call to range(...): (line 1297)
    # Processing the call arguments (line 1297)
    
    # Call to len(...): (line 1297)
    # Processing the call arguments (line 1297)
    # Getting the type of 'origin' (line 1297)
    origin_125143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 24), 'origin', False)
    # Processing the call keyword arguments (line 1297)
    kwargs_125144 = {}
    # Getting the type of 'len' (line 1297)
    len_125142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 20), 'len', False)
    # Calling len(args, kwargs) (line 1297)
    len_call_result_125145 = invoke(stypy.reporting.localization.Localization(__file__, 1297, 20), len_125142, *[origin_125143], **kwargs_125144)
    
    # Processing the call keyword arguments (line 1297)
    kwargs_125146 = {}
    # Getting the type of 'range' (line 1297)
    range_125141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 14), 'range', False)
    # Calling range(args, kwargs) (line 1297)
    range_call_result_125147 = invoke(stypy.reporting.localization.Localization(__file__, 1297, 14), range_125141, *[len_call_result_125145], **kwargs_125146)
    
    # Testing the type of a for loop iterable (line 1297)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1297, 4), range_call_result_125147)
    # Getting the type of the for loop variable (line 1297)
    for_loop_var_125148 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1297, 4), range_call_result_125147)
    # Assigning a type to the variable 'ii' (line 1297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1297, 4), 'ii', for_loop_var_125148)
    # SSA begins for a for statement (line 1297)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a UnaryOp to a Subscript (line 1298):
    
    # Assigning a UnaryOp to a Subscript (line 1298):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1298)
    ii_125149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 29), 'ii')
    # Getting the type of 'origin' (line 1298)
    origin_125150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 22), 'origin')
    # Obtaining the member '__getitem__' of a type (line 1298)
    getitem___125151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 22), origin_125150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1298)
    subscript_call_result_125152 = invoke(stypy.reporting.localization.Localization(__file__, 1298, 22), getitem___125151, ii_125149)
    
    # Applying the 'usub' unary operator (line 1298)
    result___neg___125153 = python_operator(stypy.reporting.localization.Localization(__file__, 1298, 21), 'usub', subscript_call_result_125152)
    
    # Getting the type of 'origin' (line 1298)
    origin_125154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 8), 'origin')
    # Getting the type of 'ii' (line 1298)
    ii_125155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 15), 'ii')
    # Storing an element on a container (line 1298)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1298, 8), origin_125154, (ii_125155, result___neg___125153))
    
    # Type idiom detected: calculating its left and rigth part (line 1299)
    # Getting the type of 'footprint' (line 1299)
    footprint_125156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 8), 'footprint')
    # Getting the type of 'None' (line 1299)
    None_125157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 28), 'None')
    
    (may_be_125158, more_types_in_union_125159) = may_not_be_none(footprint_125156, None_125157)

    if may_be_125158:

        if more_types_in_union_125159:
            # Runtime conditional SSA (line 1299)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 1300):
        
        # Assigning a Subscript to a Name (line 1300):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ii' (line 1300)
        ii_125160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 33), 'ii')
        # Getting the type of 'footprint' (line 1300)
        footprint_125161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 17), 'footprint')
        # Obtaining the member 'shape' of a type (line 1300)
        shape_125162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 17), footprint_125161, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1300)
        getitem___125163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 17), shape_125162, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1300)
        subscript_call_result_125164 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 17), getitem___125163, ii_125160)
        
        # Assigning a type to the variable 'sz' (line 1300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1300, 12), 'sz', subscript_call_result_125164)

        if more_types_in_union_125159:
            # Runtime conditional SSA for else branch (line 1299)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_125158) or more_types_in_union_125159):
        
        # Type idiom detected: calculating its left and rigth part (line 1301)
        # Getting the type of 'structure' (line 1301)
        structure_125165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 13), 'structure')
        # Getting the type of 'None' (line 1301)
        None_125166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 30), 'None')
        
        (may_be_125167, more_types_in_union_125168) = may_not_be_none(structure_125165, None_125166)

        if may_be_125167:

            if more_types_in_union_125168:
                # Runtime conditional SSA (line 1301)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 1302):
            
            # Assigning a Subscript to a Name (line 1302):
            
            # Obtaining the type of the subscript
            # Getting the type of 'ii' (line 1302)
            ii_125169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 33), 'ii')
            # Getting the type of 'structure' (line 1302)
            structure_125170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 17), 'structure')
            # Obtaining the member 'shape' of a type (line 1302)
            shape_125171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 17), structure_125170, 'shape')
            # Obtaining the member '__getitem__' of a type (line 1302)
            getitem___125172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 17), shape_125171, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1302)
            subscript_call_result_125173 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 17), getitem___125172, ii_125169)
            
            # Assigning a type to the variable 'sz' (line 1302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 12), 'sz', subscript_call_result_125173)

            if more_types_in_union_125168:
                # Runtime conditional SSA for else branch (line 1301)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_125167) or more_types_in_union_125168):
            
            
            # Call to isscalar(...): (line 1303)
            # Processing the call arguments (line 1303)
            # Getting the type of 'size' (line 1303)
            size_125176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 28), 'size', False)
            # Processing the call keyword arguments (line 1303)
            kwargs_125177 = {}
            # Getting the type of 'numpy' (line 1303)
            numpy_125174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 13), 'numpy', False)
            # Obtaining the member 'isscalar' of a type (line 1303)
            isscalar_125175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1303, 13), numpy_125174, 'isscalar')
            # Calling isscalar(args, kwargs) (line 1303)
            isscalar_call_result_125178 = invoke(stypy.reporting.localization.Localization(__file__, 1303, 13), isscalar_125175, *[size_125176], **kwargs_125177)
            
            # Testing the type of an if condition (line 1303)
            if_condition_125179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1303, 13), isscalar_call_result_125178)
            # Assigning a type to the variable 'if_condition_125179' (line 1303)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1303, 13), 'if_condition_125179', if_condition_125179)
            # SSA begins for if statement (line 1303)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 1304):
            
            # Assigning a Name to a Name (line 1304):
            # Getting the type of 'size' (line 1304)
            size_125180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 17), 'size')
            # Assigning a type to the variable 'sz' (line 1304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1304, 12), 'sz', size_125180)
            # SSA branch for the else part of an if statement (line 1303)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Name (line 1306):
            
            # Assigning a Subscript to a Name (line 1306):
            
            # Obtaining the type of the subscript
            # Getting the type of 'ii' (line 1306)
            ii_125181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 22), 'ii')
            # Getting the type of 'size' (line 1306)
            size_125182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 17), 'size')
            # Obtaining the member '__getitem__' of a type (line 1306)
            getitem___125183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1306, 17), size_125182, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1306)
            subscript_call_result_125184 = invoke(stypy.reporting.localization.Localization(__file__, 1306, 17), getitem___125183, ii_125181)
            
            # Assigning a type to the variable 'sz' (line 1306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1306, 12), 'sz', subscript_call_result_125184)
            # SSA join for if statement (line 1303)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_125167 and more_types_in_union_125168):
                # SSA join for if statement (line 1301)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_125158 and more_types_in_union_125159):
            # SSA join for if statement (line 1299)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'sz' (line 1307)
    sz_125185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1307, 15), 'sz')
    int_125186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1307, 20), 'int')
    # Applying the binary operator '&' (line 1307)
    result_and__125187 = python_operator(stypy.reporting.localization.Localization(__file__, 1307, 15), '&', sz_125185, int_125186)
    
    # Applying the 'not' unary operator (line 1307)
    result_not__125188 = python_operator(stypy.reporting.localization.Localization(__file__, 1307, 11), 'not', result_and__125187)
    
    # Testing the type of an if condition (line 1307)
    if_condition_125189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1307, 8), result_not__125188)
    # Assigning a type to the variable 'if_condition_125189' (line 1307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1307, 8), 'if_condition_125189', if_condition_125189)
    # SSA begins for if statement (line 1307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'origin' (line 1308)
    origin_125190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 12), 'origin')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1308)
    ii_125191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 19), 'ii')
    # Getting the type of 'origin' (line 1308)
    origin_125192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 12), 'origin')
    # Obtaining the member '__getitem__' of a type (line 1308)
    getitem___125193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1308, 12), origin_125192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1308)
    subscript_call_result_125194 = invoke(stypy.reporting.localization.Localization(__file__, 1308, 12), getitem___125193, ii_125191)
    
    int_125195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1308, 26), 'int')
    # Applying the binary operator '-=' (line 1308)
    result_isub_125196 = python_operator(stypy.reporting.localization.Localization(__file__, 1308, 12), '-=', subscript_call_result_125194, int_125195)
    # Getting the type of 'origin' (line 1308)
    origin_125197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 12), 'origin')
    # Getting the type of 'ii' (line 1308)
    ii_125198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 19), 'ii')
    # Storing an element on a container (line 1308)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1308, 12), origin_125197, (ii_125198, result_isub_125196))
    
    # SSA join for if statement (line 1307)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _min_or_max_filter(...): (line 1310)
    # Processing the call arguments (line 1310)
    # Getting the type of 'input' (line 1310)
    input_125201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1310, 38), 'input', False)
    # Getting the type of 'size' (line 1310)
    size_125202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1310, 45), 'size', False)
    # Getting the type of 'footprint' (line 1310)
    footprint_125203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1310, 51), 'footprint', False)
    # Getting the type of 'structure' (line 1310)
    structure_125204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1310, 62), 'structure', False)
    # Getting the type of 'output' (line 1311)
    output_125205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1311, 38), 'output', False)
    # Getting the type of 'mode' (line 1311)
    mode_125206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1311, 46), 'mode', False)
    # Getting the type of 'cval' (line 1311)
    cval_125207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1311, 52), 'cval', False)
    # Getting the type of 'origin' (line 1311)
    origin_125208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1311, 58), 'origin', False)
    int_125209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1311, 66), 'int')
    # Processing the call keyword arguments (line 1310)
    kwargs_125210 = {}
    # Getting the type of 'filters' (line 1310)
    filters_125199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1310, 11), 'filters', False)
    # Obtaining the member '_min_or_max_filter' of a type (line 1310)
    _min_or_max_filter_125200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1310, 11), filters_125199, '_min_or_max_filter')
    # Calling _min_or_max_filter(args, kwargs) (line 1310)
    _min_or_max_filter_call_result_125211 = invoke(stypy.reporting.localization.Localization(__file__, 1310, 11), _min_or_max_filter_125200, *[input_125201, size_125202, footprint_125203, structure_125204, output_125205, mode_125206, cval_125207, origin_125208, int_125209], **kwargs_125210)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1310, 4), 'stypy_return_type', _min_or_max_filter_call_result_125211)
    
    # ################# End of 'grey_dilation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'grey_dilation' in the type store
    # Getting the type of 'stypy_return_type' (line 1166)
    stypy_return_type_125212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125212)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'grey_dilation'
    return stypy_return_type_125212

# Assigning a type to the variable 'grey_dilation' (line 1166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1166, 0), 'grey_dilation', grey_dilation)

@norecursion
def grey_opening(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1314)
    None_125213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1314, 29), 'None')
    # Getting the type of 'None' (line 1314)
    None_125214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1314, 45), 'None')
    # Getting the type of 'None' (line 1314)
    None_125215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1314, 61), 'None')
    # Getting the type of 'None' (line 1315)
    None_125216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 24), 'None')
    str_125217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1315, 35), 'str', 'reflect')
    float_125218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1315, 51), 'float')
    int_125219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1315, 63), 'int')
    defaults = [None_125213, None_125214, None_125215, None_125216, str_125217, float_125218, int_125219]
    # Create a new context for function 'grey_opening'
    module_type_store = module_type_store.open_function_context('grey_opening', 1314, 0, False)
    
    # Passed parameters checking function
    grey_opening.stypy_localization = localization
    grey_opening.stypy_type_of_self = None
    grey_opening.stypy_type_store = module_type_store
    grey_opening.stypy_function_name = 'grey_opening'
    grey_opening.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin']
    grey_opening.stypy_varargs_param_name = None
    grey_opening.stypy_kwargs_param_name = None
    grey_opening.stypy_call_defaults = defaults
    grey_opening.stypy_call_varargs = varargs
    grey_opening.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'grey_opening', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'grey_opening', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'grey_opening(...)' code ##################

    str_125220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1388, (-1)), 'str', "\n    Multi-dimensional greyscale opening.\n\n    A greyscale opening consists in the succession of a greyscale erosion,\n    and a greyscale dilation.\n\n    Parameters\n    ----------\n    input : array_like\n        Array over which the grayscale opening is to be computed.\n    size : tuple of ints\n        Shape of a flat and full structuring element used for the grayscale\n        opening. Optional if `footprint` or `structure` is provided.\n    footprint : array of ints, optional\n        Positions of non-infinite elements of a flat structuring element\n        used for the grayscale opening.\n    structure : array of ints, optional\n        Structuring element used for the grayscale opening. `structure`\n        may be a non-flat structuring element.\n    output : array, optional\n        An array used for storing the ouput of the opening may be provided.\n    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional\n        The `mode` parameter determines how the array borders are\n        handled, where `cval` is the value when mode is equal to\n        'constant'. Default is 'reflect'\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n    origin : scalar, optional\n        The `origin` parameter controls the placement of the filter.\n        Default 0\n\n    Returns\n    -------\n    grey_opening : ndarray\n        Result of the grayscale opening of `input` with `structure`.\n\n    See also\n    --------\n    binary_opening, grey_dilation, grey_erosion, grey_closing\n    generate_binary_structure\n\n    Notes\n    -----\n    The action of a grayscale opening with a flat structuring element amounts\n    to smoothen high local maxima, whereas binary opening erases small objects.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.arange(36).reshape((6,6))\n    >>> a[3, 3] = 50\n    >>> a\n    array([[ 0,  1,  2,  3,  4,  5],\n           [ 6,  7,  8,  9, 10, 11],\n           [12, 13, 14, 15, 16, 17],\n           [18, 19, 20, 50, 22, 23],\n           [24, 25, 26, 27, 28, 29],\n           [30, 31, 32, 33, 34, 35]])\n    >>> ndimage.grey_opening(a, size=(3,3))\n    array([[ 0,  1,  2,  3,  4,  4],\n           [ 6,  7,  8,  9, 10, 10],\n           [12, 13, 14, 15, 16, 16],\n           [18, 19, 20, 22, 22, 22],\n           [24, 25, 26, 27, 28, 28],\n           [24, 25, 26, 27, 28, 28]])\n    >>> # Note that the local maximum a[3,3] has disappeared\n\n    ")
    
    # Assigning a Call to a Name (line 1389):
    
    # Assigning a Call to a Name (line 1389):
    
    # Call to grey_erosion(...): (line 1389)
    # Processing the call arguments (line 1389)
    # Getting the type of 'input' (line 1389)
    input_125222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 23), 'input', False)
    # Getting the type of 'size' (line 1389)
    size_125223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 30), 'size', False)
    # Getting the type of 'footprint' (line 1389)
    footprint_125224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 36), 'footprint', False)
    # Getting the type of 'structure' (line 1389)
    structure_125225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 47), 'structure', False)
    # Getting the type of 'None' (line 1389)
    None_125226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 58), 'None', False)
    # Getting the type of 'mode' (line 1389)
    mode_125227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 64), 'mode', False)
    # Getting the type of 'cval' (line 1390)
    cval_125228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1390, 23), 'cval', False)
    # Getting the type of 'origin' (line 1390)
    origin_125229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1390, 29), 'origin', False)
    # Processing the call keyword arguments (line 1389)
    kwargs_125230 = {}
    # Getting the type of 'grey_erosion' (line 1389)
    grey_erosion_125221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 10), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1389)
    grey_erosion_call_result_125231 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 10), grey_erosion_125221, *[input_125222, size_125223, footprint_125224, structure_125225, None_125226, mode_125227, cval_125228, origin_125229], **kwargs_125230)
    
    # Assigning a type to the variable 'tmp' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'tmp', grey_erosion_call_result_125231)
    
    # Call to grey_dilation(...): (line 1391)
    # Processing the call arguments (line 1391)
    # Getting the type of 'tmp' (line 1391)
    tmp_125233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 25), 'tmp', False)
    # Getting the type of 'size' (line 1391)
    size_125234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 30), 'size', False)
    # Getting the type of 'footprint' (line 1391)
    footprint_125235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 36), 'footprint', False)
    # Getting the type of 'structure' (line 1391)
    structure_125236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 47), 'structure', False)
    # Getting the type of 'output' (line 1391)
    output_125237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 58), 'output', False)
    # Getting the type of 'mode' (line 1391)
    mode_125238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 66), 'mode', False)
    # Getting the type of 'cval' (line 1392)
    cval_125239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 25), 'cval', False)
    # Getting the type of 'origin' (line 1392)
    origin_125240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 31), 'origin', False)
    # Processing the call keyword arguments (line 1391)
    kwargs_125241 = {}
    # Getting the type of 'grey_dilation' (line 1391)
    grey_dilation_125232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 11), 'grey_dilation', False)
    # Calling grey_dilation(args, kwargs) (line 1391)
    grey_dilation_call_result_125242 = invoke(stypy.reporting.localization.Localization(__file__, 1391, 11), grey_dilation_125232, *[tmp_125233, size_125234, footprint_125235, structure_125236, output_125237, mode_125238, cval_125239, origin_125240], **kwargs_125241)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 4), 'stypy_return_type', grey_dilation_call_result_125242)
    
    # ################# End of 'grey_opening(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'grey_opening' in the type store
    # Getting the type of 'stypy_return_type' (line 1314)
    stypy_return_type_125243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1314, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125243)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'grey_opening'
    return stypy_return_type_125243

# Assigning a type to the variable 'grey_opening' (line 1314)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1314, 0), 'grey_opening', grey_opening)

@norecursion
def grey_closing(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1395)
    None_125244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1395, 29), 'None')
    # Getting the type of 'None' (line 1395)
    None_125245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1395, 45), 'None')
    # Getting the type of 'None' (line 1395)
    None_125246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1395, 61), 'None')
    # Getting the type of 'None' (line 1396)
    None_125247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1396, 24), 'None')
    str_125248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1396, 35), 'str', 'reflect')
    float_125249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1396, 51), 'float')
    int_125250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1396, 63), 'int')
    defaults = [None_125244, None_125245, None_125246, None_125247, str_125248, float_125249, int_125250]
    # Create a new context for function 'grey_closing'
    module_type_store = module_type_store.open_function_context('grey_closing', 1395, 0, False)
    
    # Passed parameters checking function
    grey_closing.stypy_localization = localization
    grey_closing.stypy_type_of_self = None
    grey_closing.stypy_type_store = module_type_store
    grey_closing.stypy_function_name = 'grey_closing'
    grey_closing.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin']
    grey_closing.stypy_varargs_param_name = None
    grey_closing.stypy_kwargs_param_name = None
    grey_closing.stypy_call_defaults = defaults
    grey_closing.stypy_call_varargs = varargs
    grey_closing.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'grey_closing', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'grey_closing', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'grey_closing(...)' code ##################

    str_125251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1469, (-1)), 'str', "\n    Multi-dimensional greyscale closing.\n\n    A greyscale closing consists in the succession of a greyscale dilation,\n    and a greyscale erosion.\n\n    Parameters\n    ----------\n    input : array_like\n        Array over which the grayscale closing is to be computed.\n    size : tuple of ints\n        Shape of a flat and full structuring element used for the grayscale\n        closing. Optional if `footprint` or `structure` is provided.\n    footprint : array of ints, optional\n        Positions of non-infinite elements of a flat structuring element\n        used for the grayscale closing.\n    structure : array of ints, optional\n        Structuring element used for the grayscale closing. `structure`\n        may be a non-flat structuring element.\n    output : array, optional\n        An array used for storing the ouput of the closing may be provided.\n    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional\n        The `mode` parameter determines how the array borders are\n        handled, where `cval` is the value when mode is equal to\n        'constant'. Default is 'reflect'\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n    origin : scalar, optional\n        The `origin` parameter controls the placement of the filter.\n        Default 0\n\n    Returns\n    -------\n    grey_closing : ndarray\n        Result of the grayscale closing of `input` with `structure`.\n\n    See also\n    --------\n    binary_closing, grey_dilation, grey_erosion, grey_opening,\n    generate_binary_structure\n\n    Notes\n    -----\n    The action of a grayscale closing with a flat structuring element amounts\n    to smoothen deep local minima, whereas binary closing fills small holes.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.arange(36).reshape((6,6))\n    >>> a[3,3] = 0\n    >>> a\n    array([[ 0,  1,  2,  3,  4,  5],\n           [ 6,  7,  8,  9, 10, 11],\n           [12, 13, 14, 15, 16, 17],\n           [18, 19, 20,  0, 22, 23],\n           [24, 25, 26, 27, 28, 29],\n           [30, 31, 32, 33, 34, 35]])\n    >>> ndimage.grey_closing(a, size=(3,3))\n    array([[ 7,  7,  8,  9, 10, 11],\n           [ 7,  7,  8,  9, 10, 11],\n           [13, 13, 14, 15, 16, 17],\n           [19, 19, 20, 20, 22, 23],\n           [25, 25, 26, 27, 28, 29],\n           [31, 31, 32, 33, 34, 35]])\n    >>> # Note that the local minimum a[3,3] has disappeared\n\n    ")
    
    # Assigning a Call to a Name (line 1470):
    
    # Assigning a Call to a Name (line 1470):
    
    # Call to grey_dilation(...): (line 1470)
    # Processing the call arguments (line 1470)
    # Getting the type of 'input' (line 1470)
    input_125253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 24), 'input', False)
    # Getting the type of 'size' (line 1470)
    size_125254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 31), 'size', False)
    # Getting the type of 'footprint' (line 1470)
    footprint_125255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 37), 'footprint', False)
    # Getting the type of 'structure' (line 1470)
    structure_125256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 48), 'structure', False)
    # Getting the type of 'None' (line 1470)
    None_125257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 59), 'None', False)
    # Getting the type of 'mode' (line 1470)
    mode_125258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 65), 'mode', False)
    # Getting the type of 'cval' (line 1471)
    cval_125259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1471, 24), 'cval', False)
    # Getting the type of 'origin' (line 1471)
    origin_125260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1471, 30), 'origin', False)
    # Processing the call keyword arguments (line 1470)
    kwargs_125261 = {}
    # Getting the type of 'grey_dilation' (line 1470)
    grey_dilation_125252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 10), 'grey_dilation', False)
    # Calling grey_dilation(args, kwargs) (line 1470)
    grey_dilation_call_result_125262 = invoke(stypy.reporting.localization.Localization(__file__, 1470, 10), grey_dilation_125252, *[input_125253, size_125254, footprint_125255, structure_125256, None_125257, mode_125258, cval_125259, origin_125260], **kwargs_125261)
    
    # Assigning a type to the variable 'tmp' (line 1470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1470, 4), 'tmp', grey_dilation_call_result_125262)
    
    # Call to grey_erosion(...): (line 1472)
    # Processing the call arguments (line 1472)
    # Getting the type of 'tmp' (line 1472)
    tmp_125264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 24), 'tmp', False)
    # Getting the type of 'size' (line 1472)
    size_125265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 29), 'size', False)
    # Getting the type of 'footprint' (line 1472)
    footprint_125266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 35), 'footprint', False)
    # Getting the type of 'structure' (line 1472)
    structure_125267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 46), 'structure', False)
    # Getting the type of 'output' (line 1472)
    output_125268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 57), 'output', False)
    # Getting the type of 'mode' (line 1472)
    mode_125269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 65), 'mode', False)
    # Getting the type of 'cval' (line 1473)
    cval_125270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 24), 'cval', False)
    # Getting the type of 'origin' (line 1473)
    origin_125271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 30), 'origin', False)
    # Processing the call keyword arguments (line 1472)
    kwargs_125272 = {}
    # Getting the type of 'grey_erosion' (line 1472)
    grey_erosion_125263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 11), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1472)
    grey_erosion_call_result_125273 = invoke(stypy.reporting.localization.Localization(__file__, 1472, 11), grey_erosion_125263, *[tmp_125264, size_125265, footprint_125266, structure_125267, output_125268, mode_125269, cval_125270, origin_125271], **kwargs_125272)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1472, 4), 'stypy_return_type', grey_erosion_call_result_125273)
    
    # ################# End of 'grey_closing(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'grey_closing' in the type store
    # Getting the type of 'stypy_return_type' (line 1395)
    stypy_return_type_125274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1395, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125274)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'grey_closing'
    return stypy_return_type_125274

# Assigning a type to the variable 'grey_closing' (line 1395)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1395, 0), 'grey_closing', grey_closing)

@norecursion
def morphological_gradient(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1476)
    None_125275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1476, 39), 'None')
    # Getting the type of 'None' (line 1476)
    None_125276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1476, 55), 'None')
    # Getting the type of 'None' (line 1477)
    None_125277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1477, 34), 'None')
    # Getting the type of 'None' (line 1477)
    None_125278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1477, 47), 'None')
    str_125279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1477, 58), 'str', 'reflect')
    float_125280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1478, 29), 'float')
    int_125281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1478, 41), 'int')
    defaults = [None_125275, None_125276, None_125277, None_125278, str_125279, float_125280, int_125281]
    # Create a new context for function 'morphological_gradient'
    module_type_store = module_type_store.open_function_context('morphological_gradient', 1476, 0, False)
    
    # Passed parameters checking function
    morphological_gradient.stypy_localization = localization
    morphological_gradient.stypy_type_of_self = None
    morphological_gradient.stypy_type_store = module_type_store
    morphological_gradient.stypy_function_name = 'morphological_gradient'
    morphological_gradient.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin']
    morphological_gradient.stypy_varargs_param_name = None
    morphological_gradient.stypy_kwargs_param_name = None
    morphological_gradient.stypy_call_defaults = defaults
    morphological_gradient.stypy_call_varargs = varargs
    morphological_gradient.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'morphological_gradient', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'morphological_gradient', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'morphological_gradient(...)' code ##################

    str_125282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1578, (-1)), 'str', "\n    Multi-dimensional morphological gradient.\n\n    The morphological gradient is calculated as the difference between a\n    dilation and an erosion of the input with a given structuring element.\n\n    Parameters\n    ----------\n    input : array_like\n        Array over which to compute the morphlogical gradient.\n    size : tuple of ints\n        Shape of a flat and full structuring element used for the mathematical\n        morphology operations. Optional if `footprint` or `structure` is\n        provided. A larger `size` yields a more blurred gradient.\n    footprint : array of ints, optional\n        Positions of non-infinite elements of a flat structuring element\n        used for the morphology operations. Larger footprints\n        give a more blurred morphological gradient.\n    structure : array of ints, optional\n        Structuring element used for the morphology operations.\n        `structure` may be a non-flat structuring element.\n    output : array, optional\n        An array used for storing the ouput of the morphological gradient\n        may be provided.\n    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional\n        The `mode` parameter determines how the array borders are\n        handled, where `cval` is the value when mode is equal to\n        'constant'. Default is 'reflect'\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n    origin : scalar, optional\n        The `origin` parameter controls the placement of the filter.\n        Default 0\n\n    Returns\n    -------\n    morphological_gradient : ndarray\n        Morphological gradient of `input`.\n\n    See also\n    --------\n    grey_dilation, grey_erosion, ndimage.gaussian_gradient_magnitude\n\n    Notes\n    -----\n    For a flat structuring element, the morphological gradient\n    computed at a given point corresponds to the maximal difference\n    between elements of the input among the elements covered by the\n    structuring element centered on the point.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Mathematical_morphology\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((7,7), dtype=int)\n    >>> a[2:5, 2:5] = 1\n    >>> ndimage.morphological_gradient(a, size=(3,3))\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 1, 1, 0, 1, 1, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> # The morphological gradient is computed as the difference\n    >>> # between a dilation and an erosion\n    >>> ndimage.grey_dilation(a, size=(3,3)) -\\\n    ...  ndimage.grey_erosion(a, size=(3,3))\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 1, 1, 0, 1, 1, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 1, 1, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> a = np.zeros((7,7), dtype=int)\n    >>> a[2:5, 2:5] = 1\n    >>> a[4,4] = 2; a[2,3] = 3\n    >>> a\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 3, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 2, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> ndimage.morphological_gradient(a, size=(3,3))\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 3, 3, 3, 1, 0],\n           [0, 1, 3, 3, 3, 1, 0],\n           [0, 1, 3, 2, 3, 2, 0],\n           [0, 1, 1, 2, 2, 2, 0],\n           [0, 1, 1, 2, 2, 2, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n\n    ")
    
    # Assigning a Call to a Name (line 1579):
    
    # Assigning a Call to a Name (line 1579):
    
    # Call to grey_dilation(...): (line 1579)
    # Processing the call arguments (line 1579)
    # Getting the type of 'input' (line 1579)
    input_125284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 24), 'input', False)
    # Getting the type of 'size' (line 1579)
    size_125285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 31), 'size', False)
    # Getting the type of 'footprint' (line 1579)
    footprint_125286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 37), 'footprint', False)
    # Getting the type of 'structure' (line 1579)
    structure_125287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 48), 'structure', False)
    # Getting the type of 'None' (line 1579)
    None_125288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 59), 'None', False)
    # Getting the type of 'mode' (line 1579)
    mode_125289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 65), 'mode', False)
    # Getting the type of 'cval' (line 1580)
    cval_125290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1580, 24), 'cval', False)
    # Getting the type of 'origin' (line 1580)
    origin_125291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1580, 30), 'origin', False)
    # Processing the call keyword arguments (line 1579)
    kwargs_125292 = {}
    # Getting the type of 'grey_dilation' (line 1579)
    grey_dilation_125283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 10), 'grey_dilation', False)
    # Calling grey_dilation(args, kwargs) (line 1579)
    grey_dilation_call_result_125293 = invoke(stypy.reporting.localization.Localization(__file__, 1579, 10), grey_dilation_125283, *[input_125284, size_125285, footprint_125286, structure_125287, None_125288, mode_125289, cval_125290, origin_125291], **kwargs_125292)
    
    # Assigning a type to the variable 'tmp' (line 1579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1579, 4), 'tmp', grey_dilation_call_result_125293)
    
    
    # Call to isinstance(...): (line 1581)
    # Processing the call arguments (line 1581)
    # Getting the type of 'output' (line 1581)
    output_125295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 18), 'output', False)
    # Getting the type of 'numpy' (line 1581)
    numpy_125296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 26), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1581)
    ndarray_125297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1581, 26), numpy_125296, 'ndarray')
    # Processing the call keyword arguments (line 1581)
    kwargs_125298 = {}
    # Getting the type of 'isinstance' (line 1581)
    isinstance_125294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1581)
    isinstance_call_result_125299 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 7), isinstance_125294, *[output_125295, ndarray_125297], **kwargs_125298)
    
    # Testing the type of an if condition (line 1581)
    if_condition_125300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1581, 4), isinstance_call_result_125299)
    # Assigning a type to the variable 'if_condition_125300' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'if_condition_125300', if_condition_125300)
    # SSA begins for if statement (line 1581)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to grey_erosion(...): (line 1582)
    # Processing the call arguments (line 1582)
    # Getting the type of 'input' (line 1582)
    input_125302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 21), 'input', False)
    # Getting the type of 'size' (line 1582)
    size_125303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 28), 'size', False)
    # Getting the type of 'footprint' (line 1582)
    footprint_125304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 34), 'footprint', False)
    # Getting the type of 'structure' (line 1582)
    structure_125305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 45), 'structure', False)
    # Getting the type of 'output' (line 1582)
    output_125306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 56), 'output', False)
    # Getting the type of 'mode' (line 1582)
    mode_125307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 64), 'mode', False)
    # Getting the type of 'cval' (line 1583)
    cval_125308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 21), 'cval', False)
    # Getting the type of 'origin' (line 1583)
    origin_125309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 27), 'origin', False)
    # Processing the call keyword arguments (line 1582)
    kwargs_125310 = {}
    # Getting the type of 'grey_erosion' (line 1582)
    grey_erosion_125301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 8), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1582)
    grey_erosion_call_result_125311 = invoke(stypy.reporting.localization.Localization(__file__, 1582, 8), grey_erosion_125301, *[input_125302, size_125303, footprint_125304, structure_125305, output_125306, mode_125307, cval_125308, origin_125309], **kwargs_125310)
    
    
    # Call to subtract(...): (line 1584)
    # Processing the call arguments (line 1584)
    # Getting the type of 'tmp' (line 1584)
    tmp_125314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 30), 'tmp', False)
    # Getting the type of 'output' (line 1584)
    output_125315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 35), 'output', False)
    # Getting the type of 'output' (line 1584)
    output_125316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 43), 'output', False)
    # Processing the call keyword arguments (line 1584)
    kwargs_125317 = {}
    # Getting the type of 'numpy' (line 1584)
    numpy_125312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 15), 'numpy', False)
    # Obtaining the member 'subtract' of a type (line 1584)
    subtract_125313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1584, 15), numpy_125312, 'subtract')
    # Calling subtract(args, kwargs) (line 1584)
    subtract_call_result_125318 = invoke(stypy.reporting.localization.Localization(__file__, 1584, 15), subtract_125313, *[tmp_125314, output_125315, output_125316], **kwargs_125317)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1584, 8), 'stypy_return_type', subtract_call_result_125318)
    # SSA branch for the else part of an if statement (line 1581)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'tmp' (line 1586)
    tmp_125319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 16), 'tmp')
    
    # Call to grey_erosion(...): (line 1586)
    # Processing the call arguments (line 1586)
    # Getting the type of 'input' (line 1586)
    input_125321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 35), 'input', False)
    # Getting the type of 'size' (line 1586)
    size_125322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 42), 'size', False)
    # Getting the type of 'footprint' (line 1586)
    footprint_125323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 48), 'footprint', False)
    # Getting the type of 'structure' (line 1586)
    structure_125324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 59), 'structure', False)
    # Getting the type of 'None' (line 1587)
    None_125325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 35), 'None', False)
    # Getting the type of 'mode' (line 1587)
    mode_125326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 41), 'mode', False)
    # Getting the type of 'cval' (line 1587)
    cval_125327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 47), 'cval', False)
    # Getting the type of 'origin' (line 1587)
    origin_125328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 53), 'origin', False)
    # Processing the call keyword arguments (line 1586)
    kwargs_125329 = {}
    # Getting the type of 'grey_erosion' (line 1586)
    grey_erosion_125320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 22), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1586)
    grey_erosion_call_result_125330 = invoke(stypy.reporting.localization.Localization(__file__, 1586, 22), grey_erosion_125320, *[input_125321, size_125322, footprint_125323, structure_125324, None_125325, mode_125326, cval_125327, origin_125328], **kwargs_125329)
    
    # Applying the binary operator '-' (line 1586)
    result_sub_125331 = python_operator(stypy.reporting.localization.Localization(__file__, 1586, 16), '-', tmp_125319, grey_erosion_call_result_125330)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1586, 8), 'stypy_return_type', result_sub_125331)
    # SSA join for if statement (line 1581)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'morphological_gradient(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'morphological_gradient' in the type store
    # Getting the type of 'stypy_return_type' (line 1476)
    stypy_return_type_125332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1476, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'morphological_gradient'
    return stypy_return_type_125332

# Assigning a type to the variable 'morphological_gradient' (line 1476)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1476, 0), 'morphological_gradient', morphological_gradient)

@norecursion
def morphological_laplace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1590)
    None_125333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1590, 38), 'None')
    # Getting the type of 'None' (line 1590)
    None_125334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1590, 54), 'None')
    # Getting the type of 'None' (line 1591)
    None_125335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 36), 'None')
    # Getting the type of 'None' (line 1591)
    None_125336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 49), 'None')
    str_125337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, 31), 'str', 'reflect')
    float_125338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, 47), 'float')
    int_125339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, 59), 'int')
    defaults = [None_125333, None_125334, None_125335, None_125336, str_125337, float_125338, int_125339]
    # Create a new context for function 'morphological_laplace'
    module_type_store = module_type_store.open_function_context('morphological_laplace', 1590, 0, False)
    
    # Passed parameters checking function
    morphological_laplace.stypy_localization = localization
    morphological_laplace.stypy_type_of_self = None
    morphological_laplace.stypy_type_store = module_type_store
    morphological_laplace.stypy_function_name = 'morphological_laplace'
    morphological_laplace.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin']
    morphological_laplace.stypy_varargs_param_name = None
    morphological_laplace.stypy_kwargs_param_name = None
    morphological_laplace.stypy_call_defaults = defaults
    morphological_laplace.stypy_call_varargs = varargs
    morphological_laplace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'morphological_laplace', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'morphological_laplace', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'morphological_laplace(...)' code ##################

    str_125340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1623, (-1)), 'str', "\n    Multi-dimensional morphological laplace.\n\n    Parameters\n    ----------\n    input : array_like\n        Input.\n    size : int or sequence of ints, optional\n        See `structure`.\n    footprint : bool or ndarray, optional\n        See `structure`.\n    structure : structure, optional\n        Either `size`, `footprint`, or the `structure` must be provided.\n    output : ndarray, optional\n        An output array can optionally be provided.\n    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional\n        The mode parameter determines how the array borders are handled.\n        For 'constant' mode, values beyond borders are set to be `cval`.\n        Default is 'reflect'.\n    cval : scalar, optional\n        Value to fill past edges of input if mode is 'constant'.\n        Default is 0.0\n    origin : origin, optional\n        The origin parameter controls the placement of the filter.\n\n    Returns\n    -------\n    morphological_laplace : ndarray\n        Output\n\n    ")
    
    # Assigning a Call to a Name (line 1624):
    
    # Assigning a Call to a Name (line 1624):
    
    # Call to grey_dilation(...): (line 1624)
    # Processing the call arguments (line 1624)
    # Getting the type of 'input' (line 1624)
    input_125342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 25), 'input', False)
    # Getting the type of 'size' (line 1624)
    size_125343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 32), 'size', False)
    # Getting the type of 'footprint' (line 1624)
    footprint_125344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 38), 'footprint', False)
    # Getting the type of 'structure' (line 1624)
    structure_125345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 49), 'structure', False)
    # Getting the type of 'None' (line 1624)
    None_125346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 60), 'None', False)
    # Getting the type of 'mode' (line 1624)
    mode_125347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 66), 'mode', False)
    # Getting the type of 'cval' (line 1625)
    cval_125348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1625, 25), 'cval', False)
    # Getting the type of 'origin' (line 1625)
    origin_125349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1625, 31), 'origin', False)
    # Processing the call keyword arguments (line 1624)
    kwargs_125350 = {}
    # Getting the type of 'grey_dilation' (line 1624)
    grey_dilation_125341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 11), 'grey_dilation', False)
    # Calling grey_dilation(args, kwargs) (line 1624)
    grey_dilation_call_result_125351 = invoke(stypy.reporting.localization.Localization(__file__, 1624, 11), grey_dilation_125341, *[input_125342, size_125343, footprint_125344, structure_125345, None_125346, mode_125347, cval_125348, origin_125349], **kwargs_125350)
    
    # Assigning a type to the variable 'tmp1' (line 1624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1624, 4), 'tmp1', grey_dilation_call_result_125351)
    
    
    # Call to isinstance(...): (line 1626)
    # Processing the call arguments (line 1626)
    # Getting the type of 'output' (line 1626)
    output_125353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1626, 18), 'output', False)
    # Getting the type of 'numpy' (line 1626)
    numpy_125354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1626, 26), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1626)
    ndarray_125355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1626, 26), numpy_125354, 'ndarray')
    # Processing the call keyword arguments (line 1626)
    kwargs_125356 = {}
    # Getting the type of 'isinstance' (line 1626)
    isinstance_125352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1626, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1626)
    isinstance_call_result_125357 = invoke(stypy.reporting.localization.Localization(__file__, 1626, 7), isinstance_125352, *[output_125353, ndarray_125355], **kwargs_125356)
    
    # Testing the type of an if condition (line 1626)
    if_condition_125358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1626, 4), isinstance_call_result_125357)
    # Assigning a type to the variable 'if_condition_125358' (line 1626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1626, 4), 'if_condition_125358', if_condition_125358)
    # SSA begins for if statement (line 1626)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to grey_erosion(...): (line 1627)
    # Processing the call arguments (line 1627)
    # Getting the type of 'input' (line 1627)
    input_125360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 21), 'input', False)
    # Getting the type of 'size' (line 1627)
    size_125361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 28), 'size', False)
    # Getting the type of 'footprint' (line 1627)
    footprint_125362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 34), 'footprint', False)
    # Getting the type of 'structure' (line 1627)
    structure_125363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 45), 'structure', False)
    # Getting the type of 'output' (line 1627)
    output_125364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 56), 'output', False)
    # Getting the type of 'mode' (line 1627)
    mode_125365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 64), 'mode', False)
    # Getting the type of 'cval' (line 1628)
    cval_125366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1628, 21), 'cval', False)
    # Getting the type of 'origin' (line 1628)
    origin_125367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1628, 27), 'origin', False)
    # Processing the call keyword arguments (line 1627)
    kwargs_125368 = {}
    # Getting the type of 'grey_erosion' (line 1627)
    grey_erosion_125359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 8), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1627)
    grey_erosion_call_result_125369 = invoke(stypy.reporting.localization.Localization(__file__, 1627, 8), grey_erosion_125359, *[input_125360, size_125361, footprint_125362, structure_125363, output_125364, mode_125365, cval_125366, origin_125367], **kwargs_125368)
    
    
    # Call to add(...): (line 1629)
    # Processing the call arguments (line 1629)
    # Getting the type of 'tmp1' (line 1629)
    tmp1_125372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 18), 'tmp1', False)
    # Getting the type of 'output' (line 1629)
    output_125373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 24), 'output', False)
    # Getting the type of 'output' (line 1629)
    output_125374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 32), 'output', False)
    # Processing the call keyword arguments (line 1629)
    kwargs_125375 = {}
    # Getting the type of 'numpy' (line 1629)
    numpy_125370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 8), 'numpy', False)
    # Obtaining the member 'add' of a type (line 1629)
    add_125371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1629, 8), numpy_125370, 'add')
    # Calling add(args, kwargs) (line 1629)
    add_call_result_125376 = invoke(stypy.reporting.localization.Localization(__file__, 1629, 8), add_125371, *[tmp1_125372, output_125373, output_125374], **kwargs_125375)
    
    
    # Call to subtract(...): (line 1630)
    # Processing the call arguments (line 1630)
    # Getting the type of 'output' (line 1630)
    output_125379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 23), 'output', False)
    # Getting the type of 'input' (line 1630)
    input_125380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 31), 'input', False)
    # Getting the type of 'output' (line 1630)
    output_125381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 38), 'output', False)
    # Processing the call keyword arguments (line 1630)
    kwargs_125382 = {}
    # Getting the type of 'numpy' (line 1630)
    numpy_125377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 8), 'numpy', False)
    # Obtaining the member 'subtract' of a type (line 1630)
    subtract_125378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1630, 8), numpy_125377, 'subtract')
    # Calling subtract(args, kwargs) (line 1630)
    subtract_call_result_125383 = invoke(stypy.reporting.localization.Localization(__file__, 1630, 8), subtract_125378, *[output_125379, input_125380, output_125381], **kwargs_125382)
    
    
    # Call to subtract(...): (line 1631)
    # Processing the call arguments (line 1631)
    # Getting the type of 'output' (line 1631)
    output_125386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 30), 'output', False)
    # Getting the type of 'input' (line 1631)
    input_125387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 38), 'input', False)
    # Getting the type of 'output' (line 1631)
    output_125388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 45), 'output', False)
    # Processing the call keyword arguments (line 1631)
    kwargs_125389 = {}
    # Getting the type of 'numpy' (line 1631)
    numpy_125384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 15), 'numpy', False)
    # Obtaining the member 'subtract' of a type (line 1631)
    subtract_125385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1631, 15), numpy_125384, 'subtract')
    # Calling subtract(args, kwargs) (line 1631)
    subtract_call_result_125390 = invoke(stypy.reporting.localization.Localization(__file__, 1631, 15), subtract_125385, *[output_125386, input_125387, output_125388], **kwargs_125389)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1631, 8), 'stypy_return_type', subtract_call_result_125390)
    # SSA branch for the else part of an if statement (line 1626)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1633):
    
    # Assigning a Call to a Name (line 1633):
    
    # Call to grey_erosion(...): (line 1633)
    # Processing the call arguments (line 1633)
    # Getting the type of 'input' (line 1633)
    input_125392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 28), 'input', False)
    # Getting the type of 'size' (line 1633)
    size_125393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 35), 'size', False)
    # Getting the type of 'footprint' (line 1633)
    footprint_125394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 41), 'footprint', False)
    # Getting the type of 'structure' (line 1633)
    structure_125395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 52), 'structure', False)
    # Getting the type of 'None' (line 1633)
    None_125396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 63), 'None', False)
    # Getting the type of 'mode' (line 1633)
    mode_125397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 69), 'mode', False)
    # Getting the type of 'cval' (line 1634)
    cval_125398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 28), 'cval', False)
    # Getting the type of 'origin' (line 1634)
    origin_125399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 34), 'origin', False)
    # Processing the call keyword arguments (line 1633)
    kwargs_125400 = {}
    # Getting the type of 'grey_erosion' (line 1633)
    grey_erosion_125391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 15), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1633)
    grey_erosion_call_result_125401 = invoke(stypy.reporting.localization.Localization(__file__, 1633, 15), grey_erosion_125391, *[input_125392, size_125393, footprint_125394, structure_125395, None_125396, mode_125397, cval_125398, origin_125399], **kwargs_125400)
    
    # Assigning a type to the variable 'tmp2' (line 1633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1633, 8), 'tmp2', grey_erosion_call_result_125401)
    
    # Call to add(...): (line 1635)
    # Processing the call arguments (line 1635)
    # Getting the type of 'tmp1' (line 1635)
    tmp1_125404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 18), 'tmp1', False)
    # Getting the type of 'tmp2' (line 1635)
    tmp2_125405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 24), 'tmp2', False)
    # Getting the type of 'tmp2' (line 1635)
    tmp2_125406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 30), 'tmp2', False)
    # Processing the call keyword arguments (line 1635)
    kwargs_125407 = {}
    # Getting the type of 'numpy' (line 1635)
    numpy_125402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 8), 'numpy', False)
    # Obtaining the member 'add' of a type (line 1635)
    add_125403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1635, 8), numpy_125402, 'add')
    # Calling add(args, kwargs) (line 1635)
    add_call_result_125408 = invoke(stypy.reporting.localization.Localization(__file__, 1635, 8), add_125403, *[tmp1_125404, tmp2_125405, tmp2_125406], **kwargs_125407)
    
    
    # Call to subtract(...): (line 1636)
    # Processing the call arguments (line 1636)
    # Getting the type of 'tmp2' (line 1636)
    tmp2_125411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 23), 'tmp2', False)
    # Getting the type of 'input' (line 1636)
    input_125412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 29), 'input', False)
    # Getting the type of 'tmp2' (line 1636)
    tmp2_125413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 36), 'tmp2', False)
    # Processing the call keyword arguments (line 1636)
    kwargs_125414 = {}
    # Getting the type of 'numpy' (line 1636)
    numpy_125409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 8), 'numpy', False)
    # Obtaining the member 'subtract' of a type (line 1636)
    subtract_125410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1636, 8), numpy_125409, 'subtract')
    # Calling subtract(args, kwargs) (line 1636)
    subtract_call_result_125415 = invoke(stypy.reporting.localization.Localization(__file__, 1636, 8), subtract_125410, *[tmp2_125411, input_125412, tmp2_125413], **kwargs_125414)
    
    
    # Call to subtract(...): (line 1637)
    # Processing the call arguments (line 1637)
    # Getting the type of 'tmp2' (line 1637)
    tmp2_125418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 23), 'tmp2', False)
    # Getting the type of 'input' (line 1637)
    input_125419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 29), 'input', False)
    # Getting the type of 'tmp2' (line 1637)
    tmp2_125420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 36), 'tmp2', False)
    # Processing the call keyword arguments (line 1637)
    kwargs_125421 = {}
    # Getting the type of 'numpy' (line 1637)
    numpy_125416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 8), 'numpy', False)
    # Obtaining the member 'subtract' of a type (line 1637)
    subtract_125417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1637, 8), numpy_125416, 'subtract')
    # Calling subtract(args, kwargs) (line 1637)
    subtract_call_result_125422 = invoke(stypy.reporting.localization.Localization(__file__, 1637, 8), subtract_125417, *[tmp2_125418, input_125419, tmp2_125420], **kwargs_125421)
    
    # Getting the type of 'tmp2' (line 1638)
    tmp2_125423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1638, 15), 'tmp2')
    # Assigning a type to the variable 'stypy_return_type' (line 1638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1638, 8), 'stypy_return_type', tmp2_125423)
    # SSA join for if statement (line 1626)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'morphological_laplace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'morphological_laplace' in the type store
    # Getting the type of 'stypy_return_type' (line 1590)
    stypy_return_type_125424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1590, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125424)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'morphological_laplace'
    return stypy_return_type_125424

# Assigning a type to the variable 'morphological_laplace' (line 1590)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1590, 0), 'morphological_laplace', morphological_laplace)

@norecursion
def white_tophat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1641)
    None_125425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1641, 29), 'None')
    # Getting the type of 'None' (line 1641)
    None_125426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1641, 45), 'None')
    # Getting the type of 'None' (line 1641)
    None_125427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1641, 61), 'None')
    # Getting the type of 'None' (line 1642)
    None_125428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1642, 24), 'None')
    str_125429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1642, 35), 'str', 'reflect')
    float_125430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1642, 51), 'float')
    int_125431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1642, 63), 'int')
    defaults = [None_125425, None_125426, None_125427, None_125428, str_125429, float_125430, int_125431]
    # Create a new context for function 'white_tophat'
    module_type_store = module_type_store.open_function_context('white_tophat', 1641, 0, False)
    
    # Passed parameters checking function
    white_tophat.stypy_localization = localization
    white_tophat.stypy_type_of_self = None
    white_tophat.stypy_type_store = module_type_store
    white_tophat.stypy_function_name = 'white_tophat'
    white_tophat.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin']
    white_tophat.stypy_varargs_param_name = None
    white_tophat.stypy_kwargs_param_name = None
    white_tophat.stypy_call_defaults = defaults
    white_tophat.stypy_call_varargs = varargs
    white_tophat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'white_tophat', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'white_tophat', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'white_tophat(...)' code ##################

    str_125432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1681, (-1)), 'str', "\n    Multi-dimensional white tophat filter.\n\n    Parameters\n    ----------\n    input : array_like\n        Input.\n    size : tuple of ints\n        Shape of a flat and full structuring element used for the filter.\n        Optional if `footprint` or `structure` is provided.\n    footprint : array of ints, optional\n        Positions of elements of a flat structuring element\n        used for the white tophat filter.\n    structure : array of ints, optional\n        Structuring element used for the filter. `structure`\n        may be a non-flat structuring element.\n    output : array, optional\n        An array used for storing the output of the filter may be provided.\n    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional\n        The `mode` parameter determines how the array borders are\n        handled, where `cval` is the value when mode is equal to\n        'constant'. Default is 'reflect'\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'.\n        Default is 0.0.\n    origin : scalar, optional\n        The `origin` parameter controls the placement of the filter.\n        Default is 0.\n\n    Returns\n    -------\n    output : ndarray\n        Result of the filter of `input` with `structure`.\n\n    See also\n    --------\n    black_tophat\n\n    ")
    
    # Assigning a Call to a Name (line 1682):
    
    # Assigning a Call to a Name (line 1682):
    
    # Call to grey_erosion(...): (line 1682)
    # Processing the call arguments (line 1682)
    # Getting the type of 'input' (line 1682)
    input_125434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 23), 'input', False)
    # Getting the type of 'size' (line 1682)
    size_125435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 30), 'size', False)
    # Getting the type of 'footprint' (line 1682)
    footprint_125436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 36), 'footprint', False)
    # Getting the type of 'structure' (line 1682)
    structure_125437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 47), 'structure', False)
    # Getting the type of 'None' (line 1682)
    None_125438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 58), 'None', False)
    # Getting the type of 'mode' (line 1682)
    mode_125439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 64), 'mode', False)
    # Getting the type of 'cval' (line 1683)
    cval_125440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1683, 23), 'cval', False)
    # Getting the type of 'origin' (line 1683)
    origin_125441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1683, 29), 'origin', False)
    # Processing the call keyword arguments (line 1682)
    kwargs_125442 = {}
    # Getting the type of 'grey_erosion' (line 1682)
    grey_erosion_125433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 10), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1682)
    grey_erosion_call_result_125443 = invoke(stypy.reporting.localization.Localization(__file__, 1682, 10), grey_erosion_125433, *[input_125434, size_125435, footprint_125436, structure_125437, None_125438, mode_125439, cval_125440, origin_125441], **kwargs_125442)
    
    # Assigning a type to the variable 'tmp' (line 1682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1682, 4), 'tmp', grey_erosion_call_result_125443)
    
    
    # Call to isinstance(...): (line 1684)
    # Processing the call arguments (line 1684)
    # Getting the type of 'output' (line 1684)
    output_125445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 18), 'output', False)
    # Getting the type of 'numpy' (line 1684)
    numpy_125446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 26), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1684)
    ndarray_125447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1684, 26), numpy_125446, 'ndarray')
    # Processing the call keyword arguments (line 1684)
    kwargs_125448 = {}
    # Getting the type of 'isinstance' (line 1684)
    isinstance_125444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1684)
    isinstance_call_result_125449 = invoke(stypy.reporting.localization.Localization(__file__, 1684, 7), isinstance_125444, *[output_125445, ndarray_125447], **kwargs_125448)
    
    # Testing the type of an if condition (line 1684)
    if_condition_125450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1684, 4), isinstance_call_result_125449)
    # Assigning a type to the variable 'if_condition_125450' (line 1684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1684, 4), 'if_condition_125450', if_condition_125450)
    # SSA begins for if statement (line 1684)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to grey_dilation(...): (line 1685)
    # Processing the call arguments (line 1685)
    # Getting the type of 'tmp' (line 1685)
    tmp_125452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 22), 'tmp', False)
    # Getting the type of 'size' (line 1685)
    size_125453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 27), 'size', False)
    # Getting the type of 'footprint' (line 1685)
    footprint_125454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 33), 'footprint', False)
    # Getting the type of 'structure' (line 1685)
    structure_125455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 44), 'structure', False)
    # Getting the type of 'output' (line 1685)
    output_125456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 55), 'output', False)
    # Getting the type of 'mode' (line 1685)
    mode_125457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 63), 'mode', False)
    # Getting the type of 'cval' (line 1685)
    cval_125458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 69), 'cval', False)
    # Getting the type of 'origin' (line 1686)
    origin_125459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 22), 'origin', False)
    # Processing the call keyword arguments (line 1685)
    kwargs_125460 = {}
    # Getting the type of 'grey_dilation' (line 1685)
    grey_dilation_125451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1685, 8), 'grey_dilation', False)
    # Calling grey_dilation(args, kwargs) (line 1685)
    grey_dilation_call_result_125461 = invoke(stypy.reporting.localization.Localization(__file__, 1685, 8), grey_dilation_125451, *[tmp_125452, size_125453, footprint_125454, structure_125455, output_125456, mode_125457, cval_125458, origin_125459], **kwargs_125460)
    
    
    # Call to subtract(...): (line 1687)
    # Processing the call arguments (line 1687)
    # Getting the type of 'input' (line 1687)
    input_125464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1687, 30), 'input', False)
    # Getting the type of 'output' (line 1687)
    output_125465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1687, 37), 'output', False)
    # Getting the type of 'output' (line 1687)
    output_125466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1687, 45), 'output', False)
    # Processing the call keyword arguments (line 1687)
    kwargs_125467 = {}
    # Getting the type of 'numpy' (line 1687)
    numpy_125462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1687, 15), 'numpy', False)
    # Obtaining the member 'subtract' of a type (line 1687)
    subtract_125463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1687, 15), numpy_125462, 'subtract')
    # Calling subtract(args, kwargs) (line 1687)
    subtract_call_result_125468 = invoke(stypy.reporting.localization.Localization(__file__, 1687, 15), subtract_125463, *[input_125464, output_125465, output_125466], **kwargs_125467)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1687, 8), 'stypy_return_type', subtract_call_result_125468)
    # SSA branch for the else part of an if statement (line 1684)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1689):
    
    # Assigning a Call to a Name (line 1689):
    
    # Call to grey_dilation(...): (line 1689)
    # Processing the call arguments (line 1689)
    # Getting the type of 'tmp' (line 1689)
    tmp_125470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1689, 28), 'tmp', False)
    # Getting the type of 'size' (line 1689)
    size_125471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1689, 33), 'size', False)
    # Getting the type of 'footprint' (line 1689)
    footprint_125472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1689, 39), 'footprint', False)
    # Getting the type of 'structure' (line 1689)
    structure_125473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1689, 50), 'structure', False)
    # Getting the type of 'None' (line 1689)
    None_125474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1689, 61), 'None', False)
    # Getting the type of 'mode' (line 1689)
    mode_125475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1689, 67), 'mode', False)
    # Getting the type of 'cval' (line 1690)
    cval_125476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1690, 28), 'cval', False)
    # Getting the type of 'origin' (line 1690)
    origin_125477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1690, 34), 'origin', False)
    # Processing the call keyword arguments (line 1689)
    kwargs_125478 = {}
    # Getting the type of 'grey_dilation' (line 1689)
    grey_dilation_125469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1689, 14), 'grey_dilation', False)
    # Calling grey_dilation(args, kwargs) (line 1689)
    grey_dilation_call_result_125479 = invoke(stypy.reporting.localization.Localization(__file__, 1689, 14), grey_dilation_125469, *[tmp_125470, size_125471, footprint_125472, structure_125473, None_125474, mode_125475, cval_125476, origin_125477], **kwargs_125478)
    
    # Assigning a type to the variable 'tmp' (line 1689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1689, 8), 'tmp', grey_dilation_call_result_125479)
    # Getting the type of 'input' (line 1691)
    input_125480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1691, 15), 'input')
    # Getting the type of 'tmp' (line 1691)
    tmp_125481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1691, 23), 'tmp')
    # Applying the binary operator '-' (line 1691)
    result_sub_125482 = python_operator(stypy.reporting.localization.Localization(__file__, 1691, 15), '-', input_125480, tmp_125481)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1691, 8), 'stypy_return_type', result_sub_125482)
    # SSA join for if statement (line 1684)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'white_tophat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'white_tophat' in the type store
    # Getting the type of 'stypy_return_type' (line 1641)
    stypy_return_type_125483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1641, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125483)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'white_tophat'
    return stypy_return_type_125483

# Assigning a type to the variable 'white_tophat' (line 1641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1641, 0), 'white_tophat', white_tophat)

@norecursion
def black_tophat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1694)
    None_125484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1694, 29), 'None')
    # Getting the type of 'None' (line 1694)
    None_125485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1694, 45), 'None')
    # Getting the type of 'None' (line 1695)
    None_125486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1695, 27), 'None')
    # Getting the type of 'None' (line 1695)
    None_125487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1695, 40), 'None')
    str_125488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1695, 51), 'str', 'reflect')
    float_125489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1696, 22), 'float')
    int_125490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1696, 34), 'int')
    defaults = [None_125484, None_125485, None_125486, None_125487, str_125488, float_125489, int_125490]
    # Create a new context for function 'black_tophat'
    module_type_store = module_type_store.open_function_context('black_tophat', 1694, 0, False)
    
    # Passed parameters checking function
    black_tophat.stypy_localization = localization
    black_tophat.stypy_type_of_self = None
    black_tophat.stypy_type_store = module_type_store
    black_tophat.stypy_function_name = 'black_tophat'
    black_tophat.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin']
    black_tophat.stypy_varargs_param_name = None
    black_tophat.stypy_kwargs_param_name = None
    black_tophat.stypy_call_defaults = defaults
    black_tophat.stypy_call_varargs = varargs
    black_tophat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'black_tophat', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'black_tophat', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'black_tophat(...)' code ##################

    str_125491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1735, (-1)), 'str', "\n    Multi-dimensional black tophat filter.\n\n    Parameters\n    ----------\n    input : array_like\n        Input.\n    size : tuple of ints, optional\n        Shape of a flat and full structuring element used for the filter.\n        Optional if `footprint` or `structure` is provided.\n    footprint : array of ints, optional\n        Positions of non-infinite elements of a flat structuring element\n        used for the black tophat filter.\n    structure : array of ints, optional\n        Structuring element used for the filter. `structure`\n        may be a non-flat structuring element.\n    output : array, optional\n        An array used for storing the output of the filter may be provided.\n    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional\n        The `mode` parameter determines how the array borders are\n        handled, where `cval` is the value when mode is equal to\n        'constant'. Default is 'reflect'\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n    origin : scalar, optional\n        The `origin` parameter controls the placement of the filter.\n        Default 0\n\n    Returns\n    -------\n    black_tophat : ndarray\n        Result of the filter of `input` with `structure`.\n\n    See also\n    --------\n    white_tophat, grey_opening, grey_closing\n\n    ")
    
    # Assigning a Call to a Name (line 1736):
    
    # Assigning a Call to a Name (line 1736):
    
    # Call to grey_dilation(...): (line 1736)
    # Processing the call arguments (line 1736)
    # Getting the type of 'input' (line 1736)
    input_125493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 24), 'input', False)
    # Getting the type of 'size' (line 1736)
    size_125494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 31), 'size', False)
    # Getting the type of 'footprint' (line 1736)
    footprint_125495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 37), 'footprint', False)
    # Getting the type of 'structure' (line 1736)
    structure_125496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 48), 'structure', False)
    # Getting the type of 'None' (line 1736)
    None_125497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 59), 'None', False)
    # Getting the type of 'mode' (line 1736)
    mode_125498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 65), 'mode', False)
    # Getting the type of 'cval' (line 1737)
    cval_125499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1737, 24), 'cval', False)
    # Getting the type of 'origin' (line 1737)
    origin_125500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1737, 30), 'origin', False)
    # Processing the call keyword arguments (line 1736)
    kwargs_125501 = {}
    # Getting the type of 'grey_dilation' (line 1736)
    grey_dilation_125492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 10), 'grey_dilation', False)
    # Calling grey_dilation(args, kwargs) (line 1736)
    grey_dilation_call_result_125502 = invoke(stypy.reporting.localization.Localization(__file__, 1736, 10), grey_dilation_125492, *[input_125493, size_125494, footprint_125495, structure_125496, None_125497, mode_125498, cval_125499, origin_125500], **kwargs_125501)
    
    # Assigning a type to the variable 'tmp' (line 1736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1736, 4), 'tmp', grey_dilation_call_result_125502)
    
    
    # Call to isinstance(...): (line 1738)
    # Processing the call arguments (line 1738)
    # Getting the type of 'output' (line 1738)
    output_125504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 18), 'output', False)
    # Getting the type of 'numpy' (line 1738)
    numpy_125505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 26), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1738)
    ndarray_125506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1738, 26), numpy_125505, 'ndarray')
    # Processing the call keyword arguments (line 1738)
    kwargs_125507 = {}
    # Getting the type of 'isinstance' (line 1738)
    isinstance_125503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1738)
    isinstance_call_result_125508 = invoke(stypy.reporting.localization.Localization(__file__, 1738, 7), isinstance_125503, *[output_125504, ndarray_125506], **kwargs_125507)
    
    # Testing the type of an if condition (line 1738)
    if_condition_125509 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1738, 4), isinstance_call_result_125508)
    # Assigning a type to the variable 'if_condition_125509' (line 1738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1738, 4), 'if_condition_125509', if_condition_125509)
    # SSA begins for if statement (line 1738)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to grey_erosion(...): (line 1739)
    # Processing the call arguments (line 1739)
    # Getting the type of 'tmp' (line 1739)
    tmp_125511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 21), 'tmp', False)
    # Getting the type of 'size' (line 1739)
    size_125512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 26), 'size', False)
    # Getting the type of 'footprint' (line 1739)
    footprint_125513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 32), 'footprint', False)
    # Getting the type of 'structure' (line 1739)
    structure_125514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 43), 'structure', False)
    # Getting the type of 'output' (line 1739)
    output_125515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 54), 'output', False)
    # Getting the type of 'mode' (line 1739)
    mode_125516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 62), 'mode', False)
    # Getting the type of 'cval' (line 1739)
    cval_125517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 68), 'cval', False)
    # Getting the type of 'origin' (line 1740)
    origin_125518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 21), 'origin', False)
    # Processing the call keyword arguments (line 1739)
    kwargs_125519 = {}
    # Getting the type of 'grey_erosion' (line 1739)
    grey_erosion_125510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 8), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1739)
    grey_erosion_call_result_125520 = invoke(stypy.reporting.localization.Localization(__file__, 1739, 8), grey_erosion_125510, *[tmp_125511, size_125512, footprint_125513, structure_125514, output_125515, mode_125516, cval_125517, origin_125518], **kwargs_125519)
    
    
    # Call to subtract(...): (line 1741)
    # Processing the call arguments (line 1741)
    # Getting the type of 'output' (line 1741)
    output_125523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1741, 30), 'output', False)
    # Getting the type of 'input' (line 1741)
    input_125524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1741, 38), 'input', False)
    # Getting the type of 'output' (line 1741)
    output_125525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1741, 45), 'output', False)
    # Processing the call keyword arguments (line 1741)
    kwargs_125526 = {}
    # Getting the type of 'numpy' (line 1741)
    numpy_125521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1741, 15), 'numpy', False)
    # Obtaining the member 'subtract' of a type (line 1741)
    subtract_125522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1741, 15), numpy_125521, 'subtract')
    # Calling subtract(args, kwargs) (line 1741)
    subtract_call_result_125527 = invoke(stypy.reporting.localization.Localization(__file__, 1741, 15), subtract_125522, *[output_125523, input_125524, output_125525], **kwargs_125526)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1741, 8), 'stypy_return_type', subtract_call_result_125527)
    # SSA branch for the else part of an if statement (line 1738)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1743):
    
    # Assigning a Call to a Name (line 1743):
    
    # Call to grey_erosion(...): (line 1743)
    # Processing the call arguments (line 1743)
    # Getting the type of 'tmp' (line 1743)
    tmp_125529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 27), 'tmp', False)
    # Getting the type of 'size' (line 1743)
    size_125530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 32), 'size', False)
    # Getting the type of 'footprint' (line 1743)
    footprint_125531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 38), 'footprint', False)
    # Getting the type of 'structure' (line 1743)
    structure_125532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 49), 'structure', False)
    # Getting the type of 'None' (line 1743)
    None_125533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 60), 'None', False)
    # Getting the type of 'mode' (line 1743)
    mode_125534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 66), 'mode', False)
    # Getting the type of 'cval' (line 1744)
    cval_125535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1744, 27), 'cval', False)
    # Getting the type of 'origin' (line 1744)
    origin_125536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1744, 33), 'origin', False)
    # Processing the call keyword arguments (line 1743)
    kwargs_125537 = {}
    # Getting the type of 'grey_erosion' (line 1743)
    grey_erosion_125528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 14), 'grey_erosion', False)
    # Calling grey_erosion(args, kwargs) (line 1743)
    grey_erosion_call_result_125538 = invoke(stypy.reporting.localization.Localization(__file__, 1743, 14), grey_erosion_125528, *[tmp_125529, size_125530, footprint_125531, structure_125532, None_125533, mode_125534, cval_125535, origin_125536], **kwargs_125537)
    
    # Assigning a type to the variable 'tmp' (line 1743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1743, 8), 'tmp', grey_erosion_call_result_125538)
    # Getting the type of 'tmp' (line 1745)
    tmp_125539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1745, 15), 'tmp')
    # Getting the type of 'input' (line 1745)
    input_125540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1745, 21), 'input')
    # Applying the binary operator '-' (line 1745)
    result_sub_125541 = python_operator(stypy.reporting.localization.Localization(__file__, 1745, 15), '-', tmp_125539, input_125540)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1745, 8), 'stypy_return_type', result_sub_125541)
    # SSA join for if statement (line 1738)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'black_tophat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'black_tophat' in the type store
    # Getting the type of 'stypy_return_type' (line 1694)
    stypy_return_type_125542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1694, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125542)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'black_tophat'
    return stypy_return_type_125542

# Assigning a type to the variable 'black_tophat' (line 1694)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1694, 0), 'black_tophat', black_tophat)

@norecursion
def distance_transform_bf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_125543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1748, 40), 'str', 'euclidean')
    # Getting the type of 'None' (line 1748)
    None_125544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1748, 62), 'None')
    # Getting the type of 'True' (line 1749)
    True_125545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1749, 43), 'True')
    # Getting the type of 'False' (line 1749)
    False_125546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1749, 64), 'False')
    # Getting the type of 'None' (line 1750)
    None_125547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1750, 36), 'None')
    # Getting the type of 'None' (line 1750)
    None_125548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1750, 50), 'None')
    defaults = [str_125543, None_125544, True_125545, False_125546, None_125547, None_125548]
    # Create a new context for function 'distance_transform_bf'
    module_type_store = module_type_store.open_function_context('distance_transform_bf', 1748, 0, False)
    
    # Passed parameters checking function
    distance_transform_bf.stypy_localization = localization
    distance_transform_bf.stypy_type_of_self = None
    distance_transform_bf.stypy_type_store = module_type_store
    distance_transform_bf.stypy_function_name = 'distance_transform_bf'
    distance_transform_bf.stypy_param_names_list = ['input', 'metric', 'sampling', 'return_distances', 'return_indices', 'distances', 'indices']
    distance_transform_bf.stypy_varargs_param_name = None
    distance_transform_bf.stypy_kwargs_param_name = None
    distance_transform_bf.stypy_call_defaults = defaults
    distance_transform_bf.stypy_call_varargs = varargs
    distance_transform_bf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'distance_transform_bf', ['input', 'metric', 'sampling', 'return_distances', 'return_indices', 'distances', 'indices'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'distance_transform_bf', localization, ['input', 'metric', 'sampling', 'return_distances', 'return_indices', 'distances', 'indices'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'distance_transform_bf(...)' code ##################

    str_125549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1806, (-1)), 'str', "\n    Distance transform function by a brute force algorithm.\n\n    This function calculates the distance transform of the `input`, by\n    replacing each foreground (non-zero) element, with its\n    shortest distance to the background (any zero-valued element).\n\n    In addition to the distance transform, the feature transform can\n    be calculated. In this case the index of the closest background\n    element is returned along the first axis of the result.\n\n    Parameters\n    ----------\n    input : array_like\n        Input\n    metric : str, optional\n        Three types of distance metric are supported: 'euclidean', 'taxicab'\n        and 'chessboard'.\n    sampling : {int, sequence of ints}, optional\n        This parameter is only used in the case of the euclidean `metric`\n        distance transform.\n\n        The sampling along each axis can be given by the `sampling` parameter\n        which should be a sequence of length equal to the input rank, or a\n        single number in which the `sampling` is assumed to be equal along all\n        axes.\n    return_distances : bool, optional\n        The `return_distances` flag can be used to indicate if the distance\n        transform is returned.\n\n        The default is True.\n    return_indices : bool, optional\n        The `return_indices` flags can be used to indicate if the feature\n        transform is returned.\n\n        The default is False.\n    distances : float64 ndarray, optional\n        Optional output array to hold distances (if `return_distances` is\n        True).\n    indices : int64 ndarray, optional\n        Optional output array to hold indices (if `return_indices` is True).\n\n    Returns\n    -------\n    distances : ndarray\n        Distance array if `return_distances` is True.\n    indices : ndarray\n        Indices array if `return_indices` is True.\n\n    Notes\n    -----\n    This function employs a slow brute force algorithm, see also the\n    function distance_transform_cdt for more efficient taxicab and\n    chessboard algorithms.\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'return_distances' (line 1807)
    return_distances_125550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1807, 12), 'return_distances')
    # Applying the 'not' unary operator (line 1807)
    result_not__125551 = python_operator(stypy.reporting.localization.Localization(__file__, 1807, 8), 'not', return_distances_125550)
    
    
    # Getting the type of 'return_indices' (line 1807)
    return_indices_125552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1807, 39), 'return_indices')
    # Applying the 'not' unary operator (line 1807)
    result_not__125553 = python_operator(stypy.reporting.localization.Localization(__file__, 1807, 35), 'not', return_indices_125552)
    
    # Applying the binary operator 'and' (line 1807)
    result_and_keyword_125554 = python_operator(stypy.reporting.localization.Localization(__file__, 1807, 7), 'and', result_not__125551, result_not__125553)
    
    # Testing the type of an if condition (line 1807)
    if_condition_125555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1807, 4), result_and_keyword_125554)
    # Assigning a type to the variable 'if_condition_125555' (line 1807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1807, 4), 'if_condition_125555', if_condition_125555)
    # SSA begins for if statement (line 1807)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1808):
    
    # Assigning a Str to a Name (line 1808):
    str_125556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1808, 14), 'str', 'at least one of distances/indices must be specified')
    # Assigning a type to the variable 'msg' (line 1808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1808, 8), 'msg', str_125556)
    
    # Call to RuntimeError(...): (line 1809)
    # Processing the call arguments (line 1809)
    # Getting the type of 'msg' (line 1809)
    msg_125558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1809, 27), 'msg', False)
    # Processing the call keyword arguments (line 1809)
    kwargs_125559 = {}
    # Getting the type of 'RuntimeError' (line 1809)
    RuntimeError_125557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1809, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1809)
    RuntimeError_call_result_125560 = invoke(stypy.reporting.localization.Localization(__file__, 1809, 14), RuntimeError_125557, *[msg_125558], **kwargs_125559)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1809, 8), RuntimeError_call_result_125560, 'raise parameter', BaseException)
    # SSA join for if statement (line 1807)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Compare to a Name (line 1811):
    
    # Assigning a Compare to a Name (line 1811):
    
    
    # Call to asarray(...): (line 1811)
    # Processing the call arguments (line 1811)
    # Getting the type of 'input' (line 1811)
    input_125563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1811, 25), 'input', False)
    # Processing the call keyword arguments (line 1811)
    kwargs_125564 = {}
    # Getting the type of 'numpy' (line 1811)
    numpy_125561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1811, 11), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1811)
    asarray_125562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1811, 11), numpy_125561, 'asarray')
    # Calling asarray(args, kwargs) (line 1811)
    asarray_call_result_125565 = invoke(stypy.reporting.localization.Localization(__file__, 1811, 11), asarray_125562, *[input_125563], **kwargs_125564)
    
    int_125566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1811, 35), 'int')
    # Applying the binary operator '!=' (line 1811)
    result_ne_125567 = python_operator(stypy.reporting.localization.Localization(__file__, 1811, 11), '!=', asarray_call_result_125565, int_125566)
    
    # Assigning a type to the variable 'tmp1' (line 1811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1811, 4), 'tmp1', result_ne_125567)
    
    # Assigning a Call to a Name (line 1812):
    
    # Assigning a Call to a Name (line 1812):
    
    # Call to generate_binary_structure(...): (line 1812)
    # Processing the call arguments (line 1812)
    # Getting the type of 'tmp1' (line 1812)
    tmp1_125569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1812, 39), 'tmp1', False)
    # Obtaining the member 'ndim' of a type (line 1812)
    ndim_125570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1812, 39), tmp1_125569, 'ndim')
    # Getting the type of 'tmp1' (line 1812)
    tmp1_125571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1812, 50), 'tmp1', False)
    # Obtaining the member 'ndim' of a type (line 1812)
    ndim_125572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1812, 50), tmp1_125571, 'ndim')
    # Processing the call keyword arguments (line 1812)
    kwargs_125573 = {}
    # Getting the type of 'generate_binary_structure' (line 1812)
    generate_binary_structure_125568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1812, 13), 'generate_binary_structure', False)
    # Calling generate_binary_structure(args, kwargs) (line 1812)
    generate_binary_structure_call_result_125574 = invoke(stypy.reporting.localization.Localization(__file__, 1812, 13), generate_binary_structure_125568, *[ndim_125570, ndim_125572], **kwargs_125573)
    
    # Assigning a type to the variable 'struct' (line 1812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1812, 4), 'struct', generate_binary_structure_call_result_125574)
    
    # Assigning a Call to a Name (line 1813):
    
    # Assigning a Call to a Name (line 1813):
    
    # Call to binary_dilation(...): (line 1813)
    # Processing the call arguments (line 1813)
    # Getting the type of 'tmp1' (line 1813)
    tmp1_125576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1813, 27), 'tmp1', False)
    # Getting the type of 'struct' (line 1813)
    struct_125577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1813, 33), 'struct', False)
    # Processing the call keyword arguments (line 1813)
    kwargs_125578 = {}
    # Getting the type of 'binary_dilation' (line 1813)
    binary_dilation_125575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1813, 11), 'binary_dilation', False)
    # Calling binary_dilation(args, kwargs) (line 1813)
    binary_dilation_call_result_125579 = invoke(stypy.reporting.localization.Localization(__file__, 1813, 11), binary_dilation_125575, *[tmp1_125576, struct_125577], **kwargs_125578)
    
    # Assigning a type to the variable 'tmp2' (line 1813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1813, 4), 'tmp2', binary_dilation_call_result_125579)
    
    # Assigning a Call to a Name (line 1814):
    
    # Assigning a Call to a Name (line 1814):
    
    # Call to logical_xor(...): (line 1814)
    # Processing the call arguments (line 1814)
    # Getting the type of 'tmp1' (line 1814)
    tmp1_125582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1814, 29), 'tmp1', False)
    # Getting the type of 'tmp2' (line 1814)
    tmp2_125583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1814, 35), 'tmp2', False)
    # Processing the call keyword arguments (line 1814)
    kwargs_125584 = {}
    # Getting the type of 'numpy' (line 1814)
    numpy_125580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1814, 11), 'numpy', False)
    # Obtaining the member 'logical_xor' of a type (line 1814)
    logical_xor_125581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1814, 11), numpy_125580, 'logical_xor')
    # Calling logical_xor(args, kwargs) (line 1814)
    logical_xor_call_result_125585 = invoke(stypy.reporting.localization.Localization(__file__, 1814, 11), logical_xor_125581, *[tmp1_125582, tmp2_125583], **kwargs_125584)
    
    # Assigning a type to the variable 'tmp2' (line 1814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1814, 4), 'tmp2', logical_xor_call_result_125585)
    
    # Assigning a BinOp to a Name (line 1815):
    
    # Assigning a BinOp to a Name (line 1815):
    
    # Call to astype(...): (line 1815)
    # Processing the call arguments (line 1815)
    # Getting the type of 'numpy' (line 1815)
    numpy_125588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1815, 23), 'numpy', False)
    # Obtaining the member 'int8' of a type (line 1815)
    int8_125589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1815, 23), numpy_125588, 'int8')
    # Processing the call keyword arguments (line 1815)
    kwargs_125590 = {}
    # Getting the type of 'tmp1' (line 1815)
    tmp1_125586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1815, 11), 'tmp1', False)
    # Obtaining the member 'astype' of a type (line 1815)
    astype_125587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1815, 11), tmp1_125586, 'astype')
    # Calling astype(args, kwargs) (line 1815)
    astype_call_result_125591 = invoke(stypy.reporting.localization.Localization(__file__, 1815, 11), astype_125587, *[int8_125589], **kwargs_125590)
    
    
    # Call to astype(...): (line 1815)
    # Processing the call arguments (line 1815)
    # Getting the type of 'numpy' (line 1815)
    numpy_125594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1815, 49), 'numpy', False)
    # Obtaining the member 'int8' of a type (line 1815)
    int8_125595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1815, 49), numpy_125594, 'int8')
    # Processing the call keyword arguments (line 1815)
    kwargs_125596 = {}
    # Getting the type of 'tmp2' (line 1815)
    tmp2_125592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1815, 37), 'tmp2', False)
    # Obtaining the member 'astype' of a type (line 1815)
    astype_125593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1815, 37), tmp2_125592, 'astype')
    # Calling astype(args, kwargs) (line 1815)
    astype_call_result_125597 = invoke(stypy.reporting.localization.Localization(__file__, 1815, 37), astype_125593, *[int8_125595], **kwargs_125596)
    
    # Applying the binary operator '-' (line 1815)
    result_sub_125598 = python_operator(stypy.reporting.localization.Localization(__file__, 1815, 11), '-', astype_call_result_125591, astype_call_result_125597)
    
    # Assigning a type to the variable 'tmp1' (line 1815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1815, 4), 'tmp1', result_sub_125598)
    
    # Assigning a Call to a Name (line 1816):
    
    # Assigning a Call to a Name (line 1816):
    
    # Call to lower(...): (line 1816)
    # Processing the call keyword arguments (line 1816)
    kwargs_125601 = {}
    # Getting the type of 'metric' (line 1816)
    metric_125599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1816, 13), 'metric', False)
    # Obtaining the member 'lower' of a type (line 1816)
    lower_125600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1816, 13), metric_125599, 'lower')
    # Calling lower(args, kwargs) (line 1816)
    lower_call_result_125602 = invoke(stypy.reporting.localization.Localization(__file__, 1816, 13), lower_125600, *[], **kwargs_125601)
    
    # Assigning a type to the variable 'metric' (line 1816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1816, 4), 'metric', lower_call_result_125602)
    
    
    # Getting the type of 'metric' (line 1817)
    metric_125603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1817, 7), 'metric')
    str_125604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1817, 17), 'str', 'euclidean')
    # Applying the binary operator '==' (line 1817)
    result_eq_125605 = python_operator(stypy.reporting.localization.Localization(__file__, 1817, 7), '==', metric_125603, str_125604)
    
    # Testing the type of an if condition (line 1817)
    if_condition_125606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1817, 4), result_eq_125605)
    # Assigning a type to the variable 'if_condition_125606' (line 1817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1817, 4), 'if_condition_125606', if_condition_125606)
    # SSA begins for if statement (line 1817)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 1818):
    
    # Assigning a Num to a Name (line 1818):
    int_125607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1818, 17), 'int')
    # Assigning a type to the variable 'metric' (line 1818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1818, 8), 'metric', int_125607)
    # SSA branch for the else part of an if statement (line 1817)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'metric' (line 1819)
    metric_125608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1819, 9), 'metric')
    
    # Obtaining an instance of the builtin type 'list' (line 1819)
    list_125609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1819, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1819)
    # Adding element type (line 1819)
    str_125610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1819, 20), 'str', 'taxicab')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1819, 19), list_125609, str_125610)
    # Adding element type (line 1819)
    str_125611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1819, 31), 'str', 'cityblock')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1819, 19), list_125609, str_125611)
    # Adding element type (line 1819)
    str_125612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1819, 44), 'str', 'manhattan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1819, 19), list_125609, str_125612)
    
    # Applying the binary operator 'in' (line 1819)
    result_contains_125613 = python_operator(stypy.reporting.localization.Localization(__file__, 1819, 9), 'in', metric_125608, list_125609)
    
    # Testing the type of an if condition (line 1819)
    if_condition_125614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1819, 9), result_contains_125613)
    # Assigning a type to the variable 'if_condition_125614' (line 1819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1819, 9), 'if_condition_125614', if_condition_125614)
    # SSA begins for if statement (line 1819)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 1820):
    
    # Assigning a Num to a Name (line 1820):
    int_125615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1820, 17), 'int')
    # Assigning a type to the variable 'metric' (line 1820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1820, 8), 'metric', int_125615)
    # SSA branch for the else part of an if statement (line 1819)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'metric' (line 1821)
    metric_125616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1821, 9), 'metric')
    str_125617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1821, 19), 'str', 'chessboard')
    # Applying the binary operator '==' (line 1821)
    result_eq_125618 = python_operator(stypy.reporting.localization.Localization(__file__, 1821, 9), '==', metric_125616, str_125617)
    
    # Testing the type of an if condition (line 1821)
    if_condition_125619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1821, 9), result_eq_125618)
    # Assigning a type to the variable 'if_condition_125619' (line 1821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1821, 9), 'if_condition_125619', if_condition_125619)
    # SSA begins for if statement (line 1821)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 1822):
    
    # Assigning a Num to a Name (line 1822):
    int_125620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1822, 17), 'int')
    # Assigning a type to the variable 'metric' (line 1822)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1822, 8), 'metric', int_125620)
    # SSA branch for the else part of an if statement (line 1821)
    module_type_store.open_ssa_branch('else')
    
    # Call to RuntimeError(...): (line 1824)
    # Processing the call arguments (line 1824)
    str_125622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1824, 27), 'str', 'distance metric not supported')
    # Processing the call keyword arguments (line 1824)
    kwargs_125623 = {}
    # Getting the type of 'RuntimeError' (line 1824)
    RuntimeError_125621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1824)
    RuntimeError_call_result_125624 = invoke(stypy.reporting.localization.Localization(__file__, 1824, 14), RuntimeError_125621, *[str_125622], **kwargs_125623)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1824, 8), RuntimeError_call_result_125624, 'raise parameter', BaseException)
    # SSA join for if statement (line 1821)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1819)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1817)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1825)
    # Getting the type of 'sampling' (line 1825)
    sampling_125625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 4), 'sampling')
    # Getting the type of 'None' (line 1825)
    None_125626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 23), 'None')
    
    (may_be_125627, more_types_in_union_125628) = may_not_be_none(sampling_125625, None_125626)

    if may_be_125627:

        if more_types_in_union_125628:
            # Runtime conditional SSA (line 1825)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1826):
        
        # Assigning a Call to a Name (line 1826):
        
        # Call to _normalize_sequence(...): (line 1826)
        # Processing the call arguments (line 1826)
        # Getting the type of 'sampling' (line 1826)
        sampling_125631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 51), 'sampling', False)
        # Getting the type of 'tmp1' (line 1826)
        tmp1_125632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 61), 'tmp1', False)
        # Obtaining the member 'ndim' of a type (line 1826)
        ndim_125633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1826, 61), tmp1_125632, 'ndim')
        # Processing the call keyword arguments (line 1826)
        kwargs_125634 = {}
        # Getting the type of '_ni_support' (line 1826)
        _ni_support_125629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 19), '_ni_support', False)
        # Obtaining the member '_normalize_sequence' of a type (line 1826)
        _normalize_sequence_125630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1826, 19), _ni_support_125629, '_normalize_sequence')
        # Calling _normalize_sequence(args, kwargs) (line 1826)
        _normalize_sequence_call_result_125635 = invoke(stypy.reporting.localization.Localization(__file__, 1826, 19), _normalize_sequence_125630, *[sampling_125631, ndim_125633], **kwargs_125634)
        
        # Assigning a type to the variable 'sampling' (line 1826)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1826, 8), 'sampling', _normalize_sequence_call_result_125635)
        
        # Assigning a Call to a Name (line 1827):
        
        # Assigning a Call to a Name (line 1827):
        
        # Call to asarray(...): (line 1827)
        # Processing the call arguments (line 1827)
        # Getting the type of 'sampling' (line 1827)
        sampling_125638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 33), 'sampling', False)
        # Processing the call keyword arguments (line 1827)
        # Getting the type of 'numpy' (line 1827)
        numpy_125639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 49), 'numpy', False)
        # Obtaining the member 'float64' of a type (line 1827)
        float64_125640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1827, 49), numpy_125639, 'float64')
        keyword_125641 = float64_125640
        kwargs_125642 = {'dtype': keyword_125641}
        # Getting the type of 'numpy' (line 1827)
        numpy_125636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 19), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 1827)
        asarray_125637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1827, 19), numpy_125636, 'asarray')
        # Calling asarray(args, kwargs) (line 1827)
        asarray_call_result_125643 = invoke(stypy.reporting.localization.Localization(__file__, 1827, 19), asarray_125637, *[sampling_125638], **kwargs_125642)
        
        # Assigning a type to the variable 'sampling' (line 1827)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1827, 8), 'sampling', asarray_call_result_125643)
        
        
        # Getting the type of 'sampling' (line 1828)
        sampling_125644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1828, 15), 'sampling')
        # Obtaining the member 'flags' of a type (line 1828)
        flags_125645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1828, 15), sampling_125644, 'flags')
        # Obtaining the member 'contiguous' of a type (line 1828)
        contiguous_125646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1828, 15), flags_125645, 'contiguous')
        # Applying the 'not' unary operator (line 1828)
        result_not__125647 = python_operator(stypy.reporting.localization.Localization(__file__, 1828, 11), 'not', contiguous_125646)
        
        # Testing the type of an if condition (line 1828)
        if_condition_125648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1828, 8), result_not__125647)
        # Assigning a type to the variable 'if_condition_125648' (line 1828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1828, 8), 'if_condition_125648', if_condition_125648)
        # SSA begins for if statement (line 1828)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1829):
        
        # Assigning a Call to a Name (line 1829):
        
        # Call to copy(...): (line 1829)
        # Processing the call keyword arguments (line 1829)
        kwargs_125651 = {}
        # Getting the type of 'sampling' (line 1829)
        sampling_125649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1829, 23), 'sampling', False)
        # Obtaining the member 'copy' of a type (line 1829)
        copy_125650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1829, 23), sampling_125649, 'copy')
        # Calling copy(args, kwargs) (line 1829)
        copy_call_result_125652 = invoke(stypy.reporting.localization.Localization(__file__, 1829, 23), copy_125650, *[], **kwargs_125651)
        
        # Assigning a type to the variable 'sampling' (line 1829)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1829, 12), 'sampling', copy_call_result_125652)
        # SSA join for if statement (line 1828)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_125628:
            # SSA join for if statement (line 1825)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'return_indices' (line 1830)
    return_indices_125653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1830, 7), 'return_indices')
    # Testing the type of an if condition (line 1830)
    if_condition_125654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1830, 4), return_indices_125653)
    # Assigning a type to the variable 'if_condition_125654' (line 1830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1830, 4), 'if_condition_125654', if_condition_125654)
    # SSA begins for if statement (line 1830)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1831):
    
    # Assigning a Call to a Name (line 1831):
    
    # Call to zeros(...): (line 1831)
    # Processing the call arguments (line 1831)
    # Getting the type of 'tmp1' (line 1831)
    tmp1_125657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1831, 25), 'tmp1', False)
    # Obtaining the member 'shape' of a type (line 1831)
    shape_125658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1831, 25), tmp1_125657, 'shape')
    # Processing the call keyword arguments (line 1831)
    # Getting the type of 'numpy' (line 1831)
    numpy_125659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1831, 43), 'numpy', False)
    # Obtaining the member 'int32' of a type (line 1831)
    int32_125660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1831, 43), numpy_125659, 'int32')
    keyword_125661 = int32_125660
    kwargs_125662 = {'dtype': keyword_125661}
    # Getting the type of 'numpy' (line 1831)
    numpy_125655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1831, 13), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 1831)
    zeros_125656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1831, 13), numpy_125655, 'zeros')
    # Calling zeros(args, kwargs) (line 1831)
    zeros_call_result_125663 = invoke(stypy.reporting.localization.Localization(__file__, 1831, 13), zeros_125656, *[shape_125658], **kwargs_125662)
    
    # Assigning a type to the variable 'ft' (line 1831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1831, 8), 'ft', zeros_call_result_125663)
    # SSA branch for the else part of an if statement (line 1830)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 1833):
    
    # Assigning a Name to a Name (line 1833):
    # Getting the type of 'None' (line 1833)
    None_125664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1833, 13), 'None')
    # Assigning a type to the variable 'ft' (line 1833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1833, 8), 'ft', None_125664)
    # SSA join for if statement (line 1830)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_distances' (line 1834)
    return_distances_125665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 7), 'return_distances')
    # Testing the type of an if condition (line 1834)
    if_condition_125666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1834, 4), return_distances_125665)
    # Assigning a type to the variable 'if_condition_125666' (line 1834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1834, 4), 'if_condition_125666', if_condition_125666)
    # SSA begins for if statement (line 1834)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 1835)
    # Getting the type of 'distances' (line 1835)
    distances_125667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1835, 11), 'distances')
    # Getting the type of 'None' (line 1835)
    None_125668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1835, 24), 'None')
    
    (may_be_125669, more_types_in_union_125670) = may_be_none(distances_125667, None_125668)

    if may_be_125669:

        if more_types_in_union_125670:
            # Runtime conditional SSA (line 1835)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'metric' (line 1836)
        metric_125671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1836, 15), 'metric')
        int_125672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1836, 25), 'int')
        # Applying the binary operator '==' (line 1836)
        result_eq_125673 = python_operator(stypy.reporting.localization.Localization(__file__, 1836, 15), '==', metric_125671, int_125672)
        
        # Testing the type of an if condition (line 1836)
        if_condition_125674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1836, 12), result_eq_125673)
        # Assigning a type to the variable 'if_condition_125674' (line 1836)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1836, 12), 'if_condition_125674', if_condition_125674)
        # SSA begins for if statement (line 1836)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1837):
        
        # Assigning a Call to a Name (line 1837):
        
        # Call to zeros(...): (line 1837)
        # Processing the call arguments (line 1837)
        # Getting the type of 'tmp1' (line 1837)
        tmp1_125677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 33), 'tmp1', False)
        # Obtaining the member 'shape' of a type (line 1837)
        shape_125678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1837, 33), tmp1_125677, 'shape')
        # Processing the call keyword arguments (line 1837)
        # Getting the type of 'numpy' (line 1837)
        numpy_125679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 51), 'numpy', False)
        # Obtaining the member 'float64' of a type (line 1837)
        float64_125680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1837, 51), numpy_125679, 'float64')
        keyword_125681 = float64_125680
        kwargs_125682 = {'dtype': keyword_125681}
        # Getting the type of 'numpy' (line 1837)
        numpy_125675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 21), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 1837)
        zeros_125676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1837, 21), numpy_125675, 'zeros')
        # Calling zeros(args, kwargs) (line 1837)
        zeros_call_result_125683 = invoke(stypy.reporting.localization.Localization(__file__, 1837, 21), zeros_125676, *[shape_125678], **kwargs_125682)
        
        # Assigning a type to the variable 'dt' (line 1837)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1837, 16), 'dt', zeros_call_result_125683)
        # SSA branch for the else part of an if statement (line 1836)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1839):
        
        # Assigning a Call to a Name (line 1839):
        
        # Call to zeros(...): (line 1839)
        # Processing the call arguments (line 1839)
        # Getting the type of 'tmp1' (line 1839)
        tmp1_125686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 33), 'tmp1', False)
        # Obtaining the member 'shape' of a type (line 1839)
        shape_125687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1839, 33), tmp1_125686, 'shape')
        # Processing the call keyword arguments (line 1839)
        # Getting the type of 'numpy' (line 1839)
        numpy_125688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 51), 'numpy', False)
        # Obtaining the member 'uint32' of a type (line 1839)
        uint32_125689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1839, 51), numpy_125688, 'uint32')
        keyword_125690 = uint32_125689
        kwargs_125691 = {'dtype': keyword_125690}
        # Getting the type of 'numpy' (line 1839)
        numpy_125684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 21), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 1839)
        zeros_125685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1839, 21), numpy_125684, 'zeros')
        # Calling zeros(args, kwargs) (line 1839)
        zeros_call_result_125692 = invoke(stypy.reporting.localization.Localization(__file__, 1839, 21), zeros_125685, *[shape_125687], **kwargs_125691)
        
        # Assigning a type to the variable 'dt' (line 1839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1839, 16), 'dt', zeros_call_result_125692)
        # SSA join for if statement (line 1836)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_125670:
            # Runtime conditional SSA for else branch (line 1835)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_125669) or more_types_in_union_125670):
        
        
        # Getting the type of 'distances' (line 1841)
        distances_125693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1841, 15), 'distances')
        # Obtaining the member 'shape' of a type (line 1841)
        shape_125694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1841, 15), distances_125693, 'shape')
        # Getting the type of 'tmp1' (line 1841)
        tmp1_125695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1841, 34), 'tmp1')
        # Obtaining the member 'shape' of a type (line 1841)
        shape_125696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1841, 34), tmp1_125695, 'shape')
        # Applying the binary operator '!=' (line 1841)
        result_ne_125697 = python_operator(stypy.reporting.localization.Localization(__file__, 1841, 15), '!=', shape_125694, shape_125696)
        
        # Testing the type of an if condition (line 1841)
        if_condition_125698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1841, 12), result_ne_125697)
        # Assigning a type to the variable 'if_condition_125698' (line 1841)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1841, 12), 'if_condition_125698', if_condition_125698)
        # SSA begins for if statement (line 1841)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 1842)
        # Processing the call arguments (line 1842)
        str_125700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1842, 35), 'str', 'distances array has wrong shape')
        # Processing the call keyword arguments (line 1842)
        kwargs_125701 = {}
        # Getting the type of 'RuntimeError' (line 1842)
        RuntimeError_125699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1842, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 1842)
        RuntimeError_call_result_125702 = invoke(stypy.reporting.localization.Localization(__file__, 1842, 22), RuntimeError_125699, *[str_125700], **kwargs_125701)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1842, 16), RuntimeError_call_result_125702, 'raise parameter', BaseException)
        # SSA join for if statement (line 1841)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'metric' (line 1843)
        metric_125703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 15), 'metric')
        int_125704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1843, 25), 'int')
        # Applying the binary operator '==' (line 1843)
        result_eq_125705 = python_operator(stypy.reporting.localization.Localization(__file__, 1843, 15), '==', metric_125703, int_125704)
        
        # Testing the type of an if condition (line 1843)
        if_condition_125706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1843, 12), result_eq_125705)
        # Assigning a type to the variable 'if_condition_125706' (line 1843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1843, 12), 'if_condition_125706', if_condition_125706)
        # SSA begins for if statement (line 1843)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'distances' (line 1844)
        distances_125707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1844, 19), 'distances')
        # Obtaining the member 'dtype' of a type (line 1844)
        dtype_125708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1844, 19), distances_125707, 'dtype')
        # Obtaining the member 'type' of a type (line 1844)
        type_125709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1844, 19), dtype_125708, 'type')
        # Getting the type of 'numpy' (line 1844)
        numpy_125710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1844, 43), 'numpy')
        # Obtaining the member 'float64' of a type (line 1844)
        float64_125711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1844, 43), numpy_125710, 'float64')
        # Applying the binary operator '!=' (line 1844)
        result_ne_125712 = python_operator(stypy.reporting.localization.Localization(__file__, 1844, 19), '!=', type_125709, float64_125711)
        
        # Testing the type of an if condition (line 1844)
        if_condition_125713 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1844, 16), result_ne_125712)
        # Assigning a type to the variable 'if_condition_125713' (line 1844)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1844, 16), 'if_condition_125713', if_condition_125713)
        # SSA begins for if statement (line 1844)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 1845)
        # Processing the call arguments (line 1845)
        str_125715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1845, 39), 'str', 'distances array must be float64')
        # Processing the call keyword arguments (line 1845)
        kwargs_125716 = {}
        # Getting the type of 'RuntimeError' (line 1845)
        RuntimeError_125714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1845, 26), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 1845)
        RuntimeError_call_result_125717 = invoke(stypy.reporting.localization.Localization(__file__, 1845, 26), RuntimeError_125714, *[str_125715], **kwargs_125716)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1845, 20), RuntimeError_call_result_125717, 'raise parameter', BaseException)
        # SSA join for if statement (line 1844)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1843)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'distances' (line 1847)
        distances_125718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1847, 19), 'distances')
        # Obtaining the member 'dtype' of a type (line 1847)
        dtype_125719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1847, 19), distances_125718, 'dtype')
        # Obtaining the member 'type' of a type (line 1847)
        type_125720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1847, 19), dtype_125719, 'type')
        # Getting the type of 'numpy' (line 1847)
        numpy_125721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1847, 43), 'numpy')
        # Obtaining the member 'uint32' of a type (line 1847)
        uint32_125722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1847, 43), numpy_125721, 'uint32')
        # Applying the binary operator '!=' (line 1847)
        result_ne_125723 = python_operator(stypy.reporting.localization.Localization(__file__, 1847, 19), '!=', type_125720, uint32_125722)
        
        # Testing the type of an if condition (line 1847)
        if_condition_125724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1847, 16), result_ne_125723)
        # Assigning a type to the variable 'if_condition_125724' (line 1847)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1847, 16), 'if_condition_125724', if_condition_125724)
        # SSA begins for if statement (line 1847)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 1848)
        # Processing the call arguments (line 1848)
        str_125726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1848, 39), 'str', 'distances array must be uint32')
        # Processing the call keyword arguments (line 1848)
        kwargs_125727 = {}
        # Getting the type of 'RuntimeError' (line 1848)
        RuntimeError_125725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1848, 26), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 1848)
        RuntimeError_call_result_125728 = invoke(stypy.reporting.localization.Localization(__file__, 1848, 26), RuntimeError_125725, *[str_125726], **kwargs_125727)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1848, 20), RuntimeError_call_result_125728, 'raise parameter', BaseException)
        # SSA join for if statement (line 1847)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1843)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 1849):
        
        # Assigning a Name to a Name (line 1849):
        # Getting the type of 'distances' (line 1849)
        distances_125729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1849, 17), 'distances')
        # Assigning a type to the variable 'dt' (line 1849)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1849, 12), 'dt', distances_125729)

        if (may_be_125669 and more_types_in_union_125670):
            # SSA join for if statement (line 1835)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 1834)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 1851):
    
    # Assigning a Name to a Name (line 1851):
    # Getting the type of 'None' (line 1851)
    None_125730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1851, 13), 'None')
    # Assigning a type to the variable 'dt' (line 1851)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1851, 8), 'dt', None_125730)
    # SSA join for if statement (line 1834)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to distance_transform_bf(...): (line 1853)
    # Processing the call arguments (line 1853)
    # Getting the type of 'tmp1' (line 1853)
    tmp1_125733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 36), 'tmp1', False)
    # Getting the type of 'metric' (line 1853)
    metric_125734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 42), 'metric', False)
    # Getting the type of 'sampling' (line 1853)
    sampling_125735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 50), 'sampling', False)
    # Getting the type of 'dt' (line 1853)
    dt_125736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 60), 'dt', False)
    # Getting the type of 'ft' (line 1853)
    ft_125737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 64), 'ft', False)
    # Processing the call keyword arguments (line 1853)
    kwargs_125738 = {}
    # Getting the type of '_nd_image' (line 1853)
    _nd_image_125731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 4), '_nd_image', False)
    # Obtaining the member 'distance_transform_bf' of a type (line 1853)
    distance_transform_bf_125732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1853, 4), _nd_image_125731, 'distance_transform_bf')
    # Calling distance_transform_bf(args, kwargs) (line 1853)
    distance_transform_bf_call_result_125739 = invoke(stypy.reporting.localization.Localization(__file__, 1853, 4), distance_transform_bf_125732, *[tmp1_125733, metric_125734, sampling_125735, dt_125736, ft_125737], **kwargs_125738)
    
    
    # Getting the type of 'return_indices' (line 1854)
    return_indices_125740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1854, 7), 'return_indices')
    # Testing the type of an if condition (line 1854)
    if_condition_125741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1854, 4), return_indices_125740)
    # Assigning a type to the variable 'if_condition_125741' (line 1854)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1854, 4), 'if_condition_125741', if_condition_125741)
    # SSA begins for if statement (line 1854)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isinstance(...): (line 1855)
    # Processing the call arguments (line 1855)
    # Getting the type of 'indices' (line 1855)
    indices_125743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1855, 22), 'indices', False)
    # Getting the type of 'numpy' (line 1855)
    numpy_125744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1855, 31), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1855)
    ndarray_125745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1855, 31), numpy_125744, 'ndarray')
    # Processing the call keyword arguments (line 1855)
    kwargs_125746 = {}
    # Getting the type of 'isinstance' (line 1855)
    isinstance_125742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1855, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1855)
    isinstance_call_result_125747 = invoke(stypy.reporting.localization.Localization(__file__, 1855, 11), isinstance_125742, *[indices_125743, ndarray_125745], **kwargs_125746)
    
    # Testing the type of an if condition (line 1855)
    if_condition_125748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1855, 8), isinstance_call_result_125747)
    # Assigning a type to the variable 'if_condition_125748' (line 1855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1855, 8), 'if_condition_125748', if_condition_125748)
    # SSA begins for if statement (line 1855)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'indices' (line 1856)
    indices_125749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1856, 15), 'indices')
    # Obtaining the member 'dtype' of a type (line 1856)
    dtype_125750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1856, 15), indices_125749, 'dtype')
    # Obtaining the member 'type' of a type (line 1856)
    type_125751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1856, 15), dtype_125750, 'type')
    # Getting the type of 'numpy' (line 1856)
    numpy_125752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1856, 37), 'numpy')
    # Obtaining the member 'int32' of a type (line 1856)
    int32_125753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1856, 37), numpy_125752, 'int32')
    # Applying the binary operator '!=' (line 1856)
    result_ne_125754 = python_operator(stypy.reporting.localization.Localization(__file__, 1856, 15), '!=', type_125751, int32_125753)
    
    # Testing the type of an if condition (line 1856)
    if_condition_125755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1856, 12), result_ne_125754)
    # Assigning a type to the variable 'if_condition_125755' (line 1856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1856, 12), 'if_condition_125755', if_condition_125755)
    # SSA begins for if statement (line 1856)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1857)
    # Processing the call arguments (line 1857)
    str_125757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1857, 35), 'str', 'indices must of int32 type')
    # Processing the call keyword arguments (line 1857)
    kwargs_125758 = {}
    # Getting the type of 'RuntimeError' (line 1857)
    RuntimeError_125756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1857, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1857)
    RuntimeError_call_result_125759 = invoke(stypy.reporting.localization.Localization(__file__, 1857, 22), RuntimeError_125756, *[str_125757], **kwargs_125758)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1857, 16), RuntimeError_call_result_125759, 'raise parameter', BaseException)
    # SSA join for if statement (line 1856)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'indices' (line 1858)
    indices_125760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1858, 15), 'indices')
    # Obtaining the member 'shape' of a type (line 1858)
    shape_125761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1858, 15), indices_125760, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1858)
    tuple_125762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1858, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1858)
    # Adding element type (line 1858)
    # Getting the type of 'tmp1' (line 1858)
    tmp1_125763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1858, 33), 'tmp1')
    # Obtaining the member 'ndim' of a type (line 1858)
    ndim_125764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1858, 33), tmp1_125763, 'ndim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1858, 33), tuple_125762, ndim_125764)
    
    # Getting the type of 'tmp1' (line 1858)
    tmp1_125765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1858, 47), 'tmp1')
    # Obtaining the member 'shape' of a type (line 1858)
    shape_125766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1858, 47), tmp1_125765, 'shape')
    # Applying the binary operator '+' (line 1858)
    result_add_125767 = python_operator(stypy.reporting.localization.Localization(__file__, 1858, 32), '+', tuple_125762, shape_125766)
    
    # Applying the binary operator '!=' (line 1858)
    result_ne_125768 = python_operator(stypy.reporting.localization.Localization(__file__, 1858, 15), '!=', shape_125761, result_add_125767)
    
    # Testing the type of an if condition (line 1858)
    if_condition_125769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1858, 12), result_ne_125768)
    # Assigning a type to the variable 'if_condition_125769' (line 1858)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1858, 12), 'if_condition_125769', if_condition_125769)
    # SSA begins for if statement (line 1858)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1859)
    # Processing the call arguments (line 1859)
    str_125771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1859, 35), 'str', 'indices has wrong shape')
    # Processing the call keyword arguments (line 1859)
    kwargs_125772 = {}
    # Getting the type of 'RuntimeError' (line 1859)
    RuntimeError_125770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1859, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1859)
    RuntimeError_call_result_125773 = invoke(stypy.reporting.localization.Localization(__file__, 1859, 22), RuntimeError_125770, *[str_125771], **kwargs_125772)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1859, 16), RuntimeError_call_result_125773, 'raise parameter', BaseException)
    # SSA join for if statement (line 1858)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1860):
    
    # Assigning a Name to a Name (line 1860):
    # Getting the type of 'indices' (line 1860)
    indices_125774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1860, 19), 'indices')
    # Assigning a type to the variable 'tmp2' (line 1860)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1860, 12), 'tmp2', indices_125774)
    # SSA branch for the else part of an if statement (line 1855)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1862):
    
    # Assigning a Call to a Name (line 1862):
    
    # Call to indices(...): (line 1862)
    # Processing the call arguments (line 1862)
    # Getting the type of 'tmp1' (line 1862)
    tmp1_125777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1862, 33), 'tmp1', False)
    # Obtaining the member 'shape' of a type (line 1862)
    shape_125778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1862, 33), tmp1_125777, 'shape')
    # Processing the call keyword arguments (line 1862)
    # Getting the type of 'numpy' (line 1862)
    numpy_125779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1862, 51), 'numpy', False)
    # Obtaining the member 'int32' of a type (line 1862)
    int32_125780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1862, 51), numpy_125779, 'int32')
    keyword_125781 = int32_125780
    kwargs_125782 = {'dtype': keyword_125781}
    # Getting the type of 'numpy' (line 1862)
    numpy_125775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1862, 19), 'numpy', False)
    # Obtaining the member 'indices' of a type (line 1862)
    indices_125776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1862, 19), numpy_125775, 'indices')
    # Calling indices(args, kwargs) (line 1862)
    indices_call_result_125783 = invoke(stypy.reporting.localization.Localization(__file__, 1862, 19), indices_125776, *[shape_125778], **kwargs_125782)
    
    # Assigning a type to the variable 'tmp2' (line 1862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1862, 12), 'tmp2', indices_call_result_125783)
    # SSA join for if statement (line 1855)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1863):
    
    # Assigning a Call to a Name (line 1863):
    
    # Call to ravel(...): (line 1863)
    # Processing the call arguments (line 1863)
    # Getting the type of 'ft' (line 1863)
    ft_125786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1863, 25), 'ft', False)
    # Processing the call keyword arguments (line 1863)
    kwargs_125787 = {}
    # Getting the type of 'numpy' (line 1863)
    numpy_125784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1863, 13), 'numpy', False)
    # Obtaining the member 'ravel' of a type (line 1863)
    ravel_125785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1863, 13), numpy_125784, 'ravel')
    # Calling ravel(args, kwargs) (line 1863)
    ravel_call_result_125788 = invoke(stypy.reporting.localization.Localization(__file__, 1863, 13), ravel_125785, *[ft_125786], **kwargs_125787)
    
    # Assigning a type to the variable 'ft' (line 1863)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1863, 8), 'ft', ravel_call_result_125788)
    
    
    # Call to range(...): (line 1864)
    # Processing the call arguments (line 1864)
    
    # Obtaining the type of the subscript
    int_125790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1864, 35), 'int')
    # Getting the type of 'tmp2' (line 1864)
    tmp2_125791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1864, 24), 'tmp2', False)
    # Obtaining the member 'shape' of a type (line 1864)
    shape_125792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1864, 24), tmp2_125791, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1864)
    getitem___125793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1864, 24), shape_125792, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1864)
    subscript_call_result_125794 = invoke(stypy.reporting.localization.Localization(__file__, 1864, 24), getitem___125793, int_125790)
    
    # Processing the call keyword arguments (line 1864)
    kwargs_125795 = {}
    # Getting the type of 'range' (line 1864)
    range_125789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1864, 18), 'range', False)
    # Calling range(args, kwargs) (line 1864)
    range_call_result_125796 = invoke(stypy.reporting.localization.Localization(__file__, 1864, 18), range_125789, *[subscript_call_result_125794], **kwargs_125795)
    
    # Testing the type of a for loop iterable (line 1864)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1864, 8), range_call_result_125796)
    # Getting the type of the for loop variable (line 1864)
    for_loop_var_125797 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1864, 8), range_call_result_125796)
    # Assigning a type to the variable 'ii' (line 1864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1864, 8), 'ii', for_loop_var_125797)
    # SSA begins for a for statement (line 1864)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 1865):
    
    # Assigning a Subscript to a Name (line 1865):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ft' (line 1865)
    ft_125798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1865, 46), 'ft')
    
    # Call to ravel(...): (line 1865)
    # Processing the call arguments (line 1865)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1865)
    ii_125801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1865, 36), 'ii', False)
    Ellipsis_125802 = Ellipsis
    # Getting the type of 'tmp2' (line 1865)
    tmp2_125803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1865, 31), 'tmp2', False)
    # Obtaining the member '__getitem__' of a type (line 1865)
    getitem___125804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1865, 31), tmp2_125803, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1865)
    subscript_call_result_125805 = invoke(stypy.reporting.localization.Localization(__file__, 1865, 31), getitem___125804, (ii_125801, Ellipsis_125802))
    
    # Processing the call keyword arguments (line 1865)
    kwargs_125806 = {}
    # Getting the type of 'numpy' (line 1865)
    numpy_125799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1865, 19), 'numpy', False)
    # Obtaining the member 'ravel' of a type (line 1865)
    ravel_125800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1865, 19), numpy_125799, 'ravel')
    # Calling ravel(args, kwargs) (line 1865)
    ravel_call_result_125807 = invoke(stypy.reporting.localization.Localization(__file__, 1865, 19), ravel_125800, *[subscript_call_result_125805], **kwargs_125806)
    
    # Obtaining the member '__getitem__' of a type (line 1865)
    getitem___125808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1865, 19), ravel_call_result_125807, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1865)
    subscript_call_result_125809 = invoke(stypy.reporting.localization.Localization(__file__, 1865, 19), getitem___125808, ft_125798)
    
    # Assigning a type to the variable 'rtmp' (line 1865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1865, 12), 'rtmp', subscript_call_result_125809)
    
    # Assigning a Attribute to a Attribute (line 1866):
    
    # Assigning a Attribute to a Attribute (line 1866):
    # Getting the type of 'tmp1' (line 1866)
    tmp1_125810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1866, 25), 'tmp1')
    # Obtaining the member 'shape' of a type (line 1866)
    shape_125811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1866, 25), tmp1_125810, 'shape')
    # Getting the type of 'rtmp' (line 1866)
    rtmp_125812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1866, 12), 'rtmp')
    # Setting the type of the member 'shape' of a type (line 1866)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1866, 12), rtmp_125812, 'shape', shape_125811)
    
    # Assigning a Name to a Subscript (line 1867):
    
    # Assigning a Name to a Subscript (line 1867):
    # Getting the type of 'rtmp' (line 1867)
    rtmp_125813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1867, 28), 'rtmp')
    # Getting the type of 'tmp2' (line 1867)
    tmp2_125814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1867, 12), 'tmp2')
    # Getting the type of 'ii' (line 1867)
    ii_125815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1867, 17), 'ii')
    Ellipsis_125816 = Ellipsis
    # Storing an element on a container (line 1867)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1867, 12), tmp2_125814, ((ii_125815, Ellipsis_125816), rtmp_125813))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1868):
    
    # Assigning a Name to a Name (line 1868):
    # Getting the type of 'tmp2' (line 1868)
    tmp2_125817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1868, 13), 'tmp2')
    # Assigning a type to the variable 'ft' (line 1868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1868, 8), 'ft', tmp2_125817)
    # SSA join for if statement (line 1854)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 1871):
    
    # Assigning a List to a Name (line 1871):
    
    # Obtaining an instance of the builtin type 'list' (line 1871)
    list_125818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1871, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1871)
    
    # Assigning a type to the variable 'result' (line 1871)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1871, 4), 'result', list_125818)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'return_distances' (line 1872)
    return_distances_125819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1872, 7), 'return_distances')
    
    
    # Call to isinstance(...): (line 1872)
    # Processing the call arguments (line 1872)
    # Getting the type of 'distances' (line 1872)
    distances_125821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1872, 43), 'distances', False)
    # Getting the type of 'numpy' (line 1872)
    numpy_125822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1872, 54), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1872)
    ndarray_125823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1872, 54), numpy_125822, 'ndarray')
    # Processing the call keyword arguments (line 1872)
    kwargs_125824 = {}
    # Getting the type of 'isinstance' (line 1872)
    isinstance_125820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1872, 32), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1872)
    isinstance_call_result_125825 = invoke(stypy.reporting.localization.Localization(__file__, 1872, 32), isinstance_125820, *[distances_125821, ndarray_125823], **kwargs_125824)
    
    # Applying the 'not' unary operator (line 1872)
    result_not__125826 = python_operator(stypy.reporting.localization.Localization(__file__, 1872, 28), 'not', isinstance_call_result_125825)
    
    # Applying the binary operator 'and' (line 1872)
    result_and_keyword_125827 = python_operator(stypy.reporting.localization.Localization(__file__, 1872, 7), 'and', return_distances_125819, result_not__125826)
    
    # Testing the type of an if condition (line 1872)
    if_condition_125828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1872, 4), result_and_keyword_125827)
    # Assigning a type to the variable 'if_condition_125828' (line 1872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1872, 4), 'if_condition_125828', if_condition_125828)
    # SSA begins for if statement (line 1872)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 1873)
    # Processing the call arguments (line 1873)
    # Getting the type of 'dt' (line 1873)
    dt_125831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1873, 22), 'dt', False)
    # Processing the call keyword arguments (line 1873)
    kwargs_125832 = {}
    # Getting the type of 'result' (line 1873)
    result_125829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1873, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 1873)
    append_125830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1873, 8), result_125829, 'append')
    # Calling append(args, kwargs) (line 1873)
    append_call_result_125833 = invoke(stypy.reporting.localization.Localization(__file__, 1873, 8), append_125830, *[dt_125831], **kwargs_125832)
    
    # SSA join for if statement (line 1872)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'return_indices' (line 1874)
    return_indices_125834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1874, 7), 'return_indices')
    
    
    # Call to isinstance(...): (line 1874)
    # Processing the call arguments (line 1874)
    # Getting the type of 'indices' (line 1874)
    indices_125836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1874, 41), 'indices', False)
    # Getting the type of 'numpy' (line 1874)
    numpy_125837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1874, 50), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1874)
    ndarray_125838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1874, 50), numpy_125837, 'ndarray')
    # Processing the call keyword arguments (line 1874)
    kwargs_125839 = {}
    # Getting the type of 'isinstance' (line 1874)
    isinstance_125835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1874, 30), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1874)
    isinstance_call_result_125840 = invoke(stypy.reporting.localization.Localization(__file__, 1874, 30), isinstance_125835, *[indices_125836, ndarray_125838], **kwargs_125839)
    
    # Applying the 'not' unary operator (line 1874)
    result_not__125841 = python_operator(stypy.reporting.localization.Localization(__file__, 1874, 26), 'not', isinstance_call_result_125840)
    
    # Applying the binary operator 'and' (line 1874)
    result_and_keyword_125842 = python_operator(stypy.reporting.localization.Localization(__file__, 1874, 7), 'and', return_indices_125834, result_not__125841)
    
    # Testing the type of an if condition (line 1874)
    if_condition_125843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1874, 4), result_and_keyword_125842)
    # Assigning a type to the variable 'if_condition_125843' (line 1874)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1874, 4), 'if_condition_125843', if_condition_125843)
    # SSA begins for if statement (line 1874)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 1875)
    # Processing the call arguments (line 1875)
    # Getting the type of 'ft' (line 1875)
    ft_125846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1875, 22), 'ft', False)
    # Processing the call keyword arguments (line 1875)
    kwargs_125847 = {}
    # Getting the type of 'result' (line 1875)
    result_125844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1875, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 1875)
    append_125845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1875, 8), result_125844, 'append')
    # Calling append(args, kwargs) (line 1875)
    append_call_result_125848 = invoke(stypy.reporting.localization.Localization(__file__, 1875, 8), append_125845, *[ft_125846], **kwargs_125847)
    
    # SSA join for if statement (line 1874)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1877)
    # Processing the call arguments (line 1877)
    # Getting the type of 'result' (line 1877)
    result_125850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1877, 11), 'result', False)
    # Processing the call keyword arguments (line 1877)
    kwargs_125851 = {}
    # Getting the type of 'len' (line 1877)
    len_125849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1877, 7), 'len', False)
    # Calling len(args, kwargs) (line 1877)
    len_call_result_125852 = invoke(stypy.reporting.localization.Localization(__file__, 1877, 7), len_125849, *[result_125850], **kwargs_125851)
    
    int_125853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1877, 22), 'int')
    # Applying the binary operator '==' (line 1877)
    result_eq_125854 = python_operator(stypy.reporting.localization.Localization(__file__, 1877, 7), '==', len_call_result_125852, int_125853)
    
    # Testing the type of an if condition (line 1877)
    if_condition_125855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1877, 4), result_eq_125854)
    # Assigning a type to the variable 'if_condition_125855' (line 1877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1877, 4), 'if_condition_125855', if_condition_125855)
    # SSA begins for if statement (line 1877)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tuple(...): (line 1878)
    # Processing the call arguments (line 1878)
    # Getting the type of 'result' (line 1878)
    result_125857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1878, 21), 'result', False)
    # Processing the call keyword arguments (line 1878)
    kwargs_125858 = {}
    # Getting the type of 'tuple' (line 1878)
    tuple_125856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1878, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1878)
    tuple_call_result_125859 = invoke(stypy.reporting.localization.Localization(__file__, 1878, 15), tuple_125856, *[result_125857], **kwargs_125858)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1878, 8), 'stypy_return_type', tuple_call_result_125859)
    # SSA branch for the else part of an if statement (line 1877)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 1879)
    # Processing the call arguments (line 1879)
    # Getting the type of 'result' (line 1879)
    result_125861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1879, 13), 'result', False)
    # Processing the call keyword arguments (line 1879)
    kwargs_125862 = {}
    # Getting the type of 'len' (line 1879)
    len_125860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1879, 9), 'len', False)
    # Calling len(args, kwargs) (line 1879)
    len_call_result_125863 = invoke(stypy.reporting.localization.Localization(__file__, 1879, 9), len_125860, *[result_125861], **kwargs_125862)
    
    int_125864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1879, 24), 'int')
    # Applying the binary operator '==' (line 1879)
    result_eq_125865 = python_operator(stypy.reporting.localization.Localization(__file__, 1879, 9), '==', len_call_result_125863, int_125864)
    
    # Testing the type of an if condition (line 1879)
    if_condition_125866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1879, 9), result_eq_125865)
    # Assigning a type to the variable 'if_condition_125866' (line 1879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1879, 9), 'if_condition_125866', if_condition_125866)
    # SSA begins for if statement (line 1879)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_125867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1880, 22), 'int')
    # Getting the type of 'result' (line 1880)
    result_125868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1880, 15), 'result')
    # Obtaining the member '__getitem__' of a type (line 1880)
    getitem___125869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1880, 15), result_125868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1880)
    subscript_call_result_125870 = invoke(stypy.reporting.localization.Localization(__file__, 1880, 15), getitem___125869, int_125867)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1880, 8), 'stypy_return_type', subscript_call_result_125870)
    # SSA branch for the else part of an if statement (line 1879)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'None' (line 1882)
    None_125871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1882, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 1882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1882, 8), 'stypy_return_type', None_125871)
    # SSA join for if statement (line 1879)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1877)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'distance_transform_bf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'distance_transform_bf' in the type store
    # Getting the type of 'stypy_return_type' (line 1748)
    stypy_return_type_125872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1748, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125872)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'distance_transform_bf'
    return stypy_return_type_125872

# Assigning a type to the variable 'distance_transform_bf' (line 1748)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1748, 0), 'distance_transform_bf', distance_transform_bf)

@norecursion
def distance_transform_cdt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_125873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1885, 41), 'str', 'chessboard')
    # Getting the type of 'True' (line 1886)
    True_125874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1886, 41), 'True')
    # Getting the type of 'False' (line 1886)
    False_125875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1886, 62), 'False')
    # Getting the type of 'None' (line 1887)
    None_125876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1887, 34), 'None')
    # Getting the type of 'None' (line 1887)
    None_125877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1887, 48), 'None')
    defaults = [str_125873, True_125874, False_125875, None_125876, None_125877]
    # Create a new context for function 'distance_transform_cdt'
    module_type_store = module_type_store.open_function_context('distance_transform_cdt', 1885, 0, False)
    
    # Passed parameters checking function
    distance_transform_cdt.stypy_localization = localization
    distance_transform_cdt.stypy_type_of_self = None
    distance_transform_cdt.stypy_type_store = module_type_store
    distance_transform_cdt.stypy_function_name = 'distance_transform_cdt'
    distance_transform_cdt.stypy_param_names_list = ['input', 'metric', 'return_distances', 'return_indices', 'distances', 'indices']
    distance_transform_cdt.stypy_varargs_param_name = None
    distance_transform_cdt.stypy_kwargs_param_name = None
    distance_transform_cdt.stypy_call_defaults = defaults
    distance_transform_cdt.stypy_call_varargs = varargs
    distance_transform_cdt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'distance_transform_cdt', ['input', 'metric', 'return_distances', 'return_indices', 'distances', 'indices'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'distance_transform_cdt', localization, ['input', 'metric', 'return_distances', 'return_indices', 'distances', 'indices'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'distance_transform_cdt(...)' code ##################

    str_125878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1921, (-1)), 'str', "\n    Distance transform for chamfer type of transforms.\n\n    Parameters\n    ----------\n    input : array_like\n        Input\n    metric : {'chessboard', 'taxicab'}, optional\n        The `metric` determines the type of chamfering that is done. If the\n        `metric` is equal to 'taxicab' a structure is generated using\n        generate_binary_structure with a squared distance equal to 1. If\n        the `metric` is equal to 'chessboard', a `metric` is generated\n        using generate_binary_structure with a squared distance equal to\n        the dimensionality of the array. These choices correspond to the\n        common interpretations of the 'taxicab' and the 'chessboard'\n        distance metrics in two dimensions.\n\n        The default for `metric` is 'chessboard'.\n    return_distances, return_indices : bool, optional\n        The `return_distances`, and `return_indices` flags can be used to\n        indicate if the distance transform, the feature transform, or both\n        must be returned.\n\n        If the feature transform is returned (``return_indices=True``),\n        the index of the closest background element is returned along\n        the first axis of the result.\n\n        The `return_distances` default is True, and the\n        `return_indices` default is False.\n    distances, indices : ndarrays of int32, optional\n        The `distances` and `indices` arguments can be used to give optional\n        output arrays that must be the same shape as `input`.\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'return_distances' (line 1922)
    return_distances_125879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1922, 12), 'return_distances')
    # Applying the 'not' unary operator (line 1922)
    result_not__125880 = python_operator(stypy.reporting.localization.Localization(__file__, 1922, 8), 'not', return_distances_125879)
    
    
    # Getting the type of 'return_indices' (line 1922)
    return_indices_125881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1922, 39), 'return_indices')
    # Applying the 'not' unary operator (line 1922)
    result_not__125882 = python_operator(stypy.reporting.localization.Localization(__file__, 1922, 35), 'not', return_indices_125881)
    
    # Applying the binary operator 'and' (line 1922)
    result_and_keyword_125883 = python_operator(stypy.reporting.localization.Localization(__file__, 1922, 7), 'and', result_not__125880, result_not__125882)
    
    # Testing the type of an if condition (line 1922)
    if_condition_125884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1922, 4), result_and_keyword_125883)
    # Assigning a type to the variable 'if_condition_125884' (line 1922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1922, 4), 'if_condition_125884', if_condition_125884)
    # SSA begins for if statement (line 1922)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1923):
    
    # Assigning a Str to a Name (line 1923):
    str_125885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1923, 14), 'str', 'at least one of distances/indices must be specified')
    # Assigning a type to the variable 'msg' (line 1923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1923, 8), 'msg', str_125885)
    
    # Call to RuntimeError(...): (line 1924)
    # Processing the call arguments (line 1924)
    # Getting the type of 'msg' (line 1924)
    msg_125887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1924, 27), 'msg', False)
    # Processing the call keyword arguments (line 1924)
    kwargs_125888 = {}
    # Getting the type of 'RuntimeError' (line 1924)
    RuntimeError_125886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1924, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1924)
    RuntimeError_call_result_125889 = invoke(stypy.reporting.localization.Localization(__file__, 1924, 14), RuntimeError_125886, *[msg_125887], **kwargs_125888)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1924, 8), RuntimeError_call_result_125889, 'raise parameter', BaseException)
    # SSA join for if statement (line 1922)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1926):
    
    # Assigning a Call to a Name (line 1926):
    
    # Call to isinstance(...): (line 1926)
    # Processing the call arguments (line 1926)
    # Getting the type of 'indices' (line 1926)
    indices_125891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1926, 28), 'indices', False)
    # Getting the type of 'numpy' (line 1926)
    numpy_125892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1926, 37), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1926)
    ndarray_125893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1926, 37), numpy_125892, 'ndarray')
    # Processing the call keyword arguments (line 1926)
    kwargs_125894 = {}
    # Getting the type of 'isinstance' (line 1926)
    isinstance_125890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1926, 17), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1926)
    isinstance_call_result_125895 = invoke(stypy.reporting.localization.Localization(__file__, 1926, 17), isinstance_125890, *[indices_125891, ndarray_125893], **kwargs_125894)
    
    # Assigning a type to the variable 'ft_inplace' (line 1926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1926, 4), 'ft_inplace', isinstance_call_result_125895)
    
    # Assigning a Call to a Name (line 1927):
    
    # Assigning a Call to a Name (line 1927):
    
    # Call to isinstance(...): (line 1927)
    # Processing the call arguments (line 1927)
    # Getting the type of 'distances' (line 1927)
    distances_125897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1927, 28), 'distances', False)
    # Getting the type of 'numpy' (line 1927)
    numpy_125898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1927, 39), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1927)
    ndarray_125899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1927, 39), numpy_125898, 'ndarray')
    # Processing the call keyword arguments (line 1927)
    kwargs_125900 = {}
    # Getting the type of 'isinstance' (line 1927)
    isinstance_125896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1927, 17), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1927)
    isinstance_call_result_125901 = invoke(stypy.reporting.localization.Localization(__file__, 1927, 17), isinstance_125896, *[distances_125897, ndarray_125899], **kwargs_125900)
    
    # Assigning a type to the variable 'dt_inplace' (line 1927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1927, 4), 'dt_inplace', isinstance_call_result_125901)
    
    # Assigning a Call to a Name (line 1928):
    
    # Assigning a Call to a Name (line 1928):
    
    # Call to asarray(...): (line 1928)
    # Processing the call arguments (line 1928)
    # Getting the type of 'input' (line 1928)
    input_125904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1928, 26), 'input', False)
    # Processing the call keyword arguments (line 1928)
    kwargs_125905 = {}
    # Getting the type of 'numpy' (line 1928)
    numpy_125902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1928, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1928)
    asarray_125903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1928, 12), numpy_125902, 'asarray')
    # Calling asarray(args, kwargs) (line 1928)
    asarray_call_result_125906 = invoke(stypy.reporting.localization.Localization(__file__, 1928, 12), asarray_125903, *[input_125904], **kwargs_125905)
    
    # Assigning a type to the variable 'input' (line 1928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1928, 4), 'input', asarray_call_result_125906)
    
    
    # Getting the type of 'metric' (line 1929)
    metric_125907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1929, 7), 'metric')
    
    # Obtaining an instance of the builtin type 'list' (line 1929)
    list_125908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1929, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1929)
    # Adding element type (line 1929)
    str_125909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1929, 18), 'str', 'taxicab')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1929, 17), list_125908, str_125909)
    # Adding element type (line 1929)
    str_125910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1929, 29), 'str', 'cityblock')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1929, 17), list_125908, str_125910)
    # Adding element type (line 1929)
    str_125911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1929, 42), 'str', 'manhattan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1929, 17), list_125908, str_125911)
    
    # Applying the binary operator 'in' (line 1929)
    result_contains_125912 = python_operator(stypy.reporting.localization.Localization(__file__, 1929, 7), 'in', metric_125907, list_125908)
    
    # Testing the type of an if condition (line 1929)
    if_condition_125913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1929, 4), result_contains_125912)
    # Assigning a type to the variable 'if_condition_125913' (line 1929)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1929, 4), 'if_condition_125913', if_condition_125913)
    # SSA begins for if statement (line 1929)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 1930):
    
    # Assigning a Attribute to a Name (line 1930):
    # Getting the type of 'input' (line 1930)
    input_125914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1930, 15), 'input')
    # Obtaining the member 'ndim' of a type (line 1930)
    ndim_125915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1930, 15), input_125914, 'ndim')
    # Assigning a type to the variable 'rank' (line 1930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1930, 8), 'rank', ndim_125915)
    
    # Assigning a Call to a Name (line 1931):
    
    # Assigning a Call to a Name (line 1931):
    
    # Call to generate_binary_structure(...): (line 1931)
    # Processing the call arguments (line 1931)
    # Getting the type of 'rank' (line 1931)
    rank_125917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1931, 43), 'rank', False)
    int_125918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1931, 49), 'int')
    # Processing the call keyword arguments (line 1931)
    kwargs_125919 = {}
    # Getting the type of 'generate_binary_structure' (line 1931)
    generate_binary_structure_125916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1931, 17), 'generate_binary_structure', False)
    # Calling generate_binary_structure(args, kwargs) (line 1931)
    generate_binary_structure_call_result_125920 = invoke(stypy.reporting.localization.Localization(__file__, 1931, 17), generate_binary_structure_125916, *[rank_125917, int_125918], **kwargs_125919)
    
    # Assigning a type to the variable 'metric' (line 1931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1931, 8), 'metric', generate_binary_structure_call_result_125920)
    # SSA branch for the else part of an if statement (line 1929)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'metric' (line 1932)
    metric_125921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1932, 9), 'metric')
    str_125922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1932, 19), 'str', 'chessboard')
    # Applying the binary operator '==' (line 1932)
    result_eq_125923 = python_operator(stypy.reporting.localization.Localization(__file__, 1932, 9), '==', metric_125921, str_125922)
    
    # Testing the type of an if condition (line 1932)
    if_condition_125924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1932, 9), result_eq_125923)
    # Assigning a type to the variable 'if_condition_125924' (line 1932)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1932, 9), 'if_condition_125924', if_condition_125924)
    # SSA begins for if statement (line 1932)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 1933):
    
    # Assigning a Attribute to a Name (line 1933):
    # Getting the type of 'input' (line 1933)
    input_125925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1933, 15), 'input')
    # Obtaining the member 'ndim' of a type (line 1933)
    ndim_125926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1933, 15), input_125925, 'ndim')
    # Assigning a type to the variable 'rank' (line 1933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1933, 8), 'rank', ndim_125926)
    
    # Assigning a Call to a Name (line 1934):
    
    # Assigning a Call to a Name (line 1934):
    
    # Call to generate_binary_structure(...): (line 1934)
    # Processing the call arguments (line 1934)
    # Getting the type of 'rank' (line 1934)
    rank_125928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1934, 43), 'rank', False)
    # Getting the type of 'rank' (line 1934)
    rank_125929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1934, 49), 'rank', False)
    # Processing the call keyword arguments (line 1934)
    kwargs_125930 = {}
    # Getting the type of 'generate_binary_structure' (line 1934)
    generate_binary_structure_125927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1934, 17), 'generate_binary_structure', False)
    # Calling generate_binary_structure(args, kwargs) (line 1934)
    generate_binary_structure_call_result_125931 = invoke(stypy.reporting.localization.Localization(__file__, 1934, 17), generate_binary_structure_125927, *[rank_125928, rank_125929], **kwargs_125930)
    
    # Assigning a type to the variable 'metric' (line 1934)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1934, 8), 'metric', generate_binary_structure_call_result_125931)
    # SSA branch for the else part of an if statement (line 1932)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 1936)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 1937):
    
    # Assigning a Call to a Name (line 1937):
    
    # Call to asarray(...): (line 1937)
    # Processing the call arguments (line 1937)
    # Getting the type of 'metric' (line 1937)
    metric_125934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1937, 35), 'metric', False)
    # Processing the call keyword arguments (line 1937)
    kwargs_125935 = {}
    # Getting the type of 'numpy' (line 1937)
    numpy_125932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1937, 21), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1937)
    asarray_125933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1937, 21), numpy_125932, 'asarray')
    # Calling asarray(args, kwargs) (line 1937)
    asarray_call_result_125936 = invoke(stypy.reporting.localization.Localization(__file__, 1937, 21), asarray_125933, *[metric_125934], **kwargs_125935)
    
    # Assigning a type to the variable 'metric' (line 1937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1937, 12), 'metric', asarray_call_result_125936)
    # SSA branch for the except part of a try statement (line 1936)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1936)
    module_type_store.open_ssa_branch('except')
    
    # Call to RuntimeError(...): (line 1939)
    # Processing the call arguments (line 1939)
    str_125938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1939, 31), 'str', 'invalid metric provided')
    # Processing the call keyword arguments (line 1939)
    kwargs_125939 = {}
    # Getting the type of 'RuntimeError' (line 1939)
    RuntimeError_125937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1939, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1939)
    RuntimeError_call_result_125940 = invoke(stypy.reporting.localization.Localization(__file__, 1939, 18), RuntimeError_125937, *[str_125938], **kwargs_125939)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1939, 12), RuntimeError_call_result_125940, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1936)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'metric' (line 1940)
    metric_125941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1940, 17), 'metric')
    # Obtaining the member 'shape' of a type (line 1940)
    shape_125942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1940, 17), metric_125941, 'shape')
    # Testing the type of a for loop iterable (line 1940)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1940, 8), shape_125942)
    # Getting the type of the for loop variable (line 1940)
    for_loop_var_125943 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1940, 8), shape_125942)
    # Assigning a type to the variable 's' (line 1940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1940, 8), 's', for_loop_var_125943)
    # SSA begins for a for statement (line 1940)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 's' (line 1941)
    s_125944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1941, 15), 's')
    int_125945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1941, 20), 'int')
    # Applying the binary operator '!=' (line 1941)
    result_ne_125946 = python_operator(stypy.reporting.localization.Localization(__file__, 1941, 15), '!=', s_125944, int_125945)
    
    # Testing the type of an if condition (line 1941)
    if_condition_125947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1941, 12), result_ne_125946)
    # Assigning a type to the variable 'if_condition_125947' (line 1941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1941, 12), 'if_condition_125947', if_condition_125947)
    # SSA begins for if statement (line 1941)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1942)
    # Processing the call arguments (line 1942)
    str_125949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1942, 35), 'str', 'metric sizes must be equal to 3')
    # Processing the call keyword arguments (line 1942)
    kwargs_125950 = {}
    # Getting the type of 'RuntimeError' (line 1942)
    RuntimeError_125948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1942, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1942)
    RuntimeError_call_result_125951 = invoke(stypy.reporting.localization.Localization(__file__, 1942, 22), RuntimeError_125948, *[str_125949], **kwargs_125950)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1942, 16), RuntimeError_call_result_125951, 'raise parameter', BaseException)
    # SSA join for if statement (line 1941)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1932)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1929)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'metric' (line 1944)
    metric_125952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1944, 11), 'metric')
    # Obtaining the member 'flags' of a type (line 1944)
    flags_125953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1944, 11), metric_125952, 'flags')
    # Obtaining the member 'contiguous' of a type (line 1944)
    contiguous_125954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1944, 11), flags_125953, 'contiguous')
    # Applying the 'not' unary operator (line 1944)
    result_not__125955 = python_operator(stypy.reporting.localization.Localization(__file__, 1944, 7), 'not', contiguous_125954)
    
    # Testing the type of an if condition (line 1944)
    if_condition_125956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1944, 4), result_not__125955)
    # Assigning a type to the variable 'if_condition_125956' (line 1944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1944, 4), 'if_condition_125956', if_condition_125956)
    # SSA begins for if statement (line 1944)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1945):
    
    # Assigning a Call to a Name (line 1945):
    
    # Call to copy(...): (line 1945)
    # Processing the call keyword arguments (line 1945)
    kwargs_125959 = {}
    # Getting the type of 'metric' (line 1945)
    metric_125957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1945, 17), 'metric', False)
    # Obtaining the member 'copy' of a type (line 1945)
    copy_125958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1945, 17), metric_125957, 'copy')
    # Calling copy(args, kwargs) (line 1945)
    copy_call_result_125960 = invoke(stypy.reporting.localization.Localization(__file__, 1945, 17), copy_125958, *[], **kwargs_125959)
    
    # Assigning a type to the variable 'metric' (line 1945)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1945, 8), 'metric', copy_call_result_125960)
    # SSA join for if statement (line 1944)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'dt_inplace' (line 1946)
    dt_inplace_125961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1946, 7), 'dt_inplace')
    # Testing the type of an if condition (line 1946)
    if_condition_125962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1946, 4), dt_inplace_125961)
    # Assigning a type to the variable 'if_condition_125962' (line 1946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1946, 4), 'if_condition_125962', if_condition_125962)
    # SSA begins for if statement (line 1946)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'distances' (line 1947)
    distances_125963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1947, 11), 'distances')
    # Obtaining the member 'dtype' of a type (line 1947)
    dtype_125964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1947, 11), distances_125963, 'dtype')
    # Obtaining the member 'type' of a type (line 1947)
    type_125965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1947, 11), dtype_125964, 'type')
    # Getting the type of 'numpy' (line 1947)
    numpy_125966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1947, 35), 'numpy')
    # Obtaining the member 'int32' of a type (line 1947)
    int32_125967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1947, 35), numpy_125966, 'int32')
    # Applying the binary operator '!=' (line 1947)
    result_ne_125968 = python_operator(stypy.reporting.localization.Localization(__file__, 1947, 11), '!=', type_125965, int32_125967)
    
    # Testing the type of an if condition (line 1947)
    if_condition_125969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1947, 8), result_ne_125968)
    # Assigning a type to the variable 'if_condition_125969' (line 1947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1947, 8), 'if_condition_125969', if_condition_125969)
    # SSA begins for if statement (line 1947)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1948)
    # Processing the call arguments (line 1948)
    str_125971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1948, 31), 'str', 'distances must be of int32 type')
    # Processing the call keyword arguments (line 1948)
    kwargs_125972 = {}
    # Getting the type of 'RuntimeError' (line 1948)
    RuntimeError_125970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1948, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1948)
    RuntimeError_call_result_125973 = invoke(stypy.reporting.localization.Localization(__file__, 1948, 18), RuntimeError_125970, *[str_125971], **kwargs_125972)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1948, 12), RuntimeError_call_result_125973, 'raise parameter', BaseException)
    # SSA join for if statement (line 1947)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'distances' (line 1949)
    distances_125974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1949, 11), 'distances')
    # Obtaining the member 'shape' of a type (line 1949)
    shape_125975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1949, 11), distances_125974, 'shape')
    # Getting the type of 'input' (line 1949)
    input_125976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1949, 30), 'input')
    # Obtaining the member 'shape' of a type (line 1949)
    shape_125977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1949, 30), input_125976, 'shape')
    # Applying the binary operator '!=' (line 1949)
    result_ne_125978 = python_operator(stypy.reporting.localization.Localization(__file__, 1949, 11), '!=', shape_125975, shape_125977)
    
    # Testing the type of an if condition (line 1949)
    if_condition_125979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1949, 8), result_ne_125978)
    # Assigning a type to the variable 'if_condition_125979' (line 1949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1949, 8), 'if_condition_125979', if_condition_125979)
    # SSA begins for if statement (line 1949)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1950)
    # Processing the call arguments (line 1950)
    str_125981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1950, 31), 'str', 'distances has wrong shape')
    # Processing the call keyword arguments (line 1950)
    kwargs_125982 = {}
    # Getting the type of 'RuntimeError' (line 1950)
    RuntimeError_125980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1950, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1950)
    RuntimeError_call_result_125983 = invoke(stypy.reporting.localization.Localization(__file__, 1950, 18), RuntimeError_125980, *[str_125981], **kwargs_125982)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1950, 12), RuntimeError_call_result_125983, 'raise parameter', BaseException)
    # SSA join for if statement (line 1949)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1951):
    
    # Assigning a Name to a Name (line 1951):
    # Getting the type of 'distances' (line 1951)
    distances_125984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1951, 13), 'distances')
    # Assigning a type to the variable 'dt' (line 1951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1951, 8), 'dt', distances_125984)
    
    # Assigning a Call to a Subscript (line 1952):
    
    # Assigning a Call to a Subscript (line 1952):
    
    # Call to astype(...): (line 1952)
    # Processing the call arguments (line 1952)
    # Getting the type of 'numpy' (line 1952)
    numpy_125993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1952, 51), 'numpy', False)
    # Obtaining the member 'int32' of a type (line 1952)
    int32_125994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1952, 51), numpy_125993, 'int32')
    # Processing the call keyword arguments (line 1952)
    kwargs_125995 = {}
    
    # Call to where(...): (line 1952)
    # Processing the call arguments (line 1952)
    # Getting the type of 'input' (line 1952)
    input_125987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1952, 30), 'input', False)
    int_125988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1952, 37), 'int')
    int_125989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1952, 41), 'int')
    # Processing the call keyword arguments (line 1952)
    kwargs_125990 = {}
    # Getting the type of 'numpy' (line 1952)
    numpy_125985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1952, 18), 'numpy', False)
    # Obtaining the member 'where' of a type (line 1952)
    where_125986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1952, 18), numpy_125985, 'where')
    # Calling where(args, kwargs) (line 1952)
    where_call_result_125991 = invoke(stypy.reporting.localization.Localization(__file__, 1952, 18), where_125986, *[input_125987, int_125988, int_125989], **kwargs_125990)
    
    # Obtaining the member 'astype' of a type (line 1952)
    astype_125992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1952, 18), where_call_result_125991, 'astype')
    # Calling astype(args, kwargs) (line 1952)
    astype_call_result_125996 = invoke(stypy.reporting.localization.Localization(__file__, 1952, 18), astype_125992, *[int32_125994], **kwargs_125995)
    
    # Getting the type of 'dt' (line 1952)
    dt_125997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1952, 8), 'dt')
    Ellipsis_125998 = Ellipsis
    # Storing an element on a container (line 1952)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1952, 8), dt_125997, (Ellipsis_125998, astype_call_result_125996))
    # SSA branch for the else part of an if statement (line 1946)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1954):
    
    # Assigning a Call to a Name (line 1954):
    
    # Call to astype(...): (line 1954)
    # Processing the call arguments (line 1954)
    # Getting the type of 'numpy' (line 1954)
    numpy_126007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1954, 46), 'numpy', False)
    # Obtaining the member 'int32' of a type (line 1954)
    int32_126008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1954, 46), numpy_126007, 'int32')
    # Processing the call keyword arguments (line 1954)
    kwargs_126009 = {}
    
    # Call to where(...): (line 1954)
    # Processing the call arguments (line 1954)
    # Getting the type of 'input' (line 1954)
    input_126001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1954, 25), 'input', False)
    int_126002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1954, 32), 'int')
    int_126003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1954, 36), 'int')
    # Processing the call keyword arguments (line 1954)
    kwargs_126004 = {}
    # Getting the type of 'numpy' (line 1954)
    numpy_125999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1954, 13), 'numpy', False)
    # Obtaining the member 'where' of a type (line 1954)
    where_126000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1954, 13), numpy_125999, 'where')
    # Calling where(args, kwargs) (line 1954)
    where_call_result_126005 = invoke(stypy.reporting.localization.Localization(__file__, 1954, 13), where_126000, *[input_126001, int_126002, int_126003], **kwargs_126004)
    
    # Obtaining the member 'astype' of a type (line 1954)
    astype_126006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1954, 13), where_call_result_126005, 'astype')
    # Calling astype(args, kwargs) (line 1954)
    astype_call_result_126010 = invoke(stypy.reporting.localization.Localization(__file__, 1954, 13), astype_126006, *[int32_126008], **kwargs_126009)
    
    # Assigning a type to the variable 'dt' (line 1954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1954, 8), 'dt', astype_call_result_126010)
    # SSA join for if statement (line 1946)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 1956):
    
    # Assigning a Attribute to a Name (line 1956):
    # Getting the type of 'dt' (line 1956)
    dt_126011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1956, 11), 'dt')
    # Obtaining the member 'ndim' of a type (line 1956)
    ndim_126012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1956, 11), dt_126011, 'ndim')
    # Assigning a type to the variable 'rank' (line 1956)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1956, 4), 'rank', ndim_126012)
    
    # Getting the type of 'return_indices' (line 1957)
    return_indices_126013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1957, 7), 'return_indices')
    # Testing the type of an if condition (line 1957)
    if_condition_126014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1957, 4), return_indices_126013)
    # Assigning a type to the variable 'if_condition_126014' (line 1957)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1957, 4), 'if_condition_126014', if_condition_126014)
    # SSA begins for if statement (line 1957)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1958):
    
    # Assigning a Call to a Name (line 1958):
    
    # Call to product(...): (line 1958)
    # Processing the call arguments (line 1958)
    # Getting the type of 'dt' (line 1958)
    dt_126017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1958, 27), 'dt', False)
    # Obtaining the member 'shape' of a type (line 1958)
    shape_126018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1958, 27), dt_126017, 'shape')
    # Processing the call keyword arguments (line 1958)
    int_126019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1958, 41), 'int')
    keyword_126020 = int_126019
    kwargs_126021 = {'axis': keyword_126020}
    # Getting the type of 'numpy' (line 1958)
    numpy_126015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1958, 13), 'numpy', False)
    # Obtaining the member 'product' of a type (line 1958)
    product_126016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1958, 13), numpy_126015, 'product')
    # Calling product(args, kwargs) (line 1958)
    product_call_result_126022 = invoke(stypy.reporting.localization.Localization(__file__, 1958, 13), product_126016, *[shape_126018], **kwargs_126021)
    
    # Assigning a type to the variable 'sz' (line 1958)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1958, 8), 'sz', product_call_result_126022)
    
    # Assigning a Call to a Name (line 1959):
    
    # Assigning a Call to a Name (line 1959):
    
    # Call to arange(...): (line 1959)
    # Processing the call arguments (line 1959)
    # Getting the type of 'sz' (line 1959)
    sz_126025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1959, 26), 'sz', False)
    # Processing the call keyword arguments (line 1959)
    # Getting the type of 'numpy' (line 1959)
    numpy_126026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1959, 36), 'numpy', False)
    # Obtaining the member 'int32' of a type (line 1959)
    int32_126027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1959, 36), numpy_126026, 'int32')
    keyword_126028 = int32_126027
    kwargs_126029 = {'dtype': keyword_126028}
    # Getting the type of 'numpy' (line 1959)
    numpy_126023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1959, 13), 'numpy', False)
    # Obtaining the member 'arange' of a type (line 1959)
    arange_126024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1959, 13), numpy_126023, 'arange')
    # Calling arange(args, kwargs) (line 1959)
    arange_call_result_126030 = invoke(stypy.reporting.localization.Localization(__file__, 1959, 13), arange_126024, *[sz_126025], **kwargs_126029)
    
    # Assigning a type to the variable 'ft' (line 1959)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1959, 8), 'ft', arange_call_result_126030)
    
    # Assigning a Attribute to a Attribute (line 1960):
    
    # Assigning a Attribute to a Attribute (line 1960):
    # Getting the type of 'dt' (line 1960)
    dt_126031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1960, 19), 'dt')
    # Obtaining the member 'shape' of a type (line 1960)
    shape_126032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1960, 19), dt_126031, 'shape')
    # Getting the type of 'ft' (line 1960)
    ft_126033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1960, 8), 'ft')
    # Setting the type of the member 'shape' of a type (line 1960)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1960, 8), ft_126033, 'shape', shape_126032)
    # SSA branch for the else part of an if statement (line 1957)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 1962):
    
    # Assigning a Name to a Name (line 1962):
    # Getting the type of 'None' (line 1962)
    None_126034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1962, 13), 'None')
    # Assigning a type to the variable 'ft' (line 1962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1962, 8), 'ft', None_126034)
    # SSA join for if statement (line 1957)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to distance_transform_op(...): (line 1964)
    # Processing the call arguments (line 1964)
    # Getting the type of 'metric' (line 1964)
    metric_126037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1964, 36), 'metric', False)
    # Getting the type of 'dt' (line 1964)
    dt_126038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1964, 44), 'dt', False)
    # Getting the type of 'ft' (line 1964)
    ft_126039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1964, 48), 'ft', False)
    # Processing the call keyword arguments (line 1964)
    kwargs_126040 = {}
    # Getting the type of '_nd_image' (line 1964)
    _nd_image_126035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1964, 4), '_nd_image', False)
    # Obtaining the member 'distance_transform_op' of a type (line 1964)
    distance_transform_op_126036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1964, 4), _nd_image_126035, 'distance_transform_op')
    # Calling distance_transform_op(args, kwargs) (line 1964)
    distance_transform_op_call_result_126041 = invoke(stypy.reporting.localization.Localization(__file__, 1964, 4), distance_transform_op_126036, *[metric_126037, dt_126038, ft_126039], **kwargs_126040)
    
    
    # Assigning a Subscript to a Name (line 1965):
    
    # Assigning a Subscript to a Name (line 1965):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 1965)
    # Processing the call arguments (line 1965)
    
    # Obtaining an instance of the builtin type 'list' (line 1965)
    list_126043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1965, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1965)
    # Adding element type (line 1965)
    
    # Call to slice(...): (line 1965)
    # Processing the call arguments (line 1965)
    # Getting the type of 'None' (line 1965)
    None_126045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1965, 25), 'None', False)
    # Getting the type of 'None' (line 1965)
    None_126046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1965, 31), 'None', False)
    int_126047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1965, 37), 'int')
    # Processing the call keyword arguments (line 1965)
    kwargs_126048 = {}
    # Getting the type of 'slice' (line 1965)
    slice_126044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1965, 19), 'slice', False)
    # Calling slice(args, kwargs) (line 1965)
    slice_call_result_126049 = invoke(stypy.reporting.localization.Localization(__file__, 1965, 19), slice_126044, *[None_126045, None_126046, int_126047], **kwargs_126048)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1965, 18), list_126043, slice_call_result_126049)
    
    # Getting the type of 'rank' (line 1965)
    rank_126050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1965, 44), 'rank', False)
    # Applying the binary operator '*' (line 1965)
    result_mul_126051 = python_operator(stypy.reporting.localization.Localization(__file__, 1965, 18), '*', list_126043, rank_126050)
    
    # Processing the call keyword arguments (line 1965)
    kwargs_126052 = {}
    # Getting the type of 'tuple' (line 1965)
    tuple_126042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1965, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1965)
    tuple_call_result_126053 = invoke(stypy.reporting.localization.Localization(__file__, 1965, 12), tuple_126042, *[result_mul_126051], **kwargs_126052)
    
    # Getting the type of 'dt' (line 1965)
    dt_126054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1965, 9), 'dt')
    # Obtaining the member '__getitem__' of a type (line 1965)
    getitem___126055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1965, 9), dt_126054, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1965)
    subscript_call_result_126056 = invoke(stypy.reporting.localization.Localization(__file__, 1965, 9), getitem___126055, tuple_call_result_126053)
    
    # Assigning a type to the variable 'dt' (line 1965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1965, 4), 'dt', subscript_call_result_126056)
    
    # Getting the type of 'return_indices' (line 1966)
    return_indices_126057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1966, 7), 'return_indices')
    # Testing the type of an if condition (line 1966)
    if_condition_126058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1966, 4), return_indices_126057)
    # Assigning a type to the variable 'if_condition_126058' (line 1966)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1966, 4), 'if_condition_126058', if_condition_126058)
    # SSA begins for if statement (line 1966)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1967):
    
    # Assigning a Subscript to a Name (line 1967):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 1967)
    # Processing the call arguments (line 1967)
    
    # Obtaining an instance of the builtin type 'list' (line 1967)
    list_126060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1967, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1967)
    # Adding element type (line 1967)
    
    # Call to slice(...): (line 1967)
    # Processing the call arguments (line 1967)
    # Getting the type of 'None' (line 1967)
    None_126062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1967, 29), 'None', False)
    # Getting the type of 'None' (line 1967)
    None_126063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1967, 35), 'None', False)
    int_126064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1967, 41), 'int')
    # Processing the call keyword arguments (line 1967)
    kwargs_126065 = {}
    # Getting the type of 'slice' (line 1967)
    slice_126061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1967, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 1967)
    slice_call_result_126066 = invoke(stypy.reporting.localization.Localization(__file__, 1967, 23), slice_126061, *[None_126062, None_126063, int_126064], **kwargs_126065)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1967, 22), list_126060, slice_call_result_126066)
    
    # Getting the type of 'rank' (line 1967)
    rank_126067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1967, 48), 'rank', False)
    # Applying the binary operator '*' (line 1967)
    result_mul_126068 = python_operator(stypy.reporting.localization.Localization(__file__, 1967, 22), '*', list_126060, rank_126067)
    
    # Processing the call keyword arguments (line 1967)
    kwargs_126069 = {}
    # Getting the type of 'tuple' (line 1967)
    tuple_126059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1967, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1967)
    tuple_call_result_126070 = invoke(stypy.reporting.localization.Localization(__file__, 1967, 16), tuple_126059, *[result_mul_126068], **kwargs_126069)
    
    # Getting the type of 'ft' (line 1967)
    ft_126071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1967, 13), 'ft')
    # Obtaining the member '__getitem__' of a type (line 1967)
    getitem___126072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1967, 13), ft_126071, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1967)
    subscript_call_result_126073 = invoke(stypy.reporting.localization.Localization(__file__, 1967, 13), getitem___126072, tuple_call_result_126070)
    
    # Assigning a type to the variable 'ft' (line 1967)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1967, 8), 'ft', subscript_call_result_126073)
    # SSA join for if statement (line 1966)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to distance_transform_op(...): (line 1968)
    # Processing the call arguments (line 1968)
    # Getting the type of 'metric' (line 1968)
    metric_126076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1968, 36), 'metric', False)
    # Getting the type of 'dt' (line 1968)
    dt_126077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1968, 44), 'dt', False)
    # Getting the type of 'ft' (line 1968)
    ft_126078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1968, 48), 'ft', False)
    # Processing the call keyword arguments (line 1968)
    kwargs_126079 = {}
    # Getting the type of '_nd_image' (line 1968)
    _nd_image_126074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1968, 4), '_nd_image', False)
    # Obtaining the member 'distance_transform_op' of a type (line 1968)
    distance_transform_op_126075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1968, 4), _nd_image_126074, 'distance_transform_op')
    # Calling distance_transform_op(args, kwargs) (line 1968)
    distance_transform_op_call_result_126080 = invoke(stypy.reporting.localization.Localization(__file__, 1968, 4), distance_transform_op_126075, *[metric_126076, dt_126077, ft_126078], **kwargs_126079)
    
    
    # Assigning a Subscript to a Name (line 1969):
    
    # Assigning a Subscript to a Name (line 1969):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 1969)
    # Processing the call arguments (line 1969)
    
    # Obtaining an instance of the builtin type 'list' (line 1969)
    list_126082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1969, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1969)
    # Adding element type (line 1969)
    
    # Call to slice(...): (line 1969)
    # Processing the call arguments (line 1969)
    # Getting the type of 'None' (line 1969)
    None_126084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1969, 25), 'None', False)
    # Getting the type of 'None' (line 1969)
    None_126085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1969, 31), 'None', False)
    int_126086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1969, 37), 'int')
    # Processing the call keyword arguments (line 1969)
    kwargs_126087 = {}
    # Getting the type of 'slice' (line 1969)
    slice_126083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1969, 19), 'slice', False)
    # Calling slice(args, kwargs) (line 1969)
    slice_call_result_126088 = invoke(stypy.reporting.localization.Localization(__file__, 1969, 19), slice_126083, *[None_126084, None_126085, int_126086], **kwargs_126087)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1969, 18), list_126082, slice_call_result_126088)
    
    # Getting the type of 'rank' (line 1969)
    rank_126089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1969, 44), 'rank', False)
    # Applying the binary operator '*' (line 1969)
    result_mul_126090 = python_operator(stypy.reporting.localization.Localization(__file__, 1969, 18), '*', list_126082, rank_126089)
    
    # Processing the call keyword arguments (line 1969)
    kwargs_126091 = {}
    # Getting the type of 'tuple' (line 1969)
    tuple_126081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1969, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1969)
    tuple_call_result_126092 = invoke(stypy.reporting.localization.Localization(__file__, 1969, 12), tuple_126081, *[result_mul_126090], **kwargs_126091)
    
    # Getting the type of 'dt' (line 1969)
    dt_126093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1969, 9), 'dt')
    # Obtaining the member '__getitem__' of a type (line 1969)
    getitem___126094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1969, 9), dt_126093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1969)
    subscript_call_result_126095 = invoke(stypy.reporting.localization.Localization(__file__, 1969, 9), getitem___126094, tuple_call_result_126092)
    
    # Assigning a type to the variable 'dt' (line 1969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1969, 4), 'dt', subscript_call_result_126095)
    
    # Getting the type of 'return_indices' (line 1970)
    return_indices_126096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1970, 7), 'return_indices')
    # Testing the type of an if condition (line 1970)
    if_condition_126097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1970, 4), return_indices_126096)
    # Assigning a type to the variable 'if_condition_126097' (line 1970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1970, 4), 'if_condition_126097', if_condition_126097)
    # SSA begins for if statement (line 1970)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1971):
    
    # Assigning a Subscript to a Name (line 1971):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 1971)
    # Processing the call arguments (line 1971)
    
    # Obtaining an instance of the builtin type 'list' (line 1971)
    list_126099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1971, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1971)
    # Adding element type (line 1971)
    
    # Call to slice(...): (line 1971)
    # Processing the call arguments (line 1971)
    # Getting the type of 'None' (line 1971)
    None_126101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1971, 29), 'None', False)
    # Getting the type of 'None' (line 1971)
    None_126102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1971, 35), 'None', False)
    int_126103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1971, 41), 'int')
    # Processing the call keyword arguments (line 1971)
    kwargs_126104 = {}
    # Getting the type of 'slice' (line 1971)
    slice_126100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1971, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 1971)
    slice_call_result_126105 = invoke(stypy.reporting.localization.Localization(__file__, 1971, 23), slice_126100, *[None_126101, None_126102, int_126103], **kwargs_126104)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1971, 22), list_126099, slice_call_result_126105)
    
    # Getting the type of 'rank' (line 1971)
    rank_126106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1971, 48), 'rank', False)
    # Applying the binary operator '*' (line 1971)
    result_mul_126107 = python_operator(stypy.reporting.localization.Localization(__file__, 1971, 22), '*', list_126099, rank_126106)
    
    # Processing the call keyword arguments (line 1971)
    kwargs_126108 = {}
    # Getting the type of 'tuple' (line 1971)
    tuple_126098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1971, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1971)
    tuple_call_result_126109 = invoke(stypy.reporting.localization.Localization(__file__, 1971, 16), tuple_126098, *[result_mul_126107], **kwargs_126108)
    
    # Getting the type of 'ft' (line 1971)
    ft_126110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1971, 13), 'ft')
    # Obtaining the member '__getitem__' of a type (line 1971)
    getitem___126111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1971, 13), ft_126110, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1971)
    subscript_call_result_126112 = invoke(stypy.reporting.localization.Localization(__file__, 1971, 13), getitem___126111, tuple_call_result_126109)
    
    # Assigning a type to the variable 'ft' (line 1971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1971, 8), 'ft', subscript_call_result_126112)
    
    # Assigning a Call to a Name (line 1972):
    
    # Assigning a Call to a Name (line 1972):
    
    # Call to ravel(...): (line 1972)
    # Processing the call arguments (line 1972)
    # Getting the type of 'ft' (line 1972)
    ft_126115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1972, 25), 'ft', False)
    # Processing the call keyword arguments (line 1972)
    kwargs_126116 = {}
    # Getting the type of 'numpy' (line 1972)
    numpy_126113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1972, 13), 'numpy', False)
    # Obtaining the member 'ravel' of a type (line 1972)
    ravel_126114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1972, 13), numpy_126113, 'ravel')
    # Calling ravel(args, kwargs) (line 1972)
    ravel_call_result_126117 = invoke(stypy.reporting.localization.Localization(__file__, 1972, 13), ravel_126114, *[ft_126115], **kwargs_126116)
    
    # Assigning a type to the variable 'ft' (line 1972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1972, 8), 'ft', ravel_call_result_126117)
    
    # Getting the type of 'ft_inplace' (line 1973)
    ft_inplace_126118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1973, 11), 'ft_inplace')
    # Testing the type of an if condition (line 1973)
    if_condition_126119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1973, 8), ft_inplace_126118)
    # Assigning a type to the variable 'if_condition_126119' (line 1973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1973, 8), 'if_condition_126119', if_condition_126119)
    # SSA begins for if statement (line 1973)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'indices' (line 1974)
    indices_126120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1974, 15), 'indices')
    # Obtaining the member 'dtype' of a type (line 1974)
    dtype_126121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1974, 15), indices_126120, 'dtype')
    # Obtaining the member 'type' of a type (line 1974)
    type_126122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1974, 15), dtype_126121, 'type')
    # Getting the type of 'numpy' (line 1974)
    numpy_126123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1974, 37), 'numpy')
    # Obtaining the member 'int32' of a type (line 1974)
    int32_126124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1974, 37), numpy_126123, 'int32')
    # Applying the binary operator '!=' (line 1974)
    result_ne_126125 = python_operator(stypy.reporting.localization.Localization(__file__, 1974, 15), '!=', type_126122, int32_126124)
    
    # Testing the type of an if condition (line 1974)
    if_condition_126126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1974, 12), result_ne_126125)
    # Assigning a type to the variable 'if_condition_126126' (line 1974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1974, 12), 'if_condition_126126', if_condition_126126)
    # SSA begins for if statement (line 1974)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1975)
    # Processing the call arguments (line 1975)
    str_126128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1975, 35), 'str', 'indices must of int32 type')
    # Processing the call keyword arguments (line 1975)
    kwargs_126129 = {}
    # Getting the type of 'RuntimeError' (line 1975)
    RuntimeError_126127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1975, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1975)
    RuntimeError_call_result_126130 = invoke(stypy.reporting.localization.Localization(__file__, 1975, 22), RuntimeError_126127, *[str_126128], **kwargs_126129)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1975, 16), RuntimeError_call_result_126130, 'raise parameter', BaseException)
    # SSA join for if statement (line 1974)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'indices' (line 1976)
    indices_126131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1976, 15), 'indices')
    # Obtaining the member 'shape' of a type (line 1976)
    shape_126132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1976, 15), indices_126131, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1976)
    tuple_126133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1976, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1976)
    # Adding element type (line 1976)
    # Getting the type of 'dt' (line 1976)
    dt_126134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1976, 33), 'dt')
    # Obtaining the member 'ndim' of a type (line 1976)
    ndim_126135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1976, 33), dt_126134, 'ndim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1976, 33), tuple_126133, ndim_126135)
    
    # Getting the type of 'dt' (line 1976)
    dt_126136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1976, 45), 'dt')
    # Obtaining the member 'shape' of a type (line 1976)
    shape_126137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1976, 45), dt_126136, 'shape')
    # Applying the binary operator '+' (line 1976)
    result_add_126138 = python_operator(stypy.reporting.localization.Localization(__file__, 1976, 32), '+', tuple_126133, shape_126137)
    
    # Applying the binary operator '!=' (line 1976)
    result_ne_126139 = python_operator(stypy.reporting.localization.Localization(__file__, 1976, 15), '!=', shape_126132, result_add_126138)
    
    # Testing the type of an if condition (line 1976)
    if_condition_126140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1976, 12), result_ne_126139)
    # Assigning a type to the variable 'if_condition_126140' (line 1976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1976, 12), 'if_condition_126140', if_condition_126140)
    # SSA begins for if statement (line 1976)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1977)
    # Processing the call arguments (line 1977)
    str_126142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1977, 35), 'str', 'indices has wrong shape')
    # Processing the call keyword arguments (line 1977)
    kwargs_126143 = {}
    # Getting the type of 'RuntimeError' (line 1977)
    RuntimeError_126141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1977, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1977)
    RuntimeError_call_result_126144 = invoke(stypy.reporting.localization.Localization(__file__, 1977, 22), RuntimeError_126141, *[str_126142], **kwargs_126143)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1977, 16), RuntimeError_call_result_126144, 'raise parameter', BaseException)
    # SSA join for if statement (line 1976)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1978):
    
    # Assigning a Name to a Name (line 1978):
    # Getting the type of 'indices' (line 1978)
    indices_126145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1978, 18), 'indices')
    # Assigning a type to the variable 'tmp' (line 1978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1978, 12), 'tmp', indices_126145)
    # SSA branch for the else part of an if statement (line 1973)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1980):
    
    # Assigning a Call to a Name (line 1980):
    
    # Call to indices(...): (line 1980)
    # Processing the call arguments (line 1980)
    # Getting the type of 'dt' (line 1980)
    dt_126148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1980, 32), 'dt', False)
    # Obtaining the member 'shape' of a type (line 1980)
    shape_126149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1980, 32), dt_126148, 'shape')
    # Processing the call keyword arguments (line 1980)
    # Getting the type of 'numpy' (line 1980)
    numpy_126150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1980, 48), 'numpy', False)
    # Obtaining the member 'int32' of a type (line 1980)
    int32_126151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1980, 48), numpy_126150, 'int32')
    keyword_126152 = int32_126151
    kwargs_126153 = {'dtype': keyword_126152}
    # Getting the type of 'numpy' (line 1980)
    numpy_126146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1980, 18), 'numpy', False)
    # Obtaining the member 'indices' of a type (line 1980)
    indices_126147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1980, 18), numpy_126146, 'indices')
    # Calling indices(args, kwargs) (line 1980)
    indices_call_result_126154 = invoke(stypy.reporting.localization.Localization(__file__, 1980, 18), indices_126147, *[shape_126149], **kwargs_126153)
    
    # Assigning a type to the variable 'tmp' (line 1980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1980, 12), 'tmp', indices_call_result_126154)
    # SSA join for if statement (line 1973)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 1981)
    # Processing the call arguments (line 1981)
    
    # Obtaining the type of the subscript
    int_126156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1981, 34), 'int')
    # Getting the type of 'tmp' (line 1981)
    tmp_126157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1981, 24), 'tmp', False)
    # Obtaining the member 'shape' of a type (line 1981)
    shape_126158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1981, 24), tmp_126157, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1981)
    getitem___126159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1981, 24), shape_126158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1981)
    subscript_call_result_126160 = invoke(stypy.reporting.localization.Localization(__file__, 1981, 24), getitem___126159, int_126156)
    
    # Processing the call keyword arguments (line 1981)
    kwargs_126161 = {}
    # Getting the type of 'range' (line 1981)
    range_126155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1981, 18), 'range', False)
    # Calling range(args, kwargs) (line 1981)
    range_call_result_126162 = invoke(stypy.reporting.localization.Localization(__file__, 1981, 18), range_126155, *[subscript_call_result_126160], **kwargs_126161)
    
    # Testing the type of a for loop iterable (line 1981)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1981, 8), range_call_result_126162)
    # Getting the type of the for loop variable (line 1981)
    for_loop_var_126163 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1981, 8), range_call_result_126162)
    # Assigning a type to the variable 'ii' (line 1981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1981, 8), 'ii', for_loop_var_126163)
    # SSA begins for a for statement (line 1981)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 1982):
    
    # Assigning a Subscript to a Name (line 1982):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ft' (line 1982)
    ft_126164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1982, 45), 'ft')
    
    # Call to ravel(...): (line 1982)
    # Processing the call arguments (line 1982)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1982)
    ii_126167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1982, 35), 'ii', False)
    Ellipsis_126168 = Ellipsis
    # Getting the type of 'tmp' (line 1982)
    tmp_126169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1982, 31), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 1982)
    getitem___126170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1982, 31), tmp_126169, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1982)
    subscript_call_result_126171 = invoke(stypy.reporting.localization.Localization(__file__, 1982, 31), getitem___126170, (ii_126167, Ellipsis_126168))
    
    # Processing the call keyword arguments (line 1982)
    kwargs_126172 = {}
    # Getting the type of 'numpy' (line 1982)
    numpy_126165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1982, 19), 'numpy', False)
    # Obtaining the member 'ravel' of a type (line 1982)
    ravel_126166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1982, 19), numpy_126165, 'ravel')
    # Calling ravel(args, kwargs) (line 1982)
    ravel_call_result_126173 = invoke(stypy.reporting.localization.Localization(__file__, 1982, 19), ravel_126166, *[subscript_call_result_126171], **kwargs_126172)
    
    # Obtaining the member '__getitem__' of a type (line 1982)
    getitem___126174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1982, 19), ravel_call_result_126173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1982)
    subscript_call_result_126175 = invoke(stypy.reporting.localization.Localization(__file__, 1982, 19), getitem___126174, ft_126164)
    
    # Assigning a type to the variable 'rtmp' (line 1982)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1982, 12), 'rtmp', subscript_call_result_126175)
    
    # Assigning a Attribute to a Attribute (line 1983):
    
    # Assigning a Attribute to a Attribute (line 1983):
    # Getting the type of 'dt' (line 1983)
    dt_126176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1983, 25), 'dt')
    # Obtaining the member 'shape' of a type (line 1983)
    shape_126177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1983, 25), dt_126176, 'shape')
    # Getting the type of 'rtmp' (line 1983)
    rtmp_126178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1983, 12), 'rtmp')
    # Setting the type of the member 'shape' of a type (line 1983)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1983, 12), rtmp_126178, 'shape', shape_126177)
    
    # Assigning a Name to a Subscript (line 1984):
    
    # Assigning a Name to a Subscript (line 1984):
    # Getting the type of 'rtmp' (line 1984)
    rtmp_126179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1984, 27), 'rtmp')
    # Getting the type of 'tmp' (line 1984)
    tmp_126180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1984, 12), 'tmp')
    # Getting the type of 'ii' (line 1984)
    ii_126181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1984, 16), 'ii')
    Ellipsis_126182 = Ellipsis
    # Storing an element on a container (line 1984)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1984, 12), tmp_126180, ((ii_126181, Ellipsis_126182), rtmp_126179))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1985):
    
    # Assigning a Name to a Name (line 1985):
    # Getting the type of 'tmp' (line 1985)
    tmp_126183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1985, 13), 'tmp')
    # Assigning a type to the variable 'ft' (line 1985)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1985, 8), 'ft', tmp_126183)
    # SSA join for if statement (line 1970)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 1988):
    
    # Assigning a List to a Name (line 1988):
    
    # Obtaining an instance of the builtin type 'list' (line 1988)
    list_126184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1988, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1988)
    
    # Assigning a type to the variable 'result' (line 1988)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1988, 4), 'result', list_126184)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'return_distances' (line 1989)
    return_distances_126185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1989, 7), 'return_distances')
    
    # Getting the type of 'dt_inplace' (line 1989)
    dt_inplace_126186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1989, 32), 'dt_inplace')
    # Applying the 'not' unary operator (line 1989)
    result_not__126187 = python_operator(stypy.reporting.localization.Localization(__file__, 1989, 28), 'not', dt_inplace_126186)
    
    # Applying the binary operator 'and' (line 1989)
    result_and_keyword_126188 = python_operator(stypy.reporting.localization.Localization(__file__, 1989, 7), 'and', return_distances_126185, result_not__126187)
    
    # Testing the type of an if condition (line 1989)
    if_condition_126189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1989, 4), result_and_keyword_126188)
    # Assigning a type to the variable 'if_condition_126189' (line 1989)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1989, 4), 'if_condition_126189', if_condition_126189)
    # SSA begins for if statement (line 1989)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 1990)
    # Processing the call arguments (line 1990)
    # Getting the type of 'dt' (line 1990)
    dt_126192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1990, 22), 'dt', False)
    # Processing the call keyword arguments (line 1990)
    kwargs_126193 = {}
    # Getting the type of 'result' (line 1990)
    result_126190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1990, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 1990)
    append_126191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1990, 8), result_126190, 'append')
    # Calling append(args, kwargs) (line 1990)
    append_call_result_126194 = invoke(stypy.reporting.localization.Localization(__file__, 1990, 8), append_126191, *[dt_126192], **kwargs_126193)
    
    # SSA join for if statement (line 1989)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'return_indices' (line 1991)
    return_indices_126195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1991, 7), 'return_indices')
    
    # Getting the type of 'ft_inplace' (line 1991)
    ft_inplace_126196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1991, 30), 'ft_inplace')
    # Applying the 'not' unary operator (line 1991)
    result_not__126197 = python_operator(stypy.reporting.localization.Localization(__file__, 1991, 26), 'not', ft_inplace_126196)
    
    # Applying the binary operator 'and' (line 1991)
    result_and_keyword_126198 = python_operator(stypy.reporting.localization.Localization(__file__, 1991, 7), 'and', return_indices_126195, result_not__126197)
    
    # Testing the type of an if condition (line 1991)
    if_condition_126199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1991, 4), result_and_keyword_126198)
    # Assigning a type to the variable 'if_condition_126199' (line 1991)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1991, 4), 'if_condition_126199', if_condition_126199)
    # SSA begins for if statement (line 1991)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 1992)
    # Processing the call arguments (line 1992)
    # Getting the type of 'ft' (line 1992)
    ft_126202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1992, 22), 'ft', False)
    # Processing the call keyword arguments (line 1992)
    kwargs_126203 = {}
    # Getting the type of 'result' (line 1992)
    result_126200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1992, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 1992)
    append_126201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1992, 8), result_126200, 'append')
    # Calling append(args, kwargs) (line 1992)
    append_call_result_126204 = invoke(stypy.reporting.localization.Localization(__file__, 1992, 8), append_126201, *[ft_126202], **kwargs_126203)
    
    # SSA join for if statement (line 1991)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1994)
    # Processing the call arguments (line 1994)
    # Getting the type of 'result' (line 1994)
    result_126206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1994, 11), 'result', False)
    # Processing the call keyword arguments (line 1994)
    kwargs_126207 = {}
    # Getting the type of 'len' (line 1994)
    len_126205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1994, 7), 'len', False)
    # Calling len(args, kwargs) (line 1994)
    len_call_result_126208 = invoke(stypy.reporting.localization.Localization(__file__, 1994, 7), len_126205, *[result_126206], **kwargs_126207)
    
    int_126209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1994, 22), 'int')
    # Applying the binary operator '==' (line 1994)
    result_eq_126210 = python_operator(stypy.reporting.localization.Localization(__file__, 1994, 7), '==', len_call_result_126208, int_126209)
    
    # Testing the type of an if condition (line 1994)
    if_condition_126211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1994, 4), result_eq_126210)
    # Assigning a type to the variable 'if_condition_126211' (line 1994)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1994, 4), 'if_condition_126211', if_condition_126211)
    # SSA begins for if statement (line 1994)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tuple(...): (line 1995)
    # Processing the call arguments (line 1995)
    # Getting the type of 'result' (line 1995)
    result_126213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1995, 21), 'result', False)
    # Processing the call keyword arguments (line 1995)
    kwargs_126214 = {}
    # Getting the type of 'tuple' (line 1995)
    tuple_126212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1995, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1995)
    tuple_call_result_126215 = invoke(stypy.reporting.localization.Localization(__file__, 1995, 15), tuple_126212, *[result_126213], **kwargs_126214)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1995, 8), 'stypy_return_type', tuple_call_result_126215)
    # SSA branch for the else part of an if statement (line 1994)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 1996)
    # Processing the call arguments (line 1996)
    # Getting the type of 'result' (line 1996)
    result_126217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1996, 13), 'result', False)
    # Processing the call keyword arguments (line 1996)
    kwargs_126218 = {}
    # Getting the type of 'len' (line 1996)
    len_126216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1996, 9), 'len', False)
    # Calling len(args, kwargs) (line 1996)
    len_call_result_126219 = invoke(stypy.reporting.localization.Localization(__file__, 1996, 9), len_126216, *[result_126217], **kwargs_126218)
    
    int_126220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1996, 24), 'int')
    # Applying the binary operator '==' (line 1996)
    result_eq_126221 = python_operator(stypy.reporting.localization.Localization(__file__, 1996, 9), '==', len_call_result_126219, int_126220)
    
    # Testing the type of an if condition (line 1996)
    if_condition_126222 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1996, 9), result_eq_126221)
    # Assigning a type to the variable 'if_condition_126222' (line 1996)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1996, 9), 'if_condition_126222', if_condition_126222)
    # SSA begins for if statement (line 1996)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_126223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1997, 22), 'int')
    # Getting the type of 'result' (line 1997)
    result_126224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1997, 15), 'result')
    # Obtaining the member '__getitem__' of a type (line 1997)
    getitem___126225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1997, 15), result_126224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1997)
    subscript_call_result_126226 = invoke(stypy.reporting.localization.Localization(__file__, 1997, 15), getitem___126225, int_126223)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1997)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1997, 8), 'stypy_return_type', subscript_call_result_126226)
    # SSA branch for the else part of an if statement (line 1996)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'None' (line 1999)
    None_126227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1999, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 1999)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1999, 8), 'stypy_return_type', None_126227)
    # SSA join for if statement (line 1996)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1994)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'distance_transform_cdt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'distance_transform_cdt' in the type store
    # Getting the type of 'stypy_return_type' (line 1885)
    stypy_return_type_126228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1885, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126228)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'distance_transform_cdt'
    return stypy_return_type_126228

# Assigning a type to the variable 'distance_transform_cdt' (line 1885)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1885, 0), 'distance_transform_cdt', distance_transform_cdt)

@norecursion
def distance_transform_edt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2002)
    None_126229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2002, 43), 'None')
    # Getting the type of 'True' (line 2003)
    True_126230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2003, 41), 'True')
    # Getting the type of 'False' (line 2003)
    False_126231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2003, 62), 'False')
    # Getting the type of 'None' (line 2004)
    None_126232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2004, 34), 'None')
    # Getting the type of 'None' (line 2004)
    None_126233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2004, 48), 'None')
    defaults = [None_126229, True_126230, False_126231, None_126232, None_126233]
    # Create a new context for function 'distance_transform_edt'
    module_type_store = module_type_store.open_function_context('distance_transform_edt', 2002, 0, False)
    
    # Passed parameters checking function
    distance_transform_edt.stypy_localization = localization
    distance_transform_edt.stypy_type_of_self = None
    distance_transform_edt.stypy_type_store = module_type_store
    distance_transform_edt.stypy_function_name = 'distance_transform_edt'
    distance_transform_edt.stypy_param_names_list = ['input', 'sampling', 'return_distances', 'return_indices', 'distances', 'indices']
    distance_transform_edt.stypy_varargs_param_name = None
    distance_transform_edt.stypy_kwargs_param_name = None
    distance_transform_edt.stypy_call_defaults = defaults
    distance_transform_edt.stypy_call_varargs = varargs
    distance_transform_edt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'distance_transform_edt', ['input', 'sampling', 'return_distances', 'return_indices', 'distances', 'indices'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'distance_transform_edt', localization, ['input', 'sampling', 'return_distances', 'return_indices', 'distances', 'indices'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'distance_transform_edt(...)' code ##################

    str_126234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2111, (-1)), 'str', '\n    Exact euclidean distance transform.\n\n    In addition to the distance transform, the feature transform can\n    be calculated. In this case the index of the closest background\n    element is returned along the first axis of the result.\n\n    Parameters\n    ----------\n    input : array_like\n        Input data to transform. Can be any type but will be converted\n        into binary: 1 wherever input equates to True, 0 elsewhere.\n    sampling : float or int, or sequence of same, optional\n        Spacing of elements along each dimension. If a sequence, must be of\n        length equal to the input rank; if a single number, this is used for\n        all axes. If not specified, a grid spacing of unity is implied.\n    return_distances : bool, optional\n        Whether to return distance matrix. At least one of\n        return_distances/return_indices must be True. Default is True.\n    return_indices : bool, optional\n        Whether to return indices matrix. Default is False.\n    distances : ndarray, optional\n        Used for output of distance array, must be of type float64.\n    indices : ndarray, optional\n        Used for output of indices, must be of type int32.\n\n    Returns\n    -------\n    distance_transform_edt : ndarray or list of ndarrays\n        Either distance matrix, index matrix, or a list of the two,\n        depending on `return_x` flags and `distance` and `indices`\n        input parameters.\n\n    Notes\n    -----\n    The euclidean distance transform gives values of the euclidean\n    distance::\n\n                    n\n      y_i = sqrt(sum (x[i]-b[i])**2)\n                    i\n\n    where b[i] is the background point (value 0) with the smallest\n    Euclidean distance to input points x[i], and n is the\n    number of dimensions.\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.array(([0,1,1,1,1],\n    ...               [0,0,1,1,1],\n    ...               [0,1,1,1,1],\n    ...               [0,1,1,1,0],\n    ...               [0,1,1,0,0]))\n    >>> ndimage.distance_transform_edt(a)\n    array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],\n           [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],\n           [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],\n           [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],\n           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])\n\n    With a sampling of 2 units along x, 1 along y:\n\n    >>> ndimage.distance_transform_edt(a, sampling=[2,1])\n    array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],\n           [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],\n           [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],\n           [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],\n           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])\n\n    Asking for indices as well:\n\n    >>> edt, inds = ndimage.distance_transform_edt(a, return_indices=True)\n    >>> inds\n    array([[[0, 0, 1, 1, 3],\n            [1, 1, 1, 1, 3],\n            [2, 2, 1, 3, 3],\n            [3, 3, 4, 4, 3],\n            [4, 4, 4, 4, 4]],\n           [[0, 0, 1, 1, 4],\n            [0, 1, 1, 1, 4],\n            [0, 0, 1, 4, 4],\n            [0, 0, 3, 3, 4],\n            [0, 0, 3, 3, 4]]])\n\n    With arrays provided for inplace outputs:\n\n    >>> indices = np.zeros(((np.ndim(a),) + a.shape), dtype=np.int32)\n    >>> ndimage.distance_transform_edt(a, return_indices=True, indices=indices)\n    array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],\n           [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],\n           [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],\n           [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],\n           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])\n    >>> indices\n    array([[[0, 0, 1, 1, 3],\n            [1, 1, 1, 1, 3],\n            [2, 2, 1, 3, 3],\n            [3, 3, 4, 4, 3],\n            [4, 4, 4, 4, 4]],\n           [[0, 0, 1, 1, 4],\n            [0, 1, 1, 1, 4],\n            [0, 0, 1, 4, 4],\n            [0, 0, 3, 3, 4],\n            [0, 0, 3, 3, 4]]])\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'return_distances' (line 2112)
    return_distances_126235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2112, 12), 'return_distances')
    # Applying the 'not' unary operator (line 2112)
    result_not__126236 = python_operator(stypy.reporting.localization.Localization(__file__, 2112, 8), 'not', return_distances_126235)
    
    
    # Getting the type of 'return_indices' (line 2112)
    return_indices_126237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2112, 39), 'return_indices')
    # Applying the 'not' unary operator (line 2112)
    result_not__126238 = python_operator(stypy.reporting.localization.Localization(__file__, 2112, 35), 'not', return_indices_126237)
    
    # Applying the binary operator 'and' (line 2112)
    result_and_keyword_126239 = python_operator(stypy.reporting.localization.Localization(__file__, 2112, 7), 'and', result_not__126236, result_not__126238)
    
    # Testing the type of an if condition (line 2112)
    if_condition_126240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2112, 4), result_and_keyword_126239)
    # Assigning a type to the variable 'if_condition_126240' (line 2112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2112, 4), 'if_condition_126240', if_condition_126240)
    # SSA begins for if statement (line 2112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 2113):
    
    # Assigning a Str to a Name (line 2113):
    str_126241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2113, 14), 'str', 'at least one of distances/indices must be specified')
    # Assigning a type to the variable 'msg' (line 2113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2113, 8), 'msg', str_126241)
    
    # Call to RuntimeError(...): (line 2114)
    # Processing the call arguments (line 2114)
    # Getting the type of 'msg' (line 2114)
    msg_126243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2114, 27), 'msg', False)
    # Processing the call keyword arguments (line 2114)
    kwargs_126244 = {}
    # Getting the type of 'RuntimeError' (line 2114)
    RuntimeError_126242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2114, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 2114)
    RuntimeError_call_result_126245 = invoke(stypy.reporting.localization.Localization(__file__, 2114, 14), RuntimeError_126242, *[msg_126243], **kwargs_126244)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 2114, 8), RuntimeError_call_result_126245, 'raise parameter', BaseException)
    # SSA join for if statement (line 2112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 2116):
    
    # Assigning a Call to a Name (line 2116):
    
    # Call to isinstance(...): (line 2116)
    # Processing the call arguments (line 2116)
    # Getting the type of 'indices' (line 2116)
    indices_126247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2116, 28), 'indices', False)
    # Getting the type of 'numpy' (line 2116)
    numpy_126248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2116, 37), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 2116)
    ndarray_126249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2116, 37), numpy_126248, 'ndarray')
    # Processing the call keyword arguments (line 2116)
    kwargs_126250 = {}
    # Getting the type of 'isinstance' (line 2116)
    isinstance_126246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2116, 17), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2116)
    isinstance_call_result_126251 = invoke(stypy.reporting.localization.Localization(__file__, 2116, 17), isinstance_126246, *[indices_126247, ndarray_126249], **kwargs_126250)
    
    # Assigning a type to the variable 'ft_inplace' (line 2116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2116, 4), 'ft_inplace', isinstance_call_result_126251)
    
    # Assigning a Call to a Name (line 2117):
    
    # Assigning a Call to a Name (line 2117):
    
    # Call to isinstance(...): (line 2117)
    # Processing the call arguments (line 2117)
    # Getting the type of 'distances' (line 2117)
    distances_126253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2117, 28), 'distances', False)
    # Getting the type of 'numpy' (line 2117)
    numpy_126254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2117, 39), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 2117)
    ndarray_126255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2117, 39), numpy_126254, 'ndarray')
    # Processing the call keyword arguments (line 2117)
    kwargs_126256 = {}
    # Getting the type of 'isinstance' (line 2117)
    isinstance_126252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2117, 17), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2117)
    isinstance_call_result_126257 = invoke(stypy.reporting.localization.Localization(__file__, 2117, 17), isinstance_126252, *[distances_126253, ndarray_126255], **kwargs_126256)
    
    # Assigning a type to the variable 'dt_inplace' (line 2117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2117, 4), 'dt_inplace', isinstance_call_result_126257)
    
    # Assigning a Call to a Name (line 2119):
    
    # Assigning a Call to a Name (line 2119):
    
    # Call to atleast_1d(...): (line 2119)
    # Processing the call arguments (line 2119)
    
    # Call to astype(...): (line 2119)
    # Processing the call arguments (line 2119)
    # Getting the type of 'numpy' (line 2119)
    numpy_126268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2119, 61), 'numpy', False)
    # Obtaining the member 'int8' of a type (line 2119)
    int8_126269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2119, 61), numpy_126268, 'int8')
    # Processing the call keyword arguments (line 2119)
    kwargs_126270 = {}
    
    # Call to where(...): (line 2119)
    # Processing the call arguments (line 2119)
    # Getting the type of 'input' (line 2119)
    input_126262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2119, 41), 'input', False)
    int_126263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2119, 48), 'int')
    int_126264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2119, 51), 'int')
    # Processing the call keyword arguments (line 2119)
    kwargs_126265 = {}
    # Getting the type of 'numpy' (line 2119)
    numpy_126260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2119, 29), 'numpy', False)
    # Obtaining the member 'where' of a type (line 2119)
    where_126261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2119, 29), numpy_126260, 'where')
    # Calling where(args, kwargs) (line 2119)
    where_call_result_126266 = invoke(stypy.reporting.localization.Localization(__file__, 2119, 29), where_126261, *[input_126262, int_126263, int_126264], **kwargs_126265)
    
    # Obtaining the member 'astype' of a type (line 2119)
    astype_126267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2119, 29), where_call_result_126266, 'astype')
    # Calling astype(args, kwargs) (line 2119)
    astype_call_result_126271 = invoke(stypy.reporting.localization.Localization(__file__, 2119, 29), astype_126267, *[int8_126269], **kwargs_126270)
    
    # Processing the call keyword arguments (line 2119)
    kwargs_126272 = {}
    # Getting the type of 'numpy' (line 2119)
    numpy_126258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2119, 12), 'numpy', False)
    # Obtaining the member 'atleast_1d' of a type (line 2119)
    atleast_1d_126259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2119, 12), numpy_126258, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 2119)
    atleast_1d_call_result_126273 = invoke(stypy.reporting.localization.Localization(__file__, 2119, 12), atleast_1d_126259, *[astype_call_result_126271], **kwargs_126272)
    
    # Assigning a type to the variable 'input' (line 2119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2119, 4), 'input', atleast_1d_call_result_126273)
    
    # Type idiom detected: calculating its left and rigth part (line 2120)
    # Getting the type of 'sampling' (line 2120)
    sampling_126274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2120, 4), 'sampling')
    # Getting the type of 'None' (line 2120)
    None_126275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2120, 23), 'None')
    
    (may_be_126276, more_types_in_union_126277) = may_not_be_none(sampling_126274, None_126275)

    if may_be_126276:

        if more_types_in_union_126277:
            # Runtime conditional SSA (line 2120)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 2121):
        
        # Assigning a Call to a Name (line 2121):
        
        # Call to _normalize_sequence(...): (line 2121)
        # Processing the call arguments (line 2121)
        # Getting the type of 'sampling' (line 2121)
        sampling_126280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2121, 51), 'sampling', False)
        # Getting the type of 'input' (line 2121)
        input_126281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2121, 61), 'input', False)
        # Obtaining the member 'ndim' of a type (line 2121)
        ndim_126282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2121, 61), input_126281, 'ndim')
        # Processing the call keyword arguments (line 2121)
        kwargs_126283 = {}
        # Getting the type of '_ni_support' (line 2121)
        _ni_support_126278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2121, 19), '_ni_support', False)
        # Obtaining the member '_normalize_sequence' of a type (line 2121)
        _normalize_sequence_126279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2121, 19), _ni_support_126278, '_normalize_sequence')
        # Calling _normalize_sequence(args, kwargs) (line 2121)
        _normalize_sequence_call_result_126284 = invoke(stypy.reporting.localization.Localization(__file__, 2121, 19), _normalize_sequence_126279, *[sampling_126280, ndim_126282], **kwargs_126283)
        
        # Assigning a type to the variable 'sampling' (line 2121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2121, 8), 'sampling', _normalize_sequence_call_result_126284)
        
        # Assigning a Call to a Name (line 2122):
        
        # Assigning a Call to a Name (line 2122):
        
        # Call to asarray(...): (line 2122)
        # Processing the call arguments (line 2122)
        # Getting the type of 'sampling' (line 2122)
        sampling_126287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2122, 33), 'sampling', False)
        # Processing the call keyword arguments (line 2122)
        # Getting the type of 'numpy' (line 2122)
        numpy_126288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2122, 49), 'numpy', False)
        # Obtaining the member 'float64' of a type (line 2122)
        float64_126289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2122, 49), numpy_126288, 'float64')
        keyword_126290 = float64_126289
        kwargs_126291 = {'dtype': keyword_126290}
        # Getting the type of 'numpy' (line 2122)
        numpy_126285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2122, 19), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 2122)
        asarray_126286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2122, 19), numpy_126285, 'asarray')
        # Calling asarray(args, kwargs) (line 2122)
        asarray_call_result_126292 = invoke(stypy.reporting.localization.Localization(__file__, 2122, 19), asarray_126286, *[sampling_126287], **kwargs_126291)
        
        # Assigning a type to the variable 'sampling' (line 2122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2122, 8), 'sampling', asarray_call_result_126292)
        
        
        # Getting the type of 'sampling' (line 2123)
        sampling_126293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2123, 15), 'sampling')
        # Obtaining the member 'flags' of a type (line 2123)
        flags_126294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2123, 15), sampling_126293, 'flags')
        # Obtaining the member 'contiguous' of a type (line 2123)
        contiguous_126295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2123, 15), flags_126294, 'contiguous')
        # Applying the 'not' unary operator (line 2123)
        result_not__126296 = python_operator(stypy.reporting.localization.Localization(__file__, 2123, 11), 'not', contiguous_126295)
        
        # Testing the type of an if condition (line 2123)
        if_condition_126297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2123, 8), result_not__126296)
        # Assigning a type to the variable 'if_condition_126297' (line 2123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2123, 8), 'if_condition_126297', if_condition_126297)
        # SSA begins for if statement (line 2123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 2124):
        
        # Assigning a Call to a Name (line 2124):
        
        # Call to copy(...): (line 2124)
        # Processing the call keyword arguments (line 2124)
        kwargs_126300 = {}
        # Getting the type of 'sampling' (line 2124)
        sampling_126298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2124, 23), 'sampling', False)
        # Obtaining the member 'copy' of a type (line 2124)
        copy_126299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2124, 23), sampling_126298, 'copy')
        # Calling copy(args, kwargs) (line 2124)
        copy_call_result_126301 = invoke(stypy.reporting.localization.Localization(__file__, 2124, 23), copy_126299, *[], **kwargs_126300)
        
        # Assigning a type to the variable 'sampling' (line 2124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2124, 12), 'sampling', copy_call_result_126301)
        # SSA join for if statement (line 2123)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_126277:
            # SSA join for if statement (line 2120)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'ft_inplace' (line 2126)
    ft_inplace_126302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2126, 7), 'ft_inplace')
    # Testing the type of an if condition (line 2126)
    if_condition_126303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2126, 4), ft_inplace_126302)
    # Assigning a type to the variable 'if_condition_126303' (line 2126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2126, 4), 'if_condition_126303', if_condition_126303)
    # SSA begins for if statement (line 2126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 2127):
    
    # Assigning a Name to a Name (line 2127):
    # Getting the type of 'indices' (line 2127)
    indices_126304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2127, 13), 'indices')
    # Assigning a type to the variable 'ft' (line 2127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2127, 8), 'ft', indices_126304)
    
    
    # Getting the type of 'ft' (line 2128)
    ft_126305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2128, 11), 'ft')
    # Obtaining the member 'shape' of a type (line 2128)
    shape_126306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2128, 11), ft_126305, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 2128)
    tuple_126307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2128, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 2128)
    # Adding element type (line 2128)
    # Getting the type of 'input' (line 2128)
    input_126308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2128, 24), 'input')
    # Obtaining the member 'ndim' of a type (line 2128)
    ndim_126309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2128, 24), input_126308, 'ndim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2128, 24), tuple_126307, ndim_126309)
    
    # Getting the type of 'input' (line 2128)
    input_126310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2128, 39), 'input')
    # Obtaining the member 'shape' of a type (line 2128)
    shape_126311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2128, 39), input_126310, 'shape')
    # Applying the binary operator '+' (line 2128)
    result_add_126312 = python_operator(stypy.reporting.localization.Localization(__file__, 2128, 23), '+', tuple_126307, shape_126311)
    
    # Applying the binary operator '!=' (line 2128)
    result_ne_126313 = python_operator(stypy.reporting.localization.Localization(__file__, 2128, 11), '!=', shape_126306, result_add_126312)
    
    # Testing the type of an if condition (line 2128)
    if_condition_126314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2128, 8), result_ne_126313)
    # Assigning a type to the variable 'if_condition_126314' (line 2128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2128, 8), 'if_condition_126314', if_condition_126314)
    # SSA begins for if statement (line 2128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 2129)
    # Processing the call arguments (line 2129)
    str_126316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2129, 31), 'str', 'indices has wrong shape')
    # Processing the call keyword arguments (line 2129)
    kwargs_126317 = {}
    # Getting the type of 'RuntimeError' (line 2129)
    RuntimeError_126315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2129, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 2129)
    RuntimeError_call_result_126318 = invoke(stypy.reporting.localization.Localization(__file__, 2129, 18), RuntimeError_126315, *[str_126316], **kwargs_126317)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 2129, 12), RuntimeError_call_result_126318, 'raise parameter', BaseException)
    # SSA join for if statement (line 2128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ft' (line 2130)
    ft_126319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2130, 11), 'ft')
    # Obtaining the member 'dtype' of a type (line 2130)
    dtype_126320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2130, 11), ft_126319, 'dtype')
    # Obtaining the member 'type' of a type (line 2130)
    type_126321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2130, 11), dtype_126320, 'type')
    # Getting the type of 'numpy' (line 2130)
    numpy_126322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2130, 28), 'numpy')
    # Obtaining the member 'int32' of a type (line 2130)
    int32_126323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2130, 28), numpy_126322, 'int32')
    # Applying the binary operator '!=' (line 2130)
    result_ne_126324 = python_operator(stypy.reporting.localization.Localization(__file__, 2130, 11), '!=', type_126321, int32_126323)
    
    # Testing the type of an if condition (line 2130)
    if_condition_126325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2130, 8), result_ne_126324)
    # Assigning a type to the variable 'if_condition_126325' (line 2130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2130, 8), 'if_condition_126325', if_condition_126325)
    # SSA begins for if statement (line 2130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 2131)
    # Processing the call arguments (line 2131)
    str_126327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2131, 31), 'str', 'indices must be of int32 type')
    # Processing the call keyword arguments (line 2131)
    kwargs_126328 = {}
    # Getting the type of 'RuntimeError' (line 2131)
    RuntimeError_126326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2131, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 2131)
    RuntimeError_call_result_126329 = invoke(stypy.reporting.localization.Localization(__file__, 2131, 18), RuntimeError_126326, *[str_126327], **kwargs_126328)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 2131, 12), RuntimeError_call_result_126329, 'raise parameter', BaseException)
    # SSA join for if statement (line 2130)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 2126)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 2133):
    
    # Assigning a Call to a Name (line 2133):
    
    # Call to zeros(...): (line 2133)
    # Processing the call arguments (line 2133)
    
    # Obtaining an instance of the builtin type 'tuple' (line 2133)
    tuple_126332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2133, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 2133)
    # Adding element type (line 2133)
    # Getting the type of 'input' (line 2133)
    input_126333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2133, 26), 'input', False)
    # Obtaining the member 'ndim' of a type (line 2133)
    ndim_126334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2133, 26), input_126333, 'ndim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2133, 26), tuple_126332, ndim_126334)
    
    # Getting the type of 'input' (line 2133)
    input_126335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2133, 41), 'input', False)
    # Obtaining the member 'shape' of a type (line 2133)
    shape_126336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2133, 41), input_126335, 'shape')
    # Applying the binary operator '+' (line 2133)
    result_add_126337 = python_operator(stypy.reporting.localization.Localization(__file__, 2133, 25), '+', tuple_126332, shape_126336)
    
    # Processing the call keyword arguments (line 2133)
    # Getting the type of 'numpy' (line 2134)
    numpy_126338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2134, 34), 'numpy', False)
    # Obtaining the member 'int32' of a type (line 2134)
    int32_126339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2134, 34), numpy_126338, 'int32')
    keyword_126340 = int32_126339
    kwargs_126341 = {'dtype': keyword_126340}
    # Getting the type of 'numpy' (line 2133)
    numpy_126330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2133, 13), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 2133)
    zeros_126331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2133, 13), numpy_126330, 'zeros')
    # Calling zeros(args, kwargs) (line 2133)
    zeros_call_result_126342 = invoke(stypy.reporting.localization.Localization(__file__, 2133, 13), zeros_126331, *[result_add_126337], **kwargs_126341)
    
    # Assigning a type to the variable 'ft' (line 2133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2133, 8), 'ft', zeros_call_result_126342)
    # SSA join for if statement (line 2126)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to euclidean_feature_transform(...): (line 2136)
    # Processing the call arguments (line 2136)
    # Getting the type of 'input' (line 2136)
    input_126345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2136, 42), 'input', False)
    # Getting the type of 'sampling' (line 2136)
    sampling_126346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2136, 49), 'sampling', False)
    # Getting the type of 'ft' (line 2136)
    ft_126347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2136, 59), 'ft', False)
    # Processing the call keyword arguments (line 2136)
    kwargs_126348 = {}
    # Getting the type of '_nd_image' (line 2136)
    _nd_image_126343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2136, 4), '_nd_image', False)
    # Obtaining the member 'euclidean_feature_transform' of a type (line 2136)
    euclidean_feature_transform_126344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2136, 4), _nd_image_126343, 'euclidean_feature_transform')
    # Calling euclidean_feature_transform(args, kwargs) (line 2136)
    euclidean_feature_transform_call_result_126349 = invoke(stypy.reporting.localization.Localization(__file__, 2136, 4), euclidean_feature_transform_126344, *[input_126345, sampling_126346, ft_126347], **kwargs_126348)
    
    
    # Getting the type of 'return_distances' (line 2138)
    return_distances_126350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2138, 7), 'return_distances')
    # Testing the type of an if condition (line 2138)
    if_condition_126351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2138, 4), return_distances_126350)
    # Assigning a type to the variable 'if_condition_126351' (line 2138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2138, 4), 'if_condition_126351', if_condition_126351)
    # SSA begins for if statement (line 2138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 2139):
    
    # Assigning a BinOp to a Name (line 2139):
    # Getting the type of 'ft' (line 2139)
    ft_126352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2139, 13), 'ft')
    
    # Call to indices(...): (line 2139)
    # Processing the call arguments (line 2139)
    # Getting the type of 'input' (line 2139)
    input_126355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2139, 32), 'input', False)
    # Obtaining the member 'shape' of a type (line 2139)
    shape_126356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2139, 32), input_126355, 'shape')
    # Processing the call keyword arguments (line 2139)
    # Getting the type of 'ft' (line 2139)
    ft_126357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2139, 51), 'ft', False)
    # Obtaining the member 'dtype' of a type (line 2139)
    dtype_126358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2139, 51), ft_126357, 'dtype')
    keyword_126359 = dtype_126358
    kwargs_126360 = {'dtype': keyword_126359}
    # Getting the type of 'numpy' (line 2139)
    numpy_126353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2139, 18), 'numpy', False)
    # Obtaining the member 'indices' of a type (line 2139)
    indices_126354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2139, 18), numpy_126353, 'indices')
    # Calling indices(args, kwargs) (line 2139)
    indices_call_result_126361 = invoke(stypy.reporting.localization.Localization(__file__, 2139, 18), indices_126354, *[shape_126356], **kwargs_126360)
    
    # Applying the binary operator '-' (line 2139)
    result_sub_126362 = python_operator(stypy.reporting.localization.Localization(__file__, 2139, 13), '-', ft_126352, indices_call_result_126361)
    
    # Assigning a type to the variable 'dt' (line 2139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2139, 8), 'dt', result_sub_126362)
    
    # Assigning a Call to a Name (line 2140):
    
    # Assigning a Call to a Name (line 2140):
    
    # Call to astype(...): (line 2140)
    # Processing the call arguments (line 2140)
    # Getting the type of 'numpy' (line 2140)
    numpy_126365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2140, 23), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 2140)
    float64_126366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2140, 23), numpy_126365, 'float64')
    # Processing the call keyword arguments (line 2140)
    kwargs_126367 = {}
    # Getting the type of 'dt' (line 2140)
    dt_126363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2140, 13), 'dt', False)
    # Obtaining the member 'astype' of a type (line 2140)
    astype_126364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2140, 13), dt_126363, 'astype')
    # Calling astype(args, kwargs) (line 2140)
    astype_call_result_126368 = invoke(stypy.reporting.localization.Localization(__file__, 2140, 13), astype_126364, *[float64_126366], **kwargs_126367)
    
    # Assigning a type to the variable 'dt' (line 2140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2140, 8), 'dt', astype_call_result_126368)
    
    # Type idiom detected: calculating its left and rigth part (line 2141)
    # Getting the type of 'sampling' (line 2141)
    sampling_126369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2141, 8), 'sampling')
    # Getting the type of 'None' (line 2141)
    None_126370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2141, 27), 'None')
    
    (may_be_126371, more_types_in_union_126372) = may_not_be_none(sampling_126369, None_126370)

    if may_be_126371:

        if more_types_in_union_126372:
            # Runtime conditional SSA (line 2141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to range(...): (line 2142)
        # Processing the call arguments (line 2142)
        
        # Call to len(...): (line 2142)
        # Processing the call arguments (line 2142)
        # Getting the type of 'sampling' (line 2142)
        sampling_126375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2142, 32), 'sampling', False)
        # Processing the call keyword arguments (line 2142)
        kwargs_126376 = {}
        # Getting the type of 'len' (line 2142)
        len_126374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2142, 28), 'len', False)
        # Calling len(args, kwargs) (line 2142)
        len_call_result_126377 = invoke(stypy.reporting.localization.Localization(__file__, 2142, 28), len_126374, *[sampling_126375], **kwargs_126376)
        
        # Processing the call keyword arguments (line 2142)
        kwargs_126378 = {}
        # Getting the type of 'range' (line 2142)
        range_126373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2142, 22), 'range', False)
        # Calling range(args, kwargs) (line 2142)
        range_call_result_126379 = invoke(stypy.reporting.localization.Localization(__file__, 2142, 22), range_126373, *[len_call_result_126377], **kwargs_126378)
        
        # Testing the type of a for loop iterable (line 2142)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 2142, 12), range_call_result_126379)
        # Getting the type of the for loop variable (line 2142)
        for_loop_var_126380 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 2142, 12), range_call_result_126379)
        # Assigning a type to the variable 'ii' (line 2142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2142, 12), 'ii', for_loop_var_126380)
        # SSA begins for a for statement (line 2142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'dt' (line 2143)
        dt_126381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2143, 16), 'dt')
        
        # Obtaining the type of the subscript
        # Getting the type of 'ii' (line 2143)
        ii_126382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2143, 19), 'ii')
        Ellipsis_126383 = Ellipsis
        # Getting the type of 'dt' (line 2143)
        dt_126384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2143, 16), 'dt')
        # Obtaining the member '__getitem__' of a type (line 2143)
        getitem___126385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2143, 16), dt_126384, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 2143)
        subscript_call_result_126386 = invoke(stypy.reporting.localization.Localization(__file__, 2143, 16), getitem___126385, (ii_126382, Ellipsis_126383))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ii' (line 2143)
        ii_126387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2143, 40), 'ii')
        # Getting the type of 'sampling' (line 2143)
        sampling_126388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2143, 31), 'sampling')
        # Obtaining the member '__getitem__' of a type (line 2143)
        getitem___126389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2143, 31), sampling_126388, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 2143)
        subscript_call_result_126390 = invoke(stypy.reporting.localization.Localization(__file__, 2143, 31), getitem___126389, ii_126387)
        
        # Applying the binary operator '*=' (line 2143)
        result_imul_126391 = python_operator(stypy.reporting.localization.Localization(__file__, 2143, 16), '*=', subscript_call_result_126386, subscript_call_result_126390)
        # Getting the type of 'dt' (line 2143)
        dt_126392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2143, 16), 'dt')
        # Getting the type of 'ii' (line 2143)
        ii_126393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2143, 19), 'ii')
        Ellipsis_126394 = Ellipsis
        # Storing an element on a container (line 2143)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2143, 16), dt_126392, ((ii_126393, Ellipsis_126394), result_imul_126391))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_126372:
            # SSA join for if statement (line 2141)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to multiply(...): (line 2144)
    # Processing the call arguments (line 2144)
    # Getting the type of 'dt' (line 2144)
    dt_126397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2144, 23), 'dt', False)
    # Getting the type of 'dt' (line 2144)
    dt_126398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2144, 27), 'dt', False)
    # Getting the type of 'dt' (line 2144)
    dt_126399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2144, 31), 'dt', False)
    # Processing the call keyword arguments (line 2144)
    kwargs_126400 = {}
    # Getting the type of 'numpy' (line 2144)
    numpy_126395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2144, 8), 'numpy', False)
    # Obtaining the member 'multiply' of a type (line 2144)
    multiply_126396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2144, 8), numpy_126395, 'multiply')
    # Calling multiply(args, kwargs) (line 2144)
    multiply_call_result_126401 = invoke(stypy.reporting.localization.Localization(__file__, 2144, 8), multiply_126396, *[dt_126397, dt_126398, dt_126399], **kwargs_126400)
    
    
    # Getting the type of 'dt_inplace' (line 2145)
    dt_inplace_126402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2145, 11), 'dt_inplace')
    # Testing the type of an if condition (line 2145)
    if_condition_126403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2145, 8), dt_inplace_126402)
    # Assigning a type to the variable 'if_condition_126403' (line 2145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2145, 8), 'if_condition_126403', if_condition_126403)
    # SSA begins for if statement (line 2145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 2146):
    
    # Assigning a Call to a Name (line 2146):
    
    # Call to reduce(...): (line 2146)
    # Processing the call arguments (line 2146)
    # Getting the type of 'dt' (line 2146)
    dt_126407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2146, 34), 'dt', False)
    # Processing the call keyword arguments (line 2146)
    int_126408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2146, 43), 'int')
    keyword_126409 = int_126408
    kwargs_126410 = {'axis': keyword_126409}
    # Getting the type of 'numpy' (line 2146)
    numpy_126404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2146, 17), 'numpy', False)
    # Obtaining the member 'add' of a type (line 2146)
    add_126405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2146, 17), numpy_126404, 'add')
    # Obtaining the member 'reduce' of a type (line 2146)
    reduce_126406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2146, 17), add_126405, 'reduce')
    # Calling reduce(args, kwargs) (line 2146)
    reduce_call_result_126411 = invoke(stypy.reporting.localization.Localization(__file__, 2146, 17), reduce_126406, *[dt_126407], **kwargs_126410)
    
    # Assigning a type to the variable 'dt' (line 2146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2146, 12), 'dt', reduce_call_result_126411)
    
    
    # Getting the type of 'distances' (line 2147)
    distances_126412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2147, 15), 'distances')
    # Obtaining the member 'shape' of a type (line 2147)
    shape_126413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2147, 15), distances_126412, 'shape')
    # Getting the type of 'dt' (line 2147)
    dt_126414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2147, 34), 'dt')
    # Obtaining the member 'shape' of a type (line 2147)
    shape_126415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2147, 34), dt_126414, 'shape')
    # Applying the binary operator '!=' (line 2147)
    result_ne_126416 = python_operator(stypy.reporting.localization.Localization(__file__, 2147, 15), '!=', shape_126413, shape_126415)
    
    # Testing the type of an if condition (line 2147)
    if_condition_126417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2147, 12), result_ne_126416)
    # Assigning a type to the variable 'if_condition_126417' (line 2147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2147, 12), 'if_condition_126417', if_condition_126417)
    # SSA begins for if statement (line 2147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 2148)
    # Processing the call arguments (line 2148)
    str_126419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2148, 35), 'str', 'indices has wrong shape')
    # Processing the call keyword arguments (line 2148)
    kwargs_126420 = {}
    # Getting the type of 'RuntimeError' (line 2148)
    RuntimeError_126418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2148, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 2148)
    RuntimeError_call_result_126421 = invoke(stypy.reporting.localization.Localization(__file__, 2148, 22), RuntimeError_126418, *[str_126419], **kwargs_126420)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 2148, 16), RuntimeError_call_result_126421, 'raise parameter', BaseException)
    # SSA join for if statement (line 2147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'distances' (line 2149)
    distances_126422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2149, 15), 'distances')
    # Obtaining the member 'dtype' of a type (line 2149)
    dtype_126423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2149, 15), distances_126422, 'dtype')
    # Obtaining the member 'type' of a type (line 2149)
    type_126424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2149, 15), dtype_126423, 'type')
    # Getting the type of 'numpy' (line 2149)
    numpy_126425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2149, 39), 'numpy')
    # Obtaining the member 'float64' of a type (line 2149)
    float64_126426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2149, 39), numpy_126425, 'float64')
    # Applying the binary operator '!=' (line 2149)
    result_ne_126427 = python_operator(stypy.reporting.localization.Localization(__file__, 2149, 15), '!=', type_126424, float64_126426)
    
    # Testing the type of an if condition (line 2149)
    if_condition_126428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2149, 12), result_ne_126427)
    # Assigning a type to the variable 'if_condition_126428' (line 2149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2149, 12), 'if_condition_126428', if_condition_126428)
    # SSA begins for if statement (line 2149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 2150)
    # Processing the call arguments (line 2150)
    str_126430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2150, 35), 'str', 'indices must be of float64 type')
    # Processing the call keyword arguments (line 2150)
    kwargs_126431 = {}
    # Getting the type of 'RuntimeError' (line 2150)
    RuntimeError_126429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2150, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 2150)
    RuntimeError_call_result_126432 = invoke(stypy.reporting.localization.Localization(__file__, 2150, 22), RuntimeError_126429, *[str_126430], **kwargs_126431)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 2150, 16), RuntimeError_call_result_126432, 'raise parameter', BaseException)
    # SSA join for if statement (line 2149)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sqrt(...): (line 2151)
    # Processing the call arguments (line 2151)
    # Getting the type of 'dt' (line 2151)
    dt_126435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2151, 23), 'dt', False)
    # Getting the type of 'distances' (line 2151)
    distances_126436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2151, 27), 'distances', False)
    # Processing the call keyword arguments (line 2151)
    kwargs_126437 = {}
    # Getting the type of 'numpy' (line 2151)
    numpy_126433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2151, 12), 'numpy', False)
    # Obtaining the member 'sqrt' of a type (line 2151)
    sqrt_126434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2151, 12), numpy_126433, 'sqrt')
    # Calling sqrt(args, kwargs) (line 2151)
    sqrt_call_result_126438 = invoke(stypy.reporting.localization.Localization(__file__, 2151, 12), sqrt_126434, *[dt_126435, distances_126436], **kwargs_126437)
    
    # SSA branch for the else part of an if statement (line 2145)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 2153):
    
    # Assigning a Call to a Name (line 2153):
    
    # Call to reduce(...): (line 2153)
    # Processing the call arguments (line 2153)
    # Getting the type of 'dt' (line 2153)
    dt_126442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2153, 34), 'dt', False)
    # Processing the call keyword arguments (line 2153)
    int_126443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2153, 43), 'int')
    keyword_126444 = int_126443
    kwargs_126445 = {'axis': keyword_126444}
    # Getting the type of 'numpy' (line 2153)
    numpy_126439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2153, 17), 'numpy', False)
    # Obtaining the member 'add' of a type (line 2153)
    add_126440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2153, 17), numpy_126439, 'add')
    # Obtaining the member 'reduce' of a type (line 2153)
    reduce_126441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2153, 17), add_126440, 'reduce')
    # Calling reduce(args, kwargs) (line 2153)
    reduce_call_result_126446 = invoke(stypy.reporting.localization.Localization(__file__, 2153, 17), reduce_126441, *[dt_126442], **kwargs_126445)
    
    # Assigning a type to the variable 'dt' (line 2153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2153, 12), 'dt', reduce_call_result_126446)
    
    # Assigning a Call to a Name (line 2154):
    
    # Assigning a Call to a Name (line 2154):
    
    # Call to sqrt(...): (line 2154)
    # Processing the call arguments (line 2154)
    # Getting the type of 'dt' (line 2154)
    dt_126449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2154, 28), 'dt', False)
    # Processing the call keyword arguments (line 2154)
    kwargs_126450 = {}
    # Getting the type of 'numpy' (line 2154)
    numpy_126447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2154, 17), 'numpy', False)
    # Obtaining the member 'sqrt' of a type (line 2154)
    sqrt_126448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2154, 17), numpy_126447, 'sqrt')
    # Calling sqrt(args, kwargs) (line 2154)
    sqrt_call_result_126451 = invoke(stypy.reporting.localization.Localization(__file__, 2154, 17), sqrt_126448, *[dt_126449], **kwargs_126450)
    
    # Assigning a type to the variable 'dt' (line 2154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2154, 12), 'dt', sqrt_call_result_126451)
    # SSA join for if statement (line 2145)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 2138)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 2157):
    
    # Assigning a List to a Name (line 2157):
    
    # Obtaining an instance of the builtin type 'list' (line 2157)
    list_126452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2157, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 2157)
    
    # Assigning a type to the variable 'result' (line 2157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2157, 4), 'result', list_126452)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'return_distances' (line 2158)
    return_distances_126453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2158, 7), 'return_distances')
    
    # Getting the type of 'dt_inplace' (line 2158)
    dt_inplace_126454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2158, 32), 'dt_inplace')
    # Applying the 'not' unary operator (line 2158)
    result_not__126455 = python_operator(stypy.reporting.localization.Localization(__file__, 2158, 28), 'not', dt_inplace_126454)
    
    # Applying the binary operator 'and' (line 2158)
    result_and_keyword_126456 = python_operator(stypy.reporting.localization.Localization(__file__, 2158, 7), 'and', return_distances_126453, result_not__126455)
    
    # Testing the type of an if condition (line 2158)
    if_condition_126457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2158, 4), result_and_keyword_126456)
    # Assigning a type to the variable 'if_condition_126457' (line 2158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2158, 4), 'if_condition_126457', if_condition_126457)
    # SSA begins for if statement (line 2158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 2159)
    # Processing the call arguments (line 2159)
    # Getting the type of 'dt' (line 2159)
    dt_126460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2159, 22), 'dt', False)
    # Processing the call keyword arguments (line 2159)
    kwargs_126461 = {}
    # Getting the type of 'result' (line 2159)
    result_126458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2159, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 2159)
    append_126459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2159, 8), result_126458, 'append')
    # Calling append(args, kwargs) (line 2159)
    append_call_result_126462 = invoke(stypy.reporting.localization.Localization(__file__, 2159, 8), append_126459, *[dt_126460], **kwargs_126461)
    
    # SSA join for if statement (line 2158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'return_indices' (line 2160)
    return_indices_126463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2160, 7), 'return_indices')
    
    # Getting the type of 'ft_inplace' (line 2160)
    ft_inplace_126464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2160, 30), 'ft_inplace')
    # Applying the 'not' unary operator (line 2160)
    result_not__126465 = python_operator(stypy.reporting.localization.Localization(__file__, 2160, 26), 'not', ft_inplace_126464)
    
    # Applying the binary operator 'and' (line 2160)
    result_and_keyword_126466 = python_operator(stypy.reporting.localization.Localization(__file__, 2160, 7), 'and', return_indices_126463, result_not__126465)
    
    # Testing the type of an if condition (line 2160)
    if_condition_126467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2160, 4), result_and_keyword_126466)
    # Assigning a type to the variable 'if_condition_126467' (line 2160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2160, 4), 'if_condition_126467', if_condition_126467)
    # SSA begins for if statement (line 2160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 2161)
    # Processing the call arguments (line 2161)
    # Getting the type of 'ft' (line 2161)
    ft_126470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2161, 22), 'ft', False)
    # Processing the call keyword arguments (line 2161)
    kwargs_126471 = {}
    # Getting the type of 'result' (line 2161)
    result_126468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2161, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 2161)
    append_126469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2161, 8), result_126468, 'append')
    # Calling append(args, kwargs) (line 2161)
    append_call_result_126472 = invoke(stypy.reporting.localization.Localization(__file__, 2161, 8), append_126469, *[ft_126470], **kwargs_126471)
    
    # SSA join for if statement (line 2160)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 2163)
    # Processing the call arguments (line 2163)
    # Getting the type of 'result' (line 2163)
    result_126474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2163, 11), 'result', False)
    # Processing the call keyword arguments (line 2163)
    kwargs_126475 = {}
    # Getting the type of 'len' (line 2163)
    len_126473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2163, 7), 'len', False)
    # Calling len(args, kwargs) (line 2163)
    len_call_result_126476 = invoke(stypy.reporting.localization.Localization(__file__, 2163, 7), len_126473, *[result_126474], **kwargs_126475)
    
    int_126477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2163, 22), 'int')
    # Applying the binary operator '==' (line 2163)
    result_eq_126478 = python_operator(stypy.reporting.localization.Localization(__file__, 2163, 7), '==', len_call_result_126476, int_126477)
    
    # Testing the type of an if condition (line 2163)
    if_condition_126479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2163, 4), result_eq_126478)
    # Assigning a type to the variable 'if_condition_126479' (line 2163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2163, 4), 'if_condition_126479', if_condition_126479)
    # SSA begins for if statement (line 2163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tuple(...): (line 2164)
    # Processing the call arguments (line 2164)
    # Getting the type of 'result' (line 2164)
    result_126481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2164, 21), 'result', False)
    # Processing the call keyword arguments (line 2164)
    kwargs_126482 = {}
    # Getting the type of 'tuple' (line 2164)
    tuple_126480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2164, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 2164)
    tuple_call_result_126483 = invoke(stypy.reporting.localization.Localization(__file__, 2164, 15), tuple_126480, *[result_126481], **kwargs_126482)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2164, 8), 'stypy_return_type', tuple_call_result_126483)
    # SSA branch for the else part of an if statement (line 2163)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 2165)
    # Processing the call arguments (line 2165)
    # Getting the type of 'result' (line 2165)
    result_126485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2165, 13), 'result', False)
    # Processing the call keyword arguments (line 2165)
    kwargs_126486 = {}
    # Getting the type of 'len' (line 2165)
    len_126484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2165, 9), 'len', False)
    # Calling len(args, kwargs) (line 2165)
    len_call_result_126487 = invoke(stypy.reporting.localization.Localization(__file__, 2165, 9), len_126484, *[result_126485], **kwargs_126486)
    
    int_126488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2165, 24), 'int')
    # Applying the binary operator '==' (line 2165)
    result_eq_126489 = python_operator(stypy.reporting.localization.Localization(__file__, 2165, 9), '==', len_call_result_126487, int_126488)
    
    # Testing the type of an if condition (line 2165)
    if_condition_126490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2165, 9), result_eq_126489)
    # Assigning a type to the variable 'if_condition_126490' (line 2165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2165, 9), 'if_condition_126490', if_condition_126490)
    # SSA begins for if statement (line 2165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_126491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2166, 22), 'int')
    # Getting the type of 'result' (line 2166)
    result_126492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2166, 15), 'result')
    # Obtaining the member '__getitem__' of a type (line 2166)
    getitem___126493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2166, 15), result_126492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 2166)
    subscript_call_result_126494 = invoke(stypy.reporting.localization.Localization(__file__, 2166, 15), getitem___126493, int_126491)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2166, 8), 'stypy_return_type', subscript_call_result_126494)
    # SSA branch for the else part of an if statement (line 2165)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'None' (line 2168)
    None_126495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2168, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 2168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2168, 8), 'stypy_return_type', None_126495)
    # SSA join for if statement (line 2165)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 2163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'distance_transform_edt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'distance_transform_edt' in the type store
    # Getting the type of 'stypy_return_type' (line 2002)
    stypy_return_type_126496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2002, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126496)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'distance_transform_edt'
    return stypy_return_type_126496

# Assigning a type to the variable 'distance_transform_edt' (line 2002)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2002, 0), 'distance_transform_edt', distance_transform_edt)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
