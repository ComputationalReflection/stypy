
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A collection of image utilities using the Python Imaging Library (PIL).
3: 
4: Note that PIL is not a dependency of SciPy and this module is not
5: available on systems that don't have PIL installed.
6: 
7: '''
8: from __future__ import division, print_function, absolute_import
9: 
10: # Functions which need the PIL
11: 
12: import numpy
13: import tempfile
14: 
15: from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,
16:                    transpose, iscomplexobj, uint8, issubdtype, array)
17: 
18: try:
19:     from PIL import Image, ImageFilter
20: except ImportError:
21:     import Image
22:     import ImageFilter
23: 
24: 
25: if not hasattr(Image, 'frombytes'):
26:     Image.frombytes = Image.fromstring
27: 
28: __all__ = ['fromimage', 'toimage', 'imsave', 'imread', 'bytescale',
29:            'imrotate', 'imresize', 'imshow', 'imfilter']
30: 
31: 
32: @numpy.deprecate(message="`bytescale` is deprecated in SciPy 1.0.0, "
33:                          "and will be removed in 1.2.0.")
34: def bytescale(data, cmin=None, cmax=None, high=255, low=0):
35:     '''
36:     Byte scales an array (image).
37: 
38:     Byte scaling means converting the input image to uint8 dtype and scaling
39:     the range to ``(low, high)`` (default 0-255).
40:     If the input image already has dtype uint8, no scaling is done.
41: 
42:     This function is only available if Python Imaging Library (PIL) is installed.
43: 
44:     Parameters
45:     ----------
46:     data : ndarray
47:         PIL image data array.
48:     cmin : scalar, optional
49:         Bias scaling of small values. Default is ``data.min()``.
50:     cmax : scalar, optional
51:         Bias scaling of large values. Default is ``data.max()``.
52:     high : scalar, optional
53:         Scale max value to `high`.  Default is 255.
54:     low : scalar, optional
55:         Scale min value to `low`.  Default is 0.
56: 
57:     Returns
58:     -------
59:     img_array : uint8 ndarray
60:         The byte-scaled array.
61: 
62:     Examples
63:     --------
64:     >>> from scipy.misc import bytescale
65:     >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
66:     ...                 [ 73.88003259,  80.91433048,   4.88878881],
67:     ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
68:     >>> bytescale(img)
69:     array([[255,   0, 236],
70:            [205, 225,   4],
71:            [140,  90,  70]], dtype=uint8)
72:     >>> bytescale(img, high=200, low=100)
73:     array([[200, 100, 192],
74:            [180, 188, 102],
75:            [155, 135, 128]], dtype=uint8)
76:     >>> bytescale(img, cmin=0, cmax=255)
77:     array([[91,  3, 84],
78:            [74, 81,  5],
79:            [52, 34, 28]], dtype=uint8)
80: 
81:     '''
82:     if data.dtype == uint8:
83:         return data
84: 
85:     if high > 255:
86:         raise ValueError("`high` should be less than or equal to 255.")
87:     if low < 0:
88:         raise ValueError("`low` should be greater than or equal to 0.")
89:     if high < low:
90:         raise ValueError("`high` should be greater than or equal to `low`.")
91: 
92:     if cmin is None:
93:         cmin = data.min()
94:     if cmax is None:
95:         cmax = data.max()
96: 
97:     cscale = cmax - cmin
98:     if cscale < 0:
99:         raise ValueError("`cmax` should be larger than `cmin`.")
100:     elif cscale == 0:
101:         cscale = 1
102: 
103:     scale = float(high - low) / cscale
104:     bytedata = (data - cmin) * scale + low
105:     return (bytedata.clip(low, high) + 0.5).astype(uint8)
106: 
107: 
108: @numpy.deprecate(message="`imread` is deprecated in SciPy 1.0.0, "
109:                          "and will be removed in 1.2.0.\n"
110:                          "Use ``imageio.imread`` instead.")
111: def imread(name, flatten=False, mode=None):
112:     '''
113:     Read an image from a file as an array.
114: 
115:     This function is only available if Python Imaging Library (PIL) is installed.
116: 
117:     Parameters
118:     ----------
119:     name : str or file object
120:         The file name or file object to be read.
121:     flatten : bool, optional
122:         If True, flattens the color layers into a single gray-scale layer.
123:     mode : str, optional
124:         Mode to convert image to, e.g. ``'RGB'``.  See the Notes for more
125:         details.
126: 
127:     Returns
128:     -------
129:     imread : ndarray
130:         The array obtained by reading the image.
131: 
132:     Notes
133:     -----
134:     `imread` uses the Python Imaging Library (PIL) to read an image.
135:     The following notes are from the PIL documentation.
136: 
137:     `mode` can be one of the following strings:
138: 
139:     * 'L' (8-bit pixels, black and white)
140:     * 'P' (8-bit pixels, mapped to any other mode using a color palette)
141:     * 'RGB' (3x8-bit pixels, true color)
142:     * 'RGBA' (4x8-bit pixels, true color with transparency mask)
143:     * 'CMYK' (4x8-bit pixels, color separation)
144:     * 'YCbCr' (3x8-bit pixels, color video format)
145:     * 'I' (32-bit signed integer pixels)
146:     * 'F' (32-bit floating point pixels)
147: 
148:     PIL also provides limited support for a few special modes, including
149:     'LA' ('L' with alpha), 'RGBX' (true color with padding) and 'RGBa'
150:     (true color with premultiplied alpha).
151: 
152:     When translating a color image to black and white (mode 'L', 'I' or
153:     'F'), the library uses the ITU-R 601-2 luma transform::
154: 
155:         L = R * 299/1000 + G * 587/1000 + B * 114/1000
156: 
157:     When `flatten` is True, the image is converted using mode 'F'.
158:     When `mode` is not None and `flatten` is True, the image is first
159:     converted according to `mode`, and the result is then flattened using
160:     mode 'F'.
161: 
162:     '''
163: 
164:     im = Image.open(name)
165:     return fromimage(im, flatten=flatten, mode=mode)
166: 
167: 
168: @numpy.deprecate(message="`imsave` is deprecated in SciPy 1.0.0, "
169:                          "and will be removed in 1.2.0.\n"
170:                          "Use ``imageio.imwrite`` instead.")
171: def imsave(name, arr, format=None):
172:     '''
173:     Save an array as an image.
174: 
175:     This function is only available if Python Imaging Library (PIL) is installed.
176: 
177:     .. warning::
178: 
179:         This function uses `bytescale` under the hood to rescale images to use
180:         the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
181:         It will also cast data for 2-D images to ``uint32`` for ``mode=None``
182:         (which is the default).
183: 
184:     Parameters
185:     ----------
186:     name : str or file object
187:         Output file name or file object.
188:     arr : ndarray, MxN or MxNx3 or MxNx4
189:         Array containing image values.  If the shape is ``MxN``, the array
190:         represents a grey-level image.  Shape ``MxNx3`` stores the red, green
191:         and blue bands along the last dimension.  An alpha layer may be
192:         included, specified as the last colour band of an ``MxNx4`` array.
193:     format : str
194:         Image format. If omitted, the format to use is determined from the
195:         file name extension. If a file object was used instead of a file name,
196:         this parameter should always be used.
197: 
198:     Examples
199:     --------
200:     Construct an array of gradient intensity values and save to file:
201: 
202:     >>> from scipy.misc import imsave
203:     >>> x = np.zeros((255, 255))
204:     >>> x = np.zeros((255, 255), dtype=np.uint8)
205:     >>> x[:] = np.arange(255)
206:     >>> imsave('gradient.png', x)
207: 
208:     Construct an array with three colour bands (R, G, B) and store to file:
209: 
210:     >>> rgb = np.zeros((255, 255, 3), dtype=np.uint8)
211:     >>> rgb[..., 0] = np.arange(255)
212:     >>> rgb[..., 1] = 55
213:     >>> rgb[..., 2] = 1 - np.arange(255)
214:     >>> imsave('rgb_gradient.png', rgb)
215: 
216:     '''
217:     im = toimage(arr, channel_axis=2)
218:     if format is None:
219:         im.save(name)
220:     else:
221:         im.save(name, format)
222:     return
223: 
224: 
225: @numpy.deprecate(message="`fromimage` is deprecated in SciPy 1.0.0. "
226:                          "and will be removed in 1.2.0.\n"
227:                          "Use ``np.asarray(im)`` instead.")
228: def fromimage(im, flatten=False, mode=None):
229:     '''
230:     Return a copy of a PIL image as a numpy array.
231: 
232:     This function is only available if Python Imaging Library (PIL) is installed.
233: 
234:     Parameters
235:     ----------
236:     im : PIL image
237:         Input image.
238:     flatten : bool
239:         If true, convert the output to grey-scale.
240:     mode : str, optional
241:         Mode to convert image to, e.g. ``'RGB'``.  See the Notes of the
242:         `imread` docstring for more details.
243: 
244:     Returns
245:     -------
246:     fromimage : ndarray
247:         The different colour bands/channels are stored in the
248:         third dimension, such that a grey-image is MxN, an
249:         RGB-image MxNx3 and an RGBA-image MxNx4.
250: 
251:     '''
252:     if not Image.isImageType(im):
253:         raise TypeError("Input is not a PIL image.")
254: 
255:     if mode is not None:
256:         if mode != im.mode:
257:             im = im.convert(mode)
258:     elif im.mode == 'P':
259:         # Mode 'P' means there is an indexed "palette".  If we leave the mode
260:         # as 'P', then when we do `a = array(im)` below, `a` will be a 2-D
261:         # containing the indices into the palette, and not a 3-D array
262:         # containing the RGB or RGBA values.
263:         if 'transparency' in im.info:
264:             im = im.convert('RGBA')
265:         else:
266:             im = im.convert('RGB')
267: 
268:     if flatten:
269:         im = im.convert('F')
270:     elif im.mode == '1':
271:         # Workaround for crash in PIL. When im is 1-bit, the call array(im)
272:         # can cause a seg. fault, or generate garbage. See
273:         # https://github.com/scipy/scipy/issues/2138 and
274:         # https://github.com/python-pillow/Pillow/issues/350.
275:         #
276:         # This converts im from a 1-bit image to an 8-bit image.
277:         im = im.convert('L')
278: 
279:     a = array(im)
280:     return a
281: 
282: _errstr = "Mode is unknown or incompatible with input array shape."
283: 
284: 
285: @numpy.deprecate(message="`toimage` is deprecated in SciPy 1.0.0, "
286:                          "and will be removed in 1.2.0.\n"
287:             "Use Pillow's ``Image.fromarray`` directly instead.")
288: def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
289:             mode=None, channel_axis=None):
290:     '''Takes a numpy array and returns a PIL image.
291: 
292:     This function is only available if Python Imaging Library (PIL) is installed.
293: 
294:     The mode of the PIL image depends on the array shape and the `pal` and
295:     `mode` keywords.
296: 
297:     For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
298:     (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
299:     is given as 'F' or 'I' in which case a float and/or integer array is made.
300: 
301:     .. warning::
302: 
303:         This function uses `bytescale` under the hood to rescale images to use
304:         the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
305:         It will also cast data for 2-D images to ``uint32`` for ``mode=None``
306:         (which is the default).
307: 
308:     Notes
309:     -----
310:     For 3-D arrays, the `channel_axis` argument tells which dimension of the
311:     array holds the channel data.
312: 
313:     For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
314:     by default or 'YCbCr' if selected.
315: 
316:     The numpy array must be either 2 dimensional or 3 dimensional.
317: 
318:     '''
319:     data = asarray(arr)
320:     if iscomplexobj(data):
321:         raise ValueError("Cannot convert a complex-valued array.")
322:     shape = list(data.shape)
323:     valid = len(shape) == 2 or ((len(shape) == 3) and
324:                                 ((3 in shape) or (4 in shape)))
325:     if not valid:
326:         raise ValueError("'arr' does not have a suitable array shape for "
327:                          "any mode.")
328:     if len(shape) == 2:
329:         shape = (shape[1], shape[0])  # columns show up first
330:         if mode == 'F':
331:             data32 = data.astype(numpy.float32)
332:             image = Image.frombytes(mode, shape, data32.tostring())
333:             return image
334:         if mode in [None, 'L', 'P']:
335:             bytedata = bytescale(data, high=high, low=low,
336:                                  cmin=cmin, cmax=cmax)
337:             image = Image.frombytes('L', shape, bytedata.tostring())
338:             if pal is not None:
339:                 image.putpalette(asarray(pal, dtype=uint8).tostring())
340:                 # Becomes a mode='P' automagically.
341:             elif mode == 'P':  # default gray-scale
342:                 pal = (arange(0, 256, 1, dtype=uint8)[:, newaxis] *
343:                        ones((3,), dtype=uint8)[newaxis, :])
344:                 image.putpalette(asarray(pal, dtype=uint8).tostring())
345:             return image
346:         if mode == '1':  # high input gives threshold for 1
347:             bytedata = (data > high)
348:             image = Image.frombytes('1', shape, bytedata.tostring())
349:             return image
350:         if cmin is None:
351:             cmin = amin(ravel(data))
352:         if cmax is None:
353:             cmax = amax(ravel(data))
354:         data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
355:         if mode == 'I':
356:             data32 = data.astype(numpy.uint32)
357:             image = Image.frombytes(mode, shape, data32.tostring())
358:         else:
359:             raise ValueError(_errstr)
360:         return image
361: 
362:     # if here then 3-d array with a 3 or a 4 in the shape length.
363:     # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
364:     if channel_axis is None:
365:         if (3 in shape):
366:             ca = numpy.flatnonzero(asarray(shape) == 3)[0]
367:         else:
368:             ca = numpy.flatnonzero(asarray(shape) == 4)
369:             if len(ca):
370:                 ca = ca[0]
371:             else:
372:                 raise ValueError("Could not find channel dimension.")
373:     else:
374:         ca = channel_axis
375: 
376:     numch = shape[ca]
377:     if numch not in [3, 4]:
378:         raise ValueError("Channel axis dimension is not valid.")
379: 
380:     bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
381:     if ca == 2:
382:         strdata = bytedata.tostring()
383:         shape = (shape[1], shape[0])
384:     elif ca == 1:
385:         strdata = transpose(bytedata, (0, 2, 1)).tostring()
386:         shape = (shape[2], shape[0])
387:     elif ca == 0:
388:         strdata = transpose(bytedata, (1, 2, 0)).tostring()
389:         shape = (shape[2], shape[1])
390:     if mode is None:
391:         if numch == 3:
392:             mode = 'RGB'
393:         else:
394:             mode = 'RGBA'
395: 
396:     if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
397:         raise ValueError(_errstr)
398: 
399:     if mode in ['RGB', 'YCbCr']:
400:         if numch != 3:
401:             raise ValueError("Invalid array shape for mode.")
402:     if mode in ['RGBA', 'CMYK']:
403:         if numch != 4:
404:             raise ValueError("Invalid array shape for mode.")
405: 
406:     # Here we know data and mode is correct
407:     image = Image.frombytes(mode, shape, strdata)
408:     return image
409: 
410: 
411: @numpy.deprecate(message="`imrotate` is deprecated in SciPy 1.0.0, "
412:                          "and will be removed in 1.2.0.\n"
413:                          "Use ``skimage.transform.rotate`` instead.")
414: def imrotate(arr, angle, interp='bilinear'):
415:     '''
416:     Rotate an image counter-clockwise by angle degrees.
417: 
418:     This function is only available if Python Imaging Library (PIL) is installed.
419: 
420:     .. warning::
421: 
422:         This function uses `bytescale` under the hood to rescale images to use
423:         the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
424:         It will also cast data for 2-D images to ``uint32`` for ``mode=None``
425:         (which is the default).
426: 
427:     Parameters
428:     ----------
429:     arr : ndarray
430:         Input array of image to be rotated.
431:     angle : float
432:         The angle of rotation.
433:     interp : str, optional
434:         Interpolation
435: 
436:         - 'nearest' :  for nearest neighbor
437:         - 'bilinear' : for bilinear
438:         - 'lanczos' : for lanczos
439:         - 'cubic' : for bicubic
440:         - 'bicubic' : for bicubic
441: 
442:     Returns
443:     -------
444:     imrotate : ndarray
445:         The rotated array of image.
446: 
447:     '''
448:     arr = asarray(arr)
449:     func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
450:     im = toimage(arr)
451:     im = im.rotate(angle, resample=func[interp])
452:     return fromimage(im)
453: 
454: 
455: @numpy.deprecate(message="`imshow` is deprecated in SciPy 1.0.0, "
456:                          "and will be removed in 1.2.0.\n"
457:                          "Use ``matplotlib.pyplot.imshow`` instead.")
458: def imshow(arr):
459:     '''
460:     Simple showing of an image through an external viewer.
461: 
462:     This function is only available if Python Imaging Library (PIL) is installed.
463: 
464:     Uses the image viewer specified by the environment variable
465:     SCIPY_PIL_IMAGE_VIEWER, or if that is not defined then `see`,
466:     to view a temporary file generated from array data.
467: 
468:     .. warning::
469: 
470:         This function uses `bytescale` under the hood to rescale images to use
471:         the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
472:         It will also cast data for 2-D images to ``uint32`` for ``mode=None``
473:         (which is the default).
474: 
475:     Parameters
476:     ----------
477:     arr : ndarray
478:         Array of image data to show.
479: 
480:     Returns
481:     -------
482:     None
483: 
484:     Examples
485:     --------
486:     >>> a = np.tile(np.arange(255), (255,1))
487:     >>> from scipy import misc
488:     >>> misc.imshow(a)
489: 
490:     '''
491:     im = toimage(arr)
492:     fnum, fname = tempfile.mkstemp('.png')
493:     try:
494:         im.save(fname)
495:     except:
496:         raise RuntimeError("Error saving temporary image data.")
497: 
498:     import os
499:     os.close(fnum)
500: 
501:     cmd = os.environ.get('SCIPY_PIL_IMAGE_VIEWER', 'see')
502:     status = os.system("%s %s" % (cmd, fname))
503: 
504:     os.unlink(fname)
505:     if status != 0:
506:         raise RuntimeError('Could not execute image viewer.')
507: 
508: 
509: @numpy.deprecate(message="`imresize` is deprecated in SciPy 1.0.0, "
510:                          "and will be removed in 1.2.0.\n"
511:                          "Use ``skimage.transform.resize`` instead.")
512: def imresize(arr, size, interp='bilinear', mode=None):
513:     '''
514:     Resize an image.
515: 
516:     This function is only available if Python Imaging Library (PIL) is installed.
517: 
518:     .. warning::
519: 
520:         This function uses `bytescale` under the hood to rescale images to use
521:         the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
522:         It will also cast data for 2-D images to ``uint32`` for ``mode=None``
523:         (which is the default).
524: 
525:     Parameters
526:     ----------
527:     arr : ndarray
528:         The array of image to be resized.
529:     size : int, float or tuple
530:         * int   - Percentage of current size.
531:         * float - Fraction of current size.
532:         * tuple - Size of the output image (height, width).
533: 
534:     interp : str, optional
535:         Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
536:         'bicubic' or 'cubic').
537:     mode : str, optional
538:         The PIL image mode ('P', 'L', etc.) to convert `arr` before resizing.
539:         If ``mode=None`` (the default), 2-D images will be treated like
540:         ``mode='L'``, i.e. casting to long integer.  For 3-D and 4-D arrays,
541:         `mode` will be set to ``'RGB'`` and ``'RGBA'`` respectively.
542: 
543:     Returns
544:     -------
545:     imresize : ndarray
546:         The resized array of image.
547: 
548:     See Also
549:     --------
550:     toimage : Implicitly used to convert `arr` according to `mode`.
551:     scipy.ndimage.zoom : More generic implementation that does not use PIL.
552: 
553:     '''
554:     im = toimage(arr, mode=mode)
555:     ts = type(size)
556:     if issubdtype(ts, numpy.signedinteger):
557:         percent = size / 100.0
558:         size = tuple((array(im.size)*percent).astype(int))
559:     elif issubdtype(type(size), numpy.floating):
560:         size = tuple((array(im.size)*size).astype(int))
561:     else:
562:         size = (size[1], size[0])
563:     func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
564:     imnew = im.resize(size, resample=func[interp])
565:     return fromimage(imnew)
566: 
567: 
568: @numpy.deprecate(message="`imfilter` is deprecated in SciPy 1.0.0, "
569:                          "and will be removed in 1.2.0.\n"
570:                          "Use Pillow filtering functionality directly.")
571: def imfilter(arr, ftype):
572:     '''
573:     Simple filtering of an image.
574: 
575:     This function is only available if Python Imaging Library (PIL) is installed.
576: 
577:     .. warning::
578: 
579:         This function uses `bytescale` under the hood to rescale images to use
580:         the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
581:         It will also cast data for 2-D images to ``uint32`` for ``mode=None``
582:         (which is the default).
583: 
584:     Parameters
585:     ----------
586:     arr : ndarray
587:         The array of Image in which the filter is to be applied.
588:     ftype : str
589:         The filter that has to be applied. Legal values are:
590:         'blur', 'contour', 'detail', 'edge_enhance', 'edge_enhance_more',
591:         'emboss', 'find_edges', 'smooth', 'smooth_more', 'sharpen'.
592: 
593:     Returns
594:     -------
595:     imfilter : ndarray
596:         The array with filter applied.
597: 
598:     Raises
599:     ------
600:     ValueError
601:         *Unknown filter type.*  If the filter you are trying
602:         to apply is unsupported.
603: 
604:     '''
605:     _tdict = {'blur': ImageFilter.BLUR,
606:               'contour': ImageFilter.CONTOUR,
607:               'detail': ImageFilter.DETAIL,
608:               'edge_enhance': ImageFilter.EDGE_ENHANCE,
609:               'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
610:               'emboss': ImageFilter.EMBOSS,
611:               'find_edges': ImageFilter.FIND_EDGES,
612:               'smooth': ImageFilter.SMOOTH,
613:               'smooth_more': ImageFilter.SMOOTH_MORE,
614:               'sharpen': ImageFilter.SHARPEN
615:               }
616: 
617:     im = toimage(arr)
618:     if ftype not in _tdict:
619:         raise ValueError("Unknown filter type.")
620:     return fromimage(im.filter(_tdict[ftype]))
621: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_114244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', "\nA collection of image utilities using the Python Imaging Library (PIL).\n\nNote that PIL is not a dependency of SciPy and this module is not\navailable on systems that don't have PIL installed.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import numpy' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_114245 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy')

if (type(import_114245) is not StypyTypeError):

    if (import_114245 != 'pyd_module'):
        __import__(import_114245)
        sys_modules_114246 = sys.modules[import_114245]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', sys_modules_114246.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', import_114245)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import tempfile' statement (line 13)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy import amin, amax, ravel, asarray, arange, ones, newaxis, transpose, iscomplexobj, uint8, issubdtype, array' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_114247 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy')

if (type(import_114247) is not StypyTypeError):

    if (import_114247 != 'pyd_module'):
        __import__(import_114247)
        sys_modules_114248 = sys.modules[import_114247]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy', sys_modules_114248.module_type_store, module_type_store, ['amin', 'amax', 'ravel', 'asarray', 'arange', 'ones', 'newaxis', 'transpose', 'iscomplexobj', 'uint8', 'issubdtype', 'array'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_114248, sys_modules_114248.module_type_store, module_type_store)
    else:
        from numpy import amin, amax, ravel, asarray, arange, ones, newaxis, transpose, iscomplexobj, uint8, issubdtype, array

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy', None, module_type_store, ['amin', 'amax', 'ravel', 'asarray', 'arange', 'ones', 'newaxis', 'transpose', 'iscomplexobj', 'uint8', 'issubdtype', 'array'], [amin, amax, ravel, asarray, arange, ones, newaxis, transpose, iscomplexobj, uint8, issubdtype, array])

else:
    # Assigning a type to the variable 'numpy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy', import_114247)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')



# SSA begins for try-except statement (line 18)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 4))

# 'from PIL import Image, ImageFilter' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_114249 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'PIL')

if (type(import_114249) is not StypyTypeError):

    if (import_114249 != 'pyd_module'):
        __import__(import_114249)
        sys_modules_114250 = sys.modules[import_114249]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'PIL', sys_modules_114250.module_type_store, module_type_store, ['Image', 'ImageFilter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 4), __file__, sys_modules_114250, sys_modules_114250.module_type_store, module_type_store)
    else:
        from PIL import Image, ImageFilter

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'PIL', None, module_type_store, ['Image', 'ImageFilter'], [Image, ImageFilter])

else:
    # Assigning a type to the variable 'PIL' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'PIL', import_114249)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

# SSA branch for the except part of a try statement (line 18)
# SSA branch for the except 'ImportError' branch of a try statement (line 18)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 4))

# 'import Image' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_114251 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'Image')

if (type(import_114251) is not StypyTypeError):

    if (import_114251 != 'pyd_module'):
        __import__(import_114251)
        sys_modules_114252 = sys.modules[import_114251]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'Image', sys_modules_114252.module_type_store, module_type_store)
    else:
        import Image

        import_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'Image', Image, module_type_store)

else:
    # Assigning a type to the variable 'Image' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'Image', import_114251)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 4))

# 'import ImageFilter' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_114253 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'ImageFilter')

if (type(import_114253) is not StypyTypeError):

    if (import_114253 != 'pyd_module'):
        __import__(import_114253)
        sys_modules_114254 = sys.modules[import_114253]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'ImageFilter', sys_modules_114254.module_type_store, module_type_store)
    else:
        import ImageFilter

        import_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'ImageFilter', ImageFilter, module_type_store)

else:
    # Assigning a type to the variable 'ImageFilter' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ImageFilter', import_114253)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

# SSA join for try-except statement (line 18)
module_type_store = module_type_store.join_ssa_context()


# Type idiom detected: calculating its left and rigth part (line 25)
str_114255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'str', 'frombytes')
# Getting the type of 'Image' (line 25)
Image_114256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'Image')

(may_be_114257, more_types_in_union_114258) = may_not_provide_member(str_114255, Image_114256)

if may_be_114257:

    if more_types_in_union_114258:
        # Runtime conditional SSA (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    # Assigning a type to the variable 'Image' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'Image', remove_member_provider_from_union(Image_114256, 'frombytes'))
    
    # Assigning a Attribute to a Attribute (line 26):
    
    # Assigning a Attribute to a Attribute (line 26):
    # Getting the type of 'Image' (line 26)
    Image_114259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 22), 'Image')
    # Obtaining the member 'fromstring' of a type (line 26)
    fromstring_114260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 22), Image_114259, 'fromstring')
    # Getting the type of 'Image' (line 26)
    Image_114261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'Image')
    # Setting the type of the member 'frombytes' of a type (line 26)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), Image_114261, 'frombytes', fromstring_114260)

    if more_types_in_union_114258:
        # SSA join for if statement (line 25)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a List to a Name (line 28):

# Assigning a List to a Name (line 28):
__all__ = ['fromimage', 'toimage', 'imsave', 'imread', 'bytescale', 'imrotate', 'imresize', 'imshow', 'imfilter']
module_type_store.set_exportable_members(['fromimage', 'toimage', 'imsave', 'imread', 'bytescale', 'imrotate', 'imresize', 'imshow', 'imfilter'])

# Obtaining an instance of the builtin type 'list' (line 28)
list_114262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_114263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'str', 'fromimage')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114263)
# Adding element type (line 28)
str_114264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 24), 'str', 'toimage')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114264)
# Adding element type (line 28)
str_114265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 35), 'str', 'imsave')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114265)
# Adding element type (line 28)
str_114266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 45), 'str', 'imread')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114266)
# Adding element type (line 28)
str_114267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 55), 'str', 'bytescale')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114267)
# Adding element type (line 28)
str_114268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'str', 'imrotate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114268)
# Adding element type (line 28)
str_114269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'str', 'imresize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114269)
# Adding element type (line 28)
str_114270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 35), 'str', 'imshow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114270)
# Adding element type (line 28)
str_114271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 45), 'str', 'imfilter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_114262, str_114271)

# Assigning a type to the variable '__all__' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '__all__', list_114262)

@norecursion
def bytescale(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 34)
    None_114272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'None')
    # Getting the type of 'None' (line 34)
    None_114273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 36), 'None')
    int_114274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 47), 'int')
    int_114275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 56), 'int')
    defaults = [None_114272, None_114273, int_114274, int_114275]
    # Create a new context for function 'bytescale'
    module_type_store = module_type_store.open_function_context('bytescale', 32, 0, False)
    
    # Passed parameters checking function
    bytescale.stypy_localization = localization
    bytescale.stypy_type_of_self = None
    bytescale.stypy_type_store = module_type_store
    bytescale.stypy_function_name = 'bytescale'
    bytescale.stypy_param_names_list = ['data', 'cmin', 'cmax', 'high', 'low']
    bytescale.stypy_varargs_param_name = None
    bytescale.stypy_kwargs_param_name = None
    bytescale.stypy_call_defaults = defaults
    bytescale.stypy_call_varargs = varargs
    bytescale.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bytescale', ['data', 'cmin', 'cmax', 'high', 'low'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bytescale', localization, ['data', 'cmin', 'cmax', 'high', 'low'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bytescale(...)' code ##################

    str_114276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', '\n    Byte scales an array (image).\n\n    Byte scaling means converting the input image to uint8 dtype and scaling\n    the range to ``(low, high)`` (default 0-255).\n    If the input image already has dtype uint8, no scaling is done.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    Parameters\n    ----------\n    data : ndarray\n        PIL image data array.\n    cmin : scalar, optional\n        Bias scaling of small values. Default is ``data.min()``.\n    cmax : scalar, optional\n        Bias scaling of large values. Default is ``data.max()``.\n    high : scalar, optional\n        Scale max value to `high`.  Default is 255.\n    low : scalar, optional\n        Scale min value to `low`.  Default is 0.\n\n    Returns\n    -------\n    img_array : uint8 ndarray\n        The byte-scaled array.\n\n    Examples\n    --------\n    >>> from scipy.misc import bytescale\n    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],\n    ...                 [ 73.88003259,  80.91433048,   4.88878881],\n    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])\n    >>> bytescale(img)\n    array([[255,   0, 236],\n           [205, 225,   4],\n           [140,  90,  70]], dtype=uint8)\n    >>> bytescale(img, high=200, low=100)\n    array([[200, 100, 192],\n           [180, 188, 102],\n           [155, 135, 128]], dtype=uint8)\n    >>> bytescale(img, cmin=0, cmax=255)\n    array([[91,  3, 84],\n           [74, 81,  5],\n           [52, 34, 28]], dtype=uint8)\n\n    ')
    
    
    # Getting the type of 'data' (line 82)
    data_114277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'data')
    # Obtaining the member 'dtype' of a type (line 82)
    dtype_114278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 7), data_114277, 'dtype')
    # Getting the type of 'uint8' (line 82)
    uint8_114279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'uint8')
    # Applying the binary operator '==' (line 82)
    result_eq_114280 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 7), '==', dtype_114278, uint8_114279)
    
    # Testing the type of an if condition (line 82)
    if_condition_114281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), result_eq_114280)
    # Assigning a type to the variable 'if_condition_114281' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_114281', if_condition_114281)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'data' (line 83)
    data_114282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'data')
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', data_114282)
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'high' (line 85)
    high_114283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 7), 'high')
    int_114284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 14), 'int')
    # Applying the binary operator '>' (line 85)
    result_gt_114285 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), '>', high_114283, int_114284)
    
    # Testing the type of an if condition (line 85)
    if_condition_114286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 4), result_gt_114285)
    # Assigning a type to the variable 'if_condition_114286' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'if_condition_114286', if_condition_114286)
    # SSA begins for if statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 86)
    # Processing the call arguments (line 86)
    str_114288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'str', '`high` should be less than or equal to 255.')
    # Processing the call keyword arguments (line 86)
    kwargs_114289 = {}
    # Getting the type of 'ValueError' (line 86)
    ValueError_114287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 86)
    ValueError_call_result_114290 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), ValueError_114287, *[str_114288], **kwargs_114289)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 86, 8), ValueError_call_result_114290, 'raise parameter', BaseException)
    # SSA join for if statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'low' (line 87)
    low_114291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 7), 'low')
    int_114292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 13), 'int')
    # Applying the binary operator '<' (line 87)
    result_lt_114293 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 7), '<', low_114291, int_114292)
    
    # Testing the type of an if condition (line 87)
    if_condition_114294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 4), result_lt_114293)
    # Assigning a type to the variable 'if_condition_114294' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'if_condition_114294', if_condition_114294)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 88)
    # Processing the call arguments (line 88)
    str_114296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'str', '`low` should be greater than or equal to 0.')
    # Processing the call keyword arguments (line 88)
    kwargs_114297 = {}
    # Getting the type of 'ValueError' (line 88)
    ValueError_114295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 88)
    ValueError_call_result_114298 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), ValueError_114295, *[str_114296], **kwargs_114297)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 8), ValueError_call_result_114298, 'raise parameter', BaseException)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'high' (line 89)
    high_114299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'high')
    # Getting the type of 'low' (line 89)
    low_114300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 14), 'low')
    # Applying the binary operator '<' (line 89)
    result_lt_114301 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 7), '<', high_114299, low_114300)
    
    # Testing the type of an if condition (line 89)
    if_condition_114302 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), result_lt_114301)
    # Assigning a type to the variable 'if_condition_114302' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_114302', if_condition_114302)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 90)
    # Processing the call arguments (line 90)
    str_114304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'str', '`high` should be greater than or equal to `low`.')
    # Processing the call keyword arguments (line 90)
    kwargs_114305 = {}
    # Getting the type of 'ValueError' (line 90)
    ValueError_114303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 90)
    ValueError_call_result_114306 = invoke(stypy.reporting.localization.Localization(__file__, 90, 14), ValueError_114303, *[str_114304], **kwargs_114305)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 90, 8), ValueError_call_result_114306, 'raise parameter', BaseException)
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 92)
    # Getting the type of 'cmin' (line 92)
    cmin_114307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 7), 'cmin')
    # Getting the type of 'None' (line 92)
    None_114308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'None')
    
    (may_be_114309, more_types_in_union_114310) = may_be_none(cmin_114307, None_114308)

    if may_be_114309:

        if more_types_in_union_114310:
            # Runtime conditional SSA (line 92)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to min(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_114313 = {}
        # Getting the type of 'data' (line 93)
        data_114311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'data', False)
        # Obtaining the member 'min' of a type (line 93)
        min_114312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), data_114311, 'min')
        # Calling min(args, kwargs) (line 93)
        min_call_result_114314 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), min_114312, *[], **kwargs_114313)
        
        # Assigning a type to the variable 'cmin' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'cmin', min_call_result_114314)

        if more_types_in_union_114310:
            # SSA join for if statement (line 92)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 94)
    # Getting the type of 'cmax' (line 94)
    cmax_114315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'cmax')
    # Getting the type of 'None' (line 94)
    None_114316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'None')
    
    (may_be_114317, more_types_in_union_114318) = may_be_none(cmax_114315, None_114316)

    if may_be_114317:

        if more_types_in_union_114318:
            # Runtime conditional SSA (line 94)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to max(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_114321 = {}
        # Getting the type of 'data' (line 95)
        data_114319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'data', False)
        # Obtaining the member 'max' of a type (line 95)
        max_114320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), data_114319, 'max')
        # Calling max(args, kwargs) (line 95)
        max_call_result_114322 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), max_114320, *[], **kwargs_114321)
        
        # Assigning a type to the variable 'cmax' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'cmax', max_call_result_114322)

        if more_types_in_union_114318:
            # SSA join for if statement (line 94)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 97):
    
    # Assigning a BinOp to a Name (line 97):
    # Getting the type of 'cmax' (line 97)
    cmax_114323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'cmax')
    # Getting the type of 'cmin' (line 97)
    cmin_114324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'cmin')
    # Applying the binary operator '-' (line 97)
    result_sub_114325 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 13), '-', cmax_114323, cmin_114324)
    
    # Assigning a type to the variable 'cscale' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'cscale', result_sub_114325)
    
    
    # Getting the type of 'cscale' (line 98)
    cscale_114326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'cscale')
    int_114327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'int')
    # Applying the binary operator '<' (line 98)
    result_lt_114328 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 7), '<', cscale_114326, int_114327)
    
    # Testing the type of an if condition (line 98)
    if_condition_114329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 4), result_lt_114328)
    # Assigning a type to the variable 'if_condition_114329' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'if_condition_114329', if_condition_114329)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 99)
    # Processing the call arguments (line 99)
    str_114331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 25), 'str', '`cmax` should be larger than `cmin`.')
    # Processing the call keyword arguments (line 99)
    kwargs_114332 = {}
    # Getting the type of 'ValueError' (line 99)
    ValueError_114330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 99)
    ValueError_call_result_114333 = invoke(stypy.reporting.localization.Localization(__file__, 99, 14), ValueError_114330, *[str_114331], **kwargs_114332)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 8), ValueError_call_result_114333, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 98)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'cscale' (line 100)
    cscale_114334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'cscale')
    int_114335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'int')
    # Applying the binary operator '==' (line 100)
    result_eq_114336 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 9), '==', cscale_114334, int_114335)
    
    # Testing the type of an if condition (line 100)
    if_condition_114337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 9), result_eq_114336)
    # Assigning a type to the variable 'if_condition_114337' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'if_condition_114337', if_condition_114337)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 101):
    
    # Assigning a Num to a Name (line 101):
    int_114338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'int')
    # Assigning a type to the variable 'cscale' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'cscale', int_114338)
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 103):
    
    # Assigning a BinOp to a Name (line 103):
    
    # Call to float(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'high' (line 103)
    high_114340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'high', False)
    # Getting the type of 'low' (line 103)
    low_114341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'low', False)
    # Applying the binary operator '-' (line 103)
    result_sub_114342 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 18), '-', high_114340, low_114341)
    
    # Processing the call keyword arguments (line 103)
    kwargs_114343 = {}
    # Getting the type of 'float' (line 103)
    float_114339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'float', False)
    # Calling float(args, kwargs) (line 103)
    float_call_result_114344 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), float_114339, *[result_sub_114342], **kwargs_114343)
    
    # Getting the type of 'cscale' (line 103)
    cscale_114345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 32), 'cscale')
    # Applying the binary operator 'div' (line 103)
    result_div_114346 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 12), 'div', float_call_result_114344, cscale_114345)
    
    # Assigning a type to the variable 'scale' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'scale', result_div_114346)
    
    # Assigning a BinOp to a Name (line 104):
    
    # Assigning a BinOp to a Name (line 104):
    # Getting the type of 'data' (line 104)
    data_114347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'data')
    # Getting the type of 'cmin' (line 104)
    cmin_114348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'cmin')
    # Applying the binary operator '-' (line 104)
    result_sub_114349 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 16), '-', data_114347, cmin_114348)
    
    # Getting the type of 'scale' (line 104)
    scale_114350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'scale')
    # Applying the binary operator '*' (line 104)
    result_mul_114351 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), '*', result_sub_114349, scale_114350)
    
    # Getting the type of 'low' (line 104)
    low_114352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'low')
    # Applying the binary operator '+' (line 104)
    result_add_114353 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), '+', result_mul_114351, low_114352)
    
    # Assigning a type to the variable 'bytedata' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'bytedata', result_add_114353)
    
    # Call to astype(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'uint8' (line 105)
    uint8_114363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 51), 'uint8', False)
    # Processing the call keyword arguments (line 105)
    kwargs_114364 = {}
    
    # Call to clip(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'low' (line 105)
    low_114356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'low', False)
    # Getting the type of 'high' (line 105)
    high_114357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'high', False)
    # Processing the call keyword arguments (line 105)
    kwargs_114358 = {}
    # Getting the type of 'bytedata' (line 105)
    bytedata_114354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'bytedata', False)
    # Obtaining the member 'clip' of a type (line 105)
    clip_114355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), bytedata_114354, 'clip')
    # Calling clip(args, kwargs) (line 105)
    clip_call_result_114359 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), clip_114355, *[low_114356, high_114357], **kwargs_114358)
    
    float_114360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 39), 'float')
    # Applying the binary operator '+' (line 105)
    result_add_114361 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 12), '+', clip_call_result_114359, float_114360)
    
    # Obtaining the member 'astype' of a type (line 105)
    astype_114362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), result_add_114361, 'astype')
    # Calling astype(args, kwargs) (line 105)
    astype_call_result_114365 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), astype_114362, *[uint8_114363], **kwargs_114364)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', astype_call_result_114365)
    
    # ################# End of 'bytescale(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bytescale' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_114366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114366)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bytescale'
    return stypy_return_type_114366

# Assigning a type to the variable 'bytescale' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'bytescale', bytescale)

@norecursion
def imread(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 111)
    False_114367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'False')
    # Getting the type of 'None' (line 111)
    None_114368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 37), 'None')
    defaults = [False_114367, None_114368]
    # Create a new context for function 'imread'
    module_type_store = module_type_store.open_function_context('imread', 108, 0, False)
    
    # Passed parameters checking function
    imread.stypy_localization = localization
    imread.stypy_type_of_self = None
    imread.stypy_type_store = module_type_store
    imread.stypy_function_name = 'imread'
    imread.stypy_param_names_list = ['name', 'flatten', 'mode']
    imread.stypy_varargs_param_name = None
    imread.stypy_kwargs_param_name = None
    imread.stypy_call_defaults = defaults
    imread.stypy_call_varargs = varargs
    imread.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'imread', ['name', 'flatten', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'imread', localization, ['name', 'flatten', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'imread(...)' code ##################

    str_114369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', "\n    Read an image from a file as an array.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    Parameters\n    ----------\n    name : str or file object\n        The file name or file object to be read.\n    flatten : bool, optional\n        If True, flattens the color layers into a single gray-scale layer.\n    mode : str, optional\n        Mode to convert image to, e.g. ``'RGB'``.  See the Notes for more\n        details.\n\n    Returns\n    -------\n    imread : ndarray\n        The array obtained by reading the image.\n\n    Notes\n    -----\n    `imread` uses the Python Imaging Library (PIL) to read an image.\n    The following notes are from the PIL documentation.\n\n    `mode` can be one of the following strings:\n\n    * 'L' (8-bit pixels, black and white)\n    * 'P' (8-bit pixels, mapped to any other mode using a color palette)\n    * 'RGB' (3x8-bit pixels, true color)\n    * 'RGBA' (4x8-bit pixels, true color with transparency mask)\n    * 'CMYK' (4x8-bit pixels, color separation)\n    * 'YCbCr' (3x8-bit pixels, color video format)\n    * 'I' (32-bit signed integer pixels)\n    * 'F' (32-bit floating point pixels)\n\n    PIL also provides limited support for a few special modes, including\n    'LA' ('L' with alpha), 'RGBX' (true color with padding) and 'RGBa'\n    (true color with premultiplied alpha).\n\n    When translating a color image to black and white (mode 'L', 'I' or\n    'F'), the library uses the ITU-R 601-2 luma transform::\n\n        L = R * 299/1000 + G * 587/1000 + B * 114/1000\n\n    When `flatten` is True, the image is converted using mode 'F'.\n    When `mode` is not None and `flatten` is True, the image is first\n    converted according to `mode`, and the result is then flattened using\n    mode 'F'.\n\n    ")
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to open(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'name' (line 164)
    name_114372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'name', False)
    # Processing the call keyword arguments (line 164)
    kwargs_114373 = {}
    # Getting the type of 'Image' (line 164)
    Image_114370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 9), 'Image', False)
    # Obtaining the member 'open' of a type (line 164)
    open_114371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 9), Image_114370, 'open')
    # Calling open(args, kwargs) (line 164)
    open_call_result_114374 = invoke(stypy.reporting.localization.Localization(__file__, 164, 9), open_114371, *[name_114372], **kwargs_114373)
    
    # Assigning a type to the variable 'im' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'im', open_call_result_114374)
    
    # Call to fromimage(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'im' (line 165)
    im_114376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'im', False)
    # Processing the call keyword arguments (line 165)
    # Getting the type of 'flatten' (line 165)
    flatten_114377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'flatten', False)
    keyword_114378 = flatten_114377
    # Getting the type of 'mode' (line 165)
    mode_114379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 47), 'mode', False)
    keyword_114380 = mode_114379
    kwargs_114381 = {'flatten': keyword_114378, 'mode': keyword_114380}
    # Getting the type of 'fromimage' (line 165)
    fromimage_114375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'fromimage', False)
    # Calling fromimage(args, kwargs) (line 165)
    fromimage_call_result_114382 = invoke(stypy.reporting.localization.Localization(__file__, 165, 11), fromimage_114375, *[im_114376], **kwargs_114381)
    
    # Assigning a type to the variable 'stypy_return_type' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type', fromimage_call_result_114382)
    
    # ################# End of 'imread(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'imread' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_114383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114383)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'imread'
    return stypy_return_type_114383

# Assigning a type to the variable 'imread' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'imread', imread)

@norecursion
def imsave(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 171)
    None_114384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'None')
    defaults = [None_114384]
    # Create a new context for function 'imsave'
    module_type_store = module_type_store.open_function_context('imsave', 168, 0, False)
    
    # Passed parameters checking function
    imsave.stypy_localization = localization
    imsave.stypy_type_of_self = None
    imsave.stypy_type_store = module_type_store
    imsave.stypy_function_name = 'imsave'
    imsave.stypy_param_names_list = ['name', 'arr', 'format']
    imsave.stypy_varargs_param_name = None
    imsave.stypy_kwargs_param_name = None
    imsave.stypy_call_defaults = defaults
    imsave.stypy_call_varargs = varargs
    imsave.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'imsave', ['name', 'arr', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'imsave', localization, ['name', 'arr', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'imsave(...)' code ##################

    str_114385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, (-1)), 'str', "\n    Save an array as an image.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    .. warning::\n\n        This function uses `bytescale` under the hood to rescale images to use\n        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.\n        It will also cast data for 2-D images to ``uint32`` for ``mode=None``\n        (which is the default).\n\n    Parameters\n    ----------\n    name : str or file object\n        Output file name or file object.\n    arr : ndarray, MxN or MxNx3 or MxNx4\n        Array containing image values.  If the shape is ``MxN``, the array\n        represents a grey-level image.  Shape ``MxNx3`` stores the red, green\n        and blue bands along the last dimension.  An alpha layer may be\n        included, specified as the last colour band of an ``MxNx4`` array.\n    format : str\n        Image format. If omitted, the format to use is determined from the\n        file name extension. If a file object was used instead of a file name,\n        this parameter should always be used.\n\n    Examples\n    --------\n    Construct an array of gradient intensity values and save to file:\n\n    >>> from scipy.misc import imsave\n    >>> x = np.zeros((255, 255))\n    >>> x = np.zeros((255, 255), dtype=np.uint8)\n    >>> x[:] = np.arange(255)\n    >>> imsave('gradient.png', x)\n\n    Construct an array with three colour bands (R, G, B) and store to file:\n\n    >>> rgb = np.zeros((255, 255, 3), dtype=np.uint8)\n    >>> rgb[..., 0] = np.arange(255)\n    >>> rgb[..., 1] = 55\n    >>> rgb[..., 2] = 1 - np.arange(255)\n    >>> imsave('rgb_gradient.png', rgb)\n\n    ")
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to toimage(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'arr' (line 217)
    arr_114387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), 'arr', False)
    # Processing the call keyword arguments (line 217)
    int_114388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'int')
    keyword_114389 = int_114388
    kwargs_114390 = {'channel_axis': keyword_114389}
    # Getting the type of 'toimage' (line 217)
    toimage_114386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 9), 'toimage', False)
    # Calling toimage(args, kwargs) (line 217)
    toimage_call_result_114391 = invoke(stypy.reporting.localization.Localization(__file__, 217, 9), toimage_114386, *[arr_114387], **kwargs_114390)
    
    # Assigning a type to the variable 'im' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'im', toimage_call_result_114391)
    
    # Type idiom detected: calculating its left and rigth part (line 218)
    # Getting the type of 'format' (line 218)
    format_114392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 7), 'format')
    # Getting the type of 'None' (line 218)
    None_114393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 17), 'None')
    
    (may_be_114394, more_types_in_union_114395) = may_be_none(format_114392, None_114393)

    if may_be_114394:

        if more_types_in_union_114395:
            # Runtime conditional SSA (line 218)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to save(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'name' (line 219)
        name_114398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'name', False)
        # Processing the call keyword arguments (line 219)
        kwargs_114399 = {}
        # Getting the type of 'im' (line 219)
        im_114396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'im', False)
        # Obtaining the member 'save' of a type (line 219)
        save_114397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), im_114396, 'save')
        # Calling save(args, kwargs) (line 219)
        save_call_result_114400 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), save_114397, *[name_114398], **kwargs_114399)
        

        if more_types_in_union_114395:
            # Runtime conditional SSA for else branch (line 218)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_114394) or more_types_in_union_114395):
        
        # Call to save(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'name' (line 221)
        name_114403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'name', False)
        # Getting the type of 'format' (line 221)
        format_114404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'format', False)
        # Processing the call keyword arguments (line 221)
        kwargs_114405 = {}
        # Getting the type of 'im' (line 221)
        im_114401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'im', False)
        # Obtaining the member 'save' of a type (line 221)
        save_114402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), im_114401, 'save')
        # Calling save(args, kwargs) (line 221)
        save_call_result_114406 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), save_114402, *[name_114403, format_114404], **kwargs_114405)
        

        if (may_be_114394 and more_types_in_union_114395):
            # SSA join for if statement (line 218)
            module_type_store = module_type_store.join_ssa_context()


    
    # Assigning a type to the variable 'stypy_return_type' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'imsave(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'imsave' in the type store
    # Getting the type of 'stypy_return_type' (line 168)
    stypy_return_type_114407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114407)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'imsave'
    return stypy_return_type_114407

# Assigning a type to the variable 'imsave' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'imsave', imsave)

@norecursion
def fromimage(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 228)
    False_114408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'False')
    # Getting the type of 'None' (line 228)
    None_114409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 'None')
    defaults = [False_114408, None_114409]
    # Create a new context for function 'fromimage'
    module_type_store = module_type_store.open_function_context('fromimage', 225, 0, False)
    
    # Passed parameters checking function
    fromimage.stypy_localization = localization
    fromimage.stypy_type_of_self = None
    fromimage.stypy_type_store = module_type_store
    fromimage.stypy_function_name = 'fromimage'
    fromimage.stypy_param_names_list = ['im', 'flatten', 'mode']
    fromimage.stypy_varargs_param_name = None
    fromimage.stypy_kwargs_param_name = None
    fromimage.stypy_call_defaults = defaults
    fromimage.stypy_call_varargs = varargs
    fromimage.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fromimage', ['im', 'flatten', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fromimage', localization, ['im', 'flatten', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fromimage(...)' code ##################

    str_114410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, (-1)), 'str', "\n    Return a copy of a PIL image as a numpy array.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    Parameters\n    ----------\n    im : PIL image\n        Input image.\n    flatten : bool\n        If true, convert the output to grey-scale.\n    mode : str, optional\n        Mode to convert image to, e.g. ``'RGB'``.  See the Notes of the\n        `imread` docstring for more details.\n\n    Returns\n    -------\n    fromimage : ndarray\n        The different colour bands/channels are stored in the\n        third dimension, such that a grey-image is MxN, an\n        RGB-image MxNx3 and an RGBA-image MxNx4.\n\n    ")
    
    
    
    # Call to isImageType(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'im' (line 252)
    im_114413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'im', False)
    # Processing the call keyword arguments (line 252)
    kwargs_114414 = {}
    # Getting the type of 'Image' (line 252)
    Image_114411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 11), 'Image', False)
    # Obtaining the member 'isImageType' of a type (line 252)
    isImageType_114412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 11), Image_114411, 'isImageType')
    # Calling isImageType(args, kwargs) (line 252)
    isImageType_call_result_114415 = invoke(stypy.reporting.localization.Localization(__file__, 252, 11), isImageType_114412, *[im_114413], **kwargs_114414)
    
    # Applying the 'not' unary operator (line 252)
    result_not__114416 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 7), 'not', isImageType_call_result_114415)
    
    # Testing the type of an if condition (line 252)
    if_condition_114417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 4), result_not__114416)
    # Assigning a type to the variable 'if_condition_114417' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'if_condition_114417', if_condition_114417)
    # SSA begins for if statement (line 252)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 253)
    # Processing the call arguments (line 253)
    str_114419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 24), 'str', 'Input is not a PIL image.')
    # Processing the call keyword arguments (line 253)
    kwargs_114420 = {}
    # Getting the type of 'TypeError' (line 253)
    TypeError_114418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 253)
    TypeError_call_result_114421 = invoke(stypy.reporting.localization.Localization(__file__, 253, 14), TypeError_114418, *[str_114419], **kwargs_114420)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 253, 8), TypeError_call_result_114421, 'raise parameter', BaseException)
    # SSA join for if statement (line 252)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 255)
    # Getting the type of 'mode' (line 255)
    mode_114422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'mode')
    # Getting the type of 'None' (line 255)
    None_114423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 19), 'None')
    
    (may_be_114424, more_types_in_union_114425) = may_not_be_none(mode_114422, None_114423)

    if may_be_114424:

        if more_types_in_union_114425:
            # Runtime conditional SSA (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'mode' (line 256)
        mode_114426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'mode')
        # Getting the type of 'im' (line 256)
        im_114427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'im')
        # Obtaining the member 'mode' of a type (line 256)
        mode_114428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 19), im_114427, 'mode')
        # Applying the binary operator '!=' (line 256)
        result_ne_114429 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), '!=', mode_114426, mode_114428)
        
        # Testing the type of an if condition (line 256)
        if_condition_114430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 8), result_ne_114429)
        # Assigning a type to the variable 'if_condition_114430' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'if_condition_114430', if_condition_114430)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to convert(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'mode' (line 257)
        mode_114433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'mode', False)
        # Processing the call keyword arguments (line 257)
        kwargs_114434 = {}
        # Getting the type of 'im' (line 257)
        im_114431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 17), 'im', False)
        # Obtaining the member 'convert' of a type (line 257)
        convert_114432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 17), im_114431, 'convert')
        # Calling convert(args, kwargs) (line 257)
        convert_call_result_114435 = invoke(stypy.reporting.localization.Localization(__file__, 257, 17), convert_114432, *[mode_114433], **kwargs_114434)
        
        # Assigning a type to the variable 'im' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'im', convert_call_result_114435)
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_114425:
            # Runtime conditional SSA for else branch (line 255)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_114424) or more_types_in_union_114425):
        
        
        # Getting the type of 'im' (line 258)
        im_114436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 9), 'im')
        # Obtaining the member 'mode' of a type (line 258)
        mode_114437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 9), im_114436, 'mode')
        str_114438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 20), 'str', 'P')
        # Applying the binary operator '==' (line 258)
        result_eq_114439 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 9), '==', mode_114437, str_114438)
        
        # Testing the type of an if condition (line 258)
        if_condition_114440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 9), result_eq_114439)
        # Assigning a type to the variable 'if_condition_114440' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 9), 'if_condition_114440', if_condition_114440)
        # SSA begins for if statement (line 258)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        str_114441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 11), 'str', 'transparency')
        # Getting the type of 'im' (line 263)
        im_114442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'im')
        # Obtaining the member 'info' of a type (line 263)
        info_114443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 29), im_114442, 'info')
        # Applying the binary operator 'in' (line 263)
        result_contains_114444 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 11), 'in', str_114441, info_114443)
        
        # Testing the type of an if condition (line 263)
        if_condition_114445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 8), result_contains_114444)
        # Assigning a type to the variable 'if_condition_114445' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'if_condition_114445', if_condition_114445)
        # SSA begins for if statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to convert(...): (line 264)
        # Processing the call arguments (line 264)
        str_114448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 28), 'str', 'RGBA')
        # Processing the call keyword arguments (line 264)
        kwargs_114449 = {}
        # Getting the type of 'im' (line 264)
        im_114446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 17), 'im', False)
        # Obtaining the member 'convert' of a type (line 264)
        convert_114447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 17), im_114446, 'convert')
        # Calling convert(args, kwargs) (line 264)
        convert_call_result_114450 = invoke(stypy.reporting.localization.Localization(__file__, 264, 17), convert_114447, *[str_114448], **kwargs_114449)
        
        # Assigning a type to the variable 'im' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'im', convert_call_result_114450)
        # SSA branch for the else part of an if statement (line 263)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to convert(...): (line 266)
        # Processing the call arguments (line 266)
        str_114453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 28), 'str', 'RGB')
        # Processing the call keyword arguments (line 266)
        kwargs_114454 = {}
        # Getting the type of 'im' (line 266)
        im_114451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'im', False)
        # Obtaining the member 'convert' of a type (line 266)
        convert_114452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 17), im_114451, 'convert')
        # Calling convert(args, kwargs) (line 266)
        convert_call_result_114455 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), convert_114452, *[str_114453], **kwargs_114454)
        
        # Assigning a type to the variable 'im' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'im', convert_call_result_114455)
        # SSA join for if statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 258)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_114424 and more_types_in_union_114425):
            # SSA join for if statement (line 255)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'flatten' (line 268)
    flatten_114456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 7), 'flatten')
    # Testing the type of an if condition (line 268)
    if_condition_114457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 4), flatten_114456)
    # Assigning a type to the variable 'if_condition_114457' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'if_condition_114457', if_condition_114457)
    # SSA begins for if statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to convert(...): (line 269)
    # Processing the call arguments (line 269)
    str_114460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 24), 'str', 'F')
    # Processing the call keyword arguments (line 269)
    kwargs_114461 = {}
    # Getting the type of 'im' (line 269)
    im_114458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 13), 'im', False)
    # Obtaining the member 'convert' of a type (line 269)
    convert_114459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 13), im_114458, 'convert')
    # Calling convert(args, kwargs) (line 269)
    convert_call_result_114462 = invoke(stypy.reporting.localization.Localization(__file__, 269, 13), convert_114459, *[str_114460], **kwargs_114461)
    
    # Assigning a type to the variable 'im' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'im', convert_call_result_114462)
    # SSA branch for the else part of an if statement (line 268)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'im' (line 270)
    im_114463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 9), 'im')
    # Obtaining the member 'mode' of a type (line 270)
    mode_114464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 9), im_114463, 'mode')
    str_114465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 20), 'str', '1')
    # Applying the binary operator '==' (line 270)
    result_eq_114466 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 9), '==', mode_114464, str_114465)
    
    # Testing the type of an if condition (line 270)
    if_condition_114467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 9), result_eq_114466)
    # Assigning a type to the variable 'if_condition_114467' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 9), 'if_condition_114467', if_condition_114467)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 277):
    
    # Assigning a Call to a Name (line 277):
    
    # Call to convert(...): (line 277)
    # Processing the call arguments (line 277)
    str_114470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'str', 'L')
    # Processing the call keyword arguments (line 277)
    kwargs_114471 = {}
    # Getting the type of 'im' (line 277)
    im_114468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'im', False)
    # Obtaining the member 'convert' of a type (line 277)
    convert_114469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 13), im_114468, 'convert')
    # Calling convert(args, kwargs) (line 277)
    convert_call_result_114472 = invoke(stypy.reporting.localization.Localization(__file__, 277, 13), convert_114469, *[str_114470], **kwargs_114471)
    
    # Assigning a type to the variable 'im' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'im', convert_call_result_114472)
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 268)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to array(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'im' (line 279)
    im_114474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 14), 'im', False)
    # Processing the call keyword arguments (line 279)
    kwargs_114475 = {}
    # Getting the type of 'array' (line 279)
    array_114473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'array', False)
    # Calling array(args, kwargs) (line 279)
    array_call_result_114476 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), array_114473, *[im_114474], **kwargs_114475)
    
    # Assigning a type to the variable 'a' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'a', array_call_result_114476)
    # Getting the type of 'a' (line 280)
    a_114477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type', a_114477)
    
    # ################# End of 'fromimage(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fromimage' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_114478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114478)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fromimage'
    return stypy_return_type_114478

# Assigning a type to the variable 'fromimage' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'fromimage', fromimage)

# Assigning a Str to a Name (line 282):

# Assigning a Str to a Name (line 282):
str_114479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 10), 'str', 'Mode is unknown or incompatible with input array shape.')
# Assigning a type to the variable '_errstr' (line 282)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), '_errstr', str_114479)

@norecursion
def toimage(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_114480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 22), 'int')
    int_114481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 31), 'int')
    # Getting the type of 'None' (line 288)
    None_114482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 39), 'None')
    # Getting the type of 'None' (line 288)
    None_114483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 50), 'None')
    # Getting the type of 'None' (line 288)
    None_114484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 60), 'None')
    # Getting the type of 'None' (line 289)
    None_114485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 17), 'None')
    # Getting the type of 'None' (line 289)
    None_114486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 36), 'None')
    defaults = [int_114480, int_114481, None_114482, None_114483, None_114484, None_114485, None_114486]
    # Create a new context for function 'toimage'
    module_type_store = module_type_store.open_function_context('toimage', 285, 0, False)
    
    # Passed parameters checking function
    toimage.stypy_localization = localization
    toimage.stypy_type_of_self = None
    toimage.stypy_type_store = module_type_store
    toimage.stypy_function_name = 'toimage'
    toimage.stypy_param_names_list = ['arr', 'high', 'low', 'cmin', 'cmax', 'pal', 'mode', 'channel_axis']
    toimage.stypy_varargs_param_name = None
    toimage.stypy_kwargs_param_name = None
    toimage.stypy_call_defaults = defaults
    toimage.stypy_call_varargs = varargs
    toimage.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'toimage', ['arr', 'high', 'low', 'cmin', 'cmax', 'pal', 'mode', 'channel_axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'toimage', localization, ['arr', 'high', 'low', 'cmin', 'cmax', 'pal', 'mode', 'channel_axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'toimage(...)' code ##################

    str_114487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, (-1)), 'str', "Takes a numpy array and returns a PIL image.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    The mode of the PIL image depends on the array shape and the `pal` and\n    `mode` keywords.\n\n    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values\n    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode\n    is given as 'F' or 'I' in which case a float and/or integer array is made.\n\n    .. warning::\n\n        This function uses `bytescale` under the hood to rescale images to use\n        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.\n        It will also cast data for 2-D images to ``uint32`` for ``mode=None``\n        (which is the default).\n\n    Notes\n    -----\n    For 3-D arrays, the `channel_axis` argument tells which dimension of the\n    array holds the channel data.\n\n    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'\n    by default or 'YCbCr' if selected.\n\n    The numpy array must be either 2 dimensional or 3 dimensional.\n\n    ")
    
    # Assigning a Call to a Name (line 319):
    
    # Assigning a Call to a Name (line 319):
    
    # Call to asarray(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'arr' (line 319)
    arr_114489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'arr', False)
    # Processing the call keyword arguments (line 319)
    kwargs_114490 = {}
    # Getting the type of 'asarray' (line 319)
    asarray_114488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'asarray', False)
    # Calling asarray(args, kwargs) (line 319)
    asarray_call_result_114491 = invoke(stypy.reporting.localization.Localization(__file__, 319, 11), asarray_114488, *[arr_114489], **kwargs_114490)
    
    # Assigning a type to the variable 'data' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'data', asarray_call_result_114491)
    
    
    # Call to iscomplexobj(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'data' (line 320)
    data_114493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'data', False)
    # Processing the call keyword arguments (line 320)
    kwargs_114494 = {}
    # Getting the type of 'iscomplexobj' (line 320)
    iscomplexobj_114492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 320)
    iscomplexobj_call_result_114495 = invoke(stypy.reporting.localization.Localization(__file__, 320, 7), iscomplexobj_114492, *[data_114493], **kwargs_114494)
    
    # Testing the type of an if condition (line 320)
    if_condition_114496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 4), iscomplexobj_call_result_114495)
    # Assigning a type to the variable 'if_condition_114496' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'if_condition_114496', if_condition_114496)
    # SSA begins for if statement (line 320)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 321)
    # Processing the call arguments (line 321)
    str_114498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 25), 'str', 'Cannot convert a complex-valued array.')
    # Processing the call keyword arguments (line 321)
    kwargs_114499 = {}
    # Getting the type of 'ValueError' (line 321)
    ValueError_114497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 321)
    ValueError_call_result_114500 = invoke(stypy.reporting.localization.Localization(__file__, 321, 14), ValueError_114497, *[str_114498], **kwargs_114499)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 321, 8), ValueError_call_result_114500, 'raise parameter', BaseException)
    # SSA join for if statement (line 320)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to list(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'data' (line 322)
    data_114502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'data', False)
    # Obtaining the member 'shape' of a type (line 322)
    shape_114503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 17), data_114502, 'shape')
    # Processing the call keyword arguments (line 322)
    kwargs_114504 = {}
    # Getting the type of 'list' (line 322)
    list_114501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'list', False)
    # Calling list(args, kwargs) (line 322)
    list_call_result_114505 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), list_114501, *[shape_114503], **kwargs_114504)
    
    # Assigning a type to the variable 'shape' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'shape', list_call_result_114505)
    
    # Assigning a BoolOp to a Name (line 323):
    
    # Assigning a BoolOp to a Name (line 323):
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'shape' (line 323)
    shape_114507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'shape', False)
    # Processing the call keyword arguments (line 323)
    kwargs_114508 = {}
    # Getting the type of 'len' (line 323)
    len_114506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'len', False)
    # Calling len(args, kwargs) (line 323)
    len_call_result_114509 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), len_114506, *[shape_114507], **kwargs_114508)
    
    int_114510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 26), 'int')
    # Applying the binary operator '==' (line 323)
    result_eq_114511 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 12), '==', len_call_result_114509, int_114510)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'shape' (line 323)
    shape_114513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 37), 'shape', False)
    # Processing the call keyword arguments (line 323)
    kwargs_114514 = {}
    # Getting the type of 'len' (line 323)
    len_114512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 33), 'len', False)
    # Calling len(args, kwargs) (line 323)
    len_call_result_114515 = invoke(stypy.reporting.localization.Localization(__file__, 323, 33), len_114512, *[shape_114513], **kwargs_114514)
    
    int_114516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 47), 'int')
    # Applying the binary operator '==' (line 323)
    result_eq_114517 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 33), '==', len_call_result_114515, int_114516)
    
    
    # Evaluating a boolean operation
    
    int_114518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 34), 'int')
    # Getting the type of 'shape' (line 324)
    shape_114519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 39), 'shape')
    # Applying the binary operator 'in' (line 324)
    result_contains_114520 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 34), 'in', int_114518, shape_114519)
    
    
    int_114521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 50), 'int')
    # Getting the type of 'shape' (line 324)
    shape_114522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 55), 'shape')
    # Applying the binary operator 'in' (line 324)
    result_contains_114523 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 50), 'in', int_114521, shape_114522)
    
    # Applying the binary operator 'or' (line 324)
    result_or_keyword_114524 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 33), 'or', result_contains_114520, result_contains_114523)
    
    # Applying the binary operator 'and' (line 323)
    result_and_keyword_114525 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 32), 'and', result_eq_114517, result_or_keyword_114524)
    
    # Applying the binary operator 'or' (line 323)
    result_or_keyword_114526 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 12), 'or', result_eq_114511, result_and_keyword_114525)
    
    # Assigning a type to the variable 'valid' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'valid', result_or_keyword_114526)
    
    
    # Getting the type of 'valid' (line 325)
    valid_114527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'valid')
    # Applying the 'not' unary operator (line 325)
    result_not__114528 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 7), 'not', valid_114527)
    
    # Testing the type of an if condition (line 325)
    if_condition_114529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 4), result_not__114528)
    # Assigning a type to the variable 'if_condition_114529' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'if_condition_114529', if_condition_114529)
    # SSA begins for if statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 326)
    # Processing the call arguments (line 326)
    str_114531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 25), 'str', "'arr' does not have a suitable array shape for any mode.")
    # Processing the call keyword arguments (line 326)
    kwargs_114532 = {}
    # Getting the type of 'ValueError' (line 326)
    ValueError_114530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 326)
    ValueError_call_result_114533 = invoke(stypy.reporting.localization.Localization(__file__, 326, 14), ValueError_114530, *[str_114531], **kwargs_114532)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 326, 8), ValueError_call_result_114533, 'raise parameter', BaseException)
    # SSA join for if statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'shape' (line 328)
    shape_114535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'shape', False)
    # Processing the call keyword arguments (line 328)
    kwargs_114536 = {}
    # Getting the type of 'len' (line 328)
    len_114534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 7), 'len', False)
    # Calling len(args, kwargs) (line 328)
    len_call_result_114537 = invoke(stypy.reporting.localization.Localization(__file__, 328, 7), len_114534, *[shape_114535], **kwargs_114536)
    
    int_114538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 21), 'int')
    # Applying the binary operator '==' (line 328)
    result_eq_114539 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 7), '==', len_call_result_114537, int_114538)
    
    # Testing the type of an if condition (line 328)
    if_condition_114540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 4), result_eq_114539)
    # Assigning a type to the variable 'if_condition_114540' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'if_condition_114540', if_condition_114540)
    # SSA begins for if statement (line 328)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 329):
    
    # Assigning a Tuple to a Name (line 329):
    
    # Obtaining an instance of the builtin type 'tuple' (line 329)
    tuple_114541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 329)
    # Adding element type (line 329)
    
    # Obtaining the type of the subscript
    int_114542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 23), 'int')
    # Getting the type of 'shape' (line 329)
    shape_114543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 17), 'shape')
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___114544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 17), shape_114543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 329)
    subscript_call_result_114545 = invoke(stypy.reporting.localization.Localization(__file__, 329, 17), getitem___114544, int_114542)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 17), tuple_114541, subscript_call_result_114545)
    # Adding element type (line 329)
    
    # Obtaining the type of the subscript
    int_114546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'int')
    # Getting the type of 'shape' (line 329)
    shape_114547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'shape')
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___114548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 27), shape_114547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 329)
    subscript_call_result_114549 = invoke(stypy.reporting.localization.Localization(__file__, 329, 27), getitem___114548, int_114546)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 17), tuple_114541, subscript_call_result_114549)
    
    # Assigning a type to the variable 'shape' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'shape', tuple_114541)
    
    
    # Getting the type of 'mode' (line 330)
    mode_114550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'mode')
    str_114551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 19), 'str', 'F')
    # Applying the binary operator '==' (line 330)
    result_eq_114552 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 11), '==', mode_114550, str_114551)
    
    # Testing the type of an if condition (line 330)
    if_condition_114553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 8), result_eq_114552)
    # Assigning a type to the variable 'if_condition_114553' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'if_condition_114553', if_condition_114553)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 331):
    
    # Assigning a Call to a Name (line 331):
    
    # Call to astype(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'numpy' (line 331)
    numpy_114556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'numpy', False)
    # Obtaining the member 'float32' of a type (line 331)
    float32_114557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 33), numpy_114556, 'float32')
    # Processing the call keyword arguments (line 331)
    kwargs_114558 = {}
    # Getting the type of 'data' (line 331)
    data_114554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'data', False)
    # Obtaining the member 'astype' of a type (line 331)
    astype_114555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 21), data_114554, 'astype')
    # Calling astype(args, kwargs) (line 331)
    astype_call_result_114559 = invoke(stypy.reporting.localization.Localization(__file__, 331, 21), astype_114555, *[float32_114557], **kwargs_114558)
    
    # Assigning a type to the variable 'data32' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'data32', astype_call_result_114559)
    
    # Assigning a Call to a Name (line 332):
    
    # Assigning a Call to a Name (line 332):
    
    # Call to frombytes(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'mode' (line 332)
    mode_114562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 36), 'mode', False)
    # Getting the type of 'shape' (line 332)
    shape_114563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 42), 'shape', False)
    
    # Call to tostring(...): (line 332)
    # Processing the call keyword arguments (line 332)
    kwargs_114566 = {}
    # Getting the type of 'data32' (line 332)
    data32_114564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 49), 'data32', False)
    # Obtaining the member 'tostring' of a type (line 332)
    tostring_114565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 49), data32_114564, 'tostring')
    # Calling tostring(args, kwargs) (line 332)
    tostring_call_result_114567 = invoke(stypy.reporting.localization.Localization(__file__, 332, 49), tostring_114565, *[], **kwargs_114566)
    
    # Processing the call keyword arguments (line 332)
    kwargs_114568 = {}
    # Getting the type of 'Image' (line 332)
    Image_114560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 20), 'Image', False)
    # Obtaining the member 'frombytes' of a type (line 332)
    frombytes_114561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 20), Image_114560, 'frombytes')
    # Calling frombytes(args, kwargs) (line 332)
    frombytes_call_result_114569 = invoke(stypy.reporting.localization.Localization(__file__, 332, 20), frombytes_114561, *[mode_114562, shape_114563, tostring_call_result_114567], **kwargs_114568)
    
    # Assigning a type to the variable 'image' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'image', frombytes_call_result_114569)
    # Getting the type of 'image' (line 333)
    image_114570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 19), 'image')
    # Assigning a type to the variable 'stypy_return_type' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'stypy_return_type', image_114570)
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 334)
    mode_114571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 334)
    list_114572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 334)
    # Adding element type (line 334)
    # Getting the type of 'None' (line 334)
    None_114573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 19), list_114572, None_114573)
    # Adding element type (line 334)
    str_114574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 26), 'str', 'L')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 19), list_114572, str_114574)
    # Adding element type (line 334)
    str_114575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 31), 'str', 'P')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 19), list_114572, str_114575)
    
    # Applying the binary operator 'in' (line 334)
    result_contains_114576 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), 'in', mode_114571, list_114572)
    
    # Testing the type of an if condition (line 334)
    if_condition_114577 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), result_contains_114576)
    # Assigning a type to the variable 'if_condition_114577' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_114577', if_condition_114577)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 335):
    
    # Assigning a Call to a Name (line 335):
    
    # Call to bytescale(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'data' (line 335)
    data_114579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 33), 'data', False)
    # Processing the call keyword arguments (line 335)
    # Getting the type of 'high' (line 335)
    high_114580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 44), 'high', False)
    keyword_114581 = high_114580
    # Getting the type of 'low' (line 335)
    low_114582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 54), 'low', False)
    keyword_114583 = low_114582
    # Getting the type of 'cmin' (line 336)
    cmin_114584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 38), 'cmin', False)
    keyword_114585 = cmin_114584
    # Getting the type of 'cmax' (line 336)
    cmax_114586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 49), 'cmax', False)
    keyword_114587 = cmax_114586
    kwargs_114588 = {'high': keyword_114581, 'cmax': keyword_114587, 'low': keyword_114583, 'cmin': keyword_114585}
    # Getting the type of 'bytescale' (line 335)
    bytescale_114578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 23), 'bytescale', False)
    # Calling bytescale(args, kwargs) (line 335)
    bytescale_call_result_114589 = invoke(stypy.reporting.localization.Localization(__file__, 335, 23), bytescale_114578, *[data_114579], **kwargs_114588)
    
    # Assigning a type to the variable 'bytedata' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'bytedata', bytescale_call_result_114589)
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to frombytes(...): (line 337)
    # Processing the call arguments (line 337)
    str_114592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 36), 'str', 'L')
    # Getting the type of 'shape' (line 337)
    shape_114593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 41), 'shape', False)
    
    # Call to tostring(...): (line 337)
    # Processing the call keyword arguments (line 337)
    kwargs_114596 = {}
    # Getting the type of 'bytedata' (line 337)
    bytedata_114594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 48), 'bytedata', False)
    # Obtaining the member 'tostring' of a type (line 337)
    tostring_114595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 48), bytedata_114594, 'tostring')
    # Calling tostring(args, kwargs) (line 337)
    tostring_call_result_114597 = invoke(stypy.reporting.localization.Localization(__file__, 337, 48), tostring_114595, *[], **kwargs_114596)
    
    # Processing the call keyword arguments (line 337)
    kwargs_114598 = {}
    # Getting the type of 'Image' (line 337)
    Image_114590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'Image', False)
    # Obtaining the member 'frombytes' of a type (line 337)
    frombytes_114591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 20), Image_114590, 'frombytes')
    # Calling frombytes(args, kwargs) (line 337)
    frombytes_call_result_114599 = invoke(stypy.reporting.localization.Localization(__file__, 337, 20), frombytes_114591, *[str_114592, shape_114593, tostring_call_result_114597], **kwargs_114598)
    
    # Assigning a type to the variable 'image' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'image', frombytes_call_result_114599)
    
    # Type idiom detected: calculating its left and rigth part (line 338)
    # Getting the type of 'pal' (line 338)
    pal_114600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'pal')
    # Getting the type of 'None' (line 338)
    None_114601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 26), 'None')
    
    (may_be_114602, more_types_in_union_114603) = may_not_be_none(pal_114600, None_114601)

    if may_be_114602:

        if more_types_in_union_114603:
            # Runtime conditional SSA (line 338)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to putpalette(...): (line 339)
        # Processing the call arguments (line 339)
        
        # Call to tostring(...): (line 339)
        # Processing the call keyword arguments (line 339)
        kwargs_114613 = {}
        
        # Call to asarray(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'pal' (line 339)
        pal_114607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 41), 'pal', False)
        # Processing the call keyword arguments (line 339)
        # Getting the type of 'uint8' (line 339)
        uint8_114608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 52), 'uint8', False)
        keyword_114609 = uint8_114608
        kwargs_114610 = {'dtype': keyword_114609}
        # Getting the type of 'asarray' (line 339)
        asarray_114606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 33), 'asarray', False)
        # Calling asarray(args, kwargs) (line 339)
        asarray_call_result_114611 = invoke(stypy.reporting.localization.Localization(__file__, 339, 33), asarray_114606, *[pal_114607], **kwargs_114610)
        
        # Obtaining the member 'tostring' of a type (line 339)
        tostring_114612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 33), asarray_call_result_114611, 'tostring')
        # Calling tostring(args, kwargs) (line 339)
        tostring_call_result_114614 = invoke(stypy.reporting.localization.Localization(__file__, 339, 33), tostring_114612, *[], **kwargs_114613)
        
        # Processing the call keyword arguments (line 339)
        kwargs_114615 = {}
        # Getting the type of 'image' (line 339)
        image_114604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'image', False)
        # Obtaining the member 'putpalette' of a type (line 339)
        putpalette_114605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 16), image_114604, 'putpalette')
        # Calling putpalette(args, kwargs) (line 339)
        putpalette_call_result_114616 = invoke(stypy.reporting.localization.Localization(__file__, 339, 16), putpalette_114605, *[tostring_call_result_114614], **kwargs_114615)
        

        if more_types_in_union_114603:
            # Runtime conditional SSA for else branch (line 338)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_114602) or more_types_in_union_114603):
        
        
        # Getting the type of 'mode' (line 341)
        mode_114617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'mode')
        str_114618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 25), 'str', 'P')
        # Applying the binary operator '==' (line 341)
        result_eq_114619 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 17), '==', mode_114617, str_114618)
        
        # Testing the type of an if condition (line 341)
        if_condition_114620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 17), result_eq_114619)
        # Assigning a type to the variable 'if_condition_114620' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'if_condition_114620', if_condition_114620)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 342):
        
        # Assigning a BinOp to a Name (line 342):
        
        # Obtaining the type of the subscript
        slice_114621 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 342, 23), None, None, None)
        # Getting the type of 'newaxis' (line 342)
        newaxis_114622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 57), 'newaxis')
        
        # Call to arange(...): (line 342)
        # Processing the call arguments (line 342)
        int_114624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 30), 'int')
        int_114625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 33), 'int')
        int_114626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 38), 'int')
        # Processing the call keyword arguments (line 342)
        # Getting the type of 'uint8' (line 342)
        uint8_114627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 47), 'uint8', False)
        keyword_114628 = uint8_114627
        kwargs_114629 = {'dtype': keyword_114628}
        # Getting the type of 'arange' (line 342)
        arange_114623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 23), 'arange', False)
        # Calling arange(args, kwargs) (line 342)
        arange_call_result_114630 = invoke(stypy.reporting.localization.Localization(__file__, 342, 23), arange_114623, *[int_114624, int_114625, int_114626], **kwargs_114629)
        
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___114631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 23), arange_call_result_114630, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_114632 = invoke(stypy.reporting.localization.Localization(__file__, 342, 23), getitem___114631, (slice_114621, newaxis_114622))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'newaxis' (line 343)
        newaxis_114633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 47), 'newaxis')
        slice_114634 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 343, 23), None, None, None)
        
        # Call to ones(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining an instance of the builtin type 'tuple' (line 343)
        tuple_114636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 343)
        # Adding element type (line 343)
        int_114637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 29), tuple_114636, int_114637)
        
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'uint8' (line 343)
        uint8_114638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 40), 'uint8', False)
        keyword_114639 = uint8_114638
        kwargs_114640 = {'dtype': keyword_114639}
        # Getting the type of 'ones' (line 343)
        ones_114635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'ones', False)
        # Calling ones(args, kwargs) (line 343)
        ones_call_result_114641 = invoke(stypy.reporting.localization.Localization(__file__, 343, 23), ones_114635, *[tuple_114636], **kwargs_114640)
        
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___114642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 23), ones_call_result_114641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_114643 = invoke(stypy.reporting.localization.Localization(__file__, 343, 23), getitem___114642, (newaxis_114633, slice_114634))
        
        # Applying the binary operator '*' (line 342)
        result_mul_114644 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 23), '*', subscript_call_result_114632, subscript_call_result_114643)
        
        # Assigning a type to the variable 'pal' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'pal', result_mul_114644)
        
        # Call to putpalette(...): (line 344)
        # Processing the call arguments (line 344)
        
        # Call to tostring(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_114654 = {}
        
        # Call to asarray(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'pal' (line 344)
        pal_114648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 41), 'pal', False)
        # Processing the call keyword arguments (line 344)
        # Getting the type of 'uint8' (line 344)
        uint8_114649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 52), 'uint8', False)
        keyword_114650 = uint8_114649
        kwargs_114651 = {'dtype': keyword_114650}
        # Getting the type of 'asarray' (line 344)
        asarray_114647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), 'asarray', False)
        # Calling asarray(args, kwargs) (line 344)
        asarray_call_result_114652 = invoke(stypy.reporting.localization.Localization(__file__, 344, 33), asarray_114647, *[pal_114648], **kwargs_114651)
        
        # Obtaining the member 'tostring' of a type (line 344)
        tostring_114653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 33), asarray_call_result_114652, 'tostring')
        # Calling tostring(args, kwargs) (line 344)
        tostring_call_result_114655 = invoke(stypy.reporting.localization.Localization(__file__, 344, 33), tostring_114653, *[], **kwargs_114654)
        
        # Processing the call keyword arguments (line 344)
        kwargs_114656 = {}
        # Getting the type of 'image' (line 344)
        image_114645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'image', False)
        # Obtaining the member 'putpalette' of a type (line 344)
        putpalette_114646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 16), image_114645, 'putpalette')
        # Calling putpalette(args, kwargs) (line 344)
        putpalette_call_result_114657 = invoke(stypy.reporting.localization.Localization(__file__, 344, 16), putpalette_114646, *[tostring_call_result_114655], **kwargs_114656)
        
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_114602 and more_types_in_union_114603):
            # SSA join for if statement (line 338)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'image' (line 345)
    image_114658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'image')
    # Assigning a type to the variable 'stypy_return_type' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', image_114658)
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 346)
    mode_114659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 11), 'mode')
    str_114660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 19), 'str', '1')
    # Applying the binary operator '==' (line 346)
    result_eq_114661 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 11), '==', mode_114659, str_114660)
    
    # Testing the type of an if condition (line 346)
    if_condition_114662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 8), result_eq_114661)
    # Assigning a type to the variable 'if_condition_114662' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'if_condition_114662', if_condition_114662)
    # SSA begins for if statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Compare to a Name (line 347):
    
    # Assigning a Compare to a Name (line 347):
    
    # Getting the type of 'data' (line 347)
    data_114663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'data')
    # Getting the type of 'high' (line 347)
    high_114664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 31), 'high')
    # Applying the binary operator '>' (line 347)
    result_gt_114665 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 24), '>', data_114663, high_114664)
    
    # Assigning a type to the variable 'bytedata' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'bytedata', result_gt_114665)
    
    # Assigning a Call to a Name (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to frombytes(...): (line 348)
    # Processing the call arguments (line 348)
    str_114668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 36), 'str', '1')
    # Getting the type of 'shape' (line 348)
    shape_114669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 41), 'shape', False)
    
    # Call to tostring(...): (line 348)
    # Processing the call keyword arguments (line 348)
    kwargs_114672 = {}
    # Getting the type of 'bytedata' (line 348)
    bytedata_114670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 48), 'bytedata', False)
    # Obtaining the member 'tostring' of a type (line 348)
    tostring_114671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 48), bytedata_114670, 'tostring')
    # Calling tostring(args, kwargs) (line 348)
    tostring_call_result_114673 = invoke(stypy.reporting.localization.Localization(__file__, 348, 48), tostring_114671, *[], **kwargs_114672)
    
    # Processing the call keyword arguments (line 348)
    kwargs_114674 = {}
    # Getting the type of 'Image' (line 348)
    Image_114666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'Image', False)
    # Obtaining the member 'frombytes' of a type (line 348)
    frombytes_114667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 20), Image_114666, 'frombytes')
    # Calling frombytes(args, kwargs) (line 348)
    frombytes_call_result_114675 = invoke(stypy.reporting.localization.Localization(__file__, 348, 20), frombytes_114667, *[str_114668, shape_114669, tostring_call_result_114673], **kwargs_114674)
    
    # Assigning a type to the variable 'image' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'image', frombytes_call_result_114675)
    # Getting the type of 'image' (line 349)
    image_114676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'image')
    # Assigning a type to the variable 'stypy_return_type' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'stypy_return_type', image_114676)
    # SSA join for if statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 350)
    # Getting the type of 'cmin' (line 350)
    cmin_114677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'cmin')
    # Getting the type of 'None' (line 350)
    None_114678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'None')
    
    (may_be_114679, more_types_in_union_114680) = may_be_none(cmin_114677, None_114678)

    if may_be_114679:

        if more_types_in_union_114680:
            # Runtime conditional SSA (line 350)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to amin(...): (line 351)
        # Processing the call arguments (line 351)
        
        # Call to ravel(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'data' (line 351)
        data_114683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 30), 'data', False)
        # Processing the call keyword arguments (line 351)
        kwargs_114684 = {}
        # Getting the type of 'ravel' (line 351)
        ravel_114682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'ravel', False)
        # Calling ravel(args, kwargs) (line 351)
        ravel_call_result_114685 = invoke(stypy.reporting.localization.Localization(__file__, 351, 24), ravel_114682, *[data_114683], **kwargs_114684)
        
        # Processing the call keyword arguments (line 351)
        kwargs_114686 = {}
        # Getting the type of 'amin' (line 351)
        amin_114681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'amin', False)
        # Calling amin(args, kwargs) (line 351)
        amin_call_result_114687 = invoke(stypy.reporting.localization.Localization(__file__, 351, 19), amin_114681, *[ravel_call_result_114685], **kwargs_114686)
        
        # Assigning a type to the variable 'cmin' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'cmin', amin_call_result_114687)

        if more_types_in_union_114680:
            # SSA join for if statement (line 350)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 352)
    # Getting the type of 'cmax' (line 352)
    cmax_114688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 11), 'cmax')
    # Getting the type of 'None' (line 352)
    None_114689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 19), 'None')
    
    (may_be_114690, more_types_in_union_114691) = may_be_none(cmax_114688, None_114689)

    if may_be_114690:

        if more_types_in_union_114691:
            # Runtime conditional SSA (line 352)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Call to amax(...): (line 353)
        # Processing the call arguments (line 353)
        
        # Call to ravel(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'data' (line 353)
        data_114694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'data', False)
        # Processing the call keyword arguments (line 353)
        kwargs_114695 = {}
        # Getting the type of 'ravel' (line 353)
        ravel_114693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), 'ravel', False)
        # Calling ravel(args, kwargs) (line 353)
        ravel_call_result_114696 = invoke(stypy.reporting.localization.Localization(__file__, 353, 24), ravel_114693, *[data_114694], **kwargs_114695)
        
        # Processing the call keyword arguments (line 353)
        kwargs_114697 = {}
        # Getting the type of 'amax' (line 353)
        amax_114692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'amax', False)
        # Calling amax(args, kwargs) (line 353)
        amax_call_result_114698 = invoke(stypy.reporting.localization.Localization(__file__, 353, 19), amax_114692, *[ravel_call_result_114696], **kwargs_114697)
        
        # Assigning a type to the variable 'cmax' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'cmax', amax_call_result_114698)

        if more_types_in_union_114691:
            # SSA join for if statement (line 352)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 354):
    
    # Assigning a BinOp to a Name (line 354):
    # Getting the type of 'data' (line 354)
    data_114699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'data')
    float_114700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 21), 'float')
    # Applying the binary operator '*' (line 354)
    result_mul_114701 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 16), '*', data_114699, float_114700)
    
    # Getting the type of 'cmin' (line 354)
    cmin_114702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'cmin')
    # Applying the binary operator '-' (line 354)
    result_sub_114703 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 16), '-', result_mul_114701, cmin_114702)
    
    # Getting the type of 'high' (line 354)
    high_114704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 34), 'high')
    # Getting the type of 'low' (line 354)
    low_114705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 41), 'low')
    # Applying the binary operator '-' (line 354)
    result_sub_114706 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 34), '-', high_114704, low_114705)
    
    # Applying the binary operator '*' (line 354)
    result_mul_114707 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 15), '*', result_sub_114703, result_sub_114706)
    
    # Getting the type of 'cmax' (line 354)
    cmax_114708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 47), 'cmax')
    # Getting the type of 'cmin' (line 354)
    cmin_114709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 54), 'cmin')
    # Applying the binary operator '-' (line 354)
    result_sub_114710 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 47), '-', cmax_114708, cmin_114709)
    
    # Applying the binary operator 'div' (line 354)
    result_div_114711 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 45), 'div', result_mul_114707, result_sub_114710)
    
    # Getting the type of 'low' (line 354)
    low_114712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 62), 'low')
    # Applying the binary operator '+' (line 354)
    result_add_114713 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 15), '+', result_div_114711, low_114712)
    
    # Assigning a type to the variable 'data' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'data', result_add_114713)
    
    
    # Getting the type of 'mode' (line 355)
    mode_114714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'mode')
    str_114715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 19), 'str', 'I')
    # Applying the binary operator '==' (line 355)
    result_eq_114716 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 11), '==', mode_114714, str_114715)
    
    # Testing the type of an if condition (line 355)
    if_condition_114717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), result_eq_114716)
    # Assigning a type to the variable 'if_condition_114717' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_114717', if_condition_114717)
    # SSA begins for if statement (line 355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 356):
    
    # Assigning a Call to a Name (line 356):
    
    # Call to astype(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'numpy' (line 356)
    numpy_114720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), 'numpy', False)
    # Obtaining the member 'uint32' of a type (line 356)
    uint32_114721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 33), numpy_114720, 'uint32')
    # Processing the call keyword arguments (line 356)
    kwargs_114722 = {}
    # Getting the type of 'data' (line 356)
    data_114718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'data', False)
    # Obtaining the member 'astype' of a type (line 356)
    astype_114719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 21), data_114718, 'astype')
    # Calling astype(args, kwargs) (line 356)
    astype_call_result_114723 = invoke(stypy.reporting.localization.Localization(__file__, 356, 21), astype_114719, *[uint32_114721], **kwargs_114722)
    
    # Assigning a type to the variable 'data32' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'data32', astype_call_result_114723)
    
    # Assigning a Call to a Name (line 357):
    
    # Assigning a Call to a Name (line 357):
    
    # Call to frombytes(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'mode' (line 357)
    mode_114726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 36), 'mode', False)
    # Getting the type of 'shape' (line 357)
    shape_114727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 42), 'shape', False)
    
    # Call to tostring(...): (line 357)
    # Processing the call keyword arguments (line 357)
    kwargs_114730 = {}
    # Getting the type of 'data32' (line 357)
    data32_114728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 49), 'data32', False)
    # Obtaining the member 'tostring' of a type (line 357)
    tostring_114729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 49), data32_114728, 'tostring')
    # Calling tostring(args, kwargs) (line 357)
    tostring_call_result_114731 = invoke(stypy.reporting.localization.Localization(__file__, 357, 49), tostring_114729, *[], **kwargs_114730)
    
    # Processing the call keyword arguments (line 357)
    kwargs_114732 = {}
    # Getting the type of 'Image' (line 357)
    Image_114724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'Image', False)
    # Obtaining the member 'frombytes' of a type (line 357)
    frombytes_114725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 20), Image_114724, 'frombytes')
    # Calling frombytes(args, kwargs) (line 357)
    frombytes_call_result_114733 = invoke(stypy.reporting.localization.Localization(__file__, 357, 20), frombytes_114725, *[mode_114726, shape_114727, tostring_call_result_114731], **kwargs_114732)
    
    # Assigning a type to the variable 'image' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'image', frombytes_call_result_114733)
    # SSA branch for the else part of an if statement (line 355)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of '_errstr' (line 359)
    _errstr_114735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 29), '_errstr', False)
    # Processing the call keyword arguments (line 359)
    kwargs_114736 = {}
    # Getting the type of 'ValueError' (line 359)
    ValueError_114734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 359)
    ValueError_call_result_114737 = invoke(stypy.reporting.localization.Localization(__file__, 359, 18), ValueError_114734, *[_errstr_114735], **kwargs_114736)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 359, 12), ValueError_call_result_114737, 'raise parameter', BaseException)
    # SSA join for if statement (line 355)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'image' (line 360)
    image_114738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'image')
    # Assigning a type to the variable 'stypy_return_type' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'stypy_return_type', image_114738)
    # SSA join for if statement (line 328)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 364)
    # Getting the type of 'channel_axis' (line 364)
    channel_axis_114739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 7), 'channel_axis')
    # Getting the type of 'None' (line 364)
    None_114740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'None')
    
    (may_be_114741, more_types_in_union_114742) = may_be_none(channel_axis_114739, None_114740)

    if may_be_114741:

        if more_types_in_union_114742:
            # Runtime conditional SSA (line 364)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        int_114743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 12), 'int')
        # Getting the type of 'shape' (line 365)
        shape_114744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 17), 'shape')
        # Applying the binary operator 'in' (line 365)
        result_contains_114745 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 12), 'in', int_114743, shape_114744)
        
        # Testing the type of an if condition (line 365)
        if_condition_114746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 8), result_contains_114745)
        # Assigning a type to the variable 'if_condition_114746' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'if_condition_114746', if_condition_114746)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 366):
        
        # Assigning a Subscript to a Name (line 366):
        
        # Obtaining the type of the subscript
        int_114747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 56), 'int')
        
        # Call to flatnonzero(...): (line 366)
        # Processing the call arguments (line 366)
        
        
        # Call to asarray(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'shape' (line 366)
        shape_114751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 43), 'shape', False)
        # Processing the call keyword arguments (line 366)
        kwargs_114752 = {}
        # Getting the type of 'asarray' (line 366)
        asarray_114750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 35), 'asarray', False)
        # Calling asarray(args, kwargs) (line 366)
        asarray_call_result_114753 = invoke(stypy.reporting.localization.Localization(__file__, 366, 35), asarray_114750, *[shape_114751], **kwargs_114752)
        
        int_114754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 53), 'int')
        # Applying the binary operator '==' (line 366)
        result_eq_114755 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 35), '==', asarray_call_result_114753, int_114754)
        
        # Processing the call keyword arguments (line 366)
        kwargs_114756 = {}
        # Getting the type of 'numpy' (line 366)
        numpy_114748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 17), 'numpy', False)
        # Obtaining the member 'flatnonzero' of a type (line 366)
        flatnonzero_114749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 17), numpy_114748, 'flatnonzero')
        # Calling flatnonzero(args, kwargs) (line 366)
        flatnonzero_call_result_114757 = invoke(stypy.reporting.localization.Localization(__file__, 366, 17), flatnonzero_114749, *[result_eq_114755], **kwargs_114756)
        
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___114758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 17), flatnonzero_call_result_114757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_114759 = invoke(stypy.reporting.localization.Localization(__file__, 366, 17), getitem___114758, int_114747)
        
        # Assigning a type to the variable 'ca' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'ca', subscript_call_result_114759)
        # SSA branch for the else part of an if statement (line 365)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to flatnonzero(...): (line 368)
        # Processing the call arguments (line 368)
        
        
        # Call to asarray(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'shape' (line 368)
        shape_114763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 43), 'shape', False)
        # Processing the call keyword arguments (line 368)
        kwargs_114764 = {}
        # Getting the type of 'asarray' (line 368)
        asarray_114762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'asarray', False)
        # Calling asarray(args, kwargs) (line 368)
        asarray_call_result_114765 = invoke(stypy.reporting.localization.Localization(__file__, 368, 35), asarray_114762, *[shape_114763], **kwargs_114764)
        
        int_114766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 53), 'int')
        # Applying the binary operator '==' (line 368)
        result_eq_114767 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 35), '==', asarray_call_result_114765, int_114766)
        
        # Processing the call keyword arguments (line 368)
        kwargs_114768 = {}
        # Getting the type of 'numpy' (line 368)
        numpy_114760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 17), 'numpy', False)
        # Obtaining the member 'flatnonzero' of a type (line 368)
        flatnonzero_114761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 17), numpy_114760, 'flatnonzero')
        # Calling flatnonzero(args, kwargs) (line 368)
        flatnonzero_call_result_114769 = invoke(stypy.reporting.localization.Localization(__file__, 368, 17), flatnonzero_114761, *[result_eq_114767], **kwargs_114768)
        
        # Assigning a type to the variable 'ca' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'ca', flatnonzero_call_result_114769)
        
        
        # Call to len(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'ca' (line 369)
        ca_114771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'ca', False)
        # Processing the call keyword arguments (line 369)
        kwargs_114772 = {}
        # Getting the type of 'len' (line 369)
        len_114770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'len', False)
        # Calling len(args, kwargs) (line 369)
        len_call_result_114773 = invoke(stypy.reporting.localization.Localization(__file__, 369, 15), len_114770, *[ca_114771], **kwargs_114772)
        
        # Testing the type of an if condition (line 369)
        if_condition_114774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 12), len_call_result_114773)
        # Assigning a type to the variable 'if_condition_114774' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'if_condition_114774', if_condition_114774)
        # SSA begins for if statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 370):
        
        # Assigning a Subscript to a Name (line 370):
        
        # Obtaining the type of the subscript
        int_114775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 24), 'int')
        # Getting the type of 'ca' (line 370)
        ca_114776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 21), 'ca')
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___114777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 21), ca_114776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_114778 = invoke(stypy.reporting.localization.Localization(__file__, 370, 21), getitem___114777, int_114775)
        
        # Assigning a type to the variable 'ca' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'ca', subscript_call_result_114778)
        # SSA branch for the else part of an if statement (line 369)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 372)
        # Processing the call arguments (line 372)
        str_114780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 33), 'str', 'Could not find channel dimension.')
        # Processing the call keyword arguments (line 372)
        kwargs_114781 = {}
        # Getting the type of 'ValueError' (line 372)
        ValueError_114779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 372)
        ValueError_call_result_114782 = invoke(stypy.reporting.localization.Localization(__file__, 372, 22), ValueError_114779, *[str_114780], **kwargs_114781)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 372, 16), ValueError_call_result_114782, 'raise parameter', BaseException)
        # SSA join for if statement (line 369)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_114742:
            # Runtime conditional SSA for else branch (line 364)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_114741) or more_types_in_union_114742):
        
        # Assigning a Name to a Name (line 374):
        
        # Assigning a Name to a Name (line 374):
        # Getting the type of 'channel_axis' (line 374)
        channel_axis_114783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 13), 'channel_axis')
        # Assigning a type to the variable 'ca' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'ca', channel_axis_114783)

        if (may_be_114741 and more_types_in_union_114742):
            # SSA join for if statement (line 364)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 376):
    
    # Assigning a Subscript to a Name (line 376):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ca' (line 376)
    ca_114784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 18), 'ca')
    # Getting the type of 'shape' (line 376)
    shape_114785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'shape')
    # Obtaining the member '__getitem__' of a type (line 376)
    getitem___114786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), shape_114785, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 376)
    subscript_call_result_114787 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), getitem___114786, ca_114784)
    
    # Assigning a type to the variable 'numch' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'numch', subscript_call_result_114787)
    
    
    # Getting the type of 'numch' (line 377)
    numch_114788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 7), 'numch')
    
    # Obtaining an instance of the builtin type 'list' (line 377)
    list_114789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 377)
    # Adding element type (line 377)
    int_114790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 20), list_114789, int_114790)
    # Adding element type (line 377)
    int_114791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 20), list_114789, int_114791)
    
    # Applying the binary operator 'notin' (line 377)
    result_contains_114792 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 7), 'notin', numch_114788, list_114789)
    
    # Testing the type of an if condition (line 377)
    if_condition_114793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 4), result_contains_114792)
    # Assigning a type to the variable 'if_condition_114793' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'if_condition_114793', if_condition_114793)
    # SSA begins for if statement (line 377)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 378)
    # Processing the call arguments (line 378)
    str_114795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 25), 'str', 'Channel axis dimension is not valid.')
    # Processing the call keyword arguments (line 378)
    kwargs_114796 = {}
    # Getting the type of 'ValueError' (line 378)
    ValueError_114794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 378)
    ValueError_call_result_114797 = invoke(stypy.reporting.localization.Localization(__file__, 378, 14), ValueError_114794, *[str_114795], **kwargs_114796)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 378, 8), ValueError_call_result_114797, 'raise parameter', BaseException)
    # SSA join for if statement (line 377)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 380):
    
    # Assigning a Call to a Name (line 380):
    
    # Call to bytescale(...): (line 380)
    # Processing the call arguments (line 380)
    # Getting the type of 'data' (line 380)
    data_114799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 25), 'data', False)
    # Processing the call keyword arguments (line 380)
    # Getting the type of 'high' (line 380)
    high_114800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 36), 'high', False)
    keyword_114801 = high_114800
    # Getting the type of 'low' (line 380)
    low_114802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 46), 'low', False)
    keyword_114803 = low_114802
    # Getting the type of 'cmin' (line 380)
    cmin_114804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 56), 'cmin', False)
    keyword_114805 = cmin_114804
    # Getting the type of 'cmax' (line 380)
    cmax_114806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 67), 'cmax', False)
    keyword_114807 = cmax_114806
    kwargs_114808 = {'high': keyword_114801, 'cmax': keyword_114807, 'low': keyword_114803, 'cmin': keyword_114805}
    # Getting the type of 'bytescale' (line 380)
    bytescale_114798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'bytescale', False)
    # Calling bytescale(args, kwargs) (line 380)
    bytescale_call_result_114809 = invoke(stypy.reporting.localization.Localization(__file__, 380, 15), bytescale_114798, *[data_114799], **kwargs_114808)
    
    # Assigning a type to the variable 'bytedata' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'bytedata', bytescale_call_result_114809)
    
    
    # Getting the type of 'ca' (line 381)
    ca_114810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 7), 'ca')
    int_114811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 13), 'int')
    # Applying the binary operator '==' (line 381)
    result_eq_114812 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 7), '==', ca_114810, int_114811)
    
    # Testing the type of an if condition (line 381)
    if_condition_114813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 4), result_eq_114812)
    # Assigning a type to the variable 'if_condition_114813' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'if_condition_114813', if_condition_114813)
    # SSA begins for if statement (line 381)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 382):
    
    # Assigning a Call to a Name (line 382):
    
    # Call to tostring(...): (line 382)
    # Processing the call keyword arguments (line 382)
    kwargs_114816 = {}
    # Getting the type of 'bytedata' (line 382)
    bytedata_114814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 18), 'bytedata', False)
    # Obtaining the member 'tostring' of a type (line 382)
    tostring_114815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 18), bytedata_114814, 'tostring')
    # Calling tostring(args, kwargs) (line 382)
    tostring_call_result_114817 = invoke(stypy.reporting.localization.Localization(__file__, 382, 18), tostring_114815, *[], **kwargs_114816)
    
    # Assigning a type to the variable 'strdata' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'strdata', tostring_call_result_114817)
    
    # Assigning a Tuple to a Name (line 383):
    
    # Assigning a Tuple to a Name (line 383):
    
    # Obtaining an instance of the builtin type 'tuple' (line 383)
    tuple_114818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 383)
    # Adding element type (line 383)
    
    # Obtaining the type of the subscript
    int_114819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 23), 'int')
    # Getting the type of 'shape' (line 383)
    shape_114820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 17), 'shape')
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___114821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 17), shape_114820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_114822 = invoke(stypy.reporting.localization.Localization(__file__, 383, 17), getitem___114821, int_114819)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 17), tuple_114818, subscript_call_result_114822)
    # Adding element type (line 383)
    
    # Obtaining the type of the subscript
    int_114823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 33), 'int')
    # Getting the type of 'shape' (line 383)
    shape_114824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 27), 'shape')
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___114825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 27), shape_114824, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_114826 = invoke(stypy.reporting.localization.Localization(__file__, 383, 27), getitem___114825, int_114823)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 17), tuple_114818, subscript_call_result_114826)
    
    # Assigning a type to the variable 'shape' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'shape', tuple_114818)
    # SSA branch for the else part of an if statement (line 381)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ca' (line 384)
    ca_114827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 9), 'ca')
    int_114828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 15), 'int')
    # Applying the binary operator '==' (line 384)
    result_eq_114829 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 9), '==', ca_114827, int_114828)
    
    # Testing the type of an if condition (line 384)
    if_condition_114830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 9), result_eq_114829)
    # Assigning a type to the variable 'if_condition_114830' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 9), 'if_condition_114830', if_condition_114830)
    # SSA begins for if statement (line 384)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 385):
    
    # Assigning a Call to a Name (line 385):
    
    # Call to tostring(...): (line 385)
    # Processing the call keyword arguments (line 385)
    kwargs_114840 = {}
    
    # Call to transpose(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'bytedata' (line 385)
    bytedata_114832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'bytedata', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 385)
    tuple_114833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 385)
    # Adding element type (line 385)
    int_114834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 39), tuple_114833, int_114834)
    # Adding element type (line 385)
    int_114835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 39), tuple_114833, int_114835)
    # Adding element type (line 385)
    int_114836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 39), tuple_114833, int_114836)
    
    # Processing the call keyword arguments (line 385)
    kwargs_114837 = {}
    # Getting the type of 'transpose' (line 385)
    transpose_114831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'transpose', False)
    # Calling transpose(args, kwargs) (line 385)
    transpose_call_result_114838 = invoke(stypy.reporting.localization.Localization(__file__, 385, 18), transpose_114831, *[bytedata_114832, tuple_114833], **kwargs_114837)
    
    # Obtaining the member 'tostring' of a type (line 385)
    tostring_114839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), transpose_call_result_114838, 'tostring')
    # Calling tostring(args, kwargs) (line 385)
    tostring_call_result_114841 = invoke(stypy.reporting.localization.Localization(__file__, 385, 18), tostring_114839, *[], **kwargs_114840)
    
    # Assigning a type to the variable 'strdata' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'strdata', tostring_call_result_114841)
    
    # Assigning a Tuple to a Name (line 386):
    
    # Assigning a Tuple to a Name (line 386):
    
    # Obtaining an instance of the builtin type 'tuple' (line 386)
    tuple_114842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 386)
    # Adding element type (line 386)
    
    # Obtaining the type of the subscript
    int_114843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 23), 'int')
    # Getting the type of 'shape' (line 386)
    shape_114844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'shape')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___114845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 17), shape_114844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_114846 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), getitem___114845, int_114843)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), tuple_114842, subscript_call_result_114846)
    # Adding element type (line 386)
    
    # Obtaining the type of the subscript
    int_114847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 33), 'int')
    # Getting the type of 'shape' (line 386)
    shape_114848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 27), 'shape')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___114849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 27), shape_114848, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_114850 = invoke(stypy.reporting.localization.Localization(__file__, 386, 27), getitem___114849, int_114847)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), tuple_114842, subscript_call_result_114850)
    
    # Assigning a type to the variable 'shape' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'shape', tuple_114842)
    # SSA branch for the else part of an if statement (line 384)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ca' (line 387)
    ca_114851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 9), 'ca')
    int_114852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 15), 'int')
    # Applying the binary operator '==' (line 387)
    result_eq_114853 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 9), '==', ca_114851, int_114852)
    
    # Testing the type of an if condition (line 387)
    if_condition_114854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 9), result_eq_114853)
    # Assigning a type to the variable 'if_condition_114854' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 9), 'if_condition_114854', if_condition_114854)
    # SSA begins for if statement (line 387)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to tostring(...): (line 388)
    # Processing the call keyword arguments (line 388)
    kwargs_114864 = {}
    
    # Call to transpose(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'bytedata' (line 388)
    bytedata_114856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 28), 'bytedata', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 388)
    tuple_114857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 388)
    # Adding element type (line 388)
    int_114858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 39), tuple_114857, int_114858)
    # Adding element type (line 388)
    int_114859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 39), tuple_114857, int_114859)
    # Adding element type (line 388)
    int_114860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 39), tuple_114857, int_114860)
    
    # Processing the call keyword arguments (line 388)
    kwargs_114861 = {}
    # Getting the type of 'transpose' (line 388)
    transpose_114855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 18), 'transpose', False)
    # Calling transpose(args, kwargs) (line 388)
    transpose_call_result_114862 = invoke(stypy.reporting.localization.Localization(__file__, 388, 18), transpose_114855, *[bytedata_114856, tuple_114857], **kwargs_114861)
    
    # Obtaining the member 'tostring' of a type (line 388)
    tostring_114863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 18), transpose_call_result_114862, 'tostring')
    # Calling tostring(args, kwargs) (line 388)
    tostring_call_result_114865 = invoke(stypy.reporting.localization.Localization(__file__, 388, 18), tostring_114863, *[], **kwargs_114864)
    
    # Assigning a type to the variable 'strdata' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'strdata', tostring_call_result_114865)
    
    # Assigning a Tuple to a Name (line 389):
    
    # Assigning a Tuple to a Name (line 389):
    
    # Obtaining an instance of the builtin type 'tuple' (line 389)
    tuple_114866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 389)
    # Adding element type (line 389)
    
    # Obtaining the type of the subscript
    int_114867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 23), 'int')
    # Getting the type of 'shape' (line 389)
    shape_114868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'shape')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___114869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 17), shape_114868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_114870 = invoke(stypy.reporting.localization.Localization(__file__, 389, 17), getitem___114869, int_114867)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 17), tuple_114866, subscript_call_result_114870)
    # Adding element type (line 389)
    
    # Obtaining the type of the subscript
    int_114871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 33), 'int')
    # Getting the type of 'shape' (line 389)
    shape_114872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 27), 'shape')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___114873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 27), shape_114872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_114874 = invoke(stypy.reporting.localization.Localization(__file__, 389, 27), getitem___114873, int_114871)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 17), tuple_114866, subscript_call_result_114874)
    
    # Assigning a type to the variable 'shape' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'shape', tuple_114866)
    # SSA join for if statement (line 387)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 384)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 381)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 390)
    # Getting the type of 'mode' (line 390)
    mode_114875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 7), 'mode')
    # Getting the type of 'None' (line 390)
    None_114876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'None')
    
    (may_be_114877, more_types_in_union_114878) = may_be_none(mode_114875, None_114876)

    if may_be_114877:

        if more_types_in_union_114878:
            # Runtime conditional SSA (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'numch' (line 391)
        numch_114879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 11), 'numch')
        int_114880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 20), 'int')
        # Applying the binary operator '==' (line 391)
        result_eq_114881 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 11), '==', numch_114879, int_114880)
        
        # Testing the type of an if condition (line 391)
        if_condition_114882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 8), result_eq_114881)
        # Assigning a type to the variable 'if_condition_114882' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'if_condition_114882', if_condition_114882)
        # SSA begins for if statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 392):
        
        # Assigning a Str to a Name (line 392):
        str_114883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 19), 'str', 'RGB')
        # Assigning a type to the variable 'mode' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'mode', str_114883)
        # SSA branch for the else part of an if statement (line 391)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 394):
        
        # Assigning a Str to a Name (line 394):
        str_114884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 19), 'str', 'RGBA')
        # Assigning a type to the variable 'mode' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'mode', str_114884)
        # SSA join for if statement (line 391)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_114878:
            # SSA join for if statement (line 390)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'mode' (line 396)
    mode_114885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 396)
    list_114886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 396)
    # Adding element type (line 396)
    str_114887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 20), 'str', 'RGB')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 19), list_114886, str_114887)
    # Adding element type (line 396)
    str_114888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 27), 'str', 'RGBA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 19), list_114886, str_114888)
    # Adding element type (line 396)
    str_114889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 35), 'str', 'YCbCr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 19), list_114886, str_114889)
    # Adding element type (line 396)
    str_114890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 44), 'str', 'CMYK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 19), list_114886, str_114890)
    
    # Applying the binary operator 'notin' (line 396)
    result_contains_114891 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 7), 'notin', mode_114885, list_114886)
    
    # Testing the type of an if condition (line 396)
    if_condition_114892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 4), result_contains_114891)
    # Assigning a type to the variable 'if_condition_114892' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'if_condition_114892', if_condition_114892)
    # SSA begins for if statement (line 396)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of '_errstr' (line 397)
    _errstr_114894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 25), '_errstr', False)
    # Processing the call keyword arguments (line 397)
    kwargs_114895 = {}
    # Getting the type of 'ValueError' (line 397)
    ValueError_114893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 397)
    ValueError_call_result_114896 = invoke(stypy.reporting.localization.Localization(__file__, 397, 14), ValueError_114893, *[_errstr_114894], **kwargs_114895)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 397, 8), ValueError_call_result_114896, 'raise parameter', BaseException)
    # SSA join for if statement (line 396)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 399)
    mode_114897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 399)
    list_114898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 399)
    # Adding element type (line 399)
    str_114899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 16), 'str', 'RGB')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 15), list_114898, str_114899)
    # Adding element type (line 399)
    str_114900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 23), 'str', 'YCbCr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 15), list_114898, str_114900)
    
    # Applying the binary operator 'in' (line 399)
    result_contains_114901 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 7), 'in', mode_114897, list_114898)
    
    # Testing the type of an if condition (line 399)
    if_condition_114902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 4), result_contains_114901)
    # Assigning a type to the variable 'if_condition_114902' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'if_condition_114902', if_condition_114902)
    # SSA begins for if statement (line 399)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'numch' (line 400)
    numch_114903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 11), 'numch')
    int_114904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 20), 'int')
    # Applying the binary operator '!=' (line 400)
    result_ne_114905 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 11), '!=', numch_114903, int_114904)
    
    # Testing the type of an if condition (line 400)
    if_condition_114906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 8), result_ne_114905)
    # Assigning a type to the variable 'if_condition_114906' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'if_condition_114906', if_condition_114906)
    # SSA begins for if statement (line 400)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 401)
    # Processing the call arguments (line 401)
    str_114908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 29), 'str', 'Invalid array shape for mode.')
    # Processing the call keyword arguments (line 401)
    kwargs_114909 = {}
    # Getting the type of 'ValueError' (line 401)
    ValueError_114907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 401)
    ValueError_call_result_114910 = invoke(stypy.reporting.localization.Localization(__file__, 401, 18), ValueError_114907, *[str_114908], **kwargs_114909)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 401, 12), ValueError_call_result_114910, 'raise parameter', BaseException)
    # SSA join for if statement (line 400)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 399)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 402)
    mode_114911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 402)
    list_114912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 402)
    # Adding element type (line 402)
    str_114913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 16), 'str', 'RGBA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 15), list_114912, str_114913)
    # Adding element type (line 402)
    str_114914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 24), 'str', 'CMYK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 15), list_114912, str_114914)
    
    # Applying the binary operator 'in' (line 402)
    result_contains_114915 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 7), 'in', mode_114911, list_114912)
    
    # Testing the type of an if condition (line 402)
    if_condition_114916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 4), result_contains_114915)
    # Assigning a type to the variable 'if_condition_114916' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'if_condition_114916', if_condition_114916)
    # SSA begins for if statement (line 402)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'numch' (line 403)
    numch_114917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'numch')
    int_114918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 20), 'int')
    # Applying the binary operator '!=' (line 403)
    result_ne_114919 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 11), '!=', numch_114917, int_114918)
    
    # Testing the type of an if condition (line 403)
    if_condition_114920 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 8), result_ne_114919)
    # Assigning a type to the variable 'if_condition_114920' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'if_condition_114920', if_condition_114920)
    # SSA begins for if statement (line 403)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 404)
    # Processing the call arguments (line 404)
    str_114922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 29), 'str', 'Invalid array shape for mode.')
    # Processing the call keyword arguments (line 404)
    kwargs_114923 = {}
    # Getting the type of 'ValueError' (line 404)
    ValueError_114921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 404)
    ValueError_call_result_114924 = invoke(stypy.reporting.localization.Localization(__file__, 404, 18), ValueError_114921, *[str_114922], **kwargs_114923)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 404, 12), ValueError_call_result_114924, 'raise parameter', BaseException)
    # SSA join for if statement (line 403)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 402)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 407):
    
    # Assigning a Call to a Name (line 407):
    
    # Call to frombytes(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'mode' (line 407)
    mode_114927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 28), 'mode', False)
    # Getting the type of 'shape' (line 407)
    shape_114928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 34), 'shape', False)
    # Getting the type of 'strdata' (line 407)
    strdata_114929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 41), 'strdata', False)
    # Processing the call keyword arguments (line 407)
    kwargs_114930 = {}
    # Getting the type of 'Image' (line 407)
    Image_114925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'Image', False)
    # Obtaining the member 'frombytes' of a type (line 407)
    frombytes_114926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), Image_114925, 'frombytes')
    # Calling frombytes(args, kwargs) (line 407)
    frombytes_call_result_114931 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), frombytes_114926, *[mode_114927, shape_114928, strdata_114929], **kwargs_114930)
    
    # Assigning a type to the variable 'image' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'image', frombytes_call_result_114931)
    # Getting the type of 'image' (line 408)
    image_114932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 11), 'image')
    # Assigning a type to the variable 'stypy_return_type' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'stypy_return_type', image_114932)
    
    # ################# End of 'toimage(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'toimage' in the type store
    # Getting the type of 'stypy_return_type' (line 285)
    stypy_return_type_114933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114933)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'toimage'
    return stypy_return_type_114933

# Assigning a type to the variable 'toimage' (line 285)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'toimage', toimage)

@norecursion
def imrotate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_114934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 32), 'str', 'bilinear')
    defaults = [str_114934]
    # Create a new context for function 'imrotate'
    module_type_store = module_type_store.open_function_context('imrotate', 411, 0, False)
    
    # Passed parameters checking function
    imrotate.stypy_localization = localization
    imrotate.stypy_type_of_self = None
    imrotate.stypy_type_store = module_type_store
    imrotate.stypy_function_name = 'imrotate'
    imrotate.stypy_param_names_list = ['arr', 'angle', 'interp']
    imrotate.stypy_varargs_param_name = None
    imrotate.stypy_kwargs_param_name = None
    imrotate.stypy_call_defaults = defaults
    imrotate.stypy_call_varargs = varargs
    imrotate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'imrotate', ['arr', 'angle', 'interp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'imrotate', localization, ['arr', 'angle', 'interp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'imrotate(...)' code ##################

    str_114935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, (-1)), 'str', "\n    Rotate an image counter-clockwise by angle degrees.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    .. warning::\n\n        This function uses `bytescale` under the hood to rescale images to use\n        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.\n        It will also cast data for 2-D images to ``uint32`` for ``mode=None``\n        (which is the default).\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of image to be rotated.\n    angle : float\n        The angle of rotation.\n    interp : str, optional\n        Interpolation\n\n        - 'nearest' :  for nearest neighbor\n        - 'bilinear' : for bilinear\n        - 'lanczos' : for lanczos\n        - 'cubic' : for bicubic\n        - 'bicubic' : for bicubic\n\n    Returns\n    -------\n    imrotate : ndarray\n        The rotated array of image.\n\n    ")
    
    # Assigning a Call to a Name (line 448):
    
    # Assigning a Call to a Name (line 448):
    
    # Call to asarray(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'arr' (line 448)
    arr_114937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 18), 'arr', False)
    # Processing the call keyword arguments (line 448)
    kwargs_114938 = {}
    # Getting the type of 'asarray' (line 448)
    asarray_114936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 448)
    asarray_call_result_114939 = invoke(stypy.reporting.localization.Localization(__file__, 448, 10), asarray_114936, *[arr_114937], **kwargs_114938)
    
    # Assigning a type to the variable 'arr' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'arr', asarray_call_result_114939)
    
    # Assigning a Dict to a Name (line 449):
    
    # Assigning a Dict to a Name (line 449):
    
    # Obtaining an instance of the builtin type 'dict' (line 449)
    dict_114940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 449)
    # Adding element type (key, value) (line 449)
    str_114941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 12), 'str', 'nearest')
    int_114942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 23), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 11), dict_114940, (str_114941, int_114942))
    # Adding element type (key, value) (line 449)
    str_114943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 26), 'str', 'lanczos')
    int_114944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 37), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 11), dict_114940, (str_114943, int_114944))
    # Adding element type (key, value) (line 449)
    str_114945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 40), 'str', 'bilinear')
    int_114946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 52), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 11), dict_114940, (str_114945, int_114946))
    # Adding element type (key, value) (line 449)
    str_114947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 55), 'str', 'bicubic')
    int_114948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 66), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 11), dict_114940, (str_114947, int_114948))
    # Adding element type (key, value) (line 449)
    str_114949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 69), 'str', 'cubic')
    int_114950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 78), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 11), dict_114940, (str_114949, int_114950))
    
    # Assigning a type to the variable 'func' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'func', dict_114940)
    
    # Assigning a Call to a Name (line 450):
    
    # Assigning a Call to a Name (line 450):
    
    # Call to toimage(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 'arr' (line 450)
    arr_114952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 17), 'arr', False)
    # Processing the call keyword arguments (line 450)
    kwargs_114953 = {}
    # Getting the type of 'toimage' (line 450)
    toimage_114951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 9), 'toimage', False)
    # Calling toimage(args, kwargs) (line 450)
    toimage_call_result_114954 = invoke(stypy.reporting.localization.Localization(__file__, 450, 9), toimage_114951, *[arr_114952], **kwargs_114953)
    
    # Assigning a type to the variable 'im' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'im', toimage_call_result_114954)
    
    # Assigning a Call to a Name (line 451):
    
    # Assigning a Call to a Name (line 451):
    
    # Call to rotate(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'angle' (line 451)
    angle_114957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 19), 'angle', False)
    # Processing the call keyword arguments (line 451)
    
    # Obtaining the type of the subscript
    # Getting the type of 'interp' (line 451)
    interp_114958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 40), 'interp', False)
    # Getting the type of 'func' (line 451)
    func_114959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 35), 'func', False)
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___114960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 35), func_114959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_114961 = invoke(stypy.reporting.localization.Localization(__file__, 451, 35), getitem___114960, interp_114958)
    
    keyword_114962 = subscript_call_result_114961
    kwargs_114963 = {'resample': keyword_114962}
    # Getting the type of 'im' (line 451)
    im_114955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 9), 'im', False)
    # Obtaining the member 'rotate' of a type (line 451)
    rotate_114956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 9), im_114955, 'rotate')
    # Calling rotate(args, kwargs) (line 451)
    rotate_call_result_114964 = invoke(stypy.reporting.localization.Localization(__file__, 451, 9), rotate_114956, *[angle_114957], **kwargs_114963)
    
    # Assigning a type to the variable 'im' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'im', rotate_call_result_114964)
    
    # Call to fromimage(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'im' (line 452)
    im_114966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 21), 'im', False)
    # Processing the call keyword arguments (line 452)
    kwargs_114967 = {}
    # Getting the type of 'fromimage' (line 452)
    fromimage_114965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 11), 'fromimage', False)
    # Calling fromimage(args, kwargs) (line 452)
    fromimage_call_result_114968 = invoke(stypy.reporting.localization.Localization(__file__, 452, 11), fromimage_114965, *[im_114966], **kwargs_114967)
    
    # Assigning a type to the variable 'stypy_return_type' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type', fromimage_call_result_114968)
    
    # ################# End of 'imrotate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'imrotate' in the type store
    # Getting the type of 'stypy_return_type' (line 411)
    stypy_return_type_114969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114969)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'imrotate'
    return stypy_return_type_114969

# Assigning a type to the variable 'imrotate' (line 411)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 0), 'imrotate', imrotate)

@norecursion
def imshow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'imshow'
    module_type_store = module_type_store.open_function_context('imshow', 455, 0, False)
    
    # Passed parameters checking function
    imshow.stypy_localization = localization
    imshow.stypy_type_of_self = None
    imshow.stypy_type_store = module_type_store
    imshow.stypy_function_name = 'imshow'
    imshow.stypy_param_names_list = ['arr']
    imshow.stypy_varargs_param_name = None
    imshow.stypy_kwargs_param_name = None
    imshow.stypy_call_defaults = defaults
    imshow.stypy_call_varargs = varargs
    imshow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'imshow', ['arr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'imshow', localization, ['arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'imshow(...)' code ##################

    str_114970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, (-1)), 'str', "\n    Simple showing of an image through an external viewer.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    Uses the image viewer specified by the environment variable\n    SCIPY_PIL_IMAGE_VIEWER, or if that is not defined then `see`,\n    to view a temporary file generated from array data.\n\n    .. warning::\n\n        This function uses `bytescale` under the hood to rescale images to use\n        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.\n        It will also cast data for 2-D images to ``uint32`` for ``mode=None``\n        (which is the default).\n\n    Parameters\n    ----------\n    arr : ndarray\n        Array of image data to show.\n\n    Returns\n    -------\n    None\n\n    Examples\n    --------\n    >>> a = np.tile(np.arange(255), (255,1))\n    >>> from scipy import misc\n    >>> misc.imshow(a)\n\n    ")
    
    # Assigning a Call to a Name (line 491):
    
    # Assigning a Call to a Name (line 491):
    
    # Call to toimage(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'arr' (line 491)
    arr_114972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 17), 'arr', False)
    # Processing the call keyword arguments (line 491)
    kwargs_114973 = {}
    # Getting the type of 'toimage' (line 491)
    toimage_114971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 9), 'toimage', False)
    # Calling toimage(args, kwargs) (line 491)
    toimage_call_result_114974 = invoke(stypy.reporting.localization.Localization(__file__, 491, 9), toimage_114971, *[arr_114972], **kwargs_114973)
    
    # Assigning a type to the variable 'im' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'im', toimage_call_result_114974)
    
    # Assigning a Call to a Tuple (line 492):
    
    # Assigning a Subscript to a Name (line 492):
    
    # Obtaining the type of the subscript
    int_114975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 4), 'int')
    
    # Call to mkstemp(...): (line 492)
    # Processing the call arguments (line 492)
    str_114978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 35), 'str', '.png')
    # Processing the call keyword arguments (line 492)
    kwargs_114979 = {}
    # Getting the type of 'tempfile' (line 492)
    tempfile_114976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 18), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 492)
    mkstemp_114977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 18), tempfile_114976, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 492)
    mkstemp_call_result_114980 = invoke(stypy.reporting.localization.Localization(__file__, 492, 18), mkstemp_114977, *[str_114978], **kwargs_114979)
    
    # Obtaining the member '__getitem__' of a type (line 492)
    getitem___114981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 4), mkstemp_call_result_114980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 492)
    subscript_call_result_114982 = invoke(stypy.reporting.localization.Localization(__file__, 492, 4), getitem___114981, int_114975)
    
    # Assigning a type to the variable 'tuple_var_assignment_114242' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'tuple_var_assignment_114242', subscript_call_result_114982)
    
    # Assigning a Subscript to a Name (line 492):
    
    # Obtaining the type of the subscript
    int_114983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 4), 'int')
    
    # Call to mkstemp(...): (line 492)
    # Processing the call arguments (line 492)
    str_114986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 35), 'str', '.png')
    # Processing the call keyword arguments (line 492)
    kwargs_114987 = {}
    # Getting the type of 'tempfile' (line 492)
    tempfile_114984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 18), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 492)
    mkstemp_114985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 18), tempfile_114984, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 492)
    mkstemp_call_result_114988 = invoke(stypy.reporting.localization.Localization(__file__, 492, 18), mkstemp_114985, *[str_114986], **kwargs_114987)
    
    # Obtaining the member '__getitem__' of a type (line 492)
    getitem___114989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 4), mkstemp_call_result_114988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 492)
    subscript_call_result_114990 = invoke(stypy.reporting.localization.Localization(__file__, 492, 4), getitem___114989, int_114983)
    
    # Assigning a type to the variable 'tuple_var_assignment_114243' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'tuple_var_assignment_114243', subscript_call_result_114990)
    
    # Assigning a Name to a Name (line 492):
    # Getting the type of 'tuple_var_assignment_114242' (line 492)
    tuple_var_assignment_114242_114991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'tuple_var_assignment_114242')
    # Assigning a type to the variable 'fnum' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'fnum', tuple_var_assignment_114242_114991)
    
    # Assigning a Name to a Name (line 492):
    # Getting the type of 'tuple_var_assignment_114243' (line 492)
    tuple_var_assignment_114243_114992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'tuple_var_assignment_114243')
    # Assigning a type to the variable 'fname' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 10), 'fname', tuple_var_assignment_114243_114992)
    
    
    # SSA begins for try-except statement (line 493)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to save(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'fname' (line 494)
    fname_114995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 16), 'fname', False)
    # Processing the call keyword arguments (line 494)
    kwargs_114996 = {}
    # Getting the type of 'im' (line 494)
    im_114993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'im', False)
    # Obtaining the member 'save' of a type (line 494)
    save_114994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), im_114993, 'save')
    # Calling save(args, kwargs) (line 494)
    save_call_result_114997 = invoke(stypy.reporting.localization.Localization(__file__, 494, 8), save_114994, *[fname_114995], **kwargs_114996)
    
    # SSA branch for the except part of a try statement (line 493)
    # SSA branch for the except '<any exception>' branch of a try statement (line 493)
    module_type_store.open_ssa_branch('except')
    
    # Call to RuntimeError(...): (line 496)
    # Processing the call arguments (line 496)
    str_114999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 27), 'str', 'Error saving temporary image data.')
    # Processing the call keyword arguments (line 496)
    kwargs_115000 = {}
    # Getting the type of 'RuntimeError' (line 496)
    RuntimeError_114998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 496)
    RuntimeError_call_result_115001 = invoke(stypy.reporting.localization.Localization(__file__, 496, 14), RuntimeError_114998, *[str_114999], **kwargs_115000)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 496, 8), RuntimeError_call_result_115001, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 493)
    module_type_store = module_type_store.join_ssa_context()
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 498, 4))
    
    # 'import os' statement (line 498)
    import os

    import_module(stypy.reporting.localization.Localization(__file__, 498, 4), 'os', os, module_type_store)
    
    
    # Call to close(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'fnum' (line 499)
    fnum_115004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'fnum', False)
    # Processing the call keyword arguments (line 499)
    kwargs_115005 = {}
    # Getting the type of 'os' (line 499)
    os_115002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'os', False)
    # Obtaining the member 'close' of a type (line 499)
    close_115003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 4), os_115002, 'close')
    # Calling close(args, kwargs) (line 499)
    close_call_result_115006 = invoke(stypy.reporting.localization.Localization(__file__, 499, 4), close_115003, *[fnum_115004], **kwargs_115005)
    
    
    # Assigning a Call to a Name (line 501):
    
    # Assigning a Call to a Name (line 501):
    
    # Call to get(...): (line 501)
    # Processing the call arguments (line 501)
    str_115010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 25), 'str', 'SCIPY_PIL_IMAGE_VIEWER')
    str_115011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 51), 'str', 'see')
    # Processing the call keyword arguments (line 501)
    kwargs_115012 = {}
    # Getting the type of 'os' (line 501)
    os_115007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 10), 'os', False)
    # Obtaining the member 'environ' of a type (line 501)
    environ_115008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 10), os_115007, 'environ')
    # Obtaining the member 'get' of a type (line 501)
    get_115009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 10), environ_115008, 'get')
    # Calling get(args, kwargs) (line 501)
    get_call_result_115013 = invoke(stypy.reporting.localization.Localization(__file__, 501, 10), get_115009, *[str_115010, str_115011], **kwargs_115012)
    
    # Assigning a type to the variable 'cmd' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'cmd', get_call_result_115013)
    
    # Assigning a Call to a Name (line 502):
    
    # Assigning a Call to a Name (line 502):
    
    # Call to system(...): (line 502)
    # Processing the call arguments (line 502)
    str_115016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 23), 'str', '%s %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 502)
    tuple_115017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 502)
    # Adding element type (line 502)
    # Getting the type of 'cmd' (line 502)
    cmd_115018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 34), 'cmd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 34), tuple_115017, cmd_115018)
    # Adding element type (line 502)
    # Getting the type of 'fname' (line 502)
    fname_115019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 39), 'fname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 34), tuple_115017, fname_115019)
    
    # Applying the binary operator '%' (line 502)
    result_mod_115020 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 23), '%', str_115016, tuple_115017)
    
    # Processing the call keyword arguments (line 502)
    kwargs_115021 = {}
    # Getting the type of 'os' (line 502)
    os_115014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 13), 'os', False)
    # Obtaining the member 'system' of a type (line 502)
    system_115015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 13), os_115014, 'system')
    # Calling system(args, kwargs) (line 502)
    system_call_result_115022 = invoke(stypy.reporting.localization.Localization(__file__, 502, 13), system_115015, *[result_mod_115020], **kwargs_115021)
    
    # Assigning a type to the variable 'status' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'status', system_call_result_115022)
    
    # Call to unlink(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'fname' (line 504)
    fname_115025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 14), 'fname', False)
    # Processing the call keyword arguments (line 504)
    kwargs_115026 = {}
    # Getting the type of 'os' (line 504)
    os_115023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'os', False)
    # Obtaining the member 'unlink' of a type (line 504)
    unlink_115024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 4), os_115023, 'unlink')
    # Calling unlink(args, kwargs) (line 504)
    unlink_call_result_115027 = invoke(stypy.reporting.localization.Localization(__file__, 504, 4), unlink_115024, *[fname_115025], **kwargs_115026)
    
    
    
    # Getting the type of 'status' (line 505)
    status_115028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 7), 'status')
    int_115029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 17), 'int')
    # Applying the binary operator '!=' (line 505)
    result_ne_115030 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), '!=', status_115028, int_115029)
    
    # Testing the type of an if condition (line 505)
    if_condition_115031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 4), result_ne_115030)
    # Assigning a type to the variable 'if_condition_115031' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'if_condition_115031', if_condition_115031)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 506)
    # Processing the call arguments (line 506)
    str_115033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 27), 'str', 'Could not execute image viewer.')
    # Processing the call keyword arguments (line 506)
    kwargs_115034 = {}
    # Getting the type of 'RuntimeError' (line 506)
    RuntimeError_115032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 506)
    RuntimeError_call_result_115035 = invoke(stypy.reporting.localization.Localization(__file__, 506, 14), RuntimeError_115032, *[str_115033], **kwargs_115034)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 506, 8), RuntimeError_call_result_115035, 'raise parameter', BaseException)
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'imshow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'imshow' in the type store
    # Getting the type of 'stypy_return_type' (line 455)
    stypy_return_type_115036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115036)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'imshow'
    return stypy_return_type_115036

# Assigning a type to the variable 'imshow' (line 455)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), 'imshow', imshow)

@norecursion
def imresize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_115037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 31), 'str', 'bilinear')
    # Getting the type of 'None' (line 512)
    None_115038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 48), 'None')
    defaults = [str_115037, None_115038]
    # Create a new context for function 'imresize'
    module_type_store = module_type_store.open_function_context('imresize', 509, 0, False)
    
    # Passed parameters checking function
    imresize.stypy_localization = localization
    imresize.stypy_type_of_self = None
    imresize.stypy_type_store = module_type_store
    imresize.stypy_function_name = 'imresize'
    imresize.stypy_param_names_list = ['arr', 'size', 'interp', 'mode']
    imresize.stypy_varargs_param_name = None
    imresize.stypy_kwargs_param_name = None
    imresize.stypy_call_defaults = defaults
    imresize.stypy_call_varargs = varargs
    imresize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'imresize', ['arr', 'size', 'interp', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'imresize', localization, ['arr', 'size', 'interp', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'imresize(...)' code ##################

    str_115039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, (-1)), 'str', "\n    Resize an image.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    .. warning::\n\n        This function uses `bytescale` under the hood to rescale images to use\n        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.\n        It will also cast data for 2-D images to ``uint32`` for ``mode=None``\n        (which is the default).\n\n    Parameters\n    ----------\n    arr : ndarray\n        The array of image to be resized.\n    size : int, float or tuple\n        * int   - Percentage of current size.\n        * float - Fraction of current size.\n        * tuple - Size of the output image (height, width).\n\n    interp : str, optional\n        Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',\n        'bicubic' or 'cubic').\n    mode : str, optional\n        The PIL image mode ('P', 'L', etc.) to convert `arr` before resizing.\n        If ``mode=None`` (the default), 2-D images will be treated like\n        ``mode='L'``, i.e. casting to long integer.  For 3-D and 4-D arrays,\n        `mode` will be set to ``'RGB'`` and ``'RGBA'`` respectively.\n\n    Returns\n    -------\n    imresize : ndarray\n        The resized array of image.\n\n    See Also\n    --------\n    toimage : Implicitly used to convert `arr` according to `mode`.\n    scipy.ndimage.zoom : More generic implementation that does not use PIL.\n\n    ")
    
    # Assigning a Call to a Name (line 554):
    
    # Assigning a Call to a Name (line 554):
    
    # Call to toimage(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'arr' (line 554)
    arr_115041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 17), 'arr', False)
    # Processing the call keyword arguments (line 554)
    # Getting the type of 'mode' (line 554)
    mode_115042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 27), 'mode', False)
    keyword_115043 = mode_115042
    kwargs_115044 = {'mode': keyword_115043}
    # Getting the type of 'toimage' (line 554)
    toimage_115040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 9), 'toimage', False)
    # Calling toimage(args, kwargs) (line 554)
    toimage_call_result_115045 = invoke(stypy.reporting.localization.Localization(__file__, 554, 9), toimage_115040, *[arr_115041], **kwargs_115044)
    
    # Assigning a type to the variable 'im' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'im', toimage_call_result_115045)
    
    # Assigning a Call to a Name (line 555):
    
    # Assigning a Call to a Name (line 555):
    
    # Call to type(...): (line 555)
    # Processing the call arguments (line 555)
    # Getting the type of 'size' (line 555)
    size_115047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 14), 'size', False)
    # Processing the call keyword arguments (line 555)
    kwargs_115048 = {}
    # Getting the type of 'type' (line 555)
    type_115046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 9), 'type', False)
    # Calling type(args, kwargs) (line 555)
    type_call_result_115049 = invoke(stypy.reporting.localization.Localization(__file__, 555, 9), type_115046, *[size_115047], **kwargs_115048)
    
    # Assigning a type to the variable 'ts' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'ts', type_call_result_115049)
    
    
    # Call to issubdtype(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'ts' (line 556)
    ts_115051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 18), 'ts', False)
    # Getting the type of 'numpy' (line 556)
    numpy_115052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 22), 'numpy', False)
    # Obtaining the member 'signedinteger' of a type (line 556)
    signedinteger_115053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 22), numpy_115052, 'signedinteger')
    # Processing the call keyword arguments (line 556)
    kwargs_115054 = {}
    # Getting the type of 'issubdtype' (line 556)
    issubdtype_115050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 7), 'issubdtype', False)
    # Calling issubdtype(args, kwargs) (line 556)
    issubdtype_call_result_115055 = invoke(stypy.reporting.localization.Localization(__file__, 556, 7), issubdtype_115050, *[ts_115051, signedinteger_115053], **kwargs_115054)
    
    # Testing the type of an if condition (line 556)
    if_condition_115056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 4), issubdtype_call_result_115055)
    # Assigning a type to the variable 'if_condition_115056' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'if_condition_115056', if_condition_115056)
    # SSA begins for if statement (line 556)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 557):
    
    # Assigning a BinOp to a Name (line 557):
    # Getting the type of 'size' (line 557)
    size_115057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 18), 'size')
    float_115058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 25), 'float')
    # Applying the binary operator 'div' (line 557)
    result_div_115059 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 18), 'div', size_115057, float_115058)
    
    # Assigning a type to the variable 'percent' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'percent', result_div_115059)
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to tuple(...): (line 558)
    # Processing the call arguments (line 558)
    
    # Call to astype(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'int' (line 558)
    int_115069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 53), 'int', False)
    # Processing the call keyword arguments (line 558)
    kwargs_115070 = {}
    
    # Call to array(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'im' (line 558)
    im_115062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 28), 'im', False)
    # Obtaining the member 'size' of a type (line 558)
    size_115063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 28), im_115062, 'size')
    # Processing the call keyword arguments (line 558)
    kwargs_115064 = {}
    # Getting the type of 'array' (line 558)
    array_115061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 22), 'array', False)
    # Calling array(args, kwargs) (line 558)
    array_call_result_115065 = invoke(stypy.reporting.localization.Localization(__file__, 558, 22), array_115061, *[size_115063], **kwargs_115064)
    
    # Getting the type of 'percent' (line 558)
    percent_115066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 37), 'percent', False)
    # Applying the binary operator '*' (line 558)
    result_mul_115067 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 22), '*', array_call_result_115065, percent_115066)
    
    # Obtaining the member 'astype' of a type (line 558)
    astype_115068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 22), result_mul_115067, 'astype')
    # Calling astype(args, kwargs) (line 558)
    astype_call_result_115071 = invoke(stypy.reporting.localization.Localization(__file__, 558, 22), astype_115068, *[int_115069], **kwargs_115070)
    
    # Processing the call keyword arguments (line 558)
    kwargs_115072 = {}
    # Getting the type of 'tuple' (line 558)
    tuple_115060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 558)
    tuple_call_result_115073 = invoke(stypy.reporting.localization.Localization(__file__, 558, 15), tuple_115060, *[astype_call_result_115071], **kwargs_115072)
    
    # Assigning a type to the variable 'size' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'size', tuple_call_result_115073)
    # SSA branch for the else part of an if statement (line 556)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to issubdtype(...): (line 559)
    # Processing the call arguments (line 559)
    
    # Call to type(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'size' (line 559)
    size_115076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 25), 'size', False)
    # Processing the call keyword arguments (line 559)
    kwargs_115077 = {}
    # Getting the type of 'type' (line 559)
    type_115075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 'type', False)
    # Calling type(args, kwargs) (line 559)
    type_call_result_115078 = invoke(stypy.reporting.localization.Localization(__file__, 559, 20), type_115075, *[size_115076], **kwargs_115077)
    
    # Getting the type of 'numpy' (line 559)
    numpy_115079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 32), 'numpy', False)
    # Obtaining the member 'floating' of a type (line 559)
    floating_115080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 32), numpy_115079, 'floating')
    # Processing the call keyword arguments (line 559)
    kwargs_115081 = {}
    # Getting the type of 'issubdtype' (line 559)
    issubdtype_115074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 9), 'issubdtype', False)
    # Calling issubdtype(args, kwargs) (line 559)
    issubdtype_call_result_115082 = invoke(stypy.reporting.localization.Localization(__file__, 559, 9), issubdtype_115074, *[type_call_result_115078, floating_115080], **kwargs_115081)
    
    # Testing the type of an if condition (line 559)
    if_condition_115083 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 9), issubdtype_call_result_115082)
    # Assigning a type to the variable 'if_condition_115083' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 9), 'if_condition_115083', if_condition_115083)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 560):
    
    # Assigning a Call to a Name (line 560):
    
    # Call to tuple(...): (line 560)
    # Processing the call arguments (line 560)
    
    # Call to astype(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'int' (line 560)
    int_115093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 50), 'int', False)
    # Processing the call keyword arguments (line 560)
    kwargs_115094 = {}
    
    # Call to array(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'im' (line 560)
    im_115086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 28), 'im', False)
    # Obtaining the member 'size' of a type (line 560)
    size_115087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 28), im_115086, 'size')
    # Processing the call keyword arguments (line 560)
    kwargs_115088 = {}
    # Getting the type of 'array' (line 560)
    array_115085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 22), 'array', False)
    # Calling array(args, kwargs) (line 560)
    array_call_result_115089 = invoke(stypy.reporting.localization.Localization(__file__, 560, 22), array_115085, *[size_115087], **kwargs_115088)
    
    # Getting the type of 'size' (line 560)
    size_115090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 37), 'size', False)
    # Applying the binary operator '*' (line 560)
    result_mul_115091 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 22), '*', array_call_result_115089, size_115090)
    
    # Obtaining the member 'astype' of a type (line 560)
    astype_115092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 22), result_mul_115091, 'astype')
    # Calling astype(args, kwargs) (line 560)
    astype_call_result_115095 = invoke(stypy.reporting.localization.Localization(__file__, 560, 22), astype_115092, *[int_115093], **kwargs_115094)
    
    # Processing the call keyword arguments (line 560)
    kwargs_115096 = {}
    # Getting the type of 'tuple' (line 560)
    tuple_115084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 560)
    tuple_call_result_115097 = invoke(stypy.reporting.localization.Localization(__file__, 560, 15), tuple_115084, *[astype_call_result_115095], **kwargs_115096)
    
    # Assigning a type to the variable 'size' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'size', tuple_call_result_115097)
    # SSA branch for the else part of an if statement (line 559)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 562):
    
    # Assigning a Tuple to a Name (line 562):
    
    # Obtaining an instance of the builtin type 'tuple' (line 562)
    tuple_115098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 562)
    # Adding element type (line 562)
    
    # Obtaining the type of the subscript
    int_115099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 21), 'int')
    # Getting the type of 'size' (line 562)
    size_115100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'size')
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___115101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 16), size_115100, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_115102 = invoke(stypy.reporting.localization.Localization(__file__, 562, 16), getitem___115101, int_115099)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 16), tuple_115098, subscript_call_result_115102)
    # Adding element type (line 562)
    
    # Obtaining the type of the subscript
    int_115103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 30), 'int')
    # Getting the type of 'size' (line 562)
    size_115104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 25), 'size')
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___115105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 25), size_115104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_115106 = invoke(stypy.reporting.localization.Localization(__file__, 562, 25), getitem___115105, int_115103)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 16), tuple_115098, subscript_call_result_115106)
    
    # Assigning a type to the variable 'size' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'size', tuple_115098)
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 556)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 563):
    
    # Assigning a Dict to a Name (line 563):
    
    # Obtaining an instance of the builtin type 'dict' (line 563)
    dict_115107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 563)
    # Adding element type (key, value) (line 563)
    str_115108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 12), 'str', 'nearest')
    int_115109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 23), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 11), dict_115107, (str_115108, int_115109))
    # Adding element type (key, value) (line 563)
    str_115110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 26), 'str', 'lanczos')
    int_115111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 37), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 11), dict_115107, (str_115110, int_115111))
    # Adding element type (key, value) (line 563)
    str_115112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 40), 'str', 'bilinear')
    int_115113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 52), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 11), dict_115107, (str_115112, int_115113))
    # Adding element type (key, value) (line 563)
    str_115114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 55), 'str', 'bicubic')
    int_115115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 66), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 11), dict_115107, (str_115114, int_115115))
    # Adding element type (key, value) (line 563)
    str_115116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 69), 'str', 'cubic')
    int_115117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 78), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 11), dict_115107, (str_115116, int_115117))
    
    # Assigning a type to the variable 'func' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'func', dict_115107)
    
    # Assigning a Call to a Name (line 564):
    
    # Assigning a Call to a Name (line 564):
    
    # Call to resize(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'size' (line 564)
    size_115120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 22), 'size', False)
    # Processing the call keyword arguments (line 564)
    
    # Obtaining the type of the subscript
    # Getting the type of 'interp' (line 564)
    interp_115121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 42), 'interp', False)
    # Getting the type of 'func' (line 564)
    func_115122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 37), 'func', False)
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___115123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 37), func_115122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_115124 = invoke(stypy.reporting.localization.Localization(__file__, 564, 37), getitem___115123, interp_115121)
    
    keyword_115125 = subscript_call_result_115124
    kwargs_115126 = {'resample': keyword_115125}
    # Getting the type of 'im' (line 564)
    im_115118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'im', False)
    # Obtaining the member 'resize' of a type (line 564)
    resize_115119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 12), im_115118, 'resize')
    # Calling resize(args, kwargs) (line 564)
    resize_call_result_115127 = invoke(stypy.reporting.localization.Localization(__file__, 564, 12), resize_115119, *[size_115120], **kwargs_115126)
    
    # Assigning a type to the variable 'imnew' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'imnew', resize_call_result_115127)
    
    # Call to fromimage(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'imnew' (line 565)
    imnew_115129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 21), 'imnew', False)
    # Processing the call keyword arguments (line 565)
    kwargs_115130 = {}
    # Getting the type of 'fromimage' (line 565)
    fromimage_115128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'fromimage', False)
    # Calling fromimage(args, kwargs) (line 565)
    fromimage_call_result_115131 = invoke(stypy.reporting.localization.Localization(__file__, 565, 11), fromimage_115128, *[imnew_115129], **kwargs_115130)
    
    # Assigning a type to the variable 'stypy_return_type' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'stypy_return_type', fromimage_call_result_115131)
    
    # ################# End of 'imresize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'imresize' in the type store
    # Getting the type of 'stypy_return_type' (line 509)
    stypy_return_type_115132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115132)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'imresize'
    return stypy_return_type_115132

# Assigning a type to the variable 'imresize' (line 509)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'imresize', imresize)

@norecursion
def imfilter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'imfilter'
    module_type_store = module_type_store.open_function_context('imfilter', 568, 0, False)
    
    # Passed parameters checking function
    imfilter.stypy_localization = localization
    imfilter.stypy_type_of_self = None
    imfilter.stypy_type_store = module_type_store
    imfilter.stypy_function_name = 'imfilter'
    imfilter.stypy_param_names_list = ['arr', 'ftype']
    imfilter.stypy_varargs_param_name = None
    imfilter.stypy_kwargs_param_name = None
    imfilter.stypy_call_defaults = defaults
    imfilter.stypy_call_varargs = varargs
    imfilter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'imfilter', ['arr', 'ftype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'imfilter', localization, ['arr', 'ftype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'imfilter(...)' code ##################

    str_115133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, (-1)), 'str', "\n    Simple filtering of an image.\n\n    This function is only available if Python Imaging Library (PIL) is installed.\n\n    .. warning::\n\n        This function uses `bytescale` under the hood to rescale images to use\n        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.\n        It will also cast data for 2-D images to ``uint32`` for ``mode=None``\n        (which is the default).\n\n    Parameters\n    ----------\n    arr : ndarray\n        The array of Image in which the filter is to be applied.\n    ftype : str\n        The filter that has to be applied. Legal values are:\n        'blur', 'contour', 'detail', 'edge_enhance', 'edge_enhance_more',\n        'emboss', 'find_edges', 'smooth', 'smooth_more', 'sharpen'.\n\n    Returns\n    -------\n    imfilter : ndarray\n        The array with filter applied.\n\n    Raises\n    ------\n    ValueError\n        *Unknown filter type.*  If the filter you are trying\n        to apply is unsupported.\n\n    ")
    
    # Assigning a Dict to a Name (line 605):
    
    # Assigning a Dict to a Name (line 605):
    
    # Obtaining an instance of the builtin type 'dict' (line 605)
    dict_115134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 605)
    # Adding element type (key, value) (line 605)
    str_115135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 14), 'str', 'blur')
    # Getting the type of 'ImageFilter' (line 605)
    ImageFilter_115136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 22), 'ImageFilter')
    # Obtaining the member 'BLUR' of a type (line 605)
    BLUR_115137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 22), ImageFilter_115136, 'BLUR')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115135, BLUR_115137))
    # Adding element type (key, value) (line 605)
    str_115138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 14), 'str', 'contour')
    # Getting the type of 'ImageFilter' (line 606)
    ImageFilter_115139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 25), 'ImageFilter')
    # Obtaining the member 'CONTOUR' of a type (line 606)
    CONTOUR_115140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 25), ImageFilter_115139, 'CONTOUR')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115138, CONTOUR_115140))
    # Adding element type (key, value) (line 605)
    str_115141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 14), 'str', 'detail')
    # Getting the type of 'ImageFilter' (line 607)
    ImageFilter_115142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 24), 'ImageFilter')
    # Obtaining the member 'DETAIL' of a type (line 607)
    DETAIL_115143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 24), ImageFilter_115142, 'DETAIL')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115141, DETAIL_115143))
    # Adding element type (key, value) (line 605)
    str_115144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 14), 'str', 'edge_enhance')
    # Getting the type of 'ImageFilter' (line 608)
    ImageFilter_115145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 30), 'ImageFilter')
    # Obtaining the member 'EDGE_ENHANCE' of a type (line 608)
    EDGE_ENHANCE_115146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 30), ImageFilter_115145, 'EDGE_ENHANCE')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115144, EDGE_ENHANCE_115146))
    # Adding element type (key, value) (line 605)
    str_115147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 14), 'str', 'edge_enhance_more')
    # Getting the type of 'ImageFilter' (line 609)
    ImageFilter_115148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 35), 'ImageFilter')
    # Obtaining the member 'EDGE_ENHANCE_MORE' of a type (line 609)
    EDGE_ENHANCE_MORE_115149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 35), ImageFilter_115148, 'EDGE_ENHANCE_MORE')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115147, EDGE_ENHANCE_MORE_115149))
    # Adding element type (key, value) (line 605)
    str_115150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 14), 'str', 'emboss')
    # Getting the type of 'ImageFilter' (line 610)
    ImageFilter_115151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 24), 'ImageFilter')
    # Obtaining the member 'EMBOSS' of a type (line 610)
    EMBOSS_115152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 24), ImageFilter_115151, 'EMBOSS')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115150, EMBOSS_115152))
    # Adding element type (key, value) (line 605)
    str_115153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 14), 'str', 'find_edges')
    # Getting the type of 'ImageFilter' (line 611)
    ImageFilter_115154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 28), 'ImageFilter')
    # Obtaining the member 'FIND_EDGES' of a type (line 611)
    FIND_EDGES_115155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 28), ImageFilter_115154, 'FIND_EDGES')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115153, FIND_EDGES_115155))
    # Adding element type (key, value) (line 605)
    str_115156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 14), 'str', 'smooth')
    # Getting the type of 'ImageFilter' (line 612)
    ImageFilter_115157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 24), 'ImageFilter')
    # Obtaining the member 'SMOOTH' of a type (line 612)
    SMOOTH_115158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 24), ImageFilter_115157, 'SMOOTH')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115156, SMOOTH_115158))
    # Adding element type (key, value) (line 605)
    str_115159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 14), 'str', 'smooth_more')
    # Getting the type of 'ImageFilter' (line 613)
    ImageFilter_115160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 29), 'ImageFilter')
    # Obtaining the member 'SMOOTH_MORE' of a type (line 613)
    SMOOTH_MORE_115161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 29), ImageFilter_115160, 'SMOOTH_MORE')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115159, SMOOTH_MORE_115161))
    # Adding element type (key, value) (line 605)
    str_115162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 14), 'str', 'sharpen')
    # Getting the type of 'ImageFilter' (line 614)
    ImageFilter_115163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 25), 'ImageFilter')
    # Obtaining the member 'SHARPEN' of a type (line 614)
    SHARPEN_115164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 25), ImageFilter_115163, 'SHARPEN')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 13), dict_115134, (str_115162, SHARPEN_115164))
    
    # Assigning a type to the variable '_tdict' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), '_tdict', dict_115134)
    
    # Assigning a Call to a Name (line 617):
    
    # Assigning a Call to a Name (line 617):
    
    # Call to toimage(...): (line 617)
    # Processing the call arguments (line 617)
    # Getting the type of 'arr' (line 617)
    arr_115166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 17), 'arr', False)
    # Processing the call keyword arguments (line 617)
    kwargs_115167 = {}
    # Getting the type of 'toimage' (line 617)
    toimage_115165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 9), 'toimage', False)
    # Calling toimage(args, kwargs) (line 617)
    toimage_call_result_115168 = invoke(stypy.reporting.localization.Localization(__file__, 617, 9), toimage_115165, *[arr_115166], **kwargs_115167)
    
    # Assigning a type to the variable 'im' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'im', toimage_call_result_115168)
    
    
    # Getting the type of 'ftype' (line 618)
    ftype_115169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 7), 'ftype')
    # Getting the type of '_tdict' (line 618)
    _tdict_115170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 20), '_tdict')
    # Applying the binary operator 'notin' (line 618)
    result_contains_115171 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 7), 'notin', ftype_115169, _tdict_115170)
    
    # Testing the type of an if condition (line 618)
    if_condition_115172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 618, 4), result_contains_115171)
    # Assigning a type to the variable 'if_condition_115172' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'if_condition_115172', if_condition_115172)
    # SSA begins for if statement (line 618)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 619)
    # Processing the call arguments (line 619)
    str_115174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 25), 'str', 'Unknown filter type.')
    # Processing the call keyword arguments (line 619)
    kwargs_115175 = {}
    # Getting the type of 'ValueError' (line 619)
    ValueError_115173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 619)
    ValueError_call_result_115176 = invoke(stypy.reporting.localization.Localization(__file__, 619, 14), ValueError_115173, *[str_115174], **kwargs_115175)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 619, 8), ValueError_call_result_115176, 'raise parameter', BaseException)
    # SSA join for if statement (line 618)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fromimage(...): (line 620)
    # Processing the call arguments (line 620)
    
    # Call to filter(...): (line 620)
    # Processing the call arguments (line 620)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ftype' (line 620)
    ftype_115180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 38), 'ftype', False)
    # Getting the type of '_tdict' (line 620)
    _tdict_115181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 31), '_tdict', False)
    # Obtaining the member '__getitem__' of a type (line 620)
    getitem___115182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 31), _tdict_115181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 620)
    subscript_call_result_115183 = invoke(stypy.reporting.localization.Localization(__file__, 620, 31), getitem___115182, ftype_115180)
    
    # Processing the call keyword arguments (line 620)
    kwargs_115184 = {}
    # Getting the type of 'im' (line 620)
    im_115178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 21), 'im', False)
    # Obtaining the member 'filter' of a type (line 620)
    filter_115179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 21), im_115178, 'filter')
    # Calling filter(args, kwargs) (line 620)
    filter_call_result_115185 = invoke(stypy.reporting.localization.Localization(__file__, 620, 21), filter_115179, *[subscript_call_result_115183], **kwargs_115184)
    
    # Processing the call keyword arguments (line 620)
    kwargs_115186 = {}
    # Getting the type of 'fromimage' (line 620)
    fromimage_115177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 11), 'fromimage', False)
    # Calling fromimage(args, kwargs) (line 620)
    fromimage_call_result_115187 = invoke(stypy.reporting.localization.Localization(__file__, 620, 11), fromimage_115177, *[filter_call_result_115185], **kwargs_115186)
    
    # Assigning a type to the variable 'stypy_return_type' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'stypy_return_type', fromimage_call_result_115187)
    
    # ################# End of 'imfilter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'imfilter' in the type store
    # Getting the type of 'stypy_return_type' (line 568)
    stypy_return_type_115188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115188)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'imfilter'
    return stypy_return_type_115188

# Assigning a type to the variable 'imfilter' (line 568)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 0), 'imfilter', imfilter)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
