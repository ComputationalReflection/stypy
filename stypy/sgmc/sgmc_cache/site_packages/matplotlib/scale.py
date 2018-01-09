
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import numpy as np
7: from numpy import ma
8: 
9: from matplotlib.cbook import dedent
10: from matplotlib.ticker import (NullFormatter, ScalarFormatter,
11:                                LogFormatterSciNotation, LogitFormatter)
12: from matplotlib.ticker import (NullLocator, LogLocator, AutoLocator,
13:                                SymmetricalLogLocator, LogitLocator)
14: from matplotlib.transforms import Transform, IdentityTransform
15: from matplotlib import docstring
16: 
17: 
18: class ScaleBase(object):
19:     '''
20:     The base class for all scales.
21: 
22:     Scales are separable transformations, working on a single dimension.
23: 
24:     Any subclasses will want to override:
25: 
26:       - :attr:`name`
27:       - :meth:`get_transform`
28:       - :meth:`set_default_locators_and_formatters`
29: 
30:     And optionally:
31:       - :meth:`limit_range_for_scale`
32:     '''
33:     def get_transform(self):
34:         '''
35:         Return the :class:`~matplotlib.transforms.Transform` object
36:         associated with this scale.
37:         '''
38:         raise NotImplementedError()
39: 
40:     def set_default_locators_and_formatters(self, axis):
41:         '''
42:         Set the :class:`~matplotlib.ticker.Locator` and
43:         :class:`~matplotlib.ticker.Formatter` objects on the given
44:         axis to match this scale.
45:         '''
46:         raise NotImplementedError()
47: 
48:     def limit_range_for_scale(self, vmin, vmax, minpos):
49:         '''
50:         Returns the range *vmin*, *vmax*, possibly limited to the
51:         domain supported by this scale.
52: 
53:         *minpos* should be the minimum positive value in the data.
54:          This is used by log scales to determine a minimum value.
55:         '''
56:         return vmin, vmax
57: 
58: 
59: class LinearScale(ScaleBase):
60:     '''
61:     The default linear scale.
62:     '''
63: 
64:     name = 'linear'
65: 
66:     def __init__(self, axis, **kwargs):
67:         pass
68: 
69:     def set_default_locators_and_formatters(self, axis):
70:         '''
71:         Set the locators and formatters to reasonable defaults for
72:         linear scaling.
73:         '''
74:         axis.set_major_locator(AutoLocator())
75:         axis.set_major_formatter(ScalarFormatter())
76:         axis.set_minor_locator(NullLocator())
77:         axis.set_minor_formatter(NullFormatter())
78: 
79:     def get_transform(self):
80:         '''
81:         The transform for linear scaling is just the
82:         :class:`~matplotlib.transforms.IdentityTransform`.
83:         '''
84:         return IdentityTransform()
85: 
86: 
87: class LogTransformBase(Transform):
88:     input_dims = 1
89:     output_dims = 1
90:     is_separable = True
91:     has_inverse = True
92: 
93:     def __init__(self, nonpos):
94:         Transform.__init__(self)
95:         if nonpos == 'mask':
96:             self._fill_value = np.nan
97:         else:
98:             self._fill_value = 1e-300
99: 
100:     def transform_non_affine(self, a):
101:         with np.errstate(invalid="ignore"):
102:             a = np.where(a <= 0, self._fill_value, a)
103:         return np.divide(np.log(a, out=a), np.log(self.base), out=a)
104: 
105: 
106: class InvertedLogTransformBase(Transform):
107:     input_dims = 1
108:     output_dims = 1
109:     is_separable = True
110:     has_inverse = True
111: 
112:     def transform_non_affine(self, a):
113:         return ma.power(self.base, a)
114: 
115: 
116: class Log10Transform(LogTransformBase):
117:     base = 10.0
118: 
119:     def inverted(self):
120:         return InvertedLog10Transform()
121: 
122: 
123: class InvertedLog10Transform(InvertedLogTransformBase):
124:     base = 10.0
125: 
126:     def inverted(self):
127:         return Log10Transform()
128: 
129: 
130: class Log2Transform(LogTransformBase):
131:     base = 2.0
132: 
133:     def inverted(self):
134:         return InvertedLog2Transform()
135: 
136: 
137: class InvertedLog2Transform(InvertedLogTransformBase):
138:     base = 2.0
139: 
140:     def inverted(self):
141:         return Log2Transform()
142: 
143: 
144: class NaturalLogTransform(LogTransformBase):
145:     base = np.e
146: 
147:     def inverted(self):
148:         return InvertedNaturalLogTransform()
149: 
150: 
151: class InvertedNaturalLogTransform(InvertedLogTransformBase):
152:     base = np.e
153: 
154:     def inverted(self):
155:         return NaturalLogTransform()
156: 
157: 
158: class LogTransform(LogTransformBase):
159:     def __init__(self, base, nonpos):
160:         LogTransformBase.__init__(self, nonpos)
161:         self.base = base
162: 
163:     def inverted(self):
164:         return InvertedLogTransform(self.base)
165: 
166: 
167: class InvertedLogTransform(InvertedLogTransformBase):
168:     def __init__(self, base):
169:         InvertedLogTransformBase.__init__(self)
170:         self.base = base
171: 
172:     def inverted(self):
173:         return LogTransform(self.base)
174: 
175: 
176: class LogScale(ScaleBase):
177:     '''
178:     A standard logarithmic scale.  Care is taken so non-positive
179:     values are not plotted.
180: 
181:     For computational efficiency (to push as much as possible to Numpy
182:     C code in the common cases), this scale provides different
183:     transforms depending on the base of the logarithm:
184: 
185:        - base 10 (:class:`Log10Transform`)
186:        - base 2 (:class:`Log2Transform`)
187:        - base e (:class:`NaturalLogTransform`)
188:        - arbitrary base (:class:`LogTransform`)
189:     '''
190:     name = 'log'
191: 
192:     # compatibility shim
193:     LogTransformBase = LogTransformBase
194:     Log10Transform = Log10Transform
195:     InvertedLog10Transform = InvertedLog10Transform
196:     Log2Transform = Log2Transform
197:     InvertedLog2Transform = InvertedLog2Transform
198:     NaturalLogTransform = NaturalLogTransform
199:     InvertedNaturalLogTransform = InvertedNaturalLogTransform
200:     LogTransform = LogTransform
201:     InvertedLogTransform = InvertedLogTransform
202: 
203:     def __init__(self, axis, **kwargs):
204:         '''
205:         *basex*/*basey*:
206:            The base of the logarithm
207: 
208:         *nonposx*/*nonposy*: ['mask' | 'clip' ]
209:           non-positive values in *x* or *y* can be masked as
210:           invalid, or clipped to a very small positive number
211: 
212:         *subsx*/*subsy*:
213:            Where to place the subticks between each major tick.
214:            Should be a sequence of integers.  For example, in a log10
215:            scale: ``[2, 3, 4, 5, 6, 7, 8, 9]``
216: 
217:            will place 8 logarithmically spaced minor ticks between
218:            each major tick.
219:         '''
220:         if axis.axis_name == 'x':
221:             base = kwargs.pop('basex', 10.0)
222:             subs = kwargs.pop('subsx', None)
223:             nonpos = kwargs.pop('nonposx', 'mask')
224:         else:
225:             base = kwargs.pop('basey', 10.0)
226:             subs = kwargs.pop('subsy', None)
227:             nonpos = kwargs.pop('nonposy', 'mask')
228: 
229:         if nonpos not in ['mask', 'clip']:
230:             raise ValueError("nonposx, nonposy kwarg must be 'mask' or 'clip'")
231: 
232:         if base == 10.0:
233:             self._transform = self.Log10Transform(nonpos)
234:         elif base == 2.0:
235:             self._transform = self.Log2Transform(nonpos)
236:         elif base == np.e:
237:             self._transform = self.NaturalLogTransform(nonpos)
238:         else:
239:             self._transform = self.LogTransform(base, nonpos)
240: 
241:         self.base = base
242:         self.subs = subs
243: 
244:     def set_default_locators_and_formatters(self, axis):
245:         '''
246:         Set the locators and formatters to specialized versions for
247:         log scaling.
248:         '''
249:         axis.set_major_locator(LogLocator(self.base))
250:         axis.set_major_formatter(LogFormatterSciNotation(self.base))
251:         axis.set_minor_locator(LogLocator(self.base, self.subs))
252:         axis.set_minor_formatter(
253:             LogFormatterSciNotation(self.base,
254:                                     labelOnlyBase=(self.subs is not None)))
255: 
256:     def get_transform(self):
257:         '''
258:         Return a :class:`~matplotlib.transforms.Transform` instance
259:         appropriate for the given logarithm base.
260:         '''
261:         return self._transform
262: 
263:     def limit_range_for_scale(self, vmin, vmax, minpos):
264:         '''
265:         Limit the domain to positive values.
266:         '''
267:         if not np.isfinite(minpos):
268:             minpos = 1e-300  # This value should rarely if ever
269:                              # end up with a visible effect.
270: 
271:         return (minpos if vmin <= 0 else vmin,
272:                 minpos if vmax <= 0 else vmax)
273: 
274: 
275: class SymmetricalLogTransform(Transform):
276:     input_dims = 1
277:     output_dims = 1
278:     is_separable = True
279:     has_inverse = True
280: 
281:     def __init__(self, base, linthresh, linscale):
282:         Transform.__init__(self)
283:         self.base = base
284:         self.linthresh = linthresh
285:         self.linscale = linscale
286:         self._linscale_adj = (linscale / (1.0 - self.base ** -1))
287:         self._log_base = np.log(base)
288: 
289:     def transform_non_affine(self, a):
290:         sign = np.sign(a)
291:         masked = ma.masked_inside(a,
292:                                   -self.linthresh,
293:                                   self.linthresh,
294:                                   copy=False)
295:         log = sign * self.linthresh * (
296:             self._linscale_adj +
297:             ma.log(np.abs(masked) / self.linthresh) / self._log_base)
298:         if masked.mask.any():
299:             return ma.where(masked.mask, a * self._linscale_adj, log)
300:         else:
301:             return log
302: 
303:     def inverted(self):
304:         return InvertedSymmetricalLogTransform(self.base, self.linthresh,
305:                                                self.linscale)
306: 
307: 
308: class InvertedSymmetricalLogTransform(Transform):
309:     input_dims = 1
310:     output_dims = 1
311:     is_separable = True
312:     has_inverse = True
313: 
314:     def __init__(self, base, linthresh, linscale):
315:         Transform.__init__(self)
316:         symlog = SymmetricalLogTransform(base, linthresh, linscale)
317:         self.base = base
318:         self.linthresh = linthresh
319:         self.invlinthresh = symlog.transform(linthresh)
320:         self.linscale = linscale
321:         self._linscale_adj = (linscale / (1.0 - self.base ** -1))
322: 
323:     def transform_non_affine(self, a):
324:         sign = np.sign(a)
325:         masked = ma.masked_inside(a, -self.invlinthresh,
326:                                   self.invlinthresh, copy=False)
327:         exp = sign * self.linthresh * (
328:             ma.power(self.base, (sign * (masked / self.linthresh))
329:             - self._linscale_adj))
330:         if masked.mask.any():
331:             return ma.where(masked.mask, a / self._linscale_adj, exp)
332:         else:
333:             return exp
334: 
335:     def inverted(self):
336:         return SymmetricalLogTransform(self.base,
337:                                        self.linthresh, self.linscale)
338: 
339: 
340: class SymmetricalLogScale(ScaleBase):
341:     '''
342:     The symmetrical logarithmic scale is logarithmic in both the
343:     positive and negative directions from the origin.
344: 
345:     Since the values close to zero tend toward infinity, there is a
346:     need to have a range around zero that is linear.  The parameter
347:     *linthresh* allows the user to specify the size of this range
348:     (-*linthresh*, *linthresh*).
349:     '''
350:     name = 'symlog'
351:     # compatibility shim
352:     SymmetricalLogTransform = SymmetricalLogTransform
353:     InvertedSymmetricalLogTransform = InvertedSymmetricalLogTransform
354: 
355:     def __init__(self, axis, **kwargs):
356:         '''
357:         *basex*/*basey*:
358:            The base of the logarithm
359: 
360:         *linthreshx*/*linthreshy*:
361:           A single float which defines the range (-*x*, *x*), within
362:           which the plot is linear. This avoids having the plot go to
363:           infinity around zero.
364: 
365:         *subsx*/*subsy*:
366:            Where to place the subticks between each major tick.
367:            Should be a sequence of integers.  For example, in a log10
368:            scale: ``[2, 3, 4, 5, 6, 7, 8, 9]``
369: 
370:            will place 8 logarithmically spaced minor ticks between
371:            each major tick.
372: 
373:         *linscalex*/*linscaley*:
374:            This allows the linear range (-*linthresh* to *linthresh*)
375:            to be stretched relative to the logarithmic range.  Its
376:            value is the number of decades to use for each half of the
377:            linear range.  For example, when *linscale* == 1.0 (the
378:            default), the space used for the positive and negative
379:            halves of the linear range will be equal to one decade in
380:            the logarithmic range.
381:         '''
382:         if axis.axis_name == 'x':
383:             base = kwargs.pop('basex', 10.0)
384:             linthresh = kwargs.pop('linthreshx', 2.0)
385:             subs = kwargs.pop('subsx', None)
386:             linscale = kwargs.pop('linscalex', 1.0)
387:         else:
388:             base = kwargs.pop('basey', 10.0)
389:             linthresh = kwargs.pop('linthreshy', 2.0)
390:             subs = kwargs.pop('subsy', None)
391:             linscale = kwargs.pop('linscaley', 1.0)
392: 
393:         if base <= 1.0:
394:             raise ValueError("'basex/basey' must be larger than 1")
395:         if linthresh <= 0.0:
396:             raise ValueError("'linthreshx/linthreshy' must be positive")
397:         if linscale <= 0.0:
398:             raise ValueError("'linscalex/linthreshy' must be positive")
399: 
400:         self._transform = self.SymmetricalLogTransform(base,
401:                                                        linthresh,
402:                                                        linscale)
403: 
404:         self.base = base
405:         self.linthresh = linthresh
406:         self.linscale = linscale
407:         self.subs = subs
408: 
409:     def set_default_locators_and_formatters(self, axis):
410:         '''
411:         Set the locators and formatters to specialized versions for
412:         symmetrical log scaling.
413:         '''
414:         axis.set_major_locator(SymmetricalLogLocator(self.get_transform()))
415:         axis.set_major_formatter(LogFormatterSciNotation(self.base))
416:         axis.set_minor_locator(SymmetricalLogLocator(self.get_transform(),
417:                                                      self.subs))
418:         axis.set_minor_formatter(NullFormatter())
419: 
420:     def get_transform(self):
421:         '''
422:         Return a :class:`SymmetricalLogTransform` instance.
423:         '''
424:         return self._transform
425: 
426: 
427: class LogitTransform(Transform):
428:     input_dims = 1
429:     output_dims = 1
430:     is_separable = True
431:     has_inverse = True
432: 
433:     def __init__(self, nonpos):
434:         Transform.__init__(self)
435:         if nonpos == 'mask':
436:             self._fill_value = np.nan
437:         else:
438:             self._fill_value = 1e-300
439:         self._nonpos = nonpos
440: 
441:     def transform_non_affine(self, a):
442:         '''logit transform (base 10), masked or clipped'''
443:         with np.errstate(invalid="ignore"):
444:             a = np.select(
445:                 [a <= 0, a >= 1], [self._fill_value, 1 - self._fill_value], a)
446:         return np.log10(a / (1 - a))
447: 
448:     def inverted(self):
449:         return LogisticTransform(self._nonpos)
450: 
451: 
452: class LogisticTransform(Transform):
453:     input_dims = 1
454:     output_dims = 1
455:     is_separable = True
456:     has_inverse = True
457: 
458:     def __init__(self, nonpos='mask'):
459:         Transform.__init__(self)
460:         self._nonpos = nonpos
461: 
462:     def transform_non_affine(self, a):
463:         '''logistic transform (base 10)'''
464:         return 1.0 / (1 + 10**(-a))
465: 
466:     def inverted(self):
467:         return LogitTransform(self._nonpos)
468: 
469: 
470: class LogitScale(ScaleBase):
471:     '''
472:     Logit scale for data between zero and one, both excluded.
473: 
474:     This scale is similar to a log scale close to zero and to one, and almost
475:     linear around 0.5. It maps the interval ]0, 1[ onto ]-infty, +infty[.
476:     '''
477:     name = 'logit'
478: 
479:     def __init__(self, axis, nonpos='mask'):
480:         '''
481:         *nonpos*: ['mask' | 'clip' ]
482:           values beyond ]0, 1[ can be masked as invalid, or clipped to a number
483:           very close to 0 or 1
484:         '''
485:         if nonpos not in ['mask', 'clip']:
486:             raise ValueError("nonposx, nonposy kwarg must be 'mask' or 'clip'")
487: 
488:         self._transform = LogitTransform(nonpos)
489: 
490:     def get_transform(self):
491:         '''
492:         Return a :class:`LogitTransform` instance.
493:         '''
494:         return self._transform
495: 
496:     def set_default_locators_and_formatters(self, axis):
497:         # ..., 0.01, 0.1, 0.5, 0.9, 0.99, ...
498:         axis.set_major_locator(LogitLocator())
499:         axis.set_major_formatter(LogitFormatter())
500:         axis.set_minor_locator(LogitLocator(minor=True))
501:         axis.set_minor_formatter(LogitFormatter())
502: 
503:     def limit_range_for_scale(self, vmin, vmax, minpos):
504:         '''
505:         Limit the domain to values between 0 and 1 (excluded).
506:         '''
507:         if not np.isfinite(minpos):
508:             minpos = 1e-7    # This value should rarely if ever
509:                              # end up with a visible effect.
510:         return (minpos if vmin <= 0 else vmin,
511:                 1 - minpos if vmax >= 1 else vmax)
512: 
513: 
514: _scale_mapping = {
515:     'linear': LinearScale,
516:     'log':    LogScale,
517:     'symlog': SymmetricalLogScale,
518:     'logit':  LogitScale,
519:     }
520: 
521: 
522: def get_scale_names():
523:     return sorted(_scale_mapping)
524: 
525: 
526: def scale_factory(scale, axis, **kwargs):
527:     '''
528:     Return a scale class by name.
529: 
530:     ACCEPTS: [ %(names)s ]
531:     '''
532:     scale = scale.lower()
533:     if scale is None:
534:         scale = 'linear'
535: 
536:     if scale not in _scale_mapping:
537:         raise ValueError("Unknown scale type '%s'" % scale)
538: 
539:     return _scale_mapping[scale](axis, **kwargs)
540: scale_factory.__doc__ = dedent(scale_factory.__doc__) % \
541:     {'names': " | ".join(get_scale_names())}
542: 
543: 
544: def register_scale(scale_class):
545:     '''
546:     Register a new kind of scale.
547: 
548:     *scale_class* must be a subclass of :class:`ScaleBase`.
549:     '''
550:     _scale_mapping[scale_class.name] = scale_class
551: 
552: 
553: def get_scale_docs():
554:     '''
555:     Helper function for generating docstrings related to scales.
556:     '''
557:     docs = []
558:     for name in get_scale_names():
559:         scale_class = _scale_mapping[name]
560:         docs.append("    '%s'" % name)
561:         docs.append("")
562:         class_docs = dedent(scale_class.__init__.__doc__)
563:         class_docs = "".join(["        %s\n" %
564:                               x for x in class_docs.split("\n")])
565:         docs.append(class_docs)
566:         docs.append("")
567:     return "\n".join(docs)
568: 
569: 
570: docstring.interpd.update(
571:     scale=' | '.join([repr(x) for x in get_scale_names()]),
572:     scale_docs=get_scale_docs().rstrip(),
573:     )
574: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_130259 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_130259) is not StypyTypeError):

    if (import_130259 != 'pyd_module'):
        __import__(import_130259)
        sys_modules_130260 = sys.modules[import_130259]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_130260.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_130259)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_130261 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_130261) is not StypyTypeError):

    if (import_130261 != 'pyd_module'):
        __import__(import_130261)
        sys_modules_130262 = sys.modules[import_130261]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_130262.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_130261)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy import ma' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_130263 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_130263) is not StypyTypeError):

    if (import_130263 != 'pyd_module'):
        __import__(import_130263)
        sys_modules_130264 = sys.modules[import_130263]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', sys_modules_130264.module_type_store, module_type_store, ['ma'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_130264, sys_modules_130264.module_type_store, module_type_store)
    else:
        from numpy import ma

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', None, module_type_store, ['ma'], [ma])

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_130263)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.cbook import dedent' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_130265 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.cbook')

if (type(import_130265) is not StypyTypeError):

    if (import_130265 != 'pyd_module'):
        __import__(import_130265)
        sys_modules_130266 = sys.modules[import_130265]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.cbook', sys_modules_130266.module_type_store, module_type_store, ['dedent'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_130266, sys_modules_130266.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import dedent

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.cbook', None, module_type_store, ['dedent'], [dedent])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.cbook', import_130265)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib.ticker import NullFormatter, ScalarFormatter, LogFormatterSciNotation, LogitFormatter' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_130267 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.ticker')

if (type(import_130267) is not StypyTypeError):

    if (import_130267 != 'pyd_module'):
        __import__(import_130267)
        sys_modules_130268 = sys.modules[import_130267]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.ticker', sys_modules_130268.module_type_store, module_type_store, ['NullFormatter', 'ScalarFormatter', 'LogFormatterSciNotation', 'LogitFormatter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_130268, sys_modules_130268.module_type_store, module_type_store)
    else:
        from matplotlib.ticker import NullFormatter, ScalarFormatter, LogFormatterSciNotation, LogitFormatter

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.ticker', None, module_type_store, ['NullFormatter', 'ScalarFormatter', 'LogFormatterSciNotation', 'LogitFormatter'], [NullFormatter, ScalarFormatter, LogFormatterSciNotation, LogitFormatter])

else:
    # Assigning a type to the variable 'matplotlib.ticker' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.ticker', import_130267)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.ticker import NullLocator, LogLocator, AutoLocator, SymmetricalLogLocator, LogitLocator' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_130269 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.ticker')

if (type(import_130269) is not StypyTypeError):

    if (import_130269 != 'pyd_module'):
        __import__(import_130269)
        sys_modules_130270 = sys.modules[import_130269]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.ticker', sys_modules_130270.module_type_store, module_type_store, ['NullLocator', 'LogLocator', 'AutoLocator', 'SymmetricalLogLocator', 'LogitLocator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_130270, sys_modules_130270.module_type_store, module_type_store)
    else:
        from matplotlib.ticker import NullLocator, LogLocator, AutoLocator, SymmetricalLogLocator, LogitLocator

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.ticker', None, module_type_store, ['NullLocator', 'LogLocator', 'AutoLocator', 'SymmetricalLogLocator', 'LogitLocator'], [NullLocator, LogLocator, AutoLocator, SymmetricalLogLocator, LogitLocator])

else:
    # Assigning a type to the variable 'matplotlib.ticker' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.ticker', import_130269)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib.transforms import Transform, IdentityTransform' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_130271 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.transforms')

if (type(import_130271) is not StypyTypeError):

    if (import_130271 != 'pyd_module'):
        __import__(import_130271)
        sys_modules_130272 = sys.modules[import_130271]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.transforms', sys_modules_130272.module_type_store, module_type_store, ['Transform', 'IdentityTransform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_130272, sys_modules_130272.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Transform, IdentityTransform

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.transforms', None, module_type_store, ['Transform', 'IdentityTransform'], [Transform, IdentityTransform])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.transforms', import_130271)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib import docstring' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_130273 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib')

if (type(import_130273) is not StypyTypeError):

    if (import_130273 != 'pyd_module'):
        __import__(import_130273)
        sys_modules_130274 = sys.modules[import_130273]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', sys_modules_130274.module_type_store, module_type_store, ['docstring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_130274, sys_modules_130274.module_type_store, module_type_store)
    else:
        from matplotlib import docstring

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', None, module_type_store, ['docstring'], [docstring])

else:
    # Assigning a type to the variable 'matplotlib' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', import_130273)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'ScaleBase' class

class ScaleBase(object, ):
    unicode_130275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'unicode', u'\n    The base class for all scales.\n\n    Scales are separable transformations, working on a single dimension.\n\n    Any subclasses will want to override:\n\n      - :attr:`name`\n      - :meth:`get_transform`\n      - :meth:`set_default_locators_and_formatters`\n\n    And optionally:\n      - :meth:`limit_range_for_scale`\n    ')

    @norecursion
    def get_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_transform'
        module_type_store = module_type_store.open_function_context('get_transform', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScaleBase.get_transform.__dict__.__setitem__('stypy_localization', localization)
        ScaleBase.get_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScaleBase.get_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScaleBase.get_transform.__dict__.__setitem__('stypy_function_name', 'ScaleBase.get_transform')
        ScaleBase.get_transform.__dict__.__setitem__('stypy_param_names_list', [])
        ScaleBase.get_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScaleBase.get_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScaleBase.get_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScaleBase.get_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScaleBase.get_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScaleBase.get_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScaleBase.get_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_transform(...)' code ##################

        unicode_130276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'unicode', u'\n        Return the :class:`~matplotlib.transforms.Transform` object\n        associated with this scale.\n        ')
        
        # Call to NotImplementedError(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_130278 = {}
        # Getting the type of 'NotImplementedError' (line 38)
        NotImplementedError_130277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 38)
        NotImplementedError_call_result_130279 = invoke(stypy.reporting.localization.Localization(__file__, 38, 14), NotImplementedError_130277, *[], **kwargs_130278)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 38, 8), NotImplementedError_call_result_130279, 'raise parameter', BaseException)
        
        # ################# End of 'get_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_130280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_transform'
        return stypy_return_type_130280


    @norecursion
    def set_default_locators_and_formatters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_default_locators_and_formatters'
        module_type_store = module_type_store.open_function_context('set_default_locators_and_formatters', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_localization', localization)
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_function_name', 'ScaleBase.set_default_locators_and_formatters')
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScaleBase.set_default_locators_and_formatters.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScaleBase.set_default_locators_and_formatters', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_default_locators_and_formatters', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_default_locators_and_formatters(...)' code ##################

        unicode_130281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'unicode', u'\n        Set the :class:`~matplotlib.ticker.Locator` and\n        :class:`~matplotlib.ticker.Formatter` objects on the given\n        axis to match this scale.\n        ')
        
        # Call to NotImplementedError(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_130283 = {}
        # Getting the type of 'NotImplementedError' (line 46)
        NotImplementedError_130282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 46)
        NotImplementedError_call_result_130284 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), NotImplementedError_130282, *[], **kwargs_130283)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 46, 8), NotImplementedError_call_result_130284, 'raise parameter', BaseException)
        
        # ################# End of 'set_default_locators_and_formatters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_default_locators_and_formatters' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_130285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_default_locators_and_formatters'
        return stypy_return_type_130285


    @norecursion
    def limit_range_for_scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'limit_range_for_scale'
        module_type_store = module_type_store.open_function_context('limit_range_for_scale', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_localization', localization)
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_function_name', 'ScaleBase.limit_range_for_scale')
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_param_names_list', ['vmin', 'vmax', 'minpos'])
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScaleBase.limit_range_for_scale.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScaleBase.limit_range_for_scale', ['vmin', 'vmax', 'minpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'limit_range_for_scale', localization, ['vmin', 'vmax', 'minpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'limit_range_for_scale(...)' code ##################

        unicode_130286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'unicode', u'\n        Returns the range *vmin*, *vmax*, possibly limited to the\n        domain supported by this scale.\n\n        *minpos* should be the minimum positive value in the data.\n         This is used by log scales to determine a minimum value.\n        ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_130287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        # Getting the type of 'vmin' (line 56)
        vmin_130288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'vmin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 15), tuple_130287, vmin_130288)
        # Adding element type (line 56)
        # Getting the type of 'vmax' (line 56)
        vmax_130289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'vmax')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 15), tuple_130287, vmax_130289)
        
        # Assigning a type to the variable 'stypy_return_type' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'stypy_return_type', tuple_130287)
        
        # ################# End of 'limit_range_for_scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'limit_range_for_scale' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_130290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130290)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'limit_range_for_scale'
        return stypy_return_type_130290


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 0, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScaleBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ScaleBase' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'ScaleBase', ScaleBase)
# Declaration of the 'LinearScale' class
# Getting the type of 'ScaleBase' (line 59)
ScaleBase_130291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'ScaleBase')

class LinearScale(ScaleBase_130291, ):
    unicode_130292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'unicode', u'\n    The default linear scale.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearScale.__init__', ['axis'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['axis'], arguments)
        
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


    @norecursion
    def set_default_locators_and_formatters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_default_locators_and_formatters'
        module_type_store = module_type_store.open_function_context('set_default_locators_and_formatters', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_localization', localization)
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_function_name', 'LinearScale.set_default_locators_and_formatters')
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearScale.set_default_locators_and_formatters', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_default_locators_and_formatters', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_default_locators_and_formatters(...)' code ##################

        unicode_130293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'unicode', u'\n        Set the locators and formatters to reasonable defaults for\n        linear scaling.\n        ')
        
        # Call to set_major_locator(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to AutoLocator(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_130297 = {}
        # Getting the type of 'AutoLocator' (line 74)
        AutoLocator_130296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 31), 'AutoLocator', False)
        # Calling AutoLocator(args, kwargs) (line 74)
        AutoLocator_call_result_130298 = invoke(stypy.reporting.localization.Localization(__file__, 74, 31), AutoLocator_130296, *[], **kwargs_130297)
        
        # Processing the call keyword arguments (line 74)
        kwargs_130299 = {}
        # Getting the type of 'axis' (line 74)
        axis_130294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'axis', False)
        # Obtaining the member 'set_major_locator' of a type (line 74)
        set_major_locator_130295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), axis_130294, 'set_major_locator')
        # Calling set_major_locator(args, kwargs) (line 74)
        set_major_locator_call_result_130300 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), set_major_locator_130295, *[AutoLocator_call_result_130298], **kwargs_130299)
        
        
        # Call to set_major_formatter(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to ScalarFormatter(...): (line 75)
        # Processing the call keyword arguments (line 75)
        kwargs_130304 = {}
        # Getting the type of 'ScalarFormatter' (line 75)
        ScalarFormatter_130303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 33), 'ScalarFormatter', False)
        # Calling ScalarFormatter(args, kwargs) (line 75)
        ScalarFormatter_call_result_130305 = invoke(stypy.reporting.localization.Localization(__file__, 75, 33), ScalarFormatter_130303, *[], **kwargs_130304)
        
        # Processing the call keyword arguments (line 75)
        kwargs_130306 = {}
        # Getting the type of 'axis' (line 75)
        axis_130301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'axis', False)
        # Obtaining the member 'set_major_formatter' of a type (line 75)
        set_major_formatter_130302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), axis_130301, 'set_major_formatter')
        # Calling set_major_formatter(args, kwargs) (line 75)
        set_major_formatter_call_result_130307 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), set_major_formatter_130302, *[ScalarFormatter_call_result_130305], **kwargs_130306)
        
        
        # Call to set_minor_locator(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to NullLocator(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_130311 = {}
        # Getting the type of 'NullLocator' (line 76)
        NullLocator_130310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'NullLocator', False)
        # Calling NullLocator(args, kwargs) (line 76)
        NullLocator_call_result_130312 = invoke(stypy.reporting.localization.Localization(__file__, 76, 31), NullLocator_130310, *[], **kwargs_130311)
        
        # Processing the call keyword arguments (line 76)
        kwargs_130313 = {}
        # Getting the type of 'axis' (line 76)
        axis_130308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'axis', False)
        # Obtaining the member 'set_minor_locator' of a type (line 76)
        set_minor_locator_130309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), axis_130308, 'set_minor_locator')
        # Calling set_minor_locator(args, kwargs) (line 76)
        set_minor_locator_call_result_130314 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), set_minor_locator_130309, *[NullLocator_call_result_130312], **kwargs_130313)
        
        
        # Call to set_minor_formatter(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to NullFormatter(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_130318 = {}
        # Getting the type of 'NullFormatter' (line 77)
        NullFormatter_130317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'NullFormatter', False)
        # Calling NullFormatter(args, kwargs) (line 77)
        NullFormatter_call_result_130319 = invoke(stypy.reporting.localization.Localization(__file__, 77, 33), NullFormatter_130317, *[], **kwargs_130318)
        
        # Processing the call keyword arguments (line 77)
        kwargs_130320 = {}
        # Getting the type of 'axis' (line 77)
        axis_130315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'axis', False)
        # Obtaining the member 'set_minor_formatter' of a type (line 77)
        set_minor_formatter_130316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), axis_130315, 'set_minor_formatter')
        # Calling set_minor_formatter(args, kwargs) (line 77)
        set_minor_formatter_call_result_130321 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), set_minor_formatter_130316, *[NullFormatter_call_result_130319], **kwargs_130320)
        
        
        # ################# End of 'set_default_locators_and_formatters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_default_locators_and_formatters' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_130322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_default_locators_and_formatters'
        return stypy_return_type_130322


    @norecursion
    def get_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_transform'
        module_type_store = module_type_store.open_function_context('get_transform', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearScale.get_transform.__dict__.__setitem__('stypy_localization', localization)
        LinearScale.get_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearScale.get_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearScale.get_transform.__dict__.__setitem__('stypy_function_name', 'LinearScale.get_transform')
        LinearScale.get_transform.__dict__.__setitem__('stypy_param_names_list', [])
        LinearScale.get_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearScale.get_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearScale.get_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearScale.get_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearScale.get_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearScale.get_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearScale.get_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_transform(...)' code ##################

        unicode_130323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'unicode', u'\n        The transform for linear scaling is just the\n        :class:`~matplotlib.transforms.IdentityTransform`.\n        ')
        
        # Call to IdentityTransform(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_130325 = {}
        # Getting the type of 'IdentityTransform' (line 84)
        IdentityTransform_130324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'IdentityTransform', False)
        # Calling IdentityTransform(args, kwargs) (line 84)
        IdentityTransform_call_result_130326 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), IdentityTransform_130324, *[], **kwargs_130325)
        
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', IdentityTransform_call_result_130326)
        
        # ################# End of 'get_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_130327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130327)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_transform'
        return stypy_return_type_130327


# Assigning a type to the variable 'LinearScale' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'LinearScale', LinearScale)

# Assigning a Str to a Name (line 64):
unicode_130328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 11), 'unicode', u'linear')
# Getting the type of 'LinearScale'
LinearScale_130329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LinearScale')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LinearScale_130329, 'name', unicode_130328)
# Declaration of the 'LogTransformBase' class
# Getting the type of 'Transform' (line 87)
Transform_130330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'Transform')

class LogTransformBase(Transform_130330, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogTransformBase.__init__', ['nonpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['nonpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'self' (line 94)
        self_130333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'self', False)
        # Processing the call keyword arguments (line 94)
        kwargs_130334 = {}
        # Getting the type of 'Transform' (line 94)
        Transform_130331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'Transform', False)
        # Obtaining the member '__init__' of a type (line 94)
        init___130332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), Transform_130331, '__init__')
        # Calling __init__(args, kwargs) (line 94)
        init___call_result_130335 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), init___130332, *[self_130333], **kwargs_130334)
        
        
        
        # Getting the type of 'nonpos' (line 95)
        nonpos_130336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'nonpos')
        unicode_130337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'unicode', u'mask')
        # Applying the binary operator '==' (line 95)
        result_eq_130338 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 11), '==', nonpos_130336, unicode_130337)
        
        # Testing the type of an if condition (line 95)
        if_condition_130339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), result_eq_130338)
        # Assigning a type to the variable 'if_condition_130339' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_130339', if_condition_130339)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 96):
        # Getting the type of 'np' (line 96)
        np_130340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'np')
        # Obtaining the member 'nan' of a type (line 96)
        nan_130341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 31), np_130340, 'nan')
        # Getting the type of 'self' (line 96)
        self_130342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'self')
        # Setting the type of the member '_fill_value' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), self_130342, '_fill_value', nan_130341)
        # SSA branch for the else part of an if statement (line 95)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Attribute (line 98):
        float_130343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'float')
        # Getting the type of 'self' (line 98)
        self_130344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self')
        # Setting the type of the member '_fill_value' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_130344, '_fill_value', float_130343)
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
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
        module_type_store = module_type_store.open_function_context('transform_non_affine', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'LogTransformBase.transform_non_affine')
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['a'])
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogTransformBase.transform_non_affine', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform_non_affine', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform_non_affine(...)' code ##################

        
        # Call to errstate(...): (line 101)
        # Processing the call keyword arguments (line 101)
        unicode_130347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 33), 'unicode', u'ignore')
        keyword_130348 = unicode_130347
        kwargs_130349 = {'invalid': keyword_130348}
        # Getting the type of 'np' (line 101)
        np_130345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'np', False)
        # Obtaining the member 'errstate' of a type (line 101)
        errstate_130346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), np_130345, 'errstate')
        # Calling errstate(args, kwargs) (line 101)
        errstate_call_result_130350 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), errstate_130346, *[], **kwargs_130349)
        
        with_130351 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 101, 13), errstate_call_result_130350, 'with parameter', '__enter__', '__exit__')

        if with_130351:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 101)
            enter___130352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), errstate_call_result_130350, '__enter__')
            with_enter_130353 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), enter___130352)
            
            # Assigning a Call to a Name (line 102):
            
            # Call to where(...): (line 102)
            # Processing the call arguments (line 102)
            
            # Getting the type of 'a' (line 102)
            a_130356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'a', False)
            int_130357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 30), 'int')
            # Applying the binary operator '<=' (line 102)
            result_le_130358 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 25), '<=', a_130356, int_130357)
            
            # Getting the type of 'self' (line 102)
            self_130359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 33), 'self', False)
            # Obtaining the member '_fill_value' of a type (line 102)
            _fill_value_130360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 33), self_130359, '_fill_value')
            # Getting the type of 'a' (line 102)
            a_130361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 51), 'a', False)
            # Processing the call keyword arguments (line 102)
            kwargs_130362 = {}
            # Getting the type of 'np' (line 102)
            np_130354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'np', False)
            # Obtaining the member 'where' of a type (line 102)
            where_130355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), np_130354, 'where')
            # Calling where(args, kwargs) (line 102)
            where_call_result_130363 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), where_130355, *[result_le_130358, _fill_value_130360, a_130361], **kwargs_130362)
            
            # Assigning a type to the variable 'a' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'a', where_call_result_130363)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 101)
            exit___130364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), errstate_call_result_130350, '__exit__')
            with_exit_130365 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), exit___130364, None, None, None)

        
        # Call to divide(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to log(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'a' (line 103)
        a_130370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 32), 'a', False)
        # Processing the call keyword arguments (line 103)
        # Getting the type of 'a' (line 103)
        a_130371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'a', False)
        keyword_130372 = a_130371
        kwargs_130373 = {'out': keyword_130372}
        # Getting the type of 'np' (line 103)
        np_130368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'np', False)
        # Obtaining the member 'log' of a type (line 103)
        log_130369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 25), np_130368, 'log')
        # Calling log(args, kwargs) (line 103)
        log_call_result_130374 = invoke(stypy.reporting.localization.Localization(__file__, 103, 25), log_130369, *[a_130370], **kwargs_130373)
        
        
        # Call to log(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_130377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 50), 'self', False)
        # Obtaining the member 'base' of a type (line 103)
        base_130378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 50), self_130377, 'base')
        # Processing the call keyword arguments (line 103)
        kwargs_130379 = {}
        # Getting the type of 'np' (line 103)
        np_130375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 43), 'np', False)
        # Obtaining the member 'log' of a type (line 103)
        log_130376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 43), np_130375, 'log')
        # Calling log(args, kwargs) (line 103)
        log_call_result_130380 = invoke(stypy.reporting.localization.Localization(__file__, 103, 43), log_130376, *[base_130378], **kwargs_130379)
        
        # Processing the call keyword arguments (line 103)
        # Getting the type of 'a' (line 103)
        a_130381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 66), 'a', False)
        keyword_130382 = a_130381
        kwargs_130383 = {'out': keyword_130382}
        # Getting the type of 'np' (line 103)
        np_130366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'np', False)
        # Obtaining the member 'divide' of a type (line 103)
        divide_130367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), np_130366, 'divide')
        # Calling divide(args, kwargs) (line 103)
        divide_call_result_130384 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), divide_130367, *[log_call_result_130374, log_call_result_130380], **kwargs_130383)
        
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', divide_call_result_130384)
        
        # ################# End of 'transform_non_affine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform_non_affine' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_130385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform_non_affine'
        return stypy_return_type_130385


# Assigning a type to the variable 'LogTransformBase' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'LogTransformBase', LogTransformBase)

# Assigning a Num to a Name (line 88):
int_130386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 17), 'int')
# Getting the type of 'LogTransformBase'
LogTransformBase_130387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogTransformBase')
# Setting the type of the member 'input_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogTransformBase_130387, 'input_dims', int_130386)

# Assigning a Num to a Name (line 89):
int_130388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'int')
# Getting the type of 'LogTransformBase'
LogTransformBase_130389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogTransformBase')
# Setting the type of the member 'output_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogTransformBase_130389, 'output_dims', int_130388)

# Assigning a Name to a Name (line 90):
# Getting the type of 'True' (line 90)
True_130390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'True')
# Getting the type of 'LogTransformBase'
LogTransformBase_130391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogTransformBase')
# Setting the type of the member 'is_separable' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogTransformBase_130391, 'is_separable', True_130390)

# Assigning a Name to a Name (line 91):
# Getting the type of 'True' (line 91)
True_130392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'True')
# Getting the type of 'LogTransformBase'
LogTransformBase_130393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogTransformBase')
# Setting the type of the member 'has_inverse' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogTransformBase_130393, 'has_inverse', True_130392)
# Declaration of the 'InvertedLogTransformBase' class
# Getting the type of 'Transform' (line 106)
Transform_130394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 31), 'Transform')

class InvertedLogTransformBase(Transform_130394, ):

    @norecursion
    def transform_non_affine(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'transform_non_affine'
        module_type_store = module_type_store.open_function_context('transform_non_affine', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'InvertedLogTransformBase.transform_non_affine')
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['a'])
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InvertedLogTransformBase.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLogTransformBase.transform_non_affine', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform_non_affine', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform_non_affine(...)' code ##################

        
        # Call to power(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_130397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'self', False)
        # Obtaining the member 'base' of a type (line 113)
        base_130398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), self_130397, 'base')
        # Getting the type of 'a' (line 113)
        a_130399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'a', False)
        # Processing the call keyword arguments (line 113)
        kwargs_130400 = {}
        # Getting the type of 'ma' (line 113)
        ma_130395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'ma', False)
        # Obtaining the member 'power' of a type (line 113)
        power_130396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 15), ma_130395, 'power')
        # Calling power(args, kwargs) (line 113)
        power_call_result_130401 = invoke(stypy.reporting.localization.Localization(__file__, 113, 15), power_130396, *[base_130398, a_130399], **kwargs_130400)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', power_call_result_130401)
        
        # ################# End of 'transform_non_affine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform_non_affine' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_130402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform_non_affine'
        return stypy_return_type_130402


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 106, 0, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLogTransformBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InvertedLogTransformBase' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'InvertedLogTransformBase', InvertedLogTransformBase)

# Assigning a Num to a Name (line 107):
int_130403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 17), 'int')
# Getting the type of 'InvertedLogTransformBase'
InvertedLogTransformBase_130404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedLogTransformBase')
# Setting the type of the member 'input_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedLogTransformBase_130404, 'input_dims', int_130403)

# Assigning a Num to a Name (line 108):
int_130405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'int')
# Getting the type of 'InvertedLogTransformBase'
InvertedLogTransformBase_130406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedLogTransformBase')
# Setting the type of the member 'output_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedLogTransformBase_130406, 'output_dims', int_130405)

# Assigning a Name to a Name (line 109):
# Getting the type of 'True' (line 109)
True_130407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'True')
# Getting the type of 'InvertedLogTransformBase'
InvertedLogTransformBase_130408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedLogTransformBase')
# Setting the type of the member 'is_separable' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedLogTransformBase_130408, 'is_separable', True_130407)

# Assigning a Name to a Name (line 110):
# Getting the type of 'True' (line 110)
True_130409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'True')
# Getting the type of 'InvertedLogTransformBase'
InvertedLogTransformBase_130410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedLogTransformBase')
# Setting the type of the member 'has_inverse' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedLogTransformBase_130410, 'has_inverse', True_130409)
# Declaration of the 'Log10Transform' class
# Getting the type of 'LogTransformBase' (line 116)
LogTransformBase_130411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'LogTransformBase')

class Log10Transform(LogTransformBase_130411, ):

    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log10Transform.inverted.__dict__.__setitem__('stypy_localization', localization)
        Log10Transform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log10Transform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log10Transform.inverted.__dict__.__setitem__('stypy_function_name', 'Log10Transform.inverted')
        Log10Transform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        Log10Transform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        Log10Transform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log10Transform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log10Transform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log10Transform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log10Transform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log10Transform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to InvertedLog10Transform(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_130413 = {}
        # Getting the type of 'InvertedLog10Transform' (line 120)
        InvertedLog10Transform_130412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'InvertedLog10Transform', False)
        # Calling InvertedLog10Transform(args, kwargs) (line 120)
        InvertedLog10Transform_call_result_130414 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), InvertedLog10Transform_130412, *[], **kwargs_130413)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', InvertedLog10Transform_call_result_130414)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_130415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130415)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130415


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 116, 0, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log10Transform.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Log10Transform' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'Log10Transform', Log10Transform)

# Assigning a Num to a Name (line 117):
float_130416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 11), 'float')
# Getting the type of 'Log10Transform'
Log10Transform_130417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Log10Transform')
# Setting the type of the member 'base' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Log10Transform_130417, 'base', float_130416)
# Declaration of the 'InvertedLog10Transform' class
# Getting the type of 'InvertedLogTransformBase' (line 123)
InvertedLogTransformBase_130418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'InvertedLogTransformBase')

class InvertedLog10Transform(InvertedLogTransformBase_130418, ):

    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_localization', localization)
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedLog10Transform.inverted')
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InvertedLog10Transform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLog10Transform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to Log10Transform(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_130420 = {}
        # Getting the type of 'Log10Transform' (line 127)
        Log10Transform_130419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'Log10Transform', False)
        # Calling Log10Transform(args, kwargs) (line 127)
        Log10Transform_call_result_130421 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), Log10Transform_130419, *[], **kwargs_130420)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', Log10Transform_call_result_130421)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_130422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130422


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 123, 0, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLog10Transform.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InvertedLog10Transform' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'InvertedLog10Transform', InvertedLog10Transform)

# Assigning a Num to a Name (line 124):
float_130423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 11), 'float')
# Getting the type of 'InvertedLog10Transform'
InvertedLog10Transform_130424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedLog10Transform')
# Setting the type of the member 'base' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedLog10Transform_130424, 'base', float_130423)
# Declaration of the 'Log2Transform' class
# Getting the type of 'LogTransformBase' (line 130)
LogTransformBase_130425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'LogTransformBase')

class Log2Transform(LogTransformBase_130425, ):

    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log2Transform.inverted.__dict__.__setitem__('stypy_localization', localization)
        Log2Transform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log2Transform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log2Transform.inverted.__dict__.__setitem__('stypy_function_name', 'Log2Transform.inverted')
        Log2Transform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        Log2Transform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        Log2Transform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log2Transform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log2Transform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log2Transform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log2Transform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log2Transform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to InvertedLog2Transform(...): (line 134)
        # Processing the call keyword arguments (line 134)
        kwargs_130427 = {}
        # Getting the type of 'InvertedLog2Transform' (line 134)
        InvertedLog2Transform_130426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'InvertedLog2Transform', False)
        # Calling InvertedLog2Transform(args, kwargs) (line 134)
        InvertedLog2Transform_call_result_130428 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), InvertedLog2Transform_130426, *[], **kwargs_130427)
        
        # Assigning a type to the variable 'stypy_return_type' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', InvertedLog2Transform_call_result_130428)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_130429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130429


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 130, 0, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log2Transform.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Log2Transform' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'Log2Transform', Log2Transform)

# Assigning a Num to a Name (line 131):
float_130430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 11), 'float')
# Getting the type of 'Log2Transform'
Log2Transform_130431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Log2Transform')
# Setting the type of the member 'base' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Log2Transform_130431, 'base', float_130430)
# Declaration of the 'InvertedLog2Transform' class
# Getting the type of 'InvertedLogTransformBase' (line 137)
InvertedLogTransformBase_130432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 28), 'InvertedLogTransformBase')

class InvertedLog2Transform(InvertedLogTransformBase_130432, ):

    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_localization', localization)
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedLog2Transform.inverted')
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InvertedLog2Transform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLog2Transform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to Log2Transform(...): (line 141)
        # Processing the call keyword arguments (line 141)
        kwargs_130434 = {}
        # Getting the type of 'Log2Transform' (line 141)
        Log2Transform_130433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'Log2Transform', False)
        # Calling Log2Transform(args, kwargs) (line 141)
        Log2Transform_call_result_130435 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), Log2Transform_130433, *[], **kwargs_130434)
        
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', Log2Transform_call_result_130435)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_130436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130436)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130436


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 137, 0, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLog2Transform.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InvertedLog2Transform' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'InvertedLog2Transform', InvertedLog2Transform)

# Assigning a Num to a Name (line 138):
float_130437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 11), 'float')
# Getting the type of 'InvertedLog2Transform'
InvertedLog2Transform_130438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedLog2Transform')
# Setting the type of the member 'base' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedLog2Transform_130438, 'base', float_130437)
# Declaration of the 'NaturalLogTransform' class
# Getting the type of 'LogTransformBase' (line 144)
LogTransformBase_130439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'LogTransformBase')

class NaturalLogTransform(LogTransformBase_130439, ):

    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_function_name', 'NaturalLogTransform.inverted')
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NaturalLogTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NaturalLogTransform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to InvertedNaturalLogTransform(...): (line 148)
        # Processing the call keyword arguments (line 148)
        kwargs_130441 = {}
        # Getting the type of 'InvertedNaturalLogTransform' (line 148)
        InvertedNaturalLogTransform_130440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'InvertedNaturalLogTransform', False)
        # Calling InvertedNaturalLogTransform(args, kwargs) (line 148)
        InvertedNaturalLogTransform_call_result_130442 = invoke(stypy.reporting.localization.Localization(__file__, 148, 15), InvertedNaturalLogTransform_130440, *[], **kwargs_130441)
        
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', InvertedNaturalLogTransform_call_result_130442)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_130443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130443)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130443


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 144, 0, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NaturalLogTransform.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NaturalLogTransform' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'NaturalLogTransform', NaturalLogTransform)

# Assigning a Attribute to a Name (line 145):
# Getting the type of 'np' (line 145)
np_130444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'np')
# Obtaining the member 'e' of a type (line 145)
e_130445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 11), np_130444, 'e')
# Getting the type of 'NaturalLogTransform'
NaturalLogTransform_130446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NaturalLogTransform')
# Setting the type of the member 'base' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NaturalLogTransform_130446, 'base', e_130445)
# Declaration of the 'InvertedNaturalLogTransform' class
# Getting the type of 'InvertedLogTransformBase' (line 151)
InvertedLogTransformBase_130447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'InvertedLogTransformBase')

class InvertedNaturalLogTransform(InvertedLogTransformBase_130447, ):

    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedNaturalLogTransform.inverted')
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InvertedNaturalLogTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedNaturalLogTransform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to NaturalLogTransform(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_130449 = {}
        # Getting the type of 'NaturalLogTransform' (line 155)
        NaturalLogTransform_130448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'NaturalLogTransform', False)
        # Calling NaturalLogTransform(args, kwargs) (line 155)
        NaturalLogTransform_call_result_130450 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), NaturalLogTransform_130448, *[], **kwargs_130449)
        
        # Assigning a type to the variable 'stypy_return_type' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'stypy_return_type', NaturalLogTransform_call_result_130450)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_130451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130451)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130451


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 151, 0, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedNaturalLogTransform.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InvertedNaturalLogTransform' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'InvertedNaturalLogTransform', InvertedNaturalLogTransform)

# Assigning a Attribute to a Name (line 152):
# Getting the type of 'np' (line 152)
np_130452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'np')
# Obtaining the member 'e' of a type (line 152)
e_130453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), np_130452, 'e')
# Getting the type of 'InvertedNaturalLogTransform'
InvertedNaturalLogTransform_130454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedNaturalLogTransform')
# Setting the type of the member 'base' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedNaturalLogTransform_130454, 'base', e_130453)
# Declaration of the 'LogTransform' class
# Getting the type of 'LogTransformBase' (line 158)
LogTransformBase_130455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'LogTransformBase')

class LogTransform(LogTransformBase_130455, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 159, 4, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogTransform.__init__', ['base', 'nonpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['base', 'nonpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'self' (line 160)
        self_130458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'self', False)
        # Getting the type of 'nonpos' (line 160)
        nonpos_130459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 40), 'nonpos', False)
        # Processing the call keyword arguments (line 160)
        kwargs_130460 = {}
        # Getting the type of 'LogTransformBase' (line 160)
        LogTransformBase_130456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'LogTransformBase', False)
        # Obtaining the member '__init__' of a type (line 160)
        init___130457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), LogTransformBase_130456, '__init__')
        # Calling __init__(args, kwargs) (line 160)
        init___call_result_130461 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), init___130457, *[self_130458, nonpos_130459], **kwargs_130460)
        
        
        # Assigning a Name to a Attribute (line 161):
        # Getting the type of 'base' (line 161)
        base_130462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'base')
        # Getting the type of 'self' (line 161)
        self_130463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self')
        # Setting the type of the member 'base' of a type (line 161)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), self_130463, 'base', base_130462)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
        LogTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogTransform.inverted.__dict__.__setitem__('stypy_function_name', 'LogTransform.inverted')
        LogTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        LogTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogTransform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to InvertedLogTransform(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'self' (line 164)
        self_130465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 36), 'self', False)
        # Obtaining the member 'base' of a type (line 164)
        base_130466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 36), self_130465, 'base')
        # Processing the call keyword arguments (line 164)
        kwargs_130467 = {}
        # Getting the type of 'InvertedLogTransform' (line 164)
        InvertedLogTransform_130464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'InvertedLogTransform', False)
        # Calling InvertedLogTransform(args, kwargs) (line 164)
        InvertedLogTransform_call_result_130468 = invoke(stypy.reporting.localization.Localization(__file__, 164, 15), InvertedLogTransform_130464, *[base_130466], **kwargs_130467)
        
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', InvertedLogTransform_call_result_130468)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_130469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130469)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130469


# Assigning a type to the variable 'LogTransform' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'LogTransform', LogTransform)
# Declaration of the 'InvertedLogTransform' class
# Getting the type of 'InvertedLogTransformBase' (line 167)
InvertedLogTransformBase_130470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'InvertedLogTransformBase')

class InvertedLogTransform(InvertedLogTransformBase_130470, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLogTransform.__init__', ['base'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['base'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'self' (line 169)
        self_130473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 42), 'self', False)
        # Processing the call keyword arguments (line 169)
        kwargs_130474 = {}
        # Getting the type of 'InvertedLogTransformBase' (line 169)
        InvertedLogTransformBase_130471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'InvertedLogTransformBase', False)
        # Obtaining the member '__init__' of a type (line 169)
        init___130472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), InvertedLogTransformBase_130471, '__init__')
        # Calling __init__(args, kwargs) (line 169)
        init___call_result_130475 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), init___130472, *[self_130473], **kwargs_130474)
        
        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of 'base' (line 170)
        base_130476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'base')
        # Getting the type of 'self' (line 170)
        self_130477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member 'base' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_130477, 'base', base_130476)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedLogTransform.inverted')
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InvertedLogTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedLogTransform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to LogTransform(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'self' (line 173)
        self_130479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 28), 'self', False)
        # Obtaining the member 'base' of a type (line 173)
        base_130480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 28), self_130479, 'base')
        # Processing the call keyword arguments (line 173)
        kwargs_130481 = {}
        # Getting the type of 'LogTransform' (line 173)
        LogTransform_130478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'LogTransform', False)
        # Calling LogTransform(args, kwargs) (line 173)
        LogTransform_call_result_130482 = invoke(stypy.reporting.localization.Localization(__file__, 173, 15), LogTransform_130478, *[base_130480], **kwargs_130481)
        
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', LogTransform_call_result_130482)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_130483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130483)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130483


# Assigning a type to the variable 'InvertedLogTransform' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'InvertedLogTransform', InvertedLogTransform)
# Declaration of the 'LogScale' class
# Getting the type of 'ScaleBase' (line 176)
ScaleBase_130484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'ScaleBase')

class LogScale(ScaleBase_130484, ):
    unicode_130485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, (-1)), 'unicode', u'\n    A standard logarithmic scale.  Care is taken so non-positive\n    values are not plotted.\n\n    For computational efficiency (to push as much as possible to Numpy\n    C code in the common cases), this scale provides different\n    transforms depending on the base of the logarithm:\n\n       - base 10 (:class:`Log10Transform`)\n       - base 2 (:class:`Log2Transform`)\n       - base e (:class:`NaturalLogTransform`)\n       - arbitrary base (:class:`LogTransform`)\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogScale.__init__', ['axis'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_130486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'unicode', u"\n        *basex*/*basey*:\n           The base of the logarithm\n\n        *nonposx*/*nonposy*: ['mask' | 'clip' ]\n          non-positive values in *x* or *y* can be masked as\n          invalid, or clipped to a very small positive number\n\n        *subsx*/*subsy*:\n           Where to place the subticks between each major tick.\n           Should be a sequence of integers.  For example, in a log10\n           scale: ``[2, 3, 4, 5, 6, 7, 8, 9]``\n\n           will place 8 logarithmically spaced minor ticks between\n           each major tick.\n        ")
        
        
        # Getting the type of 'axis' (line 220)
        axis_130487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'axis')
        # Obtaining the member 'axis_name' of a type (line 220)
        axis_name_130488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 11), axis_130487, 'axis_name')
        unicode_130489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 29), 'unicode', u'x')
        # Applying the binary operator '==' (line 220)
        result_eq_130490 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 11), '==', axis_name_130488, unicode_130489)
        
        # Testing the type of an if condition (line 220)
        if_condition_130491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 8), result_eq_130490)
        # Assigning a type to the variable 'if_condition_130491' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'if_condition_130491', if_condition_130491)
        # SSA begins for if statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 221):
        
        # Call to pop(...): (line 221)
        # Processing the call arguments (line 221)
        unicode_130494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 30), 'unicode', u'basex')
        float_130495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 39), 'float')
        # Processing the call keyword arguments (line 221)
        kwargs_130496 = {}
        # Getting the type of 'kwargs' (line 221)
        kwargs_130492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 221)
        pop_130493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 19), kwargs_130492, 'pop')
        # Calling pop(args, kwargs) (line 221)
        pop_call_result_130497 = invoke(stypy.reporting.localization.Localization(__file__, 221, 19), pop_130493, *[unicode_130494, float_130495], **kwargs_130496)
        
        # Assigning a type to the variable 'base' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'base', pop_call_result_130497)
        
        # Assigning a Call to a Name (line 222):
        
        # Call to pop(...): (line 222)
        # Processing the call arguments (line 222)
        unicode_130500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 30), 'unicode', u'subsx')
        # Getting the type of 'None' (line 222)
        None_130501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'None', False)
        # Processing the call keyword arguments (line 222)
        kwargs_130502 = {}
        # Getting the type of 'kwargs' (line 222)
        kwargs_130498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 222)
        pop_130499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 19), kwargs_130498, 'pop')
        # Calling pop(args, kwargs) (line 222)
        pop_call_result_130503 = invoke(stypy.reporting.localization.Localization(__file__, 222, 19), pop_130499, *[unicode_130500, None_130501], **kwargs_130502)
        
        # Assigning a type to the variable 'subs' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'subs', pop_call_result_130503)
        
        # Assigning a Call to a Name (line 223):
        
        # Call to pop(...): (line 223)
        # Processing the call arguments (line 223)
        unicode_130506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 32), 'unicode', u'nonposx')
        unicode_130507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 43), 'unicode', u'mask')
        # Processing the call keyword arguments (line 223)
        kwargs_130508 = {}
        # Getting the type of 'kwargs' (line 223)
        kwargs_130504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 223)
        pop_130505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 21), kwargs_130504, 'pop')
        # Calling pop(args, kwargs) (line 223)
        pop_call_result_130509 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), pop_130505, *[unicode_130506, unicode_130507], **kwargs_130508)
        
        # Assigning a type to the variable 'nonpos' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'nonpos', pop_call_result_130509)
        # SSA branch for the else part of an if statement (line 220)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 225):
        
        # Call to pop(...): (line 225)
        # Processing the call arguments (line 225)
        unicode_130512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 30), 'unicode', u'basey')
        float_130513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 39), 'float')
        # Processing the call keyword arguments (line 225)
        kwargs_130514 = {}
        # Getting the type of 'kwargs' (line 225)
        kwargs_130510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 225)
        pop_130511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 19), kwargs_130510, 'pop')
        # Calling pop(args, kwargs) (line 225)
        pop_call_result_130515 = invoke(stypy.reporting.localization.Localization(__file__, 225, 19), pop_130511, *[unicode_130512, float_130513], **kwargs_130514)
        
        # Assigning a type to the variable 'base' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'base', pop_call_result_130515)
        
        # Assigning a Call to a Name (line 226):
        
        # Call to pop(...): (line 226)
        # Processing the call arguments (line 226)
        unicode_130518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 30), 'unicode', u'subsy')
        # Getting the type of 'None' (line 226)
        None_130519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 39), 'None', False)
        # Processing the call keyword arguments (line 226)
        kwargs_130520 = {}
        # Getting the type of 'kwargs' (line 226)
        kwargs_130516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 226)
        pop_130517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 19), kwargs_130516, 'pop')
        # Calling pop(args, kwargs) (line 226)
        pop_call_result_130521 = invoke(stypy.reporting.localization.Localization(__file__, 226, 19), pop_130517, *[unicode_130518, None_130519], **kwargs_130520)
        
        # Assigning a type to the variable 'subs' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'subs', pop_call_result_130521)
        
        # Assigning a Call to a Name (line 227):
        
        # Call to pop(...): (line 227)
        # Processing the call arguments (line 227)
        unicode_130524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 32), 'unicode', u'nonposy')
        unicode_130525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 43), 'unicode', u'mask')
        # Processing the call keyword arguments (line 227)
        kwargs_130526 = {}
        # Getting the type of 'kwargs' (line 227)
        kwargs_130522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 227)
        pop_130523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 21), kwargs_130522, 'pop')
        # Calling pop(args, kwargs) (line 227)
        pop_call_result_130527 = invoke(stypy.reporting.localization.Localization(__file__, 227, 21), pop_130523, *[unicode_130524, unicode_130525], **kwargs_130526)
        
        # Assigning a type to the variable 'nonpos' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'nonpos', pop_call_result_130527)
        # SSA join for if statement (line 220)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'nonpos' (line 229)
        nonpos_130528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'nonpos')
        
        # Obtaining an instance of the builtin type 'list' (line 229)
        list_130529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 229)
        # Adding element type (line 229)
        unicode_130530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 26), 'unicode', u'mask')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 25), list_130529, unicode_130530)
        # Adding element type (line 229)
        unicode_130531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 34), 'unicode', u'clip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 25), list_130529, unicode_130531)
        
        # Applying the binary operator 'notin' (line 229)
        result_contains_130532 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'notin', nonpos_130528, list_130529)
        
        # Testing the type of an if condition (line 229)
        if_condition_130533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), result_contains_130532)
        # Assigning a type to the variable 'if_condition_130533' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_130533', if_condition_130533)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 230)
        # Processing the call arguments (line 230)
        unicode_130535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 29), 'unicode', u"nonposx, nonposy kwarg must be 'mask' or 'clip'")
        # Processing the call keyword arguments (line 230)
        kwargs_130536 = {}
        # Getting the type of 'ValueError' (line 230)
        ValueError_130534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 230)
        ValueError_call_result_130537 = invoke(stypy.reporting.localization.Localization(__file__, 230, 18), ValueError_130534, *[unicode_130535], **kwargs_130536)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 230, 12), ValueError_call_result_130537, 'raise parameter', BaseException)
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'base' (line 232)
        base_130538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'base')
        float_130539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 19), 'float')
        # Applying the binary operator '==' (line 232)
        result_eq_130540 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 11), '==', base_130538, float_130539)
        
        # Testing the type of an if condition (line 232)
        if_condition_130541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), result_eq_130540)
        # Assigning a type to the variable 'if_condition_130541' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_130541', if_condition_130541)
        # SSA begins for if statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 233):
        
        # Call to Log10Transform(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'nonpos' (line 233)
        nonpos_130544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 50), 'nonpos', False)
        # Processing the call keyword arguments (line 233)
        kwargs_130545 = {}
        # Getting the type of 'self' (line 233)
        self_130542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 30), 'self', False)
        # Obtaining the member 'Log10Transform' of a type (line 233)
        Log10Transform_130543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 30), self_130542, 'Log10Transform')
        # Calling Log10Transform(args, kwargs) (line 233)
        Log10Transform_call_result_130546 = invoke(stypy.reporting.localization.Localization(__file__, 233, 30), Log10Transform_130543, *[nonpos_130544], **kwargs_130545)
        
        # Getting the type of 'self' (line 233)
        self_130547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'self')
        # Setting the type of the member '_transform' of a type (line 233)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 12), self_130547, '_transform', Log10Transform_call_result_130546)
        # SSA branch for the else part of an if statement (line 232)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'base' (line 234)
        base_130548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 13), 'base')
        float_130549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'float')
        # Applying the binary operator '==' (line 234)
        result_eq_130550 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 13), '==', base_130548, float_130549)
        
        # Testing the type of an if condition (line 234)
        if_condition_130551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 13), result_eq_130550)
        # Assigning a type to the variable 'if_condition_130551' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 13), 'if_condition_130551', if_condition_130551)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 235):
        
        # Call to Log2Transform(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'nonpos' (line 235)
        nonpos_130554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), 'nonpos', False)
        # Processing the call keyword arguments (line 235)
        kwargs_130555 = {}
        # Getting the type of 'self' (line 235)
        self_130552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'self', False)
        # Obtaining the member 'Log2Transform' of a type (line 235)
        Log2Transform_130553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 30), self_130552, 'Log2Transform')
        # Calling Log2Transform(args, kwargs) (line 235)
        Log2Transform_call_result_130556 = invoke(stypy.reporting.localization.Localization(__file__, 235, 30), Log2Transform_130553, *[nonpos_130554], **kwargs_130555)
        
        # Getting the type of 'self' (line 235)
        self_130557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self')
        # Setting the type of the member '_transform' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), self_130557, '_transform', Log2Transform_call_result_130556)
        # SSA branch for the else part of an if statement (line 234)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'base' (line 236)
        base_130558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'base')
        # Getting the type of 'np' (line 236)
        np_130559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 21), 'np')
        # Obtaining the member 'e' of a type (line 236)
        e_130560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 21), np_130559, 'e')
        # Applying the binary operator '==' (line 236)
        result_eq_130561 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 13), '==', base_130558, e_130560)
        
        # Testing the type of an if condition (line 236)
        if_condition_130562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 13), result_eq_130561)
        # Assigning a type to the variable 'if_condition_130562' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'if_condition_130562', if_condition_130562)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 237):
        
        # Call to NaturalLogTransform(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'nonpos' (line 237)
        nonpos_130565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 55), 'nonpos', False)
        # Processing the call keyword arguments (line 237)
        kwargs_130566 = {}
        # Getting the type of 'self' (line 237)
        self_130563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 30), 'self', False)
        # Obtaining the member 'NaturalLogTransform' of a type (line 237)
        NaturalLogTransform_130564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 30), self_130563, 'NaturalLogTransform')
        # Calling NaturalLogTransform(args, kwargs) (line 237)
        NaturalLogTransform_call_result_130567 = invoke(stypy.reporting.localization.Localization(__file__, 237, 30), NaturalLogTransform_130564, *[nonpos_130565], **kwargs_130566)
        
        # Getting the type of 'self' (line 237)
        self_130568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self')
        # Setting the type of the member '_transform' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_130568, '_transform', NaturalLogTransform_call_result_130567)
        # SSA branch for the else part of an if statement (line 236)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 239):
        
        # Call to LogTransform(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'base' (line 239)
        base_130571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 48), 'base', False)
        # Getting the type of 'nonpos' (line 239)
        nonpos_130572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 54), 'nonpos', False)
        # Processing the call keyword arguments (line 239)
        kwargs_130573 = {}
        # Getting the type of 'self' (line 239)
        self_130569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 30), 'self', False)
        # Obtaining the member 'LogTransform' of a type (line 239)
        LogTransform_130570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 30), self_130569, 'LogTransform')
        # Calling LogTransform(args, kwargs) (line 239)
        LogTransform_call_result_130574 = invoke(stypy.reporting.localization.Localization(__file__, 239, 30), LogTransform_130570, *[base_130571, nonpos_130572], **kwargs_130573)
        
        # Getting the type of 'self' (line 239)
        self_130575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'self')
        # Setting the type of the member '_transform' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), self_130575, '_transform', LogTransform_call_result_130574)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 232)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 241):
        # Getting the type of 'base' (line 241)
        base_130576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'base')
        # Getting the type of 'self' (line 241)
        self_130577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self')
        # Setting the type of the member 'base' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_130577, 'base', base_130576)
        
        # Assigning a Name to a Attribute (line 242):
        # Getting the type of 'subs' (line 242)
        subs_130578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'subs')
        # Getting the type of 'self' (line 242)
        self_130579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self')
        # Setting the type of the member 'subs' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_130579, 'subs', subs_130578)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_default_locators_and_formatters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_default_locators_and_formatters'
        module_type_store = module_type_store.open_function_context('set_default_locators_and_formatters', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_localization', localization)
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_function_name', 'LogScale.set_default_locators_and_formatters')
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogScale.set_default_locators_and_formatters', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_default_locators_and_formatters', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_default_locators_and_formatters(...)' code ##################

        unicode_130580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, (-1)), 'unicode', u'\n        Set the locators and formatters to specialized versions for\n        log scaling.\n        ')
        
        # Call to set_major_locator(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Call to LogLocator(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'self' (line 249)
        self_130584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'self', False)
        # Obtaining the member 'base' of a type (line 249)
        base_130585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 42), self_130584, 'base')
        # Processing the call keyword arguments (line 249)
        kwargs_130586 = {}
        # Getting the type of 'LogLocator' (line 249)
        LogLocator_130583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 31), 'LogLocator', False)
        # Calling LogLocator(args, kwargs) (line 249)
        LogLocator_call_result_130587 = invoke(stypy.reporting.localization.Localization(__file__, 249, 31), LogLocator_130583, *[base_130585], **kwargs_130586)
        
        # Processing the call keyword arguments (line 249)
        kwargs_130588 = {}
        # Getting the type of 'axis' (line 249)
        axis_130581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'axis', False)
        # Obtaining the member 'set_major_locator' of a type (line 249)
        set_major_locator_130582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), axis_130581, 'set_major_locator')
        # Calling set_major_locator(args, kwargs) (line 249)
        set_major_locator_call_result_130589 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), set_major_locator_130582, *[LogLocator_call_result_130587], **kwargs_130588)
        
        
        # Call to set_major_formatter(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to LogFormatterSciNotation(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'self' (line 250)
        self_130593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 57), 'self', False)
        # Obtaining the member 'base' of a type (line 250)
        base_130594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 57), self_130593, 'base')
        # Processing the call keyword arguments (line 250)
        kwargs_130595 = {}
        # Getting the type of 'LogFormatterSciNotation' (line 250)
        LogFormatterSciNotation_130592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 33), 'LogFormatterSciNotation', False)
        # Calling LogFormatterSciNotation(args, kwargs) (line 250)
        LogFormatterSciNotation_call_result_130596 = invoke(stypy.reporting.localization.Localization(__file__, 250, 33), LogFormatterSciNotation_130592, *[base_130594], **kwargs_130595)
        
        # Processing the call keyword arguments (line 250)
        kwargs_130597 = {}
        # Getting the type of 'axis' (line 250)
        axis_130590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'axis', False)
        # Obtaining the member 'set_major_formatter' of a type (line 250)
        set_major_formatter_130591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), axis_130590, 'set_major_formatter')
        # Calling set_major_formatter(args, kwargs) (line 250)
        set_major_formatter_call_result_130598 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), set_major_formatter_130591, *[LogFormatterSciNotation_call_result_130596], **kwargs_130597)
        
        
        # Call to set_minor_locator(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Call to LogLocator(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'self' (line 251)
        self_130602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 42), 'self', False)
        # Obtaining the member 'base' of a type (line 251)
        base_130603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 42), self_130602, 'base')
        # Getting the type of 'self' (line 251)
        self_130604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 53), 'self', False)
        # Obtaining the member 'subs' of a type (line 251)
        subs_130605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 53), self_130604, 'subs')
        # Processing the call keyword arguments (line 251)
        kwargs_130606 = {}
        # Getting the type of 'LogLocator' (line 251)
        LogLocator_130601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'LogLocator', False)
        # Calling LogLocator(args, kwargs) (line 251)
        LogLocator_call_result_130607 = invoke(stypy.reporting.localization.Localization(__file__, 251, 31), LogLocator_130601, *[base_130603, subs_130605], **kwargs_130606)
        
        # Processing the call keyword arguments (line 251)
        kwargs_130608 = {}
        # Getting the type of 'axis' (line 251)
        axis_130599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'axis', False)
        # Obtaining the member 'set_minor_locator' of a type (line 251)
        set_minor_locator_130600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), axis_130599, 'set_minor_locator')
        # Calling set_minor_locator(args, kwargs) (line 251)
        set_minor_locator_call_result_130609 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), set_minor_locator_130600, *[LogLocator_call_result_130607], **kwargs_130608)
        
        
        # Call to set_minor_formatter(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Call to LogFormatterSciNotation(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'self' (line 253)
        self_130613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 36), 'self', False)
        # Obtaining the member 'base' of a type (line 253)
        base_130614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 36), self_130613, 'base')
        # Processing the call keyword arguments (line 253)
        
        # Getting the type of 'self' (line 254)
        self_130615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 51), 'self', False)
        # Obtaining the member 'subs' of a type (line 254)
        subs_130616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 51), self_130615, 'subs')
        # Getting the type of 'None' (line 254)
        None_130617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 68), 'None', False)
        # Applying the binary operator 'isnot' (line 254)
        result_is_not_130618 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 51), 'isnot', subs_130616, None_130617)
        
        keyword_130619 = result_is_not_130618
        kwargs_130620 = {'labelOnlyBase': keyword_130619}
        # Getting the type of 'LogFormatterSciNotation' (line 253)
        LogFormatterSciNotation_130612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'LogFormatterSciNotation', False)
        # Calling LogFormatterSciNotation(args, kwargs) (line 253)
        LogFormatterSciNotation_call_result_130621 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), LogFormatterSciNotation_130612, *[base_130614], **kwargs_130620)
        
        # Processing the call keyword arguments (line 252)
        kwargs_130622 = {}
        # Getting the type of 'axis' (line 252)
        axis_130610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'axis', False)
        # Obtaining the member 'set_minor_formatter' of a type (line 252)
        set_minor_formatter_130611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), axis_130610, 'set_minor_formatter')
        # Calling set_minor_formatter(args, kwargs) (line 252)
        set_minor_formatter_call_result_130623 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), set_minor_formatter_130611, *[LogFormatterSciNotation_call_result_130621], **kwargs_130622)
        
        
        # ################# End of 'set_default_locators_and_formatters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_default_locators_and_formatters' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_130624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130624)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_default_locators_and_formatters'
        return stypy_return_type_130624


    @norecursion
    def get_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_transform'
        module_type_store = module_type_store.open_function_context('get_transform', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogScale.get_transform.__dict__.__setitem__('stypy_localization', localization)
        LogScale.get_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogScale.get_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogScale.get_transform.__dict__.__setitem__('stypy_function_name', 'LogScale.get_transform')
        LogScale.get_transform.__dict__.__setitem__('stypy_param_names_list', [])
        LogScale.get_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogScale.get_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogScale.get_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogScale.get_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogScale.get_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogScale.get_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogScale.get_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_transform(...)' code ##################

        unicode_130625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, (-1)), 'unicode', u'\n        Return a :class:`~matplotlib.transforms.Transform` instance\n        appropriate for the given logarithm base.\n        ')
        # Getting the type of 'self' (line 261)
        self_130626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'self')
        # Obtaining the member '_transform' of a type (line 261)
        _transform_130627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), self_130626, '_transform')
        # Assigning a type to the variable 'stypy_return_type' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'stypy_return_type', _transform_130627)
        
        # ################# End of 'get_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_130628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130628)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_transform'
        return stypy_return_type_130628


    @norecursion
    def limit_range_for_scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'limit_range_for_scale'
        module_type_store = module_type_store.open_function_context('limit_range_for_scale', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_localization', localization)
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_function_name', 'LogScale.limit_range_for_scale')
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_param_names_list', ['vmin', 'vmax', 'minpos'])
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogScale.limit_range_for_scale.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogScale.limit_range_for_scale', ['vmin', 'vmax', 'minpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'limit_range_for_scale', localization, ['vmin', 'vmax', 'minpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'limit_range_for_scale(...)' code ##################

        unicode_130629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, (-1)), 'unicode', u'\n        Limit the domain to positive values.\n        ')
        
        
        
        # Call to isfinite(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'minpos' (line 267)
        minpos_130632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 27), 'minpos', False)
        # Processing the call keyword arguments (line 267)
        kwargs_130633 = {}
        # Getting the type of 'np' (line 267)
        np_130630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 267)
        isfinite_130631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 15), np_130630, 'isfinite')
        # Calling isfinite(args, kwargs) (line 267)
        isfinite_call_result_130634 = invoke(stypy.reporting.localization.Localization(__file__, 267, 15), isfinite_130631, *[minpos_130632], **kwargs_130633)
        
        # Applying the 'not' unary operator (line 267)
        result_not__130635 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 11), 'not', isfinite_call_result_130634)
        
        # Testing the type of an if condition (line 267)
        if_condition_130636 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 8), result_not__130635)
        # Assigning a type to the variable 'if_condition_130636' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'if_condition_130636', if_condition_130636)
        # SSA begins for if statement (line 267)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 268):
        float_130637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 21), 'float')
        # Assigning a type to the variable 'minpos' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'minpos', float_130637)
        # SSA join for if statement (line 267)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 271)
        tuple_130638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 271)
        # Adding element type (line 271)
        
        
        # Getting the type of 'vmin' (line 271)
        vmin_130639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'vmin')
        int_130640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 34), 'int')
        # Applying the binary operator '<=' (line 271)
        result_le_130641 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 26), '<=', vmin_130639, int_130640)
        
        # Testing the type of an if expression (line 271)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 16), result_le_130641)
        # SSA begins for if expression (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'minpos' (line 271)
        minpos_130642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'minpos')
        # SSA branch for the else part of an if expression (line 271)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'vmin' (line 271)
        vmin_130643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 41), 'vmin')
        # SSA join for if expression (line 271)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_130644 = union_type.UnionType.add(minpos_130642, vmin_130643)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 16), tuple_130638, if_exp_130644)
        # Adding element type (line 271)
        
        
        # Getting the type of 'vmax' (line 272)
        vmax_130645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 26), 'vmax')
        int_130646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 34), 'int')
        # Applying the binary operator '<=' (line 272)
        result_le_130647 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 26), '<=', vmax_130645, int_130646)
        
        # Testing the type of an if expression (line 272)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 16), result_le_130647)
        # SSA begins for if expression (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'minpos' (line 272)
        minpos_130648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'minpos')
        # SSA branch for the else part of an if expression (line 272)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'vmax' (line 272)
        vmax_130649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 41), 'vmax')
        # SSA join for if expression (line 272)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_130650 = union_type.UnionType.add(minpos_130648, vmax_130649)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 16), tuple_130638, if_exp_130650)
        
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type', tuple_130638)
        
        # ################# End of 'limit_range_for_scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'limit_range_for_scale' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_130651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130651)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'limit_range_for_scale'
        return stypy_return_type_130651


# Assigning a type to the variable 'LogScale' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'LogScale', LogScale)

# Assigning a Str to a Name (line 190):
unicode_130652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 11), 'unicode', u'log')
# Getting the type of 'LogScale'
LogScale_130653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130653, 'name', unicode_130652)

# Assigning a Name to a Name (line 193):
# Getting the type of 'LogScale'
LogScale_130654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'LogTransformBase' of a type
LogTransformBase_130655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130654, 'LogTransformBase')
# Getting the type of 'LogScale'
LogScale_130656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'LogTransformBase' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130656, 'LogTransformBase', LogTransformBase_130655)

# Assigning a Name to a Name (line 194):
# Getting the type of 'LogScale'
LogScale_130657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'Log10Transform' of a type
Log10Transform_130658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130657, 'Log10Transform')
# Getting the type of 'LogScale'
LogScale_130659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'Log10Transform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130659, 'Log10Transform', Log10Transform_130658)

# Assigning a Name to a Name (line 195):
# Getting the type of 'LogScale'
LogScale_130660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'InvertedLog10Transform' of a type
InvertedLog10Transform_130661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130660, 'InvertedLog10Transform')
# Getting the type of 'LogScale'
LogScale_130662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'InvertedLog10Transform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130662, 'InvertedLog10Transform', InvertedLog10Transform_130661)

# Assigning a Name to a Name (line 196):
# Getting the type of 'LogScale'
LogScale_130663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'Log2Transform' of a type
Log2Transform_130664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130663, 'Log2Transform')
# Getting the type of 'LogScale'
LogScale_130665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'Log2Transform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130665, 'Log2Transform', Log2Transform_130664)

# Assigning a Name to a Name (line 197):
# Getting the type of 'LogScale'
LogScale_130666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'InvertedLog2Transform' of a type
InvertedLog2Transform_130667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130666, 'InvertedLog2Transform')
# Getting the type of 'LogScale'
LogScale_130668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'InvertedLog2Transform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130668, 'InvertedLog2Transform', InvertedLog2Transform_130667)

# Assigning a Name to a Name (line 198):
# Getting the type of 'LogScale'
LogScale_130669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'NaturalLogTransform' of a type
NaturalLogTransform_130670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130669, 'NaturalLogTransform')
# Getting the type of 'LogScale'
LogScale_130671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'NaturalLogTransform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130671, 'NaturalLogTransform', NaturalLogTransform_130670)

# Assigning a Name to a Name (line 199):
# Getting the type of 'LogScale'
LogScale_130672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'InvertedNaturalLogTransform' of a type
InvertedNaturalLogTransform_130673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130672, 'InvertedNaturalLogTransform')
# Getting the type of 'LogScale'
LogScale_130674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'InvertedNaturalLogTransform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130674, 'InvertedNaturalLogTransform', InvertedNaturalLogTransform_130673)

# Assigning a Name to a Name (line 200):
# Getting the type of 'LogScale'
LogScale_130675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'LogTransform' of a type
LogTransform_130676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130675, 'LogTransform')
# Getting the type of 'LogScale'
LogScale_130677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'LogTransform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130677, 'LogTransform', LogTransform_130676)

# Assigning a Name to a Name (line 201):
# Getting the type of 'LogScale'
LogScale_130678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Obtaining the member 'InvertedLogTransform' of a type
InvertedLogTransform_130679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130678, 'InvertedLogTransform')
# Getting the type of 'LogScale'
LogScale_130680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogScale')
# Setting the type of the member 'InvertedLogTransform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogScale_130680, 'InvertedLogTransform', InvertedLogTransform_130679)
# Declaration of the 'SymmetricalLogTransform' class
# Getting the type of 'Transform' (line 275)
Transform_130681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 30), 'Transform')

class SymmetricalLogTransform(Transform_130681, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SymmetricalLogTransform.__init__', ['base', 'linthresh', 'linscale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['base', 'linthresh', 'linscale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'self' (line 282)
        self_130684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 'self', False)
        # Processing the call keyword arguments (line 282)
        kwargs_130685 = {}
        # Getting the type of 'Transform' (line 282)
        Transform_130682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'Transform', False)
        # Obtaining the member '__init__' of a type (line 282)
        init___130683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), Transform_130682, '__init__')
        # Calling __init__(args, kwargs) (line 282)
        init___call_result_130686 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), init___130683, *[self_130684], **kwargs_130685)
        
        
        # Assigning a Name to a Attribute (line 283):
        # Getting the type of 'base' (line 283)
        base_130687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'base')
        # Getting the type of 'self' (line 283)
        self_130688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self')
        # Setting the type of the member 'base' of a type (line 283)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_130688, 'base', base_130687)
        
        # Assigning a Name to a Attribute (line 284):
        # Getting the type of 'linthresh' (line 284)
        linthresh_130689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'linthresh')
        # Getting the type of 'self' (line 284)
        self_130690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self')
        # Setting the type of the member 'linthresh' of a type (line 284)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_130690, 'linthresh', linthresh_130689)
        
        # Assigning a Name to a Attribute (line 285):
        # Getting the type of 'linscale' (line 285)
        linscale_130691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'linscale')
        # Getting the type of 'self' (line 285)
        self_130692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self')
        # Setting the type of the member 'linscale' of a type (line 285)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_130692, 'linscale', linscale_130691)
        
        # Assigning a BinOp to a Attribute (line 286):
        # Getting the type of 'linscale' (line 286)
        linscale_130693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 30), 'linscale')
        float_130694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 42), 'float')
        # Getting the type of 'self' (line 286)
        self_130695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 48), 'self')
        # Obtaining the member 'base' of a type (line 286)
        base_130696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 48), self_130695, 'base')
        int_130697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 61), 'int')
        # Applying the binary operator '**' (line 286)
        result_pow_130698 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 48), '**', base_130696, int_130697)
        
        # Applying the binary operator '-' (line 286)
        result_sub_130699 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 42), '-', float_130694, result_pow_130698)
        
        # Applying the binary operator 'div' (line 286)
        result_div_130700 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 30), 'div', linscale_130693, result_sub_130699)
        
        # Getting the type of 'self' (line 286)
        self_130701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self')
        # Setting the type of the member '_linscale_adj' of a type (line 286)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_130701, '_linscale_adj', result_div_130700)
        
        # Assigning a Call to a Attribute (line 287):
        
        # Call to log(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'base' (line 287)
        base_130704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'base', False)
        # Processing the call keyword arguments (line 287)
        kwargs_130705 = {}
        # Getting the type of 'np' (line 287)
        np_130702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'np', False)
        # Obtaining the member 'log' of a type (line 287)
        log_130703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 25), np_130702, 'log')
        # Calling log(args, kwargs) (line 287)
        log_call_result_130706 = invoke(stypy.reporting.localization.Localization(__file__, 287, 25), log_130703, *[base_130704], **kwargs_130705)
        
        # Getting the type of 'self' (line 287)
        self_130707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self')
        # Setting the type of the member '_log_base' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_130707, '_log_base', log_call_result_130706)
        
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
        module_type_store = module_type_store.open_function_context('transform_non_affine', 289, 4, False)
        # Assigning a type to the variable 'self' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'SymmetricalLogTransform.transform_non_affine')
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['a'])
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SymmetricalLogTransform.transform_non_affine', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform_non_affine', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform_non_affine(...)' code ##################

        
        # Assigning a Call to a Name (line 290):
        
        # Call to sign(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'a' (line 290)
        a_130710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 23), 'a', False)
        # Processing the call keyword arguments (line 290)
        kwargs_130711 = {}
        # Getting the type of 'np' (line 290)
        np_130708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'np', False)
        # Obtaining the member 'sign' of a type (line 290)
        sign_130709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 15), np_130708, 'sign')
        # Calling sign(args, kwargs) (line 290)
        sign_call_result_130712 = invoke(stypy.reporting.localization.Localization(__file__, 290, 15), sign_130709, *[a_130710], **kwargs_130711)
        
        # Assigning a type to the variable 'sign' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'sign', sign_call_result_130712)
        
        # Assigning a Call to a Name (line 291):
        
        # Call to masked_inside(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'a' (line 291)
        a_130715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 34), 'a', False)
        
        # Getting the type of 'self' (line 292)
        self_130716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 35), 'self', False)
        # Obtaining the member 'linthresh' of a type (line 292)
        linthresh_130717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 35), self_130716, 'linthresh')
        # Applying the 'usub' unary operator (line 292)
        result___neg___130718 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 34), 'usub', linthresh_130717)
        
        # Getting the type of 'self' (line 293)
        self_130719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 34), 'self', False)
        # Obtaining the member 'linthresh' of a type (line 293)
        linthresh_130720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 34), self_130719, 'linthresh')
        # Processing the call keyword arguments (line 291)
        # Getting the type of 'False' (line 294)
        False_130721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'False', False)
        keyword_130722 = False_130721
        kwargs_130723 = {'copy': keyword_130722}
        # Getting the type of 'ma' (line 291)
        ma_130713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 17), 'ma', False)
        # Obtaining the member 'masked_inside' of a type (line 291)
        masked_inside_130714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 17), ma_130713, 'masked_inside')
        # Calling masked_inside(args, kwargs) (line 291)
        masked_inside_call_result_130724 = invoke(stypy.reporting.localization.Localization(__file__, 291, 17), masked_inside_130714, *[a_130715, result___neg___130718, linthresh_130720], **kwargs_130723)
        
        # Assigning a type to the variable 'masked' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'masked', masked_inside_call_result_130724)
        
        # Assigning a BinOp to a Name (line 295):
        # Getting the type of 'sign' (line 295)
        sign_130725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'sign')
        # Getting the type of 'self' (line 295)
        self_130726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'self')
        # Obtaining the member 'linthresh' of a type (line 295)
        linthresh_130727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 21), self_130726, 'linthresh')
        # Applying the binary operator '*' (line 295)
        result_mul_130728 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 14), '*', sign_130725, linthresh_130727)
        
        # Getting the type of 'self' (line 296)
        self_130729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'self')
        # Obtaining the member '_linscale_adj' of a type (line 296)
        _linscale_adj_130730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), self_130729, '_linscale_adj')
        
        # Call to log(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Call to abs(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'masked' (line 297)
        masked_130735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 26), 'masked', False)
        # Processing the call keyword arguments (line 297)
        kwargs_130736 = {}
        # Getting the type of 'np' (line 297)
        np_130733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'np', False)
        # Obtaining the member 'abs' of a type (line 297)
        abs_130734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 19), np_130733, 'abs')
        # Calling abs(args, kwargs) (line 297)
        abs_call_result_130737 = invoke(stypy.reporting.localization.Localization(__file__, 297, 19), abs_130734, *[masked_130735], **kwargs_130736)
        
        # Getting the type of 'self' (line 297)
        self_130738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 36), 'self', False)
        # Obtaining the member 'linthresh' of a type (line 297)
        linthresh_130739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 36), self_130738, 'linthresh')
        # Applying the binary operator 'div' (line 297)
        result_div_130740 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 19), 'div', abs_call_result_130737, linthresh_130739)
        
        # Processing the call keyword arguments (line 297)
        kwargs_130741 = {}
        # Getting the type of 'ma' (line 297)
        ma_130731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'ma', False)
        # Obtaining the member 'log' of a type (line 297)
        log_130732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 12), ma_130731, 'log')
        # Calling log(args, kwargs) (line 297)
        log_call_result_130742 = invoke(stypy.reporting.localization.Localization(__file__, 297, 12), log_130732, *[result_div_130740], **kwargs_130741)
        
        # Getting the type of 'self' (line 297)
        self_130743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 54), 'self')
        # Obtaining the member '_log_base' of a type (line 297)
        _log_base_130744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 54), self_130743, '_log_base')
        # Applying the binary operator 'div' (line 297)
        result_div_130745 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 12), 'div', log_call_result_130742, _log_base_130744)
        
        # Applying the binary operator '+' (line 296)
        result_add_130746 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 12), '+', _linscale_adj_130730, result_div_130745)
        
        # Applying the binary operator '*' (line 295)
        result_mul_130747 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 36), '*', result_mul_130728, result_add_130746)
        
        # Assigning a type to the variable 'log' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'log', result_mul_130747)
        
        
        # Call to any(...): (line 298)
        # Processing the call keyword arguments (line 298)
        kwargs_130751 = {}
        # Getting the type of 'masked' (line 298)
        masked_130748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'masked', False)
        # Obtaining the member 'mask' of a type (line 298)
        mask_130749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), masked_130748, 'mask')
        # Obtaining the member 'any' of a type (line 298)
        any_130750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), mask_130749, 'any')
        # Calling any(args, kwargs) (line 298)
        any_call_result_130752 = invoke(stypy.reporting.localization.Localization(__file__, 298, 11), any_130750, *[], **kwargs_130751)
        
        # Testing the type of an if condition (line 298)
        if_condition_130753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), any_call_result_130752)
        # Assigning a type to the variable 'if_condition_130753' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_130753', if_condition_130753)
        # SSA begins for if statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to where(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'masked' (line 299)
        masked_130756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 'masked', False)
        # Obtaining the member 'mask' of a type (line 299)
        mask_130757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 28), masked_130756, 'mask')
        # Getting the type of 'a' (line 299)
        a_130758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 41), 'a', False)
        # Getting the type of 'self' (line 299)
        self_130759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 45), 'self', False)
        # Obtaining the member '_linscale_adj' of a type (line 299)
        _linscale_adj_130760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 45), self_130759, '_linscale_adj')
        # Applying the binary operator '*' (line 299)
        result_mul_130761 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 41), '*', a_130758, _linscale_adj_130760)
        
        # Getting the type of 'log' (line 299)
        log_130762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 65), 'log', False)
        # Processing the call keyword arguments (line 299)
        kwargs_130763 = {}
        # Getting the type of 'ma' (line 299)
        ma_130754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 19), 'ma', False)
        # Obtaining the member 'where' of a type (line 299)
        where_130755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 19), ma_130754, 'where')
        # Calling where(args, kwargs) (line 299)
        where_call_result_130764 = invoke(stypy.reporting.localization.Localization(__file__, 299, 19), where_130755, *[mask_130757, result_mul_130761, log_130762], **kwargs_130763)
        
        # Assigning a type to the variable 'stypy_return_type' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'stypy_return_type', where_call_result_130764)
        # SSA branch for the else part of an if statement (line 298)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'log' (line 301)
        log_130765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 19), 'log')
        # Assigning a type to the variable 'stypy_return_type' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'stypy_return_type', log_130765)
        # SSA join for if statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'transform_non_affine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform_non_affine' in the type store
        # Getting the type of 'stypy_return_type' (line 289)
        stypy_return_type_130766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform_non_affine'
        return stypy_return_type_130766


    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_function_name', 'SymmetricalLogTransform.inverted')
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SymmetricalLogTransform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to InvertedSymmetricalLogTransform(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'self' (line 304)
        self_130768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 47), 'self', False)
        # Obtaining the member 'base' of a type (line 304)
        base_130769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 47), self_130768, 'base')
        # Getting the type of 'self' (line 304)
        self_130770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 58), 'self', False)
        # Obtaining the member 'linthresh' of a type (line 304)
        linthresh_130771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 58), self_130770, 'linthresh')
        # Getting the type of 'self' (line 305)
        self_130772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 47), 'self', False)
        # Obtaining the member 'linscale' of a type (line 305)
        linscale_130773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 47), self_130772, 'linscale')
        # Processing the call keyword arguments (line 304)
        kwargs_130774 = {}
        # Getting the type of 'InvertedSymmetricalLogTransform' (line 304)
        InvertedSymmetricalLogTransform_130767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'InvertedSymmetricalLogTransform', False)
        # Calling InvertedSymmetricalLogTransform(args, kwargs) (line 304)
        InvertedSymmetricalLogTransform_call_result_130775 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), InvertedSymmetricalLogTransform_130767, *[base_130769, linthresh_130771, linscale_130773], **kwargs_130774)
        
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', InvertedSymmetricalLogTransform_call_result_130775)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_130776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130776)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130776


# Assigning a type to the variable 'SymmetricalLogTransform' (line 275)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'SymmetricalLogTransform', SymmetricalLogTransform)

# Assigning a Num to a Name (line 276):
int_130777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 17), 'int')
# Getting the type of 'SymmetricalLogTransform'
SymmetricalLogTransform_130778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogTransform')
# Setting the type of the member 'input_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogTransform_130778, 'input_dims', int_130777)

# Assigning a Num to a Name (line 277):
int_130779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 18), 'int')
# Getting the type of 'SymmetricalLogTransform'
SymmetricalLogTransform_130780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogTransform')
# Setting the type of the member 'output_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogTransform_130780, 'output_dims', int_130779)

# Assigning a Name to a Name (line 278):
# Getting the type of 'True' (line 278)
True_130781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'True')
# Getting the type of 'SymmetricalLogTransform'
SymmetricalLogTransform_130782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogTransform')
# Setting the type of the member 'is_separable' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogTransform_130782, 'is_separable', True_130781)

# Assigning a Name to a Name (line 279):
# Getting the type of 'True' (line 279)
True_130783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 18), 'True')
# Getting the type of 'SymmetricalLogTransform'
SymmetricalLogTransform_130784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogTransform')
# Setting the type of the member 'has_inverse' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogTransform_130784, 'has_inverse', True_130783)
# Declaration of the 'InvertedSymmetricalLogTransform' class
# Getting the type of 'Transform' (line 308)
Transform_130785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 38), 'Transform')

class InvertedSymmetricalLogTransform(Transform_130785, ):

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedSymmetricalLogTransform.__init__', ['base', 'linthresh', 'linscale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['base', 'linthresh', 'linscale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'self' (line 315)
        self_130788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 27), 'self', False)
        # Processing the call keyword arguments (line 315)
        kwargs_130789 = {}
        # Getting the type of 'Transform' (line 315)
        Transform_130786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'Transform', False)
        # Obtaining the member '__init__' of a type (line 315)
        init___130787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), Transform_130786, '__init__')
        # Calling __init__(args, kwargs) (line 315)
        init___call_result_130790 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), init___130787, *[self_130788], **kwargs_130789)
        
        
        # Assigning a Call to a Name (line 316):
        
        # Call to SymmetricalLogTransform(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'base' (line 316)
        base_130792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 41), 'base', False)
        # Getting the type of 'linthresh' (line 316)
        linthresh_130793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'linthresh', False)
        # Getting the type of 'linscale' (line 316)
        linscale_130794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 58), 'linscale', False)
        # Processing the call keyword arguments (line 316)
        kwargs_130795 = {}
        # Getting the type of 'SymmetricalLogTransform' (line 316)
        SymmetricalLogTransform_130791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 17), 'SymmetricalLogTransform', False)
        # Calling SymmetricalLogTransform(args, kwargs) (line 316)
        SymmetricalLogTransform_call_result_130796 = invoke(stypy.reporting.localization.Localization(__file__, 316, 17), SymmetricalLogTransform_130791, *[base_130792, linthresh_130793, linscale_130794], **kwargs_130795)
        
        # Assigning a type to the variable 'symlog' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'symlog', SymmetricalLogTransform_call_result_130796)
        
        # Assigning a Name to a Attribute (line 317):
        # Getting the type of 'base' (line 317)
        base_130797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'base')
        # Getting the type of 'self' (line 317)
        self_130798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self')
        # Setting the type of the member 'base' of a type (line 317)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_130798, 'base', base_130797)
        
        # Assigning a Name to a Attribute (line 318):
        # Getting the type of 'linthresh' (line 318)
        linthresh_130799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'linthresh')
        # Getting the type of 'self' (line 318)
        self_130800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'self')
        # Setting the type of the member 'linthresh' of a type (line 318)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), self_130800, 'linthresh', linthresh_130799)
        
        # Assigning a Call to a Attribute (line 319):
        
        # Call to transform(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'linthresh' (line 319)
        linthresh_130803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 45), 'linthresh', False)
        # Processing the call keyword arguments (line 319)
        kwargs_130804 = {}
        # Getting the type of 'symlog' (line 319)
        symlog_130801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 28), 'symlog', False)
        # Obtaining the member 'transform' of a type (line 319)
        transform_130802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 28), symlog_130801, 'transform')
        # Calling transform(args, kwargs) (line 319)
        transform_call_result_130805 = invoke(stypy.reporting.localization.Localization(__file__, 319, 28), transform_130802, *[linthresh_130803], **kwargs_130804)
        
        # Getting the type of 'self' (line 319)
        self_130806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'self')
        # Setting the type of the member 'invlinthresh' of a type (line 319)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), self_130806, 'invlinthresh', transform_call_result_130805)
        
        # Assigning a Name to a Attribute (line 320):
        # Getting the type of 'linscale' (line 320)
        linscale_130807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'linscale')
        # Getting the type of 'self' (line 320)
        self_130808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'self')
        # Setting the type of the member 'linscale' of a type (line 320)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), self_130808, 'linscale', linscale_130807)
        
        # Assigning a BinOp to a Attribute (line 321):
        # Getting the type of 'linscale' (line 321)
        linscale_130809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'linscale')
        float_130810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 42), 'float')
        # Getting the type of 'self' (line 321)
        self_130811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 48), 'self')
        # Obtaining the member 'base' of a type (line 321)
        base_130812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 48), self_130811, 'base')
        int_130813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 61), 'int')
        # Applying the binary operator '**' (line 321)
        result_pow_130814 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 48), '**', base_130812, int_130813)
        
        # Applying the binary operator '-' (line 321)
        result_sub_130815 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 42), '-', float_130810, result_pow_130814)
        
        # Applying the binary operator 'div' (line 321)
        result_div_130816 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 30), 'div', linscale_130809, result_sub_130815)
        
        # Getting the type of 'self' (line 321)
        self_130817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self')
        # Setting the type of the member '_linscale_adj' of a type (line 321)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_130817, '_linscale_adj', result_div_130816)
        
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
        module_type_store = module_type_store.open_function_context('transform_non_affine', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'InvertedSymmetricalLogTransform.transform_non_affine')
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['a'])
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InvertedSymmetricalLogTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedSymmetricalLogTransform.transform_non_affine', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform_non_affine', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform_non_affine(...)' code ##################

        
        # Assigning a Call to a Name (line 324):
        
        # Call to sign(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'a' (line 324)
        a_130820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'a', False)
        # Processing the call keyword arguments (line 324)
        kwargs_130821 = {}
        # Getting the type of 'np' (line 324)
        np_130818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'np', False)
        # Obtaining the member 'sign' of a type (line 324)
        sign_130819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 15), np_130818, 'sign')
        # Calling sign(args, kwargs) (line 324)
        sign_call_result_130822 = invoke(stypy.reporting.localization.Localization(__file__, 324, 15), sign_130819, *[a_130820], **kwargs_130821)
        
        # Assigning a type to the variable 'sign' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'sign', sign_call_result_130822)
        
        # Assigning a Call to a Name (line 325):
        
        # Call to masked_inside(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'a' (line 325)
        a_130825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 34), 'a', False)
        
        # Getting the type of 'self' (line 325)
        self_130826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'self', False)
        # Obtaining the member 'invlinthresh' of a type (line 325)
        invlinthresh_130827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 38), self_130826, 'invlinthresh')
        # Applying the 'usub' unary operator (line 325)
        result___neg___130828 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 37), 'usub', invlinthresh_130827)
        
        # Getting the type of 'self' (line 326)
        self_130829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), 'self', False)
        # Obtaining the member 'invlinthresh' of a type (line 326)
        invlinthresh_130830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 34), self_130829, 'invlinthresh')
        # Processing the call keyword arguments (line 325)
        # Getting the type of 'False' (line 326)
        False_130831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 58), 'False', False)
        keyword_130832 = False_130831
        kwargs_130833 = {'copy': keyword_130832}
        # Getting the type of 'ma' (line 325)
        ma_130823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 17), 'ma', False)
        # Obtaining the member 'masked_inside' of a type (line 325)
        masked_inside_130824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 17), ma_130823, 'masked_inside')
        # Calling masked_inside(args, kwargs) (line 325)
        masked_inside_call_result_130834 = invoke(stypy.reporting.localization.Localization(__file__, 325, 17), masked_inside_130824, *[a_130825, result___neg___130828, invlinthresh_130830], **kwargs_130833)
        
        # Assigning a type to the variable 'masked' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'masked', masked_inside_call_result_130834)
        
        # Assigning a BinOp to a Name (line 327):
        # Getting the type of 'sign' (line 327)
        sign_130835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 14), 'sign')
        # Getting the type of 'self' (line 327)
        self_130836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'self')
        # Obtaining the member 'linthresh' of a type (line 327)
        linthresh_130837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 21), self_130836, 'linthresh')
        # Applying the binary operator '*' (line 327)
        result_mul_130838 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 14), '*', sign_130835, linthresh_130837)
        
        
        # Call to power(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'self' (line 328)
        self_130841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 'self', False)
        # Obtaining the member 'base' of a type (line 328)
        base_130842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 21), self_130841, 'base')
        # Getting the type of 'sign' (line 328)
        sign_130843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 33), 'sign', False)
        # Getting the type of 'masked' (line 328)
        masked_130844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 41), 'masked', False)
        # Getting the type of 'self' (line 328)
        self_130845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 50), 'self', False)
        # Obtaining the member 'linthresh' of a type (line 328)
        linthresh_130846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 50), self_130845, 'linthresh')
        # Applying the binary operator 'div' (line 328)
        result_div_130847 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 41), 'div', masked_130844, linthresh_130846)
        
        # Applying the binary operator '*' (line 328)
        result_mul_130848 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 33), '*', sign_130843, result_div_130847)
        
        # Getting the type of 'self' (line 329)
        self_130849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 14), 'self', False)
        # Obtaining the member '_linscale_adj' of a type (line 329)
        _linscale_adj_130850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 14), self_130849, '_linscale_adj')
        # Applying the binary operator '-' (line 328)
        result_sub_130851 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 32), '-', result_mul_130848, _linscale_adj_130850)
        
        # Processing the call keyword arguments (line 328)
        kwargs_130852 = {}
        # Getting the type of 'ma' (line 328)
        ma_130839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'ma', False)
        # Obtaining the member 'power' of a type (line 328)
        power_130840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), ma_130839, 'power')
        # Calling power(args, kwargs) (line 328)
        power_call_result_130853 = invoke(stypy.reporting.localization.Localization(__file__, 328, 12), power_130840, *[base_130842, result_sub_130851], **kwargs_130852)
        
        # Applying the binary operator '*' (line 327)
        result_mul_130854 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 36), '*', result_mul_130838, power_call_result_130853)
        
        # Assigning a type to the variable 'exp' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'exp', result_mul_130854)
        
        
        # Call to any(...): (line 330)
        # Processing the call keyword arguments (line 330)
        kwargs_130858 = {}
        # Getting the type of 'masked' (line 330)
        masked_130855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'masked', False)
        # Obtaining the member 'mask' of a type (line 330)
        mask_130856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 11), masked_130855, 'mask')
        # Obtaining the member 'any' of a type (line 330)
        any_130857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 11), mask_130856, 'any')
        # Calling any(args, kwargs) (line 330)
        any_call_result_130859 = invoke(stypy.reporting.localization.Localization(__file__, 330, 11), any_130857, *[], **kwargs_130858)
        
        # Testing the type of an if condition (line 330)
        if_condition_130860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 8), any_call_result_130859)
        # Assigning a type to the variable 'if_condition_130860' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'if_condition_130860', if_condition_130860)
        # SSA begins for if statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to where(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'masked' (line 331)
        masked_130863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'masked', False)
        # Obtaining the member 'mask' of a type (line 331)
        mask_130864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 28), masked_130863, 'mask')
        # Getting the type of 'a' (line 331)
        a_130865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 41), 'a', False)
        # Getting the type of 'self' (line 331)
        self_130866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 45), 'self', False)
        # Obtaining the member '_linscale_adj' of a type (line 331)
        _linscale_adj_130867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 45), self_130866, '_linscale_adj')
        # Applying the binary operator 'div' (line 331)
        result_div_130868 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 41), 'div', a_130865, _linscale_adj_130867)
        
        # Getting the type of 'exp' (line 331)
        exp_130869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 65), 'exp', False)
        # Processing the call keyword arguments (line 331)
        kwargs_130870 = {}
        # Getting the type of 'ma' (line 331)
        ma_130861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'ma', False)
        # Obtaining the member 'where' of a type (line 331)
        where_130862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 19), ma_130861, 'where')
        # Calling where(args, kwargs) (line 331)
        where_call_result_130871 = invoke(stypy.reporting.localization.Localization(__file__, 331, 19), where_130862, *[mask_130864, result_div_130868, exp_130869], **kwargs_130870)
        
        # Assigning a type to the variable 'stypy_return_type' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'stypy_return_type', where_call_result_130871)
        # SSA branch for the else part of an if statement (line 330)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'exp' (line 333)
        exp_130872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 19), 'exp')
        # Assigning a type to the variable 'stypy_return_type' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'stypy_return_type', exp_130872)
        # SSA join for if statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'transform_non_affine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform_non_affine' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_130873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130873)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform_non_affine'
        return stypy_return_type_130873


    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 335, 4, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_function_name', 'InvertedSymmetricalLogTransform.inverted')
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InvertedSymmetricalLogTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InvertedSymmetricalLogTransform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to SymmetricalLogTransform(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'self' (line 336)
        self_130875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 39), 'self', False)
        # Obtaining the member 'base' of a type (line 336)
        base_130876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 39), self_130875, 'base')
        # Getting the type of 'self' (line 337)
        self_130877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 39), 'self', False)
        # Obtaining the member 'linthresh' of a type (line 337)
        linthresh_130878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 39), self_130877, 'linthresh')
        # Getting the type of 'self' (line 337)
        self_130879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 55), 'self', False)
        # Obtaining the member 'linscale' of a type (line 337)
        linscale_130880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 55), self_130879, 'linscale')
        # Processing the call keyword arguments (line 336)
        kwargs_130881 = {}
        # Getting the type of 'SymmetricalLogTransform' (line 336)
        SymmetricalLogTransform_130874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 15), 'SymmetricalLogTransform', False)
        # Calling SymmetricalLogTransform(args, kwargs) (line 336)
        SymmetricalLogTransform_call_result_130882 = invoke(stypy.reporting.localization.Localization(__file__, 336, 15), SymmetricalLogTransform_130874, *[base_130876, linthresh_130878, linscale_130880], **kwargs_130881)
        
        # Assigning a type to the variable 'stypy_return_type' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'stypy_return_type', SymmetricalLogTransform_call_result_130882)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_130883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130883)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_130883


# Assigning a type to the variable 'InvertedSymmetricalLogTransform' (line 308)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'InvertedSymmetricalLogTransform', InvertedSymmetricalLogTransform)

# Assigning a Num to a Name (line 309):
int_130884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 17), 'int')
# Getting the type of 'InvertedSymmetricalLogTransform'
InvertedSymmetricalLogTransform_130885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedSymmetricalLogTransform')
# Setting the type of the member 'input_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedSymmetricalLogTransform_130885, 'input_dims', int_130884)

# Assigning a Num to a Name (line 310):
int_130886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 18), 'int')
# Getting the type of 'InvertedSymmetricalLogTransform'
InvertedSymmetricalLogTransform_130887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedSymmetricalLogTransform')
# Setting the type of the member 'output_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedSymmetricalLogTransform_130887, 'output_dims', int_130886)

# Assigning a Name to a Name (line 311):
# Getting the type of 'True' (line 311)
True_130888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 19), 'True')
# Getting the type of 'InvertedSymmetricalLogTransform'
InvertedSymmetricalLogTransform_130889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedSymmetricalLogTransform')
# Setting the type of the member 'is_separable' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedSymmetricalLogTransform_130889, 'is_separable', True_130888)

# Assigning a Name to a Name (line 312):
# Getting the type of 'True' (line 312)
True_130890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 18), 'True')
# Getting the type of 'InvertedSymmetricalLogTransform'
InvertedSymmetricalLogTransform_130891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InvertedSymmetricalLogTransform')
# Setting the type of the member 'has_inverse' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InvertedSymmetricalLogTransform_130891, 'has_inverse', True_130890)
# Declaration of the 'SymmetricalLogScale' class
# Getting the type of 'ScaleBase' (line 340)
ScaleBase_130892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 26), 'ScaleBase')

class SymmetricalLogScale(ScaleBase_130892, ):
    unicode_130893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, (-1)), 'unicode', u'\n    The symmetrical logarithmic scale is logarithmic in both the\n    positive and negative directions from the origin.\n\n    Since the values close to zero tend toward infinity, there is a\n    need to have a range around zero that is linear.  The parameter\n    *linthresh* allows the user to specify the size of this range\n    (-*linthresh*, *linthresh*).\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SymmetricalLogScale.__init__', ['axis'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_130894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, (-1)), 'unicode', u'\n        *basex*/*basey*:\n           The base of the logarithm\n\n        *linthreshx*/*linthreshy*:\n          A single float which defines the range (-*x*, *x*), within\n          which the plot is linear. This avoids having the plot go to\n          infinity around zero.\n\n        *subsx*/*subsy*:\n           Where to place the subticks between each major tick.\n           Should be a sequence of integers.  For example, in a log10\n           scale: ``[2, 3, 4, 5, 6, 7, 8, 9]``\n\n           will place 8 logarithmically spaced minor ticks between\n           each major tick.\n\n        *linscalex*/*linscaley*:\n           This allows the linear range (-*linthresh* to *linthresh*)\n           to be stretched relative to the logarithmic range.  Its\n           value is the number of decades to use for each half of the\n           linear range.  For example, when *linscale* == 1.0 (the\n           default), the space used for the positive and negative\n           halves of the linear range will be equal to one decade in\n           the logarithmic range.\n        ')
        
        
        # Getting the type of 'axis' (line 382)
        axis_130895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 11), 'axis')
        # Obtaining the member 'axis_name' of a type (line 382)
        axis_name_130896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 11), axis_130895, 'axis_name')
        unicode_130897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 29), 'unicode', u'x')
        # Applying the binary operator '==' (line 382)
        result_eq_130898 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 11), '==', axis_name_130896, unicode_130897)
        
        # Testing the type of an if condition (line 382)
        if_condition_130899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 8), result_eq_130898)
        # Assigning a type to the variable 'if_condition_130899' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'if_condition_130899', if_condition_130899)
        # SSA begins for if statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 383):
        
        # Call to pop(...): (line 383)
        # Processing the call arguments (line 383)
        unicode_130902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 30), 'unicode', u'basex')
        float_130903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 39), 'float')
        # Processing the call keyword arguments (line 383)
        kwargs_130904 = {}
        # Getting the type of 'kwargs' (line 383)
        kwargs_130900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 383)
        pop_130901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 19), kwargs_130900, 'pop')
        # Calling pop(args, kwargs) (line 383)
        pop_call_result_130905 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), pop_130901, *[unicode_130902, float_130903], **kwargs_130904)
        
        # Assigning a type to the variable 'base' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'base', pop_call_result_130905)
        
        # Assigning a Call to a Name (line 384):
        
        # Call to pop(...): (line 384)
        # Processing the call arguments (line 384)
        unicode_130908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 35), 'unicode', u'linthreshx')
        float_130909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 49), 'float')
        # Processing the call keyword arguments (line 384)
        kwargs_130910 = {}
        # Getting the type of 'kwargs' (line 384)
        kwargs_130906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 384)
        pop_130907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 24), kwargs_130906, 'pop')
        # Calling pop(args, kwargs) (line 384)
        pop_call_result_130911 = invoke(stypy.reporting.localization.Localization(__file__, 384, 24), pop_130907, *[unicode_130908, float_130909], **kwargs_130910)
        
        # Assigning a type to the variable 'linthresh' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'linthresh', pop_call_result_130911)
        
        # Assigning a Call to a Name (line 385):
        
        # Call to pop(...): (line 385)
        # Processing the call arguments (line 385)
        unicode_130914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 30), 'unicode', u'subsx')
        # Getting the type of 'None' (line 385)
        None_130915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 39), 'None', False)
        # Processing the call keyword arguments (line 385)
        kwargs_130916 = {}
        # Getting the type of 'kwargs' (line 385)
        kwargs_130912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 385)
        pop_130913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 19), kwargs_130912, 'pop')
        # Calling pop(args, kwargs) (line 385)
        pop_call_result_130917 = invoke(stypy.reporting.localization.Localization(__file__, 385, 19), pop_130913, *[unicode_130914, None_130915], **kwargs_130916)
        
        # Assigning a type to the variable 'subs' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'subs', pop_call_result_130917)
        
        # Assigning a Call to a Name (line 386):
        
        # Call to pop(...): (line 386)
        # Processing the call arguments (line 386)
        unicode_130920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 34), 'unicode', u'linscalex')
        float_130921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 47), 'float')
        # Processing the call keyword arguments (line 386)
        kwargs_130922 = {}
        # Getting the type of 'kwargs' (line 386)
        kwargs_130918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 23), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 386)
        pop_130919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 23), kwargs_130918, 'pop')
        # Calling pop(args, kwargs) (line 386)
        pop_call_result_130923 = invoke(stypy.reporting.localization.Localization(__file__, 386, 23), pop_130919, *[unicode_130920, float_130921], **kwargs_130922)
        
        # Assigning a type to the variable 'linscale' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'linscale', pop_call_result_130923)
        # SSA branch for the else part of an if statement (line 382)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 388):
        
        # Call to pop(...): (line 388)
        # Processing the call arguments (line 388)
        unicode_130926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 30), 'unicode', u'basey')
        float_130927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 39), 'float')
        # Processing the call keyword arguments (line 388)
        kwargs_130928 = {}
        # Getting the type of 'kwargs' (line 388)
        kwargs_130924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 388)
        pop_130925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 19), kwargs_130924, 'pop')
        # Calling pop(args, kwargs) (line 388)
        pop_call_result_130929 = invoke(stypy.reporting.localization.Localization(__file__, 388, 19), pop_130925, *[unicode_130926, float_130927], **kwargs_130928)
        
        # Assigning a type to the variable 'base' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'base', pop_call_result_130929)
        
        # Assigning a Call to a Name (line 389):
        
        # Call to pop(...): (line 389)
        # Processing the call arguments (line 389)
        unicode_130932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 35), 'unicode', u'linthreshy')
        float_130933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 49), 'float')
        # Processing the call keyword arguments (line 389)
        kwargs_130934 = {}
        # Getting the type of 'kwargs' (line 389)
        kwargs_130930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 24), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 389)
        pop_130931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 24), kwargs_130930, 'pop')
        # Calling pop(args, kwargs) (line 389)
        pop_call_result_130935 = invoke(stypy.reporting.localization.Localization(__file__, 389, 24), pop_130931, *[unicode_130932, float_130933], **kwargs_130934)
        
        # Assigning a type to the variable 'linthresh' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'linthresh', pop_call_result_130935)
        
        # Assigning a Call to a Name (line 390):
        
        # Call to pop(...): (line 390)
        # Processing the call arguments (line 390)
        unicode_130938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 30), 'unicode', u'subsy')
        # Getting the type of 'None' (line 390)
        None_130939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 39), 'None', False)
        # Processing the call keyword arguments (line 390)
        kwargs_130940 = {}
        # Getting the type of 'kwargs' (line 390)
        kwargs_130936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 390)
        pop_130937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), kwargs_130936, 'pop')
        # Calling pop(args, kwargs) (line 390)
        pop_call_result_130941 = invoke(stypy.reporting.localization.Localization(__file__, 390, 19), pop_130937, *[unicode_130938, None_130939], **kwargs_130940)
        
        # Assigning a type to the variable 'subs' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'subs', pop_call_result_130941)
        
        # Assigning a Call to a Name (line 391):
        
        # Call to pop(...): (line 391)
        # Processing the call arguments (line 391)
        unicode_130944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 34), 'unicode', u'linscaley')
        float_130945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 47), 'float')
        # Processing the call keyword arguments (line 391)
        kwargs_130946 = {}
        # Getting the type of 'kwargs' (line 391)
        kwargs_130942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 391)
        pop_130943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 23), kwargs_130942, 'pop')
        # Calling pop(args, kwargs) (line 391)
        pop_call_result_130947 = invoke(stypy.reporting.localization.Localization(__file__, 391, 23), pop_130943, *[unicode_130944, float_130945], **kwargs_130946)
        
        # Assigning a type to the variable 'linscale' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'linscale', pop_call_result_130947)
        # SSA join for if statement (line 382)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'base' (line 393)
        base_130948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'base')
        float_130949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 19), 'float')
        # Applying the binary operator '<=' (line 393)
        result_le_130950 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 11), '<=', base_130948, float_130949)
        
        # Testing the type of an if condition (line 393)
        if_condition_130951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 8), result_le_130950)
        # Assigning a type to the variable 'if_condition_130951' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'if_condition_130951', if_condition_130951)
        # SSA begins for if statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 394)
        # Processing the call arguments (line 394)
        unicode_130953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 29), 'unicode', u"'basex/basey' must be larger than 1")
        # Processing the call keyword arguments (line 394)
        kwargs_130954 = {}
        # Getting the type of 'ValueError' (line 394)
        ValueError_130952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 394)
        ValueError_call_result_130955 = invoke(stypy.reporting.localization.Localization(__file__, 394, 18), ValueError_130952, *[unicode_130953], **kwargs_130954)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 394, 12), ValueError_call_result_130955, 'raise parameter', BaseException)
        # SSA join for if statement (line 393)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'linthresh' (line 395)
        linthresh_130956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'linthresh')
        float_130957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 24), 'float')
        # Applying the binary operator '<=' (line 395)
        result_le_130958 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 11), '<=', linthresh_130956, float_130957)
        
        # Testing the type of an if condition (line 395)
        if_condition_130959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 8), result_le_130958)
        # Assigning a type to the variable 'if_condition_130959' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'if_condition_130959', if_condition_130959)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 396)
        # Processing the call arguments (line 396)
        unicode_130961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 29), 'unicode', u"'linthreshx/linthreshy' must be positive")
        # Processing the call keyword arguments (line 396)
        kwargs_130962 = {}
        # Getting the type of 'ValueError' (line 396)
        ValueError_130960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 396)
        ValueError_call_result_130963 = invoke(stypy.reporting.localization.Localization(__file__, 396, 18), ValueError_130960, *[unicode_130961], **kwargs_130962)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 396, 12), ValueError_call_result_130963, 'raise parameter', BaseException)
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'linscale' (line 397)
        linscale_130964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 11), 'linscale')
        float_130965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 23), 'float')
        # Applying the binary operator '<=' (line 397)
        result_le_130966 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 11), '<=', linscale_130964, float_130965)
        
        # Testing the type of an if condition (line 397)
        if_condition_130967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 8), result_le_130966)
        # Assigning a type to the variable 'if_condition_130967' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'if_condition_130967', if_condition_130967)
        # SSA begins for if statement (line 397)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 398)
        # Processing the call arguments (line 398)
        unicode_130969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 29), 'unicode', u"'linscalex/linthreshy' must be positive")
        # Processing the call keyword arguments (line 398)
        kwargs_130970 = {}
        # Getting the type of 'ValueError' (line 398)
        ValueError_130968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 398)
        ValueError_call_result_130971 = invoke(stypy.reporting.localization.Localization(__file__, 398, 18), ValueError_130968, *[unicode_130969], **kwargs_130970)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 398, 12), ValueError_call_result_130971, 'raise parameter', BaseException)
        # SSA join for if statement (line 397)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 400):
        
        # Call to SymmetricalLogTransform(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'base' (line 400)
        base_130974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 55), 'base', False)
        # Getting the type of 'linthresh' (line 401)
        linthresh_130975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 55), 'linthresh', False)
        # Getting the type of 'linscale' (line 402)
        linscale_130976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 55), 'linscale', False)
        # Processing the call keyword arguments (line 400)
        kwargs_130977 = {}
        # Getting the type of 'self' (line 400)
        self_130972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 26), 'self', False)
        # Obtaining the member 'SymmetricalLogTransform' of a type (line 400)
        SymmetricalLogTransform_130973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 26), self_130972, 'SymmetricalLogTransform')
        # Calling SymmetricalLogTransform(args, kwargs) (line 400)
        SymmetricalLogTransform_call_result_130978 = invoke(stypy.reporting.localization.Localization(__file__, 400, 26), SymmetricalLogTransform_130973, *[base_130974, linthresh_130975, linscale_130976], **kwargs_130977)
        
        # Getting the type of 'self' (line 400)
        self_130979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'self')
        # Setting the type of the member '_transform' of a type (line 400)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), self_130979, '_transform', SymmetricalLogTransform_call_result_130978)
        
        # Assigning a Name to a Attribute (line 404):
        # Getting the type of 'base' (line 404)
        base_130980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 20), 'base')
        # Getting the type of 'self' (line 404)
        self_130981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'self')
        # Setting the type of the member 'base' of a type (line 404)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), self_130981, 'base', base_130980)
        
        # Assigning a Name to a Attribute (line 405):
        # Getting the type of 'linthresh' (line 405)
        linthresh_130982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'linthresh')
        # Getting the type of 'self' (line 405)
        self_130983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'self')
        # Setting the type of the member 'linthresh' of a type (line 405)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), self_130983, 'linthresh', linthresh_130982)
        
        # Assigning a Name to a Attribute (line 406):
        # Getting the type of 'linscale' (line 406)
        linscale_130984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 24), 'linscale')
        # Getting the type of 'self' (line 406)
        self_130985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'self')
        # Setting the type of the member 'linscale' of a type (line 406)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_130985, 'linscale', linscale_130984)
        
        # Assigning a Name to a Attribute (line 407):
        # Getting the type of 'subs' (line 407)
        subs_130986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 20), 'subs')
        # Getting the type of 'self' (line 407)
        self_130987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'self')
        # Setting the type of the member 'subs' of a type (line 407)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 8), self_130987, 'subs', subs_130986)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_default_locators_and_formatters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_default_locators_and_formatters'
        module_type_store = module_type_store.open_function_context('set_default_locators_and_formatters', 409, 4, False)
        # Assigning a type to the variable 'self' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_localization', localization)
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_store', module_type_store)
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_function_name', 'SymmetricalLogScale.set_default_locators_and_formatters')
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_varargs_param_name', None)
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_defaults', defaults)
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_varargs', varargs)
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SymmetricalLogScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SymmetricalLogScale.set_default_locators_and_formatters', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_default_locators_and_formatters', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_default_locators_and_formatters(...)' code ##################

        unicode_130988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'unicode', u'\n        Set the locators and formatters to specialized versions for\n        symmetrical log scaling.\n        ')
        
        # Call to set_major_locator(...): (line 414)
        # Processing the call arguments (line 414)
        
        # Call to SymmetricalLogLocator(...): (line 414)
        # Processing the call arguments (line 414)
        
        # Call to get_transform(...): (line 414)
        # Processing the call keyword arguments (line 414)
        kwargs_130994 = {}
        # Getting the type of 'self' (line 414)
        self_130992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 53), 'self', False)
        # Obtaining the member 'get_transform' of a type (line 414)
        get_transform_130993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 53), self_130992, 'get_transform')
        # Calling get_transform(args, kwargs) (line 414)
        get_transform_call_result_130995 = invoke(stypy.reporting.localization.Localization(__file__, 414, 53), get_transform_130993, *[], **kwargs_130994)
        
        # Processing the call keyword arguments (line 414)
        kwargs_130996 = {}
        # Getting the type of 'SymmetricalLogLocator' (line 414)
        SymmetricalLogLocator_130991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 31), 'SymmetricalLogLocator', False)
        # Calling SymmetricalLogLocator(args, kwargs) (line 414)
        SymmetricalLogLocator_call_result_130997 = invoke(stypy.reporting.localization.Localization(__file__, 414, 31), SymmetricalLogLocator_130991, *[get_transform_call_result_130995], **kwargs_130996)
        
        # Processing the call keyword arguments (line 414)
        kwargs_130998 = {}
        # Getting the type of 'axis' (line 414)
        axis_130989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'axis', False)
        # Obtaining the member 'set_major_locator' of a type (line 414)
        set_major_locator_130990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), axis_130989, 'set_major_locator')
        # Calling set_major_locator(args, kwargs) (line 414)
        set_major_locator_call_result_130999 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), set_major_locator_130990, *[SymmetricalLogLocator_call_result_130997], **kwargs_130998)
        
        
        # Call to set_major_formatter(...): (line 415)
        # Processing the call arguments (line 415)
        
        # Call to LogFormatterSciNotation(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'self' (line 415)
        self_131003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 57), 'self', False)
        # Obtaining the member 'base' of a type (line 415)
        base_131004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 57), self_131003, 'base')
        # Processing the call keyword arguments (line 415)
        kwargs_131005 = {}
        # Getting the type of 'LogFormatterSciNotation' (line 415)
        LogFormatterSciNotation_131002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 33), 'LogFormatterSciNotation', False)
        # Calling LogFormatterSciNotation(args, kwargs) (line 415)
        LogFormatterSciNotation_call_result_131006 = invoke(stypy.reporting.localization.Localization(__file__, 415, 33), LogFormatterSciNotation_131002, *[base_131004], **kwargs_131005)
        
        # Processing the call keyword arguments (line 415)
        kwargs_131007 = {}
        # Getting the type of 'axis' (line 415)
        axis_131000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'axis', False)
        # Obtaining the member 'set_major_formatter' of a type (line 415)
        set_major_formatter_131001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), axis_131000, 'set_major_formatter')
        # Calling set_major_formatter(args, kwargs) (line 415)
        set_major_formatter_call_result_131008 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), set_major_formatter_131001, *[LogFormatterSciNotation_call_result_131006], **kwargs_131007)
        
        
        # Call to set_minor_locator(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Call to SymmetricalLogLocator(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Call to get_transform(...): (line 416)
        # Processing the call keyword arguments (line 416)
        kwargs_131014 = {}
        # Getting the type of 'self' (line 416)
        self_131012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 53), 'self', False)
        # Obtaining the member 'get_transform' of a type (line 416)
        get_transform_131013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 53), self_131012, 'get_transform')
        # Calling get_transform(args, kwargs) (line 416)
        get_transform_call_result_131015 = invoke(stypy.reporting.localization.Localization(__file__, 416, 53), get_transform_131013, *[], **kwargs_131014)
        
        # Getting the type of 'self' (line 417)
        self_131016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 53), 'self', False)
        # Obtaining the member 'subs' of a type (line 417)
        subs_131017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 53), self_131016, 'subs')
        # Processing the call keyword arguments (line 416)
        kwargs_131018 = {}
        # Getting the type of 'SymmetricalLogLocator' (line 416)
        SymmetricalLogLocator_131011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 31), 'SymmetricalLogLocator', False)
        # Calling SymmetricalLogLocator(args, kwargs) (line 416)
        SymmetricalLogLocator_call_result_131019 = invoke(stypy.reporting.localization.Localization(__file__, 416, 31), SymmetricalLogLocator_131011, *[get_transform_call_result_131015, subs_131017], **kwargs_131018)
        
        # Processing the call keyword arguments (line 416)
        kwargs_131020 = {}
        # Getting the type of 'axis' (line 416)
        axis_131009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'axis', False)
        # Obtaining the member 'set_minor_locator' of a type (line 416)
        set_minor_locator_131010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), axis_131009, 'set_minor_locator')
        # Calling set_minor_locator(args, kwargs) (line 416)
        set_minor_locator_call_result_131021 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), set_minor_locator_131010, *[SymmetricalLogLocator_call_result_131019], **kwargs_131020)
        
        
        # Call to set_minor_formatter(...): (line 418)
        # Processing the call arguments (line 418)
        
        # Call to NullFormatter(...): (line 418)
        # Processing the call keyword arguments (line 418)
        kwargs_131025 = {}
        # Getting the type of 'NullFormatter' (line 418)
        NullFormatter_131024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 33), 'NullFormatter', False)
        # Calling NullFormatter(args, kwargs) (line 418)
        NullFormatter_call_result_131026 = invoke(stypy.reporting.localization.Localization(__file__, 418, 33), NullFormatter_131024, *[], **kwargs_131025)
        
        # Processing the call keyword arguments (line 418)
        kwargs_131027 = {}
        # Getting the type of 'axis' (line 418)
        axis_131022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'axis', False)
        # Obtaining the member 'set_minor_formatter' of a type (line 418)
        set_minor_formatter_131023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), axis_131022, 'set_minor_formatter')
        # Calling set_minor_formatter(args, kwargs) (line 418)
        set_minor_formatter_call_result_131028 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), set_minor_formatter_131023, *[NullFormatter_call_result_131026], **kwargs_131027)
        
        
        # ################# End of 'set_default_locators_and_formatters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_default_locators_and_formatters' in the type store
        # Getting the type of 'stypy_return_type' (line 409)
        stypy_return_type_131029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_default_locators_and_formatters'
        return stypy_return_type_131029


    @norecursion
    def get_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_transform'
        module_type_store = module_type_store.open_function_context('get_transform', 420, 4, False)
        # Assigning a type to the variable 'self' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_localization', localization)
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_function_name', 'SymmetricalLogScale.get_transform')
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_param_names_list', [])
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SymmetricalLogScale.get_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SymmetricalLogScale.get_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_transform(...)' code ##################

        unicode_131030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, (-1)), 'unicode', u'\n        Return a :class:`SymmetricalLogTransform` instance.\n        ')
        # Getting the type of 'self' (line 424)
        self_131031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'self')
        # Obtaining the member '_transform' of a type (line 424)
        _transform_131032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 15), self_131031, '_transform')
        # Assigning a type to the variable 'stypy_return_type' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'stypy_return_type', _transform_131032)
        
        # ################# End of 'get_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 420)
        stypy_return_type_131033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131033)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_transform'
        return stypy_return_type_131033


# Assigning a type to the variable 'SymmetricalLogScale' (line 340)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 0), 'SymmetricalLogScale', SymmetricalLogScale)

# Assigning a Str to a Name (line 350):
unicode_131034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 11), 'unicode', u'symlog')
# Getting the type of 'SymmetricalLogScale'
SymmetricalLogScale_131035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogScale')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogScale_131035, 'name', unicode_131034)

# Assigning a Name to a Name (line 352):
# Getting the type of 'SymmetricalLogScale'
SymmetricalLogScale_131036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogScale')
# Obtaining the member 'SymmetricalLogTransform' of a type
SymmetricalLogTransform_131037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogScale_131036, 'SymmetricalLogTransform')
# Getting the type of 'SymmetricalLogScale'
SymmetricalLogScale_131038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogScale')
# Setting the type of the member 'SymmetricalLogTransform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogScale_131038, 'SymmetricalLogTransform', SymmetricalLogTransform_131037)

# Assigning a Name to a Name (line 353):
# Getting the type of 'SymmetricalLogScale'
SymmetricalLogScale_131039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogScale')
# Obtaining the member 'InvertedSymmetricalLogTransform' of a type
InvertedSymmetricalLogTransform_131040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogScale_131039, 'InvertedSymmetricalLogTransform')
# Getting the type of 'SymmetricalLogScale'
SymmetricalLogScale_131041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SymmetricalLogScale')
# Setting the type of the member 'InvertedSymmetricalLogTransform' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SymmetricalLogScale_131041, 'InvertedSymmetricalLogTransform', InvertedSymmetricalLogTransform_131040)
# Declaration of the 'LogitTransform' class
# Getting the type of 'Transform' (line 427)
Transform_131042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 21), 'Transform')

class LogitTransform(Transform_131042, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 433, 4, False)
        # Assigning a type to the variable 'self' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogitTransform.__init__', ['nonpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['nonpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'self' (line 434)
        self_131045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 27), 'self', False)
        # Processing the call keyword arguments (line 434)
        kwargs_131046 = {}
        # Getting the type of 'Transform' (line 434)
        Transform_131043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'Transform', False)
        # Obtaining the member '__init__' of a type (line 434)
        init___131044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), Transform_131043, '__init__')
        # Calling __init__(args, kwargs) (line 434)
        init___call_result_131047 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), init___131044, *[self_131045], **kwargs_131046)
        
        
        
        # Getting the type of 'nonpos' (line 435)
        nonpos_131048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 11), 'nonpos')
        unicode_131049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 21), 'unicode', u'mask')
        # Applying the binary operator '==' (line 435)
        result_eq_131050 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 11), '==', nonpos_131048, unicode_131049)
        
        # Testing the type of an if condition (line 435)
        if_condition_131051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 8), result_eq_131050)
        # Assigning a type to the variable 'if_condition_131051' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'if_condition_131051', if_condition_131051)
        # SSA begins for if statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 436):
        # Getting the type of 'np' (line 436)
        np_131052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 31), 'np')
        # Obtaining the member 'nan' of a type (line 436)
        nan_131053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 31), np_131052, 'nan')
        # Getting the type of 'self' (line 436)
        self_131054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'self')
        # Setting the type of the member '_fill_value' of a type (line 436)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), self_131054, '_fill_value', nan_131053)
        # SSA branch for the else part of an if statement (line 435)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Attribute (line 438):
        float_131055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 31), 'float')
        # Getting the type of 'self' (line 438)
        self_131056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'self')
        # Setting the type of the member '_fill_value' of a type (line 438)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), self_131056, '_fill_value', float_131055)
        # SSA join for if statement (line 435)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 439):
        # Getting the type of 'nonpos' (line 439)
        nonpos_131057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'nonpos')
        # Getting the type of 'self' (line 439)
        self_131058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'self')
        # Setting the type of the member '_nonpos' of a type (line 439)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), self_131058, '_nonpos', nonpos_131057)
        
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
        module_type_store = module_type_store.open_function_context('transform_non_affine', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'LogitTransform.transform_non_affine')
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['a'])
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogitTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogitTransform.transform_non_affine', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform_non_affine', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform_non_affine(...)' code ##################

        unicode_131059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 8), 'unicode', u'logit transform (base 10), masked or clipped')
        
        # Call to errstate(...): (line 443)
        # Processing the call keyword arguments (line 443)
        unicode_131062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 33), 'unicode', u'ignore')
        keyword_131063 = unicode_131062
        kwargs_131064 = {'invalid': keyword_131063}
        # Getting the type of 'np' (line 443)
        np_131060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 13), 'np', False)
        # Obtaining the member 'errstate' of a type (line 443)
        errstate_131061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 13), np_131060, 'errstate')
        # Calling errstate(args, kwargs) (line 443)
        errstate_call_result_131065 = invoke(stypy.reporting.localization.Localization(__file__, 443, 13), errstate_131061, *[], **kwargs_131064)
        
        with_131066 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 443, 13), errstate_call_result_131065, 'with parameter', '__enter__', '__exit__')

        if with_131066:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 443)
            enter___131067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 13), errstate_call_result_131065, '__enter__')
            with_enter_131068 = invoke(stypy.reporting.localization.Localization(__file__, 443, 13), enter___131067)
            
            # Assigning a Call to a Name (line 444):
            
            # Call to select(...): (line 444)
            # Processing the call arguments (line 444)
            
            # Obtaining an instance of the builtin type 'list' (line 445)
            list_131071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 16), 'list')
            # Adding type elements to the builtin type 'list' instance (line 445)
            # Adding element type (line 445)
            
            # Getting the type of 'a' (line 445)
            a_131072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 17), 'a', False)
            int_131073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 22), 'int')
            # Applying the binary operator '<=' (line 445)
            result_le_131074 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 17), '<=', a_131072, int_131073)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 16), list_131071, result_le_131074)
            # Adding element type (line 445)
            
            # Getting the type of 'a' (line 445)
            a_131075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 25), 'a', False)
            int_131076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 30), 'int')
            # Applying the binary operator '>=' (line 445)
            result_ge_131077 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 25), '>=', a_131075, int_131076)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 16), list_131071, result_ge_131077)
            
            
            # Obtaining an instance of the builtin type 'list' (line 445)
            list_131078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 34), 'list')
            # Adding type elements to the builtin type 'list' instance (line 445)
            # Adding element type (line 445)
            # Getting the type of 'self' (line 445)
            self_131079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 35), 'self', False)
            # Obtaining the member '_fill_value' of a type (line 445)
            _fill_value_131080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 35), self_131079, '_fill_value')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 34), list_131078, _fill_value_131080)
            # Adding element type (line 445)
            int_131081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 53), 'int')
            # Getting the type of 'self' (line 445)
            self_131082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 57), 'self', False)
            # Obtaining the member '_fill_value' of a type (line 445)
            _fill_value_131083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 57), self_131082, '_fill_value')
            # Applying the binary operator '-' (line 445)
            result_sub_131084 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 53), '-', int_131081, _fill_value_131083)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 34), list_131078, result_sub_131084)
            
            # Getting the type of 'a' (line 445)
            a_131085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 76), 'a', False)
            # Processing the call keyword arguments (line 444)
            kwargs_131086 = {}
            # Getting the type of 'np' (line 444)
            np_131069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'np', False)
            # Obtaining the member 'select' of a type (line 444)
            select_131070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 16), np_131069, 'select')
            # Calling select(args, kwargs) (line 444)
            select_call_result_131087 = invoke(stypy.reporting.localization.Localization(__file__, 444, 16), select_131070, *[list_131071, list_131078, a_131085], **kwargs_131086)
            
            # Assigning a type to the variable 'a' (line 444)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'a', select_call_result_131087)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 443)
            exit___131088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 13), errstate_call_result_131065, '__exit__')
            with_exit_131089 = invoke(stypy.reporting.localization.Localization(__file__, 443, 13), exit___131088, None, None, None)

        
        # Call to log10(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'a' (line 446)
        a_131092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 24), 'a', False)
        int_131093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 29), 'int')
        # Getting the type of 'a' (line 446)
        a_131094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 33), 'a', False)
        # Applying the binary operator '-' (line 446)
        result_sub_131095 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 29), '-', int_131093, a_131094)
        
        # Applying the binary operator 'div' (line 446)
        result_div_131096 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 24), 'div', a_131092, result_sub_131095)
        
        # Processing the call keyword arguments (line 446)
        kwargs_131097 = {}
        # Getting the type of 'np' (line 446)
        np_131090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 15), 'np', False)
        # Obtaining the member 'log10' of a type (line 446)
        log10_131091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 15), np_131090, 'log10')
        # Calling log10(args, kwargs) (line 446)
        log10_call_result_131098 = invoke(stypy.reporting.localization.Localization(__file__, 446, 15), log10_131091, *[result_div_131096], **kwargs_131097)
        
        # Assigning a type to the variable 'stypy_return_type' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'stypy_return_type', log10_call_result_131098)
        
        # ################# End of 'transform_non_affine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform_non_affine' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_131099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131099)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform_non_affine'
        return stypy_return_type_131099


    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 448, 4, False)
        # Assigning a type to the variable 'self' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogitTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
        LogitTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogitTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogitTransform.inverted.__dict__.__setitem__('stypy_function_name', 'LogitTransform.inverted')
        LogitTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        LogitTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogitTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogitTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogitTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogitTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogitTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogitTransform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to LogisticTransform(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'self' (line 449)
        self_131101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 33), 'self', False)
        # Obtaining the member '_nonpos' of a type (line 449)
        _nonpos_131102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 33), self_131101, '_nonpos')
        # Processing the call keyword arguments (line 449)
        kwargs_131103 = {}
        # Getting the type of 'LogisticTransform' (line 449)
        LogisticTransform_131100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'LogisticTransform', False)
        # Calling LogisticTransform(args, kwargs) (line 449)
        LogisticTransform_call_result_131104 = invoke(stypy.reporting.localization.Localization(__file__, 449, 15), LogisticTransform_131100, *[_nonpos_131102], **kwargs_131103)
        
        # Assigning a type to the variable 'stypy_return_type' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'stypy_return_type', LogisticTransform_call_result_131104)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 448)
        stypy_return_type_131105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131105)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_131105


# Assigning a type to the variable 'LogitTransform' (line 427)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), 'LogitTransform', LogitTransform)

# Assigning a Num to a Name (line 428):
int_131106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 17), 'int')
# Getting the type of 'LogitTransform'
LogitTransform_131107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogitTransform')
# Setting the type of the member 'input_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogitTransform_131107, 'input_dims', int_131106)

# Assigning a Num to a Name (line 429):
int_131108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 18), 'int')
# Getting the type of 'LogitTransform'
LogitTransform_131109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogitTransform')
# Setting the type of the member 'output_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogitTransform_131109, 'output_dims', int_131108)

# Assigning a Name to a Name (line 430):
# Getting the type of 'True' (line 430)
True_131110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'True')
# Getting the type of 'LogitTransform'
LogitTransform_131111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogitTransform')
# Setting the type of the member 'is_separable' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogitTransform_131111, 'is_separable', True_131110)

# Assigning a Name to a Name (line 431):
# Getting the type of 'True' (line 431)
True_131112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), 'True')
# Getting the type of 'LogitTransform'
LogitTransform_131113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogitTransform')
# Setting the type of the member 'has_inverse' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogitTransform_131113, 'has_inverse', True_131112)
# Declaration of the 'LogisticTransform' class
# Getting the type of 'Transform' (line 452)
Transform_131114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 24), 'Transform')

class LogisticTransform(Transform_131114, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_131115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 30), 'unicode', u'mask')
        defaults = [unicode_131115]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 458, 4, False)
        # Assigning a type to the variable 'self' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogisticTransform.__init__', ['nonpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['nonpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'self' (line 459)
        self_131118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 27), 'self', False)
        # Processing the call keyword arguments (line 459)
        kwargs_131119 = {}
        # Getting the type of 'Transform' (line 459)
        Transform_131116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'Transform', False)
        # Obtaining the member '__init__' of a type (line 459)
        init___131117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), Transform_131116, '__init__')
        # Calling __init__(args, kwargs) (line 459)
        init___call_result_131120 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), init___131117, *[self_131118], **kwargs_131119)
        
        
        # Assigning a Name to a Attribute (line 460):
        # Getting the type of 'nonpos' (line 460)
        nonpos_131121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'nonpos')
        # Getting the type of 'self' (line 460)
        self_131122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'self')
        # Setting the type of the member '_nonpos' of a type (line 460)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), self_131122, '_nonpos', nonpos_131121)
        
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
        module_type_store = module_type_store.open_function_context('transform_non_affine', 462, 4, False)
        # Assigning a type to the variable 'self' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_localization', localization)
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_function_name', 'LogisticTransform.transform_non_affine')
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_param_names_list', ['a'])
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogisticTransform.transform_non_affine.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogisticTransform.transform_non_affine', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform_non_affine', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform_non_affine(...)' code ##################

        unicode_131123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 8), 'unicode', u'logistic transform (base 10)')
        float_131124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 15), 'float')
        int_131125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 22), 'int')
        int_131126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 26), 'int')
        
        # Getting the type of 'a' (line 464)
        a_131127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 32), 'a')
        # Applying the 'usub' unary operator (line 464)
        result___neg___131128 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 31), 'usub', a_131127)
        
        # Applying the binary operator '**' (line 464)
        result_pow_131129 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 26), '**', int_131126, result___neg___131128)
        
        # Applying the binary operator '+' (line 464)
        result_add_131130 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 22), '+', int_131125, result_pow_131129)
        
        # Applying the binary operator 'div' (line 464)
        result_div_131131 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 15), 'div', float_131124, result_add_131130)
        
        # Assigning a type to the variable 'stypy_return_type' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'stypy_return_type', result_div_131131)
        
        # ################# End of 'transform_non_affine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform_non_affine' in the type store
        # Getting the type of 'stypy_return_type' (line 462)
        stypy_return_type_131132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131132)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform_non_affine'
        return stypy_return_type_131132


    @norecursion
    def inverted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inverted'
        module_type_store = module_type_store.open_function_context('inverted', 466, 4, False)
        # Assigning a type to the variable 'self' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogisticTransform.inverted.__dict__.__setitem__('stypy_localization', localization)
        LogisticTransform.inverted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogisticTransform.inverted.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogisticTransform.inverted.__dict__.__setitem__('stypy_function_name', 'LogisticTransform.inverted')
        LogisticTransform.inverted.__dict__.__setitem__('stypy_param_names_list', [])
        LogisticTransform.inverted.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogisticTransform.inverted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogisticTransform.inverted.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogisticTransform.inverted.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogisticTransform.inverted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogisticTransform.inverted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogisticTransform.inverted', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to LogitTransform(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'self' (line 467)
        self_131134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 30), 'self', False)
        # Obtaining the member '_nonpos' of a type (line 467)
        _nonpos_131135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 30), self_131134, '_nonpos')
        # Processing the call keyword arguments (line 467)
        kwargs_131136 = {}
        # Getting the type of 'LogitTransform' (line 467)
        LogitTransform_131133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'LogitTransform', False)
        # Calling LogitTransform(args, kwargs) (line 467)
        LogitTransform_call_result_131137 = invoke(stypy.reporting.localization.Localization(__file__, 467, 15), LogitTransform_131133, *[_nonpos_131135], **kwargs_131136)
        
        # Assigning a type to the variable 'stypy_return_type' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'stypy_return_type', LogitTransform_call_result_131137)
        
        # ################# End of 'inverted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inverted' in the type store
        # Getting the type of 'stypy_return_type' (line 466)
        stypy_return_type_131138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131138)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inverted'
        return stypy_return_type_131138


# Assigning a type to the variable 'LogisticTransform' (line 452)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 0), 'LogisticTransform', LogisticTransform)

# Assigning a Num to a Name (line 453):
int_131139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 17), 'int')
# Getting the type of 'LogisticTransform'
LogisticTransform_131140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogisticTransform')
# Setting the type of the member 'input_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogisticTransform_131140, 'input_dims', int_131139)

# Assigning a Num to a Name (line 454):
int_131141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 18), 'int')
# Getting the type of 'LogisticTransform'
LogisticTransform_131142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogisticTransform')
# Setting the type of the member 'output_dims' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogisticTransform_131142, 'output_dims', int_131141)

# Assigning a Name to a Name (line 455):
# Getting the type of 'True' (line 455)
True_131143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 19), 'True')
# Getting the type of 'LogisticTransform'
LogisticTransform_131144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogisticTransform')
# Setting the type of the member 'is_separable' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogisticTransform_131144, 'is_separable', True_131143)

# Assigning a Name to a Name (line 456):
# Getting the type of 'True' (line 456)
True_131145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 18), 'True')
# Getting the type of 'LogisticTransform'
LogisticTransform_131146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogisticTransform')
# Setting the type of the member 'has_inverse' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogisticTransform_131146, 'has_inverse', True_131145)
# Declaration of the 'LogitScale' class
# Getting the type of 'ScaleBase' (line 470)
ScaleBase_131147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 17), 'ScaleBase')

class LogitScale(ScaleBase_131147, ):
    unicode_131148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, (-1)), 'unicode', u'\n    Logit scale for data between zero and one, both excluded.\n\n    This scale is similar to a log scale close to zero and to one, and almost\n    linear around 0.5. It maps the interval ]0, 1[ onto ]-infty, +infty[.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_131149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 36), 'unicode', u'mask')
        defaults = [unicode_131149]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 479, 4, False)
        # Assigning a type to the variable 'self' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogitScale.__init__', ['axis', 'nonpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['axis', 'nonpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_131150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, (-1)), 'unicode', u"\n        *nonpos*: ['mask' | 'clip' ]\n          values beyond ]0, 1[ can be masked as invalid, or clipped to a number\n          very close to 0 or 1\n        ")
        
        
        # Getting the type of 'nonpos' (line 485)
        nonpos_131151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 11), 'nonpos')
        
        # Obtaining an instance of the builtin type 'list' (line 485)
        list_131152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 485)
        # Adding element type (line 485)
        unicode_131153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 26), 'unicode', u'mask')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 25), list_131152, unicode_131153)
        # Adding element type (line 485)
        unicode_131154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 34), 'unicode', u'clip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 25), list_131152, unicode_131154)
        
        # Applying the binary operator 'notin' (line 485)
        result_contains_131155 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 11), 'notin', nonpos_131151, list_131152)
        
        # Testing the type of an if condition (line 485)
        if_condition_131156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 8), result_contains_131155)
        # Assigning a type to the variable 'if_condition_131156' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'if_condition_131156', if_condition_131156)
        # SSA begins for if statement (line 485)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 486)
        # Processing the call arguments (line 486)
        unicode_131158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 29), 'unicode', u"nonposx, nonposy kwarg must be 'mask' or 'clip'")
        # Processing the call keyword arguments (line 486)
        kwargs_131159 = {}
        # Getting the type of 'ValueError' (line 486)
        ValueError_131157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 486)
        ValueError_call_result_131160 = invoke(stypy.reporting.localization.Localization(__file__, 486, 18), ValueError_131157, *[unicode_131158], **kwargs_131159)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 486, 12), ValueError_call_result_131160, 'raise parameter', BaseException)
        # SSA join for if statement (line 485)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 488):
        
        # Call to LogitTransform(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'nonpos' (line 488)
        nonpos_131162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 41), 'nonpos', False)
        # Processing the call keyword arguments (line 488)
        kwargs_131163 = {}
        # Getting the type of 'LogitTransform' (line 488)
        LogitTransform_131161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 26), 'LogitTransform', False)
        # Calling LogitTransform(args, kwargs) (line 488)
        LogitTransform_call_result_131164 = invoke(stypy.reporting.localization.Localization(__file__, 488, 26), LogitTransform_131161, *[nonpos_131162], **kwargs_131163)
        
        # Getting the type of 'self' (line 488)
        self_131165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'self')
        # Setting the type of the member '_transform' of a type (line 488)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), self_131165, '_transform', LogitTransform_call_result_131164)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_transform'
        module_type_store = module_type_store.open_function_context('get_transform', 490, 4, False)
        # Assigning a type to the variable 'self' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogitScale.get_transform.__dict__.__setitem__('stypy_localization', localization)
        LogitScale.get_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogitScale.get_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogitScale.get_transform.__dict__.__setitem__('stypy_function_name', 'LogitScale.get_transform')
        LogitScale.get_transform.__dict__.__setitem__('stypy_param_names_list', [])
        LogitScale.get_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogitScale.get_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogitScale.get_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogitScale.get_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogitScale.get_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogitScale.get_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogitScale.get_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_transform(...)' code ##################

        unicode_131166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, (-1)), 'unicode', u'\n        Return a :class:`LogitTransform` instance.\n        ')
        # Getting the type of 'self' (line 494)
        self_131167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'self')
        # Obtaining the member '_transform' of a type (line 494)
        _transform_131168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 15), self_131167, '_transform')
        # Assigning a type to the variable 'stypy_return_type' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'stypy_return_type', _transform_131168)
        
        # ################# End of 'get_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 490)
        stypy_return_type_131169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_transform'
        return stypy_return_type_131169


    @norecursion
    def set_default_locators_and_formatters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_default_locators_and_formatters'
        module_type_store = module_type_store.open_function_context('set_default_locators_and_formatters', 496, 4, False)
        # Assigning a type to the variable 'self' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_localization', localization)
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_function_name', 'LogitScale.set_default_locators_and_formatters')
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogitScale.set_default_locators_and_formatters.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogitScale.set_default_locators_and_formatters', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_default_locators_and_formatters', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_default_locators_and_formatters(...)' code ##################

        
        # Call to set_major_locator(...): (line 498)
        # Processing the call arguments (line 498)
        
        # Call to LogitLocator(...): (line 498)
        # Processing the call keyword arguments (line 498)
        kwargs_131173 = {}
        # Getting the type of 'LogitLocator' (line 498)
        LogitLocator_131172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 'LogitLocator', False)
        # Calling LogitLocator(args, kwargs) (line 498)
        LogitLocator_call_result_131174 = invoke(stypy.reporting.localization.Localization(__file__, 498, 31), LogitLocator_131172, *[], **kwargs_131173)
        
        # Processing the call keyword arguments (line 498)
        kwargs_131175 = {}
        # Getting the type of 'axis' (line 498)
        axis_131170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'axis', False)
        # Obtaining the member 'set_major_locator' of a type (line 498)
        set_major_locator_131171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), axis_131170, 'set_major_locator')
        # Calling set_major_locator(args, kwargs) (line 498)
        set_major_locator_call_result_131176 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), set_major_locator_131171, *[LogitLocator_call_result_131174], **kwargs_131175)
        
        
        # Call to set_major_formatter(...): (line 499)
        # Processing the call arguments (line 499)
        
        # Call to LogitFormatter(...): (line 499)
        # Processing the call keyword arguments (line 499)
        kwargs_131180 = {}
        # Getting the type of 'LogitFormatter' (line 499)
        LogitFormatter_131179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 33), 'LogitFormatter', False)
        # Calling LogitFormatter(args, kwargs) (line 499)
        LogitFormatter_call_result_131181 = invoke(stypy.reporting.localization.Localization(__file__, 499, 33), LogitFormatter_131179, *[], **kwargs_131180)
        
        # Processing the call keyword arguments (line 499)
        kwargs_131182 = {}
        # Getting the type of 'axis' (line 499)
        axis_131177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'axis', False)
        # Obtaining the member 'set_major_formatter' of a type (line 499)
        set_major_formatter_131178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), axis_131177, 'set_major_formatter')
        # Calling set_major_formatter(args, kwargs) (line 499)
        set_major_formatter_call_result_131183 = invoke(stypy.reporting.localization.Localization(__file__, 499, 8), set_major_formatter_131178, *[LogitFormatter_call_result_131181], **kwargs_131182)
        
        
        # Call to set_minor_locator(...): (line 500)
        # Processing the call arguments (line 500)
        
        # Call to LogitLocator(...): (line 500)
        # Processing the call keyword arguments (line 500)
        # Getting the type of 'True' (line 500)
        True_131187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 50), 'True', False)
        keyword_131188 = True_131187
        kwargs_131189 = {'minor': keyword_131188}
        # Getting the type of 'LogitLocator' (line 500)
        LogitLocator_131186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 31), 'LogitLocator', False)
        # Calling LogitLocator(args, kwargs) (line 500)
        LogitLocator_call_result_131190 = invoke(stypy.reporting.localization.Localization(__file__, 500, 31), LogitLocator_131186, *[], **kwargs_131189)
        
        # Processing the call keyword arguments (line 500)
        kwargs_131191 = {}
        # Getting the type of 'axis' (line 500)
        axis_131184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'axis', False)
        # Obtaining the member 'set_minor_locator' of a type (line 500)
        set_minor_locator_131185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), axis_131184, 'set_minor_locator')
        # Calling set_minor_locator(args, kwargs) (line 500)
        set_minor_locator_call_result_131192 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), set_minor_locator_131185, *[LogitLocator_call_result_131190], **kwargs_131191)
        
        
        # Call to set_minor_formatter(...): (line 501)
        # Processing the call arguments (line 501)
        
        # Call to LogitFormatter(...): (line 501)
        # Processing the call keyword arguments (line 501)
        kwargs_131196 = {}
        # Getting the type of 'LogitFormatter' (line 501)
        LogitFormatter_131195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 33), 'LogitFormatter', False)
        # Calling LogitFormatter(args, kwargs) (line 501)
        LogitFormatter_call_result_131197 = invoke(stypy.reporting.localization.Localization(__file__, 501, 33), LogitFormatter_131195, *[], **kwargs_131196)
        
        # Processing the call keyword arguments (line 501)
        kwargs_131198 = {}
        # Getting the type of 'axis' (line 501)
        axis_131193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'axis', False)
        # Obtaining the member 'set_minor_formatter' of a type (line 501)
        set_minor_formatter_131194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), axis_131193, 'set_minor_formatter')
        # Calling set_minor_formatter(args, kwargs) (line 501)
        set_minor_formatter_call_result_131199 = invoke(stypy.reporting.localization.Localization(__file__, 501, 8), set_minor_formatter_131194, *[LogitFormatter_call_result_131197], **kwargs_131198)
        
        
        # ################# End of 'set_default_locators_and_formatters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_default_locators_and_formatters' in the type store
        # Getting the type of 'stypy_return_type' (line 496)
        stypy_return_type_131200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_default_locators_and_formatters'
        return stypy_return_type_131200


    @norecursion
    def limit_range_for_scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'limit_range_for_scale'
        module_type_store = module_type_store.open_function_context('limit_range_for_scale', 503, 4, False)
        # Assigning a type to the variable 'self' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_localization', localization)
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_function_name', 'LogitScale.limit_range_for_scale')
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_param_names_list', ['vmin', 'vmax', 'minpos'])
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LogitScale.limit_range_for_scale.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LogitScale.limit_range_for_scale', ['vmin', 'vmax', 'minpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'limit_range_for_scale', localization, ['vmin', 'vmax', 'minpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'limit_range_for_scale(...)' code ##################

        unicode_131201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, (-1)), 'unicode', u'\n        Limit the domain to values between 0 and 1 (excluded).\n        ')
        
        
        
        # Call to isfinite(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'minpos' (line 507)
        minpos_131204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 27), 'minpos', False)
        # Processing the call keyword arguments (line 507)
        kwargs_131205 = {}
        # Getting the type of 'np' (line 507)
        np_131202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 15), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 507)
        isfinite_131203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 15), np_131202, 'isfinite')
        # Calling isfinite(args, kwargs) (line 507)
        isfinite_call_result_131206 = invoke(stypy.reporting.localization.Localization(__file__, 507, 15), isfinite_131203, *[minpos_131204], **kwargs_131205)
        
        # Applying the 'not' unary operator (line 507)
        result_not__131207 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 11), 'not', isfinite_call_result_131206)
        
        # Testing the type of an if condition (line 507)
        if_condition_131208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 8), result_not__131207)
        # Assigning a type to the variable 'if_condition_131208' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'if_condition_131208', if_condition_131208)
        # SSA begins for if statement (line 507)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 508):
        float_131209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 21), 'float')
        # Assigning a type to the variable 'minpos' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'minpos', float_131209)
        # SSA join for if statement (line 507)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 510)
        tuple_131210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 510)
        # Adding element type (line 510)
        
        
        # Getting the type of 'vmin' (line 510)
        vmin_131211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'vmin')
        int_131212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 34), 'int')
        # Applying the binary operator '<=' (line 510)
        result_le_131213 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 26), '<=', vmin_131211, int_131212)
        
        # Testing the type of an if expression (line 510)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 16), result_le_131213)
        # SSA begins for if expression (line 510)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'minpos' (line 510)
        minpos_131214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'minpos')
        # SSA branch for the else part of an if expression (line 510)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'vmin' (line 510)
        vmin_131215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 41), 'vmin')
        # SSA join for if expression (line 510)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_131216 = union_type.UnionType.add(minpos_131214, vmin_131215)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 16), tuple_131210, if_exp_131216)
        # Adding element type (line 510)
        
        
        # Getting the type of 'vmax' (line 511)
        vmax_131217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 30), 'vmax')
        int_131218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 38), 'int')
        # Applying the binary operator '>=' (line 511)
        result_ge_131219 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 30), '>=', vmax_131217, int_131218)
        
        # Testing the type of an if expression (line 511)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 16), result_ge_131219)
        # SSA begins for if expression (line 511)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_131220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 16), 'int')
        # Getting the type of 'minpos' (line 511)
        minpos_131221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 20), 'minpos')
        # Applying the binary operator '-' (line 511)
        result_sub_131222 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 16), '-', int_131220, minpos_131221)
        
        # SSA branch for the else part of an if expression (line 511)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'vmax' (line 511)
        vmax_131223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 45), 'vmax')
        # SSA join for if expression (line 511)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_131224 = union_type.UnionType.add(result_sub_131222, vmax_131223)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 16), tuple_131210, if_exp_131224)
        
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'stypy_return_type', tuple_131210)
        
        # ################# End of 'limit_range_for_scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'limit_range_for_scale' in the type store
        # Getting the type of 'stypy_return_type' (line 503)
        stypy_return_type_131225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'limit_range_for_scale'
        return stypy_return_type_131225


# Assigning a type to the variable 'LogitScale' (line 470)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 0), 'LogitScale', LogitScale)

# Assigning a Str to a Name (line 477):
unicode_131226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 11), 'unicode', u'logit')
# Getting the type of 'LogitScale'
LogitScale_131227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LogitScale')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LogitScale_131227, 'name', unicode_131226)

# Assigning a Dict to a Name (line 514):

# Obtaining an instance of the builtin type 'dict' (line 514)
dict_131228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 514)
# Adding element type (key, value) (line 514)
unicode_131229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 4), 'unicode', u'linear')
# Getting the type of 'LinearScale' (line 515)
LinearScale_131230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 14), 'LinearScale')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 17), dict_131228, (unicode_131229, LinearScale_131230))
# Adding element type (key, value) (line 514)
unicode_131231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 4), 'unicode', u'log')
# Getting the type of 'LogScale' (line 516)
LogScale_131232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 14), 'LogScale')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 17), dict_131228, (unicode_131231, LogScale_131232))
# Adding element type (key, value) (line 514)
unicode_131233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 4), 'unicode', u'symlog')
# Getting the type of 'SymmetricalLogScale' (line 517)
SymmetricalLogScale_131234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 14), 'SymmetricalLogScale')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 17), dict_131228, (unicode_131233, SymmetricalLogScale_131234))
# Adding element type (key, value) (line 514)
unicode_131235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 4), 'unicode', u'logit')
# Getting the type of 'LogitScale' (line 518)
LogitScale_131236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 14), 'LogitScale')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 17), dict_131228, (unicode_131235, LogitScale_131236))

# Assigning a type to the variable '_scale_mapping' (line 514)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), '_scale_mapping', dict_131228)

@norecursion
def get_scale_names(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_scale_names'
    module_type_store = module_type_store.open_function_context('get_scale_names', 522, 0, False)
    
    # Passed parameters checking function
    get_scale_names.stypy_localization = localization
    get_scale_names.stypy_type_of_self = None
    get_scale_names.stypy_type_store = module_type_store
    get_scale_names.stypy_function_name = 'get_scale_names'
    get_scale_names.stypy_param_names_list = []
    get_scale_names.stypy_varargs_param_name = None
    get_scale_names.stypy_kwargs_param_name = None
    get_scale_names.stypy_call_defaults = defaults
    get_scale_names.stypy_call_varargs = varargs
    get_scale_names.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_scale_names', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_scale_names', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_scale_names(...)' code ##################

    
    # Call to sorted(...): (line 523)
    # Processing the call arguments (line 523)
    # Getting the type of '_scale_mapping' (line 523)
    _scale_mapping_131238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 18), '_scale_mapping', False)
    # Processing the call keyword arguments (line 523)
    kwargs_131239 = {}
    # Getting the type of 'sorted' (line 523)
    sorted_131237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'sorted', False)
    # Calling sorted(args, kwargs) (line 523)
    sorted_call_result_131240 = invoke(stypy.reporting.localization.Localization(__file__, 523, 11), sorted_131237, *[_scale_mapping_131238], **kwargs_131239)
    
    # Assigning a type to the variable 'stypy_return_type' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type', sorted_call_result_131240)
    
    # ################# End of 'get_scale_names(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_scale_names' in the type store
    # Getting the type of 'stypy_return_type' (line 522)
    stypy_return_type_131241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131241)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_scale_names'
    return stypy_return_type_131241

# Assigning a type to the variable 'get_scale_names' (line 522)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 0), 'get_scale_names', get_scale_names)

@norecursion
def scale_factory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'scale_factory'
    module_type_store = module_type_store.open_function_context('scale_factory', 526, 0, False)
    
    # Passed parameters checking function
    scale_factory.stypy_localization = localization
    scale_factory.stypy_type_of_self = None
    scale_factory.stypy_type_store = module_type_store
    scale_factory.stypy_function_name = 'scale_factory'
    scale_factory.stypy_param_names_list = ['scale', 'axis']
    scale_factory.stypy_varargs_param_name = None
    scale_factory.stypy_kwargs_param_name = 'kwargs'
    scale_factory.stypy_call_defaults = defaults
    scale_factory.stypy_call_varargs = varargs
    scale_factory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'scale_factory', ['scale', 'axis'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'scale_factory', localization, ['scale', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'scale_factory(...)' code ##################

    unicode_131242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, (-1)), 'unicode', u'\n    Return a scale class by name.\n\n    ACCEPTS: [ %(names)s ]\n    ')
    
    # Assigning a Call to a Name (line 532):
    
    # Call to lower(...): (line 532)
    # Processing the call keyword arguments (line 532)
    kwargs_131245 = {}
    # Getting the type of 'scale' (line 532)
    scale_131243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'scale', False)
    # Obtaining the member 'lower' of a type (line 532)
    lower_131244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 12), scale_131243, 'lower')
    # Calling lower(args, kwargs) (line 532)
    lower_call_result_131246 = invoke(stypy.reporting.localization.Localization(__file__, 532, 12), lower_131244, *[], **kwargs_131245)
    
    # Assigning a type to the variable 'scale' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'scale', lower_call_result_131246)
    
    # Type idiom detected: calculating its left and rigth part (line 533)
    # Getting the type of 'scale' (line 533)
    scale_131247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 7), 'scale')
    # Getting the type of 'None' (line 533)
    None_131248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 16), 'None')
    
    (may_be_131249, more_types_in_union_131250) = may_be_none(scale_131247, None_131248)

    if may_be_131249:

        if more_types_in_union_131250:
            # Runtime conditional SSA (line 533)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 534):
        unicode_131251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 16), 'unicode', u'linear')
        # Assigning a type to the variable 'scale' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'scale', unicode_131251)

        if more_types_in_union_131250:
            # SSA join for if statement (line 533)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'scale' (line 536)
    scale_131252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 7), 'scale')
    # Getting the type of '_scale_mapping' (line 536)
    _scale_mapping_131253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 20), '_scale_mapping')
    # Applying the binary operator 'notin' (line 536)
    result_contains_131254 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 7), 'notin', scale_131252, _scale_mapping_131253)
    
    # Testing the type of an if condition (line 536)
    if_condition_131255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 4), result_contains_131254)
    # Assigning a type to the variable 'if_condition_131255' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'if_condition_131255', if_condition_131255)
    # SSA begins for if statement (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 537)
    # Processing the call arguments (line 537)
    unicode_131257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 25), 'unicode', u"Unknown scale type '%s'")
    # Getting the type of 'scale' (line 537)
    scale_131258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 53), 'scale', False)
    # Applying the binary operator '%' (line 537)
    result_mod_131259 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 25), '%', unicode_131257, scale_131258)
    
    # Processing the call keyword arguments (line 537)
    kwargs_131260 = {}
    # Getting the type of 'ValueError' (line 537)
    ValueError_131256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 537)
    ValueError_call_result_131261 = invoke(stypy.reporting.localization.Localization(__file__, 537, 14), ValueError_131256, *[result_mod_131259], **kwargs_131260)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 537, 8), ValueError_call_result_131261, 'raise parameter', BaseException)
    # SSA join for if statement (line 536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to (...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'axis' (line 539)
    axis_131266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 33), 'axis', False)
    # Processing the call keyword arguments (line 539)
    # Getting the type of 'kwargs' (line 539)
    kwargs_131267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 41), 'kwargs', False)
    kwargs_131268 = {'kwargs_131267': kwargs_131267}
    
    # Obtaining the type of the subscript
    # Getting the type of 'scale' (line 539)
    scale_131262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 26), 'scale', False)
    # Getting the type of '_scale_mapping' (line 539)
    _scale_mapping_131263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 11), '_scale_mapping', False)
    # Obtaining the member '__getitem__' of a type (line 539)
    getitem___131264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 11), _scale_mapping_131263, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 539)
    subscript_call_result_131265 = invoke(stypy.reporting.localization.Localization(__file__, 539, 11), getitem___131264, scale_131262)
    
    # Calling (args, kwargs) (line 539)
    _call_result_131269 = invoke(stypy.reporting.localization.Localization(__file__, 539, 11), subscript_call_result_131265, *[axis_131266], **kwargs_131268)
    
    # Assigning a type to the variable 'stypy_return_type' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'stypy_return_type', _call_result_131269)
    
    # ################# End of 'scale_factory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'scale_factory' in the type store
    # Getting the type of 'stypy_return_type' (line 526)
    stypy_return_type_131270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131270)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'scale_factory'
    return stypy_return_type_131270

# Assigning a type to the variable 'scale_factory' (line 526)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 0), 'scale_factory', scale_factory)

# Assigning a BinOp to a Attribute (line 540):

# Call to dedent(...): (line 540)
# Processing the call arguments (line 540)
# Getting the type of 'scale_factory' (line 540)
scale_factory_131272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 31), 'scale_factory', False)
# Obtaining the member '__doc__' of a type (line 540)
doc___131273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 31), scale_factory_131272, '__doc__')
# Processing the call keyword arguments (line 540)
kwargs_131274 = {}
# Getting the type of 'dedent' (line 540)
dedent_131271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 24), 'dedent', False)
# Calling dedent(args, kwargs) (line 540)
dedent_call_result_131275 = invoke(stypy.reporting.localization.Localization(__file__, 540, 24), dedent_131271, *[doc___131273], **kwargs_131274)


# Obtaining an instance of the builtin type 'dict' (line 541)
dict_131276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 541)
# Adding element type (key, value) (line 541)
unicode_131277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 5), 'unicode', u'names')

# Call to join(...): (line 541)
# Processing the call arguments (line 541)

# Call to get_scale_names(...): (line 541)
# Processing the call keyword arguments (line 541)
kwargs_131281 = {}
# Getting the type of 'get_scale_names' (line 541)
get_scale_names_131280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 25), 'get_scale_names', False)
# Calling get_scale_names(args, kwargs) (line 541)
get_scale_names_call_result_131282 = invoke(stypy.reporting.localization.Localization(__file__, 541, 25), get_scale_names_131280, *[], **kwargs_131281)

# Processing the call keyword arguments (line 541)
kwargs_131283 = {}
unicode_131278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 14), 'unicode', u' | ')
# Obtaining the member 'join' of a type (line 541)
join_131279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 14), unicode_131278, 'join')
# Calling join(args, kwargs) (line 541)
join_call_result_131284 = invoke(stypy.reporting.localization.Localization(__file__, 541, 14), join_131279, *[get_scale_names_call_result_131282], **kwargs_131283)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 4), dict_131276, (unicode_131277, join_call_result_131284))

# Applying the binary operator '%' (line 540)
result_mod_131285 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 24), '%', dedent_call_result_131275, dict_131276)

# Getting the type of 'scale_factory' (line 540)
scale_factory_131286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 0), 'scale_factory')
# Setting the type of the member '__doc__' of a type (line 540)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 0), scale_factory_131286, '__doc__', result_mod_131285)

@norecursion
def register_scale(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'register_scale'
    module_type_store = module_type_store.open_function_context('register_scale', 544, 0, False)
    
    # Passed parameters checking function
    register_scale.stypy_localization = localization
    register_scale.stypy_type_of_self = None
    register_scale.stypy_type_store = module_type_store
    register_scale.stypy_function_name = 'register_scale'
    register_scale.stypy_param_names_list = ['scale_class']
    register_scale.stypy_varargs_param_name = None
    register_scale.stypy_kwargs_param_name = None
    register_scale.stypy_call_defaults = defaults
    register_scale.stypy_call_varargs = varargs
    register_scale.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'register_scale', ['scale_class'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'register_scale', localization, ['scale_class'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'register_scale(...)' code ##################

    unicode_131287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, (-1)), 'unicode', u'\n    Register a new kind of scale.\n\n    *scale_class* must be a subclass of :class:`ScaleBase`.\n    ')
    
    # Assigning a Name to a Subscript (line 550):
    # Getting the type of 'scale_class' (line 550)
    scale_class_131288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 39), 'scale_class')
    # Getting the type of '_scale_mapping' (line 550)
    _scale_mapping_131289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), '_scale_mapping')
    # Getting the type of 'scale_class' (line 550)
    scale_class_131290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'scale_class')
    # Obtaining the member 'name' of a type (line 550)
    name_131291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 19), scale_class_131290, 'name')
    # Storing an element on a container (line 550)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 4), _scale_mapping_131289, (name_131291, scale_class_131288))
    
    # ################# End of 'register_scale(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'register_scale' in the type store
    # Getting the type of 'stypy_return_type' (line 544)
    stypy_return_type_131292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131292)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'register_scale'
    return stypy_return_type_131292

# Assigning a type to the variable 'register_scale' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'register_scale', register_scale)

@norecursion
def get_scale_docs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_scale_docs'
    module_type_store = module_type_store.open_function_context('get_scale_docs', 553, 0, False)
    
    # Passed parameters checking function
    get_scale_docs.stypy_localization = localization
    get_scale_docs.stypy_type_of_self = None
    get_scale_docs.stypy_type_store = module_type_store
    get_scale_docs.stypy_function_name = 'get_scale_docs'
    get_scale_docs.stypy_param_names_list = []
    get_scale_docs.stypy_varargs_param_name = None
    get_scale_docs.stypy_kwargs_param_name = None
    get_scale_docs.stypy_call_defaults = defaults
    get_scale_docs.stypy_call_varargs = varargs
    get_scale_docs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_scale_docs', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_scale_docs', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_scale_docs(...)' code ##################

    unicode_131293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, (-1)), 'unicode', u'\n    Helper function for generating docstrings related to scales.\n    ')
    
    # Assigning a List to a Name (line 557):
    
    # Obtaining an instance of the builtin type 'list' (line 557)
    list_131294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 557)
    
    # Assigning a type to the variable 'docs' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'docs', list_131294)
    
    
    # Call to get_scale_names(...): (line 558)
    # Processing the call keyword arguments (line 558)
    kwargs_131296 = {}
    # Getting the type of 'get_scale_names' (line 558)
    get_scale_names_131295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'get_scale_names', False)
    # Calling get_scale_names(args, kwargs) (line 558)
    get_scale_names_call_result_131297 = invoke(stypy.reporting.localization.Localization(__file__, 558, 16), get_scale_names_131295, *[], **kwargs_131296)
    
    # Testing the type of a for loop iterable (line 558)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 558, 4), get_scale_names_call_result_131297)
    # Getting the type of the for loop variable (line 558)
    for_loop_var_131298 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 558, 4), get_scale_names_call_result_131297)
    # Assigning a type to the variable 'name' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'name', for_loop_var_131298)
    # SSA begins for a for statement (line 558)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 559):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 559)
    name_131299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 37), 'name')
    # Getting the type of '_scale_mapping' (line 559)
    _scale_mapping_131300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 22), '_scale_mapping')
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___131301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 22), _scale_mapping_131300, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_131302 = invoke(stypy.reporting.localization.Localization(__file__, 559, 22), getitem___131301, name_131299)
    
    # Assigning a type to the variable 'scale_class' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'scale_class', subscript_call_result_131302)
    
    # Call to append(...): (line 560)
    # Processing the call arguments (line 560)
    unicode_131305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 20), 'unicode', u"    '%s'")
    # Getting the type of 'name' (line 560)
    name_131306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 33), 'name', False)
    # Applying the binary operator '%' (line 560)
    result_mod_131307 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 20), '%', unicode_131305, name_131306)
    
    # Processing the call keyword arguments (line 560)
    kwargs_131308 = {}
    # Getting the type of 'docs' (line 560)
    docs_131303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'docs', False)
    # Obtaining the member 'append' of a type (line 560)
    append_131304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), docs_131303, 'append')
    # Calling append(args, kwargs) (line 560)
    append_call_result_131309 = invoke(stypy.reporting.localization.Localization(__file__, 560, 8), append_131304, *[result_mod_131307], **kwargs_131308)
    
    
    # Call to append(...): (line 561)
    # Processing the call arguments (line 561)
    unicode_131312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 20), 'unicode', u'')
    # Processing the call keyword arguments (line 561)
    kwargs_131313 = {}
    # Getting the type of 'docs' (line 561)
    docs_131310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'docs', False)
    # Obtaining the member 'append' of a type (line 561)
    append_131311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 8), docs_131310, 'append')
    # Calling append(args, kwargs) (line 561)
    append_call_result_131314 = invoke(stypy.reporting.localization.Localization(__file__, 561, 8), append_131311, *[unicode_131312], **kwargs_131313)
    
    
    # Assigning a Call to a Name (line 562):
    
    # Call to dedent(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'scale_class' (line 562)
    scale_class_131316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 28), 'scale_class', False)
    # Obtaining the member '__init__' of a type (line 562)
    init___131317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 28), scale_class_131316, '__init__')
    # Obtaining the member '__doc__' of a type (line 562)
    doc___131318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 28), init___131317, '__doc__')
    # Processing the call keyword arguments (line 562)
    kwargs_131319 = {}
    # Getting the type of 'dedent' (line 562)
    dedent_131315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 21), 'dedent', False)
    # Calling dedent(args, kwargs) (line 562)
    dedent_call_result_131320 = invoke(stypy.reporting.localization.Localization(__file__, 562, 21), dedent_131315, *[doc___131318], **kwargs_131319)
    
    # Assigning a type to the variable 'class_docs' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'class_docs', dedent_call_result_131320)
    
    # Assigning a Call to a Name (line 563):
    
    # Call to join(...): (line 563)
    # Processing the call arguments (line 563)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 564)
    # Processing the call arguments (line 564)
    unicode_131328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 58), 'unicode', u'\n')
    # Processing the call keyword arguments (line 564)
    kwargs_131329 = {}
    # Getting the type of 'class_docs' (line 564)
    class_docs_131326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 41), 'class_docs', False)
    # Obtaining the member 'split' of a type (line 564)
    split_131327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 41), class_docs_131326, 'split')
    # Calling split(args, kwargs) (line 564)
    split_call_result_131330 = invoke(stypy.reporting.localization.Localization(__file__, 564, 41), split_131327, *[unicode_131328], **kwargs_131329)
    
    comprehension_131331 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 30), split_call_result_131330)
    # Assigning a type to the variable 'x' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 30), 'x', comprehension_131331)
    unicode_131323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 30), 'unicode', u'        %s\n')
    # Getting the type of 'x' (line 564)
    x_131324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 30), 'x', False)
    # Applying the binary operator '%' (line 563)
    result_mod_131325 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 30), '%', unicode_131323, x_131324)
    
    list_131332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 30), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 30), list_131332, result_mod_131325)
    # Processing the call keyword arguments (line 563)
    kwargs_131333 = {}
    unicode_131321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 21), 'unicode', u'')
    # Obtaining the member 'join' of a type (line 563)
    join_131322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 21), unicode_131321, 'join')
    # Calling join(args, kwargs) (line 563)
    join_call_result_131334 = invoke(stypy.reporting.localization.Localization(__file__, 563, 21), join_131322, *[list_131332], **kwargs_131333)
    
    # Assigning a type to the variable 'class_docs' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'class_docs', join_call_result_131334)
    
    # Call to append(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'class_docs' (line 565)
    class_docs_131337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 20), 'class_docs', False)
    # Processing the call keyword arguments (line 565)
    kwargs_131338 = {}
    # Getting the type of 'docs' (line 565)
    docs_131335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'docs', False)
    # Obtaining the member 'append' of a type (line 565)
    append_131336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 8), docs_131335, 'append')
    # Calling append(args, kwargs) (line 565)
    append_call_result_131339 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), append_131336, *[class_docs_131337], **kwargs_131338)
    
    
    # Call to append(...): (line 566)
    # Processing the call arguments (line 566)
    unicode_131342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 20), 'unicode', u'')
    # Processing the call keyword arguments (line 566)
    kwargs_131343 = {}
    # Getting the type of 'docs' (line 566)
    docs_131340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'docs', False)
    # Obtaining the member 'append' of a type (line 566)
    append_131341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), docs_131340, 'append')
    # Calling append(args, kwargs) (line 566)
    append_call_result_131344 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), append_131341, *[unicode_131342], **kwargs_131343)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'docs' (line 567)
    docs_131347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 'docs', False)
    # Processing the call keyword arguments (line 567)
    kwargs_131348 = {}
    unicode_131345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 11), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 567)
    join_131346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 11), unicode_131345, 'join')
    # Calling join(args, kwargs) (line 567)
    join_call_result_131349 = invoke(stypy.reporting.localization.Localization(__file__, 567, 11), join_131346, *[docs_131347], **kwargs_131348)
    
    # Assigning a type to the variable 'stypy_return_type' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'stypy_return_type', join_call_result_131349)
    
    # ################# End of 'get_scale_docs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_scale_docs' in the type store
    # Getting the type of 'stypy_return_type' (line 553)
    stypy_return_type_131350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131350)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_scale_docs'
    return stypy_return_type_131350

# Assigning a type to the variable 'get_scale_docs' (line 553)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 0), 'get_scale_docs', get_scale_docs)

# Call to update(...): (line 570)
# Processing the call keyword arguments (line 570)

# Call to join(...): (line 571)
# Processing the call arguments (line 571)
# Calculating list comprehension
# Calculating comprehension expression

# Call to get_scale_names(...): (line 571)
# Processing the call keyword arguments (line 571)
kwargs_131361 = {}
# Getting the type of 'get_scale_names' (line 571)
get_scale_names_131360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 39), 'get_scale_names', False)
# Calling get_scale_names(args, kwargs) (line 571)
get_scale_names_call_result_131362 = invoke(stypy.reporting.localization.Localization(__file__, 571, 39), get_scale_names_131360, *[], **kwargs_131361)

comprehension_131363 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 22), get_scale_names_call_result_131362)
# Assigning a type to the variable 'x' (line 571)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 22), 'x', comprehension_131363)

# Call to repr(...): (line 571)
# Processing the call arguments (line 571)
# Getting the type of 'x' (line 571)
x_131357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 27), 'x', False)
# Processing the call keyword arguments (line 571)
kwargs_131358 = {}
# Getting the type of 'repr' (line 571)
repr_131356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 22), 'repr', False)
# Calling repr(args, kwargs) (line 571)
repr_call_result_131359 = invoke(stypy.reporting.localization.Localization(__file__, 571, 22), repr_131356, *[x_131357], **kwargs_131358)

list_131364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 22), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 22), list_131364, repr_call_result_131359)
# Processing the call keyword arguments (line 571)
kwargs_131365 = {}
unicode_131354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 10), 'unicode', u' | ')
# Obtaining the member 'join' of a type (line 571)
join_131355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 10), unicode_131354, 'join')
# Calling join(args, kwargs) (line 571)
join_call_result_131366 = invoke(stypy.reporting.localization.Localization(__file__, 571, 10), join_131355, *[list_131364], **kwargs_131365)

keyword_131367 = join_call_result_131366

# Call to rstrip(...): (line 572)
# Processing the call keyword arguments (line 572)
kwargs_131372 = {}

# Call to get_scale_docs(...): (line 572)
# Processing the call keyword arguments (line 572)
kwargs_131369 = {}
# Getting the type of 'get_scale_docs' (line 572)
get_scale_docs_131368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'get_scale_docs', False)
# Calling get_scale_docs(args, kwargs) (line 572)
get_scale_docs_call_result_131370 = invoke(stypy.reporting.localization.Localization(__file__, 572, 15), get_scale_docs_131368, *[], **kwargs_131369)

# Obtaining the member 'rstrip' of a type (line 572)
rstrip_131371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 15), get_scale_docs_call_result_131370, 'rstrip')
# Calling rstrip(args, kwargs) (line 572)
rstrip_call_result_131373 = invoke(stypy.reporting.localization.Localization(__file__, 572, 15), rstrip_131371, *[], **kwargs_131372)

keyword_131374 = rstrip_call_result_131373
kwargs_131375 = {'scale_docs': keyword_131374, 'scale': keyword_131367}
# Getting the type of 'docstring' (line 570)
docstring_131351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 0), 'docstring', False)
# Obtaining the member 'interpd' of a type (line 570)
interpd_131352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 0), docstring_131351, 'interpd')
# Obtaining the member 'update' of a type (line 570)
update_131353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 0), interpd_131352, 'update')
# Calling update(args, kwargs) (line 570)
update_call_result_131376 = invoke(stypy.reporting.localization.Localization(__file__, 570, 0), update_131353, *[], **kwargs_131375)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
