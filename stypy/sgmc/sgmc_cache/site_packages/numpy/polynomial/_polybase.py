
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Abstract base class for the various polynomial Classes.
3: 
4: The ABCPolyBase class provides the methods needed to implement the common API
5: for the various polynomial classes. It operates as a mixin, but uses the
6: abc module from the stdlib, hence it is only available for Python >= 2.6.
7: 
8: '''
9: from __future__ import division, absolute_import, print_function
10: 
11: from abc import ABCMeta, abstractmethod, abstractproperty
12: from numbers import Number
13: 
14: import numpy as np
15: from . import polyutils as pu
16: 
17: __all__ = ['ABCPolyBase']
18: 
19: class ABCPolyBase(object):
20:     '''An abstract base class for series classes.
21: 
22:     ABCPolyBase provides the standard Python numerical methods
23:     '+', '-', '*', '//', '%', 'divmod', '**', and '()' along with the
24:     methods listed below.
25: 
26:     .. versionadded:: 1.9.0
27: 
28:     Parameters
29:     ----------
30:     coef : array_like
31:         Series coefficients in order of increasing degree, i.e.,
32:         ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``, where
33:         ``P_i`` is the basis polynomials of degree ``i``.
34:     domain : (2,) array_like, optional
35:         Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
36:         to the interval ``[window[0], window[1]]`` by shifting and scaling.
37:         The default value is the derived class domain.
38:     window : (2,) array_like, optional
39:         Window, see domain for its use. The default value is the
40:         derived class window.
41: 
42:     Attributes
43:     ----------
44:     coef : (N,) ndarray
45:         Series coefficients in order of increasing degree.
46:     domain : (2,) ndarray
47:         Domain that is mapped to window.
48:     window : (2,) ndarray
49:         Window that domain is mapped to.
50: 
51:     Class Attributes
52:     ----------------
53:     maxpower : int
54:         Maximum power allowed, i.e., the largest number ``n`` such that
55:         ``p(x)**n`` is allowed. This is to limit runaway polynomial size.
56:     domain : (2,) ndarray
57:         Default domain of the class.
58:     window : (2,) ndarray
59:         Default window of the class.
60: 
61:     '''
62:     __metaclass__ = ABCMeta
63: 
64:     # Not hashable
65:     __hash__ = None
66: 
67:     # Don't let participate in array operations. Value doesn't matter.
68:     __array_priority__ = 1000
69: 
70:     # Limit runaway size. T_n^m has degree n*m
71:     maxpower = 100
72: 
73:     @abstractproperty
74:     def domain(self):
75:         pass
76: 
77:     @abstractproperty
78:     def window(self):
79:         pass
80: 
81:     @abstractproperty
82:     def nickname(self):
83:         pass
84: 
85:     @abstractmethod
86:     def _add(self):
87:         pass
88: 
89:     @abstractmethod
90:     def _sub(self):
91:         pass
92: 
93:     @abstractmethod
94:     def _mul(self):
95:         pass
96: 
97:     @abstractmethod
98:     def _div(self):
99:         pass
100: 
101:     @abstractmethod
102:     def _pow(self):
103:         pass
104: 
105:     @abstractmethod
106:     def _val(self):
107:         pass
108: 
109:     @abstractmethod
110:     def _int(self):
111:         pass
112: 
113:     @abstractmethod
114:     def _der(self):
115:         pass
116: 
117:     @abstractmethod
118:     def _fit(self):
119:         pass
120: 
121:     @abstractmethod
122:     def _line(self):
123:         pass
124: 
125:     @abstractmethod
126:     def _roots(self):
127:         pass
128: 
129:     @abstractmethod
130:     def _fromroots(self):
131:         pass
132: 
133:     def has_samecoef(self, other):
134:         '''Check if coefficients match.
135: 
136:         .. versionadded:: 1.6.0
137: 
138:         Parameters
139:         ----------
140:         other : class instance
141:             The other class must have the ``coef`` attribute.
142: 
143:         Returns
144:         -------
145:         bool : boolean
146:             True if the coefficients are the same, False otherwise.
147: 
148:         '''
149:         if len(self.coef) != len(other.coef):
150:             return False
151:         elif not np.all(self.coef == other.coef):
152:             return False
153:         else:
154:             return True
155: 
156:     def has_samedomain(self, other):
157:         '''Check if domains match.
158: 
159:         .. versionadded:: 1.6.0
160: 
161:         Parameters
162:         ----------
163:         other : class instance
164:             The other class must have the ``domain`` attribute.
165: 
166:         Returns
167:         -------
168:         bool : boolean
169:             True if the domains are the same, False otherwise.
170: 
171:         '''
172:         return np.all(self.domain == other.domain)
173: 
174:     def has_samewindow(self, other):
175:         '''Check if windows match.
176: 
177:         .. versionadded:: 1.6.0
178: 
179:         Parameters
180:         ----------
181:         other : class instance
182:             The other class must have the ``window`` attribute.
183: 
184:         Returns
185:         -------
186:         bool : boolean
187:             True if the windows are the same, False otherwise.
188: 
189:         '''
190:         return np.all(self.window == other.window)
191: 
192:     def has_sametype(self, other):
193:         '''Check if types match.
194: 
195:         .. versionadded:: 1.7.0
196: 
197:         Parameters
198:         ----------
199:         other : object
200:             Class instance.
201: 
202:         Returns
203:         -------
204:         bool : boolean
205:             True if other is same class as self
206: 
207:         '''
208:         return isinstance(other, self.__class__)
209: 
210:     def _get_coefficients(self, other):
211:         '''Interpret other as polynomial coefficients.
212: 
213:         The `other` argument is checked to see if it is of the same
214:         class as self with identical domain and window. If so,
215:         return its coefficients, otherwise return `other`.
216: 
217:         .. versionadded:: 1.9.0
218: 
219:         Parameters
220:         ----------
221:         other : anything
222:             Object to be checked.
223: 
224:         Returns
225:         -------
226:         coef:
227:             The coefficients of`other` if it is a compatible instance,
228:             of ABCPolyBase, otherwise `other`.
229: 
230:         Raises
231:         ------
232:         TypeError:
233:             When `other` is an incompatible instance of ABCPolyBase.
234: 
235:         '''
236:         if isinstance(other, ABCPolyBase):
237:             if not isinstance(other, self.__class__):
238:                 raise TypeError("Polynomial types differ")
239:             elif not np.all(self.domain == other.domain):
240:                 raise TypeError("Domains differ")
241:             elif not np.all(self.window == other.window):
242:                 raise TypeError("Windows differ")
243:             return other.coef
244:         return other
245: 
246:     def __init__(self, coef, domain=None, window=None):
247:         [coef] = pu.as_series([coef], trim=False)
248:         self.coef = coef
249: 
250:         if domain is not None:
251:             [domain] = pu.as_series([domain], trim=False)
252:             if len(domain) != 2:
253:                 raise ValueError("Domain has wrong number of elements.")
254:             self.domain = domain
255: 
256:         if window is not None:
257:             [window] = pu.as_series([window], trim=False)
258:             if len(window) != 2:
259:                 raise ValueError("Window has wrong number of elements.")
260:             self.window = window
261: 
262:     def __repr__(self):
263:         format = "%s(%s, %s, %s)"
264:         coef = repr(self.coef)[6:-1]
265:         domain = repr(self.domain)[6:-1]
266:         window = repr(self.window)[6:-1]
267:         name = self.__class__.__name__
268:         return format % (name, coef, domain, window)
269: 
270:     def __str__(self):
271:         format = "%s(%s)"
272:         coef = str(self.coef)
273:         name = self.nickname
274:         return format % (name, coef)
275: 
276:     # Pickle and copy
277: 
278:     def __getstate__(self):
279:         ret = self.__dict__.copy()
280:         ret['coef'] = self.coef.copy()
281:         ret['domain'] = self.domain.copy()
282:         ret['window'] = self.window.copy()
283:         return ret
284: 
285:     def __setstate__(self, dict):
286:         self.__dict__ = dict
287: 
288:     # Call
289: 
290:     def __call__(self, arg):
291:         off, scl = pu.mapparms(self.domain, self.window)
292:         arg = off + scl*arg
293:         return self._val(arg, self.coef)
294: 
295:     def __iter__(self):
296:         return iter(self.coef)
297: 
298:     def __len__(self):
299:         return len(self.coef)
300: 
301:     # Numeric properties.
302: 
303:     def __neg__(self):
304:         return self.__class__(-self.coef, self.domain, self.window)
305: 
306:     def __pos__(self):
307:         return self
308: 
309:     def __add__(self, other):
310:         try:
311:             othercoef = self._get_coefficients(other)
312:             coef = self._add(self.coef, othercoef)
313:         except TypeError as e:
314:             raise e
315:         except:
316:             return NotImplemented
317:         return self.__class__(coef, self.domain, self.window)
318: 
319:     def __sub__(self, other):
320:         try:
321:             othercoef = self._get_coefficients(other)
322:             coef = self._sub(self.coef, othercoef)
323:         except TypeError as e:
324:             raise e
325:         except:
326:             return NotImplemented
327:         return self.__class__(coef, self.domain, self.window)
328: 
329:     def __mul__(self, other):
330:         try:
331:             othercoef = self._get_coefficients(other)
332:             coef = self._mul(self.coef, othercoef)
333:         except TypeError as e:
334:             raise e
335:         except:
336:             return NotImplemented
337:         return self.__class__(coef, self.domain, self.window)
338: 
339:     def __div__(self, other):
340:         # set to __floordiv__,  /, for now.
341:         return self.__floordiv__(other)
342: 
343:     def __truediv__(self, other):
344:         # there is no true divide if the rhs is not a Number, although it
345:         # could return the first n elements of an infinite series.
346:         # It is hard to see where n would come from, though.
347:         if not isinstance(other, Number) or isinstance(other, bool):
348:             form = "unsupported types for true division: '%s', '%s'"
349:             raise TypeError(form % (type(self), type(other)))
350:         return self.__floordiv__(other)
351: 
352:     def __floordiv__(self, other):
353:         res = self.__divmod__(other)
354:         if res is NotImplemented:
355:             return res
356:         return res[0]
357: 
358:     def __mod__(self, other):
359:         res = self.__divmod__(other)
360:         if res is NotImplemented:
361:             return res
362:         return res[1]
363: 
364:     def __divmod__(self, other):
365:         try:
366:             othercoef = self._get_coefficients(other)
367:             quo, rem = self._div(self.coef, othercoef)
368:         except (TypeError, ZeroDivisionError) as e:
369:             raise e
370:         except:
371:             return NotImplemented
372:         quo = self.__class__(quo, self.domain, self.window)
373:         rem = self.__class__(rem, self.domain, self.window)
374:         return quo, rem
375: 
376:     def __pow__(self, other):
377:         coef = self._pow(self.coef, other, maxpower=self.maxpower)
378:         res = self.__class__(coef, self.domain, self.window)
379:         return res
380: 
381:     def __radd__(self, other):
382:         try:
383:             coef = self._add(other, self.coef)
384:         except:
385:             return NotImplemented
386:         return self.__class__(coef, self.domain, self.window)
387: 
388:     def __rsub__(self, other):
389:         try:
390:             coef = self._sub(other, self.coef)
391:         except:
392:             return NotImplemented
393:         return self.__class__(coef, self.domain, self.window)
394: 
395:     def __rmul__(self, other):
396:         try:
397:             coef = self._mul(other, self.coef)
398:         except:
399:             return NotImplemented
400:         return self.__class__(coef, self.domain, self.window)
401: 
402:     def __rdiv__(self, other):
403:         # set to __floordiv__ /.
404:         return self.__rfloordiv__(other)
405: 
406:     def __rtruediv__(self, other):
407:         # An instance of ABCPolyBase is not considered a
408:         # Number.
409:         return NotImplemented
410: 
411:     def __rfloordiv__(self, other):
412:         res = self.__rdivmod__(other)
413:         if res is NotImplemented:
414:             return res
415:         return res[0]
416: 
417:     def __rmod__(self, other):
418:         res = self.__rdivmod__(other)
419:         if res is NotImplemented:
420:             return res
421:         return res[1]
422: 
423:     def __rdivmod__(self, other):
424:         try:
425:             quo, rem = self._div(other, self.coef)
426:         except ZeroDivisionError as e:
427:             raise e
428:         except:
429:             return NotImplemented
430:         quo = self.__class__(quo, self.domain, self.window)
431:         rem = self.__class__(rem, self.domain, self.window)
432:         return quo, rem
433: 
434:     # Enhance me
435:     # some augmented arithmetic operations could be added here
436: 
437:     def __eq__(self, other):
438:         res = (isinstance(other, self.__class__) and
439:                np.all(self.domain == other.domain) and
440:                np.all(self.window == other.window) and
441:                (self.coef.shape == other.coef.shape) and
442:                np.all(self.coef == other.coef))
443:         return res
444: 
445:     def __ne__(self, other):
446:         return not self.__eq__(other)
447: 
448:     #
449:     # Extra methods.
450:     #
451: 
452:     def copy(self):
453:         '''Return a copy.
454: 
455:         Returns
456:         -------
457:         new_series : series
458:             Copy of self.
459: 
460:         '''
461:         return self.__class__(self.coef, self.domain, self.window)
462: 
463:     def degree(self):
464:         '''The degree of the series.
465: 
466:         .. versionadded:: 1.5.0
467: 
468:         Returns
469:         -------
470:         degree : int
471:             Degree of the series, one less than the number of coefficients.
472: 
473:         '''
474:         return len(self) - 1
475: 
476:     def cutdeg(self, deg):
477:         '''Truncate series to the given degree.
478: 
479:         Reduce the degree of the series to `deg` by discarding the
480:         high order terms. If `deg` is greater than the current degree a
481:         copy of the current series is returned. This can be useful in least
482:         squares where the coefficients of the high degree terms may be very
483:         small.
484: 
485:         .. versionadded:: 1.5.0
486: 
487:         Parameters
488:         ----------
489:         deg : non-negative int
490:             The series is reduced to degree `deg` by discarding the high
491:             order terms. The value of `deg` must be a non-negative integer.
492: 
493:         Returns
494:         -------
495:         new_series : series
496:             New instance of series with reduced degree.
497: 
498:         '''
499:         return self.truncate(deg + 1)
500: 
501:     def trim(self, tol=0):
502:         '''Remove trailing coefficients
503: 
504:         Remove trailing coefficients until a coefficient is reached whose
505:         absolute value greater than `tol` or the beginning of the series is
506:         reached. If all the coefficients would be removed the series is set
507:         to ``[0]``. A new series instance is returned with the new
508:         coefficients.  The current instance remains unchanged.
509: 
510:         Parameters
511:         ----------
512:         tol : non-negative number.
513:             All trailing coefficients less than `tol` will be removed.
514: 
515:         Returns
516:         -------
517:         new_series : series
518:             Contains the new set of coefficients.
519: 
520:         '''
521:         coef = pu.trimcoef(self.coef, tol)
522:         return self.__class__(coef, self.domain, self.window)
523: 
524:     def truncate(self, size):
525:         '''Truncate series to length `size`.
526: 
527:         Reduce the series to length `size` by discarding the high
528:         degree terms. The value of `size` must be a positive integer. This
529:         can be useful in least squares where the coefficients of the
530:         high degree terms may be very small.
531: 
532:         Parameters
533:         ----------
534:         size : positive int
535:             The series is reduced to length `size` by discarding the high
536:             degree terms. The value of `size` must be a positive integer.
537: 
538:         Returns
539:         -------
540:         new_series : series
541:             New instance of series with truncated coefficients.
542: 
543:         '''
544:         isize = int(size)
545:         if isize != size or isize < 1:
546:             raise ValueError("size must be a positive integer")
547:         if isize >= len(self.coef):
548:             coef = self.coef
549:         else:
550:             coef = self.coef[:isize]
551:         return self.__class__(coef, self.domain, self.window)
552: 
553:     def convert(self, domain=None, kind=None, window=None):
554:         '''Convert series to a different kind and/or domain and/or window.
555: 
556:         Parameters
557:         ----------
558:         domain : array_like, optional
559:             The domain of the converted series. If the value is None,
560:             the default domain of `kind` is used.
561:         kind : class, optional
562:             The polynomial series type class to which the current instance
563:             should be converted. If kind is None, then the class of the
564:             current instance is used.
565:         window : array_like, optional
566:             The window of the converted series. If the value is None,
567:             the default window of `kind` is used.
568: 
569:         Returns
570:         -------
571:         new_series : series
572:             The returned class can be of different type than the current
573:             instance and/or have a different domain and/or different
574:             window.
575: 
576:         Notes
577:         -----
578:         Conversion between domains and class types can result in
579:         numerically ill defined series.
580: 
581:         Examples
582:         --------
583: 
584:         '''
585:         if kind is None:
586:             kind = self.__class__
587:         if domain is None:
588:             domain = kind.domain
589:         if window is None:
590:             window = kind.window
591:         return self(kind.identity(domain, window=window))
592: 
593:     def mapparms(self):
594:         '''Return the mapping parameters.
595: 
596:         The returned values define a linear map ``off + scl*x`` that is
597:         applied to the input arguments before the series is evaluated. The
598:         map depends on the ``domain`` and ``window``; if the current
599:         ``domain`` is equal to the ``window`` the resulting map is the
600:         identity.  If the coefficients of the series instance are to be
601:         used by themselves outside this class, then the linear function
602:         must be substituted for the ``x`` in the standard representation of
603:         the base polynomials.
604: 
605:         Returns
606:         -------
607:         off, scl : float or complex
608:             The mapping function is defined by ``off + scl*x``.
609: 
610:         Notes
611:         -----
612:         If the current domain is the interval ``[l1, r1]`` and the window
613:         is ``[l2, r2]``, then the linear mapping function ``L`` is
614:         defined by the equations::
615: 
616:             L(l1) = l2
617:             L(r1) = r2
618: 
619:         '''
620:         return pu.mapparms(self.domain, self.window)
621: 
622:     def integ(self, m=1, k=[], lbnd=None):
623:         '''Integrate.
624: 
625:         Return a series instance that is the definite integral of the
626:         current series.
627: 
628:         Parameters
629:         ----------
630:         m : non-negative int
631:             The number of integrations to perform.
632:         k : array_like
633:             Integration constants. The first constant is applied to the
634:             first integration, the second to the second, and so on. The
635:             list of values must less than or equal to `m` in length and any
636:             missing values are set to zero.
637:         lbnd : Scalar
638:             The lower bound of the definite integral.
639: 
640:         Returns
641:         -------
642:         new_series : series
643:             A new series representing the integral. The domain is the same
644:             as the domain of the integrated series.
645: 
646:         '''
647:         off, scl = self.mapparms()
648:         if lbnd is None:
649:             lbnd = 0
650:         else:
651:             lbnd = off + scl*lbnd
652:         coef = self._int(self.coef, m, k, lbnd, 1./scl)
653:         return self.__class__(coef, self.domain, self.window)
654: 
655:     def deriv(self, m=1):
656:         '''Differentiate.
657: 
658:         Return a series instance of that is the derivative of the current
659:         series.
660: 
661:         Parameters
662:         ----------
663:         m : non-negative int
664:             Find the derivative of order `m`.
665: 
666:         Returns
667:         -------
668:         new_series : series
669:             A new series representing the derivative. The domain is the same
670:             as the domain of the differentiated series.
671: 
672:         '''
673:         off, scl = self.mapparms()
674:         coef = self._der(self.coef, m, scl)
675:         return self.__class__(coef, self.domain, self.window)
676: 
677:     def roots(self):
678:         '''Return the roots of the series polynomial.
679: 
680:         Compute the roots for the series. Note that the accuracy of the
681:         roots decrease the further outside the domain they lie.
682: 
683:         Returns
684:         -------
685:         roots : ndarray
686:             Array containing the roots of the series.
687: 
688:         '''
689:         roots = self._roots(self.coef)
690:         return pu.mapdomain(roots, self.window, self.domain)
691: 
692:     def linspace(self, n=100, domain=None):
693:         '''Return x, y values at equally spaced points in domain.
694: 
695:         Returns the x, y values at `n` linearly spaced points across the
696:         domain.  Here y is the value of the polynomial at the points x. By
697:         default the domain is the same as that of the series instance.
698:         This method is intended mostly as a plotting aid.
699: 
700:         .. versionadded:: 1.5.0
701: 
702:         Parameters
703:         ----------
704:         n : int, optional
705:             Number of point pairs to return. The default value is 100.
706:         domain : {None, array_like}, optional
707:             If not None, the specified domain is used instead of that of
708:             the calling instance. It should be of the form ``[beg,end]``.
709:             The default is None which case the class domain is used.
710: 
711:         Returns
712:         -------
713:         x, y : ndarray
714:             x is equal to linspace(self.domain[0], self.domain[1], n) and
715:             y is the series evaluated at element of x.
716: 
717:         '''
718:         if domain is None:
719:             domain = self.domain
720:         x = np.linspace(domain[0], domain[1], n)
721:         y = self(x)
722:         return x, y
723: 
724:     @classmethod
725:     def fit(cls, x, y, deg, domain=None, rcond=None, full=False, w=None,
726:         window=None):
727:         '''Least squares fit to data.
728: 
729:         Return a series instance that is the least squares fit to the data
730:         `y` sampled at `x`. The domain of the returned instance can be
731:         specified and this will often result in a superior fit with less
732:         chance of ill conditioning.
733: 
734:         Parameters
735:         ----------
736:         x : array_like, shape (M,)
737:             x-coordinates of the M sample points ``(x[i], y[i])``.
738:         y : array_like, shape (M,) or (M, K)
739:             y-coordinates of the sample points. Several data sets of sample
740:             points sharing the same x-coordinates can be fitted at once by
741:             passing in a 2D-array that contains one dataset per column.
742:         deg : int or 1-D array_like
743:             Degree(s) of the fitting polynomials. If `deg` is a single integer
744:             all terms up to and including the `deg`'th term are included in the
745:             fit. For Numpy versions >= 1.11 a list of integers specifying the
746:             degrees of the terms to include may be used instead.
747:         domain : {None, [beg, end], []}, optional
748:             Domain to use for the returned series. If ``None``,
749:             then a minimal domain that covers the points `x` is chosen.  If
750:             ``[]`` the class domain is used. The default value was the
751:             class domain in NumPy 1.4 and ``None`` in later versions.
752:             The ``[]`` option was added in numpy 1.5.0.
753:         rcond : float, optional
754:             Relative condition number of the fit. Singular values smaller
755:             than this relative to the largest singular value will be
756:             ignored. The default value is len(x)*eps, where eps is the
757:             relative precision of the float type, about 2e-16 in most
758:             cases.
759:         full : bool, optional
760:             Switch determining nature of return value. When it is False
761:             (the default) just the coefficients are returned, when True
762:             diagnostic information from the singular value decomposition is
763:             also returned.
764:         w : array_like, shape (M,), optional
765:             Weights. If not None the contribution of each point
766:             ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
767:             weights are chosen so that the errors of the products
768:             ``w[i]*y[i]`` all have the same variance.  The default value is
769:             None.
770: 
771:             .. versionadded:: 1.5.0
772:         window : {[beg, end]}, optional
773:             Window to use for the returned series. The default
774:             value is the default class domain
775: 
776:             .. versionadded:: 1.6.0
777: 
778:         Returns
779:         -------
780:         new_series : series
781:             A series that represents the least squares fit to the data and
782:             has the domain specified in the call.
783: 
784:         [resid, rank, sv, rcond] : list
785:             These values are only returned if `full` = True
786: 
787:             resid -- sum of squared residuals of the least squares fit
788:             rank -- the numerical rank of the scaled Vandermonde matrix
789:             sv -- singular values of the scaled Vandermonde matrix
790:             rcond -- value of `rcond`.
791: 
792:             For more details, see `linalg.lstsq`.
793: 
794:         '''
795:         if domain is None:
796:             domain = pu.getdomain(x)
797:         elif type(domain) is list and len(domain) == 0:
798:             domain = cls.domain
799: 
800:         if window is None:
801:             window = cls.window
802: 
803:         xnew = pu.mapdomain(x, domain, window)
804:         res = cls._fit(xnew, y, deg, w=w, rcond=rcond, full=full)
805:         if full:
806:             [coef, status] = res
807:             return cls(coef, domain=domain, window=window), status
808:         else:
809:             coef = res
810:             return cls(coef, domain=domain, window=window)
811: 
812:     @classmethod
813:     def fromroots(cls, roots, domain=[], window=None):
814:         '''Return series instance that has the specified roots.
815: 
816:         Returns a series representing the product
817:         ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is a
818:         list of roots.
819: 
820:         Parameters
821:         ----------
822:         roots : array_like
823:             List of roots.
824:         domain : {[], None, array_like}, optional
825:             Domain for the resulting series. If None the domain is the
826:             interval from the smallest root to the largest. If [] the
827:             domain is the class domain. The default is [].
828:         window : {None, array_like}, optional
829:             Window for the returned series. If None the class window is
830:             used. The default is None.
831: 
832:         Returns
833:         -------
834:         new_series : series
835:             Series with the specified roots.
836: 
837:         '''
838:         [roots] = pu.as_series([roots], trim=False)
839:         if domain is None:
840:             domain = pu.getdomain(roots)
841:         elif type(domain) is list and len(domain) == 0:
842:             domain = cls.domain
843: 
844:         if window is None:
845:             window = cls.window
846: 
847:         deg = len(roots)
848:         off, scl = pu.mapparms(domain, window)
849:         rnew = off + scl*roots
850:         coef = cls._fromroots(rnew) / scl**deg
851:         return cls(coef, domain=domain, window=window)
852: 
853:     @classmethod
854:     def identity(cls, domain=None, window=None):
855:         '''Identity function.
856: 
857:         If ``p`` is the returned series, then ``p(x) == x`` for all
858:         values of x.
859: 
860:         Parameters
861:         ----------
862:         domain : {None, array_like}, optional
863:             If given, the array must be of the form ``[beg, end]``, where
864:             ``beg`` and ``end`` are the endpoints of the domain. If None is
865:             given then the class domain is used. The default is None.
866:         window : {None, array_like}, optional
867:             If given, the resulting array must be if the form
868:             ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
869:             the window. If None is given then the class window is used. The
870:             default is None.
871: 
872:         Returns
873:         -------
874:         new_series : series
875:              Series of representing the identity.
876: 
877:         '''
878:         if domain is None:
879:             domain = cls.domain
880:         if window is None:
881:             window = cls.window
882:         off, scl = pu.mapparms(window, domain)
883:         coef = cls._line(off, scl)
884:         return cls(coef, domain, window)
885: 
886:     @classmethod
887:     def basis(cls, deg, domain=None, window=None):
888:         '''Series basis polynomial of degree `deg`.
889: 
890:         Returns the series representing the basis polynomial of degree `deg`.
891: 
892:         .. versionadded:: 1.7.0
893: 
894:         Parameters
895:         ----------
896:         deg : int
897:             Degree of the basis polynomial for the series. Must be >= 0.
898:         domain : {None, array_like}, optional
899:             If given, the array must be of the form ``[beg, end]``, where
900:             ``beg`` and ``end`` are the endpoints of the domain. If None is
901:             given then the class domain is used. The default is None.
902:         window : {None, array_like}, optional
903:             If given, the resulting array must be if the form
904:             ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
905:             the window. If None is given then the class window is used. The
906:             default is None.
907: 
908:         Returns
909:         -------
910:         new_series : series
911:             A series with the coefficient of the `deg` term set to one and
912:             all others zero.
913: 
914:         '''
915:         if domain is None:
916:             domain = cls.domain
917:         if window is None:
918:             window = cls.window
919:         ideg = int(deg)
920: 
921:         if ideg != deg or ideg < 0:
922:             raise ValueError("deg must be non-negative integer")
923:         return cls([0]*ideg + [1], domain, window)
924: 
925:     @classmethod
926:     def cast(cls, series, domain=None, window=None):
927:         '''Convert series to series of this class.
928: 
929:         The `series` is expected to be an instance of some polynomial
930:         series of one of the types supported by by the numpy.polynomial
931:         module, but could be some other class that supports the convert
932:         method.
933: 
934:         .. versionadded:: 1.7.0
935: 
936:         Parameters
937:         ----------
938:         series : series
939:             The series instance to be converted.
940:         domain : {None, array_like}, optional
941:             If given, the array must be of the form ``[beg, end]``, where
942:             ``beg`` and ``end`` are the endpoints of the domain. If None is
943:             given then the class domain is used. The default is None.
944:         window : {None, array_like}, optional
945:             If given, the resulting array must be if the form
946:             ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
947:             the window. If None is given then the class window is used. The
948:             default is None.
949: 
950:         Returns
951:         -------
952:         new_series : series
953:             A series of the same kind as the calling class and equal to
954:             `series` when evaluated.
955: 
956:         See Also
957:         --------
958:         convert : similar instance method
959: 
960:         '''
961:         if domain is None:
962:             domain = cls.domain
963:         if window is None:
964:             window = cls.window
965:         return series.convert(domain, cls, window)
966: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_179213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nAbstract base class for the various polynomial Classes.\n\nThe ABCPolyBase class provides the methods needed to implement the common API\nfor the various polynomial classes. It operates as a mixin, but uses the\nabc module from the stdlib, hence it is only available for Python >= 2.6.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from abc import ABCMeta, abstractmethod, abstractproperty' statement (line 11)
from abc import ABCMeta, abstractmethod, abstractproperty

import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'abc', None, module_type_store, ['ABCMeta', 'abstractmethod', 'abstractproperty'], [ABCMeta, abstractmethod, abstractproperty])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numbers import Number' statement (line 12)
from numbers import Number

import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numbers', None, module_type_store, ['Number'], [Number])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import numpy' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_179214 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_179214) is not StypyTypeError):

    if (import_179214 != 'pyd_module'):
        __import__(import_179214)
        sys_modules_179215 = sys.modules[import_179214]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', sys_modules_179215.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_179214)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.polynomial import pu' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_179216 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.polynomial')

if (type(import_179216) is not StypyTypeError):

    if (import_179216 != 'pyd_module'):
        __import__(import_179216)
        sys_modules_179217 = sys.modules[import_179216]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.polynomial', sys_modules_179217.module_type_store, module_type_store, ['polyutils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_179217, sys_modules_179217.module_type_store, module_type_store)
    else:
        from numpy.polynomial import polyutils as pu

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.polynomial', None, module_type_store, ['polyutils'], [pu])

else:
    # Assigning a type to the variable 'numpy.polynomial' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.polynomial', import_179216)

# Adding an alias
module_type_store.add_alias('pu', 'polyutils')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a List to a Name (line 17):

# Assigning a List to a Name (line 17):
__all__ = ['ABCPolyBase']
module_type_store.set_exportable_members(['ABCPolyBase'])

# Obtaining an instance of the builtin type 'list' (line 17)
list_179218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_179219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'ABCPolyBase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_179218, str_179219)

# Assigning a type to the variable '__all__' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '__all__', list_179218)
# Declaration of the 'ABCPolyBase' class

class ABCPolyBase(object, ):
    str_179220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', "An abstract base class for series classes.\n\n    ABCPolyBase provides the standard Python numerical methods\n    '+', '-', '*', '//', '%', 'divmod', '**', and '()' along with the\n    methods listed below.\n\n    .. versionadded:: 1.9.0\n\n    Parameters\n    ----------\n    coef : array_like\n        Series coefficients in order of increasing degree, i.e.,\n        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``, where\n        ``P_i`` is the basis polynomials of degree ``i``.\n    domain : (2,) array_like, optional\n        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped\n        to the interval ``[window[0], window[1]]`` by shifting and scaling.\n        The default value is the derived class domain.\n    window : (2,) array_like, optional\n        Window, see domain for its use. The default value is the\n        derived class window.\n\n    Attributes\n    ----------\n    coef : (N,) ndarray\n        Series coefficients in order of increasing degree.\n    domain : (2,) ndarray\n        Domain that is mapped to window.\n    window : (2,) ndarray\n        Window that domain is mapped to.\n\n    Class Attributes\n    ----------------\n    maxpower : int\n        Maximum power allowed, i.e., the largest number ``n`` such that\n        ``p(x)**n`` is allowed. This is to limit runaway polynomial size.\n    domain : (2,) ndarray\n        Default domain of the class.\n    window : (2,) ndarray\n        Default window of the class.\n\n    ")
    
    # Assigning a Name to a Name (line 62):
    
    # Assigning a Name to a Name (line 65):
    
    # Assigning a Num to a Name (line 68):
    
    # Assigning a Num to a Name (line 71):

    @norecursion
    def domain(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'domain'
        module_type_store = module_type_store.open_function_context('domain', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.domain.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.domain.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.domain.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.domain.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.domain')
        ABCPolyBase.domain.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.domain.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.domain.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.domain.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.domain.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.domain.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.domain.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.domain', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'domain', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'domain(...)' code ##################

        pass
        
        # ################# End of 'domain(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'domain' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_179221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179221)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'domain'
        return stypy_return_type_179221


    @norecursion
    def window(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'window'
        module_type_store = module_type_store.open_function_context('window', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.window.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.window.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.window.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.window.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.window')
        ABCPolyBase.window.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.window.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.window.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.window.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.window.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.window.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.window.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.window', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'window', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'window(...)' code ##################

        pass
        
        # ################# End of 'window(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'window' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_179222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179222)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'window'
        return stypy_return_type_179222


    @norecursion
    def nickname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'nickname'
        module_type_store = module_type_store.open_function_context('nickname', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.nickname')
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.nickname.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.nickname', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'nickname', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'nickname(...)' code ##################

        pass
        
        # ################# End of 'nickname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'nickname' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_179223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179223)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'nickname'
        return stypy_return_type_179223


    @norecursion
    def _add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add'
        module_type_store = module_type_store.open_function_context('_add', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._add.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._add.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._add.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._add')
        ABCPolyBase._add.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._add.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._add.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._add.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._add.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._add', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add(...)' code ##################

        pass
        
        # ################# End of '_add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_179224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179224)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add'
        return stypy_return_type_179224


    @norecursion
    def _sub(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sub'
        module_type_store = module_type_store.open_function_context('_sub', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._sub.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._sub.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._sub.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._sub.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._sub')
        ABCPolyBase._sub.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._sub.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._sub.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._sub.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._sub.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._sub.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._sub.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._sub', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sub', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sub(...)' code ##################

        pass
        
        # ################# End of '_sub(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sub' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_179225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sub'
        return stypy_return_type_179225


    @norecursion
    def _mul(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul'
        module_type_store = module_type_store.open_function_context('_mul', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._mul.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._mul.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._mul.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._mul.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._mul')
        ABCPolyBase._mul.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._mul.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._mul.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._mul.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._mul.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._mul.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._mul.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._mul', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul(...)' code ##################

        pass
        
        # ################# End of '_mul(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_179226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179226)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul'
        return stypy_return_type_179226


    @norecursion
    def _div(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_div'
        module_type_store = module_type_store.open_function_context('_div', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._div.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._div.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._div.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._div.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._div')
        ABCPolyBase._div.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._div.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._div.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._div.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._div.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._div.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._div.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._div', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_div', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_div(...)' code ##################

        pass
        
        # ################# End of '_div(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_div' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_179227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_div'
        return stypy_return_type_179227


    @norecursion
    def _pow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pow'
        module_type_store = module_type_store.open_function_context('_pow', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._pow.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._pow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._pow.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._pow.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._pow')
        ABCPolyBase._pow.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._pow.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._pow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._pow.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._pow.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._pow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._pow.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._pow', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pow', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pow(...)' code ##################

        pass
        
        # ################# End of '_pow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pow' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_179228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pow'
        return stypy_return_type_179228


    @norecursion
    def _val(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_val'
        module_type_store = module_type_store.open_function_context('_val', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._val.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._val.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._val.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._val.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._val')
        ABCPolyBase._val.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._val.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._val.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._val.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._val.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._val.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._val.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._val', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_val', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_val(...)' code ##################

        pass
        
        # ################# End of '_val(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_val' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_179229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179229)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_val'
        return stypy_return_type_179229


    @norecursion
    def _int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_int'
        module_type_store = module_type_store.open_function_context('_int', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._int.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._int.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._int.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._int')
        ABCPolyBase._int.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._int.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._int.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._int.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._int.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._int', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_int', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_int(...)' code ##################

        pass
        
        # ################# End of '_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_int' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_179230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_int'
        return stypy_return_type_179230


    @norecursion
    def _der(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_der'
        module_type_store = module_type_store.open_function_context('_der', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._der.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._der.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._der.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._der.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._der')
        ABCPolyBase._der.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._der.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._der.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._der.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._der.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._der.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._der.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._der', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_der', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_der(...)' code ##################

        pass
        
        # ################# End of '_der(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_der' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_179231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179231)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_der'
        return stypy_return_type_179231


    @norecursion
    def _fit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fit'
        module_type_store = module_type_store.open_function_context('_fit', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._fit.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._fit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._fit.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._fit.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._fit')
        ABCPolyBase._fit.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._fit.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._fit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._fit.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._fit.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._fit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._fit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._fit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fit(...)' code ##################

        pass
        
        # ################# End of '_fit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fit' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_179232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179232)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fit'
        return stypy_return_type_179232


    @norecursion
    def _line(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_line'
        module_type_store = module_type_store.open_function_context('_line', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._line.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._line.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._line.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._line.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._line')
        ABCPolyBase._line.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._line.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._line.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._line.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._line.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._line.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._line.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._line', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_line', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_line(...)' code ##################

        pass
        
        # ################# End of '_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_line' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_179233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179233)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_line'
        return stypy_return_type_179233


    @norecursion
    def _roots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_roots'
        module_type_store = module_type_store.open_function_context('_roots', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._roots.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._roots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._roots.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._roots.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._roots')
        ABCPolyBase._roots.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._roots.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._roots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._roots.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._roots.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._roots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._roots.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._roots', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_roots', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_roots(...)' code ##################

        pass
        
        # ################# End of '_roots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_roots' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_179234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_roots'
        return stypy_return_type_179234


    @norecursion
    def _fromroots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fromroots'
        module_type_store = module_type_store.open_function_context('_fromroots', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._fromroots')
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._fromroots.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._fromroots', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fromroots', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fromroots(...)' code ##################

        pass
        
        # ################# End of '_fromroots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fromroots' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_179235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fromroots'
        return stypy_return_type_179235


    @norecursion
    def has_samecoef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_samecoef'
        module_type_store = module_type_store.open_function_context('has_samecoef', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.has_samecoef')
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.has_samecoef.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.has_samecoef', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_samecoef', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_samecoef(...)' code ##################

        str_179236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, (-1)), 'str', 'Check if coefficients match.\n\n        .. versionadded:: 1.6.0\n\n        Parameters\n        ----------\n        other : class instance\n            The other class must have the ``coef`` attribute.\n\n        Returns\n        -------\n        bool : boolean\n            True if the coefficients are the same, False otherwise.\n\n        ')
        
        
        
        # Call to len(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'self' (line 149)
        self_179238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'self', False)
        # Obtaining the member 'coef' of a type (line 149)
        coef_179239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), self_179238, 'coef')
        # Processing the call keyword arguments (line 149)
        kwargs_179240 = {}
        # Getting the type of 'len' (line 149)
        len_179237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'len', False)
        # Calling len(args, kwargs) (line 149)
        len_call_result_179241 = invoke(stypy.reporting.localization.Localization(__file__, 149, 11), len_179237, *[coef_179239], **kwargs_179240)
        
        
        # Call to len(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'other' (line 149)
        other_179243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'other', False)
        # Obtaining the member 'coef' of a type (line 149)
        coef_179244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 33), other_179243, 'coef')
        # Processing the call keyword arguments (line 149)
        kwargs_179245 = {}
        # Getting the type of 'len' (line 149)
        len_179242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'len', False)
        # Calling len(args, kwargs) (line 149)
        len_call_result_179246 = invoke(stypy.reporting.localization.Localization(__file__, 149, 29), len_179242, *[coef_179244], **kwargs_179245)
        
        # Applying the binary operator '!=' (line 149)
        result_ne_179247 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '!=', len_call_result_179241, len_call_result_179246)
        
        # Testing the type of an if condition (line 149)
        if_condition_179248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_ne_179247)
        # Assigning a type to the variable 'if_condition_179248' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_179248', if_condition_179248)
        # SSA begins for if statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 150)
        False_179249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'stypy_return_type', False_179249)
        # SSA branch for the else part of an if statement (line 149)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to all(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Getting the type of 'self' (line 151)
        self_179252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'self', False)
        # Obtaining the member 'coef' of a type (line 151)
        coef_179253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), self_179252, 'coef')
        # Getting the type of 'other' (line 151)
        other_179254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 37), 'other', False)
        # Obtaining the member 'coef' of a type (line 151)
        coef_179255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 37), other_179254, 'coef')
        # Applying the binary operator '==' (line 151)
        result_eq_179256 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 24), '==', coef_179253, coef_179255)
        
        # Processing the call keyword arguments (line 151)
        kwargs_179257 = {}
        # Getting the type of 'np' (line 151)
        np_179250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'np', False)
        # Obtaining the member 'all' of a type (line 151)
        all_179251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 17), np_179250, 'all')
        # Calling all(args, kwargs) (line 151)
        all_call_result_179258 = invoke(stypy.reporting.localization.Localization(__file__, 151, 17), all_179251, *[result_eq_179256], **kwargs_179257)
        
        # Applying the 'not' unary operator (line 151)
        result_not__179259 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 13), 'not', all_call_result_179258)
        
        # Testing the type of an if condition (line 151)
        if_condition_179260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 13), result_not__179259)
        # Assigning a type to the variable 'if_condition_179260' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'if_condition_179260', if_condition_179260)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 152)
        False_179261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'stypy_return_type', False_179261)
        # SSA branch for the else part of an if statement (line 151)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'True' (line 154)
        True_179262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'stypy_return_type', True_179262)
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 149)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'has_samecoef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_samecoef' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_179263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_samecoef'
        return stypy_return_type_179263


    @norecursion
    def has_samedomain(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_samedomain'
        module_type_store = module_type_store.open_function_context('has_samedomain', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.has_samedomain')
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.has_samedomain.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.has_samedomain', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_samedomain', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_samedomain(...)' code ##################

        str_179264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'str', 'Check if domains match.\n\n        .. versionadded:: 1.6.0\n\n        Parameters\n        ----------\n        other : class instance\n            The other class must have the ``domain`` attribute.\n\n        Returns\n        -------\n        bool : boolean\n            True if the domains are the same, False otherwise.\n\n        ')
        
        # Call to all(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Getting the type of 'self' (line 172)
        self_179267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 'self', False)
        # Obtaining the member 'domain' of a type (line 172)
        domain_179268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 22), self_179267, 'domain')
        # Getting the type of 'other' (line 172)
        other_179269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'other', False)
        # Obtaining the member 'domain' of a type (line 172)
        domain_179270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 37), other_179269, 'domain')
        # Applying the binary operator '==' (line 172)
        result_eq_179271 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 22), '==', domain_179268, domain_179270)
        
        # Processing the call keyword arguments (line 172)
        kwargs_179272 = {}
        # Getting the type of 'np' (line 172)
        np_179265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 172)
        all_179266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 15), np_179265, 'all')
        # Calling all(args, kwargs) (line 172)
        all_call_result_179273 = invoke(stypy.reporting.localization.Localization(__file__, 172, 15), all_179266, *[result_eq_179271], **kwargs_179272)
        
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', all_call_result_179273)
        
        # ################# End of 'has_samedomain(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_samedomain' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_179274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179274)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_samedomain'
        return stypy_return_type_179274


    @norecursion
    def has_samewindow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_samewindow'
        module_type_store = module_type_store.open_function_context('has_samewindow', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.has_samewindow')
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.has_samewindow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.has_samewindow', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_samewindow', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_samewindow(...)' code ##################

        str_179275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, (-1)), 'str', 'Check if windows match.\n\n        .. versionadded:: 1.6.0\n\n        Parameters\n        ----------\n        other : class instance\n            The other class must have the ``window`` attribute.\n\n        Returns\n        -------\n        bool : boolean\n            True if the windows are the same, False otherwise.\n\n        ')
        
        # Call to all(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Getting the type of 'self' (line 190)
        self_179278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'self', False)
        # Obtaining the member 'window' of a type (line 190)
        window_179279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 22), self_179278, 'window')
        # Getting the type of 'other' (line 190)
        other_179280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 37), 'other', False)
        # Obtaining the member 'window' of a type (line 190)
        window_179281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 37), other_179280, 'window')
        # Applying the binary operator '==' (line 190)
        result_eq_179282 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 22), '==', window_179279, window_179281)
        
        # Processing the call keyword arguments (line 190)
        kwargs_179283 = {}
        # Getting the type of 'np' (line 190)
        np_179276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 190)
        all_179277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 15), np_179276, 'all')
        # Calling all(args, kwargs) (line 190)
        all_call_result_179284 = invoke(stypy.reporting.localization.Localization(__file__, 190, 15), all_179277, *[result_eq_179282], **kwargs_179283)
        
        # Assigning a type to the variable 'stypy_return_type' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'stypy_return_type', all_call_result_179284)
        
        # ################# End of 'has_samewindow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_samewindow' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_179285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_samewindow'
        return stypy_return_type_179285


    @norecursion
    def has_sametype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_sametype'
        module_type_store = module_type_store.open_function_context('has_sametype', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.has_sametype')
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.has_sametype.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.has_sametype', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_sametype', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_sametype(...)' code ##################

        str_179286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, (-1)), 'str', 'Check if types match.\n\n        .. versionadded:: 1.7.0\n\n        Parameters\n        ----------\n        other : object\n            Class instance.\n\n        Returns\n        -------\n        bool : boolean\n            True if other is same class as self\n\n        ')
        
        # Call to isinstance(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'other' (line 208)
        other_179288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'other', False)
        # Getting the type of 'self' (line 208)
        self_179289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 33), 'self', False)
        # Obtaining the member '__class__' of a type (line 208)
        class___179290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 33), self_179289, '__class__')
        # Processing the call keyword arguments (line 208)
        kwargs_179291 = {}
        # Getting the type of 'isinstance' (line 208)
        isinstance_179287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 208)
        isinstance_call_result_179292 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), isinstance_179287, *[other_179288, class___179290], **kwargs_179291)
        
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', isinstance_call_result_179292)
        
        # ################# End of 'has_sametype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_sametype' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_179293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_sametype'
        return stypy_return_type_179293


    @norecursion
    def _get_coefficients(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_coefficients'
        module_type_store = module_type_store.open_function_context('_get_coefficients', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase._get_coefficients')
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase._get_coefficients.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase._get_coefficients', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_coefficients', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_coefficients(...)' code ##################

        str_179294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, (-1)), 'str', 'Interpret other as polynomial coefficients.\n\n        The `other` argument is checked to see if it is of the same\n        class as self with identical domain and window. If so,\n        return its coefficients, otherwise return `other`.\n\n        .. versionadded:: 1.9.0\n\n        Parameters\n        ----------\n        other : anything\n            Object to be checked.\n\n        Returns\n        -------\n        coef:\n            The coefficients of`other` if it is a compatible instance,\n            of ABCPolyBase, otherwise `other`.\n\n        Raises\n        ------\n        TypeError:\n            When `other` is an incompatible instance of ABCPolyBase.\n\n        ')
        
        
        # Call to isinstance(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'other' (line 236)
        other_179296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 'other', False)
        # Getting the type of 'ABCPolyBase' (line 236)
        ABCPolyBase_179297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 29), 'ABCPolyBase', False)
        # Processing the call keyword arguments (line 236)
        kwargs_179298 = {}
        # Getting the type of 'isinstance' (line 236)
        isinstance_179295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 236)
        isinstance_call_result_179299 = invoke(stypy.reporting.localization.Localization(__file__, 236, 11), isinstance_179295, *[other_179296, ABCPolyBase_179297], **kwargs_179298)
        
        # Testing the type of an if condition (line 236)
        if_condition_179300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), isinstance_call_result_179299)
        # Assigning a type to the variable 'if_condition_179300' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_179300', if_condition_179300)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to isinstance(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'other' (line 237)
        other_179302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 30), 'other', False)
        # Getting the type of 'self' (line 237)
        self_179303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 37), 'self', False)
        # Obtaining the member '__class__' of a type (line 237)
        class___179304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 37), self_179303, '__class__')
        # Processing the call keyword arguments (line 237)
        kwargs_179305 = {}
        # Getting the type of 'isinstance' (line 237)
        isinstance_179301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 237)
        isinstance_call_result_179306 = invoke(stypy.reporting.localization.Localization(__file__, 237, 19), isinstance_179301, *[other_179302, class___179304], **kwargs_179305)
        
        # Applying the 'not' unary operator (line 237)
        result_not__179307 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 15), 'not', isinstance_call_result_179306)
        
        # Testing the type of an if condition (line 237)
        if_condition_179308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 12), result_not__179307)
        # Assigning a type to the variable 'if_condition_179308' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'if_condition_179308', if_condition_179308)
        # SSA begins for if statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 238)
        # Processing the call arguments (line 238)
        str_179310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 32), 'str', 'Polynomial types differ')
        # Processing the call keyword arguments (line 238)
        kwargs_179311 = {}
        # Getting the type of 'TypeError' (line 238)
        TypeError_179309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 238)
        TypeError_call_result_179312 = invoke(stypy.reporting.localization.Localization(__file__, 238, 22), TypeError_179309, *[str_179310], **kwargs_179311)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 238, 16), TypeError_call_result_179312, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 237)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to all(...): (line 239)
        # Processing the call arguments (line 239)
        
        # Getting the type of 'self' (line 239)
        self_179315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'self', False)
        # Obtaining the member 'domain' of a type (line 239)
        domain_179316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 28), self_179315, 'domain')
        # Getting the type of 'other' (line 239)
        other_179317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 43), 'other', False)
        # Obtaining the member 'domain' of a type (line 239)
        domain_179318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 43), other_179317, 'domain')
        # Applying the binary operator '==' (line 239)
        result_eq_179319 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 28), '==', domain_179316, domain_179318)
        
        # Processing the call keyword arguments (line 239)
        kwargs_179320 = {}
        # Getting the type of 'np' (line 239)
        np_179313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'np', False)
        # Obtaining the member 'all' of a type (line 239)
        all_179314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 21), np_179313, 'all')
        # Calling all(args, kwargs) (line 239)
        all_call_result_179321 = invoke(stypy.reporting.localization.Localization(__file__, 239, 21), all_179314, *[result_eq_179319], **kwargs_179320)
        
        # Applying the 'not' unary operator (line 239)
        result_not__179322 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 17), 'not', all_call_result_179321)
        
        # Testing the type of an if condition (line 239)
        if_condition_179323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 17), result_not__179322)
        # Assigning a type to the variable 'if_condition_179323' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 17), 'if_condition_179323', if_condition_179323)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 240)
        # Processing the call arguments (line 240)
        str_179325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 32), 'str', 'Domains differ')
        # Processing the call keyword arguments (line 240)
        kwargs_179326 = {}
        # Getting the type of 'TypeError' (line 240)
        TypeError_179324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 240)
        TypeError_call_result_179327 = invoke(stypy.reporting.localization.Localization(__file__, 240, 22), TypeError_179324, *[str_179325], **kwargs_179326)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 240, 16), TypeError_call_result_179327, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 239)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to all(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Getting the type of 'self' (line 241)
        self_179330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 28), 'self', False)
        # Obtaining the member 'window' of a type (line 241)
        window_179331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 28), self_179330, 'window')
        # Getting the type of 'other' (line 241)
        other_179332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 43), 'other', False)
        # Obtaining the member 'window' of a type (line 241)
        window_179333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 43), other_179332, 'window')
        # Applying the binary operator '==' (line 241)
        result_eq_179334 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 28), '==', window_179331, window_179333)
        
        # Processing the call keyword arguments (line 241)
        kwargs_179335 = {}
        # Getting the type of 'np' (line 241)
        np_179328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'np', False)
        # Obtaining the member 'all' of a type (line 241)
        all_179329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 21), np_179328, 'all')
        # Calling all(args, kwargs) (line 241)
        all_call_result_179336 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), all_179329, *[result_eq_179334], **kwargs_179335)
        
        # Applying the 'not' unary operator (line 241)
        result_not__179337 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 17), 'not', all_call_result_179336)
        
        # Testing the type of an if condition (line 241)
        if_condition_179338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 17), result_not__179337)
        # Assigning a type to the variable 'if_condition_179338' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'if_condition_179338', if_condition_179338)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 242)
        # Processing the call arguments (line 242)
        str_179340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 32), 'str', 'Windows differ')
        # Processing the call keyword arguments (line 242)
        kwargs_179341 = {}
        # Getting the type of 'TypeError' (line 242)
        TypeError_179339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 242)
        TypeError_call_result_179342 = invoke(stypy.reporting.localization.Localization(__file__, 242, 22), TypeError_179339, *[str_179340], **kwargs_179341)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 242, 16), TypeError_call_result_179342, 'raise parameter', BaseException)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 237)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'other' (line 243)
        other_179343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'other')
        # Obtaining the member 'coef' of a type (line 243)
        coef_179344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 19), other_179343, 'coef')
        # Assigning a type to the variable 'stypy_return_type' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'stypy_return_type', coef_179344)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'other' (line 244)
        other_179345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'other')
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'stypy_return_type', other_179345)
        
        # ################# End of '_get_coefficients(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_coefficients' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_179346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179346)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_coefficients'
        return stypy_return_type_179346


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 246)
        None_179347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 36), 'None')
        # Getting the type of 'None' (line 246)
        None_179348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 49), 'None')
        defaults = [None_179347, None_179348]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__init__', ['coef', 'domain', 'window'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['coef', 'domain', 'window'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a List (line 247):
        
        # Assigning a Call to a Name:
        
        # Call to as_series(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_179351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        # Getting the type of 'coef' (line 247)
        coef_179352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 31), 'coef', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 30), list_179351, coef_179352)
        
        # Processing the call keyword arguments (line 247)
        # Getting the type of 'False' (line 247)
        False_179353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 43), 'False', False)
        keyword_179354 = False_179353
        kwargs_179355 = {'trim': keyword_179354}
        # Getting the type of 'pu' (line 247)
        pu_179349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'pu', False)
        # Obtaining the member 'as_series' of a type (line 247)
        as_series_179350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 17), pu_179349, 'as_series')
        # Calling as_series(args, kwargs) (line 247)
        as_series_call_result_179356 = invoke(stypy.reporting.localization.Localization(__file__, 247, 17), as_series_179350, *[list_179351], **kwargs_179355)
        
        # Assigning a type to the variable 'call_assignment_179182' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'call_assignment_179182', as_series_call_result_179356)
        
        # Assigning a Call to a Name (line 247):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_179359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'int')
        # Processing the call keyword arguments
        kwargs_179360 = {}
        # Getting the type of 'call_assignment_179182' (line 247)
        call_assignment_179182_179357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'call_assignment_179182', False)
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___179358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), call_assignment_179182_179357, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_179361 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179358, *[int_179359], **kwargs_179360)
        
        # Assigning a type to the variable 'call_assignment_179183' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'call_assignment_179183', getitem___call_result_179361)
        
        # Assigning a Name to a Name (line 247):
        # Getting the type of 'call_assignment_179183' (line 247)
        call_assignment_179183_179362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'call_assignment_179183')
        # Assigning a type to the variable 'coef' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 9), 'coef', call_assignment_179183_179362)
        
        # Assigning a Name to a Attribute (line 248):
        
        # Assigning a Name to a Attribute (line 248):
        # Getting the type of 'coef' (line 248)
        coef_179363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'coef')
        # Getting the type of 'self' (line 248)
        self_179364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member 'coef' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_179364, 'coef', coef_179363)
        
        # Type idiom detected: calculating its left and rigth part (line 250)
        # Getting the type of 'domain' (line 250)
        domain_179365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'domain')
        # Getting the type of 'None' (line 250)
        None_179366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'None')
        
        (may_be_179367, more_types_in_union_179368) = may_not_be_none(domain_179365, None_179366)

        if may_be_179367:

            if more_types_in_union_179368:
                # Runtime conditional SSA (line 250)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a List (line 251):
            
            # Assigning a Call to a Name:
            
            # Call to as_series(...): (line 251)
            # Processing the call arguments (line 251)
            
            # Obtaining an instance of the builtin type 'list' (line 251)
            list_179371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 36), 'list')
            # Adding type elements to the builtin type 'list' instance (line 251)
            # Adding element type (line 251)
            # Getting the type of 'domain' (line 251)
            domain_179372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 37), 'domain', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 36), list_179371, domain_179372)
            
            # Processing the call keyword arguments (line 251)
            # Getting the type of 'False' (line 251)
            False_179373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 51), 'False', False)
            keyword_179374 = False_179373
            kwargs_179375 = {'trim': keyword_179374}
            # Getting the type of 'pu' (line 251)
            pu_179369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'pu', False)
            # Obtaining the member 'as_series' of a type (line 251)
            as_series_179370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 23), pu_179369, 'as_series')
            # Calling as_series(args, kwargs) (line 251)
            as_series_call_result_179376 = invoke(stypy.reporting.localization.Localization(__file__, 251, 23), as_series_179370, *[list_179371], **kwargs_179375)
            
            # Assigning a type to the variable 'call_assignment_179184' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'call_assignment_179184', as_series_call_result_179376)
            
            # Assigning a Call to a Name (line 251):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_179379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 12), 'int')
            # Processing the call keyword arguments
            kwargs_179380 = {}
            # Getting the type of 'call_assignment_179184' (line 251)
            call_assignment_179184_179377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'call_assignment_179184', False)
            # Obtaining the member '__getitem__' of a type (line 251)
            getitem___179378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), call_assignment_179184_179377, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_179381 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179378, *[int_179379], **kwargs_179380)
            
            # Assigning a type to the variable 'call_assignment_179185' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'call_assignment_179185', getitem___call_result_179381)
            
            # Assigning a Name to a Name (line 251):
            # Getting the type of 'call_assignment_179185' (line 251)
            call_assignment_179185_179382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'call_assignment_179185')
            # Assigning a type to the variable 'domain' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'domain', call_assignment_179185_179382)
            
            
            
            # Call to len(...): (line 252)
            # Processing the call arguments (line 252)
            # Getting the type of 'domain' (line 252)
            domain_179384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'domain', False)
            # Processing the call keyword arguments (line 252)
            kwargs_179385 = {}
            # Getting the type of 'len' (line 252)
            len_179383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'len', False)
            # Calling len(args, kwargs) (line 252)
            len_call_result_179386 = invoke(stypy.reporting.localization.Localization(__file__, 252, 15), len_179383, *[domain_179384], **kwargs_179385)
            
            int_179387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 30), 'int')
            # Applying the binary operator '!=' (line 252)
            result_ne_179388 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 15), '!=', len_call_result_179386, int_179387)
            
            # Testing the type of an if condition (line 252)
            if_condition_179389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 12), result_ne_179388)
            # Assigning a type to the variable 'if_condition_179389' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'if_condition_179389', if_condition_179389)
            # SSA begins for if statement (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 253)
            # Processing the call arguments (line 253)
            str_179391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 33), 'str', 'Domain has wrong number of elements.')
            # Processing the call keyword arguments (line 253)
            kwargs_179392 = {}
            # Getting the type of 'ValueError' (line 253)
            ValueError_179390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 253)
            ValueError_call_result_179393 = invoke(stypy.reporting.localization.Localization(__file__, 253, 22), ValueError_179390, *[str_179391], **kwargs_179392)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 253, 16), ValueError_call_result_179393, 'raise parameter', BaseException)
            # SSA join for if statement (line 252)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 254):
            
            # Assigning a Name to a Attribute (line 254):
            # Getting the type of 'domain' (line 254)
            domain_179394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'domain')
            # Getting the type of 'self' (line 254)
            self_179395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'self')
            # Setting the type of the member 'domain' of a type (line 254)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), self_179395, 'domain', domain_179394)

            if more_types_in_union_179368:
                # SSA join for if statement (line 250)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 256)
        # Getting the type of 'window' (line 256)
        window_179396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'window')
        # Getting the type of 'None' (line 256)
        None_179397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 25), 'None')
        
        (may_be_179398, more_types_in_union_179399) = may_not_be_none(window_179396, None_179397)

        if may_be_179398:

            if more_types_in_union_179399:
                # Runtime conditional SSA (line 256)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a List (line 257):
            
            # Assigning a Call to a Name:
            
            # Call to as_series(...): (line 257)
            # Processing the call arguments (line 257)
            
            # Obtaining an instance of the builtin type 'list' (line 257)
            list_179402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 36), 'list')
            # Adding type elements to the builtin type 'list' instance (line 257)
            # Adding element type (line 257)
            # Getting the type of 'window' (line 257)
            window_179403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 37), 'window', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 36), list_179402, window_179403)
            
            # Processing the call keyword arguments (line 257)
            # Getting the type of 'False' (line 257)
            False_179404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 51), 'False', False)
            keyword_179405 = False_179404
            kwargs_179406 = {'trim': keyword_179405}
            # Getting the type of 'pu' (line 257)
            pu_179400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'pu', False)
            # Obtaining the member 'as_series' of a type (line 257)
            as_series_179401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 23), pu_179400, 'as_series')
            # Calling as_series(args, kwargs) (line 257)
            as_series_call_result_179407 = invoke(stypy.reporting.localization.Localization(__file__, 257, 23), as_series_179401, *[list_179402], **kwargs_179406)
            
            # Assigning a type to the variable 'call_assignment_179186' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'call_assignment_179186', as_series_call_result_179407)
            
            # Assigning a Call to a Name (line 257):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_179410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'int')
            # Processing the call keyword arguments
            kwargs_179411 = {}
            # Getting the type of 'call_assignment_179186' (line 257)
            call_assignment_179186_179408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'call_assignment_179186', False)
            # Obtaining the member '__getitem__' of a type (line 257)
            getitem___179409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), call_assignment_179186_179408, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_179412 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179409, *[int_179410], **kwargs_179411)
            
            # Assigning a type to the variable 'call_assignment_179187' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'call_assignment_179187', getitem___call_result_179412)
            
            # Assigning a Name to a Name (line 257):
            # Getting the type of 'call_assignment_179187' (line 257)
            call_assignment_179187_179413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'call_assignment_179187')
            # Assigning a type to the variable 'window' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 13), 'window', call_assignment_179187_179413)
            
            
            
            # Call to len(...): (line 258)
            # Processing the call arguments (line 258)
            # Getting the type of 'window' (line 258)
            window_179415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'window', False)
            # Processing the call keyword arguments (line 258)
            kwargs_179416 = {}
            # Getting the type of 'len' (line 258)
            len_179414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'len', False)
            # Calling len(args, kwargs) (line 258)
            len_call_result_179417 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), len_179414, *[window_179415], **kwargs_179416)
            
            int_179418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'int')
            # Applying the binary operator '!=' (line 258)
            result_ne_179419 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 15), '!=', len_call_result_179417, int_179418)
            
            # Testing the type of an if condition (line 258)
            if_condition_179420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 12), result_ne_179419)
            # Assigning a type to the variable 'if_condition_179420' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'if_condition_179420', if_condition_179420)
            # SSA begins for if statement (line 258)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 259)
            # Processing the call arguments (line 259)
            str_179422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 33), 'str', 'Window has wrong number of elements.')
            # Processing the call keyword arguments (line 259)
            kwargs_179423 = {}
            # Getting the type of 'ValueError' (line 259)
            ValueError_179421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 259)
            ValueError_call_result_179424 = invoke(stypy.reporting.localization.Localization(__file__, 259, 22), ValueError_179421, *[str_179422], **kwargs_179423)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 259, 16), ValueError_call_result_179424, 'raise parameter', BaseException)
            # SSA join for if statement (line 258)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 260):
            
            # Assigning a Name to a Attribute (line 260):
            # Getting the type of 'window' (line 260)
            window_179425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 26), 'window')
            # Getting the type of 'self' (line 260)
            self_179426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self')
            # Setting the type of the member 'window' of a type (line 260)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_179426, 'window', window_179425)

            if more_types_in_union_179399:
                # SSA join for if statement (line 256)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 262, 4, False)
        # Assigning a type to the variable 'self' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__repr__')
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Assigning a Str to a Name (line 263):
        
        # Assigning a Str to a Name (line 263):
        str_179427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 17), 'str', '%s(%s, %s, %s)')
        # Assigning a type to the variable 'format' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'format', str_179427)
        
        # Assigning a Subscript to a Name (line 264):
        
        # Assigning a Subscript to a Name (line 264):
        
        # Obtaining the type of the subscript
        int_179428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 31), 'int')
        int_179429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 33), 'int')
        slice_179430 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 15), int_179428, int_179429, None)
        
        # Call to repr(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'self' (line 264)
        self_179432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'self', False)
        # Obtaining the member 'coef' of a type (line 264)
        coef_179433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 20), self_179432, 'coef')
        # Processing the call keyword arguments (line 264)
        kwargs_179434 = {}
        # Getting the type of 'repr' (line 264)
        repr_179431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'repr', False)
        # Calling repr(args, kwargs) (line 264)
        repr_call_result_179435 = invoke(stypy.reporting.localization.Localization(__file__, 264, 15), repr_179431, *[coef_179433], **kwargs_179434)
        
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___179436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 15), repr_call_result_179435, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_179437 = invoke(stypy.reporting.localization.Localization(__file__, 264, 15), getitem___179436, slice_179430)
        
        # Assigning a type to the variable 'coef' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'coef', subscript_call_result_179437)
        
        # Assigning a Subscript to a Name (line 265):
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_179438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 35), 'int')
        int_179439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 37), 'int')
        slice_179440 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 17), int_179438, int_179439, None)
        
        # Call to repr(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'self' (line 265)
        self_179442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'self', False)
        # Obtaining the member 'domain' of a type (line 265)
        domain_179443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 22), self_179442, 'domain')
        # Processing the call keyword arguments (line 265)
        kwargs_179444 = {}
        # Getting the type of 'repr' (line 265)
        repr_179441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'repr', False)
        # Calling repr(args, kwargs) (line 265)
        repr_call_result_179445 = invoke(stypy.reporting.localization.Localization(__file__, 265, 17), repr_179441, *[domain_179443], **kwargs_179444)
        
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___179446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 17), repr_call_result_179445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_179447 = invoke(stypy.reporting.localization.Localization(__file__, 265, 17), getitem___179446, slice_179440)
        
        # Assigning a type to the variable 'domain' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'domain', subscript_call_result_179447)
        
        # Assigning a Subscript to a Name (line 266):
        
        # Assigning a Subscript to a Name (line 266):
        
        # Obtaining the type of the subscript
        int_179448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 35), 'int')
        int_179449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 37), 'int')
        slice_179450 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 266, 17), int_179448, int_179449, None)
        
        # Call to repr(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'self' (line 266)
        self_179452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'self', False)
        # Obtaining the member 'window' of a type (line 266)
        window_179453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 22), self_179452, 'window')
        # Processing the call keyword arguments (line 266)
        kwargs_179454 = {}
        # Getting the type of 'repr' (line 266)
        repr_179451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'repr', False)
        # Calling repr(args, kwargs) (line 266)
        repr_call_result_179455 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), repr_179451, *[window_179453], **kwargs_179454)
        
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___179456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 17), repr_call_result_179455, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_179457 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), getitem___179456, slice_179450)
        
        # Assigning a type to the variable 'window' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'window', subscript_call_result_179457)
        
        # Assigning a Attribute to a Name (line 267):
        
        # Assigning a Attribute to a Name (line 267):
        # Getting the type of 'self' (line 267)
        self_179458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'self')
        # Obtaining the member '__class__' of a type (line 267)
        class___179459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 15), self_179458, '__class__')
        # Obtaining the member '__name__' of a type (line 267)
        name___179460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 15), class___179459, '__name__')
        # Assigning a type to the variable 'name' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'name', name___179460)
        # Getting the type of 'format' (line 268)
        format_179461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'format')
        
        # Obtaining an instance of the builtin type 'tuple' (line 268)
        tuple_179462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 268)
        # Adding element type (line 268)
        # Getting the type of 'name' (line 268)
        name_179463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 25), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 25), tuple_179462, name_179463)
        # Adding element type (line 268)
        # Getting the type of 'coef' (line 268)
        coef_179464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 31), 'coef')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 25), tuple_179462, coef_179464)
        # Adding element type (line 268)
        # Getting the type of 'domain' (line 268)
        domain_179465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 37), 'domain')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 25), tuple_179462, domain_179465)
        # Adding element type (line 268)
        # Getting the type of 'window' (line 268)
        window_179466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 45), 'window')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 25), tuple_179462, window_179466)
        
        # Applying the binary operator '%' (line 268)
        result_mod_179467 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 15), '%', format_179461, tuple_179462)
        
        # Assigning a type to the variable 'stypy_return_type' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'stypy_return_type', result_mod_179467)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 262)
        stypy_return_type_179468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179468)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_179468


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__str__')
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 271):
        
        # Assigning a Str to a Name (line 271):
        str_179469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 17), 'str', '%s(%s)')
        # Assigning a type to the variable 'format' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'format', str_179469)
        
        # Assigning a Call to a Name (line 272):
        
        # Assigning a Call to a Name (line 272):
        
        # Call to str(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'self' (line 272)
        self_179471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'self', False)
        # Obtaining the member 'coef' of a type (line 272)
        coef_179472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 19), self_179471, 'coef')
        # Processing the call keyword arguments (line 272)
        kwargs_179473 = {}
        # Getting the type of 'str' (line 272)
        str_179470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'str', False)
        # Calling str(args, kwargs) (line 272)
        str_call_result_179474 = invoke(stypy.reporting.localization.Localization(__file__, 272, 15), str_179470, *[coef_179472], **kwargs_179473)
        
        # Assigning a type to the variable 'coef' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'coef', str_call_result_179474)
        
        # Assigning a Attribute to a Name (line 273):
        
        # Assigning a Attribute to a Name (line 273):
        # Getting the type of 'self' (line 273)
        self_179475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'self')
        # Obtaining the member 'nickname' of a type (line 273)
        nickname_179476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 15), self_179475, 'nickname')
        # Assigning a type to the variable 'name' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'name', nickname_179476)
        # Getting the type of 'format' (line 274)
        format_179477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'format')
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_179478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'name' (line 274)
        name_179479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 25), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 25), tuple_179478, name_179479)
        # Adding element type (line 274)
        # Getting the type of 'coef' (line 274)
        coef_179480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 31), 'coef')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 25), tuple_179478, coef_179480)
        
        # Applying the binary operator '%' (line 274)
        result_mod_179481 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 15), '%', format_179477, tuple_179478)
        
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type', result_mod_179481)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_179482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179482)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_179482


    @norecursion
    def __getstate__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getstate__'
        module_type_store = module_type_store.open_function_context('__getstate__', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__getstate__')
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__getstate__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__getstate__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to copy(...): (line 279)
        # Processing the call keyword arguments (line 279)
        kwargs_179486 = {}
        # Getting the type of 'self' (line 279)
        self_179483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 14), 'self', False)
        # Obtaining the member '__dict__' of a type (line 279)
        dict___179484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 14), self_179483, '__dict__')
        # Obtaining the member 'copy' of a type (line 279)
        copy_179485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 14), dict___179484, 'copy')
        # Calling copy(args, kwargs) (line 279)
        copy_call_result_179487 = invoke(stypy.reporting.localization.Localization(__file__, 279, 14), copy_179485, *[], **kwargs_179486)
        
        # Assigning a type to the variable 'ret' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'ret', copy_call_result_179487)
        
        # Assigning a Call to a Subscript (line 280):
        
        # Assigning a Call to a Subscript (line 280):
        
        # Call to copy(...): (line 280)
        # Processing the call keyword arguments (line 280)
        kwargs_179491 = {}
        # Getting the type of 'self' (line 280)
        self_179488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 22), 'self', False)
        # Obtaining the member 'coef' of a type (line 280)
        coef_179489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 22), self_179488, 'coef')
        # Obtaining the member 'copy' of a type (line 280)
        copy_179490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 22), coef_179489, 'copy')
        # Calling copy(args, kwargs) (line 280)
        copy_call_result_179492 = invoke(stypy.reporting.localization.Localization(__file__, 280, 22), copy_179490, *[], **kwargs_179491)
        
        # Getting the type of 'ret' (line 280)
        ret_179493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'ret')
        str_179494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'str', 'coef')
        # Storing an element on a container (line 280)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 8), ret_179493, (str_179494, copy_call_result_179492))
        
        # Assigning a Call to a Subscript (line 281):
        
        # Assigning a Call to a Subscript (line 281):
        
        # Call to copy(...): (line 281)
        # Processing the call keyword arguments (line 281)
        kwargs_179498 = {}
        # Getting the type of 'self' (line 281)
        self_179495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'self', False)
        # Obtaining the member 'domain' of a type (line 281)
        domain_179496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), self_179495, 'domain')
        # Obtaining the member 'copy' of a type (line 281)
        copy_179497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), domain_179496, 'copy')
        # Calling copy(args, kwargs) (line 281)
        copy_call_result_179499 = invoke(stypy.reporting.localization.Localization(__file__, 281, 24), copy_179497, *[], **kwargs_179498)
        
        # Getting the type of 'ret' (line 281)
        ret_179500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'ret')
        str_179501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 12), 'str', 'domain')
        # Storing an element on a container (line 281)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 8), ret_179500, (str_179501, copy_call_result_179499))
        
        # Assigning a Call to a Subscript (line 282):
        
        # Assigning a Call to a Subscript (line 282):
        
        # Call to copy(...): (line 282)
        # Processing the call keyword arguments (line 282)
        kwargs_179505 = {}
        # Getting the type of 'self' (line 282)
        self_179502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'self', False)
        # Obtaining the member 'window' of a type (line 282)
        window_179503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), self_179502, 'window')
        # Obtaining the member 'copy' of a type (line 282)
        copy_179504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), window_179503, 'copy')
        # Calling copy(args, kwargs) (line 282)
        copy_call_result_179506 = invoke(stypy.reporting.localization.Localization(__file__, 282, 24), copy_179504, *[], **kwargs_179505)
        
        # Getting the type of 'ret' (line 282)
        ret_179507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'ret')
        str_179508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 12), 'str', 'window')
        # Storing an element on a container (line 282)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 8), ret_179507, (str_179508, copy_call_result_179506))
        # Getting the type of 'ret' (line 283)
        ret_179509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'stypy_return_type', ret_179509)
        
        # ################# End of '__getstate__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getstate__' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_179510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179510)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getstate__'
        return stypy_return_type_179510


    @norecursion
    def __setstate__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setstate__'
        module_type_store = module_type_store.open_function_context('__setstate__', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__setstate__')
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_param_names_list', ['dict'])
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__setstate__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__setstate__', ['dict'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setstate__', localization, ['dict'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setstate__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 286):
        
        # Assigning a Name to a Attribute (line 286):
        # Getting the type of 'dict' (line 286)
        dict_179511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 24), 'dict')
        # Getting the type of 'self' (line 286)
        self_179512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self')
        # Setting the type of the member '__dict__' of a type (line 286)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_179512, '__dict__', dict_179511)
        
        # ################# End of '__setstate__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setstate__' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_179513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setstate__'
        return stypy_return_type_179513


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__call__')
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_param_names_list', ['arg'])
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__call__', ['arg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['arg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Call to a Tuple (line 291):
        
        # Assigning a Call to a Name:
        
        # Call to mapparms(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'self' (line 291)
        self_179516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 31), 'self', False)
        # Obtaining the member 'domain' of a type (line 291)
        domain_179517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 31), self_179516, 'domain')
        # Getting the type of 'self' (line 291)
        self_179518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 44), 'self', False)
        # Obtaining the member 'window' of a type (line 291)
        window_179519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 44), self_179518, 'window')
        # Processing the call keyword arguments (line 291)
        kwargs_179520 = {}
        # Getting the type of 'pu' (line 291)
        pu_179514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 19), 'pu', False)
        # Obtaining the member 'mapparms' of a type (line 291)
        mapparms_179515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 19), pu_179514, 'mapparms')
        # Calling mapparms(args, kwargs) (line 291)
        mapparms_call_result_179521 = invoke(stypy.reporting.localization.Localization(__file__, 291, 19), mapparms_179515, *[domain_179517, window_179519], **kwargs_179520)
        
        # Assigning a type to the variable 'call_assignment_179188' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_179188', mapparms_call_result_179521)
        
        # Assigning a Call to a Name (line 291):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_179524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 8), 'int')
        # Processing the call keyword arguments
        kwargs_179525 = {}
        # Getting the type of 'call_assignment_179188' (line 291)
        call_assignment_179188_179522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_179188', False)
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___179523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), call_assignment_179188_179522, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_179526 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179523, *[int_179524], **kwargs_179525)
        
        # Assigning a type to the variable 'call_assignment_179189' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_179189', getitem___call_result_179526)
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'call_assignment_179189' (line 291)
        call_assignment_179189_179527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_179189')
        # Assigning a type to the variable 'off' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'off', call_assignment_179189_179527)
        
        # Assigning a Call to a Name (line 291):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_179530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 8), 'int')
        # Processing the call keyword arguments
        kwargs_179531 = {}
        # Getting the type of 'call_assignment_179188' (line 291)
        call_assignment_179188_179528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_179188', False)
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___179529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), call_assignment_179188_179528, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_179532 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179529, *[int_179530], **kwargs_179531)
        
        # Assigning a type to the variable 'call_assignment_179190' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_179190', getitem___call_result_179532)
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'call_assignment_179190' (line 291)
        call_assignment_179190_179533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_179190')
        # Assigning a type to the variable 'scl' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 13), 'scl', call_assignment_179190_179533)
        
        # Assigning a BinOp to a Name (line 292):
        
        # Assigning a BinOp to a Name (line 292):
        # Getting the type of 'off' (line 292)
        off_179534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'off')
        # Getting the type of 'scl' (line 292)
        scl_179535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 20), 'scl')
        # Getting the type of 'arg' (line 292)
        arg_179536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'arg')
        # Applying the binary operator '*' (line 292)
        result_mul_179537 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 20), '*', scl_179535, arg_179536)
        
        # Applying the binary operator '+' (line 292)
        result_add_179538 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 14), '+', off_179534, result_mul_179537)
        
        # Assigning a type to the variable 'arg' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'arg', result_add_179538)
        
        # Call to _val(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'arg' (line 293)
        arg_179541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 25), 'arg', False)
        # Getting the type of 'self' (line 293)
        self_179542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'self', False)
        # Obtaining the member 'coef' of a type (line 293)
        coef_179543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 30), self_179542, 'coef')
        # Processing the call keyword arguments (line 293)
        kwargs_179544 = {}
        # Getting the type of 'self' (line 293)
        self_179539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'self', False)
        # Obtaining the member '_val' of a type (line 293)
        _val_179540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), self_179539, '_val')
        # Calling _val(args, kwargs) (line 293)
        _val_call_result_179545 = invoke(stypy.reporting.localization.Localization(__file__, 293, 15), _val_179540, *[arg_179541, coef_179543], **kwargs_179544)
        
        # Assigning a type to the variable 'stypy_return_type' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'stypy_return_type', _val_call_result_179545)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_179546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179546)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_179546


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 295, 4, False)
        # Assigning a type to the variable 'self' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__iter__')
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        
        # Call to iter(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'self' (line 296)
        self_179548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'self', False)
        # Obtaining the member 'coef' of a type (line 296)
        coef_179549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 20), self_179548, 'coef')
        # Processing the call keyword arguments (line 296)
        kwargs_179550 = {}
        # Getting the type of 'iter' (line 296)
        iter_179547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'iter', False)
        # Calling iter(args, kwargs) (line 296)
        iter_call_result_179551 = invoke(stypy.reporting.localization.Localization(__file__, 296, 15), iter_179547, *[coef_179549], **kwargs_179550)
        
        # Assigning a type to the variable 'stypy_return_type' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'stypy_return_type', iter_call_result_179551)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 295)
        stypy_return_type_179552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_179552


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 298, 4, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__len__')
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        
        # Call to len(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'self' (line 299)
        self_179554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 19), 'self', False)
        # Obtaining the member 'coef' of a type (line 299)
        coef_179555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 19), self_179554, 'coef')
        # Processing the call keyword arguments (line 299)
        kwargs_179556 = {}
        # Getting the type of 'len' (line 299)
        len_179553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 15), 'len', False)
        # Calling len(args, kwargs) (line 299)
        len_call_result_179557 = invoke(stypy.reporting.localization.Localization(__file__, 299, 15), len_179553, *[coef_179555], **kwargs_179556)
        
        # Assigning a type to the variable 'stypy_return_type' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'stypy_return_type', len_call_result_179557)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 298)
        stypy_return_type_179558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179558)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_179558


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__neg__')
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__neg__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__neg__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__neg__(...)' code ##################

        
        # Call to __class__(...): (line 304)
        # Processing the call arguments (line 304)
        
        # Getting the type of 'self' (line 304)
        self_179561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 31), 'self', False)
        # Obtaining the member 'coef' of a type (line 304)
        coef_179562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 31), self_179561, 'coef')
        # Applying the 'usub' unary operator (line 304)
        result___neg___179563 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 30), 'usub', coef_179562)
        
        # Getting the type of 'self' (line 304)
        self_179564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 42), 'self', False)
        # Obtaining the member 'domain' of a type (line 304)
        domain_179565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 42), self_179564, 'domain')
        # Getting the type of 'self' (line 304)
        self_179566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 55), 'self', False)
        # Obtaining the member 'window' of a type (line 304)
        window_179567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 55), self_179566, 'window')
        # Processing the call keyword arguments (line 304)
        kwargs_179568 = {}
        # Getting the type of 'self' (line 304)
        self_179559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 304)
        class___179560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 15), self_179559, '__class__')
        # Calling __class__(args, kwargs) (line 304)
        class___call_result_179569 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), class___179560, *[result___neg___179563, domain_179565, window_179567], **kwargs_179568)
        
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', class___call_result_179569)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_179570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179570)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_179570


    @norecursion
    def __pos__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pos__'
        module_type_store = module_type_store.open_function_context('__pos__', 306, 4, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__pos__')
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__pos__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__pos__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pos__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pos__(...)' code ##################

        # Getting the type of 'self' (line 307)
        self_179571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'stypy_return_type', self_179571)
        
        # ################# End of '__pos__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pos__' in the type store
        # Getting the type of 'stypy_return_type' (line 306)
        stypy_return_type_179572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179572)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pos__'
        return stypy_return_type_179572


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__add__')
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 311):
        
        # Assigning a Call to a Name (line 311):
        
        # Call to _get_coefficients(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'other' (line 311)
        other_179575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 47), 'other', False)
        # Processing the call keyword arguments (line 311)
        kwargs_179576 = {}
        # Getting the type of 'self' (line 311)
        self_179573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'self', False)
        # Obtaining the member '_get_coefficients' of a type (line 311)
        _get_coefficients_179574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 24), self_179573, '_get_coefficients')
        # Calling _get_coefficients(args, kwargs) (line 311)
        _get_coefficients_call_result_179577 = invoke(stypy.reporting.localization.Localization(__file__, 311, 24), _get_coefficients_179574, *[other_179575], **kwargs_179576)
        
        # Assigning a type to the variable 'othercoef' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'othercoef', _get_coefficients_call_result_179577)
        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to _add(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'self' (line 312)
        self_179580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 29), 'self', False)
        # Obtaining the member 'coef' of a type (line 312)
        coef_179581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 29), self_179580, 'coef')
        # Getting the type of 'othercoef' (line 312)
        othercoef_179582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 40), 'othercoef', False)
        # Processing the call keyword arguments (line 312)
        kwargs_179583 = {}
        # Getting the type of 'self' (line 312)
        self_179578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'self', False)
        # Obtaining the member '_add' of a type (line 312)
        _add_179579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 19), self_179578, '_add')
        # Calling _add(args, kwargs) (line 312)
        _add_call_result_179584 = invoke(stypy.reporting.localization.Localization(__file__, 312, 19), _add_179579, *[coef_179581, othercoef_179582], **kwargs_179583)
        
        # Assigning a type to the variable 'coef' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'coef', _add_call_result_179584)
        # SSA branch for the except part of a try statement (line 310)
        # SSA branch for the except 'TypeError' branch of a try statement (line 310)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'TypeError' (line 313)
        TypeError_179585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'TypeError')
        # Assigning a type to the variable 'e' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'e', TypeError_179585)
        # Getting the type of 'e' (line 314)
        e_179586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 18), 'e')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 314, 12), e_179586, 'raise parameter', BaseException)
        # SSA branch for the except '<any exception>' branch of a try statement (line 310)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 316)
        NotImplemented_179587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'stypy_return_type', NotImplemented_179587)
        # SSA join for try-except statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'coef' (line 317)
        coef_179590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 30), 'coef', False)
        # Getting the type of 'self' (line 317)
        self_179591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 317)
        domain_179592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 36), self_179591, 'domain')
        # Getting the type of 'self' (line 317)
        self_179593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 317)
        window_179594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 49), self_179593, 'window')
        # Processing the call keyword arguments (line 317)
        kwargs_179595 = {}
        # Getting the type of 'self' (line 317)
        self_179588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 317)
        class___179589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 15), self_179588, '__class__')
        # Calling __class__(args, kwargs) (line 317)
        class___call_result_179596 = invoke(stypy.reporting.localization.Localization(__file__, 317, 15), class___179589, *[coef_179590, domain_179592, window_179594], **kwargs_179595)
        
        # Assigning a type to the variable 'stypy_return_type' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'stypy_return_type', class___call_result_179596)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_179597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179597)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_179597


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__sub__')
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__sub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 320)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to _get_coefficients(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'other' (line 321)
        other_179600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 47), 'other', False)
        # Processing the call keyword arguments (line 321)
        kwargs_179601 = {}
        # Getting the type of 'self' (line 321)
        self_179598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'self', False)
        # Obtaining the member '_get_coefficients' of a type (line 321)
        _get_coefficients_179599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 24), self_179598, '_get_coefficients')
        # Calling _get_coefficients(args, kwargs) (line 321)
        _get_coefficients_call_result_179602 = invoke(stypy.reporting.localization.Localization(__file__, 321, 24), _get_coefficients_179599, *[other_179600], **kwargs_179601)
        
        # Assigning a type to the variable 'othercoef' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'othercoef', _get_coefficients_call_result_179602)
        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to _sub(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'self' (line 322)
        self_179605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 29), 'self', False)
        # Obtaining the member 'coef' of a type (line 322)
        coef_179606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 29), self_179605, 'coef')
        # Getting the type of 'othercoef' (line 322)
        othercoef_179607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 40), 'othercoef', False)
        # Processing the call keyword arguments (line 322)
        kwargs_179608 = {}
        # Getting the type of 'self' (line 322)
        self_179603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 19), 'self', False)
        # Obtaining the member '_sub' of a type (line 322)
        _sub_179604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 19), self_179603, '_sub')
        # Calling _sub(args, kwargs) (line 322)
        _sub_call_result_179609 = invoke(stypy.reporting.localization.Localization(__file__, 322, 19), _sub_179604, *[coef_179606, othercoef_179607], **kwargs_179608)
        
        # Assigning a type to the variable 'coef' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'coef', _sub_call_result_179609)
        # SSA branch for the except part of a try statement (line 320)
        # SSA branch for the except 'TypeError' branch of a try statement (line 320)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'TypeError' (line 323)
        TypeError_179610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'TypeError')
        # Assigning a type to the variable 'e' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'e', TypeError_179610)
        # Getting the type of 'e' (line 324)
        e_179611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 18), 'e')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 324, 12), e_179611, 'raise parameter', BaseException)
        # SSA branch for the except '<any exception>' branch of a try statement (line 320)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 326)
        NotImplemented_179612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'stypy_return_type', NotImplemented_179612)
        # SSA join for try-except statement (line 320)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'coef' (line 327)
        coef_179615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 30), 'coef', False)
        # Getting the type of 'self' (line 327)
        self_179616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 327)
        domain_179617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 36), self_179616, 'domain')
        # Getting the type of 'self' (line 327)
        self_179618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 327)
        window_179619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 49), self_179618, 'window')
        # Processing the call keyword arguments (line 327)
        kwargs_179620 = {}
        # Getting the type of 'self' (line 327)
        self_179613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 327)
        class___179614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 15), self_179613, '__class__')
        # Calling __class__(args, kwargs) (line 327)
        class___call_result_179621 = invoke(stypy.reporting.localization.Localization(__file__, 327, 15), class___179614, *[coef_179615, domain_179617, window_179619], **kwargs_179620)
        
        # Assigning a type to the variable 'stypy_return_type' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stypy_return_type', class___call_result_179621)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_179622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_179622


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__mul__')
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__mul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 331):
        
        # Assigning a Call to a Name (line 331):
        
        # Call to _get_coefficients(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'other' (line 331)
        other_179625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 47), 'other', False)
        # Processing the call keyword arguments (line 331)
        kwargs_179626 = {}
        # Getting the type of 'self' (line 331)
        self_179623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'self', False)
        # Obtaining the member '_get_coefficients' of a type (line 331)
        _get_coefficients_179624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 24), self_179623, '_get_coefficients')
        # Calling _get_coefficients(args, kwargs) (line 331)
        _get_coefficients_call_result_179627 = invoke(stypy.reporting.localization.Localization(__file__, 331, 24), _get_coefficients_179624, *[other_179625], **kwargs_179626)
        
        # Assigning a type to the variable 'othercoef' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'othercoef', _get_coefficients_call_result_179627)
        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to _mul(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'self' (line 332)
        self_179630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 29), 'self', False)
        # Obtaining the member 'coef' of a type (line 332)
        coef_179631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 29), self_179630, 'coef')
        # Getting the type of 'othercoef' (line 332)
        othercoef_179632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 40), 'othercoef', False)
        # Processing the call keyword arguments (line 332)
        kwargs_179633 = {}
        # Getting the type of 'self' (line 332)
        self_179628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'self', False)
        # Obtaining the member '_mul' of a type (line 332)
        _mul_179629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), self_179628, '_mul')
        # Calling _mul(args, kwargs) (line 332)
        _mul_call_result_179634 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), _mul_179629, *[coef_179631, othercoef_179632], **kwargs_179633)
        
        # Assigning a type to the variable 'coef' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'coef', _mul_call_result_179634)
        # SSA branch for the except part of a try statement (line 330)
        # SSA branch for the except 'TypeError' branch of a try statement (line 330)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'TypeError' (line 333)
        TypeError_179635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'TypeError')
        # Assigning a type to the variable 'e' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'e', TypeError_179635)
        # Getting the type of 'e' (line 334)
        e_179636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 18), 'e')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 334, 12), e_179636, 'raise parameter', BaseException)
        # SSA branch for the except '<any exception>' branch of a try statement (line 330)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 336)
        NotImplemented_179637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'stypy_return_type', NotImplemented_179637)
        # SSA join for try-except statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'coef' (line 337)
        coef_179640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), 'coef', False)
        # Getting the type of 'self' (line 337)
        self_179641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 337)
        domain_179642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 36), self_179641, 'domain')
        # Getting the type of 'self' (line 337)
        self_179643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 337)
        window_179644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 49), self_179643, 'window')
        # Processing the call keyword arguments (line 337)
        kwargs_179645 = {}
        # Getting the type of 'self' (line 337)
        self_179638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 337)
        class___179639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 15), self_179638, '__class__')
        # Calling __class__(args, kwargs) (line 337)
        class___call_result_179646 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), class___179639, *[coef_179640, domain_179642, window_179644], **kwargs_179645)
        
        # Assigning a type to the variable 'stypy_return_type' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'stypy_return_type', class___call_result_179646)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_179647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179647)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_179647


    @norecursion
    def __div__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__div__'
        module_type_store = module_type_store.open_function_context('__div__', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__div__')
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__div__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__div__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__div__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__div__(...)' code ##################

        
        # Call to __floordiv__(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'other' (line 341)
        other_179650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 33), 'other', False)
        # Processing the call keyword arguments (line 341)
        kwargs_179651 = {}
        # Getting the type of 'self' (line 341)
        self_179648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'self', False)
        # Obtaining the member '__floordiv__' of a type (line 341)
        floordiv___179649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), self_179648, '__floordiv__')
        # Calling __floordiv__(args, kwargs) (line 341)
        floordiv___call_result_179652 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), floordiv___179649, *[other_179650], **kwargs_179651)
        
        # Assigning a type to the variable 'stypy_return_type' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'stypy_return_type', floordiv___call_result_179652)
        
        # ################# End of '__div__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__div__' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_179653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179653)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__div__'
        return stypy_return_type_179653


    @norecursion
    def __truediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__truediv__'
        module_type_store = module_type_store.open_function_context('__truediv__', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__truediv__')
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__truediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__truediv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__truediv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__truediv__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'other' (line 347)
        other_179655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 26), 'other', False)
        # Getting the type of 'Number' (line 347)
        Number_179656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 33), 'Number', False)
        # Processing the call keyword arguments (line 347)
        kwargs_179657 = {}
        # Getting the type of 'isinstance' (line 347)
        isinstance_179654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 347)
        isinstance_call_result_179658 = invoke(stypy.reporting.localization.Localization(__file__, 347, 15), isinstance_179654, *[other_179655, Number_179656], **kwargs_179657)
        
        # Applying the 'not' unary operator (line 347)
        result_not__179659 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 11), 'not', isinstance_call_result_179658)
        
        
        # Call to isinstance(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'other' (line 347)
        other_179661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 55), 'other', False)
        # Getting the type of 'bool' (line 347)
        bool_179662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 62), 'bool', False)
        # Processing the call keyword arguments (line 347)
        kwargs_179663 = {}
        # Getting the type of 'isinstance' (line 347)
        isinstance_179660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 44), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 347)
        isinstance_call_result_179664 = invoke(stypy.reporting.localization.Localization(__file__, 347, 44), isinstance_179660, *[other_179661, bool_179662], **kwargs_179663)
        
        # Applying the binary operator 'or' (line 347)
        result_or_keyword_179665 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 11), 'or', result_not__179659, isinstance_call_result_179664)
        
        # Testing the type of an if condition (line 347)
        if_condition_179666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 8), result_or_keyword_179665)
        # Assigning a type to the variable 'if_condition_179666' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'if_condition_179666', if_condition_179666)
        # SSA begins for if statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 348):
        
        # Assigning a Str to a Name (line 348):
        str_179667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 19), 'str', "unsupported types for true division: '%s', '%s'")
        # Assigning a type to the variable 'form' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'form', str_179667)
        
        # Call to TypeError(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'form' (line 349)
        form_179669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 28), 'form', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 349)
        tuple_179670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 349)
        # Adding element type (line 349)
        
        # Call to type(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'self' (line 349)
        self_179672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 41), 'self', False)
        # Processing the call keyword arguments (line 349)
        kwargs_179673 = {}
        # Getting the type of 'type' (line 349)
        type_179671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 36), 'type', False)
        # Calling type(args, kwargs) (line 349)
        type_call_result_179674 = invoke(stypy.reporting.localization.Localization(__file__, 349, 36), type_179671, *[self_179672], **kwargs_179673)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 36), tuple_179670, type_call_result_179674)
        # Adding element type (line 349)
        
        # Call to type(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'other' (line 349)
        other_179676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 53), 'other', False)
        # Processing the call keyword arguments (line 349)
        kwargs_179677 = {}
        # Getting the type of 'type' (line 349)
        type_179675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 48), 'type', False)
        # Calling type(args, kwargs) (line 349)
        type_call_result_179678 = invoke(stypy.reporting.localization.Localization(__file__, 349, 48), type_179675, *[other_179676], **kwargs_179677)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 36), tuple_179670, type_call_result_179678)
        
        # Applying the binary operator '%' (line 349)
        result_mod_179679 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 28), '%', form_179669, tuple_179670)
        
        # Processing the call keyword arguments (line 349)
        kwargs_179680 = {}
        # Getting the type of 'TypeError' (line 349)
        TypeError_179668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 349)
        TypeError_call_result_179681 = invoke(stypy.reporting.localization.Localization(__file__, 349, 18), TypeError_179668, *[result_mod_179679], **kwargs_179680)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 349, 12), TypeError_call_result_179681, 'raise parameter', BaseException)
        # SSA join for if statement (line 347)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __floordiv__(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'other' (line 350)
        other_179684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 33), 'other', False)
        # Processing the call keyword arguments (line 350)
        kwargs_179685 = {}
        # Getting the type of 'self' (line 350)
        self_179682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 15), 'self', False)
        # Obtaining the member '__floordiv__' of a type (line 350)
        floordiv___179683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 15), self_179682, '__floordiv__')
        # Calling __floordiv__(args, kwargs) (line 350)
        floordiv___call_result_179686 = invoke(stypy.reporting.localization.Localization(__file__, 350, 15), floordiv___179683, *[other_179684], **kwargs_179685)
        
        # Assigning a type to the variable 'stypy_return_type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type', floordiv___call_result_179686)
        
        # ################# End of '__truediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__truediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_179687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__truediv__'
        return stypy_return_type_179687


    @norecursion
    def __floordiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__floordiv__'
        module_type_store = module_type_store.open_function_context('__floordiv__', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__floordiv__')
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__floordiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__floordiv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__floordiv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__floordiv__(...)' code ##################

        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Call to __divmod__(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'other' (line 353)
        other_179690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'other', False)
        # Processing the call keyword arguments (line 353)
        kwargs_179691 = {}
        # Getting the type of 'self' (line 353)
        self_179688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'self', False)
        # Obtaining the member '__divmod__' of a type (line 353)
        divmod___179689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 14), self_179688, '__divmod__')
        # Calling __divmod__(args, kwargs) (line 353)
        divmod___call_result_179692 = invoke(stypy.reporting.localization.Localization(__file__, 353, 14), divmod___179689, *[other_179690], **kwargs_179691)
        
        # Assigning a type to the variable 'res' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'res', divmod___call_result_179692)
        
        
        # Getting the type of 'res' (line 354)
        res_179693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'res')
        # Getting the type of 'NotImplemented' (line 354)
        NotImplemented_179694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'NotImplemented')
        # Applying the binary operator 'is' (line 354)
        result_is__179695 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 11), 'is', res_179693, NotImplemented_179694)
        
        # Testing the type of an if condition (line 354)
        if_condition_179696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 8), result_is__179695)
        # Assigning a type to the variable 'if_condition_179696' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'if_condition_179696', if_condition_179696)
        # SSA begins for if statement (line 354)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'res' (line 355)
        res_179697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'stypy_return_type', res_179697)
        # SSA join for if statement (line 354)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_179698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 19), 'int')
        # Getting the type of 'res' (line 356)
        res_179699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'res')
        # Obtaining the member '__getitem__' of a type (line 356)
        getitem___179700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 15), res_179699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 356)
        subscript_call_result_179701 = invoke(stypy.reporting.localization.Localization(__file__, 356, 15), getitem___179700, int_179698)
        
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'stypy_return_type', subscript_call_result_179701)
        
        # ################# End of '__floordiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__floordiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_179702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__floordiv__'
        return stypy_return_type_179702


    @norecursion
    def __mod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mod__'
        module_type_store = module_type_store.open_function_context('__mod__', 358, 4, False)
        # Assigning a type to the variable 'self' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__mod__')
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__mod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__mod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mod__(...)' code ##################

        
        # Assigning a Call to a Name (line 359):
        
        # Assigning a Call to a Name (line 359):
        
        # Call to __divmod__(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'other' (line 359)
        other_179705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 30), 'other', False)
        # Processing the call keyword arguments (line 359)
        kwargs_179706 = {}
        # Getting the type of 'self' (line 359)
        self_179703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 14), 'self', False)
        # Obtaining the member '__divmod__' of a type (line 359)
        divmod___179704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 14), self_179703, '__divmod__')
        # Calling __divmod__(args, kwargs) (line 359)
        divmod___call_result_179707 = invoke(stypy.reporting.localization.Localization(__file__, 359, 14), divmod___179704, *[other_179705], **kwargs_179706)
        
        # Assigning a type to the variable 'res' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'res', divmod___call_result_179707)
        
        
        # Getting the type of 'res' (line 360)
        res_179708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 11), 'res')
        # Getting the type of 'NotImplemented' (line 360)
        NotImplemented_179709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 18), 'NotImplemented')
        # Applying the binary operator 'is' (line 360)
        result_is__179710 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 11), 'is', res_179708, NotImplemented_179709)
        
        # Testing the type of an if condition (line 360)
        if_condition_179711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 8), result_is__179710)
        # Assigning a type to the variable 'if_condition_179711' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'if_condition_179711', if_condition_179711)
        # SSA begins for if statement (line 360)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'res' (line 361)
        res_179712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 19), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'stypy_return_type', res_179712)
        # SSA join for if statement (line 360)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_179713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 19), 'int')
        # Getting the type of 'res' (line 362)
        res_179714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'res')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___179715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 15), res_179714, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_179716 = invoke(stypy.reporting.localization.Localization(__file__, 362, 15), getitem___179715, int_179713)
        
        # Assigning a type to the variable 'stypy_return_type' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'stypy_return_type', subscript_call_result_179716)
        
        # ################# End of '__mod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mod__' in the type store
        # Getting the type of 'stypy_return_type' (line 358)
        stypy_return_type_179717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179717)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mod__'
        return stypy_return_type_179717


    @norecursion
    def __divmod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__divmod__'
        module_type_store = module_type_store.open_function_context('__divmod__', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__divmod__')
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__divmod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__divmod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__divmod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__divmod__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 366):
        
        # Assigning a Call to a Name (line 366):
        
        # Call to _get_coefficients(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'other' (line 366)
        other_179720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 47), 'other', False)
        # Processing the call keyword arguments (line 366)
        kwargs_179721 = {}
        # Getting the type of 'self' (line 366)
        self_179718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'self', False)
        # Obtaining the member '_get_coefficients' of a type (line 366)
        _get_coefficients_179719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 24), self_179718, '_get_coefficients')
        # Calling _get_coefficients(args, kwargs) (line 366)
        _get_coefficients_call_result_179722 = invoke(stypy.reporting.localization.Localization(__file__, 366, 24), _get_coefficients_179719, *[other_179720], **kwargs_179721)
        
        # Assigning a type to the variable 'othercoef' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'othercoef', _get_coefficients_call_result_179722)
        
        # Assigning a Call to a Tuple (line 367):
        
        # Assigning a Call to a Name:
        
        # Call to _div(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'self' (line 367)
        self_179725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 33), 'self', False)
        # Obtaining the member 'coef' of a type (line 367)
        coef_179726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 33), self_179725, 'coef')
        # Getting the type of 'othercoef' (line 367)
        othercoef_179727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 44), 'othercoef', False)
        # Processing the call keyword arguments (line 367)
        kwargs_179728 = {}
        # Getting the type of 'self' (line 367)
        self_179723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'self', False)
        # Obtaining the member '_div' of a type (line 367)
        _div_179724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 23), self_179723, '_div')
        # Calling _div(args, kwargs) (line 367)
        _div_call_result_179729 = invoke(stypy.reporting.localization.Localization(__file__, 367, 23), _div_179724, *[coef_179726, othercoef_179727], **kwargs_179728)
        
        # Assigning a type to the variable 'call_assignment_179191' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'call_assignment_179191', _div_call_result_179729)
        
        # Assigning a Call to a Name (line 367):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_179732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 12), 'int')
        # Processing the call keyword arguments
        kwargs_179733 = {}
        # Getting the type of 'call_assignment_179191' (line 367)
        call_assignment_179191_179730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'call_assignment_179191', False)
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___179731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), call_assignment_179191_179730, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_179734 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179731, *[int_179732], **kwargs_179733)
        
        # Assigning a type to the variable 'call_assignment_179192' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'call_assignment_179192', getitem___call_result_179734)
        
        # Assigning a Name to a Name (line 367):
        # Getting the type of 'call_assignment_179192' (line 367)
        call_assignment_179192_179735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'call_assignment_179192')
        # Assigning a type to the variable 'quo' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'quo', call_assignment_179192_179735)
        
        # Assigning a Call to a Name (line 367):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_179738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 12), 'int')
        # Processing the call keyword arguments
        kwargs_179739 = {}
        # Getting the type of 'call_assignment_179191' (line 367)
        call_assignment_179191_179736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'call_assignment_179191', False)
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___179737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), call_assignment_179191_179736, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_179740 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179737, *[int_179738], **kwargs_179739)
        
        # Assigning a type to the variable 'call_assignment_179193' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'call_assignment_179193', getitem___call_result_179740)
        
        # Assigning a Name to a Name (line 367):
        # Getting the type of 'call_assignment_179193' (line 367)
        call_assignment_179193_179741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'call_assignment_179193')
        # Assigning a type to the variable 'rem' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 17), 'rem', call_assignment_179193_179741)
        # SSA branch for the except part of a try statement (line 365)
        # SSA branch for the except 'Tuple' branch of a try statement (line 365)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        
        # Obtaining an instance of the builtin type 'tuple' (line 368)
        tuple_179742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 368)
        # Adding element type (line 368)
        # Getting the type of 'TypeError' (line 368)
        TypeError_179743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'TypeError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 16), tuple_179742, TypeError_179743)
        # Adding element type (line 368)
        # Getting the type of 'ZeroDivisionError' (line 368)
        ZeroDivisionError_179744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 27), 'ZeroDivisionError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 16), tuple_179742, ZeroDivisionError_179744)
        
        # Assigning a type to the variable 'e' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'e', tuple_179742)
        # Getting the type of 'e' (line 369)
        e_179745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 18), 'e')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 369, 12), e_179745, 'raise parameter', BaseException)
        # SSA branch for the except '<any exception>' branch of a try statement (line 365)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 371)
        NotImplemented_179746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'stypy_return_type', NotImplemented_179746)
        # SSA join for try-except statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to __class__(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'quo' (line 372)
        quo_179749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 29), 'quo', False)
        # Getting the type of 'self' (line 372)
        self_179750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 34), 'self', False)
        # Obtaining the member 'domain' of a type (line 372)
        domain_179751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 34), self_179750, 'domain')
        # Getting the type of 'self' (line 372)
        self_179752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 47), 'self', False)
        # Obtaining the member 'window' of a type (line 372)
        window_179753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 47), self_179752, 'window')
        # Processing the call keyword arguments (line 372)
        kwargs_179754 = {}
        # Getting the type of 'self' (line 372)
        self_179747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 14), 'self', False)
        # Obtaining the member '__class__' of a type (line 372)
        class___179748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 14), self_179747, '__class__')
        # Calling __class__(args, kwargs) (line 372)
        class___call_result_179755 = invoke(stypy.reporting.localization.Localization(__file__, 372, 14), class___179748, *[quo_179749, domain_179751, window_179753], **kwargs_179754)
        
        # Assigning a type to the variable 'quo' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'quo', class___call_result_179755)
        
        # Assigning a Call to a Name (line 373):
        
        # Assigning a Call to a Name (line 373):
        
        # Call to __class__(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'rem' (line 373)
        rem_179758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 29), 'rem', False)
        # Getting the type of 'self' (line 373)
        self_179759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 34), 'self', False)
        # Obtaining the member 'domain' of a type (line 373)
        domain_179760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 34), self_179759, 'domain')
        # Getting the type of 'self' (line 373)
        self_179761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 47), 'self', False)
        # Obtaining the member 'window' of a type (line 373)
        window_179762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 47), self_179761, 'window')
        # Processing the call keyword arguments (line 373)
        kwargs_179763 = {}
        # Getting the type of 'self' (line 373)
        self_179756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 14), 'self', False)
        # Obtaining the member '__class__' of a type (line 373)
        class___179757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 14), self_179756, '__class__')
        # Calling __class__(args, kwargs) (line 373)
        class___call_result_179764 = invoke(stypy.reporting.localization.Localization(__file__, 373, 14), class___179757, *[rem_179758, domain_179760, window_179762], **kwargs_179763)
        
        # Assigning a type to the variable 'rem' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'rem', class___call_result_179764)
        
        # Obtaining an instance of the builtin type 'tuple' (line 374)
        tuple_179765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 374)
        # Adding element type (line 374)
        # Getting the type of 'quo' (line 374)
        quo_179766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'quo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 15), tuple_179765, quo_179766)
        # Adding element type (line 374)
        # Getting the type of 'rem' (line 374)
        rem_179767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'rem')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 15), tuple_179765, rem_179767)
        
        # Assigning a type to the variable 'stypy_return_type' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'stypy_return_type', tuple_179765)
        
        # ################# End of '__divmod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__divmod__' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_179768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__divmod__'
        return stypy_return_type_179768


    @norecursion
    def __pow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pow__'
        module_type_store = module_type_store.open_function_context('__pow__', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__pow__')
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__pow__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pow__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pow__(...)' code ##################

        
        # Assigning a Call to a Name (line 377):
        
        # Assigning a Call to a Name (line 377):
        
        # Call to _pow(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'self' (line 377)
        self_179771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 25), 'self', False)
        # Obtaining the member 'coef' of a type (line 377)
        coef_179772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 25), self_179771, 'coef')
        # Getting the type of 'other' (line 377)
        other_179773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 36), 'other', False)
        # Processing the call keyword arguments (line 377)
        # Getting the type of 'self' (line 377)
        self_179774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 52), 'self', False)
        # Obtaining the member 'maxpower' of a type (line 377)
        maxpower_179775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 52), self_179774, 'maxpower')
        keyword_179776 = maxpower_179775
        kwargs_179777 = {'maxpower': keyword_179776}
        # Getting the type of 'self' (line 377)
        self_179769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'self', False)
        # Obtaining the member '_pow' of a type (line 377)
        _pow_179770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 15), self_179769, '_pow')
        # Calling _pow(args, kwargs) (line 377)
        _pow_call_result_179778 = invoke(stypy.reporting.localization.Localization(__file__, 377, 15), _pow_179770, *[coef_179772, other_179773], **kwargs_179777)
        
        # Assigning a type to the variable 'coef' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'coef', _pow_call_result_179778)
        
        # Assigning a Call to a Name (line 378):
        
        # Assigning a Call to a Name (line 378):
        
        # Call to __class__(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'coef' (line 378)
        coef_179781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 29), 'coef', False)
        # Getting the type of 'self' (line 378)
        self_179782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 35), 'self', False)
        # Obtaining the member 'domain' of a type (line 378)
        domain_179783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 35), self_179782, 'domain')
        # Getting the type of 'self' (line 378)
        self_179784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 48), 'self', False)
        # Obtaining the member 'window' of a type (line 378)
        window_179785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 48), self_179784, 'window')
        # Processing the call keyword arguments (line 378)
        kwargs_179786 = {}
        # Getting the type of 'self' (line 378)
        self_179779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 14), 'self', False)
        # Obtaining the member '__class__' of a type (line 378)
        class___179780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 14), self_179779, '__class__')
        # Calling __class__(args, kwargs) (line 378)
        class___call_result_179787 = invoke(stypy.reporting.localization.Localization(__file__, 378, 14), class___179780, *[coef_179781, domain_179783, window_179785], **kwargs_179786)
        
        # Assigning a type to the variable 'res' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'res', class___call_result_179787)
        # Getting the type of 'res' (line 379)
        res_179788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', res_179788)
        
        # ################# End of '__pow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pow__' in the type store
        # Getting the type of 'stypy_return_type' (line 376)
        stypy_return_type_179789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pow__'
        return stypy_return_type_179789


    @norecursion
    def __radd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__radd__'
        module_type_store = module_type_store.open_function_context('__radd__', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__radd__')
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__radd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__radd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__radd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__radd__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to _add(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'other' (line 383)
        other_179792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 29), 'other', False)
        # Getting the type of 'self' (line 383)
        self_179793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 36), 'self', False)
        # Obtaining the member 'coef' of a type (line 383)
        coef_179794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 36), self_179793, 'coef')
        # Processing the call keyword arguments (line 383)
        kwargs_179795 = {}
        # Getting the type of 'self' (line 383)
        self_179790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'self', False)
        # Obtaining the member '_add' of a type (line 383)
        _add_179791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 19), self_179790, '_add')
        # Calling _add(args, kwargs) (line 383)
        _add_call_result_179796 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), _add_179791, *[other_179792, coef_179794], **kwargs_179795)
        
        # Assigning a type to the variable 'coef' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'coef', _add_call_result_179796)
        # SSA branch for the except part of a try statement (line 382)
        # SSA branch for the except '<any exception>' branch of a try statement (line 382)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 385)
        NotImplemented_179797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'stypy_return_type', NotImplemented_179797)
        # SSA join for try-except statement (line 382)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'coef' (line 386)
        coef_179800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 30), 'coef', False)
        # Getting the type of 'self' (line 386)
        self_179801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 386)
        domain_179802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 36), self_179801, 'domain')
        # Getting the type of 'self' (line 386)
        self_179803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 386)
        window_179804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 49), self_179803, 'window')
        # Processing the call keyword arguments (line 386)
        kwargs_179805 = {}
        # Getting the type of 'self' (line 386)
        self_179798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 386)
        class___179799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 15), self_179798, '__class__')
        # Calling __class__(args, kwargs) (line 386)
        class___call_result_179806 = invoke(stypy.reporting.localization.Localization(__file__, 386, 15), class___179799, *[coef_179800, domain_179802, window_179804], **kwargs_179805)
        
        # Assigning a type to the variable 'stypy_return_type' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'stypy_return_type', class___call_result_179806)
        
        # ################# End of '__radd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__radd__' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_179807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179807)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__radd__'
        return stypy_return_type_179807


    @norecursion
    def __rsub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rsub__'
        module_type_store = module_type_store.open_function_context('__rsub__', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__rsub__')
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__rsub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__rsub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rsub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rsub__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 390):
        
        # Assigning a Call to a Name (line 390):
        
        # Call to _sub(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'other' (line 390)
        other_179810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'other', False)
        # Getting the type of 'self' (line 390)
        self_179811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 36), 'self', False)
        # Obtaining the member 'coef' of a type (line 390)
        coef_179812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 36), self_179811, 'coef')
        # Processing the call keyword arguments (line 390)
        kwargs_179813 = {}
        # Getting the type of 'self' (line 390)
        self_179808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'self', False)
        # Obtaining the member '_sub' of a type (line 390)
        _sub_179809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), self_179808, '_sub')
        # Calling _sub(args, kwargs) (line 390)
        _sub_call_result_179814 = invoke(stypy.reporting.localization.Localization(__file__, 390, 19), _sub_179809, *[other_179810, coef_179812], **kwargs_179813)
        
        # Assigning a type to the variable 'coef' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'coef', _sub_call_result_179814)
        # SSA branch for the except part of a try statement (line 389)
        # SSA branch for the except '<any exception>' branch of a try statement (line 389)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 392)
        NotImplemented_179815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'stypy_return_type', NotImplemented_179815)
        # SSA join for try-except statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'coef' (line 393)
        coef_179818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 30), 'coef', False)
        # Getting the type of 'self' (line 393)
        self_179819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 393)
        domain_179820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 36), self_179819, 'domain')
        # Getting the type of 'self' (line 393)
        self_179821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 393)
        window_179822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 49), self_179821, 'window')
        # Processing the call keyword arguments (line 393)
        kwargs_179823 = {}
        # Getting the type of 'self' (line 393)
        self_179816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 393)
        class___179817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 15), self_179816, '__class__')
        # Calling __class__(args, kwargs) (line 393)
        class___call_result_179824 = invoke(stypy.reporting.localization.Localization(__file__, 393, 15), class___179817, *[coef_179818, domain_179820, window_179822], **kwargs_179823)
        
        # Assigning a type to the variable 'stypy_return_type' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'stypy_return_type', class___call_result_179824)
        
        # ################# End of '__rsub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rsub__' in the type store
        # Getting the type of 'stypy_return_type' (line 388)
        stypy_return_type_179825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179825)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rsub__'
        return stypy_return_type_179825


    @norecursion
    def __rmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmul__'
        module_type_store = module_type_store.open_function_context('__rmul__', 395, 4, False)
        # Assigning a type to the variable 'self' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__rmul__')
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__rmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__rmul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmul__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to _mul(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'other' (line 397)
        other_179828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 29), 'other', False)
        # Getting the type of 'self' (line 397)
        self_179829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 36), 'self', False)
        # Obtaining the member 'coef' of a type (line 397)
        coef_179830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 36), self_179829, 'coef')
        # Processing the call keyword arguments (line 397)
        kwargs_179831 = {}
        # Getting the type of 'self' (line 397)
        self_179826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 'self', False)
        # Obtaining the member '_mul' of a type (line 397)
        _mul_179827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 19), self_179826, '_mul')
        # Calling _mul(args, kwargs) (line 397)
        _mul_call_result_179832 = invoke(stypy.reporting.localization.Localization(__file__, 397, 19), _mul_179827, *[other_179828, coef_179830], **kwargs_179831)
        
        # Assigning a type to the variable 'coef' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'coef', _mul_call_result_179832)
        # SSA branch for the except part of a try statement (line 396)
        # SSA branch for the except '<any exception>' branch of a try statement (line 396)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 399)
        NotImplemented_179833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'stypy_return_type', NotImplemented_179833)
        # SSA join for try-except statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'coef' (line 400)
        coef_179836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 30), 'coef', False)
        # Getting the type of 'self' (line 400)
        self_179837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 400)
        domain_179838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 36), self_179837, 'domain')
        # Getting the type of 'self' (line 400)
        self_179839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 400)
        window_179840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 49), self_179839, 'window')
        # Processing the call keyword arguments (line 400)
        kwargs_179841 = {}
        # Getting the type of 'self' (line 400)
        self_179834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 400)
        class___179835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 15), self_179834, '__class__')
        # Calling __class__(args, kwargs) (line 400)
        class___call_result_179842 = invoke(stypy.reporting.localization.Localization(__file__, 400, 15), class___179835, *[coef_179836, domain_179838, window_179840], **kwargs_179841)
        
        # Assigning a type to the variable 'stypy_return_type' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'stypy_return_type', class___call_result_179842)
        
        # ################# End of '__rmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 395)
        stypy_return_type_179843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179843)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmul__'
        return stypy_return_type_179843


    @norecursion
    def __rdiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rdiv__'
        module_type_store = module_type_store.open_function_context('__rdiv__', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__rdiv__')
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__rdiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__rdiv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rdiv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rdiv__(...)' code ##################

        
        # Call to __rfloordiv__(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'other' (line 404)
        other_179846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 34), 'other', False)
        # Processing the call keyword arguments (line 404)
        kwargs_179847 = {}
        # Getting the type of 'self' (line 404)
        self_179844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'self', False)
        # Obtaining the member '__rfloordiv__' of a type (line 404)
        rfloordiv___179845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), self_179844, '__rfloordiv__')
        # Calling __rfloordiv__(args, kwargs) (line 404)
        rfloordiv___call_result_179848 = invoke(stypy.reporting.localization.Localization(__file__, 404, 15), rfloordiv___179845, *[other_179846], **kwargs_179847)
        
        # Assigning a type to the variable 'stypy_return_type' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'stypy_return_type', rfloordiv___call_result_179848)
        
        # ################# End of '__rdiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rdiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_179849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rdiv__'
        return stypy_return_type_179849


    @norecursion
    def __rtruediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rtruediv__'
        module_type_store = module_type_store.open_function_context('__rtruediv__', 406, 4, False)
        # Assigning a type to the variable 'self' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__rtruediv__')
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__rtruediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__rtruediv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rtruediv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rtruediv__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 409)
        NotImplemented_179850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'stypy_return_type', NotImplemented_179850)
        
        # ################# End of '__rtruediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rtruediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 406)
        stypy_return_type_179851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rtruediv__'
        return stypy_return_type_179851


    @norecursion
    def __rfloordiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rfloordiv__'
        module_type_store = module_type_store.open_function_context('__rfloordiv__', 411, 4, False)
        # Assigning a type to the variable 'self' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__rfloordiv__')
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__rfloordiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__rfloordiv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rfloordiv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rfloordiv__(...)' code ##################

        
        # Assigning a Call to a Name (line 412):
        
        # Assigning a Call to a Name (line 412):
        
        # Call to __rdivmod__(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'other' (line 412)
        other_179854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 31), 'other', False)
        # Processing the call keyword arguments (line 412)
        kwargs_179855 = {}
        # Getting the type of 'self' (line 412)
        self_179852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 14), 'self', False)
        # Obtaining the member '__rdivmod__' of a type (line 412)
        rdivmod___179853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 14), self_179852, '__rdivmod__')
        # Calling __rdivmod__(args, kwargs) (line 412)
        rdivmod___call_result_179856 = invoke(stypy.reporting.localization.Localization(__file__, 412, 14), rdivmod___179853, *[other_179854], **kwargs_179855)
        
        # Assigning a type to the variable 'res' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'res', rdivmod___call_result_179856)
        
        
        # Getting the type of 'res' (line 413)
        res_179857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), 'res')
        # Getting the type of 'NotImplemented' (line 413)
        NotImplemented_179858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 18), 'NotImplemented')
        # Applying the binary operator 'is' (line 413)
        result_is__179859 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 11), 'is', res_179857, NotImplemented_179858)
        
        # Testing the type of an if condition (line 413)
        if_condition_179860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 8), result_is__179859)
        # Assigning a type to the variable 'if_condition_179860' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'if_condition_179860', if_condition_179860)
        # SSA begins for if statement (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'res' (line 414)
        res_179861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 19), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'stypy_return_type', res_179861)
        # SSA join for if statement (line 413)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_179862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 19), 'int')
        # Getting the type of 'res' (line 415)
        res_179863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'res')
        # Obtaining the member '__getitem__' of a type (line 415)
        getitem___179864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 15), res_179863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 415)
        subscript_call_result_179865 = invoke(stypy.reporting.localization.Localization(__file__, 415, 15), getitem___179864, int_179862)
        
        # Assigning a type to the variable 'stypy_return_type' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'stypy_return_type', subscript_call_result_179865)
        
        # ################# End of '__rfloordiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rfloordiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 411)
        stypy_return_type_179866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179866)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rfloordiv__'
        return stypy_return_type_179866


    @norecursion
    def __rmod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmod__'
        module_type_store = module_type_store.open_function_context('__rmod__', 417, 4, False)
        # Assigning a type to the variable 'self' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__rmod__')
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__rmod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__rmod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmod__(...)' code ##################

        
        # Assigning a Call to a Name (line 418):
        
        # Assigning a Call to a Name (line 418):
        
        # Call to __rdivmod__(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'other' (line 418)
        other_179869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 31), 'other', False)
        # Processing the call keyword arguments (line 418)
        kwargs_179870 = {}
        # Getting the type of 'self' (line 418)
        self_179867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 14), 'self', False)
        # Obtaining the member '__rdivmod__' of a type (line 418)
        rdivmod___179868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 14), self_179867, '__rdivmod__')
        # Calling __rdivmod__(args, kwargs) (line 418)
        rdivmod___call_result_179871 = invoke(stypy.reporting.localization.Localization(__file__, 418, 14), rdivmod___179868, *[other_179869], **kwargs_179870)
        
        # Assigning a type to the variable 'res' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'res', rdivmod___call_result_179871)
        
        
        # Getting the type of 'res' (line 419)
        res_179872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'res')
        # Getting the type of 'NotImplemented' (line 419)
        NotImplemented_179873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 18), 'NotImplemented')
        # Applying the binary operator 'is' (line 419)
        result_is__179874 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 11), 'is', res_179872, NotImplemented_179873)
        
        # Testing the type of an if condition (line 419)
        if_condition_179875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 8), result_is__179874)
        # Assigning a type to the variable 'if_condition_179875' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'if_condition_179875', if_condition_179875)
        # SSA begins for if statement (line 419)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'res' (line 420)
        res_179876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 19), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'stypy_return_type', res_179876)
        # SSA join for if statement (line 419)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_179877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 19), 'int')
        # Getting the type of 'res' (line 421)
        res_179878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'res')
        # Obtaining the member '__getitem__' of a type (line 421)
        getitem___179879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 15), res_179878, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 421)
        subscript_call_result_179880 = invoke(stypy.reporting.localization.Localization(__file__, 421, 15), getitem___179879, int_179877)
        
        # Assigning a type to the variable 'stypy_return_type' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'stypy_return_type', subscript_call_result_179880)
        
        # ################# End of '__rmod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmod__' in the type store
        # Getting the type of 'stypy_return_type' (line 417)
        stypy_return_type_179881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmod__'
        return stypy_return_type_179881


    @norecursion
    def __rdivmod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rdivmod__'
        module_type_store = module_type_store.open_function_context('__rdivmod__', 423, 4, False)
        # Assigning a type to the variable 'self' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__rdivmod__')
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__rdivmod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__rdivmod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rdivmod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rdivmod__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 424)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 425):
        
        # Assigning a Call to a Name:
        
        # Call to _div(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'other' (line 425)
        other_179884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 33), 'other', False)
        # Getting the type of 'self' (line 425)
        self_179885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 40), 'self', False)
        # Obtaining the member 'coef' of a type (line 425)
        coef_179886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 40), self_179885, 'coef')
        # Processing the call keyword arguments (line 425)
        kwargs_179887 = {}
        # Getting the type of 'self' (line 425)
        self_179882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'self', False)
        # Obtaining the member '_div' of a type (line 425)
        _div_179883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 23), self_179882, '_div')
        # Calling _div(args, kwargs) (line 425)
        _div_call_result_179888 = invoke(stypy.reporting.localization.Localization(__file__, 425, 23), _div_179883, *[other_179884, coef_179886], **kwargs_179887)
        
        # Assigning a type to the variable 'call_assignment_179194' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'call_assignment_179194', _div_call_result_179888)
        
        # Assigning a Call to a Name (line 425):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_179891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 12), 'int')
        # Processing the call keyword arguments
        kwargs_179892 = {}
        # Getting the type of 'call_assignment_179194' (line 425)
        call_assignment_179194_179889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'call_assignment_179194', False)
        # Obtaining the member '__getitem__' of a type (line 425)
        getitem___179890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), call_assignment_179194_179889, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_179893 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179890, *[int_179891], **kwargs_179892)
        
        # Assigning a type to the variable 'call_assignment_179195' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'call_assignment_179195', getitem___call_result_179893)
        
        # Assigning a Name to a Name (line 425):
        # Getting the type of 'call_assignment_179195' (line 425)
        call_assignment_179195_179894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'call_assignment_179195')
        # Assigning a type to the variable 'quo' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'quo', call_assignment_179195_179894)
        
        # Assigning a Call to a Name (line 425):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_179897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 12), 'int')
        # Processing the call keyword arguments
        kwargs_179898 = {}
        # Getting the type of 'call_assignment_179194' (line 425)
        call_assignment_179194_179895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'call_assignment_179194', False)
        # Obtaining the member '__getitem__' of a type (line 425)
        getitem___179896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), call_assignment_179194_179895, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_179899 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179896, *[int_179897], **kwargs_179898)
        
        # Assigning a type to the variable 'call_assignment_179196' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'call_assignment_179196', getitem___call_result_179899)
        
        # Assigning a Name to a Name (line 425):
        # Getting the type of 'call_assignment_179196' (line 425)
        call_assignment_179196_179900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'call_assignment_179196')
        # Assigning a type to the variable 'rem' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 17), 'rem', call_assignment_179196_179900)
        # SSA branch for the except part of a try statement (line 424)
        # SSA branch for the except 'ZeroDivisionError' branch of a try statement (line 424)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ZeroDivisionError' (line 426)
        ZeroDivisionError_179901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'ZeroDivisionError')
        # Assigning a type to the variable 'e' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'e', ZeroDivisionError_179901)
        # Getting the type of 'e' (line 427)
        e_179902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 18), 'e')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 427, 12), e_179902, 'raise parameter', BaseException)
        # SSA branch for the except '<any exception>' branch of a try statement (line 424)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 429)
        NotImplemented_179903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'stypy_return_type', NotImplemented_179903)
        # SSA join for try-except statement (line 424)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 430):
        
        # Assigning a Call to a Name (line 430):
        
        # Call to __class__(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'quo' (line 430)
        quo_179906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 29), 'quo', False)
        # Getting the type of 'self' (line 430)
        self_179907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'self', False)
        # Obtaining the member 'domain' of a type (line 430)
        domain_179908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 34), self_179907, 'domain')
        # Getting the type of 'self' (line 430)
        self_179909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 47), 'self', False)
        # Obtaining the member 'window' of a type (line 430)
        window_179910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 47), self_179909, 'window')
        # Processing the call keyword arguments (line 430)
        kwargs_179911 = {}
        # Getting the type of 'self' (line 430)
        self_179904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 14), 'self', False)
        # Obtaining the member '__class__' of a type (line 430)
        class___179905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 14), self_179904, '__class__')
        # Calling __class__(args, kwargs) (line 430)
        class___call_result_179912 = invoke(stypy.reporting.localization.Localization(__file__, 430, 14), class___179905, *[quo_179906, domain_179908, window_179910], **kwargs_179911)
        
        # Assigning a type to the variable 'quo' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'quo', class___call_result_179912)
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to __class__(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'rem' (line 431)
        rem_179915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 29), 'rem', False)
        # Getting the type of 'self' (line 431)
        self_179916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 34), 'self', False)
        # Obtaining the member 'domain' of a type (line 431)
        domain_179917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 34), self_179916, 'domain')
        # Getting the type of 'self' (line 431)
        self_179918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 47), 'self', False)
        # Obtaining the member 'window' of a type (line 431)
        window_179919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 47), self_179918, 'window')
        # Processing the call keyword arguments (line 431)
        kwargs_179920 = {}
        # Getting the type of 'self' (line 431)
        self_179913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 14), 'self', False)
        # Obtaining the member '__class__' of a type (line 431)
        class___179914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 14), self_179913, '__class__')
        # Calling __class__(args, kwargs) (line 431)
        class___call_result_179921 = invoke(stypy.reporting.localization.Localization(__file__, 431, 14), class___179914, *[rem_179915, domain_179917, window_179919], **kwargs_179920)
        
        # Assigning a type to the variable 'rem' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'rem', class___call_result_179921)
        
        # Obtaining an instance of the builtin type 'tuple' (line 432)
        tuple_179922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 432)
        # Adding element type (line 432)
        # Getting the type of 'quo' (line 432)
        quo_179923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 15), 'quo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 15), tuple_179922, quo_179923)
        # Adding element type (line 432)
        # Getting the type of 'rem' (line 432)
        rem_179924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 20), 'rem')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 15), tuple_179922, rem_179924)
        
        # Assigning a type to the variable 'stypy_return_type' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'stypy_return_type', tuple_179922)
        
        # ################# End of '__rdivmod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rdivmod__' in the type store
        # Getting the type of 'stypy_return_type' (line 423)
        stypy_return_type_179925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179925)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rdivmod__'
        return stypy_return_type_179925


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 437, 4, False)
        # Assigning a type to the variable 'self' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__eq__')
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a BoolOp to a Name (line 438):
        
        # Assigning a BoolOp to a Name (line 438):
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'other' (line 438)
        other_179927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 26), 'other', False)
        # Getting the type of 'self' (line 438)
        self_179928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'self', False)
        # Obtaining the member '__class__' of a type (line 438)
        class___179929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 33), self_179928, '__class__')
        # Processing the call keyword arguments (line 438)
        kwargs_179930 = {}
        # Getting the type of 'isinstance' (line 438)
        isinstance_179926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 438)
        isinstance_call_result_179931 = invoke(stypy.reporting.localization.Localization(__file__, 438, 15), isinstance_179926, *[other_179927, class___179929], **kwargs_179930)
        
        
        # Call to all(...): (line 439)
        # Processing the call arguments (line 439)
        
        # Getting the type of 'self' (line 439)
        self_179934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 22), 'self', False)
        # Obtaining the member 'domain' of a type (line 439)
        domain_179935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 22), self_179934, 'domain')
        # Getting the type of 'other' (line 439)
        other_179936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 37), 'other', False)
        # Obtaining the member 'domain' of a type (line 439)
        domain_179937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 37), other_179936, 'domain')
        # Applying the binary operator '==' (line 439)
        result_eq_179938 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 22), '==', domain_179935, domain_179937)
        
        # Processing the call keyword arguments (line 439)
        kwargs_179939 = {}
        # Getting the type of 'np' (line 439)
        np_179932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 439)
        all_179933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 15), np_179932, 'all')
        # Calling all(args, kwargs) (line 439)
        all_call_result_179940 = invoke(stypy.reporting.localization.Localization(__file__, 439, 15), all_179933, *[result_eq_179938], **kwargs_179939)
        
        # Applying the binary operator 'and' (line 438)
        result_and_keyword_179941 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'and', isinstance_call_result_179931, all_call_result_179940)
        
        # Call to all(...): (line 440)
        # Processing the call arguments (line 440)
        
        # Getting the type of 'self' (line 440)
        self_179944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 22), 'self', False)
        # Obtaining the member 'window' of a type (line 440)
        window_179945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 22), self_179944, 'window')
        # Getting the type of 'other' (line 440)
        other_179946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 37), 'other', False)
        # Obtaining the member 'window' of a type (line 440)
        window_179947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 37), other_179946, 'window')
        # Applying the binary operator '==' (line 440)
        result_eq_179948 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 22), '==', window_179945, window_179947)
        
        # Processing the call keyword arguments (line 440)
        kwargs_179949 = {}
        # Getting the type of 'np' (line 440)
        np_179942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 440)
        all_179943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 15), np_179942, 'all')
        # Calling all(args, kwargs) (line 440)
        all_call_result_179950 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), all_179943, *[result_eq_179948], **kwargs_179949)
        
        # Applying the binary operator 'and' (line 438)
        result_and_keyword_179951 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'and', result_and_keyword_179941, all_call_result_179950)
        
        # Getting the type of 'self' (line 441)
        self_179952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'self')
        # Obtaining the member 'coef' of a type (line 441)
        coef_179953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 16), self_179952, 'coef')
        # Obtaining the member 'shape' of a type (line 441)
        shape_179954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 16), coef_179953, 'shape')
        # Getting the type of 'other' (line 441)
        other_179955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 35), 'other')
        # Obtaining the member 'coef' of a type (line 441)
        coef_179956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 35), other_179955, 'coef')
        # Obtaining the member 'shape' of a type (line 441)
        shape_179957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 35), coef_179956, 'shape')
        # Applying the binary operator '==' (line 441)
        result_eq_179958 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 16), '==', shape_179954, shape_179957)
        
        # Applying the binary operator 'and' (line 438)
        result_and_keyword_179959 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'and', result_and_keyword_179951, result_eq_179958)
        
        # Call to all(...): (line 442)
        # Processing the call arguments (line 442)
        
        # Getting the type of 'self' (line 442)
        self_179962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 22), 'self', False)
        # Obtaining the member 'coef' of a type (line 442)
        coef_179963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 22), self_179962, 'coef')
        # Getting the type of 'other' (line 442)
        other_179964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 35), 'other', False)
        # Obtaining the member 'coef' of a type (line 442)
        coef_179965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 35), other_179964, 'coef')
        # Applying the binary operator '==' (line 442)
        result_eq_179966 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 22), '==', coef_179963, coef_179965)
        
        # Processing the call keyword arguments (line 442)
        kwargs_179967 = {}
        # Getting the type of 'np' (line 442)
        np_179960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 442)
        all_179961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), np_179960, 'all')
        # Calling all(args, kwargs) (line 442)
        all_call_result_179968 = invoke(stypy.reporting.localization.Localization(__file__, 442, 15), all_179961, *[result_eq_179966], **kwargs_179967)
        
        # Applying the binary operator 'and' (line 438)
        result_and_keyword_179969 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'and', result_and_keyword_179959, all_call_result_179968)
        
        # Assigning a type to the variable 'res' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'res', result_and_keyword_179969)
        # Getting the type of 'res' (line 443)
        res_179970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'stypy_return_type', res_179970)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 437)
        stypy_return_type_179971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179971)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_179971


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 445, 4, False)
        # Assigning a type to the variable 'self' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.__ne__')
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        
        
        # Call to __eq__(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'other' (line 446)
        other_179974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 31), 'other', False)
        # Processing the call keyword arguments (line 446)
        kwargs_179975 = {}
        # Getting the type of 'self' (line 446)
        self_179972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 19), 'self', False)
        # Obtaining the member '__eq__' of a type (line 446)
        eq___179973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 19), self_179972, '__eq__')
        # Calling __eq__(args, kwargs) (line 446)
        eq___call_result_179976 = invoke(stypy.reporting.localization.Localization(__file__, 446, 19), eq___179973, *[other_179974], **kwargs_179975)
        
        # Applying the 'not' unary operator (line 446)
        result_not__179977 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 15), 'not', eq___call_result_179976)
        
        # Assigning a type to the variable 'stypy_return_type' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'stypy_return_type', result_not__179977)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 445)
        stypy_return_type_179978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179978)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_179978


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.copy.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.copy.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.copy')
        ABCPolyBase.copy.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        str_179979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, (-1)), 'str', 'Return a copy.\n\n        Returns\n        -------\n        new_series : series\n            Copy of self.\n\n        ')
        
        # Call to __class__(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'self' (line 461)
        self_179982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 30), 'self', False)
        # Obtaining the member 'coef' of a type (line 461)
        coef_179983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 30), self_179982, 'coef')
        # Getting the type of 'self' (line 461)
        self_179984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 41), 'self', False)
        # Obtaining the member 'domain' of a type (line 461)
        domain_179985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 41), self_179984, 'domain')
        # Getting the type of 'self' (line 461)
        self_179986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 54), 'self', False)
        # Obtaining the member 'window' of a type (line 461)
        window_179987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 54), self_179986, 'window')
        # Processing the call keyword arguments (line 461)
        kwargs_179988 = {}
        # Getting the type of 'self' (line 461)
        self_179980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 461)
        class___179981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 15), self_179980, '__class__')
        # Calling __class__(args, kwargs) (line 461)
        class___call_result_179989 = invoke(stypy.reporting.localization.Localization(__file__, 461, 15), class___179981, *[coef_179983, domain_179985, window_179987], **kwargs_179988)
        
        # Assigning a type to the variable 'stypy_return_type' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'stypy_return_type', class___call_result_179989)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_179990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179990)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_179990


    @norecursion
    def degree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'degree'
        module_type_store = module_type_store.open_function_context('degree', 463, 4, False)
        # Assigning a type to the variable 'self' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.degree.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.degree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.degree.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.degree.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.degree')
        ABCPolyBase.degree.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.degree.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.degree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.degree.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.degree.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.degree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.degree.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.degree', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'degree', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'degree(...)' code ##################

        str_179991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, (-1)), 'str', 'The degree of the series.\n\n        .. versionadded:: 1.5.0\n\n        Returns\n        -------\n        degree : int\n            Degree of the series, one less than the number of coefficients.\n\n        ')
        
        # Call to len(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'self' (line 474)
        self_179993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 19), 'self', False)
        # Processing the call keyword arguments (line 474)
        kwargs_179994 = {}
        # Getting the type of 'len' (line 474)
        len_179992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'len', False)
        # Calling len(args, kwargs) (line 474)
        len_call_result_179995 = invoke(stypy.reporting.localization.Localization(__file__, 474, 15), len_179992, *[self_179993], **kwargs_179994)
        
        int_179996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 27), 'int')
        # Applying the binary operator '-' (line 474)
        result_sub_179997 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 15), '-', len_call_result_179995, int_179996)
        
        # Assigning a type to the variable 'stypy_return_type' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'stypy_return_type', result_sub_179997)
        
        # ################# End of 'degree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'degree' in the type store
        # Getting the type of 'stypy_return_type' (line 463)
        stypy_return_type_179998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179998)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'degree'
        return stypy_return_type_179998


    @norecursion
    def cutdeg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cutdeg'
        module_type_store = module_type_store.open_function_context('cutdeg', 476, 4, False)
        # Assigning a type to the variable 'self' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.cutdeg')
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_param_names_list', ['deg'])
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.cutdeg.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.cutdeg', ['deg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cutdeg', localization, ['deg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cutdeg(...)' code ##################

        str_179999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, (-1)), 'str', 'Truncate series to the given degree.\n\n        Reduce the degree of the series to `deg` by discarding the\n        high order terms. If `deg` is greater than the current degree a\n        copy of the current series is returned. This can be useful in least\n        squares where the coefficients of the high degree terms may be very\n        small.\n\n        .. versionadded:: 1.5.0\n\n        Parameters\n        ----------\n        deg : non-negative int\n            The series is reduced to degree `deg` by discarding the high\n            order terms. The value of `deg` must be a non-negative integer.\n\n        Returns\n        -------\n        new_series : series\n            New instance of series with reduced degree.\n\n        ')
        
        # Call to truncate(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'deg' (line 499)
        deg_180002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 29), 'deg', False)
        int_180003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 35), 'int')
        # Applying the binary operator '+' (line 499)
        result_add_180004 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 29), '+', deg_180002, int_180003)
        
        # Processing the call keyword arguments (line 499)
        kwargs_180005 = {}
        # Getting the type of 'self' (line 499)
        self_180000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'self', False)
        # Obtaining the member 'truncate' of a type (line 499)
        truncate_180001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), self_180000, 'truncate')
        # Calling truncate(args, kwargs) (line 499)
        truncate_call_result_180006 = invoke(stypy.reporting.localization.Localization(__file__, 499, 15), truncate_180001, *[result_add_180004], **kwargs_180005)
        
        # Assigning a type to the variable 'stypy_return_type' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'stypy_return_type', truncate_call_result_180006)
        
        # ################# End of 'cutdeg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cutdeg' in the type store
        # Getting the type of 'stypy_return_type' (line 476)
        stypy_return_type_180007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180007)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cutdeg'
        return stypy_return_type_180007


    @norecursion
    def trim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_180008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 23), 'int')
        defaults = [int_180008]
        # Create a new context for function 'trim'
        module_type_store = module_type_store.open_function_context('trim', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.trim.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.trim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.trim.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.trim.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.trim')
        ABCPolyBase.trim.__dict__.__setitem__('stypy_param_names_list', ['tol'])
        ABCPolyBase.trim.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.trim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.trim.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.trim.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.trim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.trim.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.trim', ['tol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trim', localization, ['tol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trim(...)' code ##################

        str_180009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, (-1)), 'str', 'Remove trailing coefficients\n\n        Remove trailing coefficients until a coefficient is reached whose\n        absolute value greater than `tol` or the beginning of the series is\n        reached. If all the coefficients would be removed the series is set\n        to ``[0]``. A new series instance is returned with the new\n        coefficients.  The current instance remains unchanged.\n\n        Parameters\n        ----------\n        tol : non-negative number.\n            All trailing coefficients less than `tol` will be removed.\n\n        Returns\n        -------\n        new_series : series\n            Contains the new set of coefficients.\n\n        ')
        
        # Assigning a Call to a Name (line 521):
        
        # Assigning a Call to a Name (line 521):
        
        # Call to trimcoef(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'self' (line 521)
        self_180012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 27), 'self', False)
        # Obtaining the member 'coef' of a type (line 521)
        coef_180013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 27), self_180012, 'coef')
        # Getting the type of 'tol' (line 521)
        tol_180014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 38), 'tol', False)
        # Processing the call keyword arguments (line 521)
        kwargs_180015 = {}
        # Getting the type of 'pu' (line 521)
        pu_180010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 15), 'pu', False)
        # Obtaining the member 'trimcoef' of a type (line 521)
        trimcoef_180011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 15), pu_180010, 'trimcoef')
        # Calling trimcoef(args, kwargs) (line 521)
        trimcoef_call_result_180016 = invoke(stypy.reporting.localization.Localization(__file__, 521, 15), trimcoef_180011, *[coef_180013, tol_180014], **kwargs_180015)
        
        # Assigning a type to the variable 'coef' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'coef', trimcoef_call_result_180016)
        
        # Call to __class__(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'coef' (line 522)
        coef_180019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 30), 'coef', False)
        # Getting the type of 'self' (line 522)
        self_180020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 522)
        domain_180021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 36), self_180020, 'domain')
        # Getting the type of 'self' (line 522)
        self_180022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 522)
        window_180023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 49), self_180022, 'window')
        # Processing the call keyword arguments (line 522)
        kwargs_180024 = {}
        # Getting the type of 'self' (line 522)
        self_180017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 522)
        class___180018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 15), self_180017, '__class__')
        # Calling __class__(args, kwargs) (line 522)
        class___call_result_180025 = invoke(stypy.reporting.localization.Localization(__file__, 522, 15), class___180018, *[coef_180019, domain_180021, window_180023], **kwargs_180024)
        
        # Assigning a type to the variable 'stypy_return_type' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'stypy_return_type', class___call_result_180025)
        
        # ################# End of 'trim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trim' in the type store
        # Getting the type of 'stypy_return_type' (line 501)
        stypy_return_type_180026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180026)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trim'
        return stypy_return_type_180026


    @norecursion
    def truncate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'truncate'
        module_type_store = module_type_store.open_function_context('truncate', 524, 4, False)
        # Assigning a type to the variable 'self' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.truncate')
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_param_names_list', ['size'])
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.truncate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.truncate', ['size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'truncate', localization, ['size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'truncate(...)' code ##################

        str_180027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, (-1)), 'str', 'Truncate series to length `size`.\n\n        Reduce the series to length `size` by discarding the high\n        degree terms. The value of `size` must be a positive integer. This\n        can be useful in least squares where the coefficients of the\n        high degree terms may be very small.\n\n        Parameters\n        ----------\n        size : positive int\n            The series is reduced to length `size` by discarding the high\n            degree terms. The value of `size` must be a positive integer.\n\n        Returns\n        -------\n        new_series : series\n            New instance of series with truncated coefficients.\n\n        ')
        
        # Assigning a Call to a Name (line 544):
        
        # Assigning a Call to a Name (line 544):
        
        # Call to int(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'size' (line 544)
        size_180029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 20), 'size', False)
        # Processing the call keyword arguments (line 544)
        kwargs_180030 = {}
        # Getting the type of 'int' (line 544)
        int_180028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'int', False)
        # Calling int(args, kwargs) (line 544)
        int_call_result_180031 = invoke(stypy.reporting.localization.Localization(__file__, 544, 16), int_180028, *[size_180029], **kwargs_180030)
        
        # Assigning a type to the variable 'isize' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'isize', int_call_result_180031)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'isize' (line 545)
        isize_180032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 11), 'isize')
        # Getting the type of 'size' (line 545)
        size_180033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 20), 'size')
        # Applying the binary operator '!=' (line 545)
        result_ne_180034 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 11), '!=', isize_180032, size_180033)
        
        
        # Getting the type of 'isize' (line 545)
        isize_180035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 28), 'isize')
        int_180036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 36), 'int')
        # Applying the binary operator '<' (line 545)
        result_lt_180037 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 28), '<', isize_180035, int_180036)
        
        # Applying the binary operator 'or' (line 545)
        result_or_keyword_180038 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 11), 'or', result_ne_180034, result_lt_180037)
        
        # Testing the type of an if condition (line 545)
        if_condition_180039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 545, 8), result_or_keyword_180038)
        # Assigning a type to the variable 'if_condition_180039' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'if_condition_180039', if_condition_180039)
        # SSA begins for if statement (line 545)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 546)
        # Processing the call arguments (line 546)
        str_180041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 29), 'str', 'size must be a positive integer')
        # Processing the call keyword arguments (line 546)
        kwargs_180042 = {}
        # Getting the type of 'ValueError' (line 546)
        ValueError_180040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 546)
        ValueError_call_result_180043 = invoke(stypy.reporting.localization.Localization(__file__, 546, 18), ValueError_180040, *[str_180041], **kwargs_180042)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 546, 12), ValueError_call_result_180043, 'raise parameter', BaseException)
        # SSA join for if statement (line 545)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'isize' (line 547)
        isize_180044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 11), 'isize')
        
        # Call to len(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'self' (line 547)
        self_180046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 24), 'self', False)
        # Obtaining the member 'coef' of a type (line 547)
        coef_180047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 24), self_180046, 'coef')
        # Processing the call keyword arguments (line 547)
        kwargs_180048 = {}
        # Getting the type of 'len' (line 547)
        len_180045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 20), 'len', False)
        # Calling len(args, kwargs) (line 547)
        len_call_result_180049 = invoke(stypy.reporting.localization.Localization(__file__, 547, 20), len_180045, *[coef_180047], **kwargs_180048)
        
        # Applying the binary operator '>=' (line 547)
        result_ge_180050 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 11), '>=', isize_180044, len_call_result_180049)
        
        # Testing the type of an if condition (line 547)
        if_condition_180051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 8), result_ge_180050)
        # Assigning a type to the variable 'if_condition_180051' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'if_condition_180051', if_condition_180051)
        # SSA begins for if statement (line 547)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 548):
        
        # Assigning a Attribute to a Name (line 548):
        # Getting the type of 'self' (line 548)
        self_180052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'self')
        # Obtaining the member 'coef' of a type (line 548)
        coef_180053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 19), self_180052, 'coef')
        # Assigning a type to the variable 'coef' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'coef', coef_180053)
        # SSA branch for the else part of an if statement (line 547)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 550):
        
        # Assigning a Subscript to a Name (line 550):
        
        # Obtaining the type of the subscript
        # Getting the type of 'isize' (line 550)
        isize_180054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 30), 'isize')
        slice_180055 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 550, 19), None, isize_180054, None)
        # Getting the type of 'self' (line 550)
        self_180056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'self')
        # Obtaining the member 'coef' of a type (line 550)
        coef_180057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 19), self_180056, 'coef')
        # Obtaining the member '__getitem__' of a type (line 550)
        getitem___180058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 19), coef_180057, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 550)
        subscript_call_result_180059 = invoke(stypy.reporting.localization.Localization(__file__, 550, 19), getitem___180058, slice_180055)
        
        # Assigning a type to the variable 'coef' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'coef', subscript_call_result_180059)
        # SSA join for if statement (line 547)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'coef' (line 551)
        coef_180062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 30), 'coef', False)
        # Getting the type of 'self' (line 551)
        self_180063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 551)
        domain_180064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 36), self_180063, 'domain')
        # Getting the type of 'self' (line 551)
        self_180065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 551)
        window_180066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 49), self_180065, 'window')
        # Processing the call keyword arguments (line 551)
        kwargs_180067 = {}
        # Getting the type of 'self' (line 551)
        self_180060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 551)
        class___180061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 15), self_180060, '__class__')
        # Calling __class__(args, kwargs) (line 551)
        class___call_result_180068 = invoke(stypy.reporting.localization.Localization(__file__, 551, 15), class___180061, *[coef_180062, domain_180064, window_180066], **kwargs_180067)
        
        # Assigning a type to the variable 'stypy_return_type' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'stypy_return_type', class___call_result_180068)
        
        # ################# End of 'truncate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'truncate' in the type store
        # Getting the type of 'stypy_return_type' (line 524)
        stypy_return_type_180069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'truncate'
        return stypy_return_type_180069


    @norecursion
    def convert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 553)
        None_180070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 29), 'None')
        # Getting the type of 'None' (line 553)
        None_180071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 40), 'None')
        # Getting the type of 'None' (line 553)
        None_180072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 53), 'None')
        defaults = [None_180070, None_180071, None_180072]
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 553, 4, False)
        # Assigning a type to the variable 'self' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.convert.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.convert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.convert.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.convert')
        ABCPolyBase.convert.__dict__.__setitem__('stypy_param_names_list', ['domain', 'kind', 'window'])
        ABCPolyBase.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.convert.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.convert', ['domain', 'kind', 'window'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert', localization, ['domain', 'kind', 'window'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert(...)' code ##################

        str_180073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, (-1)), 'str', 'Convert series to a different kind and/or domain and/or window.\n\n        Parameters\n        ----------\n        domain : array_like, optional\n            The domain of the converted series. If the value is None,\n            the default domain of `kind` is used.\n        kind : class, optional\n            The polynomial series type class to which the current instance\n            should be converted. If kind is None, then the class of the\n            current instance is used.\n        window : array_like, optional\n            The window of the converted series. If the value is None,\n            the default window of `kind` is used.\n\n        Returns\n        -------\n        new_series : series\n            The returned class can be of different type than the current\n            instance and/or have a different domain and/or different\n            window.\n\n        Notes\n        -----\n        Conversion between domains and class types can result in\n        numerically ill defined series.\n\n        Examples\n        --------\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 585)
        # Getting the type of 'kind' (line 585)
        kind_180074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 11), 'kind')
        # Getting the type of 'None' (line 585)
        None_180075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 19), 'None')
        
        (may_be_180076, more_types_in_union_180077) = may_be_none(kind_180074, None_180075)

        if may_be_180076:

            if more_types_in_union_180077:
                # Runtime conditional SSA (line 585)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 586):
            
            # Assigning a Attribute to a Name (line 586):
            # Getting the type of 'self' (line 586)
            self_180078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 19), 'self')
            # Obtaining the member '__class__' of a type (line 586)
            class___180079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 19), self_180078, '__class__')
            # Assigning a type to the variable 'kind' (line 586)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'kind', class___180079)

            if more_types_in_union_180077:
                # SSA join for if statement (line 585)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 587)
        # Getting the type of 'domain' (line 587)
        domain_180080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 11), 'domain')
        # Getting the type of 'None' (line 587)
        None_180081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 21), 'None')
        
        (may_be_180082, more_types_in_union_180083) = may_be_none(domain_180080, None_180081)

        if may_be_180082:

            if more_types_in_union_180083:
                # Runtime conditional SSA (line 587)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 588):
            
            # Assigning a Attribute to a Name (line 588):
            # Getting the type of 'kind' (line 588)
            kind_180084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 21), 'kind')
            # Obtaining the member 'domain' of a type (line 588)
            domain_180085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 21), kind_180084, 'domain')
            # Assigning a type to the variable 'domain' (line 588)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'domain', domain_180085)

            if more_types_in_union_180083:
                # SSA join for if statement (line 587)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 589)
        # Getting the type of 'window' (line 589)
        window_180086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'window')
        # Getting the type of 'None' (line 589)
        None_180087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'None')
        
        (may_be_180088, more_types_in_union_180089) = may_be_none(window_180086, None_180087)

        if may_be_180088:

            if more_types_in_union_180089:
                # Runtime conditional SSA (line 589)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 590):
            
            # Assigning a Attribute to a Name (line 590):
            # Getting the type of 'kind' (line 590)
            kind_180090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 21), 'kind')
            # Obtaining the member 'window' of a type (line 590)
            window_180091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 21), kind_180090, 'window')
            # Assigning a type to the variable 'window' (line 590)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'window', window_180091)

            if more_types_in_union_180089:
                # SSA join for if statement (line 589)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to self(...): (line 591)
        # Processing the call arguments (line 591)
        
        # Call to identity(...): (line 591)
        # Processing the call arguments (line 591)
        # Getting the type of 'domain' (line 591)
        domain_180095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 34), 'domain', False)
        # Processing the call keyword arguments (line 591)
        # Getting the type of 'window' (line 591)
        window_180096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 49), 'window', False)
        keyword_180097 = window_180096
        kwargs_180098 = {'window': keyword_180097}
        # Getting the type of 'kind' (line 591)
        kind_180093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 20), 'kind', False)
        # Obtaining the member 'identity' of a type (line 591)
        identity_180094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 20), kind_180093, 'identity')
        # Calling identity(args, kwargs) (line 591)
        identity_call_result_180099 = invoke(stypy.reporting.localization.Localization(__file__, 591, 20), identity_180094, *[domain_180095], **kwargs_180098)
        
        # Processing the call keyword arguments (line 591)
        kwargs_180100 = {}
        # Getting the type of 'self' (line 591)
        self_180092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'self', False)
        # Calling self(args, kwargs) (line 591)
        self_call_result_180101 = invoke(stypy.reporting.localization.Localization(__file__, 591, 15), self_180092, *[identity_call_result_180099], **kwargs_180100)
        
        # Assigning a type to the variable 'stypy_return_type' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'stypy_return_type', self_call_result_180101)
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 553)
        stypy_return_type_180102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_180102


    @norecursion
    def mapparms(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mapparms'
        module_type_store = module_type_store.open_function_context('mapparms', 593, 4, False)
        # Assigning a type to the variable 'self' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.mapparms')
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.mapparms.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.mapparms', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mapparms', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mapparms(...)' code ##################

        str_180103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, (-1)), 'str', 'Return the mapping parameters.\n\n        The returned values define a linear map ``off + scl*x`` that is\n        applied to the input arguments before the series is evaluated. The\n        map depends on the ``domain`` and ``window``; if the current\n        ``domain`` is equal to the ``window`` the resulting map is the\n        identity.  If the coefficients of the series instance are to be\n        used by themselves outside this class, then the linear function\n        must be substituted for the ``x`` in the standard representation of\n        the base polynomials.\n\n        Returns\n        -------\n        off, scl : float or complex\n            The mapping function is defined by ``off + scl*x``.\n\n        Notes\n        -----\n        If the current domain is the interval ``[l1, r1]`` and the window\n        is ``[l2, r2]``, then the linear mapping function ``L`` is\n        defined by the equations::\n\n            L(l1) = l2\n            L(r1) = r2\n\n        ')
        
        # Call to mapparms(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 'self' (line 620)
        self_180106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 27), 'self', False)
        # Obtaining the member 'domain' of a type (line 620)
        domain_180107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 27), self_180106, 'domain')
        # Getting the type of 'self' (line 620)
        self_180108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 40), 'self', False)
        # Obtaining the member 'window' of a type (line 620)
        window_180109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 40), self_180108, 'window')
        # Processing the call keyword arguments (line 620)
        kwargs_180110 = {}
        # Getting the type of 'pu' (line 620)
        pu_180104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'pu', False)
        # Obtaining the member 'mapparms' of a type (line 620)
        mapparms_180105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), pu_180104, 'mapparms')
        # Calling mapparms(args, kwargs) (line 620)
        mapparms_call_result_180111 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), mapparms_180105, *[domain_180107, window_180109], **kwargs_180110)
        
        # Assigning a type to the variable 'stypy_return_type' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'stypy_return_type', mapparms_call_result_180111)
        
        # ################# End of 'mapparms(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mapparms' in the type store
        # Getting the type of 'stypy_return_type' (line 593)
        stypy_return_type_180112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mapparms'
        return stypy_return_type_180112


    @norecursion
    def integ(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_180113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 22), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 622)
        list_180114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 622)
        
        # Getting the type of 'None' (line 622)
        None_180115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 36), 'None')
        defaults = [int_180113, list_180114, None_180115]
        # Create a new context for function 'integ'
        module_type_store = module_type_store.open_function_context('integ', 622, 4, False)
        # Assigning a type to the variable 'self' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.integ.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.integ.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.integ.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.integ.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.integ')
        ABCPolyBase.integ.__dict__.__setitem__('stypy_param_names_list', ['m', 'k', 'lbnd'])
        ABCPolyBase.integ.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.integ.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.integ.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.integ.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.integ.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.integ.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.integ', ['m', 'k', 'lbnd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integ', localization, ['m', 'k', 'lbnd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integ(...)' code ##################

        str_180116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, (-1)), 'str', 'Integrate.\n\n        Return a series instance that is the definite integral of the\n        current series.\n\n        Parameters\n        ----------\n        m : non-negative int\n            The number of integrations to perform.\n        k : array_like\n            Integration constants. The first constant is applied to the\n            first integration, the second to the second, and so on. The\n            list of values must less than or equal to `m` in length and any\n            missing values are set to zero.\n        lbnd : Scalar\n            The lower bound of the definite integral.\n\n        Returns\n        -------\n        new_series : series\n            A new series representing the integral. The domain is the same\n            as the domain of the integrated series.\n\n        ')
        
        # Assigning a Call to a Tuple (line 647):
        
        # Assigning a Call to a Name:
        
        # Call to mapparms(...): (line 647)
        # Processing the call keyword arguments (line 647)
        kwargs_180119 = {}
        # Getting the type of 'self' (line 647)
        self_180117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 19), 'self', False)
        # Obtaining the member 'mapparms' of a type (line 647)
        mapparms_180118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 19), self_180117, 'mapparms')
        # Calling mapparms(args, kwargs) (line 647)
        mapparms_call_result_180120 = invoke(stypy.reporting.localization.Localization(__file__, 647, 19), mapparms_180118, *[], **kwargs_180119)
        
        # Assigning a type to the variable 'call_assignment_179197' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'call_assignment_179197', mapparms_call_result_180120)
        
        # Assigning a Call to a Name (line 647):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180124 = {}
        # Getting the type of 'call_assignment_179197' (line 647)
        call_assignment_179197_180121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'call_assignment_179197', False)
        # Obtaining the member '__getitem__' of a type (line 647)
        getitem___180122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 8), call_assignment_179197_180121, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180125 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180122, *[int_180123], **kwargs_180124)
        
        # Assigning a type to the variable 'call_assignment_179198' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'call_assignment_179198', getitem___call_result_180125)
        
        # Assigning a Name to a Name (line 647):
        # Getting the type of 'call_assignment_179198' (line 647)
        call_assignment_179198_180126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'call_assignment_179198')
        # Assigning a type to the variable 'off' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'off', call_assignment_179198_180126)
        
        # Assigning a Call to a Name (line 647):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180130 = {}
        # Getting the type of 'call_assignment_179197' (line 647)
        call_assignment_179197_180127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'call_assignment_179197', False)
        # Obtaining the member '__getitem__' of a type (line 647)
        getitem___180128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 8), call_assignment_179197_180127, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180131 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180128, *[int_180129], **kwargs_180130)
        
        # Assigning a type to the variable 'call_assignment_179199' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'call_assignment_179199', getitem___call_result_180131)
        
        # Assigning a Name to a Name (line 647):
        # Getting the type of 'call_assignment_179199' (line 647)
        call_assignment_179199_180132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'call_assignment_179199')
        # Assigning a type to the variable 'scl' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 13), 'scl', call_assignment_179199_180132)
        
        # Type idiom detected: calculating its left and rigth part (line 648)
        # Getting the type of 'lbnd' (line 648)
        lbnd_180133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 11), 'lbnd')
        # Getting the type of 'None' (line 648)
        None_180134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 19), 'None')
        
        (may_be_180135, more_types_in_union_180136) = may_be_none(lbnd_180133, None_180134)

        if may_be_180135:

            if more_types_in_union_180136:
                # Runtime conditional SSA (line 648)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 649):
            
            # Assigning a Num to a Name (line 649):
            int_180137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 19), 'int')
            # Assigning a type to the variable 'lbnd' (line 649)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'lbnd', int_180137)

            if more_types_in_union_180136:
                # Runtime conditional SSA for else branch (line 648)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_180135) or more_types_in_union_180136):
            
            # Assigning a BinOp to a Name (line 651):
            
            # Assigning a BinOp to a Name (line 651):
            # Getting the type of 'off' (line 651)
            off_180138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 19), 'off')
            # Getting the type of 'scl' (line 651)
            scl_180139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 25), 'scl')
            # Getting the type of 'lbnd' (line 651)
            lbnd_180140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 29), 'lbnd')
            # Applying the binary operator '*' (line 651)
            result_mul_180141 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 25), '*', scl_180139, lbnd_180140)
            
            # Applying the binary operator '+' (line 651)
            result_add_180142 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 19), '+', off_180138, result_mul_180141)
            
            # Assigning a type to the variable 'lbnd' (line 651)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'lbnd', result_add_180142)

            if (may_be_180135 and more_types_in_union_180136):
                # SSA join for if statement (line 648)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 652):
        
        # Assigning a Call to a Name (line 652):
        
        # Call to _int(...): (line 652)
        # Processing the call arguments (line 652)
        # Getting the type of 'self' (line 652)
        self_180145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 25), 'self', False)
        # Obtaining the member 'coef' of a type (line 652)
        coef_180146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 25), self_180145, 'coef')
        # Getting the type of 'm' (line 652)
        m_180147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 36), 'm', False)
        # Getting the type of 'k' (line 652)
        k_180148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 39), 'k', False)
        # Getting the type of 'lbnd' (line 652)
        lbnd_180149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 42), 'lbnd', False)
        float_180150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 48), 'float')
        # Getting the type of 'scl' (line 652)
        scl_180151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 51), 'scl', False)
        # Applying the binary operator 'div' (line 652)
        result_div_180152 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 48), 'div', float_180150, scl_180151)
        
        # Processing the call keyword arguments (line 652)
        kwargs_180153 = {}
        # Getting the type of 'self' (line 652)
        self_180143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 15), 'self', False)
        # Obtaining the member '_int' of a type (line 652)
        _int_180144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 15), self_180143, '_int')
        # Calling _int(args, kwargs) (line 652)
        _int_call_result_180154 = invoke(stypy.reporting.localization.Localization(__file__, 652, 15), _int_180144, *[coef_180146, m_180147, k_180148, lbnd_180149, result_div_180152], **kwargs_180153)
        
        # Assigning a type to the variable 'coef' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'coef', _int_call_result_180154)
        
        # Call to __class__(...): (line 653)
        # Processing the call arguments (line 653)
        # Getting the type of 'coef' (line 653)
        coef_180157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 30), 'coef', False)
        # Getting the type of 'self' (line 653)
        self_180158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 653)
        domain_180159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 36), self_180158, 'domain')
        # Getting the type of 'self' (line 653)
        self_180160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 653)
        window_180161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 49), self_180160, 'window')
        # Processing the call keyword arguments (line 653)
        kwargs_180162 = {}
        # Getting the type of 'self' (line 653)
        self_180155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 653)
        class___180156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 15), self_180155, '__class__')
        # Calling __class__(args, kwargs) (line 653)
        class___call_result_180163 = invoke(stypy.reporting.localization.Localization(__file__, 653, 15), class___180156, *[coef_180157, domain_180159, window_180161], **kwargs_180162)
        
        # Assigning a type to the variable 'stypy_return_type' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'stypy_return_type', class___call_result_180163)
        
        # ################# End of 'integ(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integ' in the type store
        # Getting the type of 'stypy_return_type' (line 622)
        stypy_return_type_180164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180164)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integ'
        return stypy_return_type_180164


    @norecursion
    def deriv(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_180165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 22), 'int')
        defaults = [int_180165]
        # Create a new context for function 'deriv'
        module_type_store = module_type_store.open_function_context('deriv', 655, 4, False)
        # Assigning a type to the variable 'self' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.deriv')
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_param_names_list', ['m'])
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.deriv.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.deriv', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'deriv', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'deriv(...)' code ##################

        str_180166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, (-1)), 'str', 'Differentiate.\n\n        Return a series instance of that is the derivative of the current\n        series.\n\n        Parameters\n        ----------\n        m : non-negative int\n            Find the derivative of order `m`.\n\n        Returns\n        -------\n        new_series : series\n            A new series representing the derivative. The domain is the same\n            as the domain of the differentiated series.\n\n        ')
        
        # Assigning a Call to a Tuple (line 673):
        
        # Assigning a Call to a Name:
        
        # Call to mapparms(...): (line 673)
        # Processing the call keyword arguments (line 673)
        kwargs_180169 = {}
        # Getting the type of 'self' (line 673)
        self_180167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 19), 'self', False)
        # Obtaining the member 'mapparms' of a type (line 673)
        mapparms_180168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 19), self_180167, 'mapparms')
        # Calling mapparms(args, kwargs) (line 673)
        mapparms_call_result_180170 = invoke(stypy.reporting.localization.Localization(__file__, 673, 19), mapparms_180168, *[], **kwargs_180169)
        
        # Assigning a type to the variable 'call_assignment_179200' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'call_assignment_179200', mapparms_call_result_180170)
        
        # Assigning a Call to a Name (line 673):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180174 = {}
        # Getting the type of 'call_assignment_179200' (line 673)
        call_assignment_179200_180171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'call_assignment_179200', False)
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___180172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 8), call_assignment_179200_180171, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180175 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180172, *[int_180173], **kwargs_180174)
        
        # Assigning a type to the variable 'call_assignment_179201' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'call_assignment_179201', getitem___call_result_180175)
        
        # Assigning a Name to a Name (line 673):
        # Getting the type of 'call_assignment_179201' (line 673)
        call_assignment_179201_180176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'call_assignment_179201')
        # Assigning a type to the variable 'off' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'off', call_assignment_179201_180176)
        
        # Assigning a Call to a Name (line 673):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180180 = {}
        # Getting the type of 'call_assignment_179200' (line 673)
        call_assignment_179200_180177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'call_assignment_179200', False)
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___180178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 8), call_assignment_179200_180177, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180181 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180178, *[int_180179], **kwargs_180180)
        
        # Assigning a type to the variable 'call_assignment_179202' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'call_assignment_179202', getitem___call_result_180181)
        
        # Assigning a Name to a Name (line 673):
        # Getting the type of 'call_assignment_179202' (line 673)
        call_assignment_179202_180182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'call_assignment_179202')
        # Assigning a type to the variable 'scl' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 13), 'scl', call_assignment_179202_180182)
        
        # Assigning a Call to a Name (line 674):
        
        # Assigning a Call to a Name (line 674):
        
        # Call to _der(...): (line 674)
        # Processing the call arguments (line 674)
        # Getting the type of 'self' (line 674)
        self_180185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 25), 'self', False)
        # Obtaining the member 'coef' of a type (line 674)
        coef_180186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 25), self_180185, 'coef')
        # Getting the type of 'm' (line 674)
        m_180187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 36), 'm', False)
        # Getting the type of 'scl' (line 674)
        scl_180188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 39), 'scl', False)
        # Processing the call keyword arguments (line 674)
        kwargs_180189 = {}
        # Getting the type of 'self' (line 674)
        self_180183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 15), 'self', False)
        # Obtaining the member '_der' of a type (line 674)
        _der_180184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 15), self_180183, '_der')
        # Calling _der(args, kwargs) (line 674)
        _der_call_result_180190 = invoke(stypy.reporting.localization.Localization(__file__, 674, 15), _der_180184, *[coef_180186, m_180187, scl_180188], **kwargs_180189)
        
        # Assigning a type to the variable 'coef' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'coef', _der_call_result_180190)
        
        # Call to __class__(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'coef' (line 675)
        coef_180193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 30), 'coef', False)
        # Getting the type of 'self' (line 675)
        self_180194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 36), 'self', False)
        # Obtaining the member 'domain' of a type (line 675)
        domain_180195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 36), self_180194, 'domain')
        # Getting the type of 'self' (line 675)
        self_180196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 49), 'self', False)
        # Obtaining the member 'window' of a type (line 675)
        window_180197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 49), self_180196, 'window')
        # Processing the call keyword arguments (line 675)
        kwargs_180198 = {}
        # Getting the type of 'self' (line 675)
        self_180191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 675)
        class___180192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 15), self_180191, '__class__')
        # Calling __class__(args, kwargs) (line 675)
        class___call_result_180199 = invoke(stypy.reporting.localization.Localization(__file__, 675, 15), class___180192, *[coef_180193, domain_180195, window_180197], **kwargs_180198)
        
        # Assigning a type to the variable 'stypy_return_type' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'stypy_return_type', class___call_result_180199)
        
        # ################# End of 'deriv(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'deriv' in the type store
        # Getting the type of 'stypy_return_type' (line 655)
        stypy_return_type_180200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'deriv'
        return stypy_return_type_180200


    @norecursion
    def roots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'roots'
        module_type_store = module_type_store.open_function_context('roots', 677, 4, False)
        # Assigning a type to the variable 'self' (line 678)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.roots.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.roots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.roots.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.roots.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.roots')
        ABCPolyBase.roots.__dict__.__setitem__('stypy_param_names_list', [])
        ABCPolyBase.roots.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.roots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.roots.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.roots.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.roots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.roots.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.roots', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'roots', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'roots(...)' code ##################

        str_180201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, (-1)), 'str', 'Return the roots of the series polynomial.\n\n        Compute the roots for the series. Note that the accuracy of the\n        roots decrease the further outside the domain they lie.\n\n        Returns\n        -------\n        roots : ndarray\n            Array containing the roots of the series.\n\n        ')
        
        # Assigning a Call to a Name (line 689):
        
        # Assigning a Call to a Name (line 689):
        
        # Call to _roots(...): (line 689)
        # Processing the call arguments (line 689)
        # Getting the type of 'self' (line 689)
        self_180204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 28), 'self', False)
        # Obtaining the member 'coef' of a type (line 689)
        coef_180205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 28), self_180204, 'coef')
        # Processing the call keyword arguments (line 689)
        kwargs_180206 = {}
        # Getting the type of 'self' (line 689)
        self_180202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'self', False)
        # Obtaining the member '_roots' of a type (line 689)
        _roots_180203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 16), self_180202, '_roots')
        # Calling _roots(args, kwargs) (line 689)
        _roots_call_result_180207 = invoke(stypy.reporting.localization.Localization(__file__, 689, 16), _roots_180203, *[coef_180205], **kwargs_180206)
        
        # Assigning a type to the variable 'roots' (line 689)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'roots', _roots_call_result_180207)
        
        # Call to mapdomain(...): (line 690)
        # Processing the call arguments (line 690)
        # Getting the type of 'roots' (line 690)
        roots_180210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 28), 'roots', False)
        # Getting the type of 'self' (line 690)
        self_180211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 35), 'self', False)
        # Obtaining the member 'window' of a type (line 690)
        window_180212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 35), self_180211, 'window')
        # Getting the type of 'self' (line 690)
        self_180213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 48), 'self', False)
        # Obtaining the member 'domain' of a type (line 690)
        domain_180214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 48), self_180213, 'domain')
        # Processing the call keyword arguments (line 690)
        kwargs_180215 = {}
        # Getting the type of 'pu' (line 690)
        pu_180208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 15), 'pu', False)
        # Obtaining the member 'mapdomain' of a type (line 690)
        mapdomain_180209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 15), pu_180208, 'mapdomain')
        # Calling mapdomain(args, kwargs) (line 690)
        mapdomain_call_result_180216 = invoke(stypy.reporting.localization.Localization(__file__, 690, 15), mapdomain_180209, *[roots_180210, window_180212, domain_180214], **kwargs_180215)
        
        # Assigning a type to the variable 'stypy_return_type' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'stypy_return_type', mapdomain_call_result_180216)
        
        # ################# End of 'roots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'roots' in the type store
        # Getting the type of 'stypy_return_type' (line 677)
        stypy_return_type_180217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180217)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'roots'
        return stypy_return_type_180217


    @norecursion
    def linspace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_180218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 25), 'int')
        # Getting the type of 'None' (line 692)
        None_180219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 37), 'None')
        defaults = [int_180218, None_180219]
        # Create a new context for function 'linspace'
        module_type_store = module_type_store.open_function_context('linspace', 692, 4, False)
        # Assigning a type to the variable 'self' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.linspace')
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_param_names_list', ['n', 'domain'])
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.linspace.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.linspace', ['n', 'domain'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'linspace', localization, ['n', 'domain'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'linspace(...)' code ##################

        str_180220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, (-1)), 'str', 'Return x, y values at equally spaced points in domain.\n\n        Returns the x, y values at `n` linearly spaced points across the\n        domain.  Here y is the value of the polynomial at the points x. By\n        default the domain is the same as that of the series instance.\n        This method is intended mostly as a plotting aid.\n\n        .. versionadded:: 1.5.0\n\n        Parameters\n        ----------\n        n : int, optional\n            Number of point pairs to return. The default value is 100.\n        domain : {None, array_like}, optional\n            If not None, the specified domain is used instead of that of\n            the calling instance. It should be of the form ``[beg,end]``.\n            The default is None which case the class domain is used.\n\n        Returns\n        -------\n        x, y : ndarray\n            x is equal to linspace(self.domain[0], self.domain[1], n) and\n            y is the series evaluated at element of x.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 718)
        # Getting the type of 'domain' (line 718)
        domain_180221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 11), 'domain')
        # Getting the type of 'None' (line 718)
        None_180222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 21), 'None')
        
        (may_be_180223, more_types_in_union_180224) = may_be_none(domain_180221, None_180222)

        if may_be_180223:

            if more_types_in_union_180224:
                # Runtime conditional SSA (line 718)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 719):
            
            # Assigning a Attribute to a Name (line 719):
            # Getting the type of 'self' (line 719)
            self_180225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 21), 'self')
            # Obtaining the member 'domain' of a type (line 719)
            domain_180226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 21), self_180225, 'domain')
            # Assigning a type to the variable 'domain' (line 719)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'domain', domain_180226)

            if more_types_in_union_180224:
                # SSA join for if statement (line 718)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 720):
        
        # Assigning a Call to a Name (line 720):
        
        # Call to linspace(...): (line 720)
        # Processing the call arguments (line 720)
        
        # Obtaining the type of the subscript
        int_180229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 31), 'int')
        # Getting the type of 'domain' (line 720)
        domain_180230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 24), 'domain', False)
        # Obtaining the member '__getitem__' of a type (line 720)
        getitem___180231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 24), domain_180230, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 720)
        subscript_call_result_180232 = invoke(stypy.reporting.localization.Localization(__file__, 720, 24), getitem___180231, int_180229)
        
        
        # Obtaining the type of the subscript
        int_180233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 42), 'int')
        # Getting the type of 'domain' (line 720)
        domain_180234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 35), 'domain', False)
        # Obtaining the member '__getitem__' of a type (line 720)
        getitem___180235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 35), domain_180234, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 720)
        subscript_call_result_180236 = invoke(stypy.reporting.localization.Localization(__file__, 720, 35), getitem___180235, int_180233)
        
        # Getting the type of 'n' (line 720)
        n_180237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 46), 'n', False)
        # Processing the call keyword arguments (line 720)
        kwargs_180238 = {}
        # Getting the type of 'np' (line 720)
        np_180227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 720)
        linspace_180228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 12), np_180227, 'linspace')
        # Calling linspace(args, kwargs) (line 720)
        linspace_call_result_180239 = invoke(stypy.reporting.localization.Localization(__file__, 720, 12), linspace_180228, *[subscript_call_result_180232, subscript_call_result_180236, n_180237], **kwargs_180238)
        
        # Assigning a type to the variable 'x' (line 720)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'x', linspace_call_result_180239)
        
        # Assigning a Call to a Name (line 721):
        
        # Assigning a Call to a Name (line 721):
        
        # Call to self(...): (line 721)
        # Processing the call arguments (line 721)
        # Getting the type of 'x' (line 721)
        x_180241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 17), 'x', False)
        # Processing the call keyword arguments (line 721)
        kwargs_180242 = {}
        # Getting the type of 'self' (line 721)
        self_180240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'self', False)
        # Calling self(args, kwargs) (line 721)
        self_call_result_180243 = invoke(stypy.reporting.localization.Localization(__file__, 721, 12), self_180240, *[x_180241], **kwargs_180242)
        
        # Assigning a type to the variable 'y' (line 721)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'y', self_call_result_180243)
        
        # Obtaining an instance of the builtin type 'tuple' (line 722)
        tuple_180244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 722)
        # Adding element type (line 722)
        # Getting the type of 'x' (line 722)
        x_180245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 15), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 15), tuple_180244, x_180245)
        # Adding element type (line 722)
        # Getting the type of 'y' (line 722)
        y_180246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 18), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 15), tuple_180244, y_180246)
        
        # Assigning a type to the variable 'stypy_return_type' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'stypy_return_type', tuple_180244)
        
        # ################# End of 'linspace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'linspace' in the type store
        # Getting the type of 'stypy_return_type' (line 692)
        stypy_return_type_180247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180247)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'linspace'
        return stypy_return_type_180247


    @norecursion
    def fit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 725)
        None_180248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 35), 'None')
        # Getting the type of 'None' (line 725)
        None_180249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 47), 'None')
        # Getting the type of 'False' (line 725)
        False_180250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 58), 'False')
        # Getting the type of 'None' (line 725)
        None_180251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 67), 'None')
        # Getting the type of 'None' (line 726)
        None_180252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 15), 'None')
        defaults = [None_180248, None_180249, False_180250, None_180251, None_180252]
        # Create a new context for function 'fit'
        module_type_store = module_type_store.open_function_context('fit', 724, 4, False)
        # Assigning a type to the variable 'self' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.fit.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.fit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.fit.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.fit.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.fit')
        ABCPolyBase.fit.__dict__.__setitem__('stypy_param_names_list', ['x', 'y', 'deg', 'domain', 'rcond', 'full', 'w', 'window'])
        ABCPolyBase.fit.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.fit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.fit.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.fit.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.fit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.fit.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.fit', ['x', 'y', 'deg', 'domain', 'rcond', 'full', 'w', 'window'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fit', localization, ['x', 'y', 'deg', 'domain', 'rcond', 'full', 'w', 'window'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fit(...)' code ##################

        str_180253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, (-1)), 'str', "Least squares fit to data.\n\n        Return a series instance that is the least squares fit to the data\n        `y` sampled at `x`. The domain of the returned instance can be\n        specified and this will often result in a superior fit with less\n        chance of ill conditioning.\n\n        Parameters\n        ----------\n        x : array_like, shape (M,)\n            x-coordinates of the M sample points ``(x[i], y[i])``.\n        y : array_like, shape (M,) or (M, K)\n            y-coordinates of the sample points. Several data sets of sample\n            points sharing the same x-coordinates can be fitted at once by\n            passing in a 2D-array that contains one dataset per column.\n        deg : int or 1-D array_like\n            Degree(s) of the fitting polynomials. If `deg` is a single integer\n            all terms up to and including the `deg`'th term are included in the\n            fit. For Numpy versions >= 1.11 a list of integers specifying the\n            degrees of the terms to include may be used instead.\n        domain : {None, [beg, end], []}, optional\n            Domain to use for the returned series. If ``None``,\n            then a minimal domain that covers the points `x` is chosen.  If\n            ``[]`` the class domain is used. The default value was the\n            class domain in NumPy 1.4 and ``None`` in later versions.\n            The ``[]`` option was added in numpy 1.5.0.\n        rcond : float, optional\n            Relative condition number of the fit. Singular values smaller\n            than this relative to the largest singular value will be\n            ignored. The default value is len(x)*eps, where eps is the\n            relative precision of the float type, about 2e-16 in most\n            cases.\n        full : bool, optional\n            Switch determining nature of return value. When it is False\n            (the default) just the coefficients are returned, when True\n            diagnostic information from the singular value decomposition is\n            also returned.\n        w : array_like, shape (M,), optional\n            Weights. If not None the contribution of each point\n            ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the\n            weights are chosen so that the errors of the products\n            ``w[i]*y[i]`` all have the same variance.  The default value is\n            None.\n\n            .. versionadded:: 1.5.0\n        window : {[beg, end]}, optional\n            Window to use for the returned series. The default\n            value is the default class domain\n\n            .. versionadded:: 1.6.0\n\n        Returns\n        -------\n        new_series : series\n            A series that represents the least squares fit to the data and\n            has the domain specified in the call.\n\n        [resid, rank, sv, rcond] : list\n            These values are only returned if `full` = True\n\n            resid -- sum of squared residuals of the least squares fit\n            rank -- the numerical rank of the scaled Vandermonde matrix\n            sv -- singular values of the scaled Vandermonde matrix\n            rcond -- value of `rcond`.\n\n            For more details, see `linalg.lstsq`.\n\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 795)
        # Getting the type of 'domain' (line 795)
        domain_180254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 11), 'domain')
        # Getting the type of 'None' (line 795)
        None_180255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 21), 'None')
        
        (may_be_180256, more_types_in_union_180257) = may_be_none(domain_180254, None_180255)

        if may_be_180256:

            if more_types_in_union_180257:
                # Runtime conditional SSA (line 795)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 796):
            
            # Assigning a Call to a Name (line 796):
            
            # Call to getdomain(...): (line 796)
            # Processing the call arguments (line 796)
            # Getting the type of 'x' (line 796)
            x_180260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 34), 'x', False)
            # Processing the call keyword arguments (line 796)
            kwargs_180261 = {}
            # Getting the type of 'pu' (line 796)
            pu_180258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 21), 'pu', False)
            # Obtaining the member 'getdomain' of a type (line 796)
            getdomain_180259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 21), pu_180258, 'getdomain')
            # Calling getdomain(args, kwargs) (line 796)
            getdomain_call_result_180262 = invoke(stypy.reporting.localization.Localization(__file__, 796, 21), getdomain_180259, *[x_180260], **kwargs_180261)
            
            # Assigning a type to the variable 'domain' (line 796)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 12), 'domain', getdomain_call_result_180262)

            if more_types_in_union_180257:
                # Runtime conditional SSA for else branch (line 795)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_180256) or more_types_in_union_180257):
            
            
            # Evaluating a boolean operation
            
            
            # Call to type(...): (line 797)
            # Processing the call arguments (line 797)
            # Getting the type of 'domain' (line 797)
            domain_180264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 18), 'domain', False)
            # Processing the call keyword arguments (line 797)
            kwargs_180265 = {}
            # Getting the type of 'type' (line 797)
            type_180263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 13), 'type', False)
            # Calling type(args, kwargs) (line 797)
            type_call_result_180266 = invoke(stypy.reporting.localization.Localization(__file__, 797, 13), type_180263, *[domain_180264], **kwargs_180265)
            
            # Getting the type of 'list' (line 797)
            list_180267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 29), 'list')
            # Applying the binary operator 'is' (line 797)
            result_is__180268 = python_operator(stypy.reporting.localization.Localization(__file__, 797, 13), 'is', type_call_result_180266, list_180267)
            
            
            
            # Call to len(...): (line 797)
            # Processing the call arguments (line 797)
            # Getting the type of 'domain' (line 797)
            domain_180270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 42), 'domain', False)
            # Processing the call keyword arguments (line 797)
            kwargs_180271 = {}
            # Getting the type of 'len' (line 797)
            len_180269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 38), 'len', False)
            # Calling len(args, kwargs) (line 797)
            len_call_result_180272 = invoke(stypy.reporting.localization.Localization(__file__, 797, 38), len_180269, *[domain_180270], **kwargs_180271)
            
            int_180273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 53), 'int')
            # Applying the binary operator '==' (line 797)
            result_eq_180274 = python_operator(stypy.reporting.localization.Localization(__file__, 797, 38), '==', len_call_result_180272, int_180273)
            
            # Applying the binary operator 'and' (line 797)
            result_and_keyword_180275 = python_operator(stypy.reporting.localization.Localization(__file__, 797, 13), 'and', result_is__180268, result_eq_180274)
            
            # Testing the type of an if condition (line 797)
            if_condition_180276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 797, 13), result_and_keyword_180275)
            # Assigning a type to the variable 'if_condition_180276' (line 797)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 13), 'if_condition_180276', if_condition_180276)
            # SSA begins for if statement (line 797)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 798):
            
            # Assigning a Attribute to a Name (line 798):
            # Getting the type of 'cls' (line 798)
            cls_180277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 21), 'cls')
            # Obtaining the member 'domain' of a type (line 798)
            domain_180278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 21), cls_180277, 'domain')
            # Assigning a type to the variable 'domain' (line 798)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 12), 'domain', domain_180278)
            # SSA join for if statement (line 797)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_180256 and more_types_in_union_180257):
                # SSA join for if statement (line 795)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 800)
        # Getting the type of 'window' (line 800)
        window_180279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 11), 'window')
        # Getting the type of 'None' (line 800)
        None_180280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 21), 'None')
        
        (may_be_180281, more_types_in_union_180282) = may_be_none(window_180279, None_180280)

        if may_be_180281:

            if more_types_in_union_180282:
                # Runtime conditional SSA (line 800)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 801):
            
            # Assigning a Attribute to a Name (line 801):
            # Getting the type of 'cls' (line 801)
            cls_180283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 21), 'cls')
            # Obtaining the member 'window' of a type (line 801)
            window_180284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 21), cls_180283, 'window')
            # Assigning a type to the variable 'window' (line 801)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 12), 'window', window_180284)

            if more_types_in_union_180282:
                # SSA join for if statement (line 800)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 803):
        
        # Assigning a Call to a Name (line 803):
        
        # Call to mapdomain(...): (line 803)
        # Processing the call arguments (line 803)
        # Getting the type of 'x' (line 803)
        x_180287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 28), 'x', False)
        # Getting the type of 'domain' (line 803)
        domain_180288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 31), 'domain', False)
        # Getting the type of 'window' (line 803)
        window_180289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 39), 'window', False)
        # Processing the call keyword arguments (line 803)
        kwargs_180290 = {}
        # Getting the type of 'pu' (line 803)
        pu_180285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 15), 'pu', False)
        # Obtaining the member 'mapdomain' of a type (line 803)
        mapdomain_180286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 15), pu_180285, 'mapdomain')
        # Calling mapdomain(args, kwargs) (line 803)
        mapdomain_call_result_180291 = invoke(stypy.reporting.localization.Localization(__file__, 803, 15), mapdomain_180286, *[x_180287, domain_180288, window_180289], **kwargs_180290)
        
        # Assigning a type to the variable 'xnew' (line 803)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'xnew', mapdomain_call_result_180291)
        
        # Assigning a Call to a Name (line 804):
        
        # Assigning a Call to a Name (line 804):
        
        # Call to _fit(...): (line 804)
        # Processing the call arguments (line 804)
        # Getting the type of 'xnew' (line 804)
        xnew_180294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 23), 'xnew', False)
        # Getting the type of 'y' (line 804)
        y_180295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 29), 'y', False)
        # Getting the type of 'deg' (line 804)
        deg_180296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 32), 'deg', False)
        # Processing the call keyword arguments (line 804)
        # Getting the type of 'w' (line 804)
        w_180297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 39), 'w', False)
        keyword_180298 = w_180297
        # Getting the type of 'rcond' (line 804)
        rcond_180299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 48), 'rcond', False)
        keyword_180300 = rcond_180299
        # Getting the type of 'full' (line 804)
        full_180301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 60), 'full', False)
        keyword_180302 = full_180301
        kwargs_180303 = {'rcond': keyword_180300, 'full': keyword_180302, 'w': keyword_180298}
        # Getting the type of 'cls' (line 804)
        cls_180292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 14), 'cls', False)
        # Obtaining the member '_fit' of a type (line 804)
        _fit_180293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 14), cls_180292, '_fit')
        # Calling _fit(args, kwargs) (line 804)
        _fit_call_result_180304 = invoke(stypy.reporting.localization.Localization(__file__, 804, 14), _fit_180293, *[xnew_180294, y_180295, deg_180296], **kwargs_180303)
        
        # Assigning a type to the variable 'res' (line 804)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 8), 'res', _fit_call_result_180304)
        
        # Getting the type of 'full' (line 805)
        full_180305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 11), 'full')
        # Testing the type of an if condition (line 805)
        if_condition_180306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 805, 8), full_180305)
        # Assigning a type to the variable 'if_condition_180306' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'if_condition_180306', if_condition_180306)
        # SSA begins for if statement (line 805)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a List (line 806):
        
        # Assigning a Subscript to a Name (line 806):
        
        # Obtaining the type of the subscript
        int_180307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 12), 'int')
        # Getting the type of 'res' (line 806)
        res_180308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 29), 'res')
        # Obtaining the member '__getitem__' of a type (line 806)
        getitem___180309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 12), res_180308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 806)
        subscript_call_result_180310 = invoke(stypy.reporting.localization.Localization(__file__, 806, 12), getitem___180309, int_180307)
        
        # Assigning a type to the variable 'list_var_assignment_179203' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 12), 'list_var_assignment_179203', subscript_call_result_180310)
        
        # Assigning a Subscript to a Name (line 806):
        
        # Obtaining the type of the subscript
        int_180311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 12), 'int')
        # Getting the type of 'res' (line 806)
        res_180312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 29), 'res')
        # Obtaining the member '__getitem__' of a type (line 806)
        getitem___180313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 12), res_180312, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 806)
        subscript_call_result_180314 = invoke(stypy.reporting.localization.Localization(__file__, 806, 12), getitem___180313, int_180311)
        
        # Assigning a type to the variable 'list_var_assignment_179204' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 12), 'list_var_assignment_179204', subscript_call_result_180314)
        
        # Assigning a Name to a Name (line 806):
        # Getting the type of 'list_var_assignment_179203' (line 806)
        list_var_assignment_179203_180315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 12), 'list_var_assignment_179203')
        # Assigning a type to the variable 'coef' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 13), 'coef', list_var_assignment_179203_180315)
        
        # Assigning a Name to a Name (line 806):
        # Getting the type of 'list_var_assignment_179204' (line 806)
        list_var_assignment_179204_180316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 12), 'list_var_assignment_179204')
        # Assigning a type to the variable 'status' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 19), 'status', list_var_assignment_179204_180316)
        
        # Obtaining an instance of the builtin type 'tuple' (line 807)
        tuple_180317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 807)
        # Adding element type (line 807)
        
        # Call to cls(...): (line 807)
        # Processing the call arguments (line 807)
        # Getting the type of 'coef' (line 807)
        coef_180319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 23), 'coef', False)
        # Processing the call keyword arguments (line 807)
        # Getting the type of 'domain' (line 807)
        domain_180320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 36), 'domain', False)
        keyword_180321 = domain_180320
        # Getting the type of 'window' (line 807)
        window_180322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 51), 'window', False)
        keyword_180323 = window_180322
        kwargs_180324 = {'domain': keyword_180321, 'window': keyword_180323}
        # Getting the type of 'cls' (line 807)
        cls_180318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 19), 'cls', False)
        # Calling cls(args, kwargs) (line 807)
        cls_call_result_180325 = invoke(stypy.reporting.localization.Localization(__file__, 807, 19), cls_180318, *[coef_180319], **kwargs_180324)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 19), tuple_180317, cls_call_result_180325)
        # Adding element type (line 807)
        # Getting the type of 'status' (line 807)
        status_180326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 60), 'status')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 19), tuple_180317, status_180326)
        
        # Assigning a type to the variable 'stypy_return_type' (line 807)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 12), 'stypy_return_type', tuple_180317)
        # SSA branch for the else part of an if statement (line 805)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 809):
        
        # Assigning a Name to a Name (line 809):
        # Getting the type of 'res' (line 809)
        res_180327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 19), 'res')
        # Assigning a type to the variable 'coef' (line 809)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 12), 'coef', res_180327)
        
        # Call to cls(...): (line 810)
        # Processing the call arguments (line 810)
        # Getting the type of 'coef' (line 810)
        coef_180329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 23), 'coef', False)
        # Processing the call keyword arguments (line 810)
        # Getting the type of 'domain' (line 810)
        domain_180330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 36), 'domain', False)
        keyword_180331 = domain_180330
        # Getting the type of 'window' (line 810)
        window_180332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 51), 'window', False)
        keyword_180333 = window_180332
        kwargs_180334 = {'domain': keyword_180331, 'window': keyword_180333}
        # Getting the type of 'cls' (line 810)
        cls_180328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 19), 'cls', False)
        # Calling cls(args, kwargs) (line 810)
        cls_call_result_180335 = invoke(stypy.reporting.localization.Localization(__file__, 810, 19), cls_180328, *[coef_180329], **kwargs_180334)
        
        # Assigning a type to the variable 'stypy_return_type' (line 810)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 12), 'stypy_return_type', cls_call_result_180335)
        # SSA join for if statement (line 805)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'fit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fit' in the type store
        # Getting the type of 'stypy_return_type' (line 724)
        stypy_return_type_180336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180336)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fit'
        return stypy_return_type_180336


    @norecursion
    def fromroots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'list' (line 813)
        list_180337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 813)
        
        # Getting the type of 'None' (line 813)
        None_180338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 48), 'None')
        defaults = [list_180337, None_180338]
        # Create a new context for function 'fromroots'
        module_type_store = module_type_store.open_function_context('fromroots', 812, 4, False)
        # Assigning a type to the variable 'self' (line 813)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.fromroots')
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_param_names_list', ['roots', 'domain', 'window'])
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.fromroots.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.fromroots', ['roots', 'domain', 'window'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fromroots', localization, ['roots', 'domain', 'window'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fromroots(...)' code ##################

        str_180339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, (-1)), 'str', 'Return series instance that has the specified roots.\n\n        Returns a series representing the product\n        ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is a\n        list of roots.\n\n        Parameters\n        ----------\n        roots : array_like\n            List of roots.\n        domain : {[], None, array_like}, optional\n            Domain for the resulting series. If None the domain is the\n            interval from the smallest root to the largest. If [] the\n            domain is the class domain. The default is [].\n        window : {None, array_like}, optional\n            Window for the returned series. If None the class window is\n            used. The default is None.\n\n        Returns\n        -------\n        new_series : series\n            Series with the specified roots.\n\n        ')
        
        # Assigning a Call to a List (line 838):
        
        # Assigning a Call to a Name:
        
        # Call to as_series(...): (line 838)
        # Processing the call arguments (line 838)
        
        # Obtaining an instance of the builtin type 'list' (line 838)
        list_180342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 838)
        # Adding element type (line 838)
        # Getting the type of 'roots' (line 838)
        roots_180343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 32), 'roots', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 31), list_180342, roots_180343)
        
        # Processing the call keyword arguments (line 838)
        # Getting the type of 'False' (line 838)
        False_180344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 45), 'False', False)
        keyword_180345 = False_180344
        kwargs_180346 = {'trim': keyword_180345}
        # Getting the type of 'pu' (line 838)
        pu_180340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 18), 'pu', False)
        # Obtaining the member 'as_series' of a type (line 838)
        as_series_180341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 18), pu_180340, 'as_series')
        # Calling as_series(args, kwargs) (line 838)
        as_series_call_result_180347 = invoke(stypy.reporting.localization.Localization(__file__, 838, 18), as_series_180341, *[list_180342], **kwargs_180346)
        
        # Assigning a type to the variable 'call_assignment_179205' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'call_assignment_179205', as_series_call_result_180347)
        
        # Assigning a Call to a Name (line 838):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180351 = {}
        # Getting the type of 'call_assignment_179205' (line 838)
        call_assignment_179205_180348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'call_assignment_179205', False)
        # Obtaining the member '__getitem__' of a type (line 838)
        getitem___180349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 8), call_assignment_179205_180348, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180352 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180349, *[int_180350], **kwargs_180351)
        
        # Assigning a type to the variable 'call_assignment_179206' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'call_assignment_179206', getitem___call_result_180352)
        
        # Assigning a Name to a Name (line 838):
        # Getting the type of 'call_assignment_179206' (line 838)
        call_assignment_179206_180353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'call_assignment_179206')
        # Assigning a type to the variable 'roots' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 9), 'roots', call_assignment_179206_180353)
        
        # Type idiom detected: calculating its left and rigth part (line 839)
        # Getting the type of 'domain' (line 839)
        domain_180354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 11), 'domain')
        # Getting the type of 'None' (line 839)
        None_180355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 21), 'None')
        
        (may_be_180356, more_types_in_union_180357) = may_be_none(domain_180354, None_180355)

        if may_be_180356:

            if more_types_in_union_180357:
                # Runtime conditional SSA (line 839)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 840):
            
            # Assigning a Call to a Name (line 840):
            
            # Call to getdomain(...): (line 840)
            # Processing the call arguments (line 840)
            # Getting the type of 'roots' (line 840)
            roots_180360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 34), 'roots', False)
            # Processing the call keyword arguments (line 840)
            kwargs_180361 = {}
            # Getting the type of 'pu' (line 840)
            pu_180358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 21), 'pu', False)
            # Obtaining the member 'getdomain' of a type (line 840)
            getdomain_180359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 21), pu_180358, 'getdomain')
            # Calling getdomain(args, kwargs) (line 840)
            getdomain_call_result_180362 = invoke(stypy.reporting.localization.Localization(__file__, 840, 21), getdomain_180359, *[roots_180360], **kwargs_180361)
            
            # Assigning a type to the variable 'domain' (line 840)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 12), 'domain', getdomain_call_result_180362)

            if more_types_in_union_180357:
                # Runtime conditional SSA for else branch (line 839)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_180356) or more_types_in_union_180357):
            
            
            # Evaluating a boolean operation
            
            
            # Call to type(...): (line 841)
            # Processing the call arguments (line 841)
            # Getting the type of 'domain' (line 841)
            domain_180364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 18), 'domain', False)
            # Processing the call keyword arguments (line 841)
            kwargs_180365 = {}
            # Getting the type of 'type' (line 841)
            type_180363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 13), 'type', False)
            # Calling type(args, kwargs) (line 841)
            type_call_result_180366 = invoke(stypy.reporting.localization.Localization(__file__, 841, 13), type_180363, *[domain_180364], **kwargs_180365)
            
            # Getting the type of 'list' (line 841)
            list_180367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 29), 'list')
            # Applying the binary operator 'is' (line 841)
            result_is__180368 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 13), 'is', type_call_result_180366, list_180367)
            
            
            
            # Call to len(...): (line 841)
            # Processing the call arguments (line 841)
            # Getting the type of 'domain' (line 841)
            domain_180370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 42), 'domain', False)
            # Processing the call keyword arguments (line 841)
            kwargs_180371 = {}
            # Getting the type of 'len' (line 841)
            len_180369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 38), 'len', False)
            # Calling len(args, kwargs) (line 841)
            len_call_result_180372 = invoke(stypy.reporting.localization.Localization(__file__, 841, 38), len_180369, *[domain_180370], **kwargs_180371)
            
            int_180373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 53), 'int')
            # Applying the binary operator '==' (line 841)
            result_eq_180374 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 38), '==', len_call_result_180372, int_180373)
            
            # Applying the binary operator 'and' (line 841)
            result_and_keyword_180375 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 13), 'and', result_is__180368, result_eq_180374)
            
            # Testing the type of an if condition (line 841)
            if_condition_180376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 841, 13), result_and_keyword_180375)
            # Assigning a type to the variable 'if_condition_180376' (line 841)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 13), 'if_condition_180376', if_condition_180376)
            # SSA begins for if statement (line 841)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 842):
            
            # Assigning a Attribute to a Name (line 842):
            # Getting the type of 'cls' (line 842)
            cls_180377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 21), 'cls')
            # Obtaining the member 'domain' of a type (line 842)
            domain_180378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 21), cls_180377, 'domain')
            # Assigning a type to the variable 'domain' (line 842)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'domain', domain_180378)
            # SSA join for if statement (line 841)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_180356 and more_types_in_union_180357):
                # SSA join for if statement (line 839)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 844)
        # Getting the type of 'window' (line 844)
        window_180379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 11), 'window')
        # Getting the type of 'None' (line 844)
        None_180380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 21), 'None')
        
        (may_be_180381, more_types_in_union_180382) = may_be_none(window_180379, None_180380)

        if may_be_180381:

            if more_types_in_union_180382:
                # Runtime conditional SSA (line 844)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 845):
            
            # Assigning a Attribute to a Name (line 845):
            # Getting the type of 'cls' (line 845)
            cls_180383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 21), 'cls')
            # Obtaining the member 'window' of a type (line 845)
            window_180384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 21), cls_180383, 'window')
            # Assigning a type to the variable 'window' (line 845)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'window', window_180384)

            if more_types_in_union_180382:
                # SSA join for if statement (line 844)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 847):
        
        # Assigning a Call to a Name (line 847):
        
        # Call to len(...): (line 847)
        # Processing the call arguments (line 847)
        # Getting the type of 'roots' (line 847)
        roots_180386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 18), 'roots', False)
        # Processing the call keyword arguments (line 847)
        kwargs_180387 = {}
        # Getting the type of 'len' (line 847)
        len_180385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 14), 'len', False)
        # Calling len(args, kwargs) (line 847)
        len_call_result_180388 = invoke(stypy.reporting.localization.Localization(__file__, 847, 14), len_180385, *[roots_180386], **kwargs_180387)
        
        # Assigning a type to the variable 'deg' (line 847)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 8), 'deg', len_call_result_180388)
        
        # Assigning a Call to a Tuple (line 848):
        
        # Assigning a Call to a Name:
        
        # Call to mapparms(...): (line 848)
        # Processing the call arguments (line 848)
        # Getting the type of 'domain' (line 848)
        domain_180391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 31), 'domain', False)
        # Getting the type of 'window' (line 848)
        window_180392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 39), 'window', False)
        # Processing the call keyword arguments (line 848)
        kwargs_180393 = {}
        # Getting the type of 'pu' (line 848)
        pu_180389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 19), 'pu', False)
        # Obtaining the member 'mapparms' of a type (line 848)
        mapparms_180390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 19), pu_180389, 'mapparms')
        # Calling mapparms(args, kwargs) (line 848)
        mapparms_call_result_180394 = invoke(stypy.reporting.localization.Localization(__file__, 848, 19), mapparms_180390, *[domain_180391, window_180392], **kwargs_180393)
        
        # Assigning a type to the variable 'call_assignment_179207' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'call_assignment_179207', mapparms_call_result_180394)
        
        # Assigning a Call to a Name (line 848):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180398 = {}
        # Getting the type of 'call_assignment_179207' (line 848)
        call_assignment_179207_180395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'call_assignment_179207', False)
        # Obtaining the member '__getitem__' of a type (line 848)
        getitem___180396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 8), call_assignment_179207_180395, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180399 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180396, *[int_180397], **kwargs_180398)
        
        # Assigning a type to the variable 'call_assignment_179208' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'call_assignment_179208', getitem___call_result_180399)
        
        # Assigning a Name to a Name (line 848):
        # Getting the type of 'call_assignment_179208' (line 848)
        call_assignment_179208_180400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'call_assignment_179208')
        # Assigning a type to the variable 'off' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'off', call_assignment_179208_180400)
        
        # Assigning a Call to a Name (line 848):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180404 = {}
        # Getting the type of 'call_assignment_179207' (line 848)
        call_assignment_179207_180401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'call_assignment_179207', False)
        # Obtaining the member '__getitem__' of a type (line 848)
        getitem___180402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 8), call_assignment_179207_180401, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180405 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180402, *[int_180403], **kwargs_180404)
        
        # Assigning a type to the variable 'call_assignment_179209' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'call_assignment_179209', getitem___call_result_180405)
        
        # Assigning a Name to a Name (line 848):
        # Getting the type of 'call_assignment_179209' (line 848)
        call_assignment_179209_180406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'call_assignment_179209')
        # Assigning a type to the variable 'scl' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 13), 'scl', call_assignment_179209_180406)
        
        # Assigning a BinOp to a Name (line 849):
        
        # Assigning a BinOp to a Name (line 849):
        # Getting the type of 'off' (line 849)
        off_180407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 15), 'off')
        # Getting the type of 'scl' (line 849)
        scl_180408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 21), 'scl')
        # Getting the type of 'roots' (line 849)
        roots_180409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 25), 'roots')
        # Applying the binary operator '*' (line 849)
        result_mul_180410 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 21), '*', scl_180408, roots_180409)
        
        # Applying the binary operator '+' (line 849)
        result_add_180411 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 15), '+', off_180407, result_mul_180410)
        
        # Assigning a type to the variable 'rnew' (line 849)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'rnew', result_add_180411)
        
        # Assigning a BinOp to a Name (line 850):
        
        # Assigning a BinOp to a Name (line 850):
        
        # Call to _fromroots(...): (line 850)
        # Processing the call arguments (line 850)
        # Getting the type of 'rnew' (line 850)
        rnew_180414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 30), 'rnew', False)
        # Processing the call keyword arguments (line 850)
        kwargs_180415 = {}
        # Getting the type of 'cls' (line 850)
        cls_180412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 15), 'cls', False)
        # Obtaining the member '_fromroots' of a type (line 850)
        _fromroots_180413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 15), cls_180412, '_fromroots')
        # Calling _fromroots(args, kwargs) (line 850)
        _fromroots_call_result_180416 = invoke(stypy.reporting.localization.Localization(__file__, 850, 15), _fromroots_180413, *[rnew_180414], **kwargs_180415)
        
        # Getting the type of 'scl' (line 850)
        scl_180417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 38), 'scl')
        # Getting the type of 'deg' (line 850)
        deg_180418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 43), 'deg')
        # Applying the binary operator '**' (line 850)
        result_pow_180419 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 38), '**', scl_180417, deg_180418)
        
        # Applying the binary operator 'div' (line 850)
        result_div_180420 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 15), 'div', _fromroots_call_result_180416, result_pow_180419)
        
        # Assigning a type to the variable 'coef' (line 850)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 8), 'coef', result_div_180420)
        
        # Call to cls(...): (line 851)
        # Processing the call arguments (line 851)
        # Getting the type of 'coef' (line 851)
        coef_180422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 19), 'coef', False)
        # Processing the call keyword arguments (line 851)
        # Getting the type of 'domain' (line 851)
        domain_180423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 32), 'domain', False)
        keyword_180424 = domain_180423
        # Getting the type of 'window' (line 851)
        window_180425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 47), 'window', False)
        keyword_180426 = window_180425
        kwargs_180427 = {'domain': keyword_180424, 'window': keyword_180426}
        # Getting the type of 'cls' (line 851)
        cls_180421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 851)
        cls_call_result_180428 = invoke(stypy.reporting.localization.Localization(__file__, 851, 15), cls_180421, *[coef_180422], **kwargs_180427)
        
        # Assigning a type to the variable 'stypy_return_type' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'stypy_return_type', cls_call_result_180428)
        
        # ################# End of 'fromroots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fromroots' in the type store
        # Getting the type of 'stypy_return_type' (line 812)
        stypy_return_type_180429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fromroots'
        return stypy_return_type_180429


    @norecursion
    def identity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 854)
        None_180430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 29), 'None')
        # Getting the type of 'None' (line 854)
        None_180431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 42), 'None')
        defaults = [None_180430, None_180431]
        # Create a new context for function 'identity'
        module_type_store = module_type_store.open_function_context('identity', 853, 4, False)
        # Assigning a type to the variable 'self' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.identity.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.identity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.identity.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.identity.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.identity')
        ABCPolyBase.identity.__dict__.__setitem__('stypy_param_names_list', ['domain', 'window'])
        ABCPolyBase.identity.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.identity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.identity.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.identity.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.identity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.identity.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.identity', ['domain', 'window'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'identity', localization, ['domain', 'window'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'identity(...)' code ##################

        str_180432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, (-1)), 'str', 'Identity function.\n\n        If ``p`` is the returned series, then ``p(x) == x`` for all\n        values of x.\n\n        Parameters\n        ----------\n        domain : {None, array_like}, optional\n            If given, the array must be of the form ``[beg, end]``, where\n            ``beg`` and ``end`` are the endpoints of the domain. If None is\n            given then the class domain is used. The default is None.\n        window : {None, array_like}, optional\n            If given, the resulting array must be if the form\n            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of\n            the window. If None is given then the class window is used. The\n            default is None.\n\n        Returns\n        -------\n        new_series : series\n             Series of representing the identity.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 878)
        # Getting the type of 'domain' (line 878)
        domain_180433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 11), 'domain')
        # Getting the type of 'None' (line 878)
        None_180434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 21), 'None')
        
        (may_be_180435, more_types_in_union_180436) = may_be_none(domain_180433, None_180434)

        if may_be_180435:

            if more_types_in_union_180436:
                # Runtime conditional SSA (line 878)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 879):
            
            # Assigning a Attribute to a Name (line 879):
            # Getting the type of 'cls' (line 879)
            cls_180437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 21), 'cls')
            # Obtaining the member 'domain' of a type (line 879)
            domain_180438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 21), cls_180437, 'domain')
            # Assigning a type to the variable 'domain' (line 879)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 12), 'domain', domain_180438)

            if more_types_in_union_180436:
                # SSA join for if statement (line 878)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 880)
        # Getting the type of 'window' (line 880)
        window_180439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 11), 'window')
        # Getting the type of 'None' (line 880)
        None_180440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 21), 'None')
        
        (may_be_180441, more_types_in_union_180442) = may_be_none(window_180439, None_180440)

        if may_be_180441:

            if more_types_in_union_180442:
                # Runtime conditional SSA (line 880)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 881):
            
            # Assigning a Attribute to a Name (line 881):
            # Getting the type of 'cls' (line 881)
            cls_180443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 21), 'cls')
            # Obtaining the member 'window' of a type (line 881)
            window_180444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 21), cls_180443, 'window')
            # Assigning a type to the variable 'window' (line 881)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 12), 'window', window_180444)

            if more_types_in_union_180442:
                # SSA join for if statement (line 880)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 882):
        
        # Assigning a Call to a Name:
        
        # Call to mapparms(...): (line 882)
        # Processing the call arguments (line 882)
        # Getting the type of 'window' (line 882)
        window_180447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 31), 'window', False)
        # Getting the type of 'domain' (line 882)
        domain_180448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 39), 'domain', False)
        # Processing the call keyword arguments (line 882)
        kwargs_180449 = {}
        # Getting the type of 'pu' (line 882)
        pu_180445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 19), 'pu', False)
        # Obtaining the member 'mapparms' of a type (line 882)
        mapparms_180446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 19), pu_180445, 'mapparms')
        # Calling mapparms(args, kwargs) (line 882)
        mapparms_call_result_180450 = invoke(stypy.reporting.localization.Localization(__file__, 882, 19), mapparms_180446, *[window_180447, domain_180448], **kwargs_180449)
        
        # Assigning a type to the variable 'call_assignment_179210' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'call_assignment_179210', mapparms_call_result_180450)
        
        # Assigning a Call to a Name (line 882):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180454 = {}
        # Getting the type of 'call_assignment_179210' (line 882)
        call_assignment_179210_180451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'call_assignment_179210', False)
        # Obtaining the member '__getitem__' of a type (line 882)
        getitem___180452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 8), call_assignment_179210_180451, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180455 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180452, *[int_180453], **kwargs_180454)
        
        # Assigning a type to the variable 'call_assignment_179211' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'call_assignment_179211', getitem___call_result_180455)
        
        # Assigning a Name to a Name (line 882):
        # Getting the type of 'call_assignment_179211' (line 882)
        call_assignment_179211_180456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'call_assignment_179211')
        # Assigning a type to the variable 'off' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'off', call_assignment_179211_180456)
        
        # Assigning a Call to a Name (line 882):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180460 = {}
        # Getting the type of 'call_assignment_179210' (line 882)
        call_assignment_179210_180457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'call_assignment_179210', False)
        # Obtaining the member '__getitem__' of a type (line 882)
        getitem___180458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 8), call_assignment_179210_180457, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_180461 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___180458, *[int_180459], **kwargs_180460)
        
        # Assigning a type to the variable 'call_assignment_179212' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'call_assignment_179212', getitem___call_result_180461)
        
        # Assigning a Name to a Name (line 882):
        # Getting the type of 'call_assignment_179212' (line 882)
        call_assignment_179212_180462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'call_assignment_179212')
        # Assigning a type to the variable 'scl' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 13), 'scl', call_assignment_179212_180462)
        
        # Assigning a Call to a Name (line 883):
        
        # Assigning a Call to a Name (line 883):
        
        # Call to _line(...): (line 883)
        # Processing the call arguments (line 883)
        # Getting the type of 'off' (line 883)
        off_180465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 25), 'off', False)
        # Getting the type of 'scl' (line 883)
        scl_180466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 30), 'scl', False)
        # Processing the call keyword arguments (line 883)
        kwargs_180467 = {}
        # Getting the type of 'cls' (line 883)
        cls_180463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 15), 'cls', False)
        # Obtaining the member '_line' of a type (line 883)
        _line_180464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 15), cls_180463, '_line')
        # Calling _line(args, kwargs) (line 883)
        _line_call_result_180468 = invoke(stypy.reporting.localization.Localization(__file__, 883, 15), _line_180464, *[off_180465, scl_180466], **kwargs_180467)
        
        # Assigning a type to the variable 'coef' (line 883)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 8), 'coef', _line_call_result_180468)
        
        # Call to cls(...): (line 884)
        # Processing the call arguments (line 884)
        # Getting the type of 'coef' (line 884)
        coef_180470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 19), 'coef', False)
        # Getting the type of 'domain' (line 884)
        domain_180471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 25), 'domain', False)
        # Getting the type of 'window' (line 884)
        window_180472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 33), 'window', False)
        # Processing the call keyword arguments (line 884)
        kwargs_180473 = {}
        # Getting the type of 'cls' (line 884)
        cls_180469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 884)
        cls_call_result_180474 = invoke(stypy.reporting.localization.Localization(__file__, 884, 15), cls_180469, *[coef_180470, domain_180471, window_180472], **kwargs_180473)
        
        # Assigning a type to the variable 'stypy_return_type' (line 884)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 8), 'stypy_return_type', cls_call_result_180474)
        
        # ################# End of 'identity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'identity' in the type store
        # Getting the type of 'stypy_return_type' (line 853)
        stypy_return_type_180475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180475)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'identity'
        return stypy_return_type_180475


    @norecursion
    def basis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 887)
        None_180476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 31), 'None')
        # Getting the type of 'None' (line 887)
        None_180477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 44), 'None')
        defaults = [None_180476, None_180477]
        # Create a new context for function 'basis'
        module_type_store = module_type_store.open_function_context('basis', 886, 4, False)
        # Assigning a type to the variable 'self' (line 887)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.basis.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.basis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.basis.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.basis.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.basis')
        ABCPolyBase.basis.__dict__.__setitem__('stypy_param_names_list', ['deg', 'domain', 'window'])
        ABCPolyBase.basis.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.basis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.basis.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.basis.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.basis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.basis.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.basis', ['deg', 'domain', 'window'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'basis', localization, ['deg', 'domain', 'window'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'basis(...)' code ##################

        str_180478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, (-1)), 'str', 'Series basis polynomial of degree `deg`.\n\n        Returns the series representing the basis polynomial of degree `deg`.\n\n        .. versionadded:: 1.7.0\n\n        Parameters\n        ----------\n        deg : int\n            Degree of the basis polynomial for the series. Must be >= 0.\n        domain : {None, array_like}, optional\n            If given, the array must be of the form ``[beg, end]``, where\n            ``beg`` and ``end`` are the endpoints of the domain. If None is\n            given then the class domain is used. The default is None.\n        window : {None, array_like}, optional\n            If given, the resulting array must be if the form\n            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of\n            the window. If None is given then the class window is used. The\n            default is None.\n\n        Returns\n        -------\n        new_series : series\n            A series with the coefficient of the `deg` term set to one and\n            all others zero.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 915)
        # Getting the type of 'domain' (line 915)
        domain_180479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 11), 'domain')
        # Getting the type of 'None' (line 915)
        None_180480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 21), 'None')
        
        (may_be_180481, more_types_in_union_180482) = may_be_none(domain_180479, None_180480)

        if may_be_180481:

            if more_types_in_union_180482:
                # Runtime conditional SSA (line 915)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 916):
            
            # Assigning a Attribute to a Name (line 916):
            # Getting the type of 'cls' (line 916)
            cls_180483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 21), 'cls')
            # Obtaining the member 'domain' of a type (line 916)
            domain_180484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 21), cls_180483, 'domain')
            # Assigning a type to the variable 'domain' (line 916)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 12), 'domain', domain_180484)

            if more_types_in_union_180482:
                # SSA join for if statement (line 915)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 917)
        # Getting the type of 'window' (line 917)
        window_180485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 11), 'window')
        # Getting the type of 'None' (line 917)
        None_180486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 21), 'None')
        
        (may_be_180487, more_types_in_union_180488) = may_be_none(window_180485, None_180486)

        if may_be_180487:

            if more_types_in_union_180488:
                # Runtime conditional SSA (line 917)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 918):
            
            # Assigning a Attribute to a Name (line 918):
            # Getting the type of 'cls' (line 918)
            cls_180489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 21), 'cls')
            # Obtaining the member 'window' of a type (line 918)
            window_180490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 21), cls_180489, 'window')
            # Assigning a type to the variable 'window' (line 918)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 12), 'window', window_180490)

            if more_types_in_union_180488:
                # SSA join for if statement (line 917)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 919):
        
        # Assigning a Call to a Name (line 919):
        
        # Call to int(...): (line 919)
        # Processing the call arguments (line 919)
        # Getting the type of 'deg' (line 919)
        deg_180492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 19), 'deg', False)
        # Processing the call keyword arguments (line 919)
        kwargs_180493 = {}
        # Getting the type of 'int' (line 919)
        int_180491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 15), 'int', False)
        # Calling int(args, kwargs) (line 919)
        int_call_result_180494 = invoke(stypy.reporting.localization.Localization(__file__, 919, 15), int_180491, *[deg_180492], **kwargs_180493)
        
        # Assigning a type to the variable 'ideg' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 8), 'ideg', int_call_result_180494)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ideg' (line 921)
        ideg_180495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 11), 'ideg')
        # Getting the type of 'deg' (line 921)
        deg_180496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 19), 'deg')
        # Applying the binary operator '!=' (line 921)
        result_ne_180497 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 11), '!=', ideg_180495, deg_180496)
        
        
        # Getting the type of 'ideg' (line 921)
        ideg_180498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 26), 'ideg')
        int_180499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 33), 'int')
        # Applying the binary operator '<' (line 921)
        result_lt_180500 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 26), '<', ideg_180498, int_180499)
        
        # Applying the binary operator 'or' (line 921)
        result_or_keyword_180501 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 11), 'or', result_ne_180497, result_lt_180500)
        
        # Testing the type of an if condition (line 921)
        if_condition_180502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 921, 8), result_or_keyword_180501)
        # Assigning a type to the variable 'if_condition_180502' (line 921)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 8), 'if_condition_180502', if_condition_180502)
        # SSA begins for if statement (line 921)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 922)
        # Processing the call arguments (line 922)
        str_180504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 29), 'str', 'deg must be non-negative integer')
        # Processing the call keyword arguments (line 922)
        kwargs_180505 = {}
        # Getting the type of 'ValueError' (line 922)
        ValueError_180503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 922)
        ValueError_call_result_180506 = invoke(stypy.reporting.localization.Localization(__file__, 922, 18), ValueError_180503, *[str_180504], **kwargs_180505)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 922, 12), ValueError_call_result_180506, 'raise parameter', BaseException)
        # SSA join for if statement (line 921)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cls(...): (line 923)
        # Processing the call arguments (line 923)
        
        # Obtaining an instance of the builtin type 'list' (line 923)
        list_180508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 923)
        # Adding element type (line 923)
        int_180509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 923, 19), list_180508, int_180509)
        
        # Getting the type of 'ideg' (line 923)
        ideg_180510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 23), 'ideg', False)
        # Applying the binary operator '*' (line 923)
        result_mul_180511 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 19), '*', list_180508, ideg_180510)
        
        
        # Obtaining an instance of the builtin type 'list' (line 923)
        list_180512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 923)
        # Adding element type (line 923)
        int_180513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 923, 30), list_180512, int_180513)
        
        # Applying the binary operator '+' (line 923)
        result_add_180514 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 19), '+', result_mul_180511, list_180512)
        
        # Getting the type of 'domain' (line 923)
        domain_180515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 35), 'domain', False)
        # Getting the type of 'window' (line 923)
        window_180516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 43), 'window', False)
        # Processing the call keyword arguments (line 923)
        kwargs_180517 = {}
        # Getting the type of 'cls' (line 923)
        cls_180507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 923)
        cls_call_result_180518 = invoke(stypy.reporting.localization.Localization(__file__, 923, 15), cls_180507, *[result_add_180514, domain_180515, window_180516], **kwargs_180517)
        
        # Assigning a type to the variable 'stypy_return_type' (line 923)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'stypy_return_type', cls_call_result_180518)
        
        # ################# End of 'basis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'basis' in the type store
        # Getting the type of 'stypy_return_type' (line 886)
        stypy_return_type_180519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180519)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'basis'
        return stypy_return_type_180519


    @norecursion
    def cast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 926)
        None_180520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 33), 'None')
        # Getting the type of 'None' (line 926)
        None_180521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 46), 'None')
        defaults = [None_180520, None_180521]
        # Create a new context for function 'cast'
        module_type_store = module_type_store.open_function_context('cast', 925, 4, False)
        # Assigning a type to the variable 'self' (line 926)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ABCPolyBase.cast.__dict__.__setitem__('stypy_localization', localization)
        ABCPolyBase.cast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ABCPolyBase.cast.__dict__.__setitem__('stypy_type_store', module_type_store)
        ABCPolyBase.cast.__dict__.__setitem__('stypy_function_name', 'ABCPolyBase.cast')
        ABCPolyBase.cast.__dict__.__setitem__('stypy_param_names_list', ['series', 'domain', 'window'])
        ABCPolyBase.cast.__dict__.__setitem__('stypy_varargs_param_name', None)
        ABCPolyBase.cast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ABCPolyBase.cast.__dict__.__setitem__('stypy_call_defaults', defaults)
        ABCPolyBase.cast.__dict__.__setitem__('stypy_call_varargs', varargs)
        ABCPolyBase.cast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ABCPolyBase.cast.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ABCPolyBase.cast', ['series', 'domain', 'window'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cast', localization, ['series', 'domain', 'window'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cast(...)' code ##################

        str_180522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, (-1)), 'str', 'Convert series to series of this class.\n\n        The `series` is expected to be an instance of some polynomial\n        series of one of the types supported by by the numpy.polynomial\n        module, but could be some other class that supports the convert\n        method.\n\n        .. versionadded:: 1.7.0\n\n        Parameters\n        ----------\n        series : series\n            The series instance to be converted.\n        domain : {None, array_like}, optional\n            If given, the array must be of the form ``[beg, end]``, where\n            ``beg`` and ``end`` are the endpoints of the domain. If None is\n            given then the class domain is used. The default is None.\n        window : {None, array_like}, optional\n            If given, the resulting array must be if the form\n            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of\n            the window. If None is given then the class window is used. The\n            default is None.\n\n        Returns\n        -------\n        new_series : series\n            A series of the same kind as the calling class and equal to\n            `series` when evaluated.\n\n        See Also\n        --------\n        convert : similar instance method\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 961)
        # Getting the type of 'domain' (line 961)
        domain_180523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 11), 'domain')
        # Getting the type of 'None' (line 961)
        None_180524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 21), 'None')
        
        (may_be_180525, more_types_in_union_180526) = may_be_none(domain_180523, None_180524)

        if may_be_180525:

            if more_types_in_union_180526:
                # Runtime conditional SSA (line 961)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 962):
            
            # Assigning a Attribute to a Name (line 962):
            # Getting the type of 'cls' (line 962)
            cls_180527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 21), 'cls')
            # Obtaining the member 'domain' of a type (line 962)
            domain_180528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 21), cls_180527, 'domain')
            # Assigning a type to the variable 'domain' (line 962)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 12), 'domain', domain_180528)

            if more_types_in_union_180526:
                # SSA join for if statement (line 961)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 963)
        # Getting the type of 'window' (line 963)
        window_180529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 11), 'window')
        # Getting the type of 'None' (line 963)
        None_180530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 21), 'None')
        
        (may_be_180531, more_types_in_union_180532) = may_be_none(window_180529, None_180530)

        if may_be_180531:

            if more_types_in_union_180532:
                # Runtime conditional SSA (line 963)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 964):
            
            # Assigning a Attribute to a Name (line 964):
            # Getting the type of 'cls' (line 964)
            cls_180533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 21), 'cls')
            # Obtaining the member 'window' of a type (line 964)
            window_180534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 21), cls_180533, 'window')
            # Assigning a type to the variable 'window' (line 964)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 12), 'window', window_180534)

            if more_types_in_union_180532:
                # SSA join for if statement (line 963)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to convert(...): (line 965)
        # Processing the call arguments (line 965)
        # Getting the type of 'domain' (line 965)
        domain_180537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 30), 'domain', False)
        # Getting the type of 'cls' (line 965)
        cls_180538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 38), 'cls', False)
        # Getting the type of 'window' (line 965)
        window_180539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 43), 'window', False)
        # Processing the call keyword arguments (line 965)
        kwargs_180540 = {}
        # Getting the type of 'series' (line 965)
        series_180535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 15), 'series', False)
        # Obtaining the member 'convert' of a type (line 965)
        convert_180536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 15), series_180535, 'convert')
        # Calling convert(args, kwargs) (line 965)
        convert_call_result_180541 = invoke(stypy.reporting.localization.Localization(__file__, 965, 15), convert_180536, *[domain_180537, cls_180538, window_180539], **kwargs_180540)
        
        # Assigning a type to the variable 'stypy_return_type' (line 965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'stypy_return_type', convert_call_result_180541)
        
        # ################# End of 'cast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cast' in the type store
        # Getting the type of 'stypy_return_type' (line 925)
        stypy_return_type_180542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180542)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cast'
        return stypy_return_type_180542


# Assigning a type to the variable 'ABCPolyBase' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'ABCPolyBase', ABCPolyBase)

# Assigning a Name to a Name (line 62):
# Getting the type of 'ABCMeta' (line 62)
ABCMeta_180543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'ABCMeta')
# Getting the type of 'ABCPolyBase'
ABCPolyBase_180544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ABCPolyBase')
# Setting the type of the member '__metaclass__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ABCPolyBase_180544, '__metaclass__', ABCMeta_180543)

# Assigning a Name to a Name (line 65):
# Getting the type of 'None' (line 65)
None_180545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'None')
# Getting the type of 'ABCPolyBase'
ABCPolyBase_180546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ABCPolyBase')
# Setting the type of the member '__hash__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ABCPolyBase_180546, '__hash__', None_180545)

# Assigning a Num to a Name (line 68):
int_180547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'int')
# Getting the type of 'ABCPolyBase'
ABCPolyBase_180548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ABCPolyBase')
# Setting the type of the member '__array_priority__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ABCPolyBase_180548, '__array_priority__', int_180547)

# Assigning a Num to a Name (line 71):
int_180549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'int')
# Getting the type of 'ABCPolyBase'
ABCPolyBase_180550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ABCPolyBase')
# Setting the type of the member 'maxpower' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ABCPolyBase_180550, 'maxpower', int_180549)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
