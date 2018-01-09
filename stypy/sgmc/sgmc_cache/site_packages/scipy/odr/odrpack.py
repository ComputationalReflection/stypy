
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Python wrappers for Orthogonal Distance Regression (ODRPACK).
3: 
4: Notes
5: =====
6: 
7: * Array formats -- FORTRAN stores its arrays in memory column first, i.e. an
8:   array element A(i, j, k) will be next to A(i+1, j, k). In C and, consequently,
9:   NumPy, arrays are stored row first: A[i, j, k] is next to A[i, j, k+1]. For
10:   efficiency and convenience, the input and output arrays of the fitting
11:   function (and its Jacobians) are passed to FORTRAN without transposition.
12:   Therefore, where the ODRPACK documentation says that the X array is of shape
13:   (N, M), it will be passed to the Python function as an array of shape (M, N).
14:   If M==1, the one-dimensional case, then nothing matters; if M>1, then your
15:   Python functions will be dealing with arrays that are indexed in reverse of
16:   the ODRPACK documentation. No real biggie, but watch out for your indexing of
17:   the Jacobians: the i,j'th elements (@f_i/@x_j) evaluated at the n'th
18:   observation will be returned as jacd[j, i, n]. Except for the Jacobians, it
19:   really is easier to deal with x[0] and x[1] than x[:,0] and x[:,1]. Of course,
20:   you can always use the transpose() function from scipy explicitly.
21: 
22: * Examples -- See the accompanying file test/test.py for examples of how to set
23:   up fits of your own. Some are taken from the User's Guide; some are from
24:   other sources.
25: 
26: * Models -- Some common models are instantiated in the accompanying module
27:   models.py . Contributions are welcome.
28: 
29: Credits
30: =======
31: 
32: * Thanks to Arnold Moene and Gerard Vermeulen for fixing some killer bugs.
33: 
34: Robert Kern
35: robert.kern@gmail.com
36: 
37: '''
38: 
39: from __future__ import division, print_function, absolute_import
40: 
41: import numpy
42: from warnings import warn
43: from scipy.odr import __odrpack
44: 
45: __all__ = ['odr', 'OdrWarning', 'OdrError', 'OdrStop',
46:            'Data', 'RealData', 'Model', 'Output', 'ODR',
47:            'odr_error', 'odr_stop']
48: 
49: odr = __odrpack.odr
50: 
51: 
52: class OdrWarning(UserWarning):
53:     '''
54:     Warning indicating that the data passed into
55:     ODR will cause problems when passed into 'odr'
56:     that the user should be aware of.
57:     '''
58:     pass
59: 
60: 
61: class OdrError(Exception):
62:     '''
63:     Exception indicating an error in fitting.
64: 
65:     This is raised by `scipy.odr` if an error occurs during fitting.
66:     '''
67:     pass
68: 
69: 
70: class OdrStop(Exception):
71:     '''
72:     Exception stopping fitting.
73: 
74:     You can raise this exception in your objective function to tell
75:     `scipy.odr` to stop fitting.
76:     '''
77:     pass
78: 
79: # Backwards compatibility
80: odr_error = OdrError
81: odr_stop = OdrStop
82: 
83: __odrpack._set_exceptions(OdrError, OdrStop)
84: 
85: 
86: def _conv(obj, dtype=None):
87:     ''' Convert an object to the preferred form for input to the odr routine.
88:     '''
89: 
90:     if obj is None:
91:         return obj
92:     else:
93:         if dtype is None:
94:             obj = numpy.asarray(obj)
95:         else:
96:             obj = numpy.asarray(obj, dtype)
97:         if obj.shape == ():
98:             # Scalar.
99:             return obj.dtype.type(obj)
100:         else:
101:             return obj
102: 
103: 
104: def _report_error(info):
105:     ''' Interprets the return code of the odr routine.
106: 
107:     Parameters
108:     ----------
109:     info : int
110:         The return code of the odr routine.
111: 
112:     Returns
113:     -------
114:     problems : list(str)
115:         A list of messages about why the odr() routine stopped.
116:     '''
117: 
118:     stopreason = ('Blank',
119:                   'Sum of squares convergence',
120:                   'Parameter convergence',
121:                   'Both sum of squares and parameter convergence',
122:                   'Iteration limit reached')[info % 5]
123: 
124:     if info >= 5:
125:         # questionable results or fatal error
126: 
127:         I = (info//10000 % 10,
128:              info//1000 % 10,
129:              info//100 % 10,
130:              info//10 % 10,
131:              info % 10)
132:         problems = []
133: 
134:         if I[0] == 0:
135:             if I[1] != 0:
136:                 problems.append('Derivatives possibly not correct')
137:             if I[2] != 0:
138:                 problems.append('Error occurred in callback')
139:             if I[3] != 0:
140:                 problems.append('Problem is not full rank at solution')
141:             problems.append(stopreason)
142:         elif I[0] == 1:
143:             if I[1] != 0:
144:                 problems.append('N < 1')
145:             if I[2] != 0:
146:                 problems.append('M < 1')
147:             if I[3] != 0:
148:                 problems.append('NP < 1 or NP > N')
149:             if I[4] != 0:
150:                 problems.append('NQ < 1')
151:         elif I[0] == 2:
152:             if I[1] != 0:
153:                 problems.append('LDY and/or LDX incorrect')
154:             if I[2] != 0:
155:                 problems.append('LDWE, LD2WE, LDWD, and/or LD2WD incorrect')
156:             if I[3] != 0:
157:                 problems.append('LDIFX, LDSTPD, and/or LDSCLD incorrect')
158:             if I[4] != 0:
159:                 problems.append('LWORK and/or LIWORK too small')
160:         elif I[0] == 3:
161:             if I[1] != 0:
162:                 problems.append('STPB and/or STPD incorrect')
163:             if I[2] != 0:
164:                 problems.append('SCLB and/or SCLD incorrect')
165:             if I[3] != 0:
166:                 problems.append('WE incorrect')
167:             if I[4] != 0:
168:                 problems.append('WD incorrect')
169:         elif I[0] == 4:
170:             problems.append('Error in derivatives')
171:         elif I[0] == 5:
172:             problems.append('Error occurred in callback')
173:         elif I[0] == 6:
174:             problems.append('Numerical error detected')
175: 
176:         return problems
177: 
178:     else:
179:         return [stopreason]
180: 
181: 
182: class Data(object):
183:     '''
184:     The data to fit.
185: 
186:     Parameters
187:     ----------
188:     x : array_like
189:         Observed data for the independent variable of the regression
190:     y : array_like, optional
191:         If array-like, observed data for the dependent variable of the
192:         regression. A scalar input implies that the model to be used on
193:         the data is implicit.
194:     we : array_like, optional
195:         If `we` is a scalar, then that value is used for all data points (and
196:         all dimensions of the response variable).
197:         If `we` is a rank-1 array of length q (the dimensionality of the
198:         response variable), then this vector is the diagonal of the covariant
199:         weighting matrix for all data points.
200:         If `we` is a rank-1 array of length n (the number of data points), then
201:         the i'th element is the weight for the i'th response variable
202:         observation (single-dimensional only).
203:         If `we` is a rank-2 array of shape (q, q), then this is the full
204:         covariant weighting matrix broadcast to each observation.
205:         If `we` is a rank-2 array of shape (q, n), then `we[:,i]` is the
206:         diagonal of the covariant weighting matrix for the i'th observation.
207:         If `we` is a rank-3 array of shape (q, q, n), then `we[:,:,i]` is the
208:         full specification of the covariant weighting matrix for each
209:         observation.
210:         If the fit is implicit, then only a positive scalar value is used.
211:     wd : array_like, optional
212:         If `wd` is a scalar, then that value is used for all data points
213:         (and all dimensions of the input variable). If `wd` = 0, then the
214:         covariant weighting matrix for each observation is set to the identity
215:         matrix (so each dimension of each observation has the same weight).
216:         If `wd` is a rank-1 array of length m (the dimensionality of the input
217:         variable), then this vector is the diagonal of the covariant weighting
218:         matrix for all data points.
219:         If `wd` is a rank-1 array of length n (the number of data points), then
220:         the i'th element is the weight for the i'th input variable observation
221:         (single-dimensional only).
222:         If `wd` is a rank-2 array of shape (m, m), then this is the full
223:         covariant weighting matrix broadcast to each observation.
224:         If `wd` is a rank-2 array of shape (m, n), then `wd[:,i]` is the
225:         diagonal of the covariant weighting matrix for the i'th observation.
226:         If `wd` is a rank-3 array of shape (m, m, n), then `wd[:,:,i]` is the
227:         full specification of the covariant weighting matrix for each
228:         observation.
229:     fix : array_like of ints, optional
230:         The `fix` argument is the same as ifixx in the class ODR. It is an
231:         array of integers with the same shape as data.x that determines which
232:         input observations are treated as fixed. One can use a sequence of
233:         length m (the dimensionality of the input observations) to fix some
234:         dimensions for all observations. A value of 0 fixes the observation,
235:         a value > 0 makes it free.
236:     meta : dict, optional
237:         Free-form dictionary for metadata.
238: 
239:     Notes
240:     -----
241:     Each argument is attached to the member of the instance of the same name.
242:     The structures of `x` and `y` are described in the Model class docstring.
243:     If `y` is an integer, then the Data instance can only be used to fit with
244:     implicit models where the dimensionality of the response is equal to the
245:     specified value of `y`.
246: 
247:     The `we` argument weights the effect a deviation in the response variable
248:     has on the fit.  The `wd` argument weights the effect a deviation in the
249:     input variable has on the fit. To handle multidimensional inputs and
250:     responses easily, the structure of these arguments has the n'th
251:     dimensional axis first. These arguments heavily use the structured
252:     arguments feature of ODRPACK to conveniently and flexibly support all
253:     options. See the ODRPACK User's Guide for a full explanation of how these
254:     weights are used in the algorithm. Basically, a higher value of the weight
255:     for a particular data point makes a deviation at that point more
256:     detrimental to the fit.
257: 
258:     '''
259: 
260:     def __init__(self, x, y=None, we=None, wd=None, fix=None, meta={}):
261:         self.x = _conv(x)
262: 
263:         if not isinstance(self.x, numpy.ndarray):
264:             raise ValueError(("Expected an 'ndarray' of data for 'x', "
265:                               "but instead got data of type '{name}'").format(
266:                     name=type(self.x).__name__))
267: 
268:         self.y = _conv(y)
269:         self.we = _conv(we)
270:         self.wd = _conv(wd)
271:         self.fix = _conv(fix)
272:         self.meta = meta
273: 
274:     def set_meta(self, **kwds):
275:         ''' Update the metadata dictionary with the keywords and data provided
276:         by keywords.
277: 
278:         Examples
279:         --------
280:         ::
281: 
282:             data.set_meta(lab="Ph 7; Lab 26", title="Ag110 + Ag108 Decay")
283:         '''
284: 
285:         self.meta.update(kwds)
286: 
287:     def __getattr__(self, attr):
288:         ''' Dispatch attribute access to the metadata dictionary.
289:         '''
290:         if attr in self.meta:
291:             return self.meta[attr]
292:         else:
293:             raise AttributeError("'%s' not in metadata" % attr)
294: 
295: 
296: class RealData(Data):
297:     '''
298:     The data, with weightings as actual standard deviations and/or
299:     covariances.
300: 
301:     Parameters
302:     ----------
303:     x : array_like
304:         Observed data for the independent variable of the regression
305:     y : array_like, optional
306:         If array-like, observed data for the dependent variable of the
307:         regression. A scalar input implies that the model to be used on
308:         the data is implicit.
309:     sx : array_like, optional
310:         Standard deviations of `x`.
311:         `sx` are standard deviations of `x` and are converted to weights by
312:         dividing 1.0 by their squares.
313:     sy : array_like, optional
314:         Standard deviations of `y`.
315:         `sy` are standard deviations of `y` and are converted to weights by
316:         dividing 1.0 by their squares.
317:     covx : array_like, optional
318:         Covariance of `x`
319:         `covx` is an array of covariance matrices of `x` and are converted to
320:         weights by performing a matrix inversion on each observation's
321:         covariance matrix.
322:     covy : array_like, optional
323:         Covariance of `y`
324:         `covy` is an array of covariance matrices and are converted to
325:         weights by performing a matrix inversion on each observation's
326:         covariance matrix.
327:     fix : array_like, optional
328:         The argument and member fix is the same as Data.fix and ODR.ifixx:
329:         It is an array of integers with the same shape as `x` that
330:         determines which input observations are treated as fixed. One can
331:         use a sequence of length m (the dimensionality of the input
332:         observations) to fix some dimensions for all observations. A value
333:         of 0 fixes the observation, a value > 0 makes it free.
334:     meta : dict, optional
335:         Free-form dictionary for metadata.
336: 
337:     Notes
338:     -----
339:     The weights `wd` and `we` are computed from provided values as follows:
340: 
341:     `sx` and `sy` are converted to weights by dividing 1.0 by their squares.
342:     For example, ``wd = 1./numpy.power(`sx`, 2)``.
343: 
344:     `covx` and `covy` are arrays of covariance matrices and are converted to
345:     weights by performing a matrix inversion on each observation's covariance
346:     matrix.  For example, ``we[i] = numpy.linalg.inv(covy[i])``.
347: 
348:     These arguments follow the same structured argument conventions as wd and
349:     we only restricted by their natures: `sx` and `sy` can't be rank-3, but
350:     `covx` and `covy` can be.
351: 
352:     Only set *either* `sx` or `covx` (not both). Setting both will raise an
353:     exception.  Same with `sy` and `covy`.
354: 
355:     '''
356: 
357:     def __init__(self, x, y=None, sx=None, sy=None, covx=None, covy=None,
358:                  fix=None, meta={}):
359:         if (sx is not None) and (covx is not None):
360:             raise ValueError("cannot set both sx and covx")
361:         if (sy is not None) and (covy is not None):
362:             raise ValueError("cannot set both sy and covy")
363: 
364:         # Set flags for __getattr__
365:         self._ga_flags = {}
366:         if sx is not None:
367:             self._ga_flags['wd'] = 'sx'
368:         else:
369:             self._ga_flags['wd'] = 'covx'
370:         if sy is not None:
371:             self._ga_flags['we'] = 'sy'
372:         else:
373:             self._ga_flags['we'] = 'covy'
374: 
375:         self.x = _conv(x)
376: 
377:         if not isinstance(self.x, numpy.ndarray):
378:             raise ValueError(("Expected an 'ndarray' of data for 'x', "
379:                               "but instead got data of type '{name}'").format(
380:                     name=type(self.x).__name__))
381: 
382:         self.y = _conv(y)
383:         self.sx = _conv(sx)
384:         self.sy = _conv(sy)
385:         self.covx = _conv(covx)
386:         self.covy = _conv(covy)
387:         self.fix = _conv(fix)
388:         self.meta = meta
389: 
390:     def _sd2wt(self, sd):
391:         ''' Convert standard deviation to weights.
392:         '''
393: 
394:         return 1./numpy.power(sd, 2)
395: 
396:     def _cov2wt(self, cov):
397:         ''' Convert covariance matrix(-ices) to weights.
398:         '''
399: 
400:         from numpy.dual import inv
401: 
402:         if len(cov.shape) == 2:
403:             return inv(cov)
404:         else:
405:             weights = numpy.zeros(cov.shape, float)
406: 
407:             for i in range(cov.shape[-1]):  # n
408:                 weights[:,:,i] = inv(cov[:,:,i])
409: 
410:             return weights
411: 
412:     def __getattr__(self, attr):
413:         lookup_tbl = {('wd', 'sx'): (self._sd2wt, self.sx),
414:                       ('wd', 'covx'): (self._cov2wt, self.covx),
415:                       ('we', 'sy'): (self._sd2wt, self.sy),
416:                       ('we', 'covy'): (self._cov2wt, self.covy)}
417: 
418:         if attr not in ('wd', 'we'):
419:             if attr in self.meta:
420:                 return self.meta[attr]
421:             else:
422:                 raise AttributeError("'%s' not in metadata" % attr)
423:         else:
424:             func, arg = lookup_tbl[(attr, self._ga_flags[attr])]
425: 
426:             if arg is not None:
427:                 return func(*(arg,))
428:             else:
429:                 return None
430: 
431: 
432: class Model(object):
433:     '''
434:     The Model class stores information about the function you wish to fit.
435: 
436:     It stores the function itself, at the least, and optionally stores
437:     functions which compute the Jacobians used during fitting. Also, one
438:     can provide a function that will provide reasonable starting values
439:     for the fit parameters possibly given the set of data.
440: 
441:     Parameters
442:     ----------
443:     fcn : function
444:           fcn(beta, x) --> y
445:     fjacb : function
446:           Jacobian of fcn wrt the fit parameters beta.
447: 
448:           fjacb(beta, x) --> @f_i(x,B)/@B_j
449:     fjacd : function
450:           Jacobian of fcn wrt the (possibly multidimensional) input
451:           variable.
452: 
453:           fjacd(beta, x) --> @f_i(x,B)/@x_j
454:     extra_args : tuple, optional
455:           If specified, `extra_args` should be a tuple of extra
456:           arguments to pass to `fcn`, `fjacb`, and `fjacd`. Each will be called
457:           by `apply(fcn, (beta, x) + extra_args)`
458:     estimate : array_like of rank-1
459:           Provides estimates of the fit parameters from the data
460: 
461:           estimate(data) --> estbeta
462:     implicit : boolean
463:           If TRUE, specifies that the model
464:           is implicit; i.e `fcn(beta, x)` ~= 0 and there is no y data to fit
465:           against
466:     meta : dict, optional
467:           freeform dictionary of metadata for the model
468: 
469:     Notes
470:     -----
471:     Note that the `fcn`, `fjacb`, and `fjacd` operate on NumPy arrays and
472:     return a NumPy array. The `estimate` object takes an instance of the
473:     Data class.
474: 
475:     Here are the rules for the shapes of the argument and return
476:     arrays of the callback functions:
477: 
478:     `x`
479:         if the input data is single-dimensional, then `x` is rank-1
480:         array; i.e. ``x = array([1, 2, 3, ...]); x.shape = (n,)``
481:         If the input data is multi-dimensional, then `x` is a rank-2 array;
482:         i.e., ``x = array([[1, 2, ...], [2, 4, ...]]); x.shape = (m, n)``.
483:         In all cases, it has the same shape as the input data array passed to
484:         `odr`. `m` is the dimensionality of the input data, `n` is the number
485:         of observations.
486:     `y`
487:         if the response variable is single-dimensional, then `y` is a
488:         rank-1 array, i.e., ``y = array([2, 4, ...]); y.shape = (n,)``.
489:         If the response variable is multi-dimensional, then `y` is a rank-2
490:         array, i.e., ``y = array([[2, 4, ...], [3, 6, ...]]); y.shape =
491:         (q, n)`` where `q` is the dimensionality of the response variable.
492:     `beta`
493:         rank-1 array of length `p` where `p` is the number of parameters;
494:         i.e. ``beta = array([B_1, B_2, ..., B_p])``
495:     `fjacb`
496:         if the response variable is multi-dimensional, then the
497:         return array's shape is `(q, p, n)` such that ``fjacb(x,beta)[l,k,i] =
498:         d f_l(X,B)/d B_k`` evaluated at the i'th data point.  If `q == 1`, then
499:         the return array is only rank-2 and with shape `(p, n)`.
500:     `fjacd`
501:         as with fjacb, only the return array's shape is `(q, m, n)`
502:         such that ``fjacd(x,beta)[l,j,i] = d f_l(X,B)/d X_j`` at the i'th data
503:         point.  If `q == 1`, then the return array's shape is `(m, n)`. If
504:         `m == 1`, the shape is (q, n). If `m == q == 1`, the shape is `(n,)`.
505: 
506:     '''
507: 
508:     def __init__(self, fcn, fjacb=None, fjacd=None,
509:         extra_args=None, estimate=None, implicit=0, meta=None):
510: 
511:         self.fcn = fcn
512:         self.fjacb = fjacb
513:         self.fjacd = fjacd
514: 
515:         if extra_args is not None:
516:             extra_args = tuple(extra_args)
517: 
518:         self.extra_args = extra_args
519:         self.estimate = estimate
520:         self.implicit = implicit
521:         self.meta = meta
522: 
523:     def set_meta(self, **kwds):
524:         ''' Update the metadata dictionary with the keywords and data provided
525:         here.
526: 
527:         Examples
528:         --------
529:         set_meta(name="Exponential", equation="y = a exp(b x) + c")
530:         '''
531: 
532:         self.meta.update(kwds)
533: 
534:     def __getattr__(self, attr):
535:         ''' Dispatch attribute access to the metadata.
536:         '''
537: 
538:         if attr in self.meta:
539:             return self.meta[attr]
540:         else:
541:             raise AttributeError("'%s' not in metadata" % attr)
542: 
543: 
544: class Output(object):
545:     '''
546:     The Output class stores the output of an ODR run.
547: 
548:     Attributes
549:     ----------
550:     beta : ndarray
551:         Estimated parameter values, of shape (q,).
552:     sd_beta : ndarray
553:         Standard errors of the estimated parameters, of shape (p,).
554:     cov_beta : ndarray
555:         Covariance matrix of the estimated parameters, of shape (p,p).
556:     delta : ndarray, optional
557:         Array of estimated errors in input variables, of same shape as `x`.
558:     eps : ndarray, optional
559:         Array of estimated errors in response variables, of same shape as `y`.
560:     xplus : ndarray, optional
561:         Array of ``x + delta``.
562:     y : ndarray, optional
563:         Array ``y = fcn(x + delta)``.
564:     res_var : float, optional
565:         Residual variance.
566:     sum_square : float, optional
567:         Sum of squares error.
568:     sum_square_delta : float, optional
569:         Sum of squares of delta error.
570:     sum_square_eps : float, optional
571:         Sum of squares of eps error.
572:     inv_condnum : float, optional
573:         Inverse condition number (cf. ODRPACK UG p. 77).
574:     rel_error : float, optional
575:         Relative error in function values computed within fcn.
576:     work : ndarray, optional
577:         Final work array.
578:     work_ind : dict, optional
579:         Indices into work for drawing out values (cf. ODRPACK UG p. 83).
580:     info : int, optional
581:         Reason for returning, as output by ODRPACK (cf. ODRPACK UG p. 38).
582:     stopreason : list of str, optional
583:         `info` interpreted into English.
584: 
585:     Notes
586:     -----
587:     Takes one argument for initialization, the return value from the
588:     function `odr`. The attributes listed as "optional" above are only
589:     present if `odr` was run with ``full_output=1``.
590: 
591:     '''
592: 
593:     def __init__(self, output):
594:         self.beta = output[0]
595:         self.sd_beta = output[1]
596:         self.cov_beta = output[2]
597: 
598:         if len(output) == 4:
599:             # full output
600:             self.__dict__.update(output[3])
601:             self.stopreason = _report_error(self.info)
602: 
603:     def pprint(self):
604:         ''' Pretty-print important results.
605:         '''
606: 
607:         print('Beta:', self.beta)
608:         print('Beta Std Error:', self.sd_beta)
609:         print('Beta Covariance:', self.cov_beta)
610:         if hasattr(self, 'info'):
611:             print('Residual Variance:',self.res_var)
612:             print('Inverse Condition #:', self.inv_condnum)
613:             print('Reason(s) for Halting:')
614:             for r in self.stopreason:
615:                 print('  %s' % r)
616: 
617: 
618: class ODR(object):
619:     '''
620:     The ODR class gathers all information and coordinates the running of the
621:     main fitting routine.
622: 
623:     Members of instances of the ODR class have the same names as the arguments
624:     to the initialization routine.
625: 
626:     Parameters
627:     ----------
628:     data : Data class instance
629:         instance of the Data class
630:     model : Model class instance
631:         instance of the Model class
632: 
633:     Other Parameters
634:     ----------------
635:     beta0 : array_like of rank-1
636:         a rank-1 sequence of initial parameter values. Optional if
637:         model provides an "estimate" function to estimate these values.
638:     delta0 : array_like of floats of rank-1, optional
639:         a (double-precision) float array to hold the initial values of
640:         the errors in the input variables. Must be same shape as data.x
641:     ifixb : array_like of ints of rank-1, optional
642:         sequence of integers with the same length as beta0 that determines
643:         which parameters are held fixed. A value of 0 fixes the parameter,
644:         a value > 0 makes the parameter free.
645:     ifixx : array_like of ints with same shape as data.x, optional
646:         an array of integers with the same shape as data.x that determines
647:         which input observations are treated as fixed. One can use a sequence
648:         of length m (the dimensionality of the input observations) to fix some
649:         dimensions for all observations. A value of 0 fixes the observation,
650:         a value > 0 makes it free.
651:     job : int, optional
652:         an integer telling ODRPACK what tasks to perform. See p. 31 of the
653:         ODRPACK User's Guide if you absolutely must set the value here. Use the
654:         method set_job post-initialization for a more readable interface.
655:     iprint : int, optional
656:         an integer telling ODRPACK what to print. See pp. 33-34 of the
657:         ODRPACK User's Guide if you absolutely must set the value here. Use the
658:         method set_iprint post-initialization for a more readable interface.
659:     errfile : str, optional
660:         string with the filename to print ODRPACK errors to. *Do Not Open
661:         This File Yourself!*
662:     rptfile : str, optional
663:         string with the filename to print ODRPACK summaries to. *Do Not
664:         Open This File Yourself!*
665:     ndigit : int, optional
666:         integer specifying the number of reliable digits in the computation
667:         of the function.
668:     taufac : float, optional
669:         float specifying the initial trust region. The default value is 1.
670:         The initial trust region is equal to taufac times the length of the
671:         first computed Gauss-Newton step. taufac must be less than 1.
672:     sstol : float, optional
673:         float specifying the tolerance for convergence based on the relative
674:         change in the sum-of-squares. The default value is eps**(1/2) where eps
675:         is the smallest value such that 1 + eps > 1 for double precision
676:         computation on the machine. sstol must be less than 1.
677:     partol : float, optional
678:         float specifying the tolerance for convergence based on the relative
679:         change in the estimated parameters. The default value is eps**(2/3) for
680:         explicit models and ``eps**(1/3)`` for implicit models. partol must be less
681:         than 1.
682:     maxit : int, optional
683:         integer specifying the maximum number of iterations to perform. For
684:         first runs, maxit is the total number of iterations performed and
685:         defaults to 50.  For restarts, maxit is the number of additional
686:         iterations to perform and defaults to 10.
687:     stpb : array_like, optional
688:         sequence (``len(stpb) == len(beta0)``) of relative step sizes to compute
689:         finite difference derivatives wrt the parameters.
690:     stpd : optional
691:         array (``stpd.shape == data.x.shape`` or ``stpd.shape == (m,)``) of relative
692:         step sizes to compute finite difference derivatives wrt the input
693:         variable errors. If stpd is a rank-1 array with length m (the
694:         dimensionality of the input variable), then the values are broadcast to
695:         all observations.
696:     sclb : array_like, optional
697:         sequence (``len(stpb) == len(beta0)``) of scaling factors for the
698:         parameters.  The purpose of these scaling factors are to scale all of
699:         the parameters to around unity. Normally appropriate scaling factors
700:         are computed if this argument is not specified. Specify them yourself
701:         if the automatic procedure goes awry.
702:     scld : array_like, optional
703:         array (scld.shape == data.x.shape or scld.shape == (m,)) of scaling
704:         factors for the *errors* in the input variables. Again, these factors
705:         are automatically computed if you do not provide them. If scld.shape ==
706:         (m,), then the scaling factors are broadcast to all observations.
707:     work : ndarray, optional
708:         array to hold the double-valued working data for ODRPACK. When
709:         restarting, takes the value of self.output.work.
710:     iwork : ndarray, optional
711:         array to hold the integer-valued working data for ODRPACK. When
712:         restarting, takes the value of self.output.iwork.
713: 
714:     Attributes
715:     ----------
716:     data : Data
717:         The data for this fit
718:     model : Model
719:         The model used in fit
720:     output : Output
721:         An instance if the Output class containing all of the returned
722:         data from an invocation of ODR.run() or ODR.restart()
723: 
724:     '''
725: 
726:     def __init__(self, data, model, beta0=None, delta0=None, ifixb=None,
727:         ifixx=None, job=None, iprint=None, errfile=None, rptfile=None,
728:         ndigit=None, taufac=None, sstol=None, partol=None, maxit=None,
729:         stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None):
730: 
731:         self.data = data
732:         self.model = model
733: 
734:         if beta0 is None:
735:             if self.model.estimate is not None:
736:                 self.beta0 = _conv(self.model.estimate(self.data))
737:             else:
738:                 raise ValueError(
739:                   "must specify beta0 or provide an estimater with the model"
740:                 )
741:         else:
742:             self.beta0 = _conv(beta0)
743: 
744:         self.delta0 = _conv(delta0)
745:         # These really are 32-bit integers in FORTRAN (gfortran), even on 64-bit
746:         # platforms.
747:         # XXX: some other FORTRAN compilers may not agree.
748:         self.ifixx = _conv(ifixx, dtype=numpy.int32)
749:         self.ifixb = _conv(ifixb, dtype=numpy.int32)
750:         self.job = job
751:         self.iprint = iprint
752:         self.errfile = errfile
753:         self.rptfile = rptfile
754:         self.ndigit = ndigit
755:         self.taufac = taufac
756:         self.sstol = sstol
757:         self.partol = partol
758:         self.maxit = maxit
759:         self.stpb = _conv(stpb)
760:         self.stpd = _conv(stpd)
761:         self.sclb = _conv(sclb)
762:         self.scld = _conv(scld)
763:         self.work = _conv(work)
764:         self.iwork = _conv(iwork)
765: 
766:         self.output = None
767: 
768:         self._check()
769: 
770:     def _check(self):
771:         ''' Check the inputs for consistency, but don't bother checking things
772:         that the builtin function odr will check.
773:         '''
774: 
775:         x_s = list(self.data.x.shape)
776: 
777:         if isinstance(self.data.y, numpy.ndarray):
778:             y_s = list(self.data.y.shape)
779:             if self.model.implicit:
780:                 raise OdrError("an implicit model cannot use response data")
781:         else:
782:             # implicit model with q == self.data.y
783:             y_s = [self.data.y, x_s[-1]]
784:             if not self.model.implicit:
785:                 raise OdrError("an explicit model needs response data")
786:             self.set_job(fit_type=1)
787: 
788:         if x_s[-1] != y_s[-1]:
789:             raise OdrError("number of observations do not match")
790: 
791:         n = x_s[-1]
792: 
793:         if len(x_s) == 2:
794:             m = x_s[0]
795:         else:
796:             m = 1
797:         if len(y_s) == 2:
798:             q = y_s[0]
799:         else:
800:             q = 1
801: 
802:         p = len(self.beta0)
803: 
804:         # permissible output array shapes
805: 
806:         fcn_perms = [(q, n)]
807:         fjacd_perms = [(q, m, n)]
808:         fjacb_perms = [(q, p, n)]
809: 
810:         if q == 1:
811:             fcn_perms.append((n,))
812:             fjacd_perms.append((m, n))
813:             fjacb_perms.append((p, n))
814:         if m == 1:
815:             fjacd_perms.append((q, n))
816:         if p == 1:
817:             fjacb_perms.append((q, n))
818:         if m == q == 1:
819:             fjacd_perms.append((n,))
820:         if p == q == 1:
821:             fjacb_perms.append((n,))
822: 
823:         # try evaluating the supplied functions to make sure they provide
824:         # sensible outputs
825: 
826:         arglist = (self.beta0, self.data.x)
827:         if self.model.extra_args is not None:
828:             arglist = arglist + self.model.extra_args
829:         res = self.model.fcn(*arglist)
830: 
831:         if res.shape not in fcn_perms:
832:             print(res.shape)
833:             print(fcn_perms)
834:             raise OdrError("fcn does not output %s-shaped array" % y_s)
835: 
836:         if self.model.fjacd is not None:
837:             res = self.model.fjacd(*arglist)
838:             if res.shape not in fjacd_perms:
839:                 raise OdrError(
840:                     "fjacd does not output %s-shaped array" % repr((q, m, n)))
841:         if self.model.fjacb is not None:
842:             res = self.model.fjacb(*arglist)
843:             if res.shape not in fjacb_perms:
844:                 raise OdrError(
845:                     "fjacb does not output %s-shaped array" % repr((q, p, n)))
846: 
847:         # check shape of delta0
848: 
849:         if self.delta0 is not None and self.delta0.shape != self.data.x.shape:
850:             raise OdrError(
851:                 "delta0 is not a %s-shaped array" % repr(self.data.x.shape))
852: 
853:         if self.data.x.size == 0:
854:             warn(("Empty data detected for ODR instance. "
855:                   "Do not expect any fitting to occur"),
856:                  OdrWarning)
857: 
858:     def _gen_work(self):
859:         ''' Generate a suitable work array if one does not already exist.
860:         '''
861: 
862:         n = self.data.x.shape[-1]
863:         p = self.beta0.shape[0]
864: 
865:         if len(self.data.x.shape) == 2:
866:             m = self.data.x.shape[0]
867:         else:
868:             m = 1
869: 
870:         if self.model.implicit:
871:             q = self.data.y
872:         elif len(self.data.y.shape) == 2:
873:             q = self.data.y.shape[0]
874:         else:
875:             q = 1
876: 
877:         if self.data.we is None:
878:             ldwe = ld2we = 1
879:         elif len(self.data.we.shape) == 3:
880:             ld2we, ldwe = self.data.we.shape[1:]
881:         else:
882:             # Okay, this isn't precisely right, but for this calculation,
883:             # it's fine
884:             ldwe = 1
885:             ld2we = self.data.we.shape[1]
886: 
887:         if self.job % 10 < 2:
888:             # ODR not OLS
889:             lwork = (18 + 11*p + p*p + m + m*m + 4*n*q + 6*n*m + 2*n*q*p +
890:                      2*n*q*m + q*q + 5*q + q*(p+m) + ldwe*ld2we*q)
891:         else:
892:             # OLS not ODR
893:             lwork = (18 + 11*p + p*p + m + m*m + 4*n*q + 2*n*m + 2*n*q*p +
894:                      5*q + q*(p+m) + ldwe*ld2we*q)
895: 
896:         if isinstance(self.work, numpy.ndarray) and self.work.shape == (lwork,)\
897:                 and self.work.dtype.str.endswith('f8'):
898:             # the existing array is fine
899:             return
900:         else:
901:             self.work = numpy.zeros((lwork,), float)
902: 
903:     def set_job(self, fit_type=None, deriv=None, var_calc=None,
904:         del_init=None, restart=None):
905:         '''
906:         Sets the "job" parameter is a hopefully comprehensible way.
907: 
908:         If an argument is not specified, then the value is left as is. The
909:         default value from class initialization is for all of these options set
910:         to 0.
911: 
912:         Parameters
913:         ----------
914:         fit_type : {0, 1, 2} int
915:             0 -> explicit ODR
916: 
917:             1 -> implicit ODR
918: 
919:             2 -> ordinary least-squares
920:         deriv : {0, 1, 2, 3} int
921:             0 -> forward finite differences
922: 
923:             1 -> central finite differences
924: 
925:             2 -> user-supplied derivatives (Jacobians) with results
926:               checked by ODRPACK
927: 
928:             3 -> user-supplied derivatives, no checking
929:         var_calc : {0, 1, 2} int
930:             0 -> calculate asymptotic covariance matrix and fit
931:                  parameter uncertainties (V_B, s_B) using derivatives
932:                  recomputed at the final solution
933: 
934:             1 -> calculate V_B and s_B using derivatives from last iteration
935: 
936:             2 -> do not calculate V_B and s_B
937:         del_init : {0, 1} int
938:             0 -> initial input variable offsets set to 0
939: 
940:             1 -> initial offsets provided by user in variable "work"
941:         restart : {0, 1} int
942:             0 -> fit is not a restart
943: 
944:             1 -> fit is a restart
945: 
946:         Notes
947:         -----
948:         The permissible values are different from those given on pg. 31 of the
949:         ODRPACK User's Guide only in that one cannot specify numbers greater than
950:         the last value for each variable.
951: 
952:         If one does not supply functions to compute the Jacobians, the fitting
953:         procedure will change deriv to 0, finite differences, as a default. To
954:         initialize the input variable offsets by yourself, set del_init to 1 and
955:         put the offsets into the "work" variable correctly.
956: 
957:         '''
958: 
959:         if self.job is None:
960:             job_l = [0, 0, 0, 0, 0]
961:         else:
962:             job_l = [self.job // 10000 % 10,
963:                      self.job // 1000 % 10,
964:                      self.job // 100 % 10,
965:                      self.job // 10 % 10,
966:                      self.job % 10]
967: 
968:         if fit_type in (0, 1, 2):
969:             job_l[4] = fit_type
970:         if deriv in (0, 1, 2, 3):
971:             job_l[3] = deriv
972:         if var_calc in (0, 1, 2):
973:             job_l[2] = var_calc
974:         if del_init in (0, 1):
975:             job_l[1] = del_init
976:         if restart in (0, 1):
977:             job_l[0] = restart
978: 
979:         self.job = (job_l[0]*10000 + job_l[1]*1000 +
980:                     job_l[2]*100 + job_l[3]*10 + job_l[4])
981: 
982:     def set_iprint(self, init=None, so_init=None,
983:         iter=None, so_iter=None, iter_step=None, final=None, so_final=None):
984:         ''' Set the iprint parameter for the printing of computation reports.
985: 
986:         If any of the arguments are specified here, then they are set in the
987:         iprint member. If iprint is not set manually or with this method, then
988:         ODRPACK defaults to no printing. If no filename is specified with the
989:         member rptfile, then ODRPACK prints to stdout. One can tell ODRPACK to
990:         print to stdout in addition to the specified filename by setting the
991:         so_* arguments to this function, but one cannot specify to print to
992:         stdout but not a file since one can do that by not specifying a rptfile
993:         filename.
994: 
995:         There are three reports: initialization, iteration, and final reports.
996:         They are represented by the arguments init, iter, and final
997:         respectively.  The permissible values are 0, 1, and 2 representing "no
998:         report", "short report", and "long report" respectively.
999: 
1000:         The argument iter_step (0 <= iter_step <= 9) specifies how often to make
1001:         the iteration report; the report will be made for every iter_step'th
1002:         iteration starting with iteration one. If iter_step == 0, then no
1003:         iteration report is made, regardless of the other arguments.
1004: 
1005:         If the rptfile is None, then any so_* arguments supplied will raise an
1006:         exception.
1007:         '''
1008:         if self.iprint is None:
1009:             self.iprint = 0
1010: 
1011:         ip = [self.iprint // 1000 % 10,
1012:               self.iprint // 100 % 10,
1013:               self.iprint // 10 % 10,
1014:               self.iprint % 10]
1015: 
1016:         # make a list to convert iprint digits to/from argument inputs
1017:         #                   rptfile, stdout
1018:         ip2arg = [[0, 0],  # none,  none
1019:                   [1, 0],  # short, none
1020:                   [2, 0],  # long,  none
1021:                   [1, 1],  # short, short
1022:                   [2, 1],  # long,  short
1023:                   [1, 2],  # short, long
1024:                   [2, 2]]  # long,  long
1025: 
1026:         if (self.rptfile is None and
1027:             (so_init is not None or
1028:              so_iter is not None or
1029:              so_final is not None)):
1030:             raise OdrError(
1031:                 "no rptfile specified, cannot output to stdout twice")
1032: 
1033:         iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
1034: 
1035:         if init is not None:
1036:             iprint_l[0] = init
1037:         if so_init is not None:
1038:             iprint_l[1] = so_init
1039:         if iter is not None:
1040:             iprint_l[2] = iter
1041:         if so_iter is not None:
1042:             iprint_l[3] = so_iter
1043:         if final is not None:
1044:             iprint_l[4] = final
1045:         if so_final is not None:
1046:             iprint_l[5] = so_final
1047: 
1048:         if iter_step in range(10):
1049:             # 0..9
1050:             ip[2] = iter_step
1051: 
1052:         ip[0] = ip2arg.index(iprint_l[0:2])
1053:         ip[1] = ip2arg.index(iprint_l[2:4])
1054:         ip[3] = ip2arg.index(iprint_l[4:6])
1055: 
1056:         self.iprint = ip[0]*1000 + ip[1]*100 + ip[2]*10 + ip[3]
1057: 
1058:     def run(self):
1059:         ''' Run the fitting routine with all of the information given and with ``full_output=1``.
1060: 
1061:         Returns
1062:         -------
1063:         output : Output instance
1064:             This object is also assigned to the attribute .output .
1065:         '''
1066: 
1067:         args = (self.model.fcn, self.beta0, self.data.y, self.data.x)
1068:         kwds = {'full_output': 1}
1069:         kwd_l = ['ifixx', 'ifixb', 'job', 'iprint', 'errfile', 'rptfile',
1070:                  'ndigit', 'taufac', 'sstol', 'partol', 'maxit', 'stpb',
1071:                  'stpd', 'sclb', 'scld', 'work', 'iwork']
1072: 
1073:         if self.delta0 is not None and self.job % 1000 // 10 == 1:
1074:             # delta0 provided and fit is not a restart
1075:             self._gen_work()
1076: 
1077:             d0 = numpy.ravel(self.delta0)
1078: 
1079:             self.work[:len(d0)] = d0
1080: 
1081:         # set the kwds from other objects explicitly
1082:         if self.model.fjacb is not None:
1083:             kwds['fjacb'] = self.model.fjacb
1084:         if self.model.fjacd is not None:
1085:             kwds['fjacd'] = self.model.fjacd
1086:         if self.data.we is not None:
1087:             kwds['we'] = self.data.we
1088:         if self.data.wd is not None:
1089:             kwds['wd'] = self.data.wd
1090:         if self.model.extra_args is not None:
1091:             kwds['extra_args'] = self.model.extra_args
1092: 
1093:         # implicitly set kwds from self's members
1094:         for attr in kwd_l:
1095:             obj = getattr(self, attr)
1096:             if obj is not None:
1097:                 kwds[attr] = obj
1098: 
1099:         self.output = Output(odr(*args, **kwds))
1100: 
1101:         return self.output
1102: 
1103:     def restart(self, iter=None):
1104:         ''' Restarts the run with iter more iterations.
1105: 
1106:         Parameters
1107:         ----------
1108:         iter : int, optional
1109:             ODRPACK's default for the number of new iterations is 10.
1110: 
1111:         Returns
1112:         -------
1113:         output : Output instance
1114:             This object is also assigned to the attribute .output .
1115:         '''
1116: 
1117:         if self.output is None:
1118:             raise OdrError("cannot restart: run() has not been called before")
1119: 
1120:         self.set_job(restart=1)
1121:         self.work = self.output.work
1122:         self.iwork = self.output.iwork
1123: 
1124:         self.maxit = iter
1125: 
1126:         return self.run()
1127: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_163540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'str', "\nPython wrappers for Orthogonal Distance Regression (ODRPACK).\n\nNotes\n=====\n\n* Array formats -- FORTRAN stores its arrays in memory column first, i.e. an\n  array element A(i, j, k) will be next to A(i+1, j, k). In C and, consequently,\n  NumPy, arrays are stored row first: A[i, j, k] is next to A[i, j, k+1]. For\n  efficiency and convenience, the input and output arrays of the fitting\n  function (and its Jacobians) are passed to FORTRAN without transposition.\n  Therefore, where the ODRPACK documentation says that the X array is of shape\n  (N, M), it will be passed to the Python function as an array of shape (M, N).\n  If M==1, the one-dimensional case, then nothing matters; if M>1, then your\n  Python functions will be dealing with arrays that are indexed in reverse of\n  the ODRPACK documentation. No real biggie, but watch out for your indexing of\n  the Jacobians: the i,j'th elements (@f_i/@x_j) evaluated at the n'th\n  observation will be returned as jacd[j, i, n]. Except for the Jacobians, it\n  really is easier to deal with x[0] and x[1] than x[:,0] and x[:,1]. Of course,\n  you can always use the transpose() function from scipy explicitly.\n\n* Examples -- See the accompanying file test/test.py for examples of how to set\n  up fits of your own. Some are taken from the User's Guide; some are from\n  other sources.\n\n* Models -- Some common models are instantiated in the accompanying module\n  models.py . Contributions are welcome.\n\nCredits\n=======\n\n* Thanks to Arnold Moene and Gerard Vermeulen for fixing some killer bugs.\n\nRobert Kern\nrobert.kern@gmail.com\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'import numpy' statement (line 41)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_163541 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'numpy')

if (type(import_163541) is not StypyTypeError):

    if (import_163541 != 'pyd_module'):
        __import__(import_163541)
        sys_modules_163542 = sys.modules[import_163541]
        import_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'numpy', sys_modules_163542.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'numpy', import_163541)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 0))

# 'from warnings import warn' statement (line 42)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 0))

# 'from scipy.odr import __odrpack' statement (line 43)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_163543 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'scipy.odr')

if (type(import_163543) is not StypyTypeError):

    if (import_163543 != 'pyd_module'):
        __import__(import_163543)
        sys_modules_163544 = sys.modules[import_163543]
        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'scipy.odr', sys_modules_163544.module_type_store, module_type_store, ['__odrpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 43, 0), __file__, sys_modules_163544, sys_modules_163544.module_type_store, module_type_store)
    else:
        from scipy.odr import __odrpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'scipy.odr', None, module_type_store, ['__odrpack'], [__odrpack])

else:
    # Assigning a type to the variable 'scipy.odr' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'scipy.odr', import_163543)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')


# Assigning a List to a Name (line 45):

# Assigning a List to a Name (line 45):
__all__ = ['odr', 'OdrWarning', 'OdrError', 'OdrStop', 'Data', 'RealData', 'Model', 'Output', 'ODR', 'odr_error', 'odr_stop']
module_type_store.set_exportable_members(['odr', 'OdrWarning', 'OdrError', 'OdrStop', 'Data', 'RealData', 'Model', 'Output', 'ODR', 'odr_error', 'odr_stop'])

# Obtaining an instance of the builtin type 'list' (line 45)
list_163545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 45)
# Adding element type (line 45)
str_163546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 11), 'str', 'odr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163546)
# Adding element type (line 45)
str_163547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'str', 'OdrWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163547)
# Adding element type (line 45)
str_163548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 32), 'str', 'OdrError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163548)
# Adding element type (line 45)
str_163549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 44), 'str', 'OdrStop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163549)
# Adding element type (line 45)
str_163550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'str', 'Data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163550)
# Adding element type (line 45)
str_163551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'str', 'RealData')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163551)
# Adding element type (line 45)
str_163552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 31), 'str', 'Model')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163552)
# Adding element type (line 45)
str_163553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 40), 'str', 'Output')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163553)
# Adding element type (line 45)
str_163554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 50), 'str', 'ODR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163554)
# Adding element type (line 45)
str_163555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'str', 'odr_error')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163555)
# Adding element type (line 45)
str_163556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'str', 'odr_stop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 10), list_163545, str_163556)

# Assigning a type to the variable '__all__' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), '__all__', list_163545)

# Assigning a Attribute to a Name (line 49):

# Assigning a Attribute to a Name (line 49):
# Getting the type of '__odrpack' (line 49)
odrpack_163557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 6), '__odrpack')
# Obtaining the member 'odr' of a type (line 49)
odr_163558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 6), odrpack_163557, 'odr')
# Assigning a type to the variable 'odr' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'odr', odr_163558)
# Declaration of the 'OdrWarning' class
# Getting the type of 'UserWarning' (line 52)
UserWarning_163559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'UserWarning')

class OdrWarning(UserWarning_163559, ):
    str_163560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', "\n    Warning indicating that the data passed into\n    ODR will cause problems when passed into 'odr'\n    that the user should be aware of.\n    ")
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 52, 0, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdrWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'OdrWarning' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'OdrWarning', OdrWarning)
# Declaration of the 'OdrError' class
# Getting the type of 'Exception' (line 61)
Exception_163561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'Exception')

class OdrError(Exception_163561, ):
    str_163562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', '\n    Exception indicating an error in fitting.\n\n    This is raised by `scipy.odr` if an error occurs during fitting.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 61, 0, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdrError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'OdrError' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'OdrError', OdrError)
# Declaration of the 'OdrStop' class
# Getting the type of 'Exception' (line 70)
Exception_163563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'Exception')

class OdrStop(Exception_163563, ):
    str_163564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n    Exception stopping fitting.\n\n    You can raise this exception in your objective function to tell\n    `scipy.odr` to stop fitting.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 70, 0, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdrStop.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'OdrStop' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'OdrStop', OdrStop)

# Assigning a Name to a Name (line 80):

# Assigning a Name to a Name (line 80):
# Getting the type of 'OdrError' (line 80)
OdrError_163565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'OdrError')
# Assigning a type to the variable 'odr_error' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'odr_error', OdrError_163565)

# Assigning a Name to a Name (line 81):

# Assigning a Name to a Name (line 81):
# Getting the type of 'OdrStop' (line 81)
OdrStop_163566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'OdrStop')
# Assigning a type to the variable 'odr_stop' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'odr_stop', OdrStop_163566)

# Call to _set_exceptions(...): (line 83)
# Processing the call arguments (line 83)
# Getting the type of 'OdrError' (line 83)
OdrError_163569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), 'OdrError', False)
# Getting the type of 'OdrStop' (line 83)
OdrStop_163570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 36), 'OdrStop', False)
# Processing the call keyword arguments (line 83)
kwargs_163571 = {}
# Getting the type of '__odrpack' (line 83)
odrpack_163567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), '__odrpack', False)
# Obtaining the member '_set_exceptions' of a type (line 83)
_set_exceptions_163568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 0), odrpack_163567, '_set_exceptions')
# Calling _set_exceptions(args, kwargs) (line 83)
_set_exceptions_call_result_163572 = invoke(stypy.reporting.localization.Localization(__file__, 83, 0), _set_exceptions_163568, *[OdrError_163569, OdrStop_163570], **kwargs_163571)


@norecursion
def _conv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 86)
    None_163573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'None')
    defaults = [None_163573]
    # Create a new context for function '_conv'
    module_type_store = module_type_store.open_function_context('_conv', 86, 0, False)
    
    # Passed parameters checking function
    _conv.stypy_localization = localization
    _conv.stypy_type_of_self = None
    _conv.stypy_type_store = module_type_store
    _conv.stypy_function_name = '_conv'
    _conv.stypy_param_names_list = ['obj', 'dtype']
    _conv.stypy_varargs_param_name = None
    _conv.stypy_kwargs_param_name = None
    _conv.stypy_call_defaults = defaults
    _conv.stypy_call_varargs = varargs
    _conv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_conv', ['obj', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_conv', localization, ['obj', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_conv(...)' code ##################

    str_163574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'str', ' Convert an object to the preferred form for input to the odr routine.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 90)
    # Getting the type of 'obj' (line 90)
    obj_163575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'obj')
    # Getting the type of 'None' (line 90)
    None_163576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'None')
    
    (may_be_163577, more_types_in_union_163578) = may_be_none(obj_163575, None_163576)

    if may_be_163577:

        if more_types_in_union_163578:
            # Runtime conditional SSA (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'obj' (line 91)
        obj_163579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', obj_163579)

        if more_types_in_union_163578:
            # Runtime conditional SSA for else branch (line 90)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_163577) or more_types_in_union_163578):
        
        # Type idiom detected: calculating its left and rigth part (line 93)
        # Getting the type of 'dtype' (line 93)
        dtype_163580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'dtype')
        # Getting the type of 'None' (line 93)
        None_163581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'None')
        
        (may_be_163582, more_types_in_union_163583) = may_be_none(dtype_163580, None_163581)

        if may_be_163582:

            if more_types_in_union_163583:
                # Runtime conditional SSA (line 93)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 94):
            
            # Assigning a Call to a Name (line 94):
            
            # Call to asarray(...): (line 94)
            # Processing the call arguments (line 94)
            # Getting the type of 'obj' (line 94)
            obj_163586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'obj', False)
            # Processing the call keyword arguments (line 94)
            kwargs_163587 = {}
            # Getting the type of 'numpy' (line 94)
            numpy_163584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'numpy', False)
            # Obtaining the member 'asarray' of a type (line 94)
            asarray_163585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 18), numpy_163584, 'asarray')
            # Calling asarray(args, kwargs) (line 94)
            asarray_call_result_163588 = invoke(stypy.reporting.localization.Localization(__file__, 94, 18), asarray_163585, *[obj_163586], **kwargs_163587)
            
            # Assigning a type to the variable 'obj' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'obj', asarray_call_result_163588)

            if more_types_in_union_163583:
                # Runtime conditional SSA for else branch (line 93)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_163582) or more_types_in_union_163583):
            
            # Assigning a Call to a Name (line 96):
            
            # Assigning a Call to a Name (line 96):
            
            # Call to asarray(...): (line 96)
            # Processing the call arguments (line 96)
            # Getting the type of 'obj' (line 96)
            obj_163591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'obj', False)
            # Getting the type of 'dtype' (line 96)
            dtype_163592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'dtype', False)
            # Processing the call keyword arguments (line 96)
            kwargs_163593 = {}
            # Getting the type of 'numpy' (line 96)
            numpy_163589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'numpy', False)
            # Obtaining the member 'asarray' of a type (line 96)
            asarray_163590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), numpy_163589, 'asarray')
            # Calling asarray(args, kwargs) (line 96)
            asarray_call_result_163594 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), asarray_163590, *[obj_163591, dtype_163592], **kwargs_163593)
            
            # Assigning a type to the variable 'obj' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'obj', asarray_call_result_163594)

            if (may_be_163582 and more_types_in_union_163583):
                # SSA join for if statement (line 93)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'obj' (line 97)
        obj_163595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'obj')
        # Obtaining the member 'shape' of a type (line 97)
        shape_163596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), obj_163595, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_163597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        
        # Applying the binary operator '==' (line 97)
        result_eq_163598 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), '==', shape_163596, tuple_163597)
        
        # Testing the type of an if condition (line 97)
        if_condition_163599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_eq_163598)
        # Assigning a type to the variable 'if_condition_163599' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_163599', if_condition_163599)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to type(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'obj' (line 99)
        obj_163603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'obj', False)
        # Processing the call keyword arguments (line 99)
        kwargs_163604 = {}
        # Getting the type of 'obj' (line 99)
        obj_163600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 99)
        dtype_163601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), obj_163600, 'dtype')
        # Obtaining the member 'type' of a type (line 99)
        type_163602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), dtype_163601, 'type')
        # Calling type(args, kwargs) (line 99)
        type_call_result_163605 = invoke(stypy.reporting.localization.Localization(__file__, 99, 19), type_163602, *[obj_163603], **kwargs_163604)
        
        # Assigning a type to the variable 'stypy_return_type' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'stypy_return_type', type_call_result_163605)
        # SSA branch for the else part of an if statement (line 97)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'obj' (line 101)
        obj_163606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', obj_163606)
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_163577 and more_types_in_union_163578):
            # SSA join for if statement (line 90)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_conv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_conv' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_163607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163607)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_conv'
    return stypy_return_type_163607

# Assigning a type to the variable '_conv' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), '_conv', _conv)

@norecursion
def _report_error(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_report_error'
    module_type_store = module_type_store.open_function_context('_report_error', 104, 0, False)
    
    # Passed parameters checking function
    _report_error.stypy_localization = localization
    _report_error.stypy_type_of_self = None
    _report_error.stypy_type_store = module_type_store
    _report_error.stypy_function_name = '_report_error'
    _report_error.stypy_param_names_list = ['info']
    _report_error.stypy_varargs_param_name = None
    _report_error.stypy_kwargs_param_name = None
    _report_error.stypy_call_defaults = defaults
    _report_error.stypy_call_varargs = varargs
    _report_error.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_report_error', ['info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_report_error', localization, ['info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_report_error(...)' code ##################

    str_163608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, (-1)), 'str', ' Interprets the return code of the odr routine.\n\n    Parameters\n    ----------\n    info : int\n        The return code of the odr routine.\n\n    Returns\n    -------\n    problems : list(str)\n        A list of messages about why the odr() routine stopped.\n    ')
    
    # Assigning a Subscript to a Name (line 118):
    
    # Assigning a Subscript to a Name (line 118):
    
    # Obtaining the type of the subscript
    # Getting the type of 'info' (line 122)
    info_163609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'info')
    int_163610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 52), 'int')
    # Applying the binary operator '%' (line 122)
    result_mod_163611 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 45), '%', info_163609, int_163610)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 118)
    tuple_163612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 118)
    # Adding element type (line 118)
    str_163613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'str', 'Blank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_163612, str_163613)
    # Adding element type (line 118)
    str_163614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 18), 'str', 'Sum of squares convergence')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_163612, str_163614)
    # Adding element type (line 118)
    str_163615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'str', 'Parameter convergence')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_163612, str_163615)
    # Adding element type (line 118)
    str_163616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 18), 'str', 'Both sum of squares and parameter convergence')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_163612, str_163616)
    # Adding element type (line 118)
    str_163617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 18), 'str', 'Iteration limit reached')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_163612, str_163617)
    
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___163618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_163612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_163619 = invoke(stypy.reporting.localization.Localization(__file__, 118, 18), getitem___163618, result_mod_163611)
    
    # Assigning a type to the variable 'stopreason' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stopreason', subscript_call_result_163619)
    
    
    # Getting the type of 'info' (line 124)
    info_163620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'info')
    int_163621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'int')
    # Applying the binary operator '>=' (line 124)
    result_ge_163622 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 7), '>=', info_163620, int_163621)
    
    # Testing the type of an if condition (line 124)
    if_condition_163623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), result_ge_163622)
    # Assigning a type to the variable 'if_condition_163623' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_163623', if_condition_163623)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 127):
    
    # Assigning a Tuple to a Name (line 127):
    
    # Obtaining an instance of the builtin type 'tuple' (line 127)
    tuple_163624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 127)
    # Adding element type (line 127)
    # Getting the type of 'info' (line 127)
    info_163625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'info')
    int_163626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 19), 'int')
    # Applying the binary operator '//' (line 127)
    result_floordiv_163627 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 13), '//', info_163625, int_163626)
    
    int_163628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'int')
    # Applying the binary operator '%' (line 127)
    result_mod_163629 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 25), '%', result_floordiv_163627, int_163628)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_163624, result_mod_163629)
    # Adding element type (line 127)
    # Getting the type of 'info' (line 128)
    info_163630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'info')
    int_163631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'int')
    # Applying the binary operator '//' (line 128)
    result_floordiv_163632 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 13), '//', info_163630, int_163631)
    
    int_163633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 26), 'int')
    # Applying the binary operator '%' (line 128)
    result_mod_163634 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 24), '%', result_floordiv_163632, int_163633)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_163624, result_mod_163634)
    # Adding element type (line 127)
    # Getting the type of 'info' (line 129)
    info_163635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 13), 'info')
    int_163636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 19), 'int')
    # Applying the binary operator '//' (line 129)
    result_floordiv_163637 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 13), '//', info_163635, int_163636)
    
    int_163638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 25), 'int')
    # Applying the binary operator '%' (line 129)
    result_mod_163639 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 23), '%', result_floordiv_163637, int_163638)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_163624, result_mod_163639)
    # Adding element type (line 127)
    # Getting the type of 'info' (line 130)
    info_163640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'info')
    int_163641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'int')
    # Applying the binary operator '//' (line 130)
    result_floordiv_163642 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 13), '//', info_163640, int_163641)
    
    int_163643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 24), 'int')
    # Applying the binary operator '%' (line 130)
    result_mod_163644 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 22), '%', result_floordiv_163642, int_163643)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_163624, result_mod_163644)
    # Adding element type (line 127)
    # Getting the type of 'info' (line 131)
    info_163645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'info')
    int_163646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 20), 'int')
    # Applying the binary operator '%' (line 131)
    result_mod_163647 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 13), '%', info_163645, int_163646)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_163624, result_mod_163647)
    
    # Assigning a type to the variable 'I' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'I', tuple_163624)
    
    # Assigning a List to a Name (line 132):
    
    # Assigning a List to a Name (line 132):
    
    # Obtaining an instance of the builtin type 'list' (line 132)
    list_163648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 132)
    
    # Assigning a type to the variable 'problems' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'problems', list_163648)
    
    
    
    # Obtaining the type of the subscript
    int_163649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 13), 'int')
    # Getting the type of 'I' (line 134)
    I_163650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'I')
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___163651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), I_163650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_163652 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), getitem___163651, int_163649)
    
    int_163653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 19), 'int')
    # Applying the binary operator '==' (line 134)
    result_eq_163654 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), '==', subscript_call_result_163652, int_163653)
    
    # Testing the type of an if condition (line 134)
    if_condition_163655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), result_eq_163654)
    # Assigning a type to the variable 'if_condition_163655' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_163655', if_condition_163655)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_163656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 17), 'int')
    # Getting the type of 'I' (line 135)
    I_163657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___163658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), I_163657, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_163659 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), getitem___163658, int_163656)
    
    int_163660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'int')
    # Applying the binary operator '!=' (line 135)
    result_ne_163661 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '!=', subscript_call_result_163659, int_163660)
    
    # Testing the type of an if condition (line 135)
    if_condition_163662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), result_ne_163661)
    # Assigning a type to the variable 'if_condition_163662' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_163662', if_condition_163662)
    # SSA begins for if statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 136)
    # Processing the call arguments (line 136)
    str_163665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 32), 'str', 'Derivatives possibly not correct')
    # Processing the call keyword arguments (line 136)
    kwargs_163666 = {}
    # Getting the type of 'problems' (line 136)
    problems_163663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 136)
    append_163664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 16), problems_163663, 'append')
    # Calling append(args, kwargs) (line 136)
    append_call_result_163667 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), append_163664, *[str_163665], **kwargs_163666)
    
    # SSA join for if statement (line 135)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 17), 'int')
    # Getting the type of 'I' (line 137)
    I_163669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___163670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), I_163669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_163671 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), getitem___163670, int_163668)
    
    int_163672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'int')
    # Applying the binary operator '!=' (line 137)
    result_ne_163673 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 15), '!=', subscript_call_result_163671, int_163672)
    
    # Testing the type of an if condition (line 137)
    if_condition_163674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 12), result_ne_163673)
    # Assigning a type to the variable 'if_condition_163674' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'if_condition_163674', if_condition_163674)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 138)
    # Processing the call arguments (line 138)
    str_163677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 32), 'str', 'Error occurred in callback')
    # Processing the call keyword arguments (line 138)
    kwargs_163678 = {}
    # Getting the type of 'problems' (line 138)
    problems_163675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 138)
    append_163676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), problems_163675, 'append')
    # Calling append(args, kwargs) (line 138)
    append_call_result_163679 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), append_163676, *[str_163677], **kwargs_163678)
    
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 17), 'int')
    # Getting the type of 'I' (line 139)
    I_163681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___163682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), I_163681, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_163683 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), getitem___163682, int_163680)
    
    int_163684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 23), 'int')
    # Applying the binary operator '!=' (line 139)
    result_ne_163685 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 15), '!=', subscript_call_result_163683, int_163684)
    
    # Testing the type of an if condition (line 139)
    if_condition_163686 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 12), result_ne_163685)
    # Assigning a type to the variable 'if_condition_163686' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'if_condition_163686', if_condition_163686)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 140)
    # Processing the call arguments (line 140)
    str_163689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 32), 'str', 'Problem is not full rank at solution')
    # Processing the call keyword arguments (line 140)
    kwargs_163690 = {}
    # Getting the type of 'problems' (line 140)
    problems_163687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 140)
    append_163688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), problems_163687, 'append')
    # Calling append(args, kwargs) (line 140)
    append_call_result_163691 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), append_163688, *[str_163689], **kwargs_163690)
    
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'stopreason' (line 141)
    stopreason_163694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 28), 'stopreason', False)
    # Processing the call keyword arguments (line 141)
    kwargs_163695 = {}
    # Getting the type of 'problems' (line 141)
    problems_163692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'problems', False)
    # Obtaining the member 'append' of a type (line 141)
    append_163693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), problems_163692, 'append')
    # Calling append(args, kwargs) (line 141)
    append_call_result_163696 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), append_163693, *[stopreason_163694], **kwargs_163695)
    
    # SSA branch for the else part of an if statement (line 134)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_163697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'int')
    # Getting the type of 'I' (line 142)
    I_163698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'I')
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___163699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 13), I_163698, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_163700 = invoke(stypy.reporting.localization.Localization(__file__, 142, 13), getitem___163699, int_163697)
    
    int_163701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'int')
    # Applying the binary operator '==' (line 142)
    result_eq_163702 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 13), '==', subscript_call_result_163700, int_163701)
    
    # Testing the type of an if condition (line 142)
    if_condition_163703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 13), result_eq_163702)
    # Assigning a type to the variable 'if_condition_163703' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'if_condition_163703', if_condition_163703)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_163704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'int')
    # Getting the type of 'I' (line 143)
    I_163705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___163706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), I_163705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_163707 = invoke(stypy.reporting.localization.Localization(__file__, 143, 15), getitem___163706, int_163704)
    
    int_163708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 23), 'int')
    # Applying the binary operator '!=' (line 143)
    result_ne_163709 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), '!=', subscript_call_result_163707, int_163708)
    
    # Testing the type of an if condition (line 143)
    if_condition_163710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 12), result_ne_163709)
    # Assigning a type to the variable 'if_condition_163710' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'if_condition_163710', if_condition_163710)
    # SSA begins for if statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 144)
    # Processing the call arguments (line 144)
    str_163713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 32), 'str', 'N < 1')
    # Processing the call keyword arguments (line 144)
    kwargs_163714 = {}
    # Getting the type of 'problems' (line 144)
    problems_163711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 144)
    append_163712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), problems_163711, 'append')
    # Calling append(args, kwargs) (line 144)
    append_call_result_163715 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), append_163712, *[str_163713], **kwargs_163714)
    
    # SSA join for if statement (line 143)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 17), 'int')
    # Getting the type of 'I' (line 145)
    I_163717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 145)
    getitem___163718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 15), I_163717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 145)
    subscript_call_result_163719 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), getitem___163718, int_163716)
    
    int_163720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'int')
    # Applying the binary operator '!=' (line 145)
    result_ne_163721 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 15), '!=', subscript_call_result_163719, int_163720)
    
    # Testing the type of an if condition (line 145)
    if_condition_163722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 12), result_ne_163721)
    # Assigning a type to the variable 'if_condition_163722' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'if_condition_163722', if_condition_163722)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 146)
    # Processing the call arguments (line 146)
    str_163725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 32), 'str', 'M < 1')
    # Processing the call keyword arguments (line 146)
    kwargs_163726 = {}
    # Getting the type of 'problems' (line 146)
    problems_163723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 146)
    append_163724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), problems_163723, 'append')
    # Calling append(args, kwargs) (line 146)
    append_call_result_163727 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), append_163724, *[str_163725], **kwargs_163726)
    
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 17), 'int')
    # Getting the type of 'I' (line 147)
    I_163729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___163730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 15), I_163729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_163731 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), getitem___163730, int_163728)
    
    int_163732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'int')
    # Applying the binary operator '!=' (line 147)
    result_ne_163733 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 15), '!=', subscript_call_result_163731, int_163732)
    
    # Testing the type of an if condition (line 147)
    if_condition_163734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 12), result_ne_163733)
    # Assigning a type to the variable 'if_condition_163734' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'if_condition_163734', if_condition_163734)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 148)
    # Processing the call arguments (line 148)
    str_163737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 32), 'str', 'NP < 1 or NP > N')
    # Processing the call keyword arguments (line 148)
    kwargs_163738 = {}
    # Getting the type of 'problems' (line 148)
    problems_163735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 148)
    append_163736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), problems_163735, 'append')
    # Calling append(args, kwargs) (line 148)
    append_call_result_163739 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), append_163736, *[str_163737], **kwargs_163738)
    
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 17), 'int')
    # Getting the type of 'I' (line 149)
    I_163741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___163742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), I_163741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_163743 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), getitem___163742, int_163740)
    
    int_163744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 23), 'int')
    # Applying the binary operator '!=' (line 149)
    result_ne_163745 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), '!=', subscript_call_result_163743, int_163744)
    
    # Testing the type of an if condition (line 149)
    if_condition_163746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 12), result_ne_163745)
    # Assigning a type to the variable 'if_condition_163746' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'if_condition_163746', if_condition_163746)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 150)
    # Processing the call arguments (line 150)
    str_163749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 32), 'str', 'NQ < 1')
    # Processing the call keyword arguments (line 150)
    kwargs_163750 = {}
    # Getting the type of 'problems' (line 150)
    problems_163747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 150)
    append_163748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), problems_163747, 'append')
    # Calling append(args, kwargs) (line 150)
    append_call_result_163751 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), append_163748, *[str_163749], **kwargs_163750)
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 142)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_163752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'int')
    # Getting the type of 'I' (line 151)
    I_163753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'I')
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___163754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 13), I_163753, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_163755 = invoke(stypy.reporting.localization.Localization(__file__, 151, 13), getitem___163754, int_163752)
    
    int_163756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 21), 'int')
    # Applying the binary operator '==' (line 151)
    result_eq_163757 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 13), '==', subscript_call_result_163755, int_163756)
    
    # Testing the type of an if condition (line 151)
    if_condition_163758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 13), result_eq_163757)
    # Assigning a type to the variable 'if_condition_163758' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'if_condition_163758', if_condition_163758)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_163759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 17), 'int')
    # Getting the type of 'I' (line 152)
    I_163760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___163761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 15), I_163760, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_163762 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), getitem___163761, int_163759)
    
    int_163763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'int')
    # Applying the binary operator '!=' (line 152)
    result_ne_163764 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 15), '!=', subscript_call_result_163762, int_163763)
    
    # Testing the type of an if condition (line 152)
    if_condition_163765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 12), result_ne_163764)
    # Assigning a type to the variable 'if_condition_163765' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'if_condition_163765', if_condition_163765)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 153)
    # Processing the call arguments (line 153)
    str_163768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 32), 'str', 'LDY and/or LDX incorrect')
    # Processing the call keyword arguments (line 153)
    kwargs_163769 = {}
    # Getting the type of 'problems' (line 153)
    problems_163766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 153)
    append_163767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), problems_163766, 'append')
    # Calling append(args, kwargs) (line 153)
    append_call_result_163770 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), append_163767, *[str_163768], **kwargs_163769)
    
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 17), 'int')
    # Getting the type of 'I' (line 154)
    I_163772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___163773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 15), I_163772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_163774 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), getitem___163773, int_163771)
    
    int_163775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'int')
    # Applying the binary operator '!=' (line 154)
    result_ne_163776 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '!=', subscript_call_result_163774, int_163775)
    
    # Testing the type of an if condition (line 154)
    if_condition_163777 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 12), result_ne_163776)
    # Assigning a type to the variable 'if_condition_163777' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'if_condition_163777', if_condition_163777)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 155)
    # Processing the call arguments (line 155)
    str_163780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 32), 'str', 'LDWE, LD2WE, LDWD, and/or LD2WD incorrect')
    # Processing the call keyword arguments (line 155)
    kwargs_163781 = {}
    # Getting the type of 'problems' (line 155)
    problems_163778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 155)
    append_163779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), problems_163778, 'append')
    # Calling append(args, kwargs) (line 155)
    append_call_result_163782 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), append_163779, *[str_163780], **kwargs_163781)
    
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 17), 'int')
    # Getting the type of 'I' (line 156)
    I_163784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___163785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 15), I_163784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_163786 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), getitem___163785, int_163783)
    
    int_163787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 23), 'int')
    # Applying the binary operator '!=' (line 156)
    result_ne_163788 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 15), '!=', subscript_call_result_163786, int_163787)
    
    # Testing the type of an if condition (line 156)
    if_condition_163789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 12), result_ne_163788)
    # Assigning a type to the variable 'if_condition_163789' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'if_condition_163789', if_condition_163789)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 157)
    # Processing the call arguments (line 157)
    str_163792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 32), 'str', 'LDIFX, LDSTPD, and/or LDSCLD incorrect')
    # Processing the call keyword arguments (line 157)
    kwargs_163793 = {}
    # Getting the type of 'problems' (line 157)
    problems_163790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 157)
    append_163791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), problems_163790, 'append')
    # Calling append(args, kwargs) (line 157)
    append_call_result_163794 = invoke(stypy.reporting.localization.Localization(__file__, 157, 16), append_163791, *[str_163792], **kwargs_163793)
    
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 17), 'int')
    # Getting the type of 'I' (line 158)
    I_163796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 158)
    getitem___163797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 15), I_163796, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 158)
    subscript_call_result_163798 = invoke(stypy.reporting.localization.Localization(__file__, 158, 15), getitem___163797, int_163795)
    
    int_163799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'int')
    # Applying the binary operator '!=' (line 158)
    result_ne_163800 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 15), '!=', subscript_call_result_163798, int_163799)
    
    # Testing the type of an if condition (line 158)
    if_condition_163801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 12), result_ne_163800)
    # Assigning a type to the variable 'if_condition_163801' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'if_condition_163801', if_condition_163801)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 159)
    # Processing the call arguments (line 159)
    str_163804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 32), 'str', 'LWORK and/or LIWORK too small')
    # Processing the call keyword arguments (line 159)
    kwargs_163805 = {}
    # Getting the type of 'problems' (line 159)
    problems_163802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 159)
    append_163803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), problems_163802, 'append')
    # Calling append(args, kwargs) (line 159)
    append_call_result_163806 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), append_163803, *[str_163804], **kwargs_163805)
    
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 151)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_163807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 15), 'int')
    # Getting the type of 'I' (line 160)
    I_163808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'I')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___163809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 13), I_163808, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_163810 = invoke(stypy.reporting.localization.Localization(__file__, 160, 13), getitem___163809, int_163807)
    
    int_163811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 21), 'int')
    # Applying the binary operator '==' (line 160)
    result_eq_163812 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 13), '==', subscript_call_result_163810, int_163811)
    
    # Testing the type of an if condition (line 160)
    if_condition_163813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 13), result_eq_163812)
    # Assigning a type to the variable 'if_condition_163813' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'if_condition_163813', if_condition_163813)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_163814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 17), 'int')
    # Getting the type of 'I' (line 161)
    I_163815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___163816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), I_163815, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_163817 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), getitem___163816, int_163814)
    
    int_163818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 23), 'int')
    # Applying the binary operator '!=' (line 161)
    result_ne_163819 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '!=', subscript_call_result_163817, int_163818)
    
    # Testing the type of an if condition (line 161)
    if_condition_163820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), result_ne_163819)
    # Assigning a type to the variable 'if_condition_163820' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_163820', if_condition_163820)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 162)
    # Processing the call arguments (line 162)
    str_163823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 32), 'str', 'STPB and/or STPD incorrect')
    # Processing the call keyword arguments (line 162)
    kwargs_163824 = {}
    # Getting the type of 'problems' (line 162)
    problems_163821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 162)
    append_163822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 16), problems_163821, 'append')
    # Calling append(args, kwargs) (line 162)
    append_call_result_163825 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), append_163822, *[str_163823], **kwargs_163824)
    
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 17), 'int')
    # Getting the type of 'I' (line 163)
    I_163827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___163828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), I_163827, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_163829 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), getitem___163828, int_163826)
    
    int_163830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 23), 'int')
    # Applying the binary operator '!=' (line 163)
    result_ne_163831 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 15), '!=', subscript_call_result_163829, int_163830)
    
    # Testing the type of an if condition (line 163)
    if_condition_163832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 12), result_ne_163831)
    # Assigning a type to the variable 'if_condition_163832' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'if_condition_163832', if_condition_163832)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 164)
    # Processing the call arguments (line 164)
    str_163835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 32), 'str', 'SCLB and/or SCLD incorrect')
    # Processing the call keyword arguments (line 164)
    kwargs_163836 = {}
    # Getting the type of 'problems' (line 164)
    problems_163833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 164)
    append_163834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), problems_163833, 'append')
    # Calling append(args, kwargs) (line 164)
    append_call_result_163837 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), append_163834, *[str_163835], **kwargs_163836)
    
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 17), 'int')
    # Getting the type of 'I' (line 165)
    I_163839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___163840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 15), I_163839, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_163841 = invoke(stypy.reporting.localization.Localization(__file__, 165, 15), getitem___163840, int_163838)
    
    int_163842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'int')
    # Applying the binary operator '!=' (line 165)
    result_ne_163843 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 15), '!=', subscript_call_result_163841, int_163842)
    
    # Testing the type of an if condition (line 165)
    if_condition_163844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 12), result_ne_163843)
    # Assigning a type to the variable 'if_condition_163844' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'if_condition_163844', if_condition_163844)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 166)
    # Processing the call arguments (line 166)
    str_163847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 32), 'str', 'WE incorrect')
    # Processing the call keyword arguments (line 166)
    kwargs_163848 = {}
    # Getting the type of 'problems' (line 166)
    problems_163845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 166)
    append_163846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), problems_163845, 'append')
    # Calling append(args, kwargs) (line 166)
    append_call_result_163849 = invoke(stypy.reporting.localization.Localization(__file__, 166, 16), append_163846, *[str_163847], **kwargs_163848)
    
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_163850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 17), 'int')
    # Getting the type of 'I' (line 167)
    I_163851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'I')
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___163852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 15), I_163851, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_163853 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), getitem___163852, int_163850)
    
    int_163854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'int')
    # Applying the binary operator '!=' (line 167)
    result_ne_163855 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 15), '!=', subscript_call_result_163853, int_163854)
    
    # Testing the type of an if condition (line 167)
    if_condition_163856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 12), result_ne_163855)
    # Assigning a type to the variable 'if_condition_163856' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'if_condition_163856', if_condition_163856)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 168)
    # Processing the call arguments (line 168)
    str_163859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 32), 'str', 'WD incorrect')
    # Processing the call keyword arguments (line 168)
    kwargs_163860 = {}
    # Getting the type of 'problems' (line 168)
    problems_163857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'problems', False)
    # Obtaining the member 'append' of a type (line 168)
    append_163858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), problems_163857, 'append')
    # Calling append(args, kwargs) (line 168)
    append_call_result_163861 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), append_163858, *[str_163859], **kwargs_163860)
    
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 160)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_163862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 15), 'int')
    # Getting the type of 'I' (line 169)
    I_163863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'I')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___163864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 13), I_163863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_163865 = invoke(stypy.reporting.localization.Localization(__file__, 169, 13), getitem___163864, int_163862)
    
    int_163866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 21), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_163867 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 13), '==', subscript_call_result_163865, int_163866)
    
    # Testing the type of an if condition (line 169)
    if_condition_163868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 13), result_eq_163867)
    # Assigning a type to the variable 'if_condition_163868' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'if_condition_163868', if_condition_163868)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 170)
    # Processing the call arguments (line 170)
    str_163871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 28), 'str', 'Error in derivatives')
    # Processing the call keyword arguments (line 170)
    kwargs_163872 = {}
    # Getting the type of 'problems' (line 170)
    problems_163869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'problems', False)
    # Obtaining the member 'append' of a type (line 170)
    append_163870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), problems_163869, 'append')
    # Calling append(args, kwargs) (line 170)
    append_call_result_163873 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), append_163870, *[str_163871], **kwargs_163872)
    
    # SSA branch for the else part of an if statement (line 169)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_163874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 15), 'int')
    # Getting the type of 'I' (line 171)
    I_163875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 13), 'I')
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___163876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 13), I_163875, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_163877 = invoke(stypy.reporting.localization.Localization(__file__, 171, 13), getitem___163876, int_163874)
    
    int_163878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 21), 'int')
    # Applying the binary operator '==' (line 171)
    result_eq_163879 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 13), '==', subscript_call_result_163877, int_163878)
    
    # Testing the type of an if condition (line 171)
    if_condition_163880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 13), result_eq_163879)
    # Assigning a type to the variable 'if_condition_163880' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 13), 'if_condition_163880', if_condition_163880)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 172)
    # Processing the call arguments (line 172)
    str_163883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 28), 'str', 'Error occurred in callback')
    # Processing the call keyword arguments (line 172)
    kwargs_163884 = {}
    # Getting the type of 'problems' (line 172)
    problems_163881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'problems', False)
    # Obtaining the member 'append' of a type (line 172)
    append_163882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), problems_163881, 'append')
    # Calling append(args, kwargs) (line 172)
    append_call_result_163885 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), append_163882, *[str_163883], **kwargs_163884)
    
    # SSA branch for the else part of an if statement (line 171)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_163886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 15), 'int')
    # Getting the type of 'I' (line 173)
    I_163887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'I')
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___163888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 13), I_163887, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_163889 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), getitem___163888, int_163886)
    
    int_163890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 21), 'int')
    # Applying the binary operator '==' (line 173)
    result_eq_163891 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 13), '==', subscript_call_result_163889, int_163890)
    
    # Testing the type of an if condition (line 173)
    if_condition_163892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 13), result_eq_163891)
    # Assigning a type to the variable 'if_condition_163892' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'if_condition_163892', if_condition_163892)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 174)
    # Processing the call arguments (line 174)
    str_163895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 28), 'str', 'Numerical error detected')
    # Processing the call keyword arguments (line 174)
    kwargs_163896 = {}
    # Getting the type of 'problems' (line 174)
    problems_163893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'problems', False)
    # Obtaining the member 'append' of a type (line 174)
    append_163894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), problems_163893, 'append')
    # Calling append(args, kwargs) (line 174)
    append_call_result_163897 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), append_163894, *[str_163895], **kwargs_163896)
    
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'problems' (line 176)
    problems_163898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'problems')
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', problems_163898)
    # SSA branch for the else part of an if statement (line 124)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_163899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    # Getting the type of 'stopreason' (line 179)
    stopreason_163900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'stopreason')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 15), list_163899, stopreason_163900)
    
    # Assigning a type to the variable 'stypy_return_type' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'stypy_return_type', list_163899)
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_report_error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_report_error' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_163901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163901)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_report_error'
    return stypy_return_type_163901

# Assigning a type to the variable '_report_error' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), '_report_error', _report_error)
# Declaration of the 'Data' class

class Data(object, ):
    str_163902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, (-1)), 'str', "\n    The data to fit.\n\n    Parameters\n    ----------\n    x : array_like\n        Observed data for the independent variable of the regression\n    y : array_like, optional\n        If array-like, observed data for the dependent variable of the\n        regression. A scalar input implies that the model to be used on\n        the data is implicit.\n    we : array_like, optional\n        If `we` is a scalar, then that value is used for all data points (and\n        all dimensions of the response variable).\n        If `we` is a rank-1 array of length q (the dimensionality of the\n        response variable), then this vector is the diagonal of the covariant\n        weighting matrix for all data points.\n        If `we` is a rank-1 array of length n (the number of data points), then\n        the i'th element is the weight for the i'th response variable\n        observation (single-dimensional only).\n        If `we` is a rank-2 array of shape (q, q), then this is the full\n        covariant weighting matrix broadcast to each observation.\n        If `we` is a rank-2 array of shape (q, n), then `we[:,i]` is the\n        diagonal of the covariant weighting matrix for the i'th observation.\n        If `we` is a rank-3 array of shape (q, q, n), then `we[:,:,i]` is the\n        full specification of the covariant weighting matrix for each\n        observation.\n        If the fit is implicit, then only a positive scalar value is used.\n    wd : array_like, optional\n        If `wd` is a scalar, then that value is used for all data points\n        (and all dimensions of the input variable). If `wd` = 0, then the\n        covariant weighting matrix for each observation is set to the identity\n        matrix (so each dimension of each observation has the same weight).\n        If `wd` is a rank-1 array of length m (the dimensionality of the input\n        variable), then this vector is the diagonal of the covariant weighting\n        matrix for all data points.\n        If `wd` is a rank-1 array of length n (the number of data points), then\n        the i'th element is the weight for the i'th input variable observation\n        (single-dimensional only).\n        If `wd` is a rank-2 array of shape (m, m), then this is the full\n        covariant weighting matrix broadcast to each observation.\n        If `wd` is a rank-2 array of shape (m, n), then `wd[:,i]` is the\n        diagonal of the covariant weighting matrix for the i'th observation.\n        If `wd` is a rank-3 array of shape (m, m, n), then `wd[:,:,i]` is the\n        full specification of the covariant weighting matrix for each\n        observation.\n    fix : array_like of ints, optional\n        The `fix` argument is the same as ifixx in the class ODR. It is an\n        array of integers with the same shape as data.x that determines which\n        input observations are treated as fixed. One can use a sequence of\n        length m (the dimensionality of the input observations) to fix some\n        dimensions for all observations. A value of 0 fixes the observation,\n        a value > 0 makes it free.\n    meta : dict, optional\n        Free-form dictionary for metadata.\n\n    Notes\n    -----\n    Each argument is attached to the member of the instance of the same name.\n    The structures of `x` and `y` are described in the Model class docstring.\n    If `y` is an integer, then the Data instance can only be used to fit with\n    implicit models where the dimensionality of the response is equal to the\n    specified value of `y`.\n\n    The `we` argument weights the effect a deviation in the response variable\n    has on the fit.  The `wd` argument weights the effect a deviation in the\n    input variable has on the fit. To handle multidimensional inputs and\n    responses easily, the structure of these arguments has the n'th\n    dimensional axis first. These arguments heavily use the structured\n    arguments feature of ODRPACK to conveniently and flexibly support all\n    options. See the ODRPACK User's Guide for a full explanation of how these\n    weights are used in the algorithm. Basically, a higher value of the weight\n    for a particular data point makes a deviation at that point more\n    detrimental to the fit.\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 260)
        None_163903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 28), 'None')
        # Getting the type of 'None' (line 260)
        None_163904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'None')
        # Getting the type of 'None' (line 260)
        None_163905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 46), 'None')
        # Getting the type of 'None' (line 260)
        None_163906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 56), 'None')
        
        # Obtaining an instance of the builtin type 'dict' (line 260)
        dict_163907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 67), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 260)
        
        defaults = [None_163903, None_163904, None_163905, None_163906, dict_163907]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Data.__init__', ['x', 'y', 'we', 'wd', 'fix', 'meta'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'we', 'wd', 'fix', 'meta'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 261):
        
        # Assigning a Call to a Attribute (line 261):
        
        # Call to _conv(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'x' (line 261)
        x_163909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'x', False)
        # Processing the call keyword arguments (line 261)
        kwargs_163910 = {}
        # Getting the type of '_conv' (line 261)
        _conv_163908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 17), '_conv', False)
        # Calling _conv(args, kwargs) (line 261)
        _conv_call_result_163911 = invoke(stypy.reporting.localization.Localization(__file__, 261, 17), _conv_163908, *[x_163909], **kwargs_163910)
        
        # Getting the type of 'self' (line 261)
        self_163912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self')
        # Setting the type of the member 'x' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_163912, 'x', _conv_call_result_163911)
        
        
        
        # Call to isinstance(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'self' (line 263)
        self_163914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 26), 'self', False)
        # Obtaining the member 'x' of a type (line 263)
        x_163915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 26), self_163914, 'x')
        # Getting the type of 'numpy' (line 263)
        numpy_163916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 34), 'numpy', False)
        # Obtaining the member 'ndarray' of a type (line 263)
        ndarray_163917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 34), numpy_163916, 'ndarray')
        # Processing the call keyword arguments (line 263)
        kwargs_163918 = {}
        # Getting the type of 'isinstance' (line 263)
        isinstance_163913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 263)
        isinstance_call_result_163919 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), isinstance_163913, *[x_163915, ndarray_163917], **kwargs_163918)
        
        # Applying the 'not' unary operator (line 263)
        result_not__163920 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 11), 'not', isinstance_call_result_163919)
        
        # Testing the type of an if condition (line 263)
        if_condition_163921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 8), result_not__163920)
        # Assigning a type to the variable 'if_condition_163921' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'if_condition_163921', if_condition_163921)
        # SSA begins for if statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 264)
        # Processing the call arguments (line 264)
        
        # Call to format(...): (line 264)
        # Processing the call keyword arguments (line 264)
        
        # Call to type(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'self' (line 266)
        self_163926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 30), 'self', False)
        # Obtaining the member 'x' of a type (line 266)
        x_163927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 30), self_163926, 'x')
        # Processing the call keyword arguments (line 266)
        kwargs_163928 = {}
        # Getting the type of 'type' (line 266)
        type_163925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'type', False)
        # Calling type(args, kwargs) (line 266)
        type_call_result_163929 = invoke(stypy.reporting.localization.Localization(__file__, 266, 25), type_163925, *[x_163927], **kwargs_163928)
        
        # Obtaining the member '__name__' of a type (line 266)
        name___163930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 25), type_call_result_163929, '__name__')
        keyword_163931 = name___163930
        kwargs_163932 = {'name': keyword_163931}
        str_163923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 30), 'str', "Expected an 'ndarray' of data for 'x', but instead got data of type '{name}'")
        # Obtaining the member 'format' of a type (line 264)
        format_163924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 30), str_163923, 'format')
        # Calling format(args, kwargs) (line 264)
        format_call_result_163933 = invoke(stypy.reporting.localization.Localization(__file__, 264, 30), format_163924, *[], **kwargs_163932)
        
        # Processing the call keyword arguments (line 264)
        kwargs_163934 = {}
        # Getting the type of 'ValueError' (line 264)
        ValueError_163922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 264)
        ValueError_call_result_163935 = invoke(stypy.reporting.localization.Localization(__file__, 264, 18), ValueError_163922, *[format_call_result_163933], **kwargs_163934)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 264, 12), ValueError_call_result_163935, 'raise parameter', BaseException)
        # SSA join for if statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 268):
        
        # Assigning a Call to a Attribute (line 268):
        
        # Call to _conv(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'y' (line 268)
        y_163937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 23), 'y', False)
        # Processing the call keyword arguments (line 268)
        kwargs_163938 = {}
        # Getting the type of '_conv' (line 268)
        _conv_163936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 17), '_conv', False)
        # Calling _conv(args, kwargs) (line 268)
        _conv_call_result_163939 = invoke(stypy.reporting.localization.Localization(__file__, 268, 17), _conv_163936, *[y_163937], **kwargs_163938)
        
        # Getting the type of 'self' (line 268)
        self_163940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self')
        # Setting the type of the member 'y' of a type (line 268)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_163940, 'y', _conv_call_result_163939)
        
        # Assigning a Call to a Attribute (line 269):
        
        # Assigning a Call to a Attribute (line 269):
        
        # Call to _conv(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'we' (line 269)
        we_163942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'we', False)
        # Processing the call keyword arguments (line 269)
        kwargs_163943 = {}
        # Getting the type of '_conv' (line 269)
        _conv_163941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 18), '_conv', False)
        # Calling _conv(args, kwargs) (line 269)
        _conv_call_result_163944 = invoke(stypy.reporting.localization.Localization(__file__, 269, 18), _conv_163941, *[we_163942], **kwargs_163943)
        
        # Getting the type of 'self' (line 269)
        self_163945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Setting the type of the member 'we' of a type (line 269)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_163945, 'we', _conv_call_result_163944)
        
        # Assigning a Call to a Attribute (line 270):
        
        # Assigning a Call to a Attribute (line 270):
        
        # Call to _conv(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'wd' (line 270)
        wd_163947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 24), 'wd', False)
        # Processing the call keyword arguments (line 270)
        kwargs_163948 = {}
        # Getting the type of '_conv' (line 270)
        _conv_163946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 18), '_conv', False)
        # Calling _conv(args, kwargs) (line 270)
        _conv_call_result_163949 = invoke(stypy.reporting.localization.Localization(__file__, 270, 18), _conv_163946, *[wd_163947], **kwargs_163948)
        
        # Getting the type of 'self' (line 270)
        self_163950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Setting the type of the member 'wd' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_163950, 'wd', _conv_call_result_163949)
        
        # Assigning a Call to a Attribute (line 271):
        
        # Assigning a Call to a Attribute (line 271):
        
        # Call to _conv(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'fix' (line 271)
        fix_163952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 25), 'fix', False)
        # Processing the call keyword arguments (line 271)
        kwargs_163953 = {}
        # Getting the type of '_conv' (line 271)
        _conv_163951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), '_conv', False)
        # Calling _conv(args, kwargs) (line 271)
        _conv_call_result_163954 = invoke(stypy.reporting.localization.Localization(__file__, 271, 19), _conv_163951, *[fix_163952], **kwargs_163953)
        
        # Getting the type of 'self' (line 271)
        self_163955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'self')
        # Setting the type of the member 'fix' of a type (line 271)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), self_163955, 'fix', _conv_call_result_163954)
        
        # Assigning a Name to a Attribute (line 272):
        
        # Assigning a Name to a Attribute (line 272):
        # Getting the type of 'meta' (line 272)
        meta_163956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'meta')
        # Getting the type of 'self' (line 272)
        self_163957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member 'meta' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_163957, 'meta', meta_163956)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_meta(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_meta'
        module_type_store = module_type_store.open_function_context('set_meta', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Data.set_meta.__dict__.__setitem__('stypy_localization', localization)
        Data.set_meta.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Data.set_meta.__dict__.__setitem__('stypy_type_store', module_type_store)
        Data.set_meta.__dict__.__setitem__('stypy_function_name', 'Data.set_meta')
        Data.set_meta.__dict__.__setitem__('stypy_param_names_list', [])
        Data.set_meta.__dict__.__setitem__('stypy_varargs_param_name', None)
        Data.set_meta.__dict__.__setitem__('stypy_kwargs_param_name', 'kwds')
        Data.set_meta.__dict__.__setitem__('stypy_call_defaults', defaults)
        Data.set_meta.__dict__.__setitem__('stypy_call_varargs', varargs)
        Data.set_meta.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Data.set_meta.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Data.set_meta', [], None, 'kwds', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_meta', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_meta(...)' code ##################

        str_163958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, (-1)), 'str', ' Update the metadata dictionary with the keywords and data provided\n        by keywords.\n\n        Examples\n        --------\n        ::\n\n            data.set_meta(lab="Ph 7; Lab 26", title="Ag110 + Ag108 Decay")\n        ')
        
        # Call to update(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'kwds' (line 285)
        kwds_163962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'kwds', False)
        # Processing the call keyword arguments (line 285)
        kwargs_163963 = {}
        # Getting the type of 'self' (line 285)
        self_163959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self', False)
        # Obtaining the member 'meta' of a type (line 285)
        meta_163960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_163959, 'meta')
        # Obtaining the member 'update' of a type (line 285)
        update_163961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), meta_163960, 'update')
        # Calling update(args, kwargs) (line 285)
        update_call_result_163964 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), update_163961, *[kwds_163962], **kwargs_163963)
        
        
        # ################# End of 'set_meta(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_meta' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_163965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_163965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_meta'
        return stypy_return_type_163965


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Data.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        Data.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Data.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Data.__getattr__.__dict__.__setitem__('stypy_function_name', 'Data.__getattr__')
        Data.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        Data.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Data.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Data.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Data.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Data.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Data.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Data.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        str_163966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', ' Dispatch attribute access to the metadata dictionary.\n        ')
        
        
        # Getting the type of 'attr' (line 290)
        attr_163967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'attr')
        # Getting the type of 'self' (line 290)
        self_163968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'self')
        # Obtaining the member 'meta' of a type (line 290)
        meta_163969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 19), self_163968, 'meta')
        # Applying the binary operator 'in' (line 290)
        result_contains_163970 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), 'in', attr_163967, meta_163969)
        
        # Testing the type of an if condition (line 290)
        if_condition_163971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), result_contains_163970)
        # Assigning a type to the variable 'if_condition_163971' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_163971', if_condition_163971)
        # SSA begins for if statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 291)
        attr_163972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 29), 'attr')
        # Getting the type of 'self' (line 291)
        self_163973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 19), 'self')
        # Obtaining the member 'meta' of a type (line 291)
        meta_163974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 19), self_163973, 'meta')
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___163975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 19), meta_163974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_163976 = invoke(stypy.reporting.localization.Localization(__file__, 291, 19), getitem___163975, attr_163972)
        
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'stypy_return_type', subscript_call_result_163976)
        # SSA branch for the else part of an if statement (line 290)
        module_type_store.open_ssa_branch('else')
        
        # Call to AttributeError(...): (line 293)
        # Processing the call arguments (line 293)
        str_163978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 33), 'str', "'%s' not in metadata")
        # Getting the type of 'attr' (line 293)
        attr_163979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 58), 'attr', False)
        # Applying the binary operator '%' (line 293)
        result_mod_163980 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 33), '%', str_163978, attr_163979)
        
        # Processing the call keyword arguments (line 293)
        kwargs_163981 = {}
        # Getting the type of 'AttributeError' (line 293)
        AttributeError_163977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 293)
        AttributeError_call_result_163982 = invoke(stypy.reporting.localization.Localization(__file__, 293, 18), AttributeError_163977, *[result_mod_163980], **kwargs_163981)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 293, 12), AttributeError_call_result_163982, 'raise parameter', BaseException)
        # SSA join for if statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_163983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_163983)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_163983


# Assigning a type to the variable 'Data' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'Data', Data)
# Declaration of the 'RealData' class
# Getting the type of 'Data' (line 296)
Data_163984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'Data')

class RealData(Data_163984, ):
    str_163985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, (-1)), 'str', "\n    The data, with weightings as actual standard deviations and/or\n    covariances.\n\n    Parameters\n    ----------\n    x : array_like\n        Observed data for the independent variable of the regression\n    y : array_like, optional\n        If array-like, observed data for the dependent variable of the\n        regression. A scalar input implies that the model to be used on\n        the data is implicit.\n    sx : array_like, optional\n        Standard deviations of `x`.\n        `sx` are standard deviations of `x` and are converted to weights by\n        dividing 1.0 by their squares.\n    sy : array_like, optional\n        Standard deviations of `y`.\n        `sy` are standard deviations of `y` and are converted to weights by\n        dividing 1.0 by their squares.\n    covx : array_like, optional\n        Covariance of `x`\n        `covx` is an array of covariance matrices of `x` and are converted to\n        weights by performing a matrix inversion on each observation's\n        covariance matrix.\n    covy : array_like, optional\n        Covariance of `y`\n        `covy` is an array of covariance matrices and are converted to\n        weights by performing a matrix inversion on each observation's\n        covariance matrix.\n    fix : array_like, optional\n        The argument and member fix is the same as Data.fix and ODR.ifixx:\n        It is an array of integers with the same shape as `x` that\n        determines which input observations are treated as fixed. One can\n        use a sequence of length m (the dimensionality of the input\n        observations) to fix some dimensions for all observations. A value\n        of 0 fixes the observation, a value > 0 makes it free.\n    meta : dict, optional\n        Free-form dictionary for metadata.\n\n    Notes\n    -----\n    The weights `wd` and `we` are computed from provided values as follows:\n\n    `sx` and `sy` are converted to weights by dividing 1.0 by their squares.\n    For example, ``wd = 1./numpy.power(`sx`, 2)``.\n\n    `covx` and `covy` are arrays of covariance matrices and are converted to\n    weights by performing a matrix inversion on each observation's covariance\n    matrix.  For example, ``we[i] = numpy.linalg.inv(covy[i])``.\n\n    These arguments follow the same structured argument conventions as wd and\n    we only restricted by their natures: `sx` and `sy` can't be rank-3, but\n    `covx` and `covy` can be.\n\n    Only set *either* `sx` or `covx` (not both). Setting both will raise an\n    exception.  Same with `sy` and `covy`.\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 357)
        None_163986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 28), 'None')
        # Getting the type of 'None' (line 357)
        None_163987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 37), 'None')
        # Getting the type of 'None' (line 357)
        None_163988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 46), 'None')
        # Getting the type of 'None' (line 357)
        None_163989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 57), 'None')
        # Getting the type of 'None' (line 357)
        None_163990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 68), 'None')
        # Getting the type of 'None' (line 358)
        None_163991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'None')
        
        # Obtaining an instance of the builtin type 'dict' (line 358)
        dict_163992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 32), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 358)
        
        defaults = [None_163986, None_163987, None_163988, None_163989, None_163990, None_163991, dict_163992]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 357, 4, False)
        # Assigning a type to the variable 'self' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RealData.__init__', ['x', 'y', 'sx', 'sy', 'covx', 'covy', 'fix', 'meta'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'sx', 'sy', 'covx', 'covy', 'fix', 'meta'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sx' (line 359)
        sx_163993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'sx')
        # Getting the type of 'None' (line 359)
        None_163994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 22), 'None')
        # Applying the binary operator 'isnot' (line 359)
        result_is_not_163995 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 12), 'isnot', sx_163993, None_163994)
        
        
        # Getting the type of 'covx' (line 359)
        covx_163996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 33), 'covx')
        # Getting the type of 'None' (line 359)
        None_163997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 45), 'None')
        # Applying the binary operator 'isnot' (line 359)
        result_is_not_163998 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 33), 'isnot', covx_163996, None_163997)
        
        # Applying the binary operator 'and' (line 359)
        result_and_keyword_163999 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 11), 'and', result_is_not_163995, result_is_not_163998)
        
        # Testing the type of an if condition (line 359)
        if_condition_164000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 8), result_and_keyword_163999)
        # Assigning a type to the variable 'if_condition_164000' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'if_condition_164000', if_condition_164000)
        # SSA begins for if statement (line 359)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 360)
        # Processing the call arguments (line 360)
        str_164002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 29), 'str', 'cannot set both sx and covx')
        # Processing the call keyword arguments (line 360)
        kwargs_164003 = {}
        # Getting the type of 'ValueError' (line 360)
        ValueError_164001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 360)
        ValueError_call_result_164004 = invoke(stypy.reporting.localization.Localization(__file__, 360, 18), ValueError_164001, *[str_164002], **kwargs_164003)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 360, 12), ValueError_call_result_164004, 'raise parameter', BaseException)
        # SSA join for if statement (line 359)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sy' (line 361)
        sy_164005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'sy')
        # Getting the type of 'None' (line 361)
        None_164006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 22), 'None')
        # Applying the binary operator 'isnot' (line 361)
        result_is_not_164007 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 12), 'isnot', sy_164005, None_164006)
        
        
        # Getting the type of 'covy' (line 361)
        covy_164008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 33), 'covy')
        # Getting the type of 'None' (line 361)
        None_164009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 45), 'None')
        # Applying the binary operator 'isnot' (line 361)
        result_is_not_164010 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 33), 'isnot', covy_164008, None_164009)
        
        # Applying the binary operator 'and' (line 361)
        result_and_keyword_164011 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 11), 'and', result_is_not_164007, result_is_not_164010)
        
        # Testing the type of an if condition (line 361)
        if_condition_164012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 8), result_and_keyword_164011)
        # Assigning a type to the variable 'if_condition_164012' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'if_condition_164012', if_condition_164012)
        # SSA begins for if statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 362)
        # Processing the call arguments (line 362)
        str_164014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 29), 'str', 'cannot set both sy and covy')
        # Processing the call keyword arguments (line 362)
        kwargs_164015 = {}
        # Getting the type of 'ValueError' (line 362)
        ValueError_164013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 362)
        ValueError_call_result_164016 = invoke(stypy.reporting.localization.Localization(__file__, 362, 18), ValueError_164013, *[str_164014], **kwargs_164015)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 362, 12), ValueError_call_result_164016, 'raise parameter', BaseException)
        # SSA join for if statement (line 361)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Attribute (line 365):
        
        # Assigning a Dict to a Attribute (line 365):
        
        # Obtaining an instance of the builtin type 'dict' (line 365)
        dict_164017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 365)
        
        # Getting the type of 'self' (line 365)
        self_164018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'self')
        # Setting the type of the member '_ga_flags' of a type (line 365)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), self_164018, '_ga_flags', dict_164017)
        
        # Type idiom detected: calculating its left and rigth part (line 366)
        # Getting the type of 'sx' (line 366)
        sx_164019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'sx')
        # Getting the type of 'None' (line 366)
        None_164020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 21), 'None')
        
        (may_be_164021, more_types_in_union_164022) = may_not_be_none(sx_164019, None_164020)

        if may_be_164021:

            if more_types_in_union_164022:
                # Runtime conditional SSA (line 366)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Subscript (line 367):
            
            # Assigning a Str to a Subscript (line 367):
            str_164023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 35), 'str', 'sx')
            # Getting the type of 'self' (line 367)
            self_164024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'self')
            # Obtaining the member '_ga_flags' of a type (line 367)
            _ga_flags_164025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), self_164024, '_ga_flags')
            str_164026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 27), 'str', 'wd')
            # Storing an element on a container (line 367)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 12), _ga_flags_164025, (str_164026, str_164023))

            if more_types_in_union_164022:
                # Runtime conditional SSA for else branch (line 366)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_164021) or more_types_in_union_164022):
            
            # Assigning a Str to a Subscript (line 369):
            
            # Assigning a Str to a Subscript (line 369):
            str_164027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 35), 'str', 'covx')
            # Getting the type of 'self' (line 369)
            self_164028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'self')
            # Obtaining the member '_ga_flags' of a type (line 369)
            _ga_flags_164029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), self_164028, '_ga_flags')
            str_164030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 27), 'str', 'wd')
            # Storing an element on a container (line 369)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 12), _ga_flags_164029, (str_164030, str_164027))

            if (may_be_164021 and more_types_in_union_164022):
                # SSA join for if statement (line 366)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 370)
        # Getting the type of 'sy' (line 370)
        sy_164031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'sy')
        # Getting the type of 'None' (line 370)
        None_164032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 21), 'None')
        
        (may_be_164033, more_types_in_union_164034) = may_not_be_none(sy_164031, None_164032)

        if may_be_164033:

            if more_types_in_union_164034:
                # Runtime conditional SSA (line 370)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Subscript (line 371):
            
            # Assigning a Str to a Subscript (line 371):
            str_164035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 35), 'str', 'sy')
            # Getting the type of 'self' (line 371)
            self_164036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'self')
            # Obtaining the member '_ga_flags' of a type (line 371)
            _ga_flags_164037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 12), self_164036, '_ga_flags')
            str_164038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 27), 'str', 'we')
            # Storing an element on a container (line 371)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 12), _ga_flags_164037, (str_164038, str_164035))

            if more_types_in_union_164034:
                # Runtime conditional SSA for else branch (line 370)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_164033) or more_types_in_union_164034):
            
            # Assigning a Str to a Subscript (line 373):
            
            # Assigning a Str to a Subscript (line 373):
            str_164039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 35), 'str', 'covy')
            # Getting the type of 'self' (line 373)
            self_164040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'self')
            # Obtaining the member '_ga_flags' of a type (line 373)
            _ga_flags_164041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), self_164040, '_ga_flags')
            str_164042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 27), 'str', 'we')
            # Storing an element on a container (line 373)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 12), _ga_flags_164041, (str_164042, str_164039))

            if (may_be_164033 and more_types_in_union_164034):
                # SSA join for if statement (line 370)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 375):
        
        # Assigning a Call to a Attribute (line 375):
        
        # Call to _conv(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'x' (line 375)
        x_164044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 23), 'x', False)
        # Processing the call keyword arguments (line 375)
        kwargs_164045 = {}
        # Getting the type of '_conv' (line 375)
        _conv_164043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 17), '_conv', False)
        # Calling _conv(args, kwargs) (line 375)
        _conv_call_result_164046 = invoke(stypy.reporting.localization.Localization(__file__, 375, 17), _conv_164043, *[x_164044], **kwargs_164045)
        
        # Getting the type of 'self' (line 375)
        self_164047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'self')
        # Setting the type of the member 'x' of a type (line 375)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), self_164047, 'x', _conv_call_result_164046)
        
        
        
        # Call to isinstance(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'self' (line 377)
        self_164049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 26), 'self', False)
        # Obtaining the member 'x' of a type (line 377)
        x_164050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 26), self_164049, 'x')
        # Getting the type of 'numpy' (line 377)
        numpy_164051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 34), 'numpy', False)
        # Obtaining the member 'ndarray' of a type (line 377)
        ndarray_164052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 34), numpy_164051, 'ndarray')
        # Processing the call keyword arguments (line 377)
        kwargs_164053 = {}
        # Getting the type of 'isinstance' (line 377)
        isinstance_164048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 377)
        isinstance_call_result_164054 = invoke(stypy.reporting.localization.Localization(__file__, 377, 15), isinstance_164048, *[x_164050, ndarray_164052], **kwargs_164053)
        
        # Applying the 'not' unary operator (line 377)
        result_not__164055 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 11), 'not', isinstance_call_result_164054)
        
        # Testing the type of an if condition (line 377)
        if_condition_164056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 8), result_not__164055)
        # Assigning a type to the variable 'if_condition_164056' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'if_condition_164056', if_condition_164056)
        # SSA begins for if statement (line 377)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 378)
        # Processing the call arguments (line 378)
        
        # Call to format(...): (line 378)
        # Processing the call keyword arguments (line 378)
        
        # Call to type(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'self' (line 380)
        self_164061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 30), 'self', False)
        # Obtaining the member 'x' of a type (line 380)
        x_164062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 30), self_164061, 'x')
        # Processing the call keyword arguments (line 380)
        kwargs_164063 = {}
        # Getting the type of 'type' (line 380)
        type_164060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 25), 'type', False)
        # Calling type(args, kwargs) (line 380)
        type_call_result_164064 = invoke(stypy.reporting.localization.Localization(__file__, 380, 25), type_164060, *[x_164062], **kwargs_164063)
        
        # Obtaining the member '__name__' of a type (line 380)
        name___164065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 25), type_call_result_164064, '__name__')
        keyword_164066 = name___164065
        kwargs_164067 = {'name': keyword_164066}
        str_164058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 30), 'str', "Expected an 'ndarray' of data for 'x', but instead got data of type '{name}'")
        # Obtaining the member 'format' of a type (line 378)
        format_164059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 30), str_164058, 'format')
        # Calling format(args, kwargs) (line 378)
        format_call_result_164068 = invoke(stypy.reporting.localization.Localization(__file__, 378, 30), format_164059, *[], **kwargs_164067)
        
        # Processing the call keyword arguments (line 378)
        kwargs_164069 = {}
        # Getting the type of 'ValueError' (line 378)
        ValueError_164057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 378)
        ValueError_call_result_164070 = invoke(stypy.reporting.localization.Localization(__file__, 378, 18), ValueError_164057, *[format_call_result_164068], **kwargs_164069)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 378, 12), ValueError_call_result_164070, 'raise parameter', BaseException)
        # SSA join for if statement (line 377)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 382):
        
        # Assigning a Call to a Attribute (line 382):
        
        # Call to _conv(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'y' (line 382)
        y_164072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'y', False)
        # Processing the call keyword arguments (line 382)
        kwargs_164073 = {}
        # Getting the type of '_conv' (line 382)
        _conv_164071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 17), '_conv', False)
        # Calling _conv(args, kwargs) (line 382)
        _conv_call_result_164074 = invoke(stypy.reporting.localization.Localization(__file__, 382, 17), _conv_164071, *[y_164072], **kwargs_164073)
        
        # Getting the type of 'self' (line 382)
        self_164075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self')
        # Setting the type of the member 'y' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_164075, 'y', _conv_call_result_164074)
        
        # Assigning a Call to a Attribute (line 383):
        
        # Assigning a Call to a Attribute (line 383):
        
        # Call to _conv(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'sx' (line 383)
        sx_164077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 24), 'sx', False)
        # Processing the call keyword arguments (line 383)
        kwargs_164078 = {}
        # Getting the type of '_conv' (line 383)
        _conv_164076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 18), '_conv', False)
        # Calling _conv(args, kwargs) (line 383)
        _conv_call_result_164079 = invoke(stypy.reporting.localization.Localization(__file__, 383, 18), _conv_164076, *[sx_164077], **kwargs_164078)
        
        # Getting the type of 'self' (line 383)
        self_164080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member 'sx' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_164080, 'sx', _conv_call_result_164079)
        
        # Assigning a Call to a Attribute (line 384):
        
        # Assigning a Call to a Attribute (line 384):
        
        # Call to _conv(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'sy' (line 384)
        sy_164082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'sy', False)
        # Processing the call keyword arguments (line 384)
        kwargs_164083 = {}
        # Getting the type of '_conv' (line 384)
        _conv_164081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), '_conv', False)
        # Calling _conv(args, kwargs) (line 384)
        _conv_call_result_164084 = invoke(stypy.reporting.localization.Localization(__file__, 384, 18), _conv_164081, *[sy_164082], **kwargs_164083)
        
        # Getting the type of 'self' (line 384)
        self_164085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self')
        # Setting the type of the member 'sy' of a type (line 384)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_164085, 'sy', _conv_call_result_164084)
        
        # Assigning a Call to a Attribute (line 385):
        
        # Assigning a Call to a Attribute (line 385):
        
        # Call to _conv(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'covx' (line 385)
        covx_164087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 26), 'covx', False)
        # Processing the call keyword arguments (line 385)
        kwargs_164088 = {}
        # Getting the type of '_conv' (line 385)
        _conv_164086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), '_conv', False)
        # Calling _conv(args, kwargs) (line 385)
        _conv_call_result_164089 = invoke(stypy.reporting.localization.Localization(__file__, 385, 20), _conv_164086, *[covx_164087], **kwargs_164088)
        
        # Getting the type of 'self' (line 385)
        self_164090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self')
        # Setting the type of the member 'covx' of a type (line 385)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_164090, 'covx', _conv_call_result_164089)
        
        # Assigning a Call to a Attribute (line 386):
        
        # Assigning a Call to a Attribute (line 386):
        
        # Call to _conv(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'covy' (line 386)
        covy_164092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 26), 'covy', False)
        # Processing the call keyword arguments (line 386)
        kwargs_164093 = {}
        # Getting the type of '_conv' (line 386)
        _conv_164091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 20), '_conv', False)
        # Calling _conv(args, kwargs) (line 386)
        _conv_call_result_164094 = invoke(stypy.reporting.localization.Localization(__file__, 386, 20), _conv_164091, *[covy_164092], **kwargs_164093)
        
        # Getting the type of 'self' (line 386)
        self_164095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'self')
        # Setting the type of the member 'covy' of a type (line 386)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), self_164095, 'covy', _conv_call_result_164094)
        
        # Assigning a Call to a Attribute (line 387):
        
        # Assigning a Call to a Attribute (line 387):
        
        # Call to _conv(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'fix' (line 387)
        fix_164097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'fix', False)
        # Processing the call keyword arguments (line 387)
        kwargs_164098 = {}
        # Getting the type of '_conv' (line 387)
        _conv_164096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 19), '_conv', False)
        # Calling _conv(args, kwargs) (line 387)
        _conv_call_result_164099 = invoke(stypy.reporting.localization.Localization(__file__, 387, 19), _conv_164096, *[fix_164097], **kwargs_164098)
        
        # Getting the type of 'self' (line 387)
        self_164100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'self')
        # Setting the type of the member 'fix' of a type (line 387)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), self_164100, 'fix', _conv_call_result_164099)
        
        # Assigning a Name to a Attribute (line 388):
        
        # Assigning a Name to a Attribute (line 388):
        # Getting the type of 'meta' (line 388)
        meta_164101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'meta')
        # Getting the type of 'self' (line 388)
        self_164102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self')
        # Setting the type of the member 'meta' of a type (line 388)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_164102, 'meta', meta_164101)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _sd2wt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sd2wt'
        module_type_store = module_type_store.open_function_context('_sd2wt', 390, 4, False)
        # Assigning a type to the variable 'self' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RealData._sd2wt.__dict__.__setitem__('stypy_localization', localization)
        RealData._sd2wt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RealData._sd2wt.__dict__.__setitem__('stypy_type_store', module_type_store)
        RealData._sd2wt.__dict__.__setitem__('stypy_function_name', 'RealData._sd2wt')
        RealData._sd2wt.__dict__.__setitem__('stypy_param_names_list', ['sd'])
        RealData._sd2wt.__dict__.__setitem__('stypy_varargs_param_name', None)
        RealData._sd2wt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RealData._sd2wt.__dict__.__setitem__('stypy_call_defaults', defaults)
        RealData._sd2wt.__dict__.__setitem__('stypy_call_varargs', varargs)
        RealData._sd2wt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RealData._sd2wt.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RealData._sd2wt', ['sd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sd2wt', localization, ['sd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sd2wt(...)' code ##################

        str_164103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, (-1)), 'str', ' Convert standard deviation to weights.\n        ')
        float_164104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 15), 'float')
        
        # Call to power(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'sd' (line 394)
        sd_164107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 30), 'sd', False)
        int_164108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 34), 'int')
        # Processing the call keyword arguments (line 394)
        kwargs_164109 = {}
        # Getting the type of 'numpy' (line 394)
        numpy_164105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 18), 'numpy', False)
        # Obtaining the member 'power' of a type (line 394)
        power_164106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 18), numpy_164105, 'power')
        # Calling power(args, kwargs) (line 394)
        power_call_result_164110 = invoke(stypy.reporting.localization.Localization(__file__, 394, 18), power_164106, *[sd_164107, int_164108], **kwargs_164109)
        
        # Applying the binary operator 'div' (line 394)
        result_div_164111 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 15), 'div', float_164104, power_call_result_164110)
        
        # Assigning a type to the variable 'stypy_return_type' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'stypy_return_type', result_div_164111)
        
        # ################# End of '_sd2wt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sd2wt' in the type store
        # Getting the type of 'stypy_return_type' (line 390)
        stypy_return_type_164112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_164112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sd2wt'
        return stypy_return_type_164112


    @norecursion
    def _cov2wt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cov2wt'
        module_type_store = module_type_store.open_function_context('_cov2wt', 396, 4, False)
        # Assigning a type to the variable 'self' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RealData._cov2wt.__dict__.__setitem__('stypy_localization', localization)
        RealData._cov2wt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RealData._cov2wt.__dict__.__setitem__('stypy_type_store', module_type_store)
        RealData._cov2wt.__dict__.__setitem__('stypy_function_name', 'RealData._cov2wt')
        RealData._cov2wt.__dict__.__setitem__('stypy_param_names_list', ['cov'])
        RealData._cov2wt.__dict__.__setitem__('stypy_varargs_param_name', None)
        RealData._cov2wt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RealData._cov2wt.__dict__.__setitem__('stypy_call_defaults', defaults)
        RealData._cov2wt.__dict__.__setitem__('stypy_call_varargs', varargs)
        RealData._cov2wt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RealData._cov2wt.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RealData._cov2wt', ['cov'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cov2wt', localization, ['cov'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cov2wt(...)' code ##################

        str_164113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, (-1)), 'str', ' Convert covariance matrix(-ices) to weights.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 400, 8))
        
        # 'from numpy.dual import inv' statement (line 400)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
        import_164114 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 400, 8), 'numpy.dual')

        if (type(import_164114) is not StypyTypeError):

            if (import_164114 != 'pyd_module'):
                __import__(import_164114)
                sys_modules_164115 = sys.modules[import_164114]
                import_from_module(stypy.reporting.localization.Localization(__file__, 400, 8), 'numpy.dual', sys_modules_164115.module_type_store, module_type_store, ['inv'])
                nest_module(stypy.reporting.localization.Localization(__file__, 400, 8), __file__, sys_modules_164115, sys_modules_164115.module_type_store, module_type_store)
            else:
                from numpy.dual import inv

                import_from_module(stypy.reporting.localization.Localization(__file__, 400, 8), 'numpy.dual', None, module_type_store, ['inv'], [inv])

        else:
            # Assigning a type to the variable 'numpy.dual' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'numpy.dual', import_164114)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')
        
        
        
        
        # Call to len(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'cov' (line 402)
        cov_164117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'cov', False)
        # Obtaining the member 'shape' of a type (line 402)
        shape_164118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 15), cov_164117, 'shape')
        # Processing the call keyword arguments (line 402)
        kwargs_164119 = {}
        # Getting the type of 'len' (line 402)
        len_164116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 11), 'len', False)
        # Calling len(args, kwargs) (line 402)
        len_call_result_164120 = invoke(stypy.reporting.localization.Localization(__file__, 402, 11), len_164116, *[shape_164118], **kwargs_164119)
        
        int_164121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 29), 'int')
        # Applying the binary operator '==' (line 402)
        result_eq_164122 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 11), '==', len_call_result_164120, int_164121)
        
        # Testing the type of an if condition (line 402)
        if_condition_164123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 8), result_eq_164122)
        # Assigning a type to the variable 'if_condition_164123' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'if_condition_164123', if_condition_164123)
        # SSA begins for if statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to inv(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'cov' (line 403)
        cov_164125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 23), 'cov', False)
        # Processing the call keyword arguments (line 403)
        kwargs_164126 = {}
        # Getting the type of 'inv' (line 403)
        inv_164124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'inv', False)
        # Calling inv(args, kwargs) (line 403)
        inv_call_result_164127 = invoke(stypy.reporting.localization.Localization(__file__, 403, 19), inv_164124, *[cov_164125], **kwargs_164126)
        
        # Assigning a type to the variable 'stypy_return_type' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'stypy_return_type', inv_call_result_164127)
        # SSA branch for the else part of an if statement (line 402)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 405):
        
        # Assigning a Call to a Name (line 405):
        
        # Call to zeros(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'cov' (line 405)
        cov_164130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 34), 'cov', False)
        # Obtaining the member 'shape' of a type (line 405)
        shape_164131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 34), cov_164130, 'shape')
        # Getting the type of 'float' (line 405)
        float_164132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 45), 'float', False)
        # Processing the call keyword arguments (line 405)
        kwargs_164133 = {}
        # Getting the type of 'numpy' (line 405)
        numpy_164128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 22), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 405)
        zeros_164129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 22), numpy_164128, 'zeros')
        # Calling zeros(args, kwargs) (line 405)
        zeros_call_result_164134 = invoke(stypy.reporting.localization.Localization(__file__, 405, 22), zeros_164129, *[shape_164131, float_164132], **kwargs_164133)
        
        # Assigning a type to the variable 'weights' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'weights', zeros_call_result_164134)
        
        
        # Call to range(...): (line 407)
        # Processing the call arguments (line 407)
        
        # Obtaining the type of the subscript
        int_164136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 37), 'int')
        # Getting the type of 'cov' (line 407)
        cov_164137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 27), 'cov', False)
        # Obtaining the member 'shape' of a type (line 407)
        shape_164138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 27), cov_164137, 'shape')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___164139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 27), shape_164138, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_164140 = invoke(stypy.reporting.localization.Localization(__file__, 407, 27), getitem___164139, int_164136)
        
        # Processing the call keyword arguments (line 407)
        kwargs_164141 = {}
        # Getting the type of 'range' (line 407)
        range_164135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'range', False)
        # Calling range(args, kwargs) (line 407)
        range_call_result_164142 = invoke(stypy.reporting.localization.Localization(__file__, 407, 21), range_164135, *[subscript_call_result_164140], **kwargs_164141)
        
        # Testing the type of a for loop iterable (line 407)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 407, 12), range_call_result_164142)
        # Getting the type of the for loop variable (line 407)
        for_loop_var_164143 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 407, 12), range_call_result_164142)
        # Assigning a type to the variable 'i' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'i', for_loop_var_164143)
        # SSA begins for a for statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 408):
        
        # Assigning a Call to a Subscript (line 408):
        
        # Call to inv(...): (line 408)
        # Processing the call arguments (line 408)
        
        # Obtaining the type of the subscript
        slice_164145 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 408, 37), None, None, None)
        slice_164146 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 408, 37), None, None, None)
        # Getting the type of 'i' (line 408)
        i_164147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'i', False)
        # Getting the type of 'cov' (line 408)
        cov_164148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 37), 'cov', False)
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___164149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 37), cov_164148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 408)
        subscript_call_result_164150 = invoke(stypy.reporting.localization.Localization(__file__, 408, 37), getitem___164149, (slice_164145, slice_164146, i_164147))
        
        # Processing the call keyword arguments (line 408)
        kwargs_164151 = {}
        # Getting the type of 'inv' (line 408)
        inv_164144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 33), 'inv', False)
        # Calling inv(args, kwargs) (line 408)
        inv_call_result_164152 = invoke(stypy.reporting.localization.Localization(__file__, 408, 33), inv_164144, *[subscript_call_result_164150], **kwargs_164151)
        
        # Getting the type of 'weights' (line 408)
        weights_164153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'weights')
        slice_164154 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 408, 16), None, None, None)
        slice_164155 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 408, 16), None, None, None)
        # Getting the type of 'i' (line 408)
        i_164156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'i')
        # Storing an element on a container (line 408)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 16), weights_164153, ((slice_164154, slice_164155, i_164156), inv_call_result_164152))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'weights' (line 410)
        weights_164157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 'weights')
        # Assigning a type to the variable 'stypy_return_type' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'stypy_return_type', weights_164157)
        # SSA join for if statement (line 402)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_cov2wt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cov2wt' in the type store
        # Getting the type of 'stypy_return_type' (line 396)
        stypy_return_type_164158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_164158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cov2wt'
        return stypy_return_type_164158


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 412, 4, False)
        # Assigning a type to the variable 'self' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RealData.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        RealData.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RealData.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        RealData.__getattr__.__dict__.__setitem__('stypy_function_name', 'RealData.__getattr__')
        RealData.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        RealData.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        RealData.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RealData.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        RealData.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        RealData.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RealData.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RealData.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        # Assigning a Dict to a Name (line 413):
        
        # Assigning a Dict to a Name (line 413):
        
        # Obtaining an instance of the builtin type 'dict' (line 413)
        dict_164159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 413)
        # Adding element type (key, value) (line 413)
        
        # Obtaining an instance of the builtin type 'tuple' (line 413)
        tuple_164160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 413)
        # Adding element type (line 413)
        str_164161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 23), 'str', 'wd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 23), tuple_164160, str_164161)
        # Adding element type (line 413)
        str_164162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 29), 'str', 'sx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 23), tuple_164160, str_164162)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 413)
        tuple_164163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 413)
        # Adding element type (line 413)
        # Getting the type of 'self' (line 413)
        self_164164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 37), 'self')
        # Obtaining the member '_sd2wt' of a type (line 413)
        _sd2wt_164165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 37), self_164164, '_sd2wt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 37), tuple_164163, _sd2wt_164165)
        # Adding element type (line 413)
        # Getting the type of 'self' (line 413)
        self_164166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 50), 'self')
        # Obtaining the member 'sx' of a type (line 413)
        sx_164167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 50), self_164166, 'sx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 37), tuple_164163, sx_164167)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 21), dict_164159, (tuple_164160, tuple_164163))
        # Adding element type (key, value) (line 413)
        
        # Obtaining an instance of the builtin type 'tuple' (line 414)
        tuple_164168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 414)
        # Adding element type (line 414)
        str_164169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 23), 'str', 'wd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 23), tuple_164168, str_164169)
        # Adding element type (line 414)
        str_164170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 29), 'str', 'covx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 23), tuple_164168, str_164170)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 414)
        tuple_164171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 414)
        # Adding element type (line 414)
        # Getting the type of 'self' (line 414)
        self_164172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 39), 'self')
        # Obtaining the member '_cov2wt' of a type (line 414)
        _cov2wt_164173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 39), self_164172, '_cov2wt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 39), tuple_164171, _cov2wt_164173)
        # Adding element type (line 414)
        # Getting the type of 'self' (line 414)
        self_164174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 53), 'self')
        # Obtaining the member 'covx' of a type (line 414)
        covx_164175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 53), self_164174, 'covx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 39), tuple_164171, covx_164175)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 21), dict_164159, (tuple_164168, tuple_164171))
        # Adding element type (key, value) (line 413)
        
        # Obtaining an instance of the builtin type 'tuple' (line 415)
        tuple_164176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 415)
        # Adding element type (line 415)
        str_164177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 23), 'str', 'we')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 23), tuple_164176, str_164177)
        # Adding element type (line 415)
        str_164178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 29), 'str', 'sy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 23), tuple_164176, str_164178)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 415)
        tuple_164179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 415)
        # Adding element type (line 415)
        # Getting the type of 'self' (line 415)
        self_164180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 37), 'self')
        # Obtaining the member '_sd2wt' of a type (line 415)
        _sd2wt_164181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 37), self_164180, '_sd2wt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 37), tuple_164179, _sd2wt_164181)
        # Adding element type (line 415)
        # Getting the type of 'self' (line 415)
        self_164182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 50), 'self')
        # Obtaining the member 'sy' of a type (line 415)
        sy_164183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 50), self_164182, 'sy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 37), tuple_164179, sy_164183)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 21), dict_164159, (tuple_164176, tuple_164179))
        # Adding element type (key, value) (line 413)
        
        # Obtaining an instance of the builtin type 'tuple' (line 416)
        tuple_164184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 416)
        # Adding element type (line 416)
        str_164185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 23), 'str', 'we')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 23), tuple_164184, str_164185)
        # Adding element type (line 416)
        str_164186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 29), 'str', 'covy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 23), tuple_164184, str_164186)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 416)
        tuple_164187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 416)
        # Adding element type (line 416)
        # Getting the type of 'self' (line 416)
        self_164188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 39), 'self')
        # Obtaining the member '_cov2wt' of a type (line 416)
        _cov2wt_164189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 39), self_164188, '_cov2wt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 39), tuple_164187, _cov2wt_164189)
        # Adding element type (line 416)
        # Getting the type of 'self' (line 416)
        self_164190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 53), 'self')
        # Obtaining the member 'covy' of a type (line 416)
        covy_164191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 53), self_164190, 'covy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 39), tuple_164187, covy_164191)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 21), dict_164159, (tuple_164184, tuple_164187))
        
        # Assigning a type to the variable 'lookup_tbl' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'lookup_tbl', dict_164159)
        
        
        # Getting the type of 'attr' (line 418)
        attr_164192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), 'attr')
        
        # Obtaining an instance of the builtin type 'tuple' (line 418)
        tuple_164193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 418)
        # Adding element type (line 418)
        str_164194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 24), 'str', 'wd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 24), tuple_164193, str_164194)
        # Adding element type (line 418)
        str_164195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 30), 'str', 'we')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 24), tuple_164193, str_164195)
        
        # Applying the binary operator 'notin' (line 418)
        result_contains_164196 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 11), 'notin', attr_164192, tuple_164193)
        
        # Testing the type of an if condition (line 418)
        if_condition_164197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 8), result_contains_164196)
        # Assigning a type to the variable 'if_condition_164197' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'if_condition_164197', if_condition_164197)
        # SSA begins for if statement (line 418)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'attr' (line 419)
        attr_164198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 15), 'attr')
        # Getting the type of 'self' (line 419)
        self_164199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 23), 'self')
        # Obtaining the member 'meta' of a type (line 419)
        meta_164200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 23), self_164199, 'meta')
        # Applying the binary operator 'in' (line 419)
        result_contains_164201 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 15), 'in', attr_164198, meta_164200)
        
        # Testing the type of an if condition (line 419)
        if_condition_164202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 12), result_contains_164201)
        # Assigning a type to the variable 'if_condition_164202' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'if_condition_164202', if_condition_164202)
        # SSA begins for if statement (line 419)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 420)
        attr_164203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 33), 'attr')
        # Getting the type of 'self' (line 420)
        self_164204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 23), 'self')
        # Obtaining the member 'meta' of a type (line 420)
        meta_164205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 23), self_164204, 'meta')
        # Obtaining the member '__getitem__' of a type (line 420)
        getitem___164206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 23), meta_164205, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 420)
        subscript_call_result_164207 = invoke(stypy.reporting.localization.Localization(__file__, 420, 23), getitem___164206, attr_164203)
        
        # Assigning a type to the variable 'stypy_return_type' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'stypy_return_type', subscript_call_result_164207)
        # SSA branch for the else part of an if statement (line 419)
        module_type_store.open_ssa_branch('else')
        
        # Call to AttributeError(...): (line 422)
        # Processing the call arguments (line 422)
        str_164209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 37), 'str', "'%s' not in metadata")
        # Getting the type of 'attr' (line 422)
        attr_164210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 62), 'attr', False)
        # Applying the binary operator '%' (line 422)
        result_mod_164211 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 37), '%', str_164209, attr_164210)
        
        # Processing the call keyword arguments (line 422)
        kwargs_164212 = {}
        # Getting the type of 'AttributeError' (line 422)
        AttributeError_164208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 22), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 422)
        AttributeError_call_result_164213 = invoke(stypy.reporting.localization.Localization(__file__, 422, 22), AttributeError_164208, *[result_mod_164211], **kwargs_164212)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 422, 16), AttributeError_call_result_164213, 'raise parameter', BaseException)
        # SSA join for if statement (line 419)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 418)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Tuple (line 424):
        
        # Assigning a Subscript to a Name (line 424):
        
        # Obtaining the type of the subscript
        int_164214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 12), 'int')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 424)
        tuple_164215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 424)
        # Adding element type (line 424)
        # Getting the type of 'attr' (line 424)
        attr_164216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 36), 'attr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 36), tuple_164215, attr_164216)
        # Adding element type (line 424)
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 424)
        attr_164217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 57), 'attr')
        # Getting the type of 'self' (line 424)
        self_164218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 42), 'self')
        # Obtaining the member '_ga_flags' of a type (line 424)
        _ga_flags_164219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 42), self_164218, '_ga_flags')
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___164220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 42), _ga_flags_164219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_164221 = invoke(stypy.reporting.localization.Localization(__file__, 424, 42), getitem___164220, attr_164217)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 36), tuple_164215, subscript_call_result_164221)
        
        # Getting the type of 'lookup_tbl' (line 424)
        lookup_tbl_164222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'lookup_tbl')
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___164223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 24), lookup_tbl_164222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_164224 = invoke(stypy.reporting.localization.Localization(__file__, 424, 24), getitem___164223, tuple_164215)
        
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___164225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), subscript_call_result_164224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_164226 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), getitem___164225, int_164214)
        
        # Assigning a type to the variable 'tuple_var_assignment_163536' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'tuple_var_assignment_163536', subscript_call_result_164226)
        
        # Assigning a Subscript to a Name (line 424):
        
        # Obtaining the type of the subscript
        int_164227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 12), 'int')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 424)
        tuple_164228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 424)
        # Adding element type (line 424)
        # Getting the type of 'attr' (line 424)
        attr_164229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 36), 'attr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 36), tuple_164228, attr_164229)
        # Adding element type (line 424)
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 424)
        attr_164230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 57), 'attr')
        # Getting the type of 'self' (line 424)
        self_164231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 42), 'self')
        # Obtaining the member '_ga_flags' of a type (line 424)
        _ga_flags_164232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 42), self_164231, '_ga_flags')
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___164233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 42), _ga_flags_164232, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_164234 = invoke(stypy.reporting.localization.Localization(__file__, 424, 42), getitem___164233, attr_164230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 36), tuple_164228, subscript_call_result_164234)
        
        # Getting the type of 'lookup_tbl' (line 424)
        lookup_tbl_164235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'lookup_tbl')
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___164236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 24), lookup_tbl_164235, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_164237 = invoke(stypy.reporting.localization.Localization(__file__, 424, 24), getitem___164236, tuple_164228)
        
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___164238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), subscript_call_result_164237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_164239 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), getitem___164238, int_164227)
        
        # Assigning a type to the variable 'tuple_var_assignment_163537' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'tuple_var_assignment_163537', subscript_call_result_164239)
        
        # Assigning a Name to a Name (line 424):
        # Getting the type of 'tuple_var_assignment_163536' (line 424)
        tuple_var_assignment_163536_164240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'tuple_var_assignment_163536')
        # Assigning a type to the variable 'func' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'func', tuple_var_assignment_163536_164240)
        
        # Assigning a Name to a Name (line 424):
        # Getting the type of 'tuple_var_assignment_163537' (line 424)
        tuple_var_assignment_163537_164241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'tuple_var_assignment_163537')
        # Assigning a type to the variable 'arg' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 18), 'arg', tuple_var_assignment_163537_164241)
        
        # Type idiom detected: calculating its left and rigth part (line 426)
        # Getting the type of 'arg' (line 426)
        arg_164242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'arg')
        # Getting the type of 'None' (line 426)
        None_164243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 26), 'None')
        
        (may_be_164244, more_types_in_union_164245) = may_not_be_none(arg_164242, None_164243)

        if may_be_164244:

            if more_types_in_union_164245:
                # Runtime conditional SSA (line 426)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to func(...): (line 427)
            
            # Obtaining an instance of the builtin type 'tuple' (line 427)
            tuple_164247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 427)
            # Adding element type (line 427)
            # Getting the type of 'arg' (line 427)
            arg_164248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 30), 'arg', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 30), tuple_164247, arg_164248)
            
            # Processing the call keyword arguments (line 427)
            kwargs_164249 = {}
            # Getting the type of 'func' (line 427)
            func_164246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 23), 'func', False)
            # Calling func(args, kwargs) (line 427)
            func_call_result_164250 = invoke(stypy.reporting.localization.Localization(__file__, 427, 23), func_164246, *[tuple_164247], **kwargs_164249)
            
            # Assigning a type to the variable 'stypy_return_type' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'stypy_return_type', func_call_result_164250)

            if more_types_in_union_164245:
                # Runtime conditional SSA for else branch (line 426)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_164244) or more_types_in_union_164245):
            # Getting the type of 'None' (line 429)
            None_164251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 429)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'stypy_return_type', None_164251)

            if (may_be_164244 and more_types_in_union_164245):
                # SSA join for if statement (line 426)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 418)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 412)
        stypy_return_type_164252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_164252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_164252


# Assigning a type to the variable 'RealData' (line 296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'RealData', RealData)
# Declaration of the 'Model' class

class Model(object, ):
    str_164253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, (-1)), 'str', "\n    The Model class stores information about the function you wish to fit.\n\n    It stores the function itself, at the least, and optionally stores\n    functions which compute the Jacobians used during fitting. Also, one\n    can provide a function that will provide reasonable starting values\n    for the fit parameters possibly given the set of data.\n\n    Parameters\n    ----------\n    fcn : function\n          fcn(beta, x) --> y\n    fjacb : function\n          Jacobian of fcn wrt the fit parameters beta.\n\n          fjacb(beta, x) --> @f_i(x,B)/@B_j\n    fjacd : function\n          Jacobian of fcn wrt the (possibly multidimensional) input\n          variable.\n\n          fjacd(beta, x) --> @f_i(x,B)/@x_j\n    extra_args : tuple, optional\n          If specified, `extra_args` should be a tuple of extra\n          arguments to pass to `fcn`, `fjacb`, and `fjacd`. Each will be called\n          by `apply(fcn, (beta, x) + extra_args)`\n    estimate : array_like of rank-1\n          Provides estimates of the fit parameters from the data\n\n          estimate(data) --> estbeta\n    implicit : boolean\n          If TRUE, specifies that the model\n          is implicit; i.e `fcn(beta, x)` ~= 0 and there is no y data to fit\n          against\n    meta : dict, optional\n          freeform dictionary of metadata for the model\n\n    Notes\n    -----\n    Note that the `fcn`, `fjacb`, and `fjacd` operate on NumPy arrays and\n    return a NumPy array. The `estimate` object takes an instance of the\n    Data class.\n\n    Here are the rules for the shapes of the argument and return\n    arrays of the callback functions:\n\n    `x`\n        if the input data is single-dimensional, then `x` is rank-1\n        array; i.e. ``x = array([1, 2, 3, ...]); x.shape = (n,)``\n        If the input data is multi-dimensional, then `x` is a rank-2 array;\n        i.e., ``x = array([[1, 2, ...], [2, 4, ...]]); x.shape = (m, n)``.\n        In all cases, it has the same shape as the input data array passed to\n        `odr`. `m` is the dimensionality of the input data, `n` is the number\n        of observations.\n    `y`\n        if the response variable is single-dimensional, then `y` is a\n        rank-1 array, i.e., ``y = array([2, 4, ...]); y.shape = (n,)``.\n        If the response variable is multi-dimensional, then `y` is a rank-2\n        array, i.e., ``y = array([[2, 4, ...], [3, 6, ...]]); y.shape =\n        (q, n)`` where `q` is the dimensionality of the response variable.\n    `beta`\n        rank-1 array of length `p` where `p` is the number of parameters;\n        i.e. ``beta = array([B_1, B_2, ..., B_p])``\n    `fjacb`\n        if the response variable is multi-dimensional, then the\n        return array's shape is `(q, p, n)` such that ``fjacb(x,beta)[l,k,i] =\n        d f_l(X,B)/d B_k`` evaluated at the i'th data point.  If `q == 1`, then\n        the return array is only rank-2 and with shape `(p, n)`.\n    `fjacd`\n        as with fjacb, only the return array's shape is `(q, m, n)`\n        such that ``fjacd(x,beta)[l,j,i] = d f_l(X,B)/d X_j`` at the i'th data\n        point.  If `q == 1`, then the return array's shape is `(m, n)`. If\n        `m == 1`, the shape is (q, n). If `m == q == 1`, the shape is `(n,)`.\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 508)
        None_164254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 34), 'None')
        # Getting the type of 'None' (line 508)
        None_164255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 46), 'None')
        # Getting the type of 'None' (line 509)
        None_164256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), 'None')
        # Getting the type of 'None' (line 509)
        None_164257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 34), 'None')
        int_164258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 49), 'int')
        # Getting the type of 'None' (line 509)
        None_164259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 57), 'None')
        defaults = [None_164254, None_164255, None_164256, None_164257, int_164258, None_164259]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 508, 4, False)
        # Assigning a type to the variable 'self' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Model.__init__', ['fcn', 'fjacb', 'fjacd', 'extra_args', 'estimate', 'implicit', 'meta'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fcn', 'fjacb', 'fjacd', 'extra_args', 'estimate', 'implicit', 'meta'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 511):
        
        # Assigning a Name to a Attribute (line 511):
        # Getting the type of 'fcn' (line 511)
        fcn_164260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 19), 'fcn')
        # Getting the type of 'self' (line 511)
        self_164261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'self')
        # Setting the type of the member 'fcn' of a type (line 511)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 8), self_164261, 'fcn', fcn_164260)
        
        # Assigning a Name to a Attribute (line 512):
        
        # Assigning a Name to a Attribute (line 512):
        # Getting the type of 'fjacb' (line 512)
        fjacb_164262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 21), 'fjacb')
        # Getting the type of 'self' (line 512)
        self_164263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'self')
        # Setting the type of the member 'fjacb' of a type (line 512)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), self_164263, 'fjacb', fjacb_164262)
        
        # Assigning a Name to a Attribute (line 513):
        
        # Assigning a Name to a Attribute (line 513):
        # Getting the type of 'fjacd' (line 513)
        fjacd_164264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'fjacd')
        # Getting the type of 'self' (line 513)
        self_164265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self')
        # Setting the type of the member 'fjacd' of a type (line 513)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_164265, 'fjacd', fjacd_164264)
        
        # Type idiom detected: calculating its left and rigth part (line 515)
        # Getting the type of 'extra_args' (line 515)
        extra_args_164266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'extra_args')
        # Getting the type of 'None' (line 515)
        None_164267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 29), 'None')
        
        (may_be_164268, more_types_in_union_164269) = may_not_be_none(extra_args_164266, None_164267)

        if may_be_164268:

            if more_types_in_union_164269:
                # Runtime conditional SSA (line 515)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 516):
            
            # Assigning a Call to a Name (line 516):
            
            # Call to tuple(...): (line 516)
            # Processing the call arguments (line 516)
            # Getting the type of 'extra_args' (line 516)
            extra_args_164271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 31), 'extra_args', False)
            # Processing the call keyword arguments (line 516)
            kwargs_164272 = {}
            # Getting the type of 'tuple' (line 516)
            tuple_164270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 25), 'tuple', False)
            # Calling tuple(args, kwargs) (line 516)
            tuple_call_result_164273 = invoke(stypy.reporting.localization.Localization(__file__, 516, 25), tuple_164270, *[extra_args_164271], **kwargs_164272)
            
            # Assigning a type to the variable 'extra_args' (line 516)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'extra_args', tuple_call_result_164273)

            if more_types_in_union_164269:
                # SSA join for if statement (line 515)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 518):
        
        # Assigning a Name to a Attribute (line 518):
        # Getting the type of 'extra_args' (line 518)
        extra_args_164274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 26), 'extra_args')
        # Getting the type of 'self' (line 518)
        self_164275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'self')
        # Setting the type of the member 'extra_args' of a type (line 518)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), self_164275, 'extra_args', extra_args_164274)
        
        # Assigning a Name to a Attribute (line 519):
        
        # Assigning a Name to a Attribute (line 519):
        # Getting the type of 'estimate' (line 519)
        estimate_164276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 24), 'estimate')
        # Getting the type of 'self' (line 519)
        self_164277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'self')
        # Setting the type of the member 'estimate' of a type (line 519)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), self_164277, 'estimate', estimate_164276)
        
        # Assigning a Name to a Attribute (line 520):
        
        # Assigning a Name to a Attribute (line 520):
        # Getting the type of 'implicit' (line 520)
        implicit_164278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 24), 'implicit')
        # Getting the type of 'self' (line 520)
        self_164279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'self')
        # Setting the type of the member 'implicit' of a type (line 520)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), self_164279, 'implicit', implicit_164278)
        
        # Assigning a Name to a Attribute (line 521):
        
        # Assigning a Name to a Attribute (line 521):
        # Getting the type of 'meta' (line 521)
        meta_164280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 20), 'meta')
        # Getting the type of 'self' (line 521)
        self_164281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'self')
        # Setting the type of the member 'meta' of a type (line 521)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 8), self_164281, 'meta', meta_164280)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_meta(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_meta'
        module_type_store = module_type_store.open_function_context('set_meta', 523, 4, False)
        # Assigning a type to the variable 'self' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Model.set_meta.__dict__.__setitem__('stypy_localization', localization)
        Model.set_meta.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Model.set_meta.__dict__.__setitem__('stypy_type_store', module_type_store)
        Model.set_meta.__dict__.__setitem__('stypy_function_name', 'Model.set_meta')
        Model.set_meta.__dict__.__setitem__('stypy_param_names_list', [])
        Model.set_meta.__dict__.__setitem__('stypy_varargs_param_name', None)
        Model.set_meta.__dict__.__setitem__('stypy_kwargs_param_name', 'kwds')
        Model.set_meta.__dict__.__setitem__('stypy_call_defaults', defaults)
        Model.set_meta.__dict__.__setitem__('stypy_call_varargs', varargs)
        Model.set_meta.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Model.set_meta.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Model.set_meta', [], None, 'kwds', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_meta', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_meta(...)' code ##################

        str_164282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, (-1)), 'str', ' Update the metadata dictionary with the keywords and data provided\n        here.\n\n        Examples\n        --------\n        set_meta(name="Exponential", equation="y = a exp(b x) + c")\n        ')
        
        # Call to update(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'kwds' (line 532)
        kwds_164286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 25), 'kwds', False)
        # Processing the call keyword arguments (line 532)
        kwargs_164287 = {}
        # Getting the type of 'self' (line 532)
        self_164283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'self', False)
        # Obtaining the member 'meta' of a type (line 532)
        meta_164284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), self_164283, 'meta')
        # Obtaining the member 'update' of a type (line 532)
        update_164285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), meta_164284, 'update')
        # Calling update(args, kwargs) (line 532)
        update_call_result_164288 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), update_164285, *[kwds_164286], **kwargs_164287)
        
        
        # ################# End of 'set_meta(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_meta' in the type store
        # Getting the type of 'stypy_return_type' (line 523)
        stypy_return_type_164289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_164289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_meta'
        return stypy_return_type_164289


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 534, 4, False)
        # Assigning a type to the variable 'self' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Model.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        Model.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Model.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Model.__getattr__.__dict__.__setitem__('stypy_function_name', 'Model.__getattr__')
        Model.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        Model.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Model.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Model.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Model.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Model.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Model.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Model.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        str_164290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, (-1)), 'str', ' Dispatch attribute access to the metadata.\n        ')
        
        
        # Getting the type of 'attr' (line 538)
        attr_164291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 11), 'attr')
        # Getting the type of 'self' (line 538)
        self_164292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 19), 'self')
        # Obtaining the member 'meta' of a type (line 538)
        meta_164293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 19), self_164292, 'meta')
        # Applying the binary operator 'in' (line 538)
        result_contains_164294 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 11), 'in', attr_164291, meta_164293)
        
        # Testing the type of an if condition (line 538)
        if_condition_164295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 538, 8), result_contains_164294)
        # Assigning a type to the variable 'if_condition_164295' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'if_condition_164295', if_condition_164295)
        # SSA begins for if statement (line 538)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 539)
        attr_164296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 29), 'attr')
        # Getting the type of 'self' (line 539)
        self_164297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 19), 'self')
        # Obtaining the member 'meta' of a type (line 539)
        meta_164298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 19), self_164297, 'meta')
        # Obtaining the member '__getitem__' of a type (line 539)
        getitem___164299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 19), meta_164298, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 539)
        subscript_call_result_164300 = invoke(stypy.reporting.localization.Localization(__file__, 539, 19), getitem___164299, attr_164296)
        
        # Assigning a type to the variable 'stypy_return_type' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'stypy_return_type', subscript_call_result_164300)
        # SSA branch for the else part of an if statement (line 538)
        module_type_store.open_ssa_branch('else')
        
        # Call to AttributeError(...): (line 541)
        # Processing the call arguments (line 541)
        str_164302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 33), 'str', "'%s' not in metadata")
        # Getting the type of 'attr' (line 541)
        attr_164303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 58), 'attr', False)
        # Applying the binary operator '%' (line 541)
        result_mod_164304 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 33), '%', str_164302, attr_164303)
        
        # Processing the call keyword arguments (line 541)
        kwargs_164305 = {}
        # Getting the type of 'AttributeError' (line 541)
        AttributeError_164301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 541)
        AttributeError_call_result_164306 = invoke(stypy.reporting.localization.Localization(__file__, 541, 18), AttributeError_164301, *[result_mod_164304], **kwargs_164305)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 541, 12), AttributeError_call_result_164306, 'raise parameter', BaseException)
        # SSA join for if statement (line 538)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 534)
        stypy_return_type_164307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_164307)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_164307


# Assigning a type to the variable 'Model' (line 432)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), 'Model', Model)
# Declaration of the 'Output' class

class Output(object, ):
    str_164308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, (-1)), 'str', '\n    The Output class stores the output of an ODR run.\n\n    Attributes\n    ----------\n    beta : ndarray\n        Estimated parameter values, of shape (q,).\n    sd_beta : ndarray\n        Standard errors of the estimated parameters, of shape (p,).\n    cov_beta : ndarray\n        Covariance matrix of the estimated parameters, of shape (p,p).\n    delta : ndarray, optional\n        Array of estimated errors in input variables, of same shape as `x`.\n    eps : ndarray, optional\n        Array of estimated errors in response variables, of same shape as `y`.\n    xplus : ndarray, optional\n        Array of ``x + delta``.\n    y : ndarray, optional\n        Array ``y = fcn(x + delta)``.\n    res_var : float, optional\n        Residual variance.\n    sum_square : float, optional\n        Sum of squares error.\n    sum_square_delta : float, optional\n        Sum of squares of delta error.\n    sum_square_eps : float, optional\n        Sum of squares of eps error.\n    inv_condnum : float, optional\n        Inverse condition number (cf. ODRPACK UG p. 77).\n    rel_error : float, optional\n        Relative error in function values computed within fcn.\n    work : ndarray, optional\n        Final work array.\n    work_ind : dict, optional\n        Indices into work for drawing out values (cf. ODRPACK UG p. 83).\n    info : int, optional\n        Reason for returning, as output by ODRPACK (cf. ODRPACK UG p. 38).\n    stopreason : list of str, optional\n        `info` interpreted into English.\n\n    Notes\n    -----\n    Takes one argument for initialization, the return value from the\n    function `odr`. The attributes listed as "optional" above are only\n    present if `odr` was run with ``full_output=1``.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 593, 4, False)
        # Assigning a type to the variable 'self' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Output.__init__', ['output'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['output'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Subscript to a Attribute (line 594):
        
        # Assigning a Subscript to a Attribute (line 594):
        
        # Obtaining the type of the subscript
        int_164309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 27), 'int')
        # Getting the type of 'output' (line 594)
        output_164310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 20), 'output')
        # Obtaining the member '__getitem__' of a type (line 594)
        getitem___164311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 20), output_164310, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 594)
        subscript_call_result_164312 = invoke(stypy.reporting.localization.Localization(__file__, 594, 20), getitem___164311, int_164309)
        
        # Getting the type of 'self' (line 594)
        self_164313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'self')
        # Setting the type of the member 'beta' of a type (line 594)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 8), self_164313, 'beta', subscript_call_result_164312)
        
        # Assigning a Subscript to a Attribute (line 595):
        
        # Assigning a Subscript to a Attribute (line 595):
        
        # Obtaining the type of the subscript
        int_164314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 30), 'int')
        # Getting the type of 'output' (line 595)
        output_164315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 23), 'output')
        # Obtaining the member '__getitem__' of a type (line 595)
        getitem___164316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 23), output_164315, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 595)
        subscript_call_result_164317 = invoke(stypy.reporting.localization.Localization(__file__, 595, 23), getitem___164316, int_164314)
        
        # Getting the type of 'self' (line 595)
        self_164318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'self')
        # Setting the type of the member 'sd_beta' of a type (line 595)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 8), self_164318, 'sd_beta', subscript_call_result_164317)
        
        # Assigning a Subscript to a Attribute (line 596):
        
        # Assigning a Subscript to a Attribute (line 596):
        
        # Obtaining the type of the subscript
        int_164319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 31), 'int')
        # Getting the type of 'output' (line 596)
        output_164320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 24), 'output')
        # Obtaining the member '__getitem__' of a type (line 596)
        getitem___164321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 24), output_164320, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 596)
        subscript_call_result_164322 = invoke(stypy.reporting.localization.Localization(__file__, 596, 24), getitem___164321, int_164319)
        
        # Getting the type of 'self' (line 596)
        self_164323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'self')
        # Setting the type of the member 'cov_beta' of a type (line 596)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 8), self_164323, 'cov_beta', subscript_call_result_164322)
        
        
        
        # Call to len(...): (line 598)
        # Processing the call arguments (line 598)
        # Getting the type of 'output' (line 598)
        output_164325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'output', False)
        # Processing the call keyword arguments (line 598)
        kwargs_164326 = {}
        # Getting the type of 'len' (line 598)
        len_164324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 11), 'len', False)
        # Calling len(args, kwargs) (line 598)
        len_call_result_164327 = invoke(stypy.reporting.localization.Localization(__file__, 598, 11), len_164324, *[output_164325], **kwargs_164326)
        
        int_164328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 26), 'int')
        # Applying the binary operator '==' (line 598)
        result_eq_164329 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 11), '==', len_call_result_164327, int_164328)
        
        # Testing the type of an if condition (line 598)
        if_condition_164330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 598, 8), result_eq_164329)
        # Assigning a type to the variable 'if_condition_164330' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'if_condition_164330', if_condition_164330)
        # SSA begins for if statement (line 598)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update(...): (line 600)
        # Processing the call arguments (line 600)
        
        # Obtaining the type of the subscript
        int_164334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 40), 'int')
        # Getting the type of 'output' (line 600)
        output_164335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 33), 'output', False)
        # Obtaining the member '__getitem__' of a type (line 600)
        getitem___164336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 33), output_164335, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 600)
        subscript_call_result_164337 = invoke(stypy.reporting.localization.Localization(__file__, 600, 33), getitem___164336, int_164334)
        
        # Processing the call keyword arguments (line 600)
        kwargs_164338 = {}
        # Getting the type of 'self' (line 600)
        self_164331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'self', False)
        # Obtaining the member '__dict__' of a type (line 600)
        dict___164332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 12), self_164331, '__dict__')
        # Obtaining the member 'update' of a type (line 600)
        update_164333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 12), dict___164332, 'update')
        # Calling update(args, kwargs) (line 600)
        update_call_result_164339 = invoke(stypy.reporting.localization.Localization(__file__, 600, 12), update_164333, *[subscript_call_result_164337], **kwargs_164338)
        
        
        # Assigning a Call to a Attribute (line 601):
        
        # Assigning a Call to a Attribute (line 601):
        
        # Call to _report_error(...): (line 601)
        # Processing the call arguments (line 601)
        # Getting the type of 'self' (line 601)
        self_164341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 44), 'self', False)
        # Obtaining the member 'info' of a type (line 601)
        info_164342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 44), self_164341, 'info')
        # Processing the call keyword arguments (line 601)
        kwargs_164343 = {}
        # Getting the type of '_report_error' (line 601)
        _report_error_164340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 30), '_report_error', False)
        # Calling _report_error(args, kwargs) (line 601)
        _report_error_call_result_164344 = invoke(stypy.reporting.localization.Localization(__file__, 601, 30), _report_error_164340, *[info_164342], **kwargs_164343)
        
        # Getting the type of 'self' (line 601)
        self_164345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'self')
        # Setting the type of the member 'stopreason' of a type (line 601)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), self_164345, 'stopreason', _report_error_call_result_164344)
        # SSA join for if statement (line 598)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def pprint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pprint'
        module_type_store = module_type_store.open_function_context('pprint', 603, 4, False)
        # Assigning a type to the variable 'self' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Output.pprint.__dict__.__setitem__('stypy_localization', localization)
        Output.pprint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Output.pprint.__dict__.__setitem__('stypy_type_store', module_type_store)
        Output.pprint.__dict__.__setitem__('stypy_function_name', 'Output.pprint')
        Output.pprint.__dict__.__setitem__('stypy_param_names_list', [])
        Output.pprint.__dict__.__setitem__('stypy_varargs_param_name', None)
        Output.pprint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Output.pprint.__dict__.__setitem__('stypy_call_defaults', defaults)
        Output.pprint.__dict__.__setitem__('stypy_call_varargs', varargs)
        Output.pprint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Output.pprint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Output.pprint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pprint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pprint(...)' code ##################

        str_164346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, (-1)), 'str', ' Pretty-print important results.\n        ')
        
        # Call to print(...): (line 607)
        # Processing the call arguments (line 607)
        str_164348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 14), 'str', 'Beta:')
        # Getting the type of 'self' (line 607)
        self_164349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 23), 'self', False)
        # Obtaining the member 'beta' of a type (line 607)
        beta_164350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 23), self_164349, 'beta')
        # Processing the call keyword arguments (line 607)
        kwargs_164351 = {}
        # Getting the type of 'print' (line 607)
        print_164347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'print', False)
        # Calling print(args, kwargs) (line 607)
        print_call_result_164352 = invoke(stypy.reporting.localization.Localization(__file__, 607, 8), print_164347, *[str_164348, beta_164350], **kwargs_164351)
        
        
        # Call to print(...): (line 608)
        # Processing the call arguments (line 608)
        str_164354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 14), 'str', 'Beta Std Error:')
        # Getting the type of 'self' (line 608)
        self_164355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 33), 'self', False)
        # Obtaining the member 'sd_beta' of a type (line 608)
        sd_beta_164356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 33), self_164355, 'sd_beta')
        # Processing the call keyword arguments (line 608)
        kwargs_164357 = {}
        # Getting the type of 'print' (line 608)
        print_164353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'print', False)
        # Calling print(args, kwargs) (line 608)
        print_call_result_164358 = invoke(stypy.reporting.localization.Localization(__file__, 608, 8), print_164353, *[str_164354, sd_beta_164356], **kwargs_164357)
        
        
        # Call to print(...): (line 609)
        # Processing the call arguments (line 609)
        str_164360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 14), 'str', 'Beta Covariance:')
        # Getting the type of 'self' (line 609)
        self_164361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 34), 'self', False)
        # Obtaining the member 'cov_beta' of a type (line 609)
        cov_beta_164362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 34), self_164361, 'cov_beta')
        # Processing the call keyword arguments (line 609)
        kwargs_164363 = {}
        # Getting the type of 'print' (line 609)
        print_164359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'print', False)
        # Calling print(args, kwargs) (line 609)
        print_call_result_164364 = invoke(stypy.reporting.localization.Localization(__file__, 609, 8), print_164359, *[str_164360, cov_beta_164362], **kwargs_164363)
        
        
        # Type idiom detected: calculating its left and rigth part (line 610)
        str_164365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 25), 'str', 'info')
        # Getting the type of 'self' (line 610)
        self_164366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 19), 'self')
        
        (may_be_164367, more_types_in_union_164368) = may_provide_member(str_164365, self_164366)

        if may_be_164367:

            if more_types_in_union_164368:
                # Runtime conditional SSA (line 610)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'self', remove_not_member_provider_from_union(self_164366, 'info'))
            
            # Call to print(...): (line 611)
            # Processing the call arguments (line 611)
            str_164370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 18), 'str', 'Residual Variance:')
            # Getting the type of 'self' (line 611)
            self_164371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 39), 'self', False)
            # Obtaining the member 'res_var' of a type (line 611)
            res_var_164372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 39), self_164371, 'res_var')
            # Processing the call keyword arguments (line 611)
            kwargs_164373 = {}
            # Getting the type of 'print' (line 611)
            print_164369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'print', False)
            # Calling print(args, kwargs) (line 611)
            print_call_result_164374 = invoke(stypy.reporting.localization.Localization(__file__, 611, 12), print_164369, *[str_164370, res_var_164372], **kwargs_164373)
            
            
            # Call to print(...): (line 612)
            # Processing the call arguments (line 612)
            str_164376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 18), 'str', 'Inverse Condition #:')
            # Getting the type of 'self' (line 612)
            self_164377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 42), 'self', False)
            # Obtaining the member 'inv_condnum' of a type (line 612)
            inv_condnum_164378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 42), self_164377, 'inv_condnum')
            # Processing the call keyword arguments (line 612)
            kwargs_164379 = {}
            # Getting the type of 'print' (line 612)
            print_164375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'print', False)
            # Calling print(args, kwargs) (line 612)
            print_call_result_164380 = invoke(stypy.reporting.localization.Localization(__file__, 612, 12), print_164375, *[str_164376, inv_condnum_164378], **kwargs_164379)
            
            
            # Call to print(...): (line 613)
            # Processing the call arguments (line 613)
            str_164382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 18), 'str', 'Reason(s) for Halting:')
            # Processing the call keyword arguments (line 613)
            kwargs_164383 = {}
            # Getting the type of 'print' (line 613)
            print_164381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'print', False)
            # Calling print(args, kwargs) (line 613)
            print_call_result_164384 = invoke(stypy.reporting.localization.Localization(__file__, 613, 12), print_164381, *[str_164382], **kwargs_164383)
            
            
            # Getting the type of 'self' (line 614)
            self_164385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 21), 'self')
            # Obtaining the member 'stopreason' of a type (line 614)
            stopreason_164386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 21), self_164385, 'stopreason')
            # Testing the type of a for loop iterable (line 614)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 614, 12), stopreason_164386)
            # Getting the type of the for loop variable (line 614)
            for_loop_var_164387 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 614, 12), stopreason_164386)
            # Assigning a type to the variable 'r' (line 614)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'r', for_loop_var_164387)
            # SSA begins for a for statement (line 614)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to print(...): (line 615)
            # Processing the call arguments (line 615)
            str_164389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 22), 'str', '  %s')
            # Getting the type of 'r' (line 615)
            r_164390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 31), 'r', False)
            # Applying the binary operator '%' (line 615)
            result_mod_164391 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 22), '%', str_164389, r_164390)
            
            # Processing the call keyword arguments (line 615)
            kwargs_164392 = {}
            # Getting the type of 'print' (line 615)
            print_164388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 16), 'print', False)
            # Calling print(args, kwargs) (line 615)
            print_call_result_164393 = invoke(stypy.reporting.localization.Localization(__file__, 615, 16), print_164388, *[result_mod_164391], **kwargs_164392)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_164368:
                # SSA join for if statement (line 610)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'pprint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pprint' in the type store
        # Getting the type of 'stypy_return_type' (line 603)
        stypy_return_type_164394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_164394)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pprint'
        return stypy_return_type_164394


# Assigning a type to the variable 'Output' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'Output', Output)
# Declaration of the 'ODR' class

class ODR(object, ):
    str_164395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, (-1)), 'str', '\n    The ODR class gathers all information and coordinates the running of the\n    main fitting routine.\n\n    Members of instances of the ODR class have the same names as the arguments\n    to the initialization routine.\n\n    Parameters\n    ----------\n    data : Data class instance\n        instance of the Data class\n    model : Model class instance\n        instance of the Model class\n\n    Other Parameters\n    ----------------\n    beta0 : array_like of rank-1\n        a rank-1 sequence of initial parameter values. Optional if\n        model provides an "estimate" function to estimate these values.\n    delta0 : array_like of floats of rank-1, optional\n        a (double-precision) float array to hold the initial values of\n        the errors in the input variables. Must be same shape as data.x\n    ifixb : array_like of ints of rank-1, optional\n        sequence of integers with the same length as beta0 that determines\n        which parameters are held fixed. A value of 0 fixes the parameter,\n        a value > 0 makes the parameter free.\n    ifixx : array_like of ints with same shape as data.x, optional\n        an array of integers with the same shape as data.x that determines\n        which input observations are treated as fixed. One can use a sequence\n        of length m (the dimensionality of the input observations) to fix some\n        dimensions for all observations. A value of 0 fixes the observation,\n        a value > 0 makes it free.\n    job : int, optional\n        an integer telling ODRPACK what tasks to perform. See p. 31 of the\n        ODRPACK User\'s Guide if you absolutely must set the value here. Use the\n        method set_job post-initialization for a more readable interface.\n    iprint : int, optional\n        an integer telling ODRPACK what to print. See pp. 33-34 of the\n        ODRPACK User\'s Guide if you absolutely must set the value here. Use the\n        method set_iprint post-initialization for a more readable interface.\n    errfile : str, optional\n        string with the filename to print ODRPACK errors to. *Do Not Open\n        This File Yourself!*\n    rptfile : str, optional\n        string with the filename to print ODRPACK summaries to. *Do Not\n        Open This File Yourself!*\n    ndigit : int, optional\n        integer specifying the number of reliable digits in the computation\n        of the function.\n    taufac : float, optional\n        float specifying the initial trust region. The default value is 1.\n        The initial trust region is equal to taufac times the length of the\n        first computed Gauss-Newton step. taufac must be less than 1.\n    sstol : float, optional\n        float specifying the tolerance for convergence based on the relative\n        change in the sum-of-squares. The default value is eps**(1/2) where eps\n        is the smallest value such that 1 + eps > 1 for double precision\n        computation on the machine. sstol must be less than 1.\n    partol : float, optional\n        float specifying the tolerance for convergence based on the relative\n        change in the estimated parameters. The default value is eps**(2/3) for\n        explicit models and ``eps**(1/3)`` for implicit models. partol must be less\n        than 1.\n    maxit : int, optional\n        integer specifying the maximum number of iterations to perform. For\n        first runs, maxit is the total number of iterations performed and\n        defaults to 50.  For restarts, maxit is the number of additional\n        iterations to perform and defaults to 10.\n    stpb : array_like, optional\n        sequence (``len(stpb) == len(beta0)``) of relative step sizes to compute\n        finite difference derivatives wrt the parameters.\n    stpd : optional\n        array (``stpd.shape == data.x.shape`` or ``stpd.shape == (m,)``) of relative\n        step sizes to compute finite difference derivatives wrt the input\n        variable errors. If stpd is a rank-1 array with length m (the\n        dimensionality of the input variable), then the values are broadcast to\n        all observations.\n    sclb : array_like, optional\n        sequence (``len(stpb) == len(beta0)``) of scaling factors for the\n        parameters.  The purpose of these scaling factors are to scale all of\n        the parameters to around unity. Normally appropriate scaling factors\n        are computed if this argument is not specified. Specify them yourself\n        if the automatic procedure goes awry.\n    scld : array_like, optional\n        array (scld.shape == data.x.shape or scld.shape == (m,)) of scaling\n        factors for the *errors* in the input variables. Again, these factors\n        are automatically computed if you do not provide them. If scld.shape ==\n        (m,), then the scaling factors are broadcast to all observations.\n    work : ndarray, optional\n        array to hold the double-valued working data for ODRPACK. When\n        restarting, takes the value of self.output.work.\n    iwork : ndarray, optional\n        array to hold the integer-valued working data for ODRPACK. When\n        restarting, takes the value of self.output.iwork.\n\n    Attributes\n    ----------\n    data : Data\n        The data for this fit\n    model : Model\n        The model used in fit\n    output : Output\n        An instance if the Output class containing all of the returned\n        data from an invocation of ODR.run() or ODR.restart()\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 726)
        None_164396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 42), 'None')
        # Getting the type of 'None' (line 726)
        None_164397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 55), 'None')
        # Getting the type of 'None' (line 726)
        None_164398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 67), 'None')
        # Getting the type of 'None' (line 727)
        None_164399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 14), 'None')
        # Getting the type of 'None' (line 727)
        None_164400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 24), 'None')
        # Getting the type of 'None' (line 727)
        None_164401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 37), 'None')
        # Getting the type of 'None' (line 727)
        None_164402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 51), 'None')
        # Getting the type of 'None' (line 727)
        None_164403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 65), 'None')
        # Getting the type of 'None' (line 728)
        None_164404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 15), 'None')
        # Getting the type of 'None' (line 728)
        None_164405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 28), 'None')
        # Getting the type of 'None' (line 728)
        None_164406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 40), 'None')
        # Getting the type of 'None' (line 728)
        None_164407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 53), 'None')
        # Getting the type of 'None' (line 728)
        None_164408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 65), 'None')
        # Getting the type of 'None' (line 729)
        None_164409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), 'None')
        # Getting the type of 'None' (line 729)
        None_164410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 24), 'None')
        # Getting the type of 'None' (line 729)
        None_164411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 35), 'None')
        # Getting the type of 'None' (line 729)
        None_164412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 46), 'None')
        # Getting the type of 'None' (line 729)
        None_164413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 57), 'None')
        # Getting the type of 'None' (line 729)
        None_164414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 69), 'None')
        defaults = [None_164396, None_164397, None_164398, None_164399, None_164400, None_164401, None_164402, None_164403, None_164404, None_164405, None_164406, None_164407, None_164408, None_164409, None_164410, None_164411, None_164412, None_164413, None_164414]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 726, 4, False)
        # Assigning a type to the variable 'self' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODR.__init__', ['data', 'model', 'beta0', 'delta0', 'ifixb', 'ifixx', 'job', 'iprint', 'errfile', 'rptfile', 'ndigit', 'taufac', 'sstol', 'partol', 'maxit', 'stpb', 'stpd', 'sclb', 'scld', 'work', 'iwork'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data', 'model', 'beta0', 'delta0', 'ifixb', 'ifixx', 'job', 'iprint', 'errfile', 'rptfile', 'ndigit', 'taufac', 'sstol', 'partol', 'maxit', 'stpb', 'stpd', 'sclb', 'scld', 'work', 'iwork'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 731):
        
        # Assigning a Name to a Attribute (line 731):
        # Getting the type of 'data' (line 731)
        data_164415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 20), 'data')
        # Getting the type of 'self' (line 731)
        self_164416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'self')
        # Setting the type of the member 'data' of a type (line 731)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 8), self_164416, 'data', data_164415)
        
        # Assigning a Name to a Attribute (line 732):
        
        # Assigning a Name to a Attribute (line 732):
        # Getting the type of 'model' (line 732)
        model_164417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 21), 'model')
        # Getting the type of 'self' (line 732)
        self_164418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'self')
        # Setting the type of the member 'model' of a type (line 732)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 8), self_164418, 'model', model_164417)
        
        # Type idiom detected: calculating its left and rigth part (line 734)
        # Getting the type of 'beta0' (line 734)
        beta0_164419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 11), 'beta0')
        # Getting the type of 'None' (line 734)
        None_164420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 20), 'None')
        
        (may_be_164421, more_types_in_union_164422) = may_be_none(beta0_164419, None_164420)

        if may_be_164421:

            if more_types_in_union_164422:
                # Runtime conditional SSA (line 734)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'self' (line 735)
            self_164423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 15), 'self')
            # Obtaining the member 'model' of a type (line 735)
            model_164424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 15), self_164423, 'model')
            # Obtaining the member 'estimate' of a type (line 735)
            estimate_164425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 15), model_164424, 'estimate')
            # Getting the type of 'None' (line 735)
            None_164426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 42), 'None')
            # Applying the binary operator 'isnot' (line 735)
            result_is_not_164427 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 15), 'isnot', estimate_164425, None_164426)
            
            # Testing the type of an if condition (line 735)
            if_condition_164428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 735, 12), result_is_not_164427)
            # Assigning a type to the variable 'if_condition_164428' (line 735)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'if_condition_164428', if_condition_164428)
            # SSA begins for if statement (line 735)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 736):
            
            # Assigning a Call to a Attribute (line 736):
            
            # Call to _conv(...): (line 736)
            # Processing the call arguments (line 736)
            
            # Call to estimate(...): (line 736)
            # Processing the call arguments (line 736)
            # Getting the type of 'self' (line 736)
            self_164433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 55), 'self', False)
            # Obtaining the member 'data' of a type (line 736)
            data_164434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 55), self_164433, 'data')
            # Processing the call keyword arguments (line 736)
            kwargs_164435 = {}
            # Getting the type of 'self' (line 736)
            self_164430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 35), 'self', False)
            # Obtaining the member 'model' of a type (line 736)
            model_164431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 35), self_164430, 'model')
            # Obtaining the member 'estimate' of a type (line 736)
            estimate_164432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 35), model_164431, 'estimate')
            # Calling estimate(args, kwargs) (line 736)
            estimate_call_result_164436 = invoke(stypy.reporting.localization.Localization(__file__, 736, 35), estimate_164432, *[data_164434], **kwargs_164435)
            
            # Processing the call keyword arguments (line 736)
            kwargs_164437 = {}
            # Getting the type of '_conv' (line 736)
            _conv_164429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 29), '_conv', False)
            # Calling _conv(args, kwargs) (line 736)
            _conv_call_result_164438 = invoke(stypy.reporting.localization.Localization(__file__, 736, 29), _conv_164429, *[estimate_call_result_164436], **kwargs_164437)
            
            # Getting the type of 'self' (line 736)
            self_164439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 16), 'self')
            # Setting the type of the member 'beta0' of a type (line 736)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 16), self_164439, 'beta0', _conv_call_result_164438)
            # SSA branch for the else part of an if statement (line 735)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 738)
            # Processing the call arguments (line 738)
            str_164441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 18), 'str', 'must specify beta0 or provide an estimater with the model')
            # Processing the call keyword arguments (line 738)
            kwargs_164442 = {}
            # Getting the type of 'ValueError' (line 738)
            ValueError_164440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 738)
            ValueError_call_result_164443 = invoke(stypy.reporting.localization.Localization(__file__, 738, 22), ValueError_164440, *[str_164441], **kwargs_164442)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 738, 16), ValueError_call_result_164443, 'raise parameter', BaseException)
            # SSA join for if statement (line 735)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_164422:
                # Runtime conditional SSA for else branch (line 734)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_164421) or more_types_in_union_164422):
            
            # Assigning a Call to a Attribute (line 742):
            
            # Assigning a Call to a Attribute (line 742):
            
            # Call to _conv(...): (line 742)
            # Processing the call arguments (line 742)
            # Getting the type of 'beta0' (line 742)
            beta0_164445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 31), 'beta0', False)
            # Processing the call keyword arguments (line 742)
            kwargs_164446 = {}
            # Getting the type of '_conv' (line 742)
            _conv_164444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 25), '_conv', False)
            # Calling _conv(args, kwargs) (line 742)
            _conv_call_result_164447 = invoke(stypy.reporting.localization.Localization(__file__, 742, 25), _conv_164444, *[beta0_164445], **kwargs_164446)
            
            # Getting the type of 'self' (line 742)
            self_164448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 12), 'self')
            # Setting the type of the member 'beta0' of a type (line 742)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 12), self_164448, 'beta0', _conv_call_result_164447)

            if (may_be_164421 and more_types_in_union_164422):
                # SSA join for if statement (line 734)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 744):
        
        # Assigning a Call to a Attribute (line 744):
        
        # Call to _conv(...): (line 744)
        # Processing the call arguments (line 744)
        # Getting the type of 'delta0' (line 744)
        delta0_164450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 28), 'delta0', False)
        # Processing the call keyword arguments (line 744)
        kwargs_164451 = {}
        # Getting the type of '_conv' (line 744)
        _conv_164449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 22), '_conv', False)
        # Calling _conv(args, kwargs) (line 744)
        _conv_call_result_164452 = invoke(stypy.reporting.localization.Localization(__file__, 744, 22), _conv_164449, *[delta0_164450], **kwargs_164451)
        
        # Getting the type of 'self' (line 744)
        self_164453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'self')
        # Setting the type of the member 'delta0' of a type (line 744)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 8), self_164453, 'delta0', _conv_call_result_164452)
        
        # Assigning a Call to a Attribute (line 748):
        
        # Assigning a Call to a Attribute (line 748):
        
        # Call to _conv(...): (line 748)
        # Processing the call arguments (line 748)
        # Getting the type of 'ifixx' (line 748)
        ifixx_164455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 27), 'ifixx', False)
        # Processing the call keyword arguments (line 748)
        # Getting the type of 'numpy' (line 748)
        numpy_164456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 40), 'numpy', False)
        # Obtaining the member 'int32' of a type (line 748)
        int32_164457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 40), numpy_164456, 'int32')
        keyword_164458 = int32_164457
        kwargs_164459 = {'dtype': keyword_164458}
        # Getting the type of '_conv' (line 748)
        _conv_164454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 21), '_conv', False)
        # Calling _conv(args, kwargs) (line 748)
        _conv_call_result_164460 = invoke(stypy.reporting.localization.Localization(__file__, 748, 21), _conv_164454, *[ifixx_164455], **kwargs_164459)
        
        # Getting the type of 'self' (line 748)
        self_164461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'self')
        # Setting the type of the member 'ifixx' of a type (line 748)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 8), self_164461, 'ifixx', _conv_call_result_164460)
        
        # Assigning a Call to a Attribute (line 749):
        
        # Assigning a Call to a Attribute (line 749):
        
        # Call to _conv(...): (line 749)
        # Processing the call arguments (line 749)
        # Getting the type of 'ifixb' (line 749)
        ifixb_164463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 27), 'ifixb', False)
        # Processing the call keyword arguments (line 749)
        # Getting the type of 'numpy' (line 749)
        numpy_164464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 40), 'numpy', False)
        # Obtaining the member 'int32' of a type (line 749)
        int32_164465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 40), numpy_164464, 'int32')
        keyword_164466 = int32_164465
        kwargs_164467 = {'dtype': keyword_164466}
        # Getting the type of '_conv' (line 749)
        _conv_164462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 21), '_conv', False)
        # Calling _conv(args, kwargs) (line 749)
        _conv_call_result_164468 = invoke(stypy.reporting.localization.Localization(__file__, 749, 21), _conv_164462, *[ifixb_164463], **kwargs_164467)
        
        # Getting the type of 'self' (line 749)
        self_164469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'self')
        # Setting the type of the member 'ifixb' of a type (line 749)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 8), self_164469, 'ifixb', _conv_call_result_164468)
        
        # Assigning a Name to a Attribute (line 750):
        
        # Assigning a Name to a Attribute (line 750):
        # Getting the type of 'job' (line 750)
        job_164470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 19), 'job')
        # Getting the type of 'self' (line 750)
        self_164471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'self')
        # Setting the type of the member 'job' of a type (line 750)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 8), self_164471, 'job', job_164470)
        
        # Assigning a Name to a Attribute (line 751):
        
        # Assigning a Name to a Attribute (line 751):
        # Getting the type of 'iprint' (line 751)
        iprint_164472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 22), 'iprint')
        # Getting the type of 'self' (line 751)
        self_164473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'self')
        # Setting the type of the member 'iprint' of a type (line 751)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 8), self_164473, 'iprint', iprint_164472)
        
        # Assigning a Name to a Attribute (line 752):
        
        # Assigning a Name to a Attribute (line 752):
        # Getting the type of 'errfile' (line 752)
        errfile_164474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 23), 'errfile')
        # Getting the type of 'self' (line 752)
        self_164475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'self')
        # Setting the type of the member 'errfile' of a type (line 752)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 8), self_164475, 'errfile', errfile_164474)
        
        # Assigning a Name to a Attribute (line 753):
        
        # Assigning a Name to a Attribute (line 753):
        # Getting the type of 'rptfile' (line 753)
        rptfile_164476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 23), 'rptfile')
        # Getting the type of 'self' (line 753)
        self_164477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'self')
        # Setting the type of the member 'rptfile' of a type (line 753)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 8), self_164477, 'rptfile', rptfile_164476)
        
        # Assigning a Name to a Attribute (line 754):
        
        # Assigning a Name to a Attribute (line 754):
        # Getting the type of 'ndigit' (line 754)
        ndigit_164478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 22), 'ndigit')
        # Getting the type of 'self' (line 754)
        self_164479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'self')
        # Setting the type of the member 'ndigit' of a type (line 754)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 8), self_164479, 'ndigit', ndigit_164478)
        
        # Assigning a Name to a Attribute (line 755):
        
        # Assigning a Name to a Attribute (line 755):
        # Getting the type of 'taufac' (line 755)
        taufac_164480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 22), 'taufac')
        # Getting the type of 'self' (line 755)
        self_164481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'self')
        # Setting the type of the member 'taufac' of a type (line 755)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 8), self_164481, 'taufac', taufac_164480)
        
        # Assigning a Name to a Attribute (line 756):
        
        # Assigning a Name to a Attribute (line 756):
        # Getting the type of 'sstol' (line 756)
        sstol_164482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 21), 'sstol')
        # Getting the type of 'self' (line 756)
        self_164483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'self')
        # Setting the type of the member 'sstol' of a type (line 756)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 8), self_164483, 'sstol', sstol_164482)
        
        # Assigning a Name to a Attribute (line 757):
        
        # Assigning a Name to a Attribute (line 757):
        # Getting the type of 'partol' (line 757)
        partol_164484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 22), 'partol')
        # Getting the type of 'self' (line 757)
        self_164485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'self')
        # Setting the type of the member 'partol' of a type (line 757)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 8), self_164485, 'partol', partol_164484)
        
        # Assigning a Name to a Attribute (line 758):
        
        # Assigning a Name to a Attribute (line 758):
        # Getting the type of 'maxit' (line 758)
        maxit_164486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 21), 'maxit')
        # Getting the type of 'self' (line 758)
        self_164487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'self')
        # Setting the type of the member 'maxit' of a type (line 758)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 8), self_164487, 'maxit', maxit_164486)
        
        # Assigning a Call to a Attribute (line 759):
        
        # Assigning a Call to a Attribute (line 759):
        
        # Call to _conv(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'stpb' (line 759)
        stpb_164489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 26), 'stpb', False)
        # Processing the call keyword arguments (line 759)
        kwargs_164490 = {}
        # Getting the type of '_conv' (line 759)
        _conv_164488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 20), '_conv', False)
        # Calling _conv(args, kwargs) (line 759)
        _conv_call_result_164491 = invoke(stypy.reporting.localization.Localization(__file__, 759, 20), _conv_164488, *[stpb_164489], **kwargs_164490)
        
        # Getting the type of 'self' (line 759)
        self_164492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'self')
        # Setting the type of the member 'stpb' of a type (line 759)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 8), self_164492, 'stpb', _conv_call_result_164491)
        
        # Assigning a Call to a Attribute (line 760):
        
        # Assigning a Call to a Attribute (line 760):
        
        # Call to _conv(...): (line 760)
        # Processing the call arguments (line 760)
        # Getting the type of 'stpd' (line 760)
        stpd_164494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 26), 'stpd', False)
        # Processing the call keyword arguments (line 760)
        kwargs_164495 = {}
        # Getting the type of '_conv' (line 760)
        _conv_164493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 20), '_conv', False)
        # Calling _conv(args, kwargs) (line 760)
        _conv_call_result_164496 = invoke(stypy.reporting.localization.Localization(__file__, 760, 20), _conv_164493, *[stpd_164494], **kwargs_164495)
        
        # Getting the type of 'self' (line 760)
        self_164497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'self')
        # Setting the type of the member 'stpd' of a type (line 760)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), self_164497, 'stpd', _conv_call_result_164496)
        
        # Assigning a Call to a Attribute (line 761):
        
        # Assigning a Call to a Attribute (line 761):
        
        # Call to _conv(...): (line 761)
        # Processing the call arguments (line 761)
        # Getting the type of 'sclb' (line 761)
        sclb_164499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 26), 'sclb', False)
        # Processing the call keyword arguments (line 761)
        kwargs_164500 = {}
        # Getting the type of '_conv' (line 761)
        _conv_164498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 20), '_conv', False)
        # Calling _conv(args, kwargs) (line 761)
        _conv_call_result_164501 = invoke(stypy.reporting.localization.Localization(__file__, 761, 20), _conv_164498, *[sclb_164499], **kwargs_164500)
        
        # Getting the type of 'self' (line 761)
        self_164502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'self')
        # Setting the type of the member 'sclb' of a type (line 761)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 8), self_164502, 'sclb', _conv_call_result_164501)
        
        # Assigning a Call to a Attribute (line 762):
        
        # Assigning a Call to a Attribute (line 762):
        
        # Call to _conv(...): (line 762)
        # Processing the call arguments (line 762)
        # Getting the type of 'scld' (line 762)
        scld_164504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 26), 'scld', False)
        # Processing the call keyword arguments (line 762)
        kwargs_164505 = {}
        # Getting the type of '_conv' (line 762)
        _conv_164503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 20), '_conv', False)
        # Calling _conv(args, kwargs) (line 762)
        _conv_call_result_164506 = invoke(stypy.reporting.localization.Localization(__file__, 762, 20), _conv_164503, *[scld_164504], **kwargs_164505)
        
        # Getting the type of 'self' (line 762)
        self_164507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'self')
        # Setting the type of the member 'scld' of a type (line 762)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 8), self_164507, 'scld', _conv_call_result_164506)
        
        # Assigning a Call to a Attribute (line 763):
        
        # Assigning a Call to a Attribute (line 763):
        
        # Call to _conv(...): (line 763)
        # Processing the call arguments (line 763)
        # Getting the type of 'work' (line 763)
        work_164509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 26), 'work', False)
        # Processing the call keyword arguments (line 763)
        kwargs_164510 = {}
        # Getting the type of '_conv' (line 763)
        _conv_164508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 20), '_conv', False)
        # Calling _conv(args, kwargs) (line 763)
        _conv_call_result_164511 = invoke(stypy.reporting.localization.Localization(__file__, 763, 20), _conv_164508, *[work_164509], **kwargs_164510)
        
        # Getting the type of 'self' (line 763)
        self_164512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 8), 'self')
        # Setting the type of the member 'work' of a type (line 763)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 8), self_164512, 'work', _conv_call_result_164511)
        
        # Assigning a Call to a Attribute (line 764):
        
        # Assigning a Call to a Attribute (line 764):
        
        # Call to _conv(...): (line 764)
        # Processing the call arguments (line 764)
        # Getting the type of 'iwork' (line 764)
        iwork_164514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 27), 'iwork', False)
        # Processing the call keyword arguments (line 764)
        kwargs_164515 = {}
        # Getting the type of '_conv' (line 764)
        _conv_164513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 21), '_conv', False)
        # Calling _conv(args, kwargs) (line 764)
        _conv_call_result_164516 = invoke(stypy.reporting.localization.Localization(__file__, 764, 21), _conv_164513, *[iwork_164514], **kwargs_164515)
        
        # Getting the type of 'self' (line 764)
        self_164517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'self')
        # Setting the type of the member 'iwork' of a type (line 764)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 8), self_164517, 'iwork', _conv_call_result_164516)
        
        # Assigning a Name to a Attribute (line 766):
        
        # Assigning a Name to a Attribute (line 766):
        # Getting the type of 'None' (line 766)
        None_164518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 22), 'None')
        # Getting the type of 'self' (line 766)
        self_164519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'self')
        # Setting the type of the member 'output' of a type (line 766)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 8), self_164519, 'output', None_164518)
        
        # Call to _check(...): (line 768)
        # Processing the call keyword arguments (line 768)
        kwargs_164522 = {}
        # Getting the type of 'self' (line 768)
        self_164520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 768)
        _check_164521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 8), self_164520, '_check')
        # Calling _check(args, kwargs) (line 768)
        _check_call_result_164523 = invoke(stypy.reporting.localization.Localization(__file__, 768, 8), _check_164521, *[], **kwargs_164522)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check'
        module_type_store = module_type_store.open_function_context('_check', 770, 4, False)
        # Assigning a type to the variable 'self' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODR._check.__dict__.__setitem__('stypy_localization', localization)
        ODR._check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODR._check.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODR._check.__dict__.__setitem__('stypy_function_name', 'ODR._check')
        ODR._check.__dict__.__setitem__('stypy_param_names_list', [])
        ODR._check.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODR._check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODR._check.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODR._check.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODR._check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODR._check.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODR._check', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check(...)' code ##################

        str_164524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, (-1)), 'str', " Check the inputs for consistency, but don't bother checking things\n        that the builtin function odr will check.\n        ")
        
        # Assigning a Call to a Name (line 775):
        
        # Assigning a Call to a Name (line 775):
        
        # Call to list(...): (line 775)
        # Processing the call arguments (line 775)
        # Getting the type of 'self' (line 775)
        self_164526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 19), 'self', False)
        # Obtaining the member 'data' of a type (line 775)
        data_164527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 19), self_164526, 'data')
        # Obtaining the member 'x' of a type (line 775)
        x_164528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 19), data_164527, 'x')
        # Obtaining the member 'shape' of a type (line 775)
        shape_164529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 19), x_164528, 'shape')
        # Processing the call keyword arguments (line 775)
        kwargs_164530 = {}
        # Getting the type of 'list' (line 775)
        list_164525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 14), 'list', False)
        # Calling list(args, kwargs) (line 775)
        list_call_result_164531 = invoke(stypy.reporting.localization.Localization(__file__, 775, 14), list_164525, *[shape_164529], **kwargs_164530)
        
        # Assigning a type to the variable 'x_s' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'x_s', list_call_result_164531)
        
        
        # Call to isinstance(...): (line 777)
        # Processing the call arguments (line 777)
        # Getting the type of 'self' (line 777)
        self_164533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 22), 'self', False)
        # Obtaining the member 'data' of a type (line 777)
        data_164534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 22), self_164533, 'data')
        # Obtaining the member 'y' of a type (line 777)
        y_164535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 22), data_164534, 'y')
        # Getting the type of 'numpy' (line 777)
        numpy_164536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 35), 'numpy', False)
        # Obtaining the member 'ndarray' of a type (line 777)
        ndarray_164537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 35), numpy_164536, 'ndarray')
        # Processing the call keyword arguments (line 777)
        kwargs_164538 = {}
        # Getting the type of 'isinstance' (line 777)
        isinstance_164532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 777)
        isinstance_call_result_164539 = invoke(stypy.reporting.localization.Localization(__file__, 777, 11), isinstance_164532, *[y_164535, ndarray_164537], **kwargs_164538)
        
        # Testing the type of an if condition (line 777)
        if_condition_164540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 777, 8), isinstance_call_result_164539)
        # Assigning a type to the variable 'if_condition_164540' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'if_condition_164540', if_condition_164540)
        # SSA begins for if statement (line 777)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 778):
        
        # Assigning a Call to a Name (line 778):
        
        # Call to list(...): (line 778)
        # Processing the call arguments (line 778)
        # Getting the type of 'self' (line 778)
        self_164542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 23), 'self', False)
        # Obtaining the member 'data' of a type (line 778)
        data_164543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 23), self_164542, 'data')
        # Obtaining the member 'y' of a type (line 778)
        y_164544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 23), data_164543, 'y')
        # Obtaining the member 'shape' of a type (line 778)
        shape_164545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 23), y_164544, 'shape')
        # Processing the call keyword arguments (line 778)
        kwargs_164546 = {}
        # Getting the type of 'list' (line 778)
        list_164541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 18), 'list', False)
        # Calling list(args, kwargs) (line 778)
        list_call_result_164547 = invoke(stypy.reporting.localization.Localization(__file__, 778, 18), list_164541, *[shape_164545], **kwargs_164546)
        
        # Assigning a type to the variable 'y_s' (line 778)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 12), 'y_s', list_call_result_164547)
        
        # Getting the type of 'self' (line 779)
        self_164548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 15), 'self')
        # Obtaining the member 'model' of a type (line 779)
        model_164549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 15), self_164548, 'model')
        # Obtaining the member 'implicit' of a type (line 779)
        implicit_164550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 15), model_164549, 'implicit')
        # Testing the type of an if condition (line 779)
        if_condition_164551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 779, 12), implicit_164550)
        # Assigning a type to the variable 'if_condition_164551' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 12), 'if_condition_164551', if_condition_164551)
        # SSA begins for if statement (line 779)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to OdrError(...): (line 780)
        # Processing the call arguments (line 780)
        str_164553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 31), 'str', 'an implicit model cannot use response data')
        # Processing the call keyword arguments (line 780)
        kwargs_164554 = {}
        # Getting the type of 'OdrError' (line 780)
        OdrError_164552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 22), 'OdrError', False)
        # Calling OdrError(args, kwargs) (line 780)
        OdrError_call_result_164555 = invoke(stypy.reporting.localization.Localization(__file__, 780, 22), OdrError_164552, *[str_164553], **kwargs_164554)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 780, 16), OdrError_call_result_164555, 'raise parameter', BaseException)
        # SSA join for if statement (line 779)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 777)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 783):
        
        # Assigning a List to a Name (line 783):
        
        # Obtaining an instance of the builtin type 'list' (line 783)
        list_164556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 783)
        # Adding element type (line 783)
        # Getting the type of 'self' (line 783)
        self_164557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 19), 'self')
        # Obtaining the member 'data' of a type (line 783)
        data_164558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 19), self_164557, 'data')
        # Obtaining the member 'y' of a type (line 783)
        y_164559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 19), data_164558, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 18), list_164556, y_164559)
        # Adding element type (line 783)
        
        # Obtaining the type of the subscript
        int_164560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 36), 'int')
        # Getting the type of 'x_s' (line 783)
        x_s_164561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 32), 'x_s')
        # Obtaining the member '__getitem__' of a type (line 783)
        getitem___164562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 32), x_s_164561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 783)
        subscript_call_result_164563 = invoke(stypy.reporting.localization.Localization(__file__, 783, 32), getitem___164562, int_164560)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 18), list_164556, subscript_call_result_164563)
        
        # Assigning a type to the variable 'y_s' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'y_s', list_164556)
        
        
        # Getting the type of 'self' (line 784)
        self_164564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 19), 'self')
        # Obtaining the member 'model' of a type (line 784)
        model_164565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 19), self_164564, 'model')
        # Obtaining the member 'implicit' of a type (line 784)
        implicit_164566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 19), model_164565, 'implicit')
        # Applying the 'not' unary operator (line 784)
        result_not__164567 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 15), 'not', implicit_164566)
        
        # Testing the type of an if condition (line 784)
        if_condition_164568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 784, 12), result_not__164567)
        # Assigning a type to the variable 'if_condition_164568' (line 784)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 12), 'if_condition_164568', if_condition_164568)
        # SSA begins for if statement (line 784)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to OdrError(...): (line 785)
        # Processing the call arguments (line 785)
        str_164570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 31), 'str', 'an explicit model needs response data')
        # Processing the call keyword arguments (line 785)
        kwargs_164571 = {}
        # Getting the type of 'OdrError' (line 785)
        OdrError_164569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 22), 'OdrError', False)
        # Calling OdrError(args, kwargs) (line 785)
        OdrError_call_result_164572 = invoke(stypy.reporting.localization.Localization(__file__, 785, 22), OdrError_164569, *[str_164570], **kwargs_164571)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 785, 16), OdrError_call_result_164572, 'raise parameter', BaseException)
        # SSA join for if statement (line 784)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_job(...): (line 786)
        # Processing the call keyword arguments (line 786)
        int_164575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 34), 'int')
        keyword_164576 = int_164575
        kwargs_164577 = {'fit_type': keyword_164576}
        # Getting the type of 'self' (line 786)
        self_164573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 12), 'self', False)
        # Obtaining the member 'set_job' of a type (line 786)
        set_job_164574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 12), self_164573, 'set_job')
        # Calling set_job(args, kwargs) (line 786)
        set_job_call_result_164578 = invoke(stypy.reporting.localization.Localization(__file__, 786, 12), set_job_164574, *[], **kwargs_164577)
        
        # SSA join for if statement (line 777)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_164579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 15), 'int')
        # Getting the type of 'x_s' (line 788)
        x_s_164580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 11), 'x_s')
        # Obtaining the member '__getitem__' of a type (line 788)
        getitem___164581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 11), x_s_164580, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 788)
        subscript_call_result_164582 = invoke(stypy.reporting.localization.Localization(__file__, 788, 11), getitem___164581, int_164579)
        
        
        # Obtaining the type of the subscript
        int_164583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 26), 'int')
        # Getting the type of 'y_s' (line 788)
        y_s_164584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 22), 'y_s')
        # Obtaining the member '__getitem__' of a type (line 788)
        getitem___164585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 22), y_s_164584, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 788)
        subscript_call_result_164586 = invoke(stypy.reporting.localization.Localization(__file__, 788, 22), getitem___164585, int_164583)
        
        # Applying the binary operator '!=' (line 788)
        result_ne_164587 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 11), '!=', subscript_call_result_164582, subscript_call_result_164586)
        
        # Testing the type of an if condition (line 788)
        if_condition_164588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 788, 8), result_ne_164587)
        # Assigning a type to the variable 'if_condition_164588' (line 788)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'if_condition_164588', if_condition_164588)
        # SSA begins for if statement (line 788)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to OdrError(...): (line 789)
        # Processing the call arguments (line 789)
        str_164590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 27), 'str', 'number of observations do not match')
        # Processing the call keyword arguments (line 789)
        kwargs_164591 = {}
        # Getting the type of 'OdrError' (line 789)
        OdrError_164589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 18), 'OdrError', False)
        # Calling OdrError(args, kwargs) (line 789)
        OdrError_call_result_164592 = invoke(stypy.reporting.localization.Localization(__file__, 789, 18), OdrError_164589, *[str_164590], **kwargs_164591)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 789, 12), OdrError_call_result_164592, 'raise parameter', BaseException)
        # SSA join for if statement (line 788)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 791):
        
        # Assigning a Subscript to a Name (line 791):
        
        # Obtaining the type of the subscript
        int_164593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 16), 'int')
        # Getting the type of 'x_s' (line 791)
        x_s_164594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'x_s')
        # Obtaining the member '__getitem__' of a type (line 791)
        getitem___164595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 12), x_s_164594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 791)
        subscript_call_result_164596 = invoke(stypy.reporting.localization.Localization(__file__, 791, 12), getitem___164595, int_164593)
        
        # Assigning a type to the variable 'n' (line 791)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 8), 'n', subscript_call_result_164596)
        
        
        
        # Call to len(...): (line 793)
        # Processing the call arguments (line 793)
        # Getting the type of 'x_s' (line 793)
        x_s_164598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 15), 'x_s', False)
        # Processing the call keyword arguments (line 793)
        kwargs_164599 = {}
        # Getting the type of 'len' (line 793)
        len_164597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 11), 'len', False)
        # Calling len(args, kwargs) (line 793)
        len_call_result_164600 = invoke(stypy.reporting.localization.Localization(__file__, 793, 11), len_164597, *[x_s_164598], **kwargs_164599)
        
        int_164601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 23), 'int')
        # Applying the binary operator '==' (line 793)
        result_eq_164602 = python_operator(stypy.reporting.localization.Localization(__file__, 793, 11), '==', len_call_result_164600, int_164601)
        
        # Testing the type of an if condition (line 793)
        if_condition_164603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 793, 8), result_eq_164602)
        # Assigning a type to the variable 'if_condition_164603' (line 793)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'if_condition_164603', if_condition_164603)
        # SSA begins for if statement (line 793)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 794):
        
        # Assigning a Subscript to a Name (line 794):
        
        # Obtaining the type of the subscript
        int_164604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 20), 'int')
        # Getting the type of 'x_s' (line 794)
        x_s_164605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 16), 'x_s')
        # Obtaining the member '__getitem__' of a type (line 794)
        getitem___164606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 16), x_s_164605, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 794)
        subscript_call_result_164607 = invoke(stypy.reporting.localization.Localization(__file__, 794, 16), getitem___164606, int_164604)
        
        # Assigning a type to the variable 'm' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'm', subscript_call_result_164607)
        # SSA branch for the else part of an if statement (line 793)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 796):
        
        # Assigning a Num to a Name (line 796):
        int_164608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 16), 'int')
        # Assigning a type to the variable 'm' (line 796)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 12), 'm', int_164608)
        # SSA join for if statement (line 793)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 797)
        # Processing the call arguments (line 797)
        # Getting the type of 'y_s' (line 797)
        y_s_164610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 15), 'y_s', False)
        # Processing the call keyword arguments (line 797)
        kwargs_164611 = {}
        # Getting the type of 'len' (line 797)
        len_164609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 11), 'len', False)
        # Calling len(args, kwargs) (line 797)
        len_call_result_164612 = invoke(stypy.reporting.localization.Localization(__file__, 797, 11), len_164609, *[y_s_164610], **kwargs_164611)
        
        int_164613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 23), 'int')
        # Applying the binary operator '==' (line 797)
        result_eq_164614 = python_operator(stypy.reporting.localization.Localization(__file__, 797, 11), '==', len_call_result_164612, int_164613)
        
        # Testing the type of an if condition (line 797)
        if_condition_164615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 797, 8), result_eq_164614)
        # Assigning a type to the variable 'if_condition_164615' (line 797)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'if_condition_164615', if_condition_164615)
        # SSA begins for if statement (line 797)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 798):
        
        # Assigning a Subscript to a Name (line 798):
        
        # Obtaining the type of the subscript
        int_164616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 20), 'int')
        # Getting the type of 'y_s' (line 798)
        y_s_164617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 16), 'y_s')
        # Obtaining the member '__getitem__' of a type (line 798)
        getitem___164618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 16), y_s_164617, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 798)
        subscript_call_result_164619 = invoke(stypy.reporting.localization.Localization(__file__, 798, 16), getitem___164618, int_164616)
        
        # Assigning a type to the variable 'q' (line 798)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 12), 'q', subscript_call_result_164619)
        # SSA branch for the else part of an if statement (line 797)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 800):
        
        # Assigning a Num to a Name (line 800):
        int_164620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 16), 'int')
        # Assigning a type to the variable 'q' (line 800)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 12), 'q', int_164620)
        # SSA join for if statement (line 797)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 802):
        
        # Assigning a Call to a Name (line 802):
        
        # Call to len(...): (line 802)
        # Processing the call arguments (line 802)
        # Getting the type of 'self' (line 802)
        self_164622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 16), 'self', False)
        # Obtaining the member 'beta0' of a type (line 802)
        beta0_164623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 16), self_164622, 'beta0')
        # Processing the call keyword arguments (line 802)
        kwargs_164624 = {}
        # Getting the type of 'len' (line 802)
        len_164621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 12), 'len', False)
        # Calling len(args, kwargs) (line 802)
        len_call_result_164625 = invoke(stypy.reporting.localization.Localization(__file__, 802, 12), len_164621, *[beta0_164623], **kwargs_164624)
        
        # Assigning a type to the variable 'p' (line 802)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'p', len_call_result_164625)
        
        # Assigning a List to a Name (line 806):
        
        # Assigning a List to a Name (line 806):
        
        # Obtaining an instance of the builtin type 'list' (line 806)
        list_164626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 806)
        # Adding element type (line 806)
        
        # Obtaining an instance of the builtin type 'tuple' (line 806)
        tuple_164627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 806)
        # Adding element type (line 806)
        # Getting the type of 'q' (line 806)
        q_164628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 22), 'q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 22), tuple_164627, q_164628)
        # Adding element type (line 806)
        # Getting the type of 'n' (line 806)
        n_164629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 25), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 22), tuple_164627, n_164629)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 20), list_164626, tuple_164627)
        
        # Assigning a type to the variable 'fcn_perms' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'fcn_perms', list_164626)
        
        # Assigning a List to a Name (line 807):
        
        # Assigning a List to a Name (line 807):
        
        # Obtaining an instance of the builtin type 'list' (line 807)
        list_164630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 807)
        # Adding element type (line 807)
        
        # Obtaining an instance of the builtin type 'tuple' (line 807)
        tuple_164631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 807)
        # Adding element type (line 807)
        # Getting the type of 'q' (line 807)
        q_164632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 24), 'q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 24), tuple_164631, q_164632)
        # Adding element type (line 807)
        # Getting the type of 'm' (line 807)
        m_164633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 27), 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 24), tuple_164631, m_164633)
        # Adding element type (line 807)
        # Getting the type of 'n' (line 807)
        n_164634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 30), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 24), tuple_164631, n_164634)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 22), list_164630, tuple_164631)
        
        # Assigning a type to the variable 'fjacd_perms' (line 807)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'fjacd_perms', list_164630)
        
        # Assigning a List to a Name (line 808):
        
        # Assigning a List to a Name (line 808):
        
        # Obtaining an instance of the builtin type 'list' (line 808)
        list_164635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 808)
        # Adding element type (line 808)
        
        # Obtaining an instance of the builtin type 'tuple' (line 808)
        tuple_164636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 808)
        # Adding element type (line 808)
        # Getting the type of 'q' (line 808)
        q_164637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 24), 'q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 24), tuple_164636, q_164637)
        # Adding element type (line 808)
        # Getting the type of 'p' (line 808)
        p_164638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 27), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 24), tuple_164636, p_164638)
        # Adding element type (line 808)
        # Getting the type of 'n' (line 808)
        n_164639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 30), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 24), tuple_164636, n_164639)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 22), list_164635, tuple_164636)
        
        # Assigning a type to the variable 'fjacb_perms' (line 808)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'fjacb_perms', list_164635)
        
        
        # Getting the type of 'q' (line 810)
        q_164640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 11), 'q')
        int_164641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 16), 'int')
        # Applying the binary operator '==' (line 810)
        result_eq_164642 = python_operator(stypy.reporting.localization.Localization(__file__, 810, 11), '==', q_164640, int_164641)
        
        # Testing the type of an if condition (line 810)
        if_condition_164643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 810, 8), result_eq_164642)
        # Assigning a type to the variable 'if_condition_164643' (line 810)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'if_condition_164643', if_condition_164643)
        # SSA begins for if statement (line 810)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 811)
        # Processing the call arguments (line 811)
        
        # Obtaining an instance of the builtin type 'tuple' (line 811)
        tuple_164646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 811)
        # Adding element type (line 811)
        # Getting the type of 'n' (line 811)
        n_164647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 30), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 30), tuple_164646, n_164647)
        
        # Processing the call keyword arguments (line 811)
        kwargs_164648 = {}
        # Getting the type of 'fcn_perms' (line 811)
        fcn_perms_164644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'fcn_perms', False)
        # Obtaining the member 'append' of a type (line 811)
        append_164645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 12), fcn_perms_164644, 'append')
        # Calling append(args, kwargs) (line 811)
        append_call_result_164649 = invoke(stypy.reporting.localization.Localization(__file__, 811, 12), append_164645, *[tuple_164646], **kwargs_164648)
        
        
        # Call to append(...): (line 812)
        # Processing the call arguments (line 812)
        
        # Obtaining an instance of the builtin type 'tuple' (line 812)
        tuple_164652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 812)
        # Adding element type (line 812)
        # Getting the type of 'm' (line 812)
        m_164653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 32), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 32), tuple_164652, m_164653)
        # Adding element type (line 812)
        # Getting the type of 'n' (line 812)
        n_164654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 32), tuple_164652, n_164654)
        
        # Processing the call keyword arguments (line 812)
        kwargs_164655 = {}
        # Getting the type of 'fjacd_perms' (line 812)
        fjacd_perms_164650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'fjacd_perms', False)
        # Obtaining the member 'append' of a type (line 812)
        append_164651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 12), fjacd_perms_164650, 'append')
        # Calling append(args, kwargs) (line 812)
        append_call_result_164656 = invoke(stypy.reporting.localization.Localization(__file__, 812, 12), append_164651, *[tuple_164652], **kwargs_164655)
        
        
        # Call to append(...): (line 813)
        # Processing the call arguments (line 813)
        
        # Obtaining an instance of the builtin type 'tuple' (line 813)
        tuple_164659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 813)
        # Adding element type (line 813)
        # Getting the type of 'p' (line 813)
        p_164660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 32), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 32), tuple_164659, p_164660)
        # Adding element type (line 813)
        # Getting the type of 'n' (line 813)
        n_164661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 32), tuple_164659, n_164661)
        
        # Processing the call keyword arguments (line 813)
        kwargs_164662 = {}
        # Getting the type of 'fjacb_perms' (line 813)
        fjacb_perms_164657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 'fjacb_perms', False)
        # Obtaining the member 'append' of a type (line 813)
        append_164658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 12), fjacb_perms_164657, 'append')
        # Calling append(args, kwargs) (line 813)
        append_call_result_164663 = invoke(stypy.reporting.localization.Localization(__file__, 813, 12), append_164658, *[tuple_164659], **kwargs_164662)
        
        # SSA join for if statement (line 810)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'm' (line 814)
        m_164664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 11), 'm')
        int_164665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 16), 'int')
        # Applying the binary operator '==' (line 814)
        result_eq_164666 = python_operator(stypy.reporting.localization.Localization(__file__, 814, 11), '==', m_164664, int_164665)
        
        # Testing the type of an if condition (line 814)
        if_condition_164667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 814, 8), result_eq_164666)
        # Assigning a type to the variable 'if_condition_164667' (line 814)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'if_condition_164667', if_condition_164667)
        # SSA begins for if statement (line 814)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 815)
        # Processing the call arguments (line 815)
        
        # Obtaining an instance of the builtin type 'tuple' (line 815)
        tuple_164670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 815)
        # Adding element type (line 815)
        # Getting the type of 'q' (line 815)
        q_164671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 32), 'q', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 32), tuple_164670, q_164671)
        # Adding element type (line 815)
        # Getting the type of 'n' (line 815)
        n_164672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 32), tuple_164670, n_164672)
        
        # Processing the call keyword arguments (line 815)
        kwargs_164673 = {}
        # Getting the type of 'fjacd_perms' (line 815)
        fjacd_perms_164668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 12), 'fjacd_perms', False)
        # Obtaining the member 'append' of a type (line 815)
        append_164669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 12), fjacd_perms_164668, 'append')
        # Calling append(args, kwargs) (line 815)
        append_call_result_164674 = invoke(stypy.reporting.localization.Localization(__file__, 815, 12), append_164669, *[tuple_164670], **kwargs_164673)
        
        # SSA join for if statement (line 814)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'p' (line 816)
        p_164675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 11), 'p')
        int_164676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 16), 'int')
        # Applying the binary operator '==' (line 816)
        result_eq_164677 = python_operator(stypy.reporting.localization.Localization(__file__, 816, 11), '==', p_164675, int_164676)
        
        # Testing the type of an if condition (line 816)
        if_condition_164678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 816, 8), result_eq_164677)
        # Assigning a type to the variable 'if_condition_164678' (line 816)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'if_condition_164678', if_condition_164678)
        # SSA begins for if statement (line 816)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 817)
        # Processing the call arguments (line 817)
        
        # Obtaining an instance of the builtin type 'tuple' (line 817)
        tuple_164681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 817)
        # Adding element type (line 817)
        # Getting the type of 'q' (line 817)
        q_164682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 32), 'q', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 817, 32), tuple_164681, q_164682)
        # Adding element type (line 817)
        # Getting the type of 'n' (line 817)
        n_164683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 817, 32), tuple_164681, n_164683)
        
        # Processing the call keyword arguments (line 817)
        kwargs_164684 = {}
        # Getting the type of 'fjacb_perms' (line 817)
        fjacb_perms_164679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'fjacb_perms', False)
        # Obtaining the member 'append' of a type (line 817)
        append_164680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 12), fjacb_perms_164679, 'append')
        # Calling append(args, kwargs) (line 817)
        append_call_result_164685 = invoke(stypy.reporting.localization.Localization(__file__, 817, 12), append_164680, *[tuple_164681], **kwargs_164684)
        
        # SSA join for if statement (line 816)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'm' (line 818)
        m_164686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 11), 'm')
        # Getting the type of 'q' (line 818)
        q_164687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 16), 'q')
        # Applying the binary operator '==' (line 818)
        result_eq_164688 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 11), '==', m_164686, q_164687)
        int_164689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 21), 'int')
        # Applying the binary operator '==' (line 818)
        result_eq_164690 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 11), '==', q_164687, int_164689)
        # Applying the binary operator '&' (line 818)
        result_and__164691 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 11), '&', result_eq_164688, result_eq_164690)
        
        # Testing the type of an if condition (line 818)
        if_condition_164692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 818, 8), result_and__164691)
        # Assigning a type to the variable 'if_condition_164692' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 8), 'if_condition_164692', if_condition_164692)
        # SSA begins for if statement (line 818)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 819)
        # Processing the call arguments (line 819)
        
        # Obtaining an instance of the builtin type 'tuple' (line 819)
        tuple_164695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 819)
        # Adding element type (line 819)
        # Getting the type of 'n' (line 819)
        n_164696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 32), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 819, 32), tuple_164695, n_164696)
        
        # Processing the call keyword arguments (line 819)
        kwargs_164697 = {}
        # Getting the type of 'fjacd_perms' (line 819)
        fjacd_perms_164693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 12), 'fjacd_perms', False)
        # Obtaining the member 'append' of a type (line 819)
        append_164694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 12), fjacd_perms_164693, 'append')
        # Calling append(args, kwargs) (line 819)
        append_call_result_164698 = invoke(stypy.reporting.localization.Localization(__file__, 819, 12), append_164694, *[tuple_164695], **kwargs_164697)
        
        # SSA join for if statement (line 818)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'p' (line 820)
        p_164699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 11), 'p')
        # Getting the type of 'q' (line 820)
        q_164700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 16), 'q')
        # Applying the binary operator '==' (line 820)
        result_eq_164701 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 11), '==', p_164699, q_164700)
        int_164702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 21), 'int')
        # Applying the binary operator '==' (line 820)
        result_eq_164703 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 11), '==', q_164700, int_164702)
        # Applying the binary operator '&' (line 820)
        result_and__164704 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 11), '&', result_eq_164701, result_eq_164703)
        
        # Testing the type of an if condition (line 820)
        if_condition_164705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 820, 8), result_and__164704)
        # Assigning a type to the variable 'if_condition_164705' (line 820)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 8), 'if_condition_164705', if_condition_164705)
        # SSA begins for if statement (line 820)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 821)
        # Processing the call arguments (line 821)
        
        # Obtaining an instance of the builtin type 'tuple' (line 821)
        tuple_164708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 821)
        # Adding element type (line 821)
        # Getting the type of 'n' (line 821)
        n_164709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 32), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 32), tuple_164708, n_164709)
        
        # Processing the call keyword arguments (line 821)
        kwargs_164710 = {}
        # Getting the type of 'fjacb_perms' (line 821)
        fjacb_perms_164706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 12), 'fjacb_perms', False)
        # Obtaining the member 'append' of a type (line 821)
        append_164707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 12), fjacb_perms_164706, 'append')
        # Calling append(args, kwargs) (line 821)
        append_call_result_164711 = invoke(stypy.reporting.localization.Localization(__file__, 821, 12), append_164707, *[tuple_164708], **kwargs_164710)
        
        # SSA join for if statement (line 820)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 826):
        
        # Assigning a Tuple to a Name (line 826):
        
        # Obtaining an instance of the builtin type 'tuple' (line 826)
        tuple_164712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 826)
        # Adding element type (line 826)
        # Getting the type of 'self' (line 826)
        self_164713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 19), 'self')
        # Obtaining the member 'beta0' of a type (line 826)
        beta0_164714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 19), self_164713, 'beta0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 826, 19), tuple_164712, beta0_164714)
        # Adding element type (line 826)
        # Getting the type of 'self' (line 826)
        self_164715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 31), 'self')
        # Obtaining the member 'data' of a type (line 826)
        data_164716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 31), self_164715, 'data')
        # Obtaining the member 'x' of a type (line 826)
        x_164717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 31), data_164716, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 826, 19), tuple_164712, x_164717)
        
        # Assigning a type to the variable 'arglist' (line 826)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 8), 'arglist', tuple_164712)
        
        
        # Getting the type of 'self' (line 827)
        self_164718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 11), 'self')
        # Obtaining the member 'model' of a type (line 827)
        model_164719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 11), self_164718, 'model')
        # Obtaining the member 'extra_args' of a type (line 827)
        extra_args_164720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 11), model_164719, 'extra_args')
        # Getting the type of 'None' (line 827)
        None_164721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 40), 'None')
        # Applying the binary operator 'isnot' (line 827)
        result_is_not_164722 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 11), 'isnot', extra_args_164720, None_164721)
        
        # Testing the type of an if condition (line 827)
        if_condition_164723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 827, 8), result_is_not_164722)
        # Assigning a type to the variable 'if_condition_164723' (line 827)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'if_condition_164723', if_condition_164723)
        # SSA begins for if statement (line 827)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 828):
        
        # Assigning a BinOp to a Name (line 828):
        # Getting the type of 'arglist' (line 828)
        arglist_164724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 22), 'arglist')
        # Getting the type of 'self' (line 828)
        self_164725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 32), 'self')
        # Obtaining the member 'model' of a type (line 828)
        model_164726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 32), self_164725, 'model')
        # Obtaining the member 'extra_args' of a type (line 828)
        extra_args_164727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 32), model_164726, 'extra_args')
        # Applying the binary operator '+' (line 828)
        result_add_164728 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 22), '+', arglist_164724, extra_args_164727)
        
        # Assigning a type to the variable 'arglist' (line 828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 12), 'arglist', result_add_164728)
        # SSA join for if statement (line 827)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 829):
        
        # Assigning a Call to a Name (line 829):
        
        # Call to fcn(...): (line 829)
        # Getting the type of 'arglist' (line 829)
        arglist_164732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 30), 'arglist', False)
        # Processing the call keyword arguments (line 829)
        kwargs_164733 = {}
        # Getting the type of 'self' (line 829)
        self_164729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 14), 'self', False)
        # Obtaining the member 'model' of a type (line 829)
        model_164730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 14), self_164729, 'model')
        # Obtaining the member 'fcn' of a type (line 829)
        fcn_164731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 14), model_164730, 'fcn')
        # Calling fcn(args, kwargs) (line 829)
        fcn_call_result_164734 = invoke(stypy.reporting.localization.Localization(__file__, 829, 14), fcn_164731, *[arglist_164732], **kwargs_164733)
        
        # Assigning a type to the variable 'res' (line 829)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'res', fcn_call_result_164734)
        
        
        # Getting the type of 'res' (line 831)
        res_164735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 11), 'res')
        # Obtaining the member 'shape' of a type (line 831)
        shape_164736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 11), res_164735, 'shape')
        # Getting the type of 'fcn_perms' (line 831)
        fcn_perms_164737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 28), 'fcn_perms')
        # Applying the binary operator 'notin' (line 831)
        result_contains_164738 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 11), 'notin', shape_164736, fcn_perms_164737)
        
        # Testing the type of an if condition (line 831)
        if_condition_164739 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 831, 8), result_contains_164738)
        # Assigning a type to the variable 'if_condition_164739' (line 831)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'if_condition_164739', if_condition_164739)
        # SSA begins for if statement (line 831)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 832)
        # Processing the call arguments (line 832)
        # Getting the type of 'res' (line 832)
        res_164741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 18), 'res', False)
        # Obtaining the member 'shape' of a type (line 832)
        shape_164742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 18), res_164741, 'shape')
        # Processing the call keyword arguments (line 832)
        kwargs_164743 = {}
        # Getting the type of 'print' (line 832)
        print_164740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 12), 'print', False)
        # Calling print(args, kwargs) (line 832)
        print_call_result_164744 = invoke(stypy.reporting.localization.Localization(__file__, 832, 12), print_164740, *[shape_164742], **kwargs_164743)
        
        
        # Call to print(...): (line 833)
        # Processing the call arguments (line 833)
        # Getting the type of 'fcn_perms' (line 833)
        fcn_perms_164746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 18), 'fcn_perms', False)
        # Processing the call keyword arguments (line 833)
        kwargs_164747 = {}
        # Getting the type of 'print' (line 833)
        print_164745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 12), 'print', False)
        # Calling print(args, kwargs) (line 833)
        print_call_result_164748 = invoke(stypy.reporting.localization.Localization(__file__, 833, 12), print_164745, *[fcn_perms_164746], **kwargs_164747)
        
        
        # Call to OdrError(...): (line 834)
        # Processing the call arguments (line 834)
        str_164750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 27), 'str', 'fcn does not output %s-shaped array')
        # Getting the type of 'y_s' (line 834)
        y_s_164751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 67), 'y_s', False)
        # Applying the binary operator '%' (line 834)
        result_mod_164752 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 27), '%', str_164750, y_s_164751)
        
        # Processing the call keyword arguments (line 834)
        kwargs_164753 = {}
        # Getting the type of 'OdrError' (line 834)
        OdrError_164749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 18), 'OdrError', False)
        # Calling OdrError(args, kwargs) (line 834)
        OdrError_call_result_164754 = invoke(stypy.reporting.localization.Localization(__file__, 834, 18), OdrError_164749, *[result_mod_164752], **kwargs_164753)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 834, 12), OdrError_call_result_164754, 'raise parameter', BaseException)
        # SSA join for if statement (line 831)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 836)
        self_164755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 11), 'self')
        # Obtaining the member 'model' of a type (line 836)
        model_164756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 11), self_164755, 'model')
        # Obtaining the member 'fjacd' of a type (line 836)
        fjacd_164757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 11), model_164756, 'fjacd')
        # Getting the type of 'None' (line 836)
        None_164758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 35), 'None')
        # Applying the binary operator 'isnot' (line 836)
        result_is_not_164759 = python_operator(stypy.reporting.localization.Localization(__file__, 836, 11), 'isnot', fjacd_164757, None_164758)
        
        # Testing the type of an if condition (line 836)
        if_condition_164760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 836, 8), result_is_not_164759)
        # Assigning a type to the variable 'if_condition_164760' (line 836)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'if_condition_164760', if_condition_164760)
        # SSA begins for if statement (line 836)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 837):
        
        # Assigning a Call to a Name (line 837):
        
        # Call to fjacd(...): (line 837)
        # Getting the type of 'arglist' (line 837)
        arglist_164764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 36), 'arglist', False)
        # Processing the call keyword arguments (line 837)
        kwargs_164765 = {}
        # Getting the type of 'self' (line 837)
        self_164761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 18), 'self', False)
        # Obtaining the member 'model' of a type (line 837)
        model_164762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 18), self_164761, 'model')
        # Obtaining the member 'fjacd' of a type (line 837)
        fjacd_164763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 18), model_164762, 'fjacd')
        # Calling fjacd(args, kwargs) (line 837)
        fjacd_call_result_164766 = invoke(stypy.reporting.localization.Localization(__file__, 837, 18), fjacd_164763, *[arglist_164764], **kwargs_164765)
        
        # Assigning a type to the variable 'res' (line 837)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'res', fjacd_call_result_164766)
        
        
        # Getting the type of 'res' (line 838)
        res_164767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 15), 'res')
        # Obtaining the member 'shape' of a type (line 838)
        shape_164768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 15), res_164767, 'shape')
        # Getting the type of 'fjacd_perms' (line 838)
        fjacd_perms_164769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 32), 'fjacd_perms')
        # Applying the binary operator 'notin' (line 838)
        result_contains_164770 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 15), 'notin', shape_164768, fjacd_perms_164769)
        
        # Testing the type of an if condition (line 838)
        if_condition_164771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 838, 12), result_contains_164770)
        # Assigning a type to the variable 'if_condition_164771' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 12), 'if_condition_164771', if_condition_164771)
        # SSA begins for if statement (line 838)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to OdrError(...): (line 839)
        # Processing the call arguments (line 839)
        str_164773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 20), 'str', 'fjacd does not output %s-shaped array')
        
        # Call to repr(...): (line 840)
        # Processing the call arguments (line 840)
        
        # Obtaining an instance of the builtin type 'tuple' (line 840)
        tuple_164775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 68), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 840)
        # Adding element type (line 840)
        # Getting the type of 'q' (line 840)
        q_164776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 68), 'q', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 68), tuple_164775, q_164776)
        # Adding element type (line 840)
        # Getting the type of 'm' (line 840)
        m_164777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 71), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 68), tuple_164775, m_164777)
        # Adding element type (line 840)
        # Getting the type of 'n' (line 840)
        n_164778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 74), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 68), tuple_164775, n_164778)
        
        # Processing the call keyword arguments (line 840)
        kwargs_164779 = {}
        # Getting the type of 'repr' (line 840)
        repr_164774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 62), 'repr', False)
        # Calling repr(args, kwargs) (line 840)
        repr_call_result_164780 = invoke(stypy.reporting.localization.Localization(__file__, 840, 62), repr_164774, *[tuple_164775], **kwargs_164779)
        
        # Applying the binary operator '%' (line 840)
        result_mod_164781 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 20), '%', str_164773, repr_call_result_164780)
        
        # Processing the call keyword arguments (line 839)
        kwargs_164782 = {}
        # Getting the type of 'OdrError' (line 839)
        OdrError_164772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 22), 'OdrError', False)
        # Calling OdrError(args, kwargs) (line 839)
        OdrError_call_result_164783 = invoke(stypy.reporting.localization.Localization(__file__, 839, 22), OdrError_164772, *[result_mod_164781], **kwargs_164782)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 839, 16), OdrError_call_result_164783, 'raise parameter', BaseException)
        # SSA join for if statement (line 838)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 836)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 841)
        self_164784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 11), 'self')
        # Obtaining the member 'model' of a type (line 841)
        model_164785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 11), self_164784, 'model')
        # Obtaining the member 'fjacb' of a type (line 841)
        fjacb_164786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 11), model_164785, 'fjacb')
        # Getting the type of 'None' (line 841)
        None_164787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 35), 'None')
        # Applying the binary operator 'isnot' (line 841)
        result_is_not_164788 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 11), 'isnot', fjacb_164786, None_164787)
        
        # Testing the type of an if condition (line 841)
        if_condition_164789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 841, 8), result_is_not_164788)
        # Assigning a type to the variable 'if_condition_164789' (line 841)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 8), 'if_condition_164789', if_condition_164789)
        # SSA begins for if statement (line 841)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 842):
        
        # Assigning a Call to a Name (line 842):
        
        # Call to fjacb(...): (line 842)
        # Getting the type of 'arglist' (line 842)
        arglist_164793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 36), 'arglist', False)
        # Processing the call keyword arguments (line 842)
        kwargs_164794 = {}
        # Getting the type of 'self' (line 842)
        self_164790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 18), 'self', False)
        # Obtaining the member 'model' of a type (line 842)
        model_164791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 18), self_164790, 'model')
        # Obtaining the member 'fjacb' of a type (line 842)
        fjacb_164792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 18), model_164791, 'fjacb')
        # Calling fjacb(args, kwargs) (line 842)
        fjacb_call_result_164795 = invoke(stypy.reporting.localization.Localization(__file__, 842, 18), fjacb_164792, *[arglist_164793], **kwargs_164794)
        
        # Assigning a type to the variable 'res' (line 842)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'res', fjacb_call_result_164795)
        
        
        # Getting the type of 'res' (line 843)
        res_164796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 15), 'res')
        # Obtaining the member 'shape' of a type (line 843)
        shape_164797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 15), res_164796, 'shape')
        # Getting the type of 'fjacb_perms' (line 843)
        fjacb_perms_164798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 32), 'fjacb_perms')
        # Applying the binary operator 'notin' (line 843)
        result_contains_164799 = python_operator(stypy.reporting.localization.Localization(__file__, 843, 15), 'notin', shape_164797, fjacb_perms_164798)
        
        # Testing the type of an if condition (line 843)
        if_condition_164800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 843, 12), result_contains_164799)
        # Assigning a type to the variable 'if_condition_164800' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'if_condition_164800', if_condition_164800)
        # SSA begins for if statement (line 843)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to OdrError(...): (line 844)
        # Processing the call arguments (line 844)
        str_164802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 20), 'str', 'fjacb does not output %s-shaped array')
        
        # Call to repr(...): (line 845)
        # Processing the call arguments (line 845)
        
        # Obtaining an instance of the builtin type 'tuple' (line 845)
        tuple_164804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 68), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 845)
        # Adding element type (line 845)
        # Getting the type of 'q' (line 845)
        q_164805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 68), 'q', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 68), tuple_164804, q_164805)
        # Adding element type (line 845)
        # Getting the type of 'p' (line 845)
        p_164806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 71), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 68), tuple_164804, p_164806)
        # Adding element type (line 845)
        # Getting the type of 'n' (line 845)
        n_164807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 74), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 68), tuple_164804, n_164807)
        
        # Processing the call keyword arguments (line 845)
        kwargs_164808 = {}
        # Getting the type of 'repr' (line 845)
        repr_164803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 62), 'repr', False)
        # Calling repr(args, kwargs) (line 845)
        repr_call_result_164809 = invoke(stypy.reporting.localization.Localization(__file__, 845, 62), repr_164803, *[tuple_164804], **kwargs_164808)
        
        # Applying the binary operator '%' (line 845)
        result_mod_164810 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 20), '%', str_164802, repr_call_result_164809)
        
        # Processing the call keyword arguments (line 844)
        kwargs_164811 = {}
        # Getting the type of 'OdrError' (line 844)
        OdrError_164801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 22), 'OdrError', False)
        # Calling OdrError(args, kwargs) (line 844)
        OdrError_call_result_164812 = invoke(stypy.reporting.localization.Localization(__file__, 844, 22), OdrError_164801, *[result_mod_164810], **kwargs_164811)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 844, 16), OdrError_call_result_164812, 'raise parameter', BaseException)
        # SSA join for if statement (line 843)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 841)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 849)
        self_164813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 11), 'self')
        # Obtaining the member 'delta0' of a type (line 849)
        delta0_164814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 11), self_164813, 'delta0')
        # Getting the type of 'None' (line 849)
        None_164815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 30), 'None')
        # Applying the binary operator 'isnot' (line 849)
        result_is_not_164816 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 11), 'isnot', delta0_164814, None_164815)
        
        
        # Getting the type of 'self' (line 849)
        self_164817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 39), 'self')
        # Obtaining the member 'delta0' of a type (line 849)
        delta0_164818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 39), self_164817, 'delta0')
        # Obtaining the member 'shape' of a type (line 849)
        shape_164819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 39), delta0_164818, 'shape')
        # Getting the type of 'self' (line 849)
        self_164820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 60), 'self')
        # Obtaining the member 'data' of a type (line 849)
        data_164821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 60), self_164820, 'data')
        # Obtaining the member 'x' of a type (line 849)
        x_164822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 60), data_164821, 'x')
        # Obtaining the member 'shape' of a type (line 849)
        shape_164823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 60), x_164822, 'shape')
        # Applying the binary operator '!=' (line 849)
        result_ne_164824 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 39), '!=', shape_164819, shape_164823)
        
        # Applying the binary operator 'and' (line 849)
        result_and_keyword_164825 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 11), 'and', result_is_not_164816, result_ne_164824)
        
        # Testing the type of an if condition (line 849)
        if_condition_164826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 849, 8), result_and_keyword_164825)
        # Assigning a type to the variable 'if_condition_164826' (line 849)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'if_condition_164826', if_condition_164826)
        # SSA begins for if statement (line 849)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to OdrError(...): (line 850)
        # Processing the call arguments (line 850)
        str_164828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 16), 'str', 'delta0 is not a %s-shaped array')
        
        # Call to repr(...): (line 851)
        # Processing the call arguments (line 851)
        # Getting the type of 'self' (line 851)
        self_164830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 57), 'self', False)
        # Obtaining the member 'data' of a type (line 851)
        data_164831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 57), self_164830, 'data')
        # Obtaining the member 'x' of a type (line 851)
        x_164832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 57), data_164831, 'x')
        # Obtaining the member 'shape' of a type (line 851)
        shape_164833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 57), x_164832, 'shape')
        # Processing the call keyword arguments (line 851)
        kwargs_164834 = {}
        # Getting the type of 'repr' (line 851)
        repr_164829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 52), 'repr', False)
        # Calling repr(args, kwargs) (line 851)
        repr_call_result_164835 = invoke(stypy.reporting.localization.Localization(__file__, 851, 52), repr_164829, *[shape_164833], **kwargs_164834)
        
        # Applying the binary operator '%' (line 851)
        result_mod_164836 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 16), '%', str_164828, repr_call_result_164835)
        
        # Processing the call keyword arguments (line 850)
        kwargs_164837 = {}
        # Getting the type of 'OdrError' (line 850)
        OdrError_164827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 18), 'OdrError', False)
        # Calling OdrError(args, kwargs) (line 850)
        OdrError_call_result_164838 = invoke(stypy.reporting.localization.Localization(__file__, 850, 18), OdrError_164827, *[result_mod_164836], **kwargs_164837)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 850, 12), OdrError_call_result_164838, 'raise parameter', BaseException)
        # SSA join for if statement (line 849)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 853)
        self_164839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 11), 'self')
        # Obtaining the member 'data' of a type (line 853)
        data_164840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 11), self_164839, 'data')
        # Obtaining the member 'x' of a type (line 853)
        x_164841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 11), data_164840, 'x')
        # Obtaining the member 'size' of a type (line 853)
        size_164842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 11), x_164841, 'size')
        int_164843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 31), 'int')
        # Applying the binary operator '==' (line 853)
        result_eq_164844 = python_operator(stypy.reporting.localization.Localization(__file__, 853, 11), '==', size_164842, int_164843)
        
        # Testing the type of an if condition (line 853)
        if_condition_164845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 853, 8), result_eq_164844)
        # Assigning a type to the variable 'if_condition_164845' (line 853)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'if_condition_164845', if_condition_164845)
        # SSA begins for if statement (line 853)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 854)
        # Processing the call arguments (line 854)
        str_164847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 18), 'str', 'Empty data detected for ODR instance. Do not expect any fitting to occur')
        # Getting the type of 'OdrWarning' (line 856)
        OdrWarning_164848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 17), 'OdrWarning', False)
        # Processing the call keyword arguments (line 854)
        kwargs_164849 = {}
        # Getting the type of 'warn' (line 854)
        warn_164846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 12), 'warn', False)
        # Calling warn(args, kwargs) (line 854)
        warn_call_result_164850 = invoke(stypy.reporting.localization.Localization(__file__, 854, 12), warn_164846, *[str_164847, OdrWarning_164848], **kwargs_164849)
        
        # SSA join for if statement (line 853)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check' in the type store
        # Getting the type of 'stypy_return_type' (line 770)
        stypy_return_type_164851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_164851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check'
        return stypy_return_type_164851


    @norecursion
    def _gen_work(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_gen_work'
        module_type_store = module_type_store.open_function_context('_gen_work', 858, 4, False)
        # Assigning a type to the variable 'self' (line 859)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODR._gen_work.__dict__.__setitem__('stypy_localization', localization)
        ODR._gen_work.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODR._gen_work.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODR._gen_work.__dict__.__setitem__('stypy_function_name', 'ODR._gen_work')
        ODR._gen_work.__dict__.__setitem__('stypy_param_names_list', [])
        ODR._gen_work.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODR._gen_work.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODR._gen_work.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODR._gen_work.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODR._gen_work.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODR._gen_work.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODR._gen_work', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_gen_work', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_gen_work(...)' code ##################

        str_164852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, (-1)), 'str', ' Generate a suitable work array if one does not already exist.\n        ')
        
        # Assigning a Subscript to a Name (line 862):
        
        # Assigning a Subscript to a Name (line 862):
        
        # Obtaining the type of the subscript
        int_164853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 30), 'int')
        # Getting the type of 'self' (line 862)
        self_164854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 12), 'self')
        # Obtaining the member 'data' of a type (line 862)
        data_164855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 12), self_164854, 'data')
        # Obtaining the member 'x' of a type (line 862)
        x_164856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 12), data_164855, 'x')
        # Obtaining the member 'shape' of a type (line 862)
        shape_164857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 12), x_164856, 'shape')
        # Obtaining the member '__getitem__' of a type (line 862)
        getitem___164858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 12), shape_164857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 862)
        subscript_call_result_164859 = invoke(stypy.reporting.localization.Localization(__file__, 862, 12), getitem___164858, int_164853)
        
        # Assigning a type to the variable 'n' (line 862)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'n', subscript_call_result_164859)
        
        # Assigning a Subscript to a Name (line 863):
        
        # Assigning a Subscript to a Name (line 863):
        
        # Obtaining the type of the subscript
        int_164860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 29), 'int')
        # Getting the type of 'self' (line 863)
        self_164861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 12), 'self')
        # Obtaining the member 'beta0' of a type (line 863)
        beta0_164862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 12), self_164861, 'beta0')
        # Obtaining the member 'shape' of a type (line 863)
        shape_164863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 12), beta0_164862, 'shape')
        # Obtaining the member '__getitem__' of a type (line 863)
        getitem___164864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 12), shape_164863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 863)
        subscript_call_result_164865 = invoke(stypy.reporting.localization.Localization(__file__, 863, 12), getitem___164864, int_164860)
        
        # Assigning a type to the variable 'p' (line 863)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 8), 'p', subscript_call_result_164865)
        
        
        
        # Call to len(...): (line 865)
        # Processing the call arguments (line 865)
        # Getting the type of 'self' (line 865)
        self_164867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 15), 'self', False)
        # Obtaining the member 'data' of a type (line 865)
        data_164868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 15), self_164867, 'data')
        # Obtaining the member 'x' of a type (line 865)
        x_164869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 15), data_164868, 'x')
        # Obtaining the member 'shape' of a type (line 865)
        shape_164870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 15), x_164869, 'shape')
        # Processing the call keyword arguments (line 865)
        kwargs_164871 = {}
        # Getting the type of 'len' (line 865)
        len_164866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 11), 'len', False)
        # Calling len(args, kwargs) (line 865)
        len_call_result_164872 = invoke(stypy.reporting.localization.Localization(__file__, 865, 11), len_164866, *[shape_164870], **kwargs_164871)
        
        int_164873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 37), 'int')
        # Applying the binary operator '==' (line 865)
        result_eq_164874 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 11), '==', len_call_result_164872, int_164873)
        
        # Testing the type of an if condition (line 865)
        if_condition_164875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 865, 8), result_eq_164874)
        # Assigning a type to the variable 'if_condition_164875' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 8), 'if_condition_164875', if_condition_164875)
        # SSA begins for if statement (line 865)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 866):
        
        # Assigning a Subscript to a Name (line 866):
        
        # Obtaining the type of the subscript
        int_164876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 34), 'int')
        # Getting the type of 'self' (line 866)
        self_164877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 16), 'self')
        # Obtaining the member 'data' of a type (line 866)
        data_164878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 16), self_164877, 'data')
        # Obtaining the member 'x' of a type (line 866)
        x_164879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 16), data_164878, 'x')
        # Obtaining the member 'shape' of a type (line 866)
        shape_164880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 16), x_164879, 'shape')
        # Obtaining the member '__getitem__' of a type (line 866)
        getitem___164881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 16), shape_164880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 866)
        subscript_call_result_164882 = invoke(stypy.reporting.localization.Localization(__file__, 866, 16), getitem___164881, int_164876)
        
        # Assigning a type to the variable 'm' (line 866)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 12), 'm', subscript_call_result_164882)
        # SSA branch for the else part of an if statement (line 865)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 868):
        
        # Assigning a Num to a Name (line 868):
        int_164883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 16), 'int')
        # Assigning a type to the variable 'm' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 12), 'm', int_164883)
        # SSA join for if statement (line 865)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 870)
        self_164884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 11), 'self')
        # Obtaining the member 'model' of a type (line 870)
        model_164885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 11), self_164884, 'model')
        # Obtaining the member 'implicit' of a type (line 870)
        implicit_164886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 11), model_164885, 'implicit')
        # Testing the type of an if condition (line 870)
        if_condition_164887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 870, 8), implicit_164886)
        # Assigning a type to the variable 'if_condition_164887' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'if_condition_164887', if_condition_164887)
        # SSA begins for if statement (line 870)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 871):
        
        # Assigning a Attribute to a Name (line 871):
        # Getting the type of 'self' (line 871)
        self_164888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 16), 'self')
        # Obtaining the member 'data' of a type (line 871)
        data_164889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 16), self_164888, 'data')
        # Obtaining the member 'y' of a type (line 871)
        y_164890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 16), data_164889, 'y')
        # Assigning a type to the variable 'q' (line 871)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 12), 'q', y_164890)
        # SSA branch for the else part of an if statement (line 870)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 872)
        # Processing the call arguments (line 872)
        # Getting the type of 'self' (line 872)
        self_164892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 17), 'self', False)
        # Obtaining the member 'data' of a type (line 872)
        data_164893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 17), self_164892, 'data')
        # Obtaining the member 'y' of a type (line 872)
        y_164894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 17), data_164893, 'y')
        # Obtaining the member 'shape' of a type (line 872)
        shape_164895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 17), y_164894, 'shape')
        # Processing the call keyword arguments (line 872)
        kwargs_164896 = {}
        # Getting the type of 'len' (line 872)
        len_164891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 13), 'len', False)
        # Calling len(args, kwargs) (line 872)
        len_call_result_164897 = invoke(stypy.reporting.localization.Localization(__file__, 872, 13), len_164891, *[shape_164895], **kwargs_164896)
        
        int_164898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 39), 'int')
        # Applying the binary operator '==' (line 872)
        result_eq_164899 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 13), '==', len_call_result_164897, int_164898)
        
        # Testing the type of an if condition (line 872)
        if_condition_164900 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 872, 13), result_eq_164899)
        # Assigning a type to the variable 'if_condition_164900' (line 872)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 13), 'if_condition_164900', if_condition_164900)
        # SSA begins for if statement (line 872)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 873):
        
        # Assigning a Subscript to a Name (line 873):
        
        # Obtaining the type of the subscript
        int_164901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 34), 'int')
        # Getting the type of 'self' (line 873)
        self_164902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 16), 'self')
        # Obtaining the member 'data' of a type (line 873)
        data_164903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 16), self_164902, 'data')
        # Obtaining the member 'y' of a type (line 873)
        y_164904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 16), data_164903, 'y')
        # Obtaining the member 'shape' of a type (line 873)
        shape_164905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 16), y_164904, 'shape')
        # Obtaining the member '__getitem__' of a type (line 873)
        getitem___164906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 16), shape_164905, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 873)
        subscript_call_result_164907 = invoke(stypy.reporting.localization.Localization(__file__, 873, 16), getitem___164906, int_164901)
        
        # Assigning a type to the variable 'q' (line 873)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 12), 'q', subscript_call_result_164907)
        # SSA branch for the else part of an if statement (line 872)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 875):
        
        # Assigning a Num to a Name (line 875):
        int_164908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 16), 'int')
        # Assigning a type to the variable 'q' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'q', int_164908)
        # SSA join for if statement (line 872)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 870)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 877)
        # Getting the type of 'self' (line 877)
        self_164909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 11), 'self')
        # Obtaining the member 'data' of a type (line 877)
        data_164910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 11), self_164909, 'data')
        # Obtaining the member 'we' of a type (line 877)
        we_164911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 11), data_164910, 'we')
        # Getting the type of 'None' (line 877)
        None_164912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 27), 'None')
        
        (may_be_164913, more_types_in_union_164914) = may_be_none(we_164911, None_164912)

        if may_be_164913:

            if more_types_in_union_164914:
                # Runtime conditional SSA (line 877)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Multiple assignment of 2 elements.
            
            # Assigning a Num to a Name (line 878):
            int_164915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 27), 'int')
            # Assigning a type to the variable 'ld2we' (line 878)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 19), 'ld2we', int_164915)
            
            # Assigning a Name to a Name (line 878):
            # Getting the type of 'ld2we' (line 878)
            ld2we_164916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 19), 'ld2we')
            # Assigning a type to the variable 'ldwe' (line 878)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 12), 'ldwe', ld2we_164916)

            if more_types_in_union_164914:
                # Runtime conditional SSA for else branch (line 877)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_164913) or more_types_in_union_164914):
            
            
            
            # Call to len(...): (line 879)
            # Processing the call arguments (line 879)
            # Getting the type of 'self' (line 879)
            self_164918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 17), 'self', False)
            # Obtaining the member 'data' of a type (line 879)
            data_164919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 17), self_164918, 'data')
            # Obtaining the member 'we' of a type (line 879)
            we_164920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 17), data_164919, 'we')
            # Obtaining the member 'shape' of a type (line 879)
            shape_164921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 17), we_164920, 'shape')
            # Processing the call keyword arguments (line 879)
            kwargs_164922 = {}
            # Getting the type of 'len' (line 879)
            len_164917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 13), 'len', False)
            # Calling len(args, kwargs) (line 879)
            len_call_result_164923 = invoke(stypy.reporting.localization.Localization(__file__, 879, 13), len_164917, *[shape_164921], **kwargs_164922)
            
            int_164924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 40), 'int')
            # Applying the binary operator '==' (line 879)
            result_eq_164925 = python_operator(stypy.reporting.localization.Localization(__file__, 879, 13), '==', len_call_result_164923, int_164924)
            
            # Testing the type of an if condition (line 879)
            if_condition_164926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 879, 13), result_eq_164925)
            # Assigning a type to the variable 'if_condition_164926' (line 879)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 13), 'if_condition_164926', if_condition_164926)
            # SSA begins for if statement (line 879)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Tuple (line 880):
            
            # Assigning a Subscript to a Name (line 880):
            
            # Obtaining the type of the subscript
            int_164927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 12), 'int')
            
            # Obtaining the type of the subscript
            int_164928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 45), 'int')
            slice_164929 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 880, 26), int_164928, None, None)
            # Getting the type of 'self' (line 880)
            self_164930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 26), 'self')
            # Obtaining the member 'data' of a type (line 880)
            data_164931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 26), self_164930, 'data')
            # Obtaining the member 'we' of a type (line 880)
            we_164932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 26), data_164931, 'we')
            # Obtaining the member 'shape' of a type (line 880)
            shape_164933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 26), we_164932, 'shape')
            # Obtaining the member '__getitem__' of a type (line 880)
            getitem___164934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 26), shape_164933, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 880)
            subscript_call_result_164935 = invoke(stypy.reporting.localization.Localization(__file__, 880, 26), getitem___164934, slice_164929)
            
            # Obtaining the member '__getitem__' of a type (line 880)
            getitem___164936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 12), subscript_call_result_164935, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 880)
            subscript_call_result_164937 = invoke(stypy.reporting.localization.Localization(__file__, 880, 12), getitem___164936, int_164927)
            
            # Assigning a type to the variable 'tuple_var_assignment_163538' (line 880)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 12), 'tuple_var_assignment_163538', subscript_call_result_164937)
            
            # Assigning a Subscript to a Name (line 880):
            
            # Obtaining the type of the subscript
            int_164938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 12), 'int')
            
            # Obtaining the type of the subscript
            int_164939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 45), 'int')
            slice_164940 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 880, 26), int_164939, None, None)
            # Getting the type of 'self' (line 880)
            self_164941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 26), 'self')
            # Obtaining the member 'data' of a type (line 880)
            data_164942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 26), self_164941, 'data')
            # Obtaining the member 'we' of a type (line 880)
            we_164943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 26), data_164942, 'we')
            # Obtaining the member 'shape' of a type (line 880)
            shape_164944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 26), we_164943, 'shape')
            # Obtaining the member '__getitem__' of a type (line 880)
            getitem___164945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 26), shape_164944, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 880)
            subscript_call_result_164946 = invoke(stypy.reporting.localization.Localization(__file__, 880, 26), getitem___164945, slice_164940)
            
            # Obtaining the member '__getitem__' of a type (line 880)
            getitem___164947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 12), subscript_call_result_164946, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 880)
            subscript_call_result_164948 = invoke(stypy.reporting.localization.Localization(__file__, 880, 12), getitem___164947, int_164938)
            
            # Assigning a type to the variable 'tuple_var_assignment_163539' (line 880)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 12), 'tuple_var_assignment_163539', subscript_call_result_164948)
            
            # Assigning a Name to a Name (line 880):
            # Getting the type of 'tuple_var_assignment_163538' (line 880)
            tuple_var_assignment_163538_164949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 12), 'tuple_var_assignment_163538')
            # Assigning a type to the variable 'ld2we' (line 880)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 12), 'ld2we', tuple_var_assignment_163538_164949)
            
            # Assigning a Name to a Name (line 880):
            # Getting the type of 'tuple_var_assignment_163539' (line 880)
            tuple_var_assignment_163539_164950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 12), 'tuple_var_assignment_163539')
            # Assigning a type to the variable 'ldwe' (line 880)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 19), 'ldwe', tuple_var_assignment_163539_164950)
            # SSA branch for the else part of an if statement (line 879)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 884):
            
            # Assigning a Num to a Name (line 884):
            int_164951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 19), 'int')
            # Assigning a type to the variable 'ldwe' (line 884)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 12), 'ldwe', int_164951)
            
            # Assigning a Subscript to a Name (line 885):
            
            # Assigning a Subscript to a Name (line 885):
            
            # Obtaining the type of the subscript
            int_164952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 39), 'int')
            # Getting the type of 'self' (line 885)
            self_164953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 20), 'self')
            # Obtaining the member 'data' of a type (line 885)
            data_164954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 20), self_164953, 'data')
            # Obtaining the member 'we' of a type (line 885)
            we_164955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 20), data_164954, 'we')
            # Obtaining the member 'shape' of a type (line 885)
            shape_164956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 20), we_164955, 'shape')
            # Obtaining the member '__getitem__' of a type (line 885)
            getitem___164957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 20), shape_164956, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 885)
            subscript_call_result_164958 = invoke(stypy.reporting.localization.Localization(__file__, 885, 20), getitem___164957, int_164952)
            
            # Assigning a type to the variable 'ld2we' (line 885)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 12), 'ld2we', subscript_call_result_164958)
            # SSA join for if statement (line 879)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_164913 and more_types_in_union_164914):
                # SSA join for if statement (line 877)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 887)
        self_164959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 11), 'self')
        # Obtaining the member 'job' of a type (line 887)
        job_164960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 11), self_164959, 'job')
        int_164961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 22), 'int')
        # Applying the binary operator '%' (line 887)
        result_mod_164962 = python_operator(stypy.reporting.localization.Localization(__file__, 887, 11), '%', job_164960, int_164961)
        
        int_164963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 27), 'int')
        # Applying the binary operator '<' (line 887)
        result_lt_164964 = python_operator(stypy.reporting.localization.Localization(__file__, 887, 11), '<', result_mod_164962, int_164963)
        
        # Testing the type of an if condition (line 887)
        if_condition_164965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 887, 8), result_lt_164964)
        # Assigning a type to the variable 'if_condition_164965' (line 887)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'if_condition_164965', if_condition_164965)
        # SSA begins for if statement (line 887)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 889):
        
        # Assigning a BinOp to a Name (line 889):
        int_164966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 21), 'int')
        int_164967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 26), 'int')
        # Getting the type of 'p' (line 889)
        p_164968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 29), 'p')
        # Applying the binary operator '*' (line 889)
        result_mul_164969 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 26), '*', int_164967, p_164968)
        
        # Applying the binary operator '+' (line 889)
        result_add_164970 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 21), '+', int_164966, result_mul_164969)
        
        # Getting the type of 'p' (line 889)
        p_164971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 33), 'p')
        # Getting the type of 'p' (line 889)
        p_164972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 35), 'p')
        # Applying the binary operator '*' (line 889)
        result_mul_164973 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 33), '*', p_164971, p_164972)
        
        # Applying the binary operator '+' (line 889)
        result_add_164974 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 31), '+', result_add_164970, result_mul_164973)
        
        # Getting the type of 'm' (line 889)
        m_164975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 39), 'm')
        # Applying the binary operator '+' (line 889)
        result_add_164976 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 37), '+', result_add_164974, m_164975)
        
        # Getting the type of 'm' (line 889)
        m_164977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 43), 'm')
        # Getting the type of 'm' (line 889)
        m_164978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 45), 'm')
        # Applying the binary operator '*' (line 889)
        result_mul_164979 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 43), '*', m_164977, m_164978)
        
        # Applying the binary operator '+' (line 889)
        result_add_164980 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 41), '+', result_add_164976, result_mul_164979)
        
        int_164981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 49), 'int')
        # Getting the type of 'n' (line 889)
        n_164982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 51), 'n')
        # Applying the binary operator '*' (line 889)
        result_mul_164983 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 49), '*', int_164981, n_164982)
        
        # Getting the type of 'q' (line 889)
        q_164984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 53), 'q')
        # Applying the binary operator '*' (line 889)
        result_mul_164985 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 52), '*', result_mul_164983, q_164984)
        
        # Applying the binary operator '+' (line 889)
        result_add_164986 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 47), '+', result_add_164980, result_mul_164985)
        
        int_164987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 57), 'int')
        # Getting the type of 'n' (line 889)
        n_164988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 59), 'n')
        # Applying the binary operator '*' (line 889)
        result_mul_164989 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 57), '*', int_164987, n_164988)
        
        # Getting the type of 'm' (line 889)
        m_164990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 61), 'm')
        # Applying the binary operator '*' (line 889)
        result_mul_164991 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 60), '*', result_mul_164989, m_164990)
        
        # Applying the binary operator '+' (line 889)
        result_add_164992 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 55), '+', result_add_164986, result_mul_164991)
        
        int_164993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 65), 'int')
        # Getting the type of 'n' (line 889)
        n_164994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 67), 'n')
        # Applying the binary operator '*' (line 889)
        result_mul_164995 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 65), '*', int_164993, n_164994)
        
        # Getting the type of 'q' (line 889)
        q_164996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 69), 'q')
        # Applying the binary operator '*' (line 889)
        result_mul_164997 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 68), '*', result_mul_164995, q_164996)
        
        # Getting the type of 'p' (line 889)
        p_164998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 71), 'p')
        # Applying the binary operator '*' (line 889)
        result_mul_164999 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 70), '*', result_mul_164997, p_164998)
        
        # Applying the binary operator '+' (line 889)
        result_add_165000 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 63), '+', result_add_164992, result_mul_164999)
        
        int_165001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 21), 'int')
        # Getting the type of 'n' (line 890)
        n_165002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 23), 'n')
        # Applying the binary operator '*' (line 890)
        result_mul_165003 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 21), '*', int_165001, n_165002)
        
        # Getting the type of 'q' (line 890)
        q_165004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 25), 'q')
        # Applying the binary operator '*' (line 890)
        result_mul_165005 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 24), '*', result_mul_165003, q_165004)
        
        # Getting the type of 'm' (line 890)
        m_165006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 27), 'm')
        # Applying the binary operator '*' (line 890)
        result_mul_165007 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 26), '*', result_mul_165005, m_165006)
        
        # Applying the binary operator '+' (line 889)
        result_add_165008 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 73), '+', result_add_165000, result_mul_165007)
        
        # Getting the type of 'q' (line 890)
        q_165009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 31), 'q')
        # Getting the type of 'q' (line 890)
        q_165010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 33), 'q')
        # Applying the binary operator '*' (line 890)
        result_mul_165011 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 31), '*', q_165009, q_165010)
        
        # Applying the binary operator '+' (line 890)
        result_add_165012 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 29), '+', result_add_165008, result_mul_165011)
        
        int_165013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 37), 'int')
        # Getting the type of 'q' (line 890)
        q_165014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 39), 'q')
        # Applying the binary operator '*' (line 890)
        result_mul_165015 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 37), '*', int_165013, q_165014)
        
        # Applying the binary operator '+' (line 890)
        result_add_165016 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 35), '+', result_add_165012, result_mul_165015)
        
        # Getting the type of 'q' (line 890)
        q_165017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 43), 'q')
        # Getting the type of 'p' (line 890)
        p_165018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 46), 'p')
        # Getting the type of 'm' (line 890)
        m_165019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 48), 'm')
        # Applying the binary operator '+' (line 890)
        result_add_165020 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 46), '+', p_165018, m_165019)
        
        # Applying the binary operator '*' (line 890)
        result_mul_165021 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 43), '*', q_165017, result_add_165020)
        
        # Applying the binary operator '+' (line 890)
        result_add_165022 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 41), '+', result_add_165016, result_mul_165021)
        
        # Getting the type of 'ldwe' (line 890)
        ldwe_165023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 53), 'ldwe')
        # Getting the type of 'ld2we' (line 890)
        ld2we_165024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 58), 'ld2we')
        # Applying the binary operator '*' (line 890)
        result_mul_165025 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 53), '*', ldwe_165023, ld2we_165024)
        
        # Getting the type of 'q' (line 890)
        q_165026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 64), 'q')
        # Applying the binary operator '*' (line 890)
        result_mul_165027 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 63), '*', result_mul_165025, q_165026)
        
        # Applying the binary operator '+' (line 890)
        result_add_165028 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 51), '+', result_add_165022, result_mul_165027)
        
        # Assigning a type to the variable 'lwork' (line 889)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 12), 'lwork', result_add_165028)
        # SSA branch for the else part of an if statement (line 887)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 893):
        
        # Assigning a BinOp to a Name (line 893):
        int_165029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 21), 'int')
        int_165030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 26), 'int')
        # Getting the type of 'p' (line 893)
        p_165031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 29), 'p')
        # Applying the binary operator '*' (line 893)
        result_mul_165032 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 26), '*', int_165030, p_165031)
        
        # Applying the binary operator '+' (line 893)
        result_add_165033 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 21), '+', int_165029, result_mul_165032)
        
        # Getting the type of 'p' (line 893)
        p_165034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 33), 'p')
        # Getting the type of 'p' (line 893)
        p_165035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 35), 'p')
        # Applying the binary operator '*' (line 893)
        result_mul_165036 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 33), '*', p_165034, p_165035)
        
        # Applying the binary operator '+' (line 893)
        result_add_165037 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 31), '+', result_add_165033, result_mul_165036)
        
        # Getting the type of 'm' (line 893)
        m_165038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 39), 'm')
        # Applying the binary operator '+' (line 893)
        result_add_165039 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 37), '+', result_add_165037, m_165038)
        
        # Getting the type of 'm' (line 893)
        m_165040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 43), 'm')
        # Getting the type of 'm' (line 893)
        m_165041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 45), 'm')
        # Applying the binary operator '*' (line 893)
        result_mul_165042 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 43), '*', m_165040, m_165041)
        
        # Applying the binary operator '+' (line 893)
        result_add_165043 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 41), '+', result_add_165039, result_mul_165042)
        
        int_165044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 49), 'int')
        # Getting the type of 'n' (line 893)
        n_165045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 51), 'n')
        # Applying the binary operator '*' (line 893)
        result_mul_165046 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 49), '*', int_165044, n_165045)
        
        # Getting the type of 'q' (line 893)
        q_165047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 53), 'q')
        # Applying the binary operator '*' (line 893)
        result_mul_165048 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 52), '*', result_mul_165046, q_165047)
        
        # Applying the binary operator '+' (line 893)
        result_add_165049 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 47), '+', result_add_165043, result_mul_165048)
        
        int_165050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 57), 'int')
        # Getting the type of 'n' (line 893)
        n_165051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 59), 'n')
        # Applying the binary operator '*' (line 893)
        result_mul_165052 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 57), '*', int_165050, n_165051)
        
        # Getting the type of 'm' (line 893)
        m_165053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 61), 'm')
        # Applying the binary operator '*' (line 893)
        result_mul_165054 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 60), '*', result_mul_165052, m_165053)
        
        # Applying the binary operator '+' (line 893)
        result_add_165055 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 55), '+', result_add_165049, result_mul_165054)
        
        int_165056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 65), 'int')
        # Getting the type of 'n' (line 893)
        n_165057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 67), 'n')
        # Applying the binary operator '*' (line 893)
        result_mul_165058 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 65), '*', int_165056, n_165057)
        
        # Getting the type of 'q' (line 893)
        q_165059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 69), 'q')
        # Applying the binary operator '*' (line 893)
        result_mul_165060 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 68), '*', result_mul_165058, q_165059)
        
        # Getting the type of 'p' (line 893)
        p_165061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 71), 'p')
        # Applying the binary operator '*' (line 893)
        result_mul_165062 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 70), '*', result_mul_165060, p_165061)
        
        # Applying the binary operator '+' (line 893)
        result_add_165063 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 63), '+', result_add_165055, result_mul_165062)
        
        int_165064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 21), 'int')
        # Getting the type of 'q' (line 894)
        q_165065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 23), 'q')
        # Applying the binary operator '*' (line 894)
        result_mul_165066 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 21), '*', int_165064, q_165065)
        
        # Applying the binary operator '+' (line 893)
        result_add_165067 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 73), '+', result_add_165063, result_mul_165066)
        
        # Getting the type of 'q' (line 894)
        q_165068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 27), 'q')
        # Getting the type of 'p' (line 894)
        p_165069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 30), 'p')
        # Getting the type of 'm' (line 894)
        m_165070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 32), 'm')
        # Applying the binary operator '+' (line 894)
        result_add_165071 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 30), '+', p_165069, m_165070)
        
        # Applying the binary operator '*' (line 894)
        result_mul_165072 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 27), '*', q_165068, result_add_165071)
        
        # Applying the binary operator '+' (line 894)
        result_add_165073 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 25), '+', result_add_165067, result_mul_165072)
        
        # Getting the type of 'ldwe' (line 894)
        ldwe_165074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 37), 'ldwe')
        # Getting the type of 'ld2we' (line 894)
        ld2we_165075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 42), 'ld2we')
        # Applying the binary operator '*' (line 894)
        result_mul_165076 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 37), '*', ldwe_165074, ld2we_165075)
        
        # Getting the type of 'q' (line 894)
        q_165077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 48), 'q')
        # Applying the binary operator '*' (line 894)
        result_mul_165078 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 47), '*', result_mul_165076, q_165077)
        
        # Applying the binary operator '+' (line 894)
        result_add_165079 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 35), '+', result_add_165073, result_mul_165078)
        
        # Assigning a type to the variable 'lwork' (line 893)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 12), 'lwork', result_add_165079)
        # SSA join for if statement (line 887)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 896)
        # Processing the call arguments (line 896)
        # Getting the type of 'self' (line 896)
        self_165081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 22), 'self', False)
        # Obtaining the member 'work' of a type (line 896)
        work_165082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 22), self_165081, 'work')
        # Getting the type of 'numpy' (line 896)
        numpy_165083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 33), 'numpy', False)
        # Obtaining the member 'ndarray' of a type (line 896)
        ndarray_165084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 33), numpy_165083, 'ndarray')
        # Processing the call keyword arguments (line 896)
        kwargs_165085 = {}
        # Getting the type of 'isinstance' (line 896)
        isinstance_165080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 896)
        isinstance_call_result_165086 = invoke(stypy.reporting.localization.Localization(__file__, 896, 11), isinstance_165080, *[work_165082, ndarray_165084], **kwargs_165085)
        
        
        # Getting the type of 'self' (line 896)
        self_165087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 52), 'self')
        # Obtaining the member 'work' of a type (line 896)
        work_165088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 52), self_165087, 'work')
        # Obtaining the member 'shape' of a type (line 896)
        shape_165089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 52), work_165088, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 896)
        tuple_165090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 72), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 896)
        # Adding element type (line 896)
        # Getting the type of 'lwork' (line 896)
        lwork_165091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 72), 'lwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 896, 72), tuple_165090, lwork_165091)
        
        # Applying the binary operator '==' (line 896)
        result_eq_165092 = python_operator(stypy.reporting.localization.Localization(__file__, 896, 52), '==', shape_165089, tuple_165090)
        
        # Applying the binary operator 'and' (line 896)
        result_and_keyword_165093 = python_operator(stypy.reporting.localization.Localization(__file__, 896, 11), 'and', isinstance_call_result_165086, result_eq_165092)
        
        # Call to endswith(...): (line 897)
        # Processing the call arguments (line 897)
        str_165099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 49), 'str', 'f8')
        # Processing the call keyword arguments (line 897)
        kwargs_165100 = {}
        # Getting the type of 'self' (line 897)
        self_165094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 20), 'self', False)
        # Obtaining the member 'work' of a type (line 897)
        work_165095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 20), self_165094, 'work')
        # Obtaining the member 'dtype' of a type (line 897)
        dtype_165096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 20), work_165095, 'dtype')
        # Obtaining the member 'str' of a type (line 897)
        str_165097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 20), dtype_165096, 'str')
        # Obtaining the member 'endswith' of a type (line 897)
        endswith_165098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 20), str_165097, 'endswith')
        # Calling endswith(args, kwargs) (line 897)
        endswith_call_result_165101 = invoke(stypy.reporting.localization.Localization(__file__, 897, 20), endswith_165098, *[str_165099], **kwargs_165100)
        
        # Applying the binary operator 'and' (line 896)
        result_and_keyword_165102 = python_operator(stypy.reporting.localization.Localization(__file__, 896, 11), 'and', result_and_keyword_165093, endswith_call_result_165101)
        
        # Testing the type of an if condition (line 896)
        if_condition_165103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 896, 8), result_and_keyword_165102)
        # Assigning a type to the variable 'if_condition_165103' (line 896)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 8), 'if_condition_165103', if_condition_165103)
        # SSA begins for if statement (line 896)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'stypy_return_type', types.NoneType)
        # SSA branch for the else part of an if statement (line 896)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 901):
        
        # Assigning a Call to a Attribute (line 901):
        
        # Call to zeros(...): (line 901)
        # Processing the call arguments (line 901)
        
        # Obtaining an instance of the builtin type 'tuple' (line 901)
        tuple_165106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 901)
        # Adding element type (line 901)
        # Getting the type of 'lwork' (line 901)
        lwork_165107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 37), 'lwork', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 37), tuple_165106, lwork_165107)
        
        # Getting the type of 'float' (line 901)
        float_165108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 46), 'float', False)
        # Processing the call keyword arguments (line 901)
        kwargs_165109 = {}
        # Getting the type of 'numpy' (line 901)
        numpy_165104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 24), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 901)
        zeros_165105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 24), numpy_165104, 'zeros')
        # Calling zeros(args, kwargs) (line 901)
        zeros_call_result_165110 = invoke(stypy.reporting.localization.Localization(__file__, 901, 24), zeros_165105, *[tuple_165106, float_165108], **kwargs_165109)
        
        # Getting the type of 'self' (line 901)
        self_165111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 12), 'self')
        # Setting the type of the member 'work' of a type (line 901)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 12), self_165111, 'work', zeros_call_result_165110)
        # SSA join for if statement (line 896)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_gen_work(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_gen_work' in the type store
        # Getting the type of 'stypy_return_type' (line 858)
        stypy_return_type_165112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_gen_work'
        return stypy_return_type_165112


    @norecursion
    def set_job(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 903)
        None_165113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 31), 'None')
        # Getting the type of 'None' (line 903)
        None_165114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 43), 'None')
        # Getting the type of 'None' (line 903)
        None_165115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 58), 'None')
        # Getting the type of 'None' (line 904)
        None_165116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 17), 'None')
        # Getting the type of 'None' (line 904)
        None_165117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 31), 'None')
        defaults = [None_165113, None_165114, None_165115, None_165116, None_165117]
        # Create a new context for function 'set_job'
        module_type_store = module_type_store.open_function_context('set_job', 903, 4, False)
        # Assigning a type to the variable 'self' (line 904)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODR.set_job.__dict__.__setitem__('stypy_localization', localization)
        ODR.set_job.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODR.set_job.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODR.set_job.__dict__.__setitem__('stypy_function_name', 'ODR.set_job')
        ODR.set_job.__dict__.__setitem__('stypy_param_names_list', ['fit_type', 'deriv', 'var_calc', 'del_init', 'restart'])
        ODR.set_job.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODR.set_job.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODR.set_job.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODR.set_job.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODR.set_job.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODR.set_job.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODR.set_job', ['fit_type', 'deriv', 'var_calc', 'del_init', 'restart'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_job', localization, ['fit_type', 'deriv', 'var_calc', 'del_init', 'restart'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_job(...)' code ##################

        str_165118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, (-1)), 'str', '\n        Sets the "job" parameter is a hopefully comprehensible way.\n\n        If an argument is not specified, then the value is left as is. The\n        default value from class initialization is for all of these options set\n        to 0.\n\n        Parameters\n        ----------\n        fit_type : {0, 1, 2} int\n            0 -> explicit ODR\n\n            1 -> implicit ODR\n\n            2 -> ordinary least-squares\n        deriv : {0, 1, 2, 3} int\n            0 -> forward finite differences\n\n            1 -> central finite differences\n\n            2 -> user-supplied derivatives (Jacobians) with results\n              checked by ODRPACK\n\n            3 -> user-supplied derivatives, no checking\n        var_calc : {0, 1, 2} int\n            0 -> calculate asymptotic covariance matrix and fit\n                 parameter uncertainties (V_B, s_B) using derivatives\n                 recomputed at the final solution\n\n            1 -> calculate V_B and s_B using derivatives from last iteration\n\n            2 -> do not calculate V_B and s_B\n        del_init : {0, 1} int\n            0 -> initial input variable offsets set to 0\n\n            1 -> initial offsets provided by user in variable "work"\n        restart : {0, 1} int\n            0 -> fit is not a restart\n\n            1 -> fit is a restart\n\n        Notes\n        -----\n        The permissible values are different from those given on pg. 31 of the\n        ODRPACK User\'s Guide only in that one cannot specify numbers greater than\n        the last value for each variable.\n\n        If one does not supply functions to compute the Jacobians, the fitting\n        procedure will change deriv to 0, finite differences, as a default. To\n        initialize the input variable offsets by yourself, set del_init to 1 and\n        put the offsets into the "work" variable correctly.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 959)
        # Getting the type of 'self' (line 959)
        self_165119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 11), 'self')
        # Obtaining the member 'job' of a type (line 959)
        job_165120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 11), self_165119, 'job')
        # Getting the type of 'None' (line 959)
        None_165121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 23), 'None')
        
        (may_be_165122, more_types_in_union_165123) = may_be_none(job_165120, None_165121)

        if may_be_165122:

            if more_types_in_union_165123:
                # Runtime conditional SSA (line 959)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 960):
            
            # Assigning a List to a Name (line 960):
            
            # Obtaining an instance of the builtin type 'list' (line 960)
            list_165124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 960)
            # Adding element type (line 960)
            int_165125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 21), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 20), list_165124, int_165125)
            # Adding element type (line 960)
            int_165126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 24), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 20), list_165124, int_165126)
            # Adding element type (line 960)
            int_165127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 20), list_165124, int_165127)
            # Adding element type (line 960)
            int_165128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 30), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 20), list_165124, int_165128)
            # Adding element type (line 960)
            int_165129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 33), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 20), list_165124, int_165129)
            
            # Assigning a type to the variable 'job_l' (line 960)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 12), 'job_l', list_165124)

            if more_types_in_union_165123:
                # Runtime conditional SSA for else branch (line 959)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_165122) or more_types_in_union_165123):
            
            # Assigning a List to a Name (line 962):
            
            # Assigning a List to a Name (line 962):
            
            # Obtaining an instance of the builtin type 'list' (line 962)
            list_165130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 962)
            # Adding element type (line 962)
            # Getting the type of 'self' (line 962)
            self_165131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 21), 'self')
            # Obtaining the member 'job' of a type (line 962)
            job_165132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 21), self_165131, 'job')
            int_165133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 33), 'int')
            # Applying the binary operator '//' (line 962)
            result_floordiv_165134 = python_operator(stypy.reporting.localization.Localization(__file__, 962, 21), '//', job_165132, int_165133)
            
            int_165135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 41), 'int')
            # Applying the binary operator '%' (line 962)
            result_mod_165136 = python_operator(stypy.reporting.localization.Localization(__file__, 962, 39), '%', result_floordiv_165134, int_165135)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 20), list_165130, result_mod_165136)
            # Adding element type (line 962)
            # Getting the type of 'self' (line 963)
            self_165137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 21), 'self')
            # Obtaining the member 'job' of a type (line 963)
            job_165138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 21), self_165137, 'job')
            int_165139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 33), 'int')
            # Applying the binary operator '//' (line 963)
            result_floordiv_165140 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 21), '//', job_165138, int_165139)
            
            int_165141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 40), 'int')
            # Applying the binary operator '%' (line 963)
            result_mod_165142 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 38), '%', result_floordiv_165140, int_165141)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 20), list_165130, result_mod_165142)
            # Adding element type (line 962)
            # Getting the type of 'self' (line 964)
            self_165143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 21), 'self')
            # Obtaining the member 'job' of a type (line 964)
            job_165144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 21), self_165143, 'job')
            int_165145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 33), 'int')
            # Applying the binary operator '//' (line 964)
            result_floordiv_165146 = python_operator(stypy.reporting.localization.Localization(__file__, 964, 21), '//', job_165144, int_165145)
            
            int_165147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 39), 'int')
            # Applying the binary operator '%' (line 964)
            result_mod_165148 = python_operator(stypy.reporting.localization.Localization(__file__, 964, 37), '%', result_floordiv_165146, int_165147)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 20), list_165130, result_mod_165148)
            # Adding element type (line 962)
            # Getting the type of 'self' (line 965)
            self_165149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 21), 'self')
            # Obtaining the member 'job' of a type (line 965)
            job_165150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 21), self_165149, 'job')
            int_165151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 33), 'int')
            # Applying the binary operator '//' (line 965)
            result_floordiv_165152 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 21), '//', job_165150, int_165151)
            
            int_165153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 38), 'int')
            # Applying the binary operator '%' (line 965)
            result_mod_165154 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 36), '%', result_floordiv_165152, int_165153)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 20), list_165130, result_mod_165154)
            # Adding element type (line 962)
            # Getting the type of 'self' (line 966)
            self_165155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 21), 'self')
            # Obtaining the member 'job' of a type (line 966)
            job_165156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 21), self_165155, 'job')
            int_165157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 32), 'int')
            # Applying the binary operator '%' (line 966)
            result_mod_165158 = python_operator(stypy.reporting.localization.Localization(__file__, 966, 21), '%', job_165156, int_165157)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 20), list_165130, result_mod_165158)
            
            # Assigning a type to the variable 'job_l' (line 962)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 12), 'job_l', list_165130)

            if (may_be_165122 and more_types_in_union_165123):
                # SSA join for if statement (line 959)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'fit_type' (line 968)
        fit_type_165159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 11), 'fit_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 968)
        tuple_165160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 968)
        # Adding element type (line 968)
        int_165161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 24), tuple_165160, int_165161)
        # Adding element type (line 968)
        int_165162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 24), tuple_165160, int_165162)
        # Adding element type (line 968)
        int_165163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 24), tuple_165160, int_165163)
        
        # Applying the binary operator 'in' (line 968)
        result_contains_165164 = python_operator(stypy.reporting.localization.Localization(__file__, 968, 11), 'in', fit_type_165159, tuple_165160)
        
        # Testing the type of an if condition (line 968)
        if_condition_165165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 968, 8), result_contains_165164)
        # Assigning a type to the variable 'if_condition_165165' (line 968)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 8), 'if_condition_165165', if_condition_165165)
        # SSA begins for if statement (line 968)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 969):
        
        # Assigning a Name to a Subscript (line 969):
        # Getting the type of 'fit_type' (line 969)
        fit_type_165166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 23), 'fit_type')
        # Getting the type of 'job_l' (line 969)
        job_l_165167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 12), 'job_l')
        int_165168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 18), 'int')
        # Storing an element on a container (line 969)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 969, 12), job_l_165167, (int_165168, fit_type_165166))
        # SSA join for if statement (line 968)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'deriv' (line 970)
        deriv_165169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 11), 'deriv')
        
        # Obtaining an instance of the builtin type 'tuple' (line 970)
        tuple_165170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 970)
        # Adding element type (line 970)
        int_165171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 970, 21), tuple_165170, int_165171)
        # Adding element type (line 970)
        int_165172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 970, 21), tuple_165170, int_165172)
        # Adding element type (line 970)
        int_165173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 970, 21), tuple_165170, int_165173)
        # Adding element type (line 970)
        int_165174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 970, 21), tuple_165170, int_165174)
        
        # Applying the binary operator 'in' (line 970)
        result_contains_165175 = python_operator(stypy.reporting.localization.Localization(__file__, 970, 11), 'in', deriv_165169, tuple_165170)
        
        # Testing the type of an if condition (line 970)
        if_condition_165176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 970, 8), result_contains_165175)
        # Assigning a type to the variable 'if_condition_165176' (line 970)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 8), 'if_condition_165176', if_condition_165176)
        # SSA begins for if statement (line 970)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 971):
        
        # Assigning a Name to a Subscript (line 971):
        # Getting the type of 'deriv' (line 971)
        deriv_165177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 23), 'deriv')
        # Getting the type of 'job_l' (line 971)
        job_l_165178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 12), 'job_l')
        int_165179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 18), 'int')
        # Storing an element on a container (line 971)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 971, 12), job_l_165178, (int_165179, deriv_165177))
        # SSA join for if statement (line 970)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'var_calc' (line 972)
        var_calc_165180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 11), 'var_calc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 972)
        tuple_165181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 972)
        # Adding element type (line 972)
        int_165182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 972, 24), tuple_165181, int_165182)
        # Adding element type (line 972)
        int_165183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 972, 24), tuple_165181, int_165183)
        # Adding element type (line 972)
        int_165184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 972, 24), tuple_165181, int_165184)
        
        # Applying the binary operator 'in' (line 972)
        result_contains_165185 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 11), 'in', var_calc_165180, tuple_165181)
        
        # Testing the type of an if condition (line 972)
        if_condition_165186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 972, 8), result_contains_165185)
        # Assigning a type to the variable 'if_condition_165186' (line 972)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 8), 'if_condition_165186', if_condition_165186)
        # SSA begins for if statement (line 972)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 973):
        
        # Assigning a Name to a Subscript (line 973):
        # Getting the type of 'var_calc' (line 973)
        var_calc_165187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 23), 'var_calc')
        # Getting the type of 'job_l' (line 973)
        job_l_165188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 12), 'job_l')
        int_165189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 18), 'int')
        # Storing an element on a container (line 973)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 12), job_l_165188, (int_165189, var_calc_165187))
        # SSA join for if statement (line 972)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'del_init' (line 974)
        del_init_165190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 11), 'del_init')
        
        # Obtaining an instance of the builtin type 'tuple' (line 974)
        tuple_165191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 974)
        # Adding element type (line 974)
        int_165192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 24), tuple_165191, int_165192)
        # Adding element type (line 974)
        int_165193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 24), tuple_165191, int_165193)
        
        # Applying the binary operator 'in' (line 974)
        result_contains_165194 = python_operator(stypy.reporting.localization.Localization(__file__, 974, 11), 'in', del_init_165190, tuple_165191)
        
        # Testing the type of an if condition (line 974)
        if_condition_165195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 974, 8), result_contains_165194)
        # Assigning a type to the variable 'if_condition_165195' (line 974)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'if_condition_165195', if_condition_165195)
        # SSA begins for if statement (line 974)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 975):
        
        # Assigning a Name to a Subscript (line 975):
        # Getting the type of 'del_init' (line 975)
        del_init_165196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 23), 'del_init')
        # Getting the type of 'job_l' (line 975)
        job_l_165197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 12), 'job_l')
        int_165198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 18), 'int')
        # Storing an element on a container (line 975)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 975, 12), job_l_165197, (int_165198, del_init_165196))
        # SSA join for if statement (line 974)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'restart' (line 976)
        restart_165199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 11), 'restart')
        
        # Obtaining an instance of the builtin type 'tuple' (line 976)
        tuple_165200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 976)
        # Adding element type (line 976)
        int_165201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 976, 23), tuple_165200, int_165201)
        # Adding element type (line 976)
        int_165202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 976, 23), tuple_165200, int_165202)
        
        # Applying the binary operator 'in' (line 976)
        result_contains_165203 = python_operator(stypy.reporting.localization.Localization(__file__, 976, 11), 'in', restart_165199, tuple_165200)
        
        # Testing the type of an if condition (line 976)
        if_condition_165204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 976, 8), result_contains_165203)
        # Assigning a type to the variable 'if_condition_165204' (line 976)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 8), 'if_condition_165204', if_condition_165204)
        # SSA begins for if statement (line 976)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 977):
        
        # Assigning a Name to a Subscript (line 977):
        # Getting the type of 'restart' (line 977)
        restart_165205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 23), 'restart')
        # Getting the type of 'job_l' (line 977)
        job_l_165206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 12), 'job_l')
        int_165207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 18), 'int')
        # Storing an element on a container (line 977)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 977, 12), job_l_165206, (int_165207, restart_165205))
        # SSA join for if statement (line 976)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Attribute (line 979):
        
        # Assigning a BinOp to a Attribute (line 979):
        
        # Obtaining the type of the subscript
        int_165208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 26), 'int')
        # Getting the type of 'job_l' (line 979)
        job_l_165209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 20), 'job_l')
        # Obtaining the member '__getitem__' of a type (line 979)
        getitem___165210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 20), job_l_165209, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 979)
        subscript_call_result_165211 = invoke(stypy.reporting.localization.Localization(__file__, 979, 20), getitem___165210, int_165208)
        
        int_165212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 29), 'int')
        # Applying the binary operator '*' (line 979)
        result_mul_165213 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 20), '*', subscript_call_result_165211, int_165212)
        
        
        # Obtaining the type of the subscript
        int_165214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 43), 'int')
        # Getting the type of 'job_l' (line 979)
        job_l_165215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 37), 'job_l')
        # Obtaining the member '__getitem__' of a type (line 979)
        getitem___165216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 37), job_l_165215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 979)
        subscript_call_result_165217 = invoke(stypy.reporting.localization.Localization(__file__, 979, 37), getitem___165216, int_165214)
        
        int_165218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 46), 'int')
        # Applying the binary operator '*' (line 979)
        result_mul_165219 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 37), '*', subscript_call_result_165217, int_165218)
        
        # Applying the binary operator '+' (line 979)
        result_add_165220 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 20), '+', result_mul_165213, result_mul_165219)
        
        
        # Obtaining the type of the subscript
        int_165221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 26), 'int')
        # Getting the type of 'job_l' (line 980)
        job_l_165222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 20), 'job_l')
        # Obtaining the member '__getitem__' of a type (line 980)
        getitem___165223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 20), job_l_165222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 980)
        subscript_call_result_165224 = invoke(stypy.reporting.localization.Localization(__file__, 980, 20), getitem___165223, int_165221)
        
        int_165225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 29), 'int')
        # Applying the binary operator '*' (line 980)
        result_mul_165226 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 20), '*', subscript_call_result_165224, int_165225)
        
        # Applying the binary operator '+' (line 979)
        result_add_165227 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 51), '+', result_add_165220, result_mul_165226)
        
        
        # Obtaining the type of the subscript
        int_165228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 41), 'int')
        # Getting the type of 'job_l' (line 980)
        job_l_165229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 35), 'job_l')
        # Obtaining the member '__getitem__' of a type (line 980)
        getitem___165230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 35), job_l_165229, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 980)
        subscript_call_result_165231 = invoke(stypy.reporting.localization.Localization(__file__, 980, 35), getitem___165230, int_165228)
        
        int_165232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 44), 'int')
        # Applying the binary operator '*' (line 980)
        result_mul_165233 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 35), '*', subscript_call_result_165231, int_165232)
        
        # Applying the binary operator '+' (line 980)
        result_add_165234 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 33), '+', result_add_165227, result_mul_165233)
        
        
        # Obtaining the type of the subscript
        int_165235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 55), 'int')
        # Getting the type of 'job_l' (line 980)
        job_l_165236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 49), 'job_l')
        # Obtaining the member '__getitem__' of a type (line 980)
        getitem___165237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 49), job_l_165236, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 980)
        subscript_call_result_165238 = invoke(stypy.reporting.localization.Localization(__file__, 980, 49), getitem___165237, int_165235)
        
        # Applying the binary operator '+' (line 980)
        result_add_165239 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 47), '+', result_add_165234, subscript_call_result_165238)
        
        # Getting the type of 'self' (line 979)
        self_165240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 8), 'self')
        # Setting the type of the member 'job' of a type (line 979)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 8), self_165240, 'job', result_add_165239)
        
        # ################# End of 'set_job(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_job' in the type store
        # Getting the type of 'stypy_return_type' (line 903)
        stypy_return_type_165241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165241)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_job'
        return stypy_return_type_165241


    @norecursion
    def set_iprint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 982)
        None_165242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 30), 'None')
        # Getting the type of 'None' (line 982)
        None_165243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 44), 'None')
        # Getting the type of 'None' (line 983)
        None_165244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 13), 'None')
        # Getting the type of 'None' (line 983)
        None_165245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 27), 'None')
        # Getting the type of 'None' (line 983)
        None_165246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 43), 'None')
        # Getting the type of 'None' (line 983)
        None_165247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 55), 'None')
        # Getting the type of 'None' (line 983)
        None_165248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 70), 'None')
        defaults = [None_165242, None_165243, None_165244, None_165245, None_165246, None_165247, None_165248]
        # Create a new context for function 'set_iprint'
        module_type_store = module_type_store.open_function_context('set_iprint', 982, 4, False)
        # Assigning a type to the variable 'self' (line 983)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODR.set_iprint.__dict__.__setitem__('stypy_localization', localization)
        ODR.set_iprint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODR.set_iprint.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODR.set_iprint.__dict__.__setitem__('stypy_function_name', 'ODR.set_iprint')
        ODR.set_iprint.__dict__.__setitem__('stypy_param_names_list', ['init', 'so_init', 'iter', 'so_iter', 'iter_step', 'final', 'so_final'])
        ODR.set_iprint.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODR.set_iprint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODR.set_iprint.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODR.set_iprint.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODR.set_iprint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODR.set_iprint.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODR.set_iprint', ['init', 'so_init', 'iter', 'so_iter', 'iter_step', 'final', 'so_final'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_iprint', localization, ['init', 'so_init', 'iter', 'so_iter', 'iter_step', 'final', 'so_final'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_iprint(...)' code ##################

        str_165249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, (-1)), 'str', ' Set the iprint parameter for the printing of computation reports.\n\n        If any of the arguments are specified here, then they are set in the\n        iprint member. If iprint is not set manually or with this method, then\n        ODRPACK defaults to no printing. If no filename is specified with the\n        member rptfile, then ODRPACK prints to stdout. One can tell ODRPACK to\n        print to stdout in addition to the specified filename by setting the\n        so_* arguments to this function, but one cannot specify to print to\n        stdout but not a file since one can do that by not specifying a rptfile\n        filename.\n\n        There are three reports: initialization, iteration, and final reports.\n        They are represented by the arguments init, iter, and final\n        respectively.  The permissible values are 0, 1, and 2 representing "no\n        report", "short report", and "long report" respectively.\n\n        The argument iter_step (0 <= iter_step <= 9) specifies how often to make\n        the iteration report; the report will be made for every iter_step\'th\n        iteration starting with iteration one. If iter_step == 0, then no\n        iteration report is made, regardless of the other arguments.\n\n        If the rptfile is None, then any so_* arguments supplied will raise an\n        exception.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 1008)
        # Getting the type of 'self' (line 1008)
        self_165250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 11), 'self')
        # Obtaining the member 'iprint' of a type (line 1008)
        iprint_165251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 11), self_165250, 'iprint')
        # Getting the type of 'None' (line 1008)
        None_165252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 26), 'None')
        
        (may_be_165253, more_types_in_union_165254) = may_be_none(iprint_165251, None_165252)

        if may_be_165253:

            if more_types_in_union_165254:
                # Runtime conditional SSA (line 1008)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 1009):
            
            # Assigning a Num to a Attribute (line 1009):
            int_165255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 26), 'int')
            # Getting the type of 'self' (line 1009)
            self_165256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 12), 'self')
            # Setting the type of the member 'iprint' of a type (line 1009)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1009, 12), self_165256, 'iprint', int_165255)

            if more_types_in_union_165254:
                # SSA join for if statement (line 1008)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 1011):
        
        # Assigning a List to a Name (line 1011):
        
        # Obtaining an instance of the builtin type 'list' (line 1011)
        list_165257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1011)
        # Adding element type (line 1011)
        # Getting the type of 'self' (line 1011)
        self_165258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 14), 'self')
        # Obtaining the member 'iprint' of a type (line 1011)
        iprint_165259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1011, 14), self_165258, 'iprint')
        int_165260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 29), 'int')
        # Applying the binary operator '//' (line 1011)
        result_floordiv_165261 = python_operator(stypy.reporting.localization.Localization(__file__, 1011, 14), '//', iprint_165259, int_165260)
        
        int_165262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 36), 'int')
        # Applying the binary operator '%' (line 1011)
        result_mod_165263 = python_operator(stypy.reporting.localization.Localization(__file__, 1011, 34), '%', result_floordiv_165261, int_165262)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 13), list_165257, result_mod_165263)
        # Adding element type (line 1011)
        # Getting the type of 'self' (line 1012)
        self_165264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 14), 'self')
        # Obtaining the member 'iprint' of a type (line 1012)
        iprint_165265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 14), self_165264, 'iprint')
        int_165266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 29), 'int')
        # Applying the binary operator '//' (line 1012)
        result_floordiv_165267 = python_operator(stypy.reporting.localization.Localization(__file__, 1012, 14), '//', iprint_165265, int_165266)
        
        int_165268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 35), 'int')
        # Applying the binary operator '%' (line 1012)
        result_mod_165269 = python_operator(stypy.reporting.localization.Localization(__file__, 1012, 33), '%', result_floordiv_165267, int_165268)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 13), list_165257, result_mod_165269)
        # Adding element type (line 1011)
        # Getting the type of 'self' (line 1013)
        self_165270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 14), 'self')
        # Obtaining the member 'iprint' of a type (line 1013)
        iprint_165271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1013, 14), self_165270, 'iprint')
        int_165272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1013, 29), 'int')
        # Applying the binary operator '//' (line 1013)
        result_floordiv_165273 = python_operator(stypy.reporting.localization.Localization(__file__, 1013, 14), '//', iprint_165271, int_165272)
        
        int_165274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1013, 34), 'int')
        # Applying the binary operator '%' (line 1013)
        result_mod_165275 = python_operator(stypy.reporting.localization.Localization(__file__, 1013, 32), '%', result_floordiv_165273, int_165274)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 13), list_165257, result_mod_165275)
        # Adding element type (line 1011)
        # Getting the type of 'self' (line 1014)
        self_165276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 14), 'self')
        # Obtaining the member 'iprint' of a type (line 1014)
        iprint_165277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 14), self_165276, 'iprint')
        int_165278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, 28), 'int')
        # Applying the binary operator '%' (line 1014)
        result_mod_165279 = python_operator(stypy.reporting.localization.Localization(__file__, 1014, 14), '%', iprint_165277, int_165278)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 13), list_165257, result_mod_165279)
        
        # Assigning a type to the variable 'ip' (line 1011)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 8), 'ip', list_165257)
        
        # Assigning a List to a Name (line 1018):
        
        # Assigning a List to a Name (line 1018):
        
        # Obtaining an instance of the builtin type 'list' (line 1018)
        list_165280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1018)
        # Adding element type (line 1018)
        
        # Obtaining an instance of the builtin type 'list' (line 1018)
        list_165281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1018)
        # Adding element type (line 1018)
        int_165282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 18), list_165281, int_165282)
        # Adding element type (line 1018)
        int_165283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 18), list_165281, int_165283)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 17), list_165280, list_165281)
        # Adding element type (line 1018)
        
        # Obtaining an instance of the builtin type 'list' (line 1019)
        list_165284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1019)
        # Adding element type (line 1019)
        int_165285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1019, 18), list_165284, int_165285)
        # Adding element type (line 1019)
        int_165286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1019, 18), list_165284, int_165286)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 17), list_165280, list_165284)
        # Adding element type (line 1018)
        
        # Obtaining an instance of the builtin type 'list' (line 1020)
        list_165287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1020)
        # Adding element type (line 1020)
        int_165288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1020, 18), list_165287, int_165288)
        # Adding element type (line 1020)
        int_165289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1020, 18), list_165287, int_165289)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 17), list_165280, list_165287)
        # Adding element type (line 1018)
        
        # Obtaining an instance of the builtin type 'list' (line 1021)
        list_165290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1021)
        # Adding element type (line 1021)
        int_165291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 18), list_165290, int_165291)
        # Adding element type (line 1021)
        int_165292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 18), list_165290, int_165292)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 17), list_165280, list_165290)
        # Adding element type (line 1018)
        
        # Obtaining an instance of the builtin type 'list' (line 1022)
        list_165293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1022)
        # Adding element type (line 1022)
        int_165294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1022, 18), list_165293, int_165294)
        # Adding element type (line 1022)
        int_165295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1022, 18), list_165293, int_165295)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 17), list_165280, list_165293)
        # Adding element type (line 1018)
        
        # Obtaining an instance of the builtin type 'list' (line 1023)
        list_165296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1023)
        # Adding element type (line 1023)
        int_165297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1023, 18), list_165296, int_165297)
        # Adding element type (line 1023)
        int_165298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1023, 18), list_165296, int_165298)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 17), list_165280, list_165296)
        # Adding element type (line 1018)
        
        # Obtaining an instance of the builtin type 'list' (line 1024)
        list_165299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1024)
        # Adding element type (line 1024)
        int_165300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1024, 18), list_165299, int_165300)
        # Adding element type (line 1024)
        int_165301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1024, 18), list_165299, int_165301)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1018, 17), list_165280, list_165299)
        
        # Assigning a type to the variable 'ip2arg' (line 1018)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1018, 8), 'ip2arg', list_165280)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 1026)
        self_165302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 12), 'self')
        # Obtaining the member 'rptfile' of a type (line 1026)
        rptfile_165303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 12), self_165302, 'rptfile')
        # Getting the type of 'None' (line 1026)
        None_165304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 28), 'None')
        # Applying the binary operator 'is' (line 1026)
        result_is__165305 = python_operator(stypy.reporting.localization.Localization(__file__, 1026, 12), 'is', rptfile_165303, None_165304)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'so_init' (line 1027)
        so_init_165306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 13), 'so_init')
        # Getting the type of 'None' (line 1027)
        None_165307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 28), 'None')
        # Applying the binary operator 'isnot' (line 1027)
        result_is_not_165308 = python_operator(stypy.reporting.localization.Localization(__file__, 1027, 13), 'isnot', so_init_165306, None_165307)
        
        
        # Getting the type of 'so_iter' (line 1028)
        so_iter_165309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 13), 'so_iter')
        # Getting the type of 'None' (line 1028)
        None_165310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 28), 'None')
        # Applying the binary operator 'isnot' (line 1028)
        result_is_not_165311 = python_operator(stypy.reporting.localization.Localization(__file__, 1028, 13), 'isnot', so_iter_165309, None_165310)
        
        # Applying the binary operator 'or' (line 1027)
        result_or_keyword_165312 = python_operator(stypy.reporting.localization.Localization(__file__, 1027, 13), 'or', result_is_not_165308, result_is_not_165311)
        
        # Getting the type of 'so_final' (line 1029)
        so_final_165313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 13), 'so_final')
        # Getting the type of 'None' (line 1029)
        None_165314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 29), 'None')
        # Applying the binary operator 'isnot' (line 1029)
        result_is_not_165315 = python_operator(stypy.reporting.localization.Localization(__file__, 1029, 13), 'isnot', so_final_165313, None_165314)
        
        # Applying the binary operator 'or' (line 1027)
        result_or_keyword_165316 = python_operator(stypy.reporting.localization.Localization(__file__, 1027, 13), 'or', result_or_keyword_165312, result_is_not_165315)
        
        # Applying the binary operator 'and' (line 1026)
        result_and_keyword_165317 = python_operator(stypy.reporting.localization.Localization(__file__, 1026, 12), 'and', result_is__165305, result_or_keyword_165316)
        
        # Testing the type of an if condition (line 1026)
        if_condition_165318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1026, 8), result_and_keyword_165317)
        # Assigning a type to the variable 'if_condition_165318' (line 1026)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1026, 8), 'if_condition_165318', if_condition_165318)
        # SSA begins for if statement (line 1026)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to OdrError(...): (line 1030)
        # Processing the call arguments (line 1030)
        str_165320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 16), 'str', 'no rptfile specified, cannot output to stdout twice')
        # Processing the call keyword arguments (line 1030)
        kwargs_165321 = {}
        # Getting the type of 'OdrError' (line 1030)
        OdrError_165319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 18), 'OdrError', False)
        # Calling OdrError(args, kwargs) (line 1030)
        OdrError_call_result_165322 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 18), OdrError_165319, *[str_165320], **kwargs_165321)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1030, 12), OdrError_call_result_165322, 'raise parameter', BaseException)
        # SSA join for if statement (line 1026)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1033):
        
        # Assigning a BinOp to a Name (line 1033):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_165323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 29), 'int')
        # Getting the type of 'ip' (line 1033)
        ip_165324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 26), 'ip')
        # Obtaining the member '__getitem__' of a type (line 1033)
        getitem___165325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 26), ip_165324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
        subscript_call_result_165326 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 26), getitem___165325, int_165323)
        
        # Getting the type of 'ip2arg' (line 1033)
        ip2arg_165327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 19), 'ip2arg')
        # Obtaining the member '__getitem__' of a type (line 1033)
        getitem___165328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 19), ip2arg_165327, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
        subscript_call_result_165329 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 19), getitem___165328, subscript_call_result_165326)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_165330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 45), 'int')
        # Getting the type of 'ip' (line 1033)
        ip_165331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 42), 'ip')
        # Obtaining the member '__getitem__' of a type (line 1033)
        getitem___165332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 42), ip_165331, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
        subscript_call_result_165333 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 42), getitem___165332, int_165330)
        
        # Getting the type of 'ip2arg' (line 1033)
        ip2arg_165334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 35), 'ip2arg')
        # Obtaining the member '__getitem__' of a type (line 1033)
        getitem___165335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 35), ip2arg_165334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
        subscript_call_result_165336 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 35), getitem___165335, subscript_call_result_165333)
        
        # Applying the binary operator '+' (line 1033)
        result_add_165337 = python_operator(stypy.reporting.localization.Localization(__file__, 1033, 19), '+', subscript_call_result_165329, subscript_call_result_165336)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_165338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 61), 'int')
        # Getting the type of 'ip' (line 1033)
        ip_165339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 58), 'ip')
        # Obtaining the member '__getitem__' of a type (line 1033)
        getitem___165340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 58), ip_165339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
        subscript_call_result_165341 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 58), getitem___165340, int_165338)
        
        # Getting the type of 'ip2arg' (line 1033)
        ip2arg_165342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 51), 'ip2arg')
        # Obtaining the member '__getitem__' of a type (line 1033)
        getitem___165343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 51), ip2arg_165342, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
        subscript_call_result_165344 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 51), getitem___165343, subscript_call_result_165341)
        
        # Applying the binary operator '+' (line 1033)
        result_add_165345 = python_operator(stypy.reporting.localization.Localization(__file__, 1033, 49), '+', result_add_165337, subscript_call_result_165344)
        
        # Assigning a type to the variable 'iprint_l' (line 1033)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'iprint_l', result_add_165345)
        
        # Type idiom detected: calculating its left and rigth part (line 1035)
        # Getting the type of 'init' (line 1035)
        init_165346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 8), 'init')
        # Getting the type of 'None' (line 1035)
        None_165347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 23), 'None')
        
        (may_be_165348, more_types_in_union_165349) = may_not_be_none(init_165346, None_165347)

        if may_be_165348:

            if more_types_in_union_165349:
                # Runtime conditional SSA (line 1035)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1036):
            
            # Assigning a Name to a Subscript (line 1036):
            # Getting the type of 'init' (line 1036)
            init_165350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 26), 'init')
            # Getting the type of 'iprint_l' (line 1036)
            iprint_l_165351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 12), 'iprint_l')
            int_165352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 21), 'int')
            # Storing an element on a container (line 1036)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1036, 12), iprint_l_165351, (int_165352, init_165350))

            if more_types_in_union_165349:
                # SSA join for if statement (line 1035)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1037)
        # Getting the type of 'so_init' (line 1037)
        so_init_165353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'so_init')
        # Getting the type of 'None' (line 1037)
        None_165354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 26), 'None')
        
        (may_be_165355, more_types_in_union_165356) = may_not_be_none(so_init_165353, None_165354)

        if may_be_165355:

            if more_types_in_union_165356:
                # Runtime conditional SSA (line 1037)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1038):
            
            # Assigning a Name to a Subscript (line 1038):
            # Getting the type of 'so_init' (line 1038)
            so_init_165357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 26), 'so_init')
            # Getting the type of 'iprint_l' (line 1038)
            iprint_l_165358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 12), 'iprint_l')
            int_165359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 21), 'int')
            # Storing an element on a container (line 1038)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1038, 12), iprint_l_165358, (int_165359, so_init_165357))

            if more_types_in_union_165356:
                # SSA join for if statement (line 1037)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1039)
        # Getting the type of 'iter' (line 1039)
        iter_165360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'iter')
        # Getting the type of 'None' (line 1039)
        None_165361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 23), 'None')
        
        (may_be_165362, more_types_in_union_165363) = may_not_be_none(iter_165360, None_165361)

        if may_be_165362:

            if more_types_in_union_165363:
                # Runtime conditional SSA (line 1039)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1040):
            
            # Assigning a Name to a Subscript (line 1040):
            # Getting the type of 'iter' (line 1040)
            iter_165364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 26), 'iter')
            # Getting the type of 'iprint_l' (line 1040)
            iprint_l_165365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 12), 'iprint_l')
            int_165366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 21), 'int')
            # Storing an element on a container (line 1040)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1040, 12), iprint_l_165365, (int_165366, iter_165364))

            if more_types_in_union_165363:
                # SSA join for if statement (line 1039)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1041)
        # Getting the type of 'so_iter' (line 1041)
        so_iter_165367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 8), 'so_iter')
        # Getting the type of 'None' (line 1041)
        None_165368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 26), 'None')
        
        (may_be_165369, more_types_in_union_165370) = may_not_be_none(so_iter_165367, None_165368)

        if may_be_165369:

            if more_types_in_union_165370:
                # Runtime conditional SSA (line 1041)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1042):
            
            # Assigning a Name to a Subscript (line 1042):
            # Getting the type of 'so_iter' (line 1042)
            so_iter_165371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 26), 'so_iter')
            # Getting the type of 'iprint_l' (line 1042)
            iprint_l_165372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 12), 'iprint_l')
            int_165373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 21), 'int')
            # Storing an element on a container (line 1042)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 12), iprint_l_165372, (int_165373, so_iter_165371))

            if more_types_in_union_165370:
                # SSA join for if statement (line 1041)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1043)
        # Getting the type of 'final' (line 1043)
        final_165374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 8), 'final')
        # Getting the type of 'None' (line 1043)
        None_165375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 24), 'None')
        
        (may_be_165376, more_types_in_union_165377) = may_not_be_none(final_165374, None_165375)

        if may_be_165376:

            if more_types_in_union_165377:
                # Runtime conditional SSA (line 1043)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1044):
            
            # Assigning a Name to a Subscript (line 1044):
            # Getting the type of 'final' (line 1044)
            final_165378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 26), 'final')
            # Getting the type of 'iprint_l' (line 1044)
            iprint_l_165379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 12), 'iprint_l')
            int_165380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 21), 'int')
            # Storing an element on a container (line 1044)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1044, 12), iprint_l_165379, (int_165380, final_165378))

            if more_types_in_union_165377:
                # SSA join for if statement (line 1043)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1045)
        # Getting the type of 'so_final' (line 1045)
        so_final_165381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 8), 'so_final')
        # Getting the type of 'None' (line 1045)
        None_165382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 27), 'None')
        
        (may_be_165383, more_types_in_union_165384) = may_not_be_none(so_final_165381, None_165382)

        if may_be_165383:

            if more_types_in_union_165384:
                # Runtime conditional SSA (line 1045)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1046):
            
            # Assigning a Name to a Subscript (line 1046):
            # Getting the type of 'so_final' (line 1046)
            so_final_165385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 26), 'so_final')
            # Getting the type of 'iprint_l' (line 1046)
            iprint_l_165386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 12), 'iprint_l')
            int_165387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 21), 'int')
            # Storing an element on a container (line 1046)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1046, 12), iprint_l_165386, (int_165387, so_final_165385))

            if more_types_in_union_165384:
                # SSA join for if statement (line 1045)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'iter_step' (line 1048)
        iter_step_165388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 11), 'iter_step')
        
        # Call to range(...): (line 1048)
        # Processing the call arguments (line 1048)
        int_165390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 30), 'int')
        # Processing the call keyword arguments (line 1048)
        kwargs_165391 = {}
        # Getting the type of 'range' (line 1048)
        range_165389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 24), 'range', False)
        # Calling range(args, kwargs) (line 1048)
        range_call_result_165392 = invoke(stypy.reporting.localization.Localization(__file__, 1048, 24), range_165389, *[int_165390], **kwargs_165391)
        
        # Applying the binary operator 'in' (line 1048)
        result_contains_165393 = python_operator(stypy.reporting.localization.Localization(__file__, 1048, 11), 'in', iter_step_165388, range_call_result_165392)
        
        # Testing the type of an if condition (line 1048)
        if_condition_165394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1048, 8), result_contains_165393)
        # Assigning a type to the variable 'if_condition_165394' (line 1048)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1048, 8), 'if_condition_165394', if_condition_165394)
        # SSA begins for if statement (line 1048)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 1050):
        
        # Assigning a Name to a Subscript (line 1050):
        # Getting the type of 'iter_step' (line 1050)
        iter_step_165395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 20), 'iter_step')
        # Getting the type of 'ip' (line 1050)
        ip_165396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 12), 'ip')
        int_165397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 15), 'int')
        # Storing an element on a container (line 1050)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1050, 12), ip_165396, (int_165397, iter_step_165395))
        # SSA join for if statement (line 1048)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 1052):
        
        # Assigning a Call to a Subscript (line 1052):
        
        # Call to index(...): (line 1052)
        # Processing the call arguments (line 1052)
        
        # Obtaining the type of the subscript
        int_165400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 38), 'int')
        int_165401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 40), 'int')
        slice_165402 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1052, 29), int_165400, int_165401, None)
        # Getting the type of 'iprint_l' (line 1052)
        iprint_l_165403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 29), 'iprint_l', False)
        # Obtaining the member '__getitem__' of a type (line 1052)
        getitem___165404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 29), iprint_l_165403, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1052)
        subscript_call_result_165405 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 29), getitem___165404, slice_165402)
        
        # Processing the call keyword arguments (line 1052)
        kwargs_165406 = {}
        # Getting the type of 'ip2arg' (line 1052)
        ip2arg_165398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 16), 'ip2arg', False)
        # Obtaining the member 'index' of a type (line 1052)
        index_165399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 16), ip2arg_165398, 'index')
        # Calling index(args, kwargs) (line 1052)
        index_call_result_165407 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 16), index_165399, *[subscript_call_result_165405], **kwargs_165406)
        
        # Getting the type of 'ip' (line 1052)
        ip_165408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'ip')
        int_165409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 11), 'int')
        # Storing an element on a container (line 1052)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1052, 8), ip_165408, (int_165409, index_call_result_165407))
        
        # Assigning a Call to a Subscript (line 1053):
        
        # Assigning a Call to a Subscript (line 1053):
        
        # Call to index(...): (line 1053)
        # Processing the call arguments (line 1053)
        
        # Obtaining the type of the subscript
        int_165412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 38), 'int')
        int_165413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 40), 'int')
        slice_165414 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1053, 29), int_165412, int_165413, None)
        # Getting the type of 'iprint_l' (line 1053)
        iprint_l_165415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 29), 'iprint_l', False)
        # Obtaining the member '__getitem__' of a type (line 1053)
        getitem___165416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 29), iprint_l_165415, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1053)
        subscript_call_result_165417 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 29), getitem___165416, slice_165414)
        
        # Processing the call keyword arguments (line 1053)
        kwargs_165418 = {}
        # Getting the type of 'ip2arg' (line 1053)
        ip2arg_165410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 16), 'ip2arg', False)
        # Obtaining the member 'index' of a type (line 1053)
        index_165411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 16), ip2arg_165410, 'index')
        # Calling index(args, kwargs) (line 1053)
        index_call_result_165419 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 16), index_165411, *[subscript_call_result_165417], **kwargs_165418)
        
        # Getting the type of 'ip' (line 1053)
        ip_165420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'ip')
        int_165421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 11), 'int')
        # Storing an element on a container (line 1053)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1053, 8), ip_165420, (int_165421, index_call_result_165419))
        
        # Assigning a Call to a Subscript (line 1054):
        
        # Assigning a Call to a Subscript (line 1054):
        
        # Call to index(...): (line 1054)
        # Processing the call arguments (line 1054)
        
        # Obtaining the type of the subscript
        int_165424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1054, 38), 'int')
        int_165425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1054, 40), 'int')
        slice_165426 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1054, 29), int_165424, int_165425, None)
        # Getting the type of 'iprint_l' (line 1054)
        iprint_l_165427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 29), 'iprint_l', False)
        # Obtaining the member '__getitem__' of a type (line 1054)
        getitem___165428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 29), iprint_l_165427, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1054)
        subscript_call_result_165429 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 29), getitem___165428, slice_165426)
        
        # Processing the call keyword arguments (line 1054)
        kwargs_165430 = {}
        # Getting the type of 'ip2arg' (line 1054)
        ip2arg_165422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 16), 'ip2arg', False)
        # Obtaining the member 'index' of a type (line 1054)
        index_165423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 16), ip2arg_165422, 'index')
        # Calling index(args, kwargs) (line 1054)
        index_call_result_165431 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 16), index_165423, *[subscript_call_result_165429], **kwargs_165430)
        
        # Getting the type of 'ip' (line 1054)
        ip_165432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'ip')
        int_165433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1054, 11), 'int')
        # Storing an element on a container (line 1054)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1054, 8), ip_165432, (int_165433, index_call_result_165431))
        
        # Assigning a BinOp to a Attribute (line 1056):
        
        # Assigning a BinOp to a Attribute (line 1056):
        
        # Obtaining the type of the subscript
        int_165434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 25), 'int')
        # Getting the type of 'ip' (line 1056)
        ip_165435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 22), 'ip')
        # Obtaining the member '__getitem__' of a type (line 1056)
        getitem___165436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 22), ip_165435, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1056)
        subscript_call_result_165437 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 22), getitem___165436, int_165434)
        
        int_165438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 28), 'int')
        # Applying the binary operator '*' (line 1056)
        result_mul_165439 = python_operator(stypy.reporting.localization.Localization(__file__, 1056, 22), '*', subscript_call_result_165437, int_165438)
        
        
        # Obtaining the type of the subscript
        int_165440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 38), 'int')
        # Getting the type of 'ip' (line 1056)
        ip_165441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 35), 'ip')
        # Obtaining the member '__getitem__' of a type (line 1056)
        getitem___165442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 35), ip_165441, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1056)
        subscript_call_result_165443 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 35), getitem___165442, int_165440)
        
        int_165444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 41), 'int')
        # Applying the binary operator '*' (line 1056)
        result_mul_165445 = python_operator(stypy.reporting.localization.Localization(__file__, 1056, 35), '*', subscript_call_result_165443, int_165444)
        
        # Applying the binary operator '+' (line 1056)
        result_add_165446 = python_operator(stypy.reporting.localization.Localization(__file__, 1056, 22), '+', result_mul_165439, result_mul_165445)
        
        
        # Obtaining the type of the subscript
        int_165447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 50), 'int')
        # Getting the type of 'ip' (line 1056)
        ip_165448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 47), 'ip')
        # Obtaining the member '__getitem__' of a type (line 1056)
        getitem___165449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 47), ip_165448, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1056)
        subscript_call_result_165450 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 47), getitem___165449, int_165447)
        
        int_165451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 53), 'int')
        # Applying the binary operator '*' (line 1056)
        result_mul_165452 = python_operator(stypy.reporting.localization.Localization(__file__, 1056, 47), '*', subscript_call_result_165450, int_165451)
        
        # Applying the binary operator '+' (line 1056)
        result_add_165453 = python_operator(stypy.reporting.localization.Localization(__file__, 1056, 45), '+', result_add_165446, result_mul_165452)
        
        
        # Obtaining the type of the subscript
        int_165454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 61), 'int')
        # Getting the type of 'ip' (line 1056)
        ip_165455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 58), 'ip')
        # Obtaining the member '__getitem__' of a type (line 1056)
        getitem___165456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 58), ip_165455, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1056)
        subscript_call_result_165457 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 58), getitem___165456, int_165454)
        
        # Applying the binary operator '+' (line 1056)
        result_add_165458 = python_operator(stypy.reporting.localization.Localization(__file__, 1056, 56), '+', result_add_165453, subscript_call_result_165457)
        
        # Getting the type of 'self' (line 1056)
        self_165459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 8), 'self')
        # Setting the type of the member 'iprint' of a type (line 1056)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 8), self_165459, 'iprint', result_add_165458)
        
        # ################# End of 'set_iprint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_iprint' in the type store
        # Getting the type of 'stypy_return_type' (line 982)
        stypy_return_type_165460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165460)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_iprint'
        return stypy_return_type_165460


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 1058, 4, False)
        # Assigning a type to the variable 'self' (line 1059)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODR.run.__dict__.__setitem__('stypy_localization', localization)
        ODR.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODR.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODR.run.__dict__.__setitem__('stypy_function_name', 'ODR.run')
        ODR.run.__dict__.__setitem__('stypy_param_names_list', [])
        ODR.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODR.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODR.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODR.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODR.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODR.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODR.run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        str_165461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, (-1)), 'str', ' Run the fitting routine with all of the information given and with ``full_output=1``.\n\n        Returns\n        -------\n        output : Output instance\n            This object is also assigned to the attribute .output .\n        ')
        
        # Assigning a Tuple to a Name (line 1067):
        
        # Assigning a Tuple to a Name (line 1067):
        
        # Obtaining an instance of the builtin type 'tuple' (line 1067)
        tuple_165462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1067)
        # Adding element type (line 1067)
        # Getting the type of 'self' (line 1067)
        self_165463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 16), 'self')
        # Obtaining the member 'model' of a type (line 1067)
        model_165464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 16), self_165463, 'model')
        # Obtaining the member 'fcn' of a type (line 1067)
        fcn_165465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 16), model_165464, 'fcn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1067, 16), tuple_165462, fcn_165465)
        # Adding element type (line 1067)
        # Getting the type of 'self' (line 1067)
        self_165466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 32), 'self')
        # Obtaining the member 'beta0' of a type (line 1067)
        beta0_165467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 32), self_165466, 'beta0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1067, 16), tuple_165462, beta0_165467)
        # Adding element type (line 1067)
        # Getting the type of 'self' (line 1067)
        self_165468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 44), 'self')
        # Obtaining the member 'data' of a type (line 1067)
        data_165469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 44), self_165468, 'data')
        # Obtaining the member 'y' of a type (line 1067)
        y_165470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 44), data_165469, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1067, 16), tuple_165462, y_165470)
        # Adding element type (line 1067)
        # Getting the type of 'self' (line 1067)
        self_165471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 57), 'self')
        # Obtaining the member 'data' of a type (line 1067)
        data_165472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 57), self_165471, 'data')
        # Obtaining the member 'x' of a type (line 1067)
        x_165473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 57), data_165472, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1067, 16), tuple_165462, x_165473)
        
        # Assigning a type to the variable 'args' (line 1067)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 8), 'args', tuple_165462)
        
        # Assigning a Dict to a Name (line 1068):
        
        # Assigning a Dict to a Name (line 1068):
        
        # Obtaining an instance of the builtin type 'dict' (line 1068)
        dict_165474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1068)
        # Adding element type (key, value) (line 1068)
        str_165475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 16), 'str', 'full_output')
        int_165476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 31), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1068, 15), dict_165474, (str_165475, int_165476))
        
        # Assigning a type to the variable 'kwds' (line 1068)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1068, 8), 'kwds', dict_165474)
        
        # Assigning a List to a Name (line 1069):
        
        # Assigning a List to a Name (line 1069):
        
        # Obtaining an instance of the builtin type 'list' (line 1069)
        list_165477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1069)
        # Adding element type (line 1069)
        str_165478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 17), 'str', 'ifixx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165478)
        # Adding element type (line 1069)
        str_165479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 26), 'str', 'ifixb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165479)
        # Adding element type (line 1069)
        str_165480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 35), 'str', 'job')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165480)
        # Adding element type (line 1069)
        str_165481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 42), 'str', 'iprint')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165481)
        # Adding element type (line 1069)
        str_165482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 52), 'str', 'errfile')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165482)
        # Adding element type (line 1069)
        str_165483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 63), 'str', 'rptfile')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165483)
        # Adding element type (line 1069)
        str_165484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 17), 'str', 'ndigit')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165484)
        # Adding element type (line 1069)
        str_165485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 27), 'str', 'taufac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165485)
        # Adding element type (line 1069)
        str_165486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 37), 'str', 'sstol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165486)
        # Adding element type (line 1069)
        str_165487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 46), 'str', 'partol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165487)
        # Adding element type (line 1069)
        str_165488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 56), 'str', 'maxit')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165488)
        # Adding element type (line 1069)
        str_165489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 65), 'str', 'stpb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165489)
        # Adding element type (line 1069)
        str_165490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 17), 'str', 'stpd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165490)
        # Adding element type (line 1069)
        str_165491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 25), 'str', 'sclb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165491)
        # Adding element type (line 1069)
        str_165492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 33), 'str', 'scld')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165492)
        # Adding element type (line 1069)
        str_165493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 41), 'str', 'work')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165493)
        # Adding element type (line 1069)
        str_165494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 49), 'str', 'iwork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), list_165477, str_165494)
        
        # Assigning a type to the variable 'kwd_l' (line 1069)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 8), 'kwd_l', list_165477)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 1073)
        self_165495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 11), 'self')
        # Obtaining the member 'delta0' of a type (line 1073)
        delta0_165496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 11), self_165495, 'delta0')
        # Getting the type of 'None' (line 1073)
        None_165497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 30), 'None')
        # Applying the binary operator 'isnot' (line 1073)
        result_is_not_165498 = python_operator(stypy.reporting.localization.Localization(__file__, 1073, 11), 'isnot', delta0_165496, None_165497)
        
        
        # Getting the type of 'self' (line 1073)
        self_165499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 39), 'self')
        # Obtaining the member 'job' of a type (line 1073)
        job_165500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 39), self_165499, 'job')
        int_165501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 50), 'int')
        # Applying the binary operator '%' (line 1073)
        result_mod_165502 = python_operator(stypy.reporting.localization.Localization(__file__, 1073, 39), '%', job_165500, int_165501)
        
        int_165503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 58), 'int')
        # Applying the binary operator '//' (line 1073)
        result_floordiv_165504 = python_operator(stypy.reporting.localization.Localization(__file__, 1073, 55), '//', result_mod_165502, int_165503)
        
        int_165505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 64), 'int')
        # Applying the binary operator '==' (line 1073)
        result_eq_165506 = python_operator(stypy.reporting.localization.Localization(__file__, 1073, 39), '==', result_floordiv_165504, int_165505)
        
        # Applying the binary operator 'and' (line 1073)
        result_and_keyword_165507 = python_operator(stypy.reporting.localization.Localization(__file__, 1073, 11), 'and', result_is_not_165498, result_eq_165506)
        
        # Testing the type of an if condition (line 1073)
        if_condition_165508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1073, 8), result_and_keyword_165507)
        # Assigning a type to the variable 'if_condition_165508' (line 1073)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'if_condition_165508', if_condition_165508)
        # SSA begins for if statement (line 1073)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _gen_work(...): (line 1075)
        # Processing the call keyword arguments (line 1075)
        kwargs_165511 = {}
        # Getting the type of 'self' (line 1075)
        self_165509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 12), 'self', False)
        # Obtaining the member '_gen_work' of a type (line 1075)
        _gen_work_165510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 12), self_165509, '_gen_work')
        # Calling _gen_work(args, kwargs) (line 1075)
        _gen_work_call_result_165512 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 12), _gen_work_165510, *[], **kwargs_165511)
        
        
        # Assigning a Call to a Name (line 1077):
        
        # Assigning a Call to a Name (line 1077):
        
        # Call to ravel(...): (line 1077)
        # Processing the call arguments (line 1077)
        # Getting the type of 'self' (line 1077)
        self_165515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 29), 'self', False)
        # Obtaining the member 'delta0' of a type (line 1077)
        delta0_165516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 29), self_165515, 'delta0')
        # Processing the call keyword arguments (line 1077)
        kwargs_165517 = {}
        # Getting the type of 'numpy' (line 1077)
        numpy_165513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 17), 'numpy', False)
        # Obtaining the member 'ravel' of a type (line 1077)
        ravel_165514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 17), numpy_165513, 'ravel')
        # Calling ravel(args, kwargs) (line 1077)
        ravel_call_result_165518 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 17), ravel_165514, *[delta0_165516], **kwargs_165517)
        
        # Assigning a type to the variable 'd0' (line 1077)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 12), 'd0', ravel_call_result_165518)
        
        # Assigning a Name to a Subscript (line 1079):
        
        # Assigning a Name to a Subscript (line 1079):
        # Getting the type of 'd0' (line 1079)
        d0_165519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 34), 'd0')
        # Getting the type of 'self' (line 1079)
        self_165520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 12), 'self')
        # Obtaining the member 'work' of a type (line 1079)
        work_165521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 12), self_165520, 'work')
        
        # Call to len(...): (line 1079)
        # Processing the call arguments (line 1079)
        # Getting the type of 'd0' (line 1079)
        d0_165523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 27), 'd0', False)
        # Processing the call keyword arguments (line 1079)
        kwargs_165524 = {}
        # Getting the type of 'len' (line 1079)
        len_165522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 23), 'len', False)
        # Calling len(args, kwargs) (line 1079)
        len_call_result_165525 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 23), len_165522, *[d0_165523], **kwargs_165524)
        
        slice_165526 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1079, 12), None, len_call_result_165525, None)
        # Storing an element on a container (line 1079)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1079, 12), work_165521, (slice_165526, d0_165519))
        # SSA join for if statement (line 1073)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 1082)
        self_165527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 11), 'self')
        # Obtaining the member 'model' of a type (line 1082)
        model_165528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1082, 11), self_165527, 'model')
        # Obtaining the member 'fjacb' of a type (line 1082)
        fjacb_165529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1082, 11), model_165528, 'fjacb')
        # Getting the type of 'None' (line 1082)
        None_165530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 35), 'None')
        # Applying the binary operator 'isnot' (line 1082)
        result_is_not_165531 = python_operator(stypy.reporting.localization.Localization(__file__, 1082, 11), 'isnot', fjacb_165529, None_165530)
        
        # Testing the type of an if condition (line 1082)
        if_condition_165532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1082, 8), result_is_not_165531)
        # Assigning a type to the variable 'if_condition_165532' (line 1082)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 8), 'if_condition_165532', if_condition_165532)
        # SSA begins for if statement (line 1082)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1083):
        
        # Assigning a Attribute to a Subscript (line 1083):
        # Getting the type of 'self' (line 1083)
        self_165533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 28), 'self')
        # Obtaining the member 'model' of a type (line 1083)
        model_165534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 28), self_165533, 'model')
        # Obtaining the member 'fjacb' of a type (line 1083)
        fjacb_165535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 28), model_165534, 'fjacb')
        # Getting the type of 'kwds' (line 1083)
        kwds_165536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 12), 'kwds')
        str_165537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 17), 'str', 'fjacb')
        # Storing an element on a container (line 1083)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1083, 12), kwds_165536, (str_165537, fjacb_165535))
        # SSA join for if statement (line 1082)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 1084)
        self_165538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 11), 'self')
        # Obtaining the member 'model' of a type (line 1084)
        model_165539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1084, 11), self_165538, 'model')
        # Obtaining the member 'fjacd' of a type (line 1084)
        fjacd_165540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1084, 11), model_165539, 'fjacd')
        # Getting the type of 'None' (line 1084)
        None_165541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 35), 'None')
        # Applying the binary operator 'isnot' (line 1084)
        result_is_not_165542 = python_operator(stypy.reporting.localization.Localization(__file__, 1084, 11), 'isnot', fjacd_165540, None_165541)
        
        # Testing the type of an if condition (line 1084)
        if_condition_165543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1084, 8), result_is_not_165542)
        # Assigning a type to the variable 'if_condition_165543' (line 1084)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1084, 8), 'if_condition_165543', if_condition_165543)
        # SSA begins for if statement (line 1084)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1085):
        
        # Assigning a Attribute to a Subscript (line 1085):
        # Getting the type of 'self' (line 1085)
        self_165544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 28), 'self')
        # Obtaining the member 'model' of a type (line 1085)
        model_165545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 28), self_165544, 'model')
        # Obtaining the member 'fjacd' of a type (line 1085)
        fjacd_165546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 28), model_165545, 'fjacd')
        # Getting the type of 'kwds' (line 1085)
        kwds_165547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 12), 'kwds')
        str_165548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 17), 'str', 'fjacd')
        # Storing an element on a container (line 1085)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1085, 12), kwds_165547, (str_165548, fjacd_165546))
        # SSA join for if statement (line 1084)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 1086)
        self_165549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 11), 'self')
        # Obtaining the member 'data' of a type (line 1086)
        data_165550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 11), self_165549, 'data')
        # Obtaining the member 'we' of a type (line 1086)
        we_165551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 11), data_165550, 'we')
        # Getting the type of 'None' (line 1086)
        None_165552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 31), 'None')
        # Applying the binary operator 'isnot' (line 1086)
        result_is_not_165553 = python_operator(stypy.reporting.localization.Localization(__file__, 1086, 11), 'isnot', we_165551, None_165552)
        
        # Testing the type of an if condition (line 1086)
        if_condition_165554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1086, 8), result_is_not_165553)
        # Assigning a type to the variable 'if_condition_165554' (line 1086)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1086, 8), 'if_condition_165554', if_condition_165554)
        # SSA begins for if statement (line 1086)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1087):
        
        # Assigning a Attribute to a Subscript (line 1087):
        # Getting the type of 'self' (line 1087)
        self_165555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 25), 'self')
        # Obtaining the member 'data' of a type (line 1087)
        data_165556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 25), self_165555, 'data')
        # Obtaining the member 'we' of a type (line 1087)
        we_165557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 25), data_165556, 'we')
        # Getting the type of 'kwds' (line 1087)
        kwds_165558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 12), 'kwds')
        str_165559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 17), 'str', 'we')
        # Storing an element on a container (line 1087)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1087, 12), kwds_165558, (str_165559, we_165557))
        # SSA join for if statement (line 1086)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 1088)
        self_165560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 11), 'self')
        # Obtaining the member 'data' of a type (line 1088)
        data_165561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1088, 11), self_165560, 'data')
        # Obtaining the member 'wd' of a type (line 1088)
        wd_165562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1088, 11), data_165561, 'wd')
        # Getting the type of 'None' (line 1088)
        None_165563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 31), 'None')
        # Applying the binary operator 'isnot' (line 1088)
        result_is_not_165564 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 11), 'isnot', wd_165562, None_165563)
        
        # Testing the type of an if condition (line 1088)
        if_condition_165565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1088, 8), result_is_not_165564)
        # Assigning a type to the variable 'if_condition_165565' (line 1088)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'if_condition_165565', if_condition_165565)
        # SSA begins for if statement (line 1088)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1089):
        
        # Assigning a Attribute to a Subscript (line 1089):
        # Getting the type of 'self' (line 1089)
        self_165566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 25), 'self')
        # Obtaining the member 'data' of a type (line 1089)
        data_165567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 25), self_165566, 'data')
        # Obtaining the member 'wd' of a type (line 1089)
        wd_165568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 25), data_165567, 'wd')
        # Getting the type of 'kwds' (line 1089)
        kwds_165569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 12), 'kwds')
        str_165570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 17), 'str', 'wd')
        # Storing an element on a container (line 1089)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1089, 12), kwds_165569, (str_165570, wd_165568))
        # SSA join for if statement (line 1088)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 1090)
        self_165571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 11), 'self')
        # Obtaining the member 'model' of a type (line 1090)
        model_165572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 11), self_165571, 'model')
        # Obtaining the member 'extra_args' of a type (line 1090)
        extra_args_165573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 11), model_165572, 'extra_args')
        # Getting the type of 'None' (line 1090)
        None_165574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 40), 'None')
        # Applying the binary operator 'isnot' (line 1090)
        result_is_not_165575 = python_operator(stypy.reporting.localization.Localization(__file__, 1090, 11), 'isnot', extra_args_165573, None_165574)
        
        # Testing the type of an if condition (line 1090)
        if_condition_165576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1090, 8), result_is_not_165575)
        # Assigning a type to the variable 'if_condition_165576' (line 1090)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1090, 8), 'if_condition_165576', if_condition_165576)
        # SSA begins for if statement (line 1090)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 1091):
        
        # Assigning a Attribute to a Subscript (line 1091):
        # Getting the type of 'self' (line 1091)
        self_165577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 33), 'self')
        # Obtaining the member 'model' of a type (line 1091)
        model_165578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 33), self_165577, 'model')
        # Obtaining the member 'extra_args' of a type (line 1091)
        extra_args_165579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 33), model_165578, 'extra_args')
        # Getting the type of 'kwds' (line 1091)
        kwds_165580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 12), 'kwds')
        str_165581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 17), 'str', 'extra_args')
        # Storing an element on a container (line 1091)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1091, 12), kwds_165580, (str_165581, extra_args_165579))
        # SSA join for if statement (line 1090)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'kwd_l' (line 1094)
        kwd_l_165582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 20), 'kwd_l')
        # Testing the type of a for loop iterable (line 1094)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1094, 8), kwd_l_165582)
        # Getting the type of the for loop variable (line 1094)
        for_loop_var_165583 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1094, 8), kwd_l_165582)
        # Assigning a type to the variable 'attr' (line 1094)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 8), 'attr', for_loop_var_165583)
        # SSA begins for a for statement (line 1094)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 1095):
        
        # Assigning a Call to a Name (line 1095):
        
        # Call to getattr(...): (line 1095)
        # Processing the call arguments (line 1095)
        # Getting the type of 'self' (line 1095)
        self_165585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 26), 'self', False)
        # Getting the type of 'attr' (line 1095)
        attr_165586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 32), 'attr', False)
        # Processing the call keyword arguments (line 1095)
        kwargs_165587 = {}
        # Getting the type of 'getattr' (line 1095)
        getattr_165584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1095)
        getattr_call_result_165588 = invoke(stypy.reporting.localization.Localization(__file__, 1095, 18), getattr_165584, *[self_165585, attr_165586], **kwargs_165587)
        
        # Assigning a type to the variable 'obj' (line 1095)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1095, 12), 'obj', getattr_call_result_165588)
        
        # Type idiom detected: calculating its left and rigth part (line 1096)
        # Getting the type of 'obj' (line 1096)
        obj_165589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 12), 'obj')
        # Getting the type of 'None' (line 1096)
        None_165590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 26), 'None')
        
        (may_be_165591, more_types_in_union_165592) = may_not_be_none(obj_165589, None_165590)

        if may_be_165591:

            if more_types_in_union_165592:
                # Runtime conditional SSA (line 1096)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1097):
            
            # Assigning a Name to a Subscript (line 1097):
            # Getting the type of 'obj' (line 1097)
            obj_165593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 29), 'obj')
            # Getting the type of 'kwds' (line 1097)
            kwds_165594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 16), 'kwds')
            # Getting the type of 'attr' (line 1097)
            attr_165595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 21), 'attr')
            # Storing an element on a container (line 1097)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1097, 16), kwds_165594, (attr_165595, obj_165593))

            if more_types_in_union_165592:
                # SSA join for if statement (line 1096)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 1099):
        
        # Assigning a Call to a Attribute (line 1099):
        
        # Call to Output(...): (line 1099)
        # Processing the call arguments (line 1099)
        
        # Call to odr(...): (line 1099)
        # Getting the type of 'args' (line 1099)
        args_165598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 34), 'args', False)
        # Processing the call keyword arguments (line 1099)
        # Getting the type of 'kwds' (line 1099)
        kwds_165599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 42), 'kwds', False)
        kwargs_165600 = {'kwds_165599': kwds_165599}
        # Getting the type of 'odr' (line 1099)
        odr_165597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 29), 'odr', False)
        # Calling odr(args, kwargs) (line 1099)
        odr_call_result_165601 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 29), odr_165597, *[args_165598], **kwargs_165600)
        
        # Processing the call keyword arguments (line 1099)
        kwargs_165602 = {}
        # Getting the type of 'Output' (line 1099)
        Output_165596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 22), 'Output', False)
        # Calling Output(args, kwargs) (line 1099)
        Output_call_result_165603 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 22), Output_165596, *[odr_call_result_165601], **kwargs_165602)
        
        # Getting the type of 'self' (line 1099)
        self_165604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 8), 'self')
        # Setting the type of the member 'output' of a type (line 1099)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 8), self_165604, 'output', Output_call_result_165603)
        # Getting the type of 'self' (line 1101)
        self_165605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 15), 'self')
        # Obtaining the member 'output' of a type (line 1101)
        output_165606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1101, 15), self_165605, 'output')
        # Assigning a type to the variable 'stypy_return_type' (line 1101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 8), 'stypy_return_type', output_165606)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 1058)
        stypy_return_type_165607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_165607


    @norecursion
    def restart(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1103)
        None_165608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 27), 'None')
        defaults = [None_165608]
        # Create a new context for function 'restart'
        module_type_store = module_type_store.open_function_context('restart', 1103, 4, False)
        # Assigning a type to the variable 'self' (line 1104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ODR.restart.__dict__.__setitem__('stypy_localization', localization)
        ODR.restart.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ODR.restart.__dict__.__setitem__('stypy_type_store', module_type_store)
        ODR.restart.__dict__.__setitem__('stypy_function_name', 'ODR.restart')
        ODR.restart.__dict__.__setitem__('stypy_param_names_list', ['iter'])
        ODR.restart.__dict__.__setitem__('stypy_varargs_param_name', None)
        ODR.restart.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ODR.restart.__dict__.__setitem__('stypy_call_defaults', defaults)
        ODR.restart.__dict__.__setitem__('stypy_call_varargs', varargs)
        ODR.restart.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ODR.restart.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODR.restart', ['iter'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'restart', localization, ['iter'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'restart(...)' code ##################

        str_165609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, (-1)), 'str', " Restarts the run with iter more iterations.\n\n        Parameters\n        ----------\n        iter : int, optional\n            ODRPACK's default for the number of new iterations is 10.\n\n        Returns\n        -------\n        output : Output instance\n            This object is also assigned to the attribute .output .\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 1117)
        # Getting the type of 'self' (line 1117)
        self_165610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 11), 'self')
        # Obtaining the member 'output' of a type (line 1117)
        output_165611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1117, 11), self_165610, 'output')
        # Getting the type of 'None' (line 1117)
        None_165612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 26), 'None')
        
        (may_be_165613, more_types_in_union_165614) = may_be_none(output_165611, None_165612)

        if may_be_165613:

            if more_types_in_union_165614:
                # Runtime conditional SSA (line 1117)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to OdrError(...): (line 1118)
            # Processing the call arguments (line 1118)
            str_165616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 27), 'str', 'cannot restart: run() has not been called before')
            # Processing the call keyword arguments (line 1118)
            kwargs_165617 = {}
            # Getting the type of 'OdrError' (line 1118)
            OdrError_165615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 18), 'OdrError', False)
            # Calling OdrError(args, kwargs) (line 1118)
            OdrError_call_result_165618 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 18), OdrError_165615, *[str_165616], **kwargs_165617)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1118, 12), OdrError_call_result_165618, 'raise parameter', BaseException)

            if more_types_in_union_165614:
                # SSA join for if statement (line 1117)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to set_job(...): (line 1120)
        # Processing the call keyword arguments (line 1120)
        int_165621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 29), 'int')
        keyword_165622 = int_165621
        kwargs_165623 = {'restart': keyword_165622}
        # Getting the type of 'self' (line 1120)
        self_165619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 8), 'self', False)
        # Obtaining the member 'set_job' of a type (line 1120)
        set_job_165620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1120, 8), self_165619, 'set_job')
        # Calling set_job(args, kwargs) (line 1120)
        set_job_call_result_165624 = invoke(stypy.reporting.localization.Localization(__file__, 1120, 8), set_job_165620, *[], **kwargs_165623)
        
        
        # Assigning a Attribute to a Attribute (line 1121):
        
        # Assigning a Attribute to a Attribute (line 1121):
        # Getting the type of 'self' (line 1121)
        self_165625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 20), 'self')
        # Obtaining the member 'output' of a type (line 1121)
        output_165626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 20), self_165625, 'output')
        # Obtaining the member 'work' of a type (line 1121)
        work_165627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 20), output_165626, 'work')
        # Getting the type of 'self' (line 1121)
        self_165628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 8), 'self')
        # Setting the type of the member 'work' of a type (line 1121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 8), self_165628, 'work', work_165627)
        
        # Assigning a Attribute to a Attribute (line 1122):
        
        # Assigning a Attribute to a Attribute (line 1122):
        # Getting the type of 'self' (line 1122)
        self_165629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 21), 'self')
        # Obtaining the member 'output' of a type (line 1122)
        output_165630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 21), self_165629, 'output')
        # Obtaining the member 'iwork' of a type (line 1122)
        iwork_165631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 21), output_165630, 'iwork')
        # Getting the type of 'self' (line 1122)
        self_165632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 8), 'self')
        # Setting the type of the member 'iwork' of a type (line 1122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 8), self_165632, 'iwork', iwork_165631)
        
        # Assigning a Name to a Attribute (line 1124):
        
        # Assigning a Name to a Attribute (line 1124):
        # Getting the type of 'iter' (line 1124)
        iter_165633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 21), 'iter')
        # Getting the type of 'self' (line 1124)
        self_165634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 8), 'self')
        # Setting the type of the member 'maxit' of a type (line 1124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 8), self_165634, 'maxit', iter_165633)
        
        # Call to run(...): (line 1126)
        # Processing the call keyword arguments (line 1126)
        kwargs_165637 = {}
        # Getting the type of 'self' (line 1126)
        self_165635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 15), 'self', False)
        # Obtaining the member 'run' of a type (line 1126)
        run_165636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 15), self_165635, 'run')
        # Calling run(args, kwargs) (line 1126)
        run_call_result_165638 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 15), run_165636, *[], **kwargs_165637)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 8), 'stypy_return_type', run_call_result_165638)
        
        # ################# End of 'restart(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'restart' in the type store
        # Getting the type of 'stypy_return_type' (line 1103)
        stypy_return_type_165639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165639)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'restart'
        return stypy_return_type_165639


# Assigning a type to the variable 'ODR' (line 618)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 0), 'ODR', ODR)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
