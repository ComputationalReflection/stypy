
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Author:  Travis Oliphant  2002-2011 with contributions from
3: #          SciPy Developers 2004-2011
4: #
5: from __future__ import division, print_function, absolute_import
6: 
7: from scipy import special
8: from scipy.special import entr, logsumexp, betaln, gammaln as gamln
9: from scipy._lib._numpy_compat import broadcast_to
10: 
11: from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh
12: 
13: import numpy as np
14: 
15: from ._distn_infrastructure import (
16:         rv_discrete, _lazywhere, _ncx2_pdf, _ncx2_cdf, get_distribution_names)
17: 
18: 
19: class binom_gen(rv_discrete):
20:     '''A binomial discrete random variable.
21: 
22:     %(before_notes)s
23: 
24:     Notes
25:     -----
26:     The probability mass function for `binom` is::
27: 
28:        binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)
29: 
30:     for ``k`` in ``{0, 1,..., n}``.
31: 
32:     `binom` takes ``n`` and ``p`` as shape parameters.
33: 
34:     %(after_notes)s
35: 
36:     %(example)s
37: 
38:     '''
39:     def _rvs(self, n, p):
40:         return self._random_state.binomial(n, p, self._size)
41: 
42:     def _argcheck(self, n, p):
43:         self.b = n
44:         return (n >= 0) & (p >= 0) & (p <= 1)
45: 
46:     def _logpmf(self, x, n, p):
47:         k = floor(x)
48:         combiln = (gamln(n+1) - (gamln(k+1) + gamln(n-k+1)))
49:         return combiln + special.xlogy(k, p) + special.xlog1py(n-k, -p)
50: 
51:     def _pmf(self, x, n, p):
52:         return exp(self._logpmf(x, n, p))
53: 
54:     def _cdf(self, x, n, p):
55:         k = floor(x)
56:         vals = special.bdtr(k, n, p)
57:         return vals
58: 
59:     def _sf(self, x, n, p):
60:         k = floor(x)
61:         return special.bdtrc(k, n, p)
62: 
63:     def _ppf(self, q, n, p):
64:         vals = ceil(special.bdtrik(q, n, p))
65:         vals1 = np.maximum(vals - 1, 0)
66:         temp = special.bdtr(vals1, n, p)
67:         return np.where(temp >= q, vals1, vals)
68: 
69:     def _stats(self, n, p, moments='mv'):
70:         q = 1.0 - p
71:         mu = n * p
72:         var = n * p * q
73:         g1, g2 = None, None
74:         if 's' in moments:
75:             g1 = (q - p) / sqrt(var)
76:         if 'k' in moments:
77:             g2 = (1.0 - 6*p*q) / var
78:         return mu, var, g1, g2
79: 
80:     def _entropy(self, n, p):
81:         k = np.r_[0:n + 1]
82:         vals = self._pmf(k, n, p)
83:         return np.sum(entr(vals), axis=0)
84: binom = binom_gen(name='binom')
85: 
86: 
87: class bernoulli_gen(binom_gen):
88:     '''A Bernoulli discrete random variable.
89: 
90:     %(before_notes)s
91: 
92:     Notes
93:     -----
94:     The probability mass function for `bernoulli` is::
95: 
96:        bernoulli.pmf(k) = 1-p  if k = 0
97:                         = p    if k = 1
98: 
99:     for ``k`` in ``{0, 1}``.
100: 
101:     `bernoulli` takes ``p`` as shape parameter.
102: 
103:     %(after_notes)s
104: 
105:     %(example)s
106: 
107:     '''
108:     def _rvs(self, p):
109:         return binom_gen._rvs(self, 1, p)
110: 
111:     def _argcheck(self, p):
112:         return (p >= 0) & (p <= 1)
113: 
114:     def _logpmf(self, x, p):
115:         return binom._logpmf(x, 1, p)
116: 
117:     def _pmf(self, x, p):
118:         return binom._pmf(x, 1, p)
119: 
120:     def _cdf(self, x, p):
121:         return binom._cdf(x, 1, p)
122: 
123:     def _sf(self, x, p):
124:         return binom._sf(x, 1, p)
125: 
126:     def _ppf(self, q, p):
127:         return binom._ppf(q, 1, p)
128: 
129:     def _stats(self, p):
130:         return binom._stats(1, p)
131: 
132:     def _entropy(self, p):
133:         return entr(p) + entr(1-p)
134: bernoulli = bernoulli_gen(b=1, name='bernoulli')
135: 
136: 
137: class nbinom_gen(rv_discrete):
138:     '''A negative binomial discrete random variable.
139: 
140:     %(before_notes)s
141: 
142:     Notes
143:     -----
144:     Negative binomial distribution describes a sequence of i.i.d. Bernoulli 
145:     trials, repeated until a predefined, non-random number of successes occurs.
146: 
147:     The probability mass function of the number of failures for `nbinom` is::
148: 
149:        nbinom.pmf(k) = choose(k+n-1, n-1) * p**n * (1-p)**k
150: 
151:     for ``k >= 0``.
152: 
153:     `nbinom` takes ``n`` and ``p`` as shape parameters where n is the number of
154:     successes, whereas p is the probability of a single success.
155: 
156:     %(after_notes)s
157: 
158:     %(example)s
159: 
160:     '''
161:     def _rvs(self, n, p):
162:         return self._random_state.negative_binomial(n, p, self._size)
163: 
164:     def _argcheck(self, n, p):
165:         return (n > 0) & (p >= 0) & (p <= 1)
166: 
167:     def _pmf(self, x, n, p):
168:         return exp(self._logpmf(x, n, p))
169: 
170:     def _logpmf(self, x, n, p):
171:         coeff = gamln(n+x) - gamln(x+1) - gamln(n)
172:         return coeff + n*log(p) + special.xlog1py(x, -p)
173: 
174:     def _cdf(self, x, n, p):
175:         k = floor(x)
176:         return special.betainc(n, k+1, p)
177: 
178:     def _sf_skip(self, x, n, p):
179:         # skip because special.nbdtrc doesn't work for 0<n<1
180:         k = floor(x)
181:         return special.nbdtrc(k, n, p)
182: 
183:     def _ppf(self, q, n, p):
184:         vals = ceil(special.nbdtrik(q, n, p))
185:         vals1 = (vals-1).clip(0.0, np.inf)
186:         temp = self._cdf(vals1, n, p)
187:         return np.where(temp >= q, vals1, vals)
188: 
189:     def _stats(self, n, p):
190:         Q = 1.0 / p
191:         P = Q - 1.0
192:         mu = n*P
193:         var = n*P*Q
194:         g1 = (Q+P)/sqrt(n*P*Q)
195:         g2 = (1.0 + 6*P*Q) / (n*P*Q)
196:         return mu, var, g1, g2
197: nbinom = nbinom_gen(name='nbinom')
198: 
199: 
200: class geom_gen(rv_discrete):
201:     '''A geometric discrete random variable.
202: 
203:     %(before_notes)s
204: 
205:     Notes
206:     -----
207:     The probability mass function for `geom` is::
208: 
209:         geom.pmf(k) = (1-p)**(k-1)*p
210: 
211:     for ``k >= 1``.
212: 
213:     `geom` takes ``p`` as shape parameter.
214: 
215:     %(after_notes)s
216: 
217:     %(example)s
218: 
219:     '''
220:     def _rvs(self, p):
221:         return self._random_state.geometric(p, size=self._size)
222: 
223:     def _argcheck(self, p):
224:         return (p <= 1) & (p >= 0)
225: 
226:     def _pmf(self, k, p):
227:         return np.power(1-p, k-1) * p
228: 
229:     def _logpmf(self, k, p):
230:         return special.xlog1py(k - 1, -p) + log(p)
231: 
232:     def _cdf(self, x, p):
233:         k = floor(x)
234:         return -expm1(log1p(-p)*k)
235: 
236:     def _sf(self, x, p):
237:         return np.exp(self._logsf(x, p))
238: 
239:     def _logsf(self, x, p):
240:         k = floor(x)
241:         return k*log1p(-p)
242: 
243:     def _ppf(self, q, p):
244:         vals = ceil(log(1.0-q)/log(1-p))
245:         temp = self._cdf(vals-1, p)
246:         return np.where((temp >= q) & (vals > 0), vals-1, vals)
247: 
248:     def _stats(self, p):
249:         mu = 1.0/p
250:         qr = 1.0-p
251:         var = qr / p / p
252:         g1 = (2.0-p) / sqrt(qr)
253:         g2 = np.polyval([1, -6, 6], p)/(1.0-p)
254:         return mu, var, g1, g2
255: geom = geom_gen(a=1, name='geom', longname="A geometric")
256: 
257: 
258: class hypergeom_gen(rv_discrete):
259:     r'''A hypergeometric discrete random variable.
260: 
261:     The hypergeometric distribution models drawing objects from a bin.
262:     `M` is the total number of objects, `n` is total number of Type I objects.
263:     The random variate represents the number of Type I objects in `N` drawn
264:     without replacement from the total population.
265: 
266:     %(before_notes)s
267: 
268:     Notes
269:     -----
270:     The symbols used to denote the shape parameters (`M`, `n`, and `N`) are not
271:     universally accepted.  See the Examples for a clarification of the
272:     definitions used here.
273: 
274:     The probability mass function is defined as,
275: 
276:     .. math:: p(k, M, n, N) = \frac{\binom{n}{k} \binom{M - n}{N - k}}{\binom{M}{N}}
277: 
278:     for :math:`k \in [\max(0, N - M + n), \min(n, N)]`, where the binomial
279:     coefficients are defined as,
280: 
281:     .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.
282: 
283:     %(after_notes)s
284: 
285:     Examples
286:     --------
287:     >>> from scipy.stats import hypergeom
288:     >>> import matplotlib.pyplot as plt
289: 
290:     Suppose we have a collection of 20 animals, of which 7 are dogs.  Then if
291:     we want to know the probability of finding a given number of dogs if we
292:     choose at random 12 of the 20 animals, we can initialize a frozen
293:     distribution and plot the probability mass function:
294: 
295:     >>> [M, n, N] = [20, 7, 12]
296:     >>> rv = hypergeom(M, n, N)
297:     >>> x = np.arange(0, n+1)
298:     >>> pmf_dogs = rv.pmf(x)
299: 
300:     >>> fig = plt.figure()
301:     >>> ax = fig.add_subplot(111)
302:     >>> ax.plot(x, pmf_dogs, 'bo')
303:     >>> ax.vlines(x, 0, pmf_dogs, lw=2)
304:     >>> ax.set_xlabel('# of dogs in our group of chosen animals')
305:     >>> ax.set_ylabel('hypergeom PMF')
306:     >>> plt.show()
307: 
308:     Instead of using a frozen distribution we can also use `hypergeom`
309:     methods directly.  To for example obtain the cumulative distribution
310:     function, use:
311: 
312:     >>> prb = hypergeom.cdf(x, M, n, N)
313: 
314:     And to generate random numbers:
315: 
316:     >>> R = hypergeom.rvs(M, n, N, size=10)
317: 
318:     '''
319:     def _rvs(self, M, n, N):
320:         return self._random_state.hypergeometric(n, M-n, N, size=self._size)
321: 
322:     def _argcheck(self, M, n, N):
323:         cond = (M > 0) & (n >= 0) & (N >= 0)
324:         cond &= (n <= M) & (N <= M)
325:         self.a = np.maximum(N-(M-n), 0)
326:         self.b = np.minimum(n, N)
327:         return cond
328: 
329:     def _logpmf(self, k, M, n, N):
330:         tot, good = M, n
331:         bad = tot - good
332:         return betaln(good+1, 1) + betaln(bad+1,1) + betaln(tot-N+1, N+1)\
333:             - betaln(k+1, good-k+1) - betaln(N-k+1,bad-N+k+1)\
334:             - betaln(tot+1, 1)
335: 
336:     def _pmf(self, k, M, n, N):
337:         # same as the following but numerically more precise
338:         # return comb(good, k) * comb(bad, N-k) / comb(tot, N)
339:         return exp(self._logpmf(k, M, n, N))
340: 
341:     def _stats(self, M, n, N):
342:         # tot, good, sample_size = M, n, N
343:         # "wikipedia".replace('N', 'M').replace('n', 'N').replace('K', 'n')
344:         M, n, N = 1.*M, 1.*n, 1.*N
345:         m = M - n
346:         p = n/M
347:         mu = N*p
348: 
349:         var = m*n*N*(M - N)*1.0/(M*M*(M-1))
350:         g1 = (m - n)*(M-2*N) / (M-2.0) * sqrt((M-1.0) / (m*n*N*(M-N)))
351: 
352:         g2 = M*(M+1) - 6.*N*(M-N) - 6.*n*m
353:         g2 *= (M-1)*M*M
354:         g2 += 6.*n*N*(M-N)*m*(5.*M-6)
355:         g2 /= n * N * (M-N) * m * (M-2.) * (M-3.)
356:         return mu, var, g1, g2
357: 
358:     def _entropy(self, M, n, N):
359:         k = np.r_[N - (M - n):min(n, N) + 1]
360:         vals = self.pmf(k, M, n, N)
361:         return np.sum(entr(vals), axis=0)
362: 
363:     def _sf(self, k, M, n, N):
364:         '''More precise calculation, 1 - cdf doesn't cut it.'''
365:         # This for loop is needed because `k` can be an array. If that's the
366:         # case, the sf() method makes M, n and N arrays of the same shape. We
367:         # therefore unpack all inputs args, so we can do the manual
368:         # integration.
369:         res = []
370:         for quant, tot, good, draw in zip(k, M, n, N):
371:             # Manual integration over probability mass function. More accurate
372:             # than integrate.quad.
373:             k2 = np.arange(quant + 1, draw + 1)
374:             res.append(np.sum(self._pmf(k2, tot, good, draw)))
375:         return np.asarray(res)
376:         
377:     def _logsf(self, k, M, n, N):
378:         '''
379:         More precise calculation than log(sf)
380:         '''
381:         res = []
382:         for quant, tot, good, draw in zip(k, M, n, N):
383:             # Integration over probability mass function using logsumexp
384:             k2 = np.arange(quant + 1, draw + 1)
385:             res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
386:         return np.asarray(res)
387: hypergeom = hypergeom_gen(name='hypergeom')
388: 
389: 
390: # FIXME: Fails _cdfvec
391: class logser_gen(rv_discrete):
392:     '''A Logarithmic (Log-Series, Series) discrete random variable.
393: 
394:     %(before_notes)s
395: 
396:     Notes
397:     -----
398:     The probability mass function for `logser` is::
399: 
400:         logser.pmf(k) = - p**k / (k*log(1-p))
401: 
402:     for ``k >= 1``.
403: 
404:     `logser` takes ``p`` as shape parameter.
405: 
406:     %(after_notes)s
407: 
408:     %(example)s
409: 
410:     '''
411:     def _rvs(self, p):
412:         # looks wrong for p>0.5, too few k=1
413:         # trying to use generic is worse, no k=1 at all
414:         return self._random_state.logseries(p, size=self._size)
415: 
416:     def _argcheck(self, p):
417:         return (p > 0) & (p < 1)
418: 
419:     def _pmf(self, k, p):
420:         return -np.power(p, k) * 1.0 / k / special.log1p(-p)
421: 
422:     def _stats(self, p):
423:         r = special.log1p(-p)
424:         mu = p / (p - 1.0) / r
425:         mu2p = -p / r / (p - 1.0)**2
426:         var = mu2p - mu*mu
427:         mu3p = -p / r * (1.0+p) / (1.0 - p)**3
428:         mu3 = mu3p - 3*mu*mu2p + 2*mu**3
429:         g1 = mu3 / np.power(var, 1.5)
430: 
431:         mu4p = -p / r * (
432:             1.0 / (p-1)**2 - 6*p / (p - 1)**3 + 6*p*p / (p-1)**4)
433:         mu4 = mu4p - 4*mu3p*mu + 6*mu2p*mu*mu - 3*mu**4
434:         g2 = mu4 / var**2 - 3.0
435:         return mu, var, g1, g2
436: logser = logser_gen(a=1, name='logser', longname='A logarithmic')
437: 
438: 
439: class poisson_gen(rv_discrete):
440:     '''A Poisson discrete random variable.
441: 
442:     %(before_notes)s
443: 
444:     Notes
445:     -----
446:     The probability mass function for `poisson` is::
447: 
448:         poisson.pmf(k) = exp(-mu) * mu**k / k!
449: 
450:     for ``k >= 0``.
451: 
452:     `poisson` takes ``mu`` as shape parameter.
453: 
454:     %(after_notes)s
455: 
456:     %(example)s
457: 
458:     '''
459: 
460:     # Override rv_discrete._argcheck to allow mu=0.
461:     def _argcheck(self, mu):
462:         return mu >= 0
463: 
464:     def _rvs(self, mu):
465:         return self._random_state.poisson(mu, self._size)
466: 
467:     def _logpmf(self, k, mu):
468:         Pk = special.xlogy(k, mu) - gamln(k + 1) - mu
469:         return Pk
470: 
471:     def _pmf(self, k, mu):
472:         return exp(self._logpmf(k, mu))
473: 
474:     def _cdf(self, x, mu):
475:         k = floor(x)
476:         return special.pdtr(k, mu)
477: 
478:     def _sf(self, x, mu):
479:         k = floor(x)
480:         return special.pdtrc(k, mu)
481: 
482:     def _ppf(self, q, mu):
483:         vals = ceil(special.pdtrik(q, mu))
484:         vals1 = np.maximum(vals - 1, 0)
485:         temp = special.pdtr(vals1, mu)
486:         return np.where(temp >= q, vals1, vals)
487: 
488:     def _stats(self, mu):
489:         var = mu
490:         tmp = np.asarray(mu)
491:         mu_nonzero = tmp > 0
492:         g1 = _lazywhere(mu_nonzero, (tmp,), lambda x: sqrt(1.0/x), np.inf)
493:         g2 = _lazywhere(mu_nonzero, (tmp,), lambda x: 1.0/x, np.inf)
494:         return mu, var, g1, g2
495: 
496: poisson = poisson_gen(name="poisson", longname='A Poisson')
497: 
498: 
499: class planck_gen(rv_discrete):
500:     '''A Planck discrete exponential random variable.
501: 
502:     %(before_notes)s
503: 
504:     Notes
505:     -----
506:     The probability mass function for `planck` is::
507: 
508:         planck.pmf(k) = (1-exp(-lambda_))*exp(-lambda_*k)
509: 
510:     for ``k*lambda_ >= 0``.
511: 
512:     `planck` takes ``lambda_`` as shape parameter.
513: 
514:     %(after_notes)s
515: 
516:     %(example)s
517: 
518:     '''
519:     def _argcheck(self, lambda_):
520:         self.a = np.where(lambda_ > 0, 0, -np.inf)
521:         self.b = np.where(lambda_ > 0, np.inf, 0)
522:         return lambda_ != 0
523: 
524:     def _pmf(self, k, lambda_):
525:         fact = (1-exp(-lambda_))
526:         return fact*exp(-lambda_*k)
527: 
528:     def _cdf(self, x, lambda_):
529:         k = floor(x)
530:         return 1-exp(-lambda_*(k+1))
531: 
532:     def _sf(self, x, lambda_):
533:         return np.exp(self._logsf(x, lambda_))
534: 
535:     def _logsf(self, x, lambda_):
536:         k = floor(x)
537:         return -lambda_*(k+1)
538: 
539:     def _ppf(self, q, lambda_):
540:         vals = ceil(-1.0/lambda_ * log1p(-q)-1)
541:         vals1 = (vals-1).clip(self.a, np.inf)
542:         temp = self._cdf(vals1, lambda_)
543:         return np.where(temp >= q, vals1, vals)
544: 
545:     def _stats(self, lambda_):
546:         mu = 1/(exp(lambda_)-1)
547:         var = exp(-lambda_)/(expm1(-lambda_))**2
548:         g1 = 2*cosh(lambda_/2.0)
549:         g2 = 4+2*cosh(lambda_)
550:         return mu, var, g1, g2
551: 
552:     def _entropy(self, lambda_):
553:         l = lambda_
554:         C = (1-exp(-l))
555:         return l*exp(-l)/C - log(C)
556: planck = planck_gen(name='planck', longname='A discrete exponential ')
557: 
558: 
559: class boltzmann_gen(rv_discrete):
560:     '''A Boltzmann (Truncated Discrete Exponential) random variable.
561: 
562:     %(before_notes)s
563: 
564:     Notes
565:     -----
566:     The probability mass function for `boltzmann` is::
567: 
568:         boltzmann.pmf(k) = (1-exp(-lambda_)*exp(-lambda_*k)/(1-exp(-lambda_*N))
569: 
570:     for ``k = 0,..., N-1``.
571: 
572:     `boltzmann` takes ``lambda_`` and ``N`` as shape parameters.
573: 
574:     %(after_notes)s
575: 
576:     %(example)s
577: 
578:     '''
579:     def _pmf(self, k, lambda_, N):
580:         fact = (1-exp(-lambda_))/(1-exp(-lambda_*N))
581:         return fact*exp(-lambda_*k)
582: 
583:     def _cdf(self, x, lambda_, N):
584:         k = floor(x)
585:         return (1-exp(-lambda_*(k+1)))/(1-exp(-lambda_*N))
586: 
587:     def _ppf(self, q, lambda_, N):
588:         qnew = q*(1-exp(-lambda_*N))
589:         vals = ceil(-1.0/lambda_ * log(1-qnew)-1)
590:         vals1 = (vals-1).clip(0.0, np.inf)
591:         temp = self._cdf(vals1, lambda_, N)
592:         return np.where(temp >= q, vals1, vals)
593: 
594:     def _stats(self, lambda_, N):
595:         z = exp(-lambda_)
596:         zN = exp(-lambda_*N)
597:         mu = z/(1.0-z)-N*zN/(1-zN)
598:         var = z/(1.0-z)**2 - N*N*zN/(1-zN)**2
599:         trm = (1-zN)/(1-z)
600:         trm2 = (z*trm**2 - N*N*zN)
601:         g1 = z*(1+z)*trm**3 - N**3*zN*(1+zN)
602:         g1 = g1 / trm2**(1.5)
603:         g2 = z*(1+4*z+z*z)*trm**4 - N**4 * zN*(1+4*zN+zN*zN)
604:         g2 = g2 / trm2 / trm2
605:         return mu, var, g1, g2
606: boltzmann = boltzmann_gen(name='boltzmann',
607:         longname='A truncated discrete exponential ')
608: 
609: 
610: class randint_gen(rv_discrete):
611:     '''A uniform discrete random variable.
612: 
613:     %(before_notes)s
614: 
615:     Notes
616:     -----
617:     The probability mass function for `randint` is::
618: 
619:         randint.pmf(k) = 1./(high - low)
620: 
621:     for ``k = low, ..., high - 1``.
622: 
623:     `randint` takes ``low`` and ``high`` as shape parameters.
624: 
625:     %(after_notes)s
626: 
627:     %(example)s
628: 
629:     '''
630:     def _argcheck(self, low, high):
631:         self.a = low
632:         self.b = high - 1
633:         return (high > low)
634: 
635:     def _pmf(self, k, low, high):
636:         p = np.ones_like(k) / (high - low)
637:         return np.where((k >= low) & (k < high), p, 0.)
638: 
639:     def _cdf(self, x, low, high):
640:         k = floor(x)
641:         return (k - low + 1.) / (high - low)
642: 
643:     def _ppf(self, q, low, high):
644:         vals = ceil(q * (high - low) + low) - 1
645:         vals1 = (vals - 1).clip(low, high)
646:         temp = self._cdf(vals1, low, high)
647:         return np.where(temp >= q, vals1, vals)
648: 
649:     def _stats(self, low, high):
650:         m2, m1 = np.asarray(high), np.asarray(low)
651:         mu = (m2 + m1 - 1.0) / 2
652:         d = m2 - m1
653:         var = (d*d - 1) / 12.0
654:         g1 = 0.0
655:         g2 = -6.0/5.0 * (d*d + 1.0) / (d*d - 1.0)
656:         return mu, var, g1, g2
657: 
658:     def _rvs(self, low, high):
659:         '''An array of *size* random integers >= ``low`` and < ``high``.'''
660:         if self._size is not None:
661:             # Numpy's RandomState.randint() doesn't broadcast its arguments.
662:             # Use `broadcast_to()` to extend the shapes of low and high
663:             # up to self._size.  Then we can use the numpy.vectorize'd
664:             # randint without needing to pass it a `size` argument.
665:             low = broadcast_to(low, self._size)
666:             high = broadcast_to(high, self._size)
667:         randint = np.vectorize(self._random_state.randint, otypes=[np.int_])
668:         return randint(low, high)
669: 
670:     def _entropy(self, low, high):
671:         return log(high - low)
672: 
673: randint = randint_gen(name='randint', longname='A discrete uniform '
674:                       '(random integer)')
675: 
676: 
677: # FIXME: problems sampling.
678: class zipf_gen(rv_discrete):
679:     '''A Zipf discrete random variable.
680: 
681:     %(before_notes)s
682: 
683:     Notes
684:     -----
685:     The probability mass function for `zipf` is::
686: 
687:         zipf.pmf(k, a) = 1/(zeta(a) * k**a)
688: 
689:     for ``k >= 1``.
690: 
691:     `zipf` takes ``a`` as shape parameter.
692: 
693:     %(after_notes)s
694: 
695:     %(example)s
696: 
697:     '''
698:     def _rvs(self, a):
699:         return self._random_state.zipf(a, size=self._size)
700: 
701:     def _argcheck(self, a):
702:         return a > 1
703: 
704:     def _pmf(self, k, a):
705:         Pk = 1.0 / special.zeta(a, 1) / k**a
706:         return Pk
707: 
708:     def _munp(self, n, a):
709:         return _lazywhere(
710:             a > n + 1, (a, n),
711:             lambda a, n: special.zeta(a - n, 1) / special.zeta(a, 1),
712:             np.inf)
713: zipf = zipf_gen(a=1, name='zipf', longname='A Zipf')
714: 
715: 
716: class dlaplace_gen(rv_discrete):
717:     '''A  Laplacian discrete random variable.
718: 
719:     %(before_notes)s
720: 
721:     Notes
722:     -----
723:     The probability mass function for `dlaplace` is::
724: 
725:         dlaplace.pmf(k) = tanh(a/2) * exp(-a*abs(k))
726: 
727:     for ``a > 0``.
728: 
729:     `dlaplace` takes ``a`` as shape parameter.
730: 
731:     %(after_notes)s
732: 
733:     %(example)s
734: 
735:     '''
736:     def _pmf(self, k, a):
737:         return tanh(a/2.0) * exp(-a * abs(k))
738: 
739:     def _cdf(self, x, a):
740:         k = floor(x)
741:         f = lambda k, a: 1.0 - exp(-a * k) / (exp(a) + 1)
742:         f2 = lambda k, a: exp(a * (k+1)) / (exp(a) + 1)
743:         return _lazywhere(k >= 0, (k, a), f=f, f2=f2)
744: 
745:     def _ppf(self, q, a):
746:         const = 1 + exp(a)
747:         vals = ceil(np.where(q < 1.0 / (1 + exp(-a)), log(q*const) / a - 1,
748:                                                       -log((1-q) * const) / a))
749:         vals1 = vals - 1
750:         return np.where(self._cdf(vals1, a) >= q, vals1, vals)
751: 
752:     def _stats(self, a):
753:         ea = exp(a)
754:         mu2 = 2.*ea/(ea-1.)**2
755:         mu4 = 2.*ea*(ea**2+10.*ea+1.) / (ea-1.)**4
756:         return 0., mu2, 0., mu4/mu2**2 - 3.
757: 
758:     def _entropy(self, a):
759:         return a / sinh(a) - log(tanh(a/2.0))
760: dlaplace = dlaplace_gen(a=-np.inf,
761:                         name='dlaplace', longname='A discrete Laplacian')
762: 
763: 
764: class skellam_gen(rv_discrete):
765:     '''A  Skellam discrete random variable.
766: 
767:     %(before_notes)s
768: 
769:     Notes
770:     -----
771:     Probability distribution of the difference of two correlated or
772:     uncorrelated Poisson random variables.
773: 
774:     Let k1 and k2 be two Poisson-distributed r.v. with expected values
775:     lam1 and lam2. Then, ``k1 - k2`` follows a Skellam distribution with
776:     parameters ``mu1 = lam1 - rho*sqrt(lam1*lam2)`` and
777:     ``mu2 = lam2 - rho*sqrt(lam1*lam2)``, where rho is the correlation
778:     coefficient between k1 and k2. If the two Poisson-distributed r.v.
779:     are independent then ``rho = 0``.
780: 
781:     Parameters mu1 and mu2 must be strictly positive.
782: 
783:     For details see: http://en.wikipedia.org/wiki/Skellam_distribution
784: 
785:     `skellam` takes ``mu1`` and ``mu2`` as shape parameters.
786: 
787:     %(after_notes)s
788: 
789:     %(example)s
790: 
791:     '''
792:     def _rvs(self, mu1, mu2):
793:         n = self._size
794:         return (self._random_state.poisson(mu1, n) -
795:                 self._random_state.poisson(mu2, n))
796: 
797:     def _pmf(self, x, mu1, mu2):
798:         px = np.where(x < 0,
799:                 _ncx2_pdf(2*mu2, 2*(1-x), 2*mu1)*2,
800:                 _ncx2_pdf(2*mu1, 2*(1+x), 2*mu2)*2)
801:         # ncx2.pdf() returns nan's for extremely low probabilities
802:         return px
803: 
804:     def _cdf(self, x, mu1, mu2):
805:         x = floor(x)
806:         px = np.where(x < 0,
807:                 _ncx2_cdf(2*mu2, -2*x, 2*mu1),
808:                 1-_ncx2_cdf(2*mu1, 2*(x+1), 2*mu2))
809:         return px
810: 
811:     def _stats(self, mu1, mu2):
812:         mean = mu1 - mu2
813:         var = mu1 + mu2
814:         g1 = mean / sqrt((var)**3)
815:         g2 = 1 / var
816:         return mean, var, g1, g2
817: skellam = skellam_gen(a=-np.inf, name="skellam", longname='A Skellam')
818: 
819: 
820: # Collect names of classes and objects in this module.
821: pairs = list(globals().items())
822: _distn_names, _distn_gen_names = get_distribution_names(pairs, rv_discrete)
823: 
824: __all__ = _distn_names + _distn_gen_names
825: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy import special' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_606758 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy')

if (type(import_606758) is not StypyTypeError):

    if (import_606758 != 'pyd_module'):
        __import__(import_606758)
        sys_modules_606759 = sys.modules[import_606758]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', sys_modules_606759.module_type_store, module_type_store, ['special'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_606759, sys_modules_606759.module_type_store, module_type_store)
    else:
        from scipy import special

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', None, module_type_store, ['special'], [special])

else:
    # Assigning a type to the variable 'scipy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', import_606758)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.special import entr, logsumexp, betaln, gamln' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_606760 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special')

if (type(import_606760) is not StypyTypeError):

    if (import_606760 != 'pyd_module'):
        __import__(import_606760)
        sys_modules_606761 = sys.modules[import_606760]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', sys_modules_606761.module_type_store, module_type_store, ['entr', 'logsumexp', 'betaln', 'gammaln'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_606761, sys_modules_606761.module_type_store, module_type_store)
    else:
        from scipy.special import entr, logsumexp, betaln, gammaln as gamln

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', None, module_type_store, ['entr', 'logsumexp', 'betaln', 'gammaln'], [entr, logsumexp, betaln, gamln])

else:
    # Assigning a type to the variable 'scipy.special' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', import_606760)

# Adding an alias
module_type_store.add_alias('gamln', 'gammaln')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy._lib._numpy_compat import broadcast_to' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_606762 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._numpy_compat')

if (type(import_606762) is not StypyTypeError):

    if (import_606762 != 'pyd_module'):
        __import__(import_606762)
        sys_modules_606763 = sys.modules[import_606762]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._numpy_compat', sys_modules_606763.module_type_store, module_type_store, ['broadcast_to'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_606763, sys_modules_606763.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import broadcast_to

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['broadcast_to'], [broadcast_to])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._numpy_compat', import_606762)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_606764 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_606764) is not StypyTypeError):

    if (import_606764 != 'pyd_module'):
        __import__(import_606764)
        sys_modules_606765 = sys.modules[import_606764]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', sys_modules_606765.module_type_store, module_type_store, ['floor', 'ceil', 'log', 'exp', 'sqrt', 'log1p', 'expm1', 'tanh', 'cosh', 'sinh'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_606765, sys_modules_606765.module_type_store, module_type_store)
    else:
        from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', None, module_type_store, ['floor', 'ceil', 'log', 'exp', 'sqrt', 'log1p', 'expm1', 'tanh', 'cosh', 'sinh'], [floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh])

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_606764)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_606766 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_606766) is not StypyTypeError):

    if (import_606766 != 'pyd_module'):
        __import__(import_606766)
        sys_modules_606767 = sys.modules[import_606766]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_606767.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_606766)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.stats._distn_infrastructure import rv_discrete, _lazywhere, _ncx2_pdf, _ncx2_cdf, get_distribution_names' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_606768 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.stats._distn_infrastructure')

if (type(import_606768) is not StypyTypeError):

    if (import_606768 != 'pyd_module'):
        __import__(import_606768)
        sys_modules_606769 = sys.modules[import_606768]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.stats._distn_infrastructure', sys_modules_606769.module_type_store, module_type_store, ['rv_discrete', '_lazywhere', '_ncx2_pdf', '_ncx2_cdf', 'get_distribution_names'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_606769, sys_modules_606769.module_type_store, module_type_store)
    else:
        from scipy.stats._distn_infrastructure import rv_discrete, _lazywhere, _ncx2_pdf, _ncx2_cdf, get_distribution_names

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.stats._distn_infrastructure', None, module_type_store, ['rv_discrete', '_lazywhere', '_ncx2_pdf', '_ncx2_cdf', 'get_distribution_names'], [rv_discrete, _lazywhere, _ncx2_pdf, _ncx2_cdf, get_distribution_names])

else:
    # Assigning a type to the variable 'scipy.stats._distn_infrastructure' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.stats._distn_infrastructure', import_606768)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

# Declaration of the 'binom_gen' class
# Getting the type of 'rv_discrete' (line 19)
rv_discrete_606770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'rv_discrete')

class binom_gen(rv_discrete_606770, ):
    str_606771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', 'A binomial discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `binom` is::\n\n       binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)\n\n    for ``k`` in ``{0, 1,..., n}``.\n\n    `binom` takes ``n`` and ``p`` as shape parameters.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._rvs.__dict__.__setitem__('stypy_function_name', 'binom_gen._rvs')
        binom_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['n', 'p'])
        binom_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._rvs', ['n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Call to binomial(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'n' (line 40)
        n_606775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'n', False)
        # Getting the type of 'p' (line 40)
        p_606776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 46), 'p', False)
        # Getting the type of 'self' (line 40)
        self_606777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 49), 'self', False)
        # Obtaining the member '_size' of a type (line 40)
        _size_606778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 49), self_606777, '_size')
        # Processing the call keyword arguments (line 40)
        kwargs_606779 = {}
        # Getting the type of 'self' (line 40)
        self_606772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'self', False)
        # Obtaining the member '_random_state' of a type (line 40)
        _random_state_606773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), self_606772, '_random_state')
        # Obtaining the member 'binomial' of a type (line 40)
        binomial_606774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), _random_state_606773, 'binomial')
        # Calling binomial(args, kwargs) (line 40)
        binomial_call_result_606780 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), binomial_606774, *[n_606775, p_606776, _size_606778], **kwargs_606779)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', binomial_call_result_606780)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_606781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_606781


    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'binom_gen._argcheck')
        binom_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['n', 'p'])
        binom_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._argcheck', ['n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Assigning a Name to a Attribute (line 43):
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'n' (line 43)
        n_606782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'n')
        # Getting the type of 'self' (line 43)
        self_606783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'b' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_606783, 'b', n_606782)
        
        # Getting the type of 'n' (line 44)
        n_606784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'n')
        int_606785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'int')
        # Applying the binary operator '>=' (line 44)
        result_ge_606786 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 16), '>=', n_606784, int_606785)
        
        
        # Getting the type of 'p' (line 44)
        p_606787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'p')
        int_606788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 32), 'int')
        # Applying the binary operator '>=' (line 44)
        result_ge_606789 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 27), '>=', p_606787, int_606788)
        
        # Applying the binary operator '&' (line 44)
        result_and__606790 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 15), '&', result_ge_606786, result_ge_606789)
        
        
        # Getting the type of 'p' (line 44)
        p_606791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 38), 'p')
        int_606792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 43), 'int')
        # Applying the binary operator '<=' (line 44)
        result_le_606793 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 38), '<=', p_606791, int_606792)
        
        # Applying the binary operator '&' (line 44)
        result_and__606794 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 35), '&', result_and__606790, result_le_606793)
        
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', result_and__606794)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_606795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_606795


    @norecursion
    def _logpmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logpmf'
        module_type_store = module_type_store.open_function_context('_logpmf', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._logpmf.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._logpmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._logpmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._logpmf.__dict__.__setitem__('stypy_function_name', 'binom_gen._logpmf')
        binom_gen._logpmf.__dict__.__setitem__('stypy_param_names_list', ['x', 'n', 'p'])
        binom_gen._logpmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._logpmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._logpmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._logpmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._logpmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._logpmf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._logpmf', ['x', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logpmf', localization, ['x', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logpmf(...)' code ##################

        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to floor(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'x' (line 47)
        x_606797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'x', False)
        # Processing the call keyword arguments (line 47)
        kwargs_606798 = {}
        # Getting the type of 'floor' (line 47)
        floor_606796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 47)
        floor_call_result_606799 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), floor_606796, *[x_606797], **kwargs_606798)
        
        # Assigning a type to the variable 'k' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'k', floor_call_result_606799)
        
        # Assigning a BinOp to a Name (line 48):
        
        # Assigning a BinOp to a Name (line 48):
        
        # Call to gamln(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'n' (line 48)
        n_606801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'n', False)
        int_606802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
        # Applying the binary operator '+' (line 48)
        result_add_606803 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 25), '+', n_606801, int_606802)
        
        # Processing the call keyword arguments (line 48)
        kwargs_606804 = {}
        # Getting the type of 'gamln' (line 48)
        gamln_606800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'gamln', False)
        # Calling gamln(args, kwargs) (line 48)
        gamln_call_result_606805 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), gamln_606800, *[result_add_606803], **kwargs_606804)
        
        
        # Call to gamln(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'k' (line 48)
        k_606807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 39), 'k', False)
        int_606808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 41), 'int')
        # Applying the binary operator '+' (line 48)
        result_add_606809 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 39), '+', k_606807, int_606808)
        
        # Processing the call keyword arguments (line 48)
        kwargs_606810 = {}
        # Getting the type of 'gamln' (line 48)
        gamln_606806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'gamln', False)
        # Calling gamln(args, kwargs) (line 48)
        gamln_call_result_606811 = invoke(stypy.reporting.localization.Localization(__file__, 48, 33), gamln_606806, *[result_add_606809], **kwargs_606810)
        
        
        # Call to gamln(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'n' (line 48)
        n_606813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 52), 'n', False)
        # Getting the type of 'k' (line 48)
        k_606814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 54), 'k', False)
        # Applying the binary operator '-' (line 48)
        result_sub_606815 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 52), '-', n_606813, k_606814)
        
        int_606816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 56), 'int')
        # Applying the binary operator '+' (line 48)
        result_add_606817 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 55), '+', result_sub_606815, int_606816)
        
        # Processing the call keyword arguments (line 48)
        kwargs_606818 = {}
        # Getting the type of 'gamln' (line 48)
        gamln_606812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 46), 'gamln', False)
        # Calling gamln(args, kwargs) (line 48)
        gamln_call_result_606819 = invoke(stypy.reporting.localization.Localization(__file__, 48, 46), gamln_606812, *[result_add_606817], **kwargs_606818)
        
        # Applying the binary operator '+' (line 48)
        result_add_606820 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 33), '+', gamln_call_result_606811, gamln_call_result_606819)
        
        # Applying the binary operator '-' (line 48)
        result_sub_606821 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), '-', gamln_call_result_606805, result_add_606820)
        
        # Assigning a type to the variable 'combiln' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'combiln', result_sub_606821)
        # Getting the type of 'combiln' (line 49)
        combiln_606822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'combiln')
        
        # Call to xlogy(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'k' (line 49)
        k_606825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 39), 'k', False)
        # Getting the type of 'p' (line 49)
        p_606826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 42), 'p', False)
        # Processing the call keyword arguments (line 49)
        kwargs_606827 = {}
        # Getting the type of 'special' (line 49)
        special_606823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'special', False)
        # Obtaining the member 'xlogy' of a type (line 49)
        xlogy_606824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), special_606823, 'xlogy')
        # Calling xlogy(args, kwargs) (line 49)
        xlogy_call_result_606828 = invoke(stypy.reporting.localization.Localization(__file__, 49, 25), xlogy_606824, *[k_606825, p_606826], **kwargs_606827)
        
        # Applying the binary operator '+' (line 49)
        result_add_606829 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), '+', combiln_606822, xlogy_call_result_606828)
        
        
        # Call to xlog1py(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'n' (line 49)
        n_606832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 63), 'n', False)
        # Getting the type of 'k' (line 49)
        k_606833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 65), 'k', False)
        # Applying the binary operator '-' (line 49)
        result_sub_606834 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 63), '-', n_606832, k_606833)
        
        
        # Getting the type of 'p' (line 49)
        p_606835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 69), 'p', False)
        # Applying the 'usub' unary operator (line 49)
        result___neg___606836 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 68), 'usub', p_606835)
        
        # Processing the call keyword arguments (line 49)
        kwargs_606837 = {}
        # Getting the type of 'special' (line 49)
        special_606830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 47), 'special', False)
        # Obtaining the member 'xlog1py' of a type (line 49)
        xlog1py_606831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 47), special_606830, 'xlog1py')
        # Calling xlog1py(args, kwargs) (line 49)
        xlog1py_call_result_606838 = invoke(stypy.reporting.localization.Localization(__file__, 49, 47), xlog1py_606831, *[result_sub_606834, result___neg___606836], **kwargs_606837)
        
        # Applying the binary operator '+' (line 49)
        result_add_606839 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 45), '+', result_add_606829, xlog1py_call_result_606838)
        
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', result_add_606839)
        
        # ################# End of '_logpmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logpmf' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_606840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logpmf'
        return stypy_return_type_606840


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._pmf.__dict__.__setitem__('stypy_function_name', 'binom_gen._pmf')
        binom_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['x', 'n', 'p'])
        binom_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._pmf', ['x', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['x', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Call to exp(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Call to _logpmf(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'x' (line 52)
        x_606844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 'x', False)
        # Getting the type of 'n' (line 52)
        n_606845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'n', False)
        # Getting the type of 'p' (line 52)
        p_606846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'p', False)
        # Processing the call keyword arguments (line 52)
        kwargs_606847 = {}
        # Getting the type of 'self' (line 52)
        self_606842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'self', False)
        # Obtaining the member '_logpmf' of a type (line 52)
        _logpmf_606843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), self_606842, '_logpmf')
        # Calling _logpmf(args, kwargs) (line 52)
        _logpmf_call_result_606848 = invoke(stypy.reporting.localization.Localization(__file__, 52, 19), _logpmf_606843, *[x_606844, n_606845, p_606846], **kwargs_606847)
        
        # Processing the call keyword arguments (line 52)
        kwargs_606849 = {}
        # Getting the type of 'exp' (line 52)
        exp_606841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'exp', False)
        # Calling exp(args, kwargs) (line 52)
        exp_call_result_606850 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), exp_606841, *[_logpmf_call_result_606848], **kwargs_606849)
        
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', exp_call_result_606850)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_606851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_606851


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._cdf.__dict__.__setitem__('stypy_function_name', 'binom_gen._cdf')
        binom_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'n', 'p'])
        binom_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._cdf', ['x', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to floor(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'x' (line 55)
        x_606853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'x', False)
        # Processing the call keyword arguments (line 55)
        kwargs_606854 = {}
        # Getting the type of 'floor' (line 55)
        floor_606852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 55)
        floor_call_result_606855 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), floor_606852, *[x_606853], **kwargs_606854)
        
        # Assigning a type to the variable 'k' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'k', floor_call_result_606855)
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to bdtr(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'k' (line 56)
        k_606858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'k', False)
        # Getting the type of 'n' (line 56)
        n_606859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 31), 'n', False)
        # Getting the type of 'p' (line 56)
        p_606860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'p', False)
        # Processing the call keyword arguments (line 56)
        kwargs_606861 = {}
        # Getting the type of 'special' (line 56)
        special_606856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'special', False)
        # Obtaining the member 'bdtr' of a type (line 56)
        bdtr_606857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 15), special_606856, 'bdtr')
        # Calling bdtr(args, kwargs) (line 56)
        bdtr_call_result_606862 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), bdtr_606857, *[k_606858, n_606859, p_606860], **kwargs_606861)
        
        # Assigning a type to the variable 'vals' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'vals', bdtr_call_result_606862)
        # Getting the type of 'vals' (line 57)
        vals_606863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'vals')
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', vals_606863)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_606864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606864)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_606864


    @norecursion
    def _sf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sf'
        module_type_store = module_type_store.open_function_context('_sf', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._sf.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._sf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._sf.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._sf.__dict__.__setitem__('stypy_function_name', 'binom_gen._sf')
        binom_gen._sf.__dict__.__setitem__('stypy_param_names_list', ['x', 'n', 'p'])
        binom_gen._sf.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._sf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._sf.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._sf.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._sf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._sf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._sf', ['x', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sf', localization, ['x', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sf(...)' code ##################

        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to floor(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'x' (line 60)
        x_606866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'x', False)
        # Processing the call keyword arguments (line 60)
        kwargs_606867 = {}
        # Getting the type of 'floor' (line 60)
        floor_606865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 60)
        floor_call_result_606868 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), floor_606865, *[x_606866], **kwargs_606867)
        
        # Assigning a type to the variable 'k' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'k', floor_call_result_606868)
        
        # Call to bdtrc(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'k' (line 61)
        k_606871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'k', False)
        # Getting the type of 'n' (line 61)
        n_606872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'n', False)
        # Getting the type of 'p' (line 61)
        p_606873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'p', False)
        # Processing the call keyword arguments (line 61)
        kwargs_606874 = {}
        # Getting the type of 'special' (line 61)
        special_606869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'special', False)
        # Obtaining the member 'bdtrc' of a type (line 61)
        bdtrc_606870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), special_606869, 'bdtrc')
        # Calling bdtrc(args, kwargs) (line 61)
        bdtrc_call_result_606875 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), bdtrc_606870, *[k_606871, n_606872, p_606873], **kwargs_606874)
        
        # Assigning a type to the variable 'stypy_return_type' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', bdtrc_call_result_606875)
        
        # ################# End of '_sf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sf' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_606876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606876)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sf'
        return stypy_return_type_606876


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._ppf.__dict__.__setitem__('stypy_function_name', 'binom_gen._ppf')
        binom_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'n', 'p'])
        binom_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._ppf', ['q', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to ceil(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to bdtrik(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'q' (line 64)
        q_606880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 'q', False)
        # Getting the type of 'n' (line 64)
        n_606881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 38), 'n', False)
        # Getting the type of 'p' (line 64)
        p_606882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'p', False)
        # Processing the call keyword arguments (line 64)
        kwargs_606883 = {}
        # Getting the type of 'special' (line 64)
        special_606878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'special', False)
        # Obtaining the member 'bdtrik' of a type (line 64)
        bdtrik_606879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 20), special_606878, 'bdtrik')
        # Calling bdtrik(args, kwargs) (line 64)
        bdtrik_call_result_606884 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), bdtrik_606879, *[q_606880, n_606881, p_606882], **kwargs_606883)
        
        # Processing the call keyword arguments (line 64)
        kwargs_606885 = {}
        # Getting the type of 'ceil' (line 64)
        ceil_606877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'ceil', False)
        # Calling ceil(args, kwargs) (line 64)
        ceil_call_result_606886 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), ceil_606877, *[bdtrik_call_result_606884], **kwargs_606885)
        
        # Assigning a type to the variable 'vals' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'vals', ceil_call_result_606886)
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to maximum(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'vals' (line 65)
        vals_606889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'vals', False)
        int_606890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'int')
        # Applying the binary operator '-' (line 65)
        result_sub_606891 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 27), '-', vals_606889, int_606890)
        
        int_606892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 37), 'int')
        # Processing the call keyword arguments (line 65)
        kwargs_606893 = {}
        # Getting the type of 'np' (line 65)
        np_606887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'np', False)
        # Obtaining the member 'maximum' of a type (line 65)
        maximum_606888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), np_606887, 'maximum')
        # Calling maximum(args, kwargs) (line 65)
        maximum_call_result_606894 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), maximum_606888, *[result_sub_606891, int_606892], **kwargs_606893)
        
        # Assigning a type to the variable 'vals1' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'vals1', maximum_call_result_606894)
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to bdtr(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'vals1' (line 66)
        vals1_606897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'vals1', False)
        # Getting the type of 'n' (line 66)
        n_606898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'n', False)
        # Getting the type of 'p' (line 66)
        p_606899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'p', False)
        # Processing the call keyword arguments (line 66)
        kwargs_606900 = {}
        # Getting the type of 'special' (line 66)
        special_606895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'special', False)
        # Obtaining the member 'bdtr' of a type (line 66)
        bdtr_606896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 15), special_606895, 'bdtr')
        # Calling bdtr(args, kwargs) (line 66)
        bdtr_call_result_606901 = invoke(stypy.reporting.localization.Localization(__file__, 66, 15), bdtr_606896, *[vals1_606897, n_606898, p_606899], **kwargs_606900)
        
        # Assigning a type to the variable 'temp' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'temp', bdtr_call_result_606901)
        
        # Call to where(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Getting the type of 'temp' (line 67)
        temp_606904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'temp', False)
        # Getting the type of 'q' (line 67)
        q_606905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'q', False)
        # Applying the binary operator '>=' (line 67)
        result_ge_606906 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 24), '>=', temp_606904, q_606905)
        
        # Getting the type of 'vals1' (line 67)
        vals1_606907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 35), 'vals1', False)
        # Getting the type of 'vals' (line 67)
        vals_606908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 42), 'vals', False)
        # Processing the call keyword arguments (line 67)
        kwargs_606909 = {}
        # Getting the type of 'np' (line 67)
        np_606902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 67)
        where_606903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 15), np_606902, 'where')
        # Calling where(args, kwargs) (line 67)
        where_call_result_606910 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), where_606903, *[result_ge_606906, vals1_606907, vals_606908], **kwargs_606909)
        
        # Assigning a type to the variable 'stypy_return_type' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', where_call_result_606910)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_606911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_606911


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_606912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 35), 'str', 'mv')
        defaults = [str_606912]
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._stats.__dict__.__setitem__('stypy_function_name', 'binom_gen._stats')
        binom_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['n', 'p', 'moments'])
        binom_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._stats', ['n', 'p', 'moments'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['n', 'p', 'moments'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a BinOp to a Name (line 70):
        
        # Assigning a BinOp to a Name (line 70):
        float_606913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'float')
        # Getting the type of 'p' (line 70)
        p_606914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'p')
        # Applying the binary operator '-' (line 70)
        result_sub_606915 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 12), '-', float_606913, p_606914)
        
        # Assigning a type to the variable 'q' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'q', result_sub_606915)
        
        # Assigning a BinOp to a Name (line 71):
        
        # Assigning a BinOp to a Name (line 71):
        # Getting the type of 'n' (line 71)
        n_606916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'n')
        # Getting the type of 'p' (line 71)
        p_606917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'p')
        # Applying the binary operator '*' (line 71)
        result_mul_606918 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 13), '*', n_606916, p_606917)
        
        # Assigning a type to the variable 'mu' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'mu', result_mul_606918)
        
        # Assigning a BinOp to a Name (line 72):
        
        # Assigning a BinOp to a Name (line 72):
        # Getting the type of 'n' (line 72)
        n_606919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'n')
        # Getting the type of 'p' (line 72)
        p_606920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'p')
        # Applying the binary operator '*' (line 72)
        result_mul_606921 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 14), '*', n_606919, p_606920)
        
        # Getting the type of 'q' (line 72)
        q_606922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'q')
        # Applying the binary operator '*' (line 72)
        result_mul_606923 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 20), '*', result_mul_606921, q_606922)
        
        # Assigning a type to the variable 'var' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'var', result_mul_606923)
        
        # Assigning a Tuple to a Tuple (line 73):
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'None' (line 73)
        None_606924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'None')
        # Assigning a type to the variable 'tuple_assignment_606747' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_assignment_606747', None_606924)
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'None' (line 73)
        None_606925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'None')
        # Assigning a type to the variable 'tuple_assignment_606748' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_assignment_606748', None_606925)
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'tuple_assignment_606747' (line 73)
        tuple_assignment_606747_606926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_assignment_606747')
        # Assigning a type to the variable 'g1' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'g1', tuple_assignment_606747_606926)
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'tuple_assignment_606748' (line 73)
        tuple_assignment_606748_606927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_assignment_606748')
        # Assigning a type to the variable 'g2' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'g2', tuple_assignment_606748_606927)
        
        
        str_606928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 11), 'str', 's')
        # Getting the type of 'moments' (line 74)
        moments_606929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'moments')
        # Applying the binary operator 'in' (line 74)
        result_contains_606930 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), 'in', str_606928, moments_606929)
        
        # Testing the type of an if condition (line 74)
        if_condition_606931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), result_contains_606930)
        # Assigning a type to the variable 'if_condition_606931' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'if_condition_606931', if_condition_606931)
        # SSA begins for if statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 75):
        
        # Assigning a BinOp to a Name (line 75):
        # Getting the type of 'q' (line 75)
        q_606932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'q')
        # Getting the type of 'p' (line 75)
        p_606933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'p')
        # Applying the binary operator '-' (line 75)
        result_sub_606934 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 18), '-', q_606932, p_606933)
        
        
        # Call to sqrt(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'var' (line 75)
        var_606936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'var', False)
        # Processing the call keyword arguments (line 75)
        kwargs_606937 = {}
        # Getting the type of 'sqrt' (line 75)
        sqrt_606935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 75)
        sqrt_call_result_606938 = invoke(stypy.reporting.localization.Localization(__file__, 75, 27), sqrt_606935, *[var_606936], **kwargs_606937)
        
        # Applying the binary operator 'div' (line 75)
        result_div_606939 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 17), 'div', result_sub_606934, sqrt_call_result_606938)
        
        # Assigning a type to the variable 'g1' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'g1', result_div_606939)
        # SSA join for if statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_606940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 11), 'str', 'k')
        # Getting the type of 'moments' (line 76)
        moments_606941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'moments')
        # Applying the binary operator 'in' (line 76)
        result_contains_606942 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), 'in', str_606940, moments_606941)
        
        # Testing the type of an if condition (line 76)
        if_condition_606943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_contains_606942)
        # Assigning a type to the variable 'if_condition_606943' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_606943', if_condition_606943)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 77):
        
        # Assigning a BinOp to a Name (line 77):
        float_606944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'float')
        int_606945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'int')
        # Getting the type of 'p' (line 77)
        p_606946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'p')
        # Applying the binary operator '*' (line 77)
        result_mul_606947 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 24), '*', int_606945, p_606946)
        
        # Getting the type of 'q' (line 77)
        q_606948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'q')
        # Applying the binary operator '*' (line 77)
        result_mul_606949 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 27), '*', result_mul_606947, q_606948)
        
        # Applying the binary operator '-' (line 77)
        result_sub_606950 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 18), '-', float_606944, result_mul_606949)
        
        # Getting the type of 'var' (line 77)
        var_606951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'var')
        # Applying the binary operator 'div' (line 77)
        result_div_606952 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 17), 'div', result_sub_606950, var_606951)
        
        # Assigning a type to the variable 'g2' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'g2', result_div_606952)
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_606953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        # Adding element type (line 78)
        # Getting the type of 'mu' (line 78)
        mu_606954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 15), tuple_606953, mu_606954)
        # Adding element type (line 78)
        # Getting the type of 'var' (line 78)
        var_606955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 15), tuple_606953, var_606955)
        # Adding element type (line 78)
        # Getting the type of 'g1' (line 78)
        g1_606956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 15), tuple_606953, g1_606956)
        # Adding element type (line 78)
        # Getting the type of 'g2' (line 78)
        g2_606957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 15), tuple_606953, g2_606957)
        
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', tuple_606953)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_606958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606958)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_606958


    @norecursion
    def _entropy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_entropy'
        module_type_store = module_type_store.open_function_context('_entropy', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        binom_gen._entropy.__dict__.__setitem__('stypy_localization', localization)
        binom_gen._entropy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        binom_gen._entropy.__dict__.__setitem__('stypy_type_store', module_type_store)
        binom_gen._entropy.__dict__.__setitem__('stypy_function_name', 'binom_gen._entropy')
        binom_gen._entropy.__dict__.__setitem__('stypy_param_names_list', ['n', 'p'])
        binom_gen._entropy.__dict__.__setitem__('stypy_varargs_param_name', None)
        binom_gen._entropy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        binom_gen._entropy.__dict__.__setitem__('stypy_call_defaults', defaults)
        binom_gen._entropy.__dict__.__setitem__('stypy_call_varargs', varargs)
        binom_gen._entropy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        binom_gen._entropy.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen._entropy', ['n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_entropy', localization, ['n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_entropy(...)' code ##################

        
        # Assigning a Subscript to a Name (line 81):
        
        # Assigning a Subscript to a Name (line 81):
        
        # Obtaining the type of the subscript
        int_606959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 18), 'int')
        # Getting the type of 'n' (line 81)
        n_606960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'n')
        int_606961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'int')
        # Applying the binary operator '+' (line 81)
        result_add_606962 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 20), '+', n_606960, int_606961)
        
        slice_606963 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 81, 12), int_606959, result_add_606962, None)
        # Getting the type of 'np' (line 81)
        np_606964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'np')
        # Obtaining the member 'r_' of a type (line 81)
        r__606965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), np_606964, 'r_')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___606966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), r__606965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_606967 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), getitem___606966, slice_606963)
        
        # Assigning a type to the variable 'k' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'k', subscript_call_result_606967)
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to _pmf(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'k' (line 82)
        k_606970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'k', False)
        # Getting the type of 'n' (line 82)
        n_606971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'n', False)
        # Getting the type of 'p' (line 82)
        p_606972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 31), 'p', False)
        # Processing the call keyword arguments (line 82)
        kwargs_606973 = {}
        # Getting the type of 'self' (line 82)
        self_606968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self', False)
        # Obtaining the member '_pmf' of a type (line 82)
        _pmf_606969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_606968, '_pmf')
        # Calling _pmf(args, kwargs) (line 82)
        _pmf_call_result_606974 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), _pmf_606969, *[k_606970, n_606971, p_606972], **kwargs_606973)
        
        # Assigning a type to the variable 'vals' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'vals', _pmf_call_result_606974)
        
        # Call to sum(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to entr(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'vals' (line 83)
        vals_606978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'vals', False)
        # Processing the call keyword arguments (line 83)
        kwargs_606979 = {}
        # Getting the type of 'entr' (line 83)
        entr_606977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'entr', False)
        # Calling entr(args, kwargs) (line 83)
        entr_call_result_606980 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), entr_606977, *[vals_606978], **kwargs_606979)
        
        # Processing the call keyword arguments (line 83)
        int_606981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 39), 'int')
        keyword_606982 = int_606981
        kwargs_606983 = {'axis': keyword_606982}
        # Getting the type of 'np' (line 83)
        np_606975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'np', False)
        # Obtaining the member 'sum' of a type (line 83)
        sum_606976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), np_606975, 'sum')
        # Calling sum(args, kwargs) (line 83)
        sum_call_result_606984 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), sum_606976, *[entr_call_result_606980], **kwargs_606983)
        
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', sum_call_result_606984)
        
        # ################# End of '_entropy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_entropy' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_606985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_606985)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_entropy'
        return stypy_return_type_606985


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'binom_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'binom_gen' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'binom_gen', binom_gen)

# Assigning a Call to a Name (line 84):

# Assigning a Call to a Name (line 84):

# Call to binom_gen(...): (line 84)
# Processing the call keyword arguments (line 84)
str_606987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'str', 'binom')
keyword_606988 = str_606987
kwargs_606989 = {'name': keyword_606988}
# Getting the type of 'binom_gen' (line 84)
binom_gen_606986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'binom_gen', False)
# Calling binom_gen(args, kwargs) (line 84)
binom_gen_call_result_606990 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), binom_gen_606986, *[], **kwargs_606989)

# Assigning a type to the variable 'binom' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'binom', binom_gen_call_result_606990)
# Declaration of the 'bernoulli_gen' class
# Getting the type of 'binom_gen' (line 87)
binom_gen_606991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'binom_gen')

class bernoulli_gen(binom_gen_606991, ):
    str_606992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, (-1)), 'str', 'A Bernoulli discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `bernoulli` is::\n\n       bernoulli.pmf(k) = 1-p  if k = 0\n                        = p    if k = 1\n\n    for ``k`` in ``{0, 1}``.\n\n    `bernoulli` takes ``p`` as shape parameter.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._rvs')
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['p'])
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._rvs', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Call to _rvs(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_606995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'self', False)
        int_606996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'int')
        # Getting the type of 'p' (line 109)
        p_606997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 39), 'p', False)
        # Processing the call keyword arguments (line 109)
        kwargs_606998 = {}
        # Getting the type of 'binom_gen' (line 109)
        binom_gen_606993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'binom_gen', False)
        # Obtaining the member '_rvs' of a type (line 109)
        _rvs_606994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), binom_gen_606993, '_rvs')
        # Calling _rvs(args, kwargs) (line 109)
        _rvs_call_result_606999 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), _rvs_606994, *[self_606995, int_606996, p_606997], **kwargs_606998)
        
        # Assigning a type to the variable 'stypy_return_type' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'stypy_return_type', _rvs_call_result_606999)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_607000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607000)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_607000


    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._argcheck')
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['p'])
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._argcheck', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Getting the type of 'p' (line 112)
        p_607001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'p')
        int_607002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 21), 'int')
        # Applying the binary operator '>=' (line 112)
        result_ge_607003 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 16), '>=', p_607001, int_607002)
        
        
        # Getting the type of 'p' (line 112)
        p_607004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'p')
        int_607005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'int')
        # Applying the binary operator '<=' (line 112)
        result_le_607006 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 27), '<=', p_607004, int_607005)
        
        # Applying the binary operator '&' (line 112)
        result_and__607007 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), '&', result_ge_607003, result_le_607006)
        
        # Assigning a type to the variable 'stypy_return_type' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'stypy_return_type', result_and__607007)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_607008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607008)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_607008


    @norecursion
    def _logpmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logpmf'
        module_type_store = module_type_store.open_function_context('_logpmf', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._logpmf')
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._logpmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._logpmf', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logpmf', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logpmf(...)' code ##################

        
        # Call to _logpmf(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'x' (line 115)
        x_607011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'x', False)
        int_607012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 32), 'int')
        # Getting the type of 'p' (line 115)
        p_607013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'p', False)
        # Processing the call keyword arguments (line 115)
        kwargs_607014 = {}
        # Getting the type of 'binom' (line 115)
        binom_607009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'binom', False)
        # Obtaining the member '_logpmf' of a type (line 115)
        _logpmf_607010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 15), binom_607009, '_logpmf')
        # Calling _logpmf(args, kwargs) (line 115)
        _logpmf_call_result_607015 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), _logpmf_607010, *[x_607011, int_607012, p_607013], **kwargs_607014)
        
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', _logpmf_call_result_607015)
        
        # ################# End of '_logpmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logpmf' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_607016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logpmf'
        return stypy_return_type_607016


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._pmf')
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._pmf', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Call to _pmf(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'x' (line 118)
        x_607019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'x', False)
        int_607020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 29), 'int')
        # Getting the type of 'p' (line 118)
        p_607021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 32), 'p', False)
        # Processing the call keyword arguments (line 118)
        kwargs_607022 = {}
        # Getting the type of 'binom' (line 118)
        binom_607017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'binom', False)
        # Obtaining the member '_pmf' of a type (line 118)
        _pmf_607018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), binom_607017, '_pmf')
        # Calling _pmf(args, kwargs) (line 118)
        _pmf_call_result_607023 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), _pmf_607018, *[x_607019, int_607020, p_607021], **kwargs_607022)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', _pmf_call_result_607023)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_607024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607024)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_607024


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._cdf')
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._cdf', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Call to _cdf(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'x' (line 121)
        x_607027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 26), 'x', False)
        int_607028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'int')
        # Getting the type of 'p' (line 121)
        p_607029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'p', False)
        # Processing the call keyword arguments (line 121)
        kwargs_607030 = {}
        # Getting the type of 'binom' (line 121)
        binom_607025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'binom', False)
        # Obtaining the member '_cdf' of a type (line 121)
        _cdf_607026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), binom_607025, '_cdf')
        # Calling _cdf(args, kwargs) (line 121)
        _cdf_call_result_607031 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), _cdf_607026, *[x_607027, int_607028, p_607029], **kwargs_607030)
        
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', _cdf_call_result_607031)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_607032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607032)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_607032


    @norecursion
    def _sf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sf'
        module_type_store = module_type_store.open_function_context('_sf', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._sf.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._sf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._sf.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._sf.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._sf')
        bernoulli_gen._sf.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        bernoulli_gen._sf.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._sf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._sf.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._sf.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._sf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._sf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._sf', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sf', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sf(...)' code ##################

        
        # Call to _sf(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'x' (line 124)
        x_607035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'x', False)
        int_607036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 28), 'int')
        # Getting the type of 'p' (line 124)
        p_607037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'p', False)
        # Processing the call keyword arguments (line 124)
        kwargs_607038 = {}
        # Getting the type of 'binom' (line 124)
        binom_607033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'binom', False)
        # Obtaining the member '_sf' of a type (line 124)
        _sf_607034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), binom_607033, '_sf')
        # Calling _sf(args, kwargs) (line 124)
        _sf_call_result_607039 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), _sf_607034, *[x_607035, int_607036, p_607037], **kwargs_607038)
        
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', _sf_call_result_607039)
        
        # ################# End of '_sf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sf' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_607040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607040)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sf'
        return stypy_return_type_607040


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._ppf')
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'p'])
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._ppf', ['q', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Call to _ppf(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'q' (line 127)
        q_607043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'q', False)
        int_607044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'int')
        # Getting the type of 'p' (line 127)
        p_607045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'p', False)
        # Processing the call keyword arguments (line 127)
        kwargs_607046 = {}
        # Getting the type of 'binom' (line 127)
        binom_607041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'binom', False)
        # Obtaining the member '_ppf' of a type (line 127)
        _ppf_607042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), binom_607041, '_ppf')
        # Calling _ppf(args, kwargs) (line 127)
        _ppf_call_result_607047 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), _ppf_607042, *[q_607043, int_607044, p_607045], **kwargs_607046)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', _ppf_call_result_607047)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_607048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607048)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_607048


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._stats.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._stats')
        bernoulli_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['p'])
        bernoulli_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._stats', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Call to _stats(...): (line 130)
        # Processing the call arguments (line 130)
        int_607051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 28), 'int')
        # Getting the type of 'p' (line 130)
        p_607052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), 'p', False)
        # Processing the call keyword arguments (line 130)
        kwargs_607053 = {}
        # Getting the type of 'binom' (line 130)
        binom_607049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'binom', False)
        # Obtaining the member '_stats' of a type (line 130)
        _stats_607050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 15), binom_607049, '_stats')
        # Calling _stats(args, kwargs) (line 130)
        _stats_call_result_607054 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), _stats_607050, *[int_607051, p_607052], **kwargs_607053)
        
        # Assigning a type to the variable 'stypy_return_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'stypy_return_type', _stats_call_result_607054)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_607055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607055)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_607055


    @norecursion
    def _entropy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_entropy'
        module_type_store = module_type_store.open_function_context('_entropy', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_localization', localization)
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_type_store', module_type_store)
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_function_name', 'bernoulli_gen._entropy')
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_param_names_list', ['p'])
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_varargs_param_name', None)
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_call_defaults', defaults)
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_call_varargs', varargs)
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bernoulli_gen._entropy.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen._entropy', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_entropy', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_entropy(...)' code ##################

        
        # Call to entr(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'p' (line 133)
        p_607057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'p', False)
        # Processing the call keyword arguments (line 133)
        kwargs_607058 = {}
        # Getting the type of 'entr' (line 133)
        entr_607056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'entr', False)
        # Calling entr(args, kwargs) (line 133)
        entr_call_result_607059 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), entr_607056, *[p_607057], **kwargs_607058)
        
        
        # Call to entr(...): (line 133)
        # Processing the call arguments (line 133)
        int_607061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 30), 'int')
        # Getting the type of 'p' (line 133)
        p_607062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 32), 'p', False)
        # Applying the binary operator '-' (line 133)
        result_sub_607063 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 30), '-', int_607061, p_607062)
        
        # Processing the call keyword arguments (line 133)
        kwargs_607064 = {}
        # Getting the type of 'entr' (line 133)
        entr_607060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'entr', False)
        # Calling entr(args, kwargs) (line 133)
        entr_call_result_607065 = invoke(stypy.reporting.localization.Localization(__file__, 133, 25), entr_607060, *[result_sub_607063], **kwargs_607064)
        
        # Applying the binary operator '+' (line 133)
        result_add_607066 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 15), '+', entr_call_result_607059, entr_call_result_607065)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', result_add_607066)
        
        # ################# End of '_entropy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_entropy' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_607067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607067)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_entropy'
        return stypy_return_type_607067


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 87, 0, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bernoulli_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'bernoulli_gen' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'bernoulli_gen', bernoulli_gen)

# Assigning a Call to a Name (line 134):

# Assigning a Call to a Name (line 134):

# Call to bernoulli_gen(...): (line 134)
# Processing the call keyword arguments (line 134)
int_607069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 28), 'int')
keyword_607070 = int_607069
str_607071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'str', 'bernoulli')
keyword_607072 = str_607071
kwargs_607073 = {'b': keyword_607070, 'name': keyword_607072}
# Getting the type of 'bernoulli_gen' (line 134)
bernoulli_gen_607068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'bernoulli_gen', False)
# Calling bernoulli_gen(args, kwargs) (line 134)
bernoulli_gen_call_result_607074 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), bernoulli_gen_607068, *[], **kwargs_607073)

# Assigning a type to the variable 'bernoulli' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'bernoulli', bernoulli_gen_call_result_607074)
# Declaration of the 'nbinom_gen' class
# Getting the type of 'rv_discrete' (line 137)
rv_discrete_607075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'rv_discrete')

class nbinom_gen(rv_discrete_607075, ):
    str_607076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, (-1)), 'str', 'A negative binomial discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    Negative binomial distribution describes a sequence of i.i.d. Bernoulli \n    trials, repeated until a predefined, non-random number of successes occurs.\n\n    The probability mass function of the number of failures for `nbinom` is::\n\n       nbinom.pmf(k) = choose(k+n-1, n-1) * p**n * (1-p)**k\n\n    for ``k >= 0``.\n\n    `nbinom` takes ``n`` and ``p`` as shape parameters where n is the number of\n    successes, whereas p is the probability of a single success.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nbinom_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        nbinom_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nbinom_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        nbinom_gen._rvs.__dict__.__setitem__('stypy_function_name', 'nbinom_gen._rvs')
        nbinom_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['n', 'p'])
        nbinom_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        nbinom_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nbinom_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        nbinom_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        nbinom_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nbinom_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen._rvs', ['n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Call to negative_binomial(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'n' (line 162)
        n_607080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 52), 'n', False)
        # Getting the type of 'p' (line 162)
        p_607081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 55), 'p', False)
        # Getting the type of 'self' (line 162)
        self_607082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 58), 'self', False)
        # Obtaining the member '_size' of a type (line 162)
        _size_607083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 58), self_607082, '_size')
        # Processing the call keyword arguments (line 162)
        kwargs_607084 = {}
        # Getting the type of 'self' (line 162)
        self_607077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'self', False)
        # Obtaining the member '_random_state' of a type (line 162)
        _random_state_607078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 15), self_607077, '_random_state')
        # Obtaining the member 'negative_binomial' of a type (line 162)
        negative_binomial_607079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 15), _random_state_607078, 'negative_binomial')
        # Calling negative_binomial(args, kwargs) (line 162)
        negative_binomial_call_result_607085 = invoke(stypy.reporting.localization.Localization(__file__, 162, 15), negative_binomial_607079, *[n_607080, p_607081, _size_607083], **kwargs_607084)
        
        # Assigning a type to the variable 'stypy_return_type' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type', negative_binomial_call_result_607085)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_607086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607086)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_607086


    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'nbinom_gen._argcheck')
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['n', 'p'])
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nbinom_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen._argcheck', ['n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Getting the type of 'n' (line 165)
        n_607087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'n')
        int_607088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'int')
        # Applying the binary operator '>' (line 165)
        result_gt_607089 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 16), '>', n_607087, int_607088)
        
        
        # Getting the type of 'p' (line 165)
        p_607090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'p')
        int_607091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 31), 'int')
        # Applying the binary operator '>=' (line 165)
        result_ge_607092 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 26), '>=', p_607090, int_607091)
        
        # Applying the binary operator '&' (line 165)
        result_and__607093 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 15), '&', result_gt_607089, result_ge_607092)
        
        
        # Getting the type of 'p' (line 165)
        p_607094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'p')
        int_607095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 42), 'int')
        # Applying the binary operator '<=' (line 165)
        result_le_607096 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 37), '<=', p_607094, int_607095)
        
        # Applying the binary operator '&' (line 165)
        result_and__607097 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 34), '&', result_and__607093, result_le_607096)
        
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'stypy_return_type', result_and__607097)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_607098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607098)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_607098


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nbinom_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        nbinom_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nbinom_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        nbinom_gen._pmf.__dict__.__setitem__('stypy_function_name', 'nbinom_gen._pmf')
        nbinom_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['x', 'n', 'p'])
        nbinom_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        nbinom_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nbinom_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        nbinom_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        nbinom_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nbinom_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen._pmf', ['x', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['x', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Call to exp(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to _logpmf(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'x' (line 168)
        x_607102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'x', False)
        # Getting the type of 'n' (line 168)
        n_607103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'n', False)
        # Getting the type of 'p' (line 168)
        p_607104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 'p', False)
        # Processing the call keyword arguments (line 168)
        kwargs_607105 = {}
        # Getting the type of 'self' (line 168)
        self_607100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'self', False)
        # Obtaining the member '_logpmf' of a type (line 168)
        _logpmf_607101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 19), self_607100, '_logpmf')
        # Calling _logpmf(args, kwargs) (line 168)
        _logpmf_call_result_607106 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), _logpmf_607101, *[x_607102, n_607103, p_607104], **kwargs_607105)
        
        # Processing the call keyword arguments (line 168)
        kwargs_607107 = {}
        # Getting the type of 'exp' (line 168)
        exp_607099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'exp', False)
        # Calling exp(args, kwargs) (line 168)
        exp_call_result_607108 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), exp_607099, *[_logpmf_call_result_607106], **kwargs_607107)
        
        # Assigning a type to the variable 'stypy_return_type' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'stypy_return_type', exp_call_result_607108)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_607109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607109)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_607109


    @norecursion
    def _logpmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logpmf'
        module_type_store = module_type_store.open_function_context('_logpmf', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_localization', localization)
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_function_name', 'nbinom_gen._logpmf')
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_param_names_list', ['x', 'n', 'p'])
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nbinom_gen._logpmf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen._logpmf', ['x', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logpmf', localization, ['x', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logpmf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 171):
        
        # Assigning a BinOp to a Name (line 171):
        
        # Call to gamln(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'n' (line 171)
        n_607111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'n', False)
        # Getting the type of 'x' (line 171)
        x_607112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'x', False)
        # Applying the binary operator '+' (line 171)
        result_add_607113 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 22), '+', n_607111, x_607112)
        
        # Processing the call keyword arguments (line 171)
        kwargs_607114 = {}
        # Getting the type of 'gamln' (line 171)
        gamln_607110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'gamln', False)
        # Calling gamln(args, kwargs) (line 171)
        gamln_call_result_607115 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), gamln_607110, *[result_add_607113], **kwargs_607114)
        
        
        # Call to gamln(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'x' (line 171)
        x_607117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'x', False)
        int_607118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 37), 'int')
        # Applying the binary operator '+' (line 171)
        result_add_607119 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 35), '+', x_607117, int_607118)
        
        # Processing the call keyword arguments (line 171)
        kwargs_607120 = {}
        # Getting the type of 'gamln' (line 171)
        gamln_607116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'gamln', False)
        # Calling gamln(args, kwargs) (line 171)
        gamln_call_result_607121 = invoke(stypy.reporting.localization.Localization(__file__, 171, 29), gamln_607116, *[result_add_607119], **kwargs_607120)
        
        # Applying the binary operator '-' (line 171)
        result_sub_607122 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 16), '-', gamln_call_result_607115, gamln_call_result_607121)
        
        
        # Call to gamln(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'n' (line 171)
        n_607124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), 'n', False)
        # Processing the call keyword arguments (line 171)
        kwargs_607125 = {}
        # Getting the type of 'gamln' (line 171)
        gamln_607123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'gamln', False)
        # Calling gamln(args, kwargs) (line 171)
        gamln_call_result_607126 = invoke(stypy.reporting.localization.Localization(__file__, 171, 42), gamln_607123, *[n_607124], **kwargs_607125)
        
        # Applying the binary operator '-' (line 171)
        result_sub_607127 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 40), '-', result_sub_607122, gamln_call_result_607126)
        
        # Assigning a type to the variable 'coeff' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'coeff', result_sub_607127)
        # Getting the type of 'coeff' (line 172)
        coeff_607128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'coeff')
        # Getting the type of 'n' (line 172)
        n_607129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'n')
        
        # Call to log(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'p' (line 172)
        p_607131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 29), 'p', False)
        # Processing the call keyword arguments (line 172)
        kwargs_607132 = {}
        # Getting the type of 'log' (line 172)
        log_607130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'log', False)
        # Calling log(args, kwargs) (line 172)
        log_call_result_607133 = invoke(stypy.reporting.localization.Localization(__file__, 172, 25), log_607130, *[p_607131], **kwargs_607132)
        
        # Applying the binary operator '*' (line 172)
        result_mul_607134 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 23), '*', n_607129, log_call_result_607133)
        
        # Applying the binary operator '+' (line 172)
        result_add_607135 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), '+', coeff_607128, result_mul_607134)
        
        
        # Call to xlog1py(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'x' (line 172)
        x_607138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 50), 'x', False)
        
        # Getting the type of 'p' (line 172)
        p_607139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 54), 'p', False)
        # Applying the 'usub' unary operator (line 172)
        result___neg___607140 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 53), 'usub', p_607139)
        
        # Processing the call keyword arguments (line 172)
        kwargs_607141 = {}
        # Getting the type of 'special' (line 172)
        special_607136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'special', False)
        # Obtaining the member 'xlog1py' of a type (line 172)
        xlog1py_607137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 34), special_607136, 'xlog1py')
        # Calling xlog1py(args, kwargs) (line 172)
        xlog1py_call_result_607142 = invoke(stypy.reporting.localization.Localization(__file__, 172, 34), xlog1py_607137, *[x_607138, result___neg___607140], **kwargs_607141)
        
        # Applying the binary operator '+' (line 172)
        result_add_607143 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 32), '+', result_add_607135, xlog1py_call_result_607142)
        
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', result_add_607143)
        
        # ################# End of '_logpmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logpmf' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_607144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logpmf'
        return stypy_return_type_607144


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nbinom_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        nbinom_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nbinom_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        nbinom_gen._cdf.__dict__.__setitem__('stypy_function_name', 'nbinom_gen._cdf')
        nbinom_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'n', 'p'])
        nbinom_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        nbinom_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nbinom_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        nbinom_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        nbinom_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nbinom_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen._cdf', ['x', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to floor(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'x' (line 175)
        x_607146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'x', False)
        # Processing the call keyword arguments (line 175)
        kwargs_607147 = {}
        # Getting the type of 'floor' (line 175)
        floor_607145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 175)
        floor_call_result_607148 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), floor_607145, *[x_607146], **kwargs_607147)
        
        # Assigning a type to the variable 'k' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'k', floor_call_result_607148)
        
        # Call to betainc(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'n' (line 176)
        n_607151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 31), 'n', False)
        # Getting the type of 'k' (line 176)
        k_607152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 34), 'k', False)
        int_607153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 36), 'int')
        # Applying the binary operator '+' (line 176)
        result_add_607154 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 34), '+', k_607152, int_607153)
        
        # Getting the type of 'p' (line 176)
        p_607155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 39), 'p', False)
        # Processing the call keyword arguments (line 176)
        kwargs_607156 = {}
        # Getting the type of 'special' (line 176)
        special_607149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'special', False)
        # Obtaining the member 'betainc' of a type (line 176)
        betainc_607150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), special_607149, 'betainc')
        # Calling betainc(args, kwargs) (line 176)
        betainc_call_result_607157 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), betainc_607150, *[n_607151, result_add_607154, p_607155], **kwargs_607156)
        
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', betainc_call_result_607157)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_607158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_607158


    @norecursion
    def _sf_skip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sf_skip'
        module_type_store = module_type_store.open_function_context('_sf_skip', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_localization', localization)
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_type_store', module_type_store)
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_function_name', 'nbinom_gen._sf_skip')
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_param_names_list', ['x', 'n', 'p'])
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_varargs_param_name', None)
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_call_defaults', defaults)
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_call_varargs', varargs)
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nbinom_gen._sf_skip.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen._sf_skip', ['x', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sf_skip', localization, ['x', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sf_skip(...)' code ##################

        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to floor(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'x' (line 180)
        x_607160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 18), 'x', False)
        # Processing the call keyword arguments (line 180)
        kwargs_607161 = {}
        # Getting the type of 'floor' (line 180)
        floor_607159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 180)
        floor_call_result_607162 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), floor_607159, *[x_607160], **kwargs_607161)
        
        # Assigning a type to the variable 'k' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'k', floor_call_result_607162)
        
        # Call to nbdtrc(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'k' (line 181)
        k_607165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'k', False)
        # Getting the type of 'n' (line 181)
        n_607166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'n', False)
        # Getting the type of 'p' (line 181)
        p_607167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'p', False)
        # Processing the call keyword arguments (line 181)
        kwargs_607168 = {}
        # Getting the type of 'special' (line 181)
        special_607163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'special', False)
        # Obtaining the member 'nbdtrc' of a type (line 181)
        nbdtrc_607164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 15), special_607163, 'nbdtrc')
        # Calling nbdtrc(args, kwargs) (line 181)
        nbdtrc_call_result_607169 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), nbdtrc_607164, *[k_607165, n_607166, p_607167], **kwargs_607168)
        
        # Assigning a type to the variable 'stypy_return_type' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', nbdtrc_call_result_607169)
        
        # ################# End of '_sf_skip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sf_skip' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_607170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607170)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sf_skip'
        return stypy_return_type_607170


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nbinom_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        nbinom_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nbinom_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        nbinom_gen._ppf.__dict__.__setitem__('stypy_function_name', 'nbinom_gen._ppf')
        nbinom_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'n', 'p'])
        nbinom_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        nbinom_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nbinom_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        nbinom_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        nbinom_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nbinom_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen._ppf', ['q', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Assigning a Call to a Name (line 184):
        
        # Assigning a Call to a Name (line 184):
        
        # Call to ceil(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Call to nbdtrik(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'q' (line 184)
        q_607174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'q', False)
        # Getting the type of 'n' (line 184)
        n_607175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 39), 'n', False)
        # Getting the type of 'p' (line 184)
        p_607176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 42), 'p', False)
        # Processing the call keyword arguments (line 184)
        kwargs_607177 = {}
        # Getting the type of 'special' (line 184)
        special_607172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'special', False)
        # Obtaining the member 'nbdtrik' of a type (line 184)
        nbdtrik_607173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 20), special_607172, 'nbdtrik')
        # Calling nbdtrik(args, kwargs) (line 184)
        nbdtrik_call_result_607178 = invoke(stypy.reporting.localization.Localization(__file__, 184, 20), nbdtrik_607173, *[q_607174, n_607175, p_607176], **kwargs_607177)
        
        # Processing the call keyword arguments (line 184)
        kwargs_607179 = {}
        # Getting the type of 'ceil' (line 184)
        ceil_607171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'ceil', False)
        # Calling ceil(args, kwargs) (line 184)
        ceil_call_result_607180 = invoke(stypy.reporting.localization.Localization(__file__, 184, 15), ceil_607171, *[nbdtrik_call_result_607178], **kwargs_607179)
        
        # Assigning a type to the variable 'vals' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'vals', ceil_call_result_607180)
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to clip(...): (line 185)
        # Processing the call arguments (line 185)
        float_607185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'float')
        # Getting the type of 'np' (line 185)
        np_607186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 35), 'np', False)
        # Obtaining the member 'inf' of a type (line 185)
        inf_607187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 35), np_607186, 'inf')
        # Processing the call keyword arguments (line 185)
        kwargs_607188 = {}
        # Getting the type of 'vals' (line 185)
        vals_607181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'vals', False)
        int_607182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 22), 'int')
        # Applying the binary operator '-' (line 185)
        result_sub_607183 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 17), '-', vals_607181, int_607182)
        
        # Obtaining the member 'clip' of a type (line 185)
        clip_607184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 17), result_sub_607183, 'clip')
        # Calling clip(args, kwargs) (line 185)
        clip_call_result_607189 = invoke(stypy.reporting.localization.Localization(__file__, 185, 17), clip_607184, *[float_607185, inf_607187], **kwargs_607188)
        
        # Assigning a type to the variable 'vals1' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'vals1', clip_call_result_607189)
        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to _cdf(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'vals1' (line 186)
        vals1_607192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'vals1', False)
        # Getting the type of 'n' (line 186)
        n_607193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 32), 'n', False)
        # Getting the type of 'p' (line 186)
        p_607194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 35), 'p', False)
        # Processing the call keyword arguments (line 186)
        kwargs_607195 = {}
        # Getting the type of 'self' (line 186)
        self_607190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'self', False)
        # Obtaining the member '_cdf' of a type (line 186)
        _cdf_607191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 15), self_607190, '_cdf')
        # Calling _cdf(args, kwargs) (line 186)
        _cdf_call_result_607196 = invoke(stypy.reporting.localization.Localization(__file__, 186, 15), _cdf_607191, *[vals1_607192, n_607193, p_607194], **kwargs_607195)
        
        # Assigning a type to the variable 'temp' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'temp', _cdf_call_result_607196)
        
        # Call to where(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Getting the type of 'temp' (line 187)
        temp_607199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'temp', False)
        # Getting the type of 'q' (line 187)
        q_607200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 32), 'q', False)
        # Applying the binary operator '>=' (line 187)
        result_ge_607201 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 24), '>=', temp_607199, q_607200)
        
        # Getting the type of 'vals1' (line 187)
        vals1_607202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 35), 'vals1', False)
        # Getting the type of 'vals' (line 187)
        vals_607203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 42), 'vals', False)
        # Processing the call keyword arguments (line 187)
        kwargs_607204 = {}
        # Getting the type of 'np' (line 187)
        np_607197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 187)
        where_607198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), np_607197, 'where')
        # Calling where(args, kwargs) (line 187)
        where_call_result_607205 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), where_607198, *[result_ge_607201, vals1_607202, vals_607203], **kwargs_607204)
        
        # Assigning a type to the variable 'stypy_return_type' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', where_call_result_607205)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_607206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607206)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_607206


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nbinom_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        nbinom_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nbinom_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        nbinom_gen._stats.__dict__.__setitem__('stypy_function_name', 'nbinom_gen._stats')
        nbinom_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['n', 'p'])
        nbinom_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        nbinom_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nbinom_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        nbinom_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        nbinom_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nbinom_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen._stats', ['n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a BinOp to a Name (line 190):
        
        # Assigning a BinOp to a Name (line 190):
        float_607207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 12), 'float')
        # Getting the type of 'p' (line 190)
        p_607208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'p')
        # Applying the binary operator 'div' (line 190)
        result_div_607209 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 12), 'div', float_607207, p_607208)
        
        # Assigning a type to the variable 'Q' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'Q', result_div_607209)
        
        # Assigning a BinOp to a Name (line 191):
        
        # Assigning a BinOp to a Name (line 191):
        # Getting the type of 'Q' (line 191)
        Q_607210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'Q')
        float_607211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 16), 'float')
        # Applying the binary operator '-' (line 191)
        result_sub_607212 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 12), '-', Q_607210, float_607211)
        
        # Assigning a type to the variable 'P' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'P', result_sub_607212)
        
        # Assigning a BinOp to a Name (line 192):
        
        # Assigning a BinOp to a Name (line 192):
        # Getting the type of 'n' (line 192)
        n_607213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 13), 'n')
        # Getting the type of 'P' (line 192)
        P_607214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'P')
        # Applying the binary operator '*' (line 192)
        result_mul_607215 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 13), '*', n_607213, P_607214)
        
        # Assigning a type to the variable 'mu' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'mu', result_mul_607215)
        
        # Assigning a BinOp to a Name (line 193):
        
        # Assigning a BinOp to a Name (line 193):
        # Getting the type of 'n' (line 193)
        n_607216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 14), 'n')
        # Getting the type of 'P' (line 193)
        P_607217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'P')
        # Applying the binary operator '*' (line 193)
        result_mul_607218 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 14), '*', n_607216, P_607217)
        
        # Getting the type of 'Q' (line 193)
        Q_607219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'Q')
        # Applying the binary operator '*' (line 193)
        result_mul_607220 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 17), '*', result_mul_607218, Q_607219)
        
        # Assigning a type to the variable 'var' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'var', result_mul_607220)
        
        # Assigning a BinOp to a Name (line 194):
        
        # Assigning a BinOp to a Name (line 194):
        # Getting the type of 'Q' (line 194)
        Q_607221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 14), 'Q')
        # Getting the type of 'P' (line 194)
        P_607222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'P')
        # Applying the binary operator '+' (line 194)
        result_add_607223 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 14), '+', Q_607221, P_607222)
        
        
        # Call to sqrt(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'n' (line 194)
        n_607225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'n', False)
        # Getting the type of 'P' (line 194)
        P_607226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 26), 'P', False)
        # Applying the binary operator '*' (line 194)
        result_mul_607227 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 24), '*', n_607225, P_607226)
        
        # Getting the type of 'Q' (line 194)
        Q_607228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'Q', False)
        # Applying the binary operator '*' (line 194)
        result_mul_607229 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 27), '*', result_mul_607227, Q_607228)
        
        # Processing the call keyword arguments (line 194)
        kwargs_607230 = {}
        # Getting the type of 'sqrt' (line 194)
        sqrt_607224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 194)
        sqrt_call_result_607231 = invoke(stypy.reporting.localization.Localization(__file__, 194, 19), sqrt_607224, *[result_mul_607229], **kwargs_607230)
        
        # Applying the binary operator 'div' (line 194)
        result_div_607232 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 13), 'div', result_add_607223, sqrt_call_result_607231)
        
        # Assigning a type to the variable 'g1' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'g1', result_div_607232)
        
        # Assigning a BinOp to a Name (line 195):
        
        # Assigning a BinOp to a Name (line 195):
        float_607233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 14), 'float')
        int_607234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 20), 'int')
        # Getting the type of 'P' (line 195)
        P_607235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'P')
        # Applying the binary operator '*' (line 195)
        result_mul_607236 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 20), '*', int_607234, P_607235)
        
        # Getting the type of 'Q' (line 195)
        Q_607237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'Q')
        # Applying the binary operator '*' (line 195)
        result_mul_607238 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 23), '*', result_mul_607236, Q_607237)
        
        # Applying the binary operator '+' (line 195)
        result_add_607239 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 14), '+', float_607233, result_mul_607238)
        
        # Getting the type of 'n' (line 195)
        n_607240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 30), 'n')
        # Getting the type of 'P' (line 195)
        P_607241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 32), 'P')
        # Applying the binary operator '*' (line 195)
        result_mul_607242 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 30), '*', n_607240, P_607241)
        
        # Getting the type of 'Q' (line 195)
        Q_607243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 34), 'Q')
        # Applying the binary operator '*' (line 195)
        result_mul_607244 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 33), '*', result_mul_607242, Q_607243)
        
        # Applying the binary operator 'div' (line 195)
        result_div_607245 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 13), 'div', result_add_607239, result_mul_607244)
        
        # Assigning a type to the variable 'g2' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'g2', result_div_607245)
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_607246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        # Getting the type of 'mu' (line 196)
        mu_607247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 15), tuple_607246, mu_607247)
        # Adding element type (line 196)
        # Getting the type of 'var' (line 196)
        var_607248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 15), tuple_607246, var_607248)
        # Adding element type (line 196)
        # Getting the type of 'g1' (line 196)
        g1_607249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 15), tuple_607246, g1_607249)
        # Adding element type (line 196)
        # Getting the type of 'g2' (line 196)
        g2_607250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 15), tuple_607246, g2_607250)
        
        # Assigning a type to the variable 'stypy_return_type' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'stypy_return_type', tuple_607246)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_607251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607251)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_607251


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nbinom_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'nbinom_gen' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'nbinom_gen', nbinom_gen)

# Assigning a Call to a Name (line 197):

# Assigning a Call to a Name (line 197):

# Call to nbinom_gen(...): (line 197)
# Processing the call keyword arguments (line 197)
str_607253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 25), 'str', 'nbinom')
keyword_607254 = str_607253
kwargs_607255 = {'name': keyword_607254}
# Getting the type of 'nbinom_gen' (line 197)
nbinom_gen_607252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'nbinom_gen', False)
# Calling nbinom_gen(args, kwargs) (line 197)
nbinom_gen_call_result_607256 = invoke(stypy.reporting.localization.Localization(__file__, 197, 9), nbinom_gen_607252, *[], **kwargs_607255)

# Assigning a type to the variable 'nbinom' (line 197)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'nbinom', nbinom_gen_call_result_607256)
# Declaration of the 'geom_gen' class
# Getting the type of 'rv_discrete' (line 200)
rv_discrete_607257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'rv_discrete')

class geom_gen(rv_discrete_607257, ):
    str_607258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', 'A geometric discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `geom` is::\n\n        geom.pmf(k) = (1-p)**(k-1)*p\n\n    for ``k >= 1``.\n\n    `geom` takes ``p`` as shape parameter.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._rvs.__dict__.__setitem__('stypy_function_name', 'geom_gen._rvs')
        geom_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['p'])
        geom_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._rvs', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Call to geometric(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'p' (line 221)
        p_607262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'p', False)
        # Processing the call keyword arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_607263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 52), 'self', False)
        # Obtaining the member '_size' of a type (line 221)
        _size_607264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 52), self_607263, '_size')
        keyword_607265 = _size_607264
        kwargs_607266 = {'size': keyword_607265}
        # Getting the type of 'self' (line 221)
        self_607259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'self', False)
        # Obtaining the member '_random_state' of a type (line 221)
        _random_state_607260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 15), self_607259, '_random_state')
        # Obtaining the member 'geometric' of a type (line 221)
        geometric_607261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 15), _random_state_607260, 'geometric')
        # Calling geometric(args, kwargs) (line 221)
        geometric_call_result_607267 = invoke(stypy.reporting.localization.Localization(__file__, 221, 15), geometric_607261, *[p_607262], **kwargs_607266)
        
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', geometric_call_result_607267)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_607268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607268)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_607268


    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'geom_gen._argcheck')
        geom_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['p'])
        geom_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._argcheck', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Getting the type of 'p' (line 224)
        p_607269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'p')
        int_607270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 21), 'int')
        # Applying the binary operator '<=' (line 224)
        result_le_607271 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 16), '<=', p_607269, int_607270)
        
        
        # Getting the type of 'p' (line 224)
        p_607272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'p')
        int_607273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 32), 'int')
        # Applying the binary operator '>=' (line 224)
        result_ge_607274 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 27), '>=', p_607272, int_607273)
        
        # Applying the binary operator '&' (line 224)
        result_and__607275 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 15), '&', result_le_607271, result_ge_607274)
        
        # Assigning a type to the variable 'stypy_return_type' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type', result_and__607275)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_607276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_607276


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._pmf.__dict__.__setitem__('stypy_function_name', 'geom_gen._pmf')
        geom_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'p'])
        geom_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._pmf', ['k', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Call to power(...): (line 227)
        # Processing the call arguments (line 227)
        int_607279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 24), 'int')
        # Getting the type of 'p' (line 227)
        p_607280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'p', False)
        # Applying the binary operator '-' (line 227)
        result_sub_607281 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 24), '-', int_607279, p_607280)
        
        # Getting the type of 'k' (line 227)
        k_607282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'k', False)
        int_607283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 31), 'int')
        # Applying the binary operator '-' (line 227)
        result_sub_607284 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 29), '-', k_607282, int_607283)
        
        # Processing the call keyword arguments (line 227)
        kwargs_607285 = {}
        # Getting the type of 'np' (line 227)
        np_607277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'np', False)
        # Obtaining the member 'power' of a type (line 227)
        power_607278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 15), np_607277, 'power')
        # Calling power(args, kwargs) (line 227)
        power_call_result_607286 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), power_607278, *[result_sub_607281, result_sub_607284], **kwargs_607285)
        
        # Getting the type of 'p' (line 227)
        p_607287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 36), 'p')
        # Applying the binary operator '*' (line 227)
        result_mul_607288 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 15), '*', power_call_result_607286, p_607287)
        
        # Assigning a type to the variable 'stypy_return_type' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'stypy_return_type', result_mul_607288)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_607289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_607289


    @norecursion
    def _logpmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logpmf'
        module_type_store = module_type_store.open_function_context('_logpmf', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._logpmf.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._logpmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._logpmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._logpmf.__dict__.__setitem__('stypy_function_name', 'geom_gen._logpmf')
        geom_gen._logpmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'p'])
        geom_gen._logpmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._logpmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._logpmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._logpmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._logpmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._logpmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._logpmf', ['k', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logpmf', localization, ['k', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logpmf(...)' code ##################

        
        # Call to xlog1py(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'k' (line 230)
        k_607292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 31), 'k', False)
        int_607293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 35), 'int')
        # Applying the binary operator '-' (line 230)
        result_sub_607294 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 31), '-', k_607292, int_607293)
        
        
        # Getting the type of 'p' (line 230)
        p_607295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 39), 'p', False)
        # Applying the 'usub' unary operator (line 230)
        result___neg___607296 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 38), 'usub', p_607295)
        
        # Processing the call keyword arguments (line 230)
        kwargs_607297 = {}
        # Getting the type of 'special' (line 230)
        special_607290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'special', False)
        # Obtaining the member 'xlog1py' of a type (line 230)
        xlog1py_607291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 15), special_607290, 'xlog1py')
        # Calling xlog1py(args, kwargs) (line 230)
        xlog1py_call_result_607298 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), xlog1py_607291, *[result_sub_607294, result___neg___607296], **kwargs_607297)
        
        
        # Call to log(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'p' (line 230)
        p_607300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'p', False)
        # Processing the call keyword arguments (line 230)
        kwargs_607301 = {}
        # Getting the type of 'log' (line 230)
        log_607299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 44), 'log', False)
        # Calling log(args, kwargs) (line 230)
        log_call_result_607302 = invoke(stypy.reporting.localization.Localization(__file__, 230, 44), log_607299, *[p_607300], **kwargs_607301)
        
        # Applying the binary operator '+' (line 230)
        result_add_607303 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 15), '+', xlog1py_call_result_607298, log_call_result_607302)
        
        # Assigning a type to the variable 'stypy_return_type' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stypy_return_type', result_add_607303)
        
        # ################# End of '_logpmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logpmf' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_607304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607304)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logpmf'
        return stypy_return_type_607304


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._cdf.__dict__.__setitem__('stypy_function_name', 'geom_gen._cdf')
        geom_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        geom_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._cdf', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to floor(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'x' (line 233)
        x_607306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'x', False)
        # Processing the call keyword arguments (line 233)
        kwargs_607307 = {}
        # Getting the type of 'floor' (line 233)
        floor_607305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 233)
        floor_call_result_607308 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), floor_607305, *[x_607306], **kwargs_607307)
        
        # Assigning a type to the variable 'k' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'k', floor_call_result_607308)
        
        
        # Call to expm1(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Call to log1p(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Getting the type of 'p' (line 234)
        p_607311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'p', False)
        # Applying the 'usub' unary operator (line 234)
        result___neg___607312 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 28), 'usub', p_607311)
        
        # Processing the call keyword arguments (line 234)
        kwargs_607313 = {}
        # Getting the type of 'log1p' (line 234)
        log1p_607310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 22), 'log1p', False)
        # Calling log1p(args, kwargs) (line 234)
        log1p_call_result_607314 = invoke(stypy.reporting.localization.Localization(__file__, 234, 22), log1p_607310, *[result___neg___607312], **kwargs_607313)
        
        # Getting the type of 'k' (line 234)
        k_607315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 32), 'k', False)
        # Applying the binary operator '*' (line 234)
        result_mul_607316 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 22), '*', log1p_call_result_607314, k_607315)
        
        # Processing the call keyword arguments (line 234)
        kwargs_607317 = {}
        # Getting the type of 'expm1' (line 234)
        expm1_607309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'expm1', False)
        # Calling expm1(args, kwargs) (line 234)
        expm1_call_result_607318 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), expm1_607309, *[result_mul_607316], **kwargs_607317)
        
        # Applying the 'usub' unary operator (line 234)
        result___neg___607319 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 15), 'usub', expm1_call_result_607318)
        
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'stypy_return_type', result___neg___607319)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_607320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607320)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_607320


    @norecursion
    def _sf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sf'
        module_type_store = module_type_store.open_function_context('_sf', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._sf.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._sf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._sf.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._sf.__dict__.__setitem__('stypy_function_name', 'geom_gen._sf')
        geom_gen._sf.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        geom_gen._sf.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._sf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._sf.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._sf.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._sf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._sf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._sf', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sf', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sf(...)' code ##################

        
        # Call to exp(...): (line 237)
        # Processing the call arguments (line 237)
        
        # Call to _logsf(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'x' (line 237)
        x_607325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 34), 'x', False)
        # Getting the type of 'p' (line 237)
        p_607326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 37), 'p', False)
        # Processing the call keyword arguments (line 237)
        kwargs_607327 = {}
        # Getting the type of 'self' (line 237)
        self_607323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'self', False)
        # Obtaining the member '_logsf' of a type (line 237)
        _logsf_607324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 22), self_607323, '_logsf')
        # Calling _logsf(args, kwargs) (line 237)
        _logsf_call_result_607328 = invoke(stypy.reporting.localization.Localization(__file__, 237, 22), _logsf_607324, *[x_607325, p_607326], **kwargs_607327)
        
        # Processing the call keyword arguments (line 237)
        kwargs_607329 = {}
        # Getting the type of 'np' (line 237)
        np_607321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 237)
        exp_607322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 15), np_607321, 'exp')
        # Calling exp(args, kwargs) (line 237)
        exp_call_result_607330 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), exp_607322, *[_logsf_call_result_607328], **kwargs_607329)
        
        # Assigning a type to the variable 'stypy_return_type' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type', exp_call_result_607330)
        
        # ################# End of '_sf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sf' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_607331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607331)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sf'
        return stypy_return_type_607331


    @norecursion
    def _logsf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logsf'
        module_type_store = module_type_store.open_function_context('_logsf', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._logsf.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._logsf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._logsf.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._logsf.__dict__.__setitem__('stypy_function_name', 'geom_gen._logsf')
        geom_gen._logsf.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        geom_gen._logsf.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._logsf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._logsf.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._logsf.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._logsf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._logsf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._logsf', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logsf', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logsf(...)' code ##################

        
        # Assigning a Call to a Name (line 240):
        
        # Assigning a Call to a Name (line 240):
        
        # Call to floor(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'x' (line 240)
        x_607333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'x', False)
        # Processing the call keyword arguments (line 240)
        kwargs_607334 = {}
        # Getting the type of 'floor' (line 240)
        floor_607332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 240)
        floor_call_result_607335 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), floor_607332, *[x_607333], **kwargs_607334)
        
        # Assigning a type to the variable 'k' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'k', floor_call_result_607335)
        # Getting the type of 'k' (line 241)
        k_607336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'k')
        
        # Call to log1p(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Getting the type of 'p' (line 241)
        p_607338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'p', False)
        # Applying the 'usub' unary operator (line 241)
        result___neg___607339 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 23), 'usub', p_607338)
        
        # Processing the call keyword arguments (line 241)
        kwargs_607340 = {}
        # Getting the type of 'log1p' (line 241)
        log1p_607337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'log1p', False)
        # Calling log1p(args, kwargs) (line 241)
        log1p_call_result_607341 = invoke(stypy.reporting.localization.Localization(__file__, 241, 17), log1p_607337, *[result___neg___607339], **kwargs_607340)
        
        # Applying the binary operator '*' (line 241)
        result_mul_607342 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 15), '*', k_607336, log1p_call_result_607341)
        
        # Assigning a type to the variable 'stypy_return_type' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'stypy_return_type', result_mul_607342)
        
        # ################# End of '_logsf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logsf' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_607343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607343)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logsf'
        return stypy_return_type_607343


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._ppf.__dict__.__setitem__('stypy_function_name', 'geom_gen._ppf')
        geom_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'p'])
        geom_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._ppf', ['q', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to ceil(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Call to log(...): (line 244)
        # Processing the call arguments (line 244)
        float_607346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 24), 'float')
        # Getting the type of 'q' (line 244)
        q_607347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 28), 'q', False)
        # Applying the binary operator '-' (line 244)
        result_sub_607348 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 24), '-', float_607346, q_607347)
        
        # Processing the call keyword arguments (line 244)
        kwargs_607349 = {}
        # Getting the type of 'log' (line 244)
        log_607345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'log', False)
        # Calling log(args, kwargs) (line 244)
        log_call_result_607350 = invoke(stypy.reporting.localization.Localization(__file__, 244, 20), log_607345, *[result_sub_607348], **kwargs_607349)
        
        
        # Call to log(...): (line 244)
        # Processing the call arguments (line 244)
        int_607352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 35), 'int')
        # Getting the type of 'p' (line 244)
        p_607353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 37), 'p', False)
        # Applying the binary operator '-' (line 244)
        result_sub_607354 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 35), '-', int_607352, p_607353)
        
        # Processing the call keyword arguments (line 244)
        kwargs_607355 = {}
        # Getting the type of 'log' (line 244)
        log_607351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 31), 'log', False)
        # Calling log(args, kwargs) (line 244)
        log_call_result_607356 = invoke(stypy.reporting.localization.Localization(__file__, 244, 31), log_607351, *[result_sub_607354], **kwargs_607355)
        
        # Applying the binary operator 'div' (line 244)
        result_div_607357 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 20), 'div', log_call_result_607350, log_call_result_607356)
        
        # Processing the call keyword arguments (line 244)
        kwargs_607358 = {}
        # Getting the type of 'ceil' (line 244)
        ceil_607344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'ceil', False)
        # Calling ceil(args, kwargs) (line 244)
        ceil_call_result_607359 = invoke(stypy.reporting.localization.Localization(__file__, 244, 15), ceil_607344, *[result_div_607357], **kwargs_607358)
        
        # Assigning a type to the variable 'vals' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'vals', ceil_call_result_607359)
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to _cdf(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'vals' (line 245)
        vals_607362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 25), 'vals', False)
        int_607363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 30), 'int')
        # Applying the binary operator '-' (line 245)
        result_sub_607364 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 25), '-', vals_607362, int_607363)
        
        # Getting the type of 'p' (line 245)
        p_607365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 33), 'p', False)
        # Processing the call keyword arguments (line 245)
        kwargs_607366 = {}
        # Getting the type of 'self' (line 245)
        self_607360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'self', False)
        # Obtaining the member '_cdf' of a type (line 245)
        _cdf_607361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), self_607360, '_cdf')
        # Calling _cdf(args, kwargs) (line 245)
        _cdf_call_result_607367 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), _cdf_607361, *[result_sub_607364, p_607365], **kwargs_607366)
        
        # Assigning a type to the variable 'temp' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'temp', _cdf_call_result_607367)
        
        # Call to where(...): (line 246)
        # Processing the call arguments (line 246)
        
        # Getting the type of 'temp' (line 246)
        temp_607370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'temp', False)
        # Getting the type of 'q' (line 246)
        q_607371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 33), 'q', False)
        # Applying the binary operator '>=' (line 246)
        result_ge_607372 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 25), '>=', temp_607370, q_607371)
        
        
        # Getting the type of 'vals' (line 246)
        vals_607373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 39), 'vals', False)
        int_607374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 46), 'int')
        # Applying the binary operator '>' (line 246)
        result_gt_607375 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 39), '>', vals_607373, int_607374)
        
        # Applying the binary operator '&' (line 246)
        result_and__607376 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 24), '&', result_ge_607372, result_gt_607375)
        
        # Getting the type of 'vals' (line 246)
        vals_607377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 50), 'vals', False)
        int_607378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 55), 'int')
        # Applying the binary operator '-' (line 246)
        result_sub_607379 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 50), '-', vals_607377, int_607378)
        
        # Getting the type of 'vals' (line 246)
        vals_607380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 58), 'vals', False)
        # Processing the call keyword arguments (line 246)
        kwargs_607381 = {}
        # Getting the type of 'np' (line 246)
        np_607368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 246)
        where_607369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), np_607368, 'where')
        # Calling where(args, kwargs) (line 246)
        where_call_result_607382 = invoke(stypy.reporting.localization.Localization(__file__, 246, 15), where_607369, *[result_and__607376, result_sub_607379, vals_607380], **kwargs_607381)
        
        # Assigning a type to the variable 'stypy_return_type' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'stypy_return_type', where_call_result_607382)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_607383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607383)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_607383


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        geom_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        geom_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        geom_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        geom_gen._stats.__dict__.__setitem__('stypy_function_name', 'geom_gen._stats')
        geom_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['p'])
        geom_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        geom_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        geom_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        geom_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        geom_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        geom_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen._stats', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a BinOp to a Name (line 249):
        
        # Assigning a BinOp to a Name (line 249):
        float_607384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 13), 'float')
        # Getting the type of 'p' (line 249)
        p_607385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 17), 'p')
        # Applying the binary operator 'div' (line 249)
        result_div_607386 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 13), 'div', float_607384, p_607385)
        
        # Assigning a type to the variable 'mu' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'mu', result_div_607386)
        
        # Assigning a BinOp to a Name (line 250):
        
        # Assigning a BinOp to a Name (line 250):
        float_607387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 13), 'float')
        # Getting the type of 'p' (line 250)
        p_607388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 17), 'p')
        # Applying the binary operator '-' (line 250)
        result_sub_607389 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 13), '-', float_607387, p_607388)
        
        # Assigning a type to the variable 'qr' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'qr', result_sub_607389)
        
        # Assigning a BinOp to a Name (line 251):
        
        # Assigning a BinOp to a Name (line 251):
        # Getting the type of 'qr' (line 251)
        qr_607390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 14), 'qr')
        # Getting the type of 'p' (line 251)
        p_607391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 19), 'p')
        # Applying the binary operator 'div' (line 251)
        result_div_607392 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 14), 'div', qr_607390, p_607391)
        
        # Getting the type of 'p' (line 251)
        p_607393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'p')
        # Applying the binary operator 'div' (line 251)
        result_div_607394 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 21), 'div', result_div_607392, p_607393)
        
        # Assigning a type to the variable 'var' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'var', result_div_607394)
        
        # Assigning a BinOp to a Name (line 252):
        
        # Assigning a BinOp to a Name (line 252):
        float_607395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 14), 'float')
        # Getting the type of 'p' (line 252)
        p_607396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'p')
        # Applying the binary operator '-' (line 252)
        result_sub_607397 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 14), '-', float_607395, p_607396)
        
        
        # Call to sqrt(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'qr' (line 252)
        qr_607399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), 'qr', False)
        # Processing the call keyword arguments (line 252)
        kwargs_607400 = {}
        # Getting the type of 'sqrt' (line 252)
        sqrt_607398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 23), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 252)
        sqrt_call_result_607401 = invoke(stypy.reporting.localization.Localization(__file__, 252, 23), sqrt_607398, *[qr_607399], **kwargs_607400)
        
        # Applying the binary operator 'div' (line 252)
        result_div_607402 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 13), 'div', result_sub_607397, sqrt_call_result_607401)
        
        # Assigning a type to the variable 'g1' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'g1', result_div_607402)
        
        # Assigning a BinOp to a Name (line 253):
        
        # Assigning a BinOp to a Name (line 253):
        
        # Call to polyval(...): (line 253)
        # Processing the call arguments (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_607405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        int_607406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 24), list_607405, int_607406)
        # Adding element type (line 253)
        int_607407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 24), list_607405, int_607407)
        # Adding element type (line 253)
        int_607408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 24), list_607405, int_607408)
        
        # Getting the type of 'p' (line 253)
        p_607409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 36), 'p', False)
        # Processing the call keyword arguments (line 253)
        kwargs_607410 = {}
        # Getting the type of 'np' (line 253)
        np_607403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 13), 'np', False)
        # Obtaining the member 'polyval' of a type (line 253)
        polyval_607404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 13), np_607403, 'polyval')
        # Calling polyval(args, kwargs) (line 253)
        polyval_call_result_607411 = invoke(stypy.reporting.localization.Localization(__file__, 253, 13), polyval_607404, *[list_607405, p_607409], **kwargs_607410)
        
        float_607412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 40), 'float')
        # Getting the type of 'p' (line 253)
        p_607413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 44), 'p')
        # Applying the binary operator '-' (line 253)
        result_sub_607414 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 40), '-', float_607412, p_607413)
        
        # Applying the binary operator 'div' (line 253)
        result_div_607415 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 13), 'div', polyval_call_result_607411, result_sub_607414)
        
        # Assigning a type to the variable 'g2' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'g2', result_div_607415)
        
        # Obtaining an instance of the builtin type 'tuple' (line 254)
        tuple_607416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 254)
        # Adding element type (line 254)
        # Getting the type of 'mu' (line 254)
        mu_607417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 15), tuple_607416, mu_607417)
        # Adding element type (line 254)
        # Getting the type of 'var' (line 254)
        var_607418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 15), tuple_607416, var_607418)
        # Adding element type (line 254)
        # Getting the type of 'g1' (line 254)
        g1_607419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 15), tuple_607416, g1_607419)
        # Adding element type (line 254)
        # Getting the type of 'g2' (line 254)
        g2_607420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 15), tuple_607416, g2_607420)
        
        # Assigning a type to the variable 'stypy_return_type' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'stypy_return_type', tuple_607416)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_607421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607421)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_607421


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 200, 0, False)
        # Assigning a type to the variable 'self' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'geom_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'geom_gen' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'geom_gen', geom_gen)

# Assigning a Call to a Name (line 255):

# Assigning a Call to a Name (line 255):

# Call to geom_gen(...): (line 255)
# Processing the call keyword arguments (line 255)
int_607423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 18), 'int')
keyword_607424 = int_607423
str_607425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 26), 'str', 'geom')
keyword_607426 = str_607425
str_607427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 43), 'str', 'A geometric')
keyword_607428 = str_607427
kwargs_607429 = {'a': keyword_607424, 'name': keyword_607426, 'longname': keyword_607428}
# Getting the type of 'geom_gen' (line 255)
geom_gen_607422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 7), 'geom_gen', False)
# Calling geom_gen(args, kwargs) (line 255)
geom_gen_call_result_607430 = invoke(stypy.reporting.localization.Localization(__file__, 255, 7), geom_gen_607422, *[], **kwargs_607429)

# Assigning a type to the variable 'geom' (line 255)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'geom', geom_gen_call_result_607430)
# Declaration of the 'hypergeom_gen' class
# Getting the type of 'rv_discrete' (line 258)
rv_discrete_607431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'rv_discrete')

class hypergeom_gen(rv_discrete_607431, ):
    str_607432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, (-1)), 'str', "A hypergeometric discrete random variable.\n\n    The hypergeometric distribution models drawing objects from a bin.\n    `M` is the total number of objects, `n` is total number of Type I objects.\n    The random variate represents the number of Type I objects in `N` drawn\n    without replacement from the total population.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The symbols used to denote the shape parameters (`M`, `n`, and `N`) are not\n    universally accepted.  See the Examples for a clarification of the\n    definitions used here.\n\n    The probability mass function is defined as,\n\n    .. math:: p(k, M, n, N) = \\frac{\\binom{n}{k} \\binom{M - n}{N - k}}{\\binom{M}{N}}\n\n    for :math:`k \\in [\\max(0, N - M + n), \\min(n, N)]`, where the binomial\n    coefficients are defined as,\n\n    .. math:: \\binom{n}{k} \\equiv \\frac{n!}{k! (n - k)!}.\n\n    %(after_notes)s\n\n    Examples\n    --------\n    >>> from scipy.stats import hypergeom\n    >>> import matplotlib.pyplot as plt\n\n    Suppose we have a collection of 20 animals, of which 7 are dogs.  Then if\n    we want to know the probability of finding a given number of dogs if we\n    choose at random 12 of the 20 animals, we can initialize a frozen\n    distribution and plot the probability mass function:\n\n    >>> [M, n, N] = [20, 7, 12]\n    >>> rv = hypergeom(M, n, N)\n    >>> x = np.arange(0, n+1)\n    >>> pmf_dogs = rv.pmf(x)\n\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> ax.plot(x, pmf_dogs, 'bo')\n    >>> ax.vlines(x, 0, pmf_dogs, lw=2)\n    >>> ax.set_xlabel('# of dogs in our group of chosen animals')\n    >>> ax.set_ylabel('hypergeom PMF')\n    >>> plt.show()\n\n    Instead of using a frozen distribution we can also use `hypergeom`\n    methods directly.  To for example obtain the cumulative distribution\n    function, use:\n\n    >>> prb = hypergeom.cdf(x, M, n, N)\n\n    And to generate random numbers:\n\n    >>> R = hypergeom.rvs(M, n, N, size=10)\n\n    ")

    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_function_name', 'hypergeom_gen._rvs')
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['M', 'n', 'N'])
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        hypergeom_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen._rvs', ['M', 'n', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['M', 'n', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Call to hypergeometric(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'n' (line 320)
        n_607436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 49), 'n', False)
        # Getting the type of 'M' (line 320)
        M_607437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 52), 'M', False)
        # Getting the type of 'n' (line 320)
        n_607438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 54), 'n', False)
        # Applying the binary operator '-' (line 320)
        result_sub_607439 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 52), '-', M_607437, n_607438)
        
        # Getting the type of 'N' (line 320)
        N_607440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 57), 'N', False)
        # Processing the call keyword arguments (line 320)
        # Getting the type of 'self' (line 320)
        self_607441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 65), 'self', False)
        # Obtaining the member '_size' of a type (line 320)
        _size_607442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 65), self_607441, '_size')
        keyword_607443 = _size_607442
        kwargs_607444 = {'size': keyword_607443}
        # Getting the type of 'self' (line 320)
        self_607433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'self', False)
        # Obtaining the member '_random_state' of a type (line 320)
        _random_state_607434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 15), self_607433, '_random_state')
        # Obtaining the member 'hypergeometric' of a type (line 320)
        hypergeometric_607435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 15), _random_state_607434, 'hypergeometric')
        # Calling hypergeometric(args, kwargs) (line 320)
        hypergeometric_call_result_607445 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), hypergeometric_607435, *[n_607436, result_sub_607439, N_607440], **kwargs_607444)
        
        # Assigning a type to the variable 'stypy_return_type' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type', hypergeometric_call_result_607445)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_607446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607446)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_607446


    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'hypergeom_gen._argcheck')
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['M', 'n', 'N'])
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        hypergeom_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen._argcheck', ['M', 'n', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['M', 'n', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Assigning a BinOp to a Name (line 323):
        
        # Assigning a BinOp to a Name (line 323):
        
        # Getting the type of 'M' (line 323)
        M_607447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'M')
        int_607448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'int')
        # Applying the binary operator '>' (line 323)
        result_gt_607449 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 16), '>', M_607447, int_607448)
        
        
        # Getting the type of 'n' (line 323)
        n_607450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'n')
        int_607451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 31), 'int')
        # Applying the binary operator '>=' (line 323)
        result_ge_607452 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 26), '>=', n_607450, int_607451)
        
        # Applying the binary operator '&' (line 323)
        result_and__607453 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 15), '&', result_gt_607449, result_ge_607452)
        
        
        # Getting the type of 'N' (line 323)
        N_607454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 37), 'N')
        int_607455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 42), 'int')
        # Applying the binary operator '>=' (line 323)
        result_ge_607456 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 37), '>=', N_607454, int_607455)
        
        # Applying the binary operator '&' (line 323)
        result_and__607457 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 34), '&', result_and__607453, result_ge_607456)
        
        # Assigning a type to the variable 'cond' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'cond', result_and__607457)
        
        # Getting the type of 'cond' (line 324)
        cond_607458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'cond')
        
        # Getting the type of 'n' (line 324)
        n_607459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 17), 'n')
        # Getting the type of 'M' (line 324)
        M_607460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 22), 'M')
        # Applying the binary operator '<=' (line 324)
        result_le_607461 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 17), '<=', n_607459, M_607460)
        
        
        # Getting the type of 'N' (line 324)
        N_607462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'N')
        # Getting the type of 'M' (line 324)
        M_607463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 33), 'M')
        # Applying the binary operator '<=' (line 324)
        result_le_607464 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 28), '<=', N_607462, M_607463)
        
        # Applying the binary operator '&' (line 324)
        result_and__607465 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 16), '&', result_le_607461, result_le_607464)
        
        # Applying the binary operator '&=' (line 324)
        result_iand_607466 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 8), '&=', cond_607458, result_and__607465)
        # Assigning a type to the variable 'cond' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'cond', result_iand_607466)
        
        
        # Assigning a Call to a Attribute (line 325):
        
        # Assigning a Call to a Attribute (line 325):
        
        # Call to maximum(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'N' (line 325)
        N_607469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 28), 'N', False)
        # Getting the type of 'M' (line 325)
        M_607470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 31), 'M', False)
        # Getting the type of 'n' (line 325)
        n_607471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 33), 'n', False)
        # Applying the binary operator '-' (line 325)
        result_sub_607472 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 31), '-', M_607470, n_607471)
        
        # Applying the binary operator '-' (line 325)
        result_sub_607473 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 28), '-', N_607469, result_sub_607472)
        
        int_607474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 37), 'int')
        # Processing the call keyword arguments (line 325)
        kwargs_607475 = {}
        # Getting the type of 'np' (line 325)
        np_607467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 17), 'np', False)
        # Obtaining the member 'maximum' of a type (line 325)
        maximum_607468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 17), np_607467, 'maximum')
        # Calling maximum(args, kwargs) (line 325)
        maximum_call_result_607476 = invoke(stypy.reporting.localization.Localization(__file__, 325, 17), maximum_607468, *[result_sub_607473, int_607474], **kwargs_607475)
        
        # Getting the type of 'self' (line 325)
        self_607477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'self')
        # Setting the type of the member 'a' of a type (line 325)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), self_607477, 'a', maximum_call_result_607476)
        
        # Assigning a Call to a Attribute (line 326):
        
        # Assigning a Call to a Attribute (line 326):
        
        # Call to minimum(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'n' (line 326)
        n_607480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 28), 'n', False)
        # Getting the type of 'N' (line 326)
        N_607481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 31), 'N', False)
        # Processing the call keyword arguments (line 326)
        kwargs_607482 = {}
        # Getting the type of 'np' (line 326)
        np_607478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 17), 'np', False)
        # Obtaining the member 'minimum' of a type (line 326)
        minimum_607479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 17), np_607478, 'minimum')
        # Calling minimum(args, kwargs) (line 326)
        minimum_call_result_607483 = invoke(stypy.reporting.localization.Localization(__file__, 326, 17), minimum_607479, *[n_607480, N_607481], **kwargs_607482)
        
        # Getting the type of 'self' (line 326)
        self_607484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self')
        # Setting the type of the member 'b' of a type (line 326)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_607484, 'b', minimum_call_result_607483)
        # Getting the type of 'cond' (line 327)
        cond_607485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'cond')
        # Assigning a type to the variable 'stypy_return_type' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stypy_return_type', cond_607485)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_607486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607486)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_607486


    @norecursion
    def _logpmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logpmf'
        module_type_store = module_type_store.open_function_context('_logpmf', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_localization', localization)
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_function_name', 'hypergeom_gen._logpmf')
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'M', 'n', 'N'])
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        hypergeom_gen._logpmf.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen._logpmf', ['k', 'M', 'n', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logpmf', localization, ['k', 'M', 'n', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logpmf(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 330):
        
        # Assigning a Name to a Name (line 330):
        # Getting the type of 'M' (line 330)
        M_607487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 'M')
        # Assigning a type to the variable 'tuple_assignment_606749' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_assignment_606749', M_607487)
        
        # Assigning a Name to a Name (line 330):
        # Getting the type of 'n' (line 330)
        n_607488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 23), 'n')
        # Assigning a type to the variable 'tuple_assignment_606750' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_assignment_606750', n_607488)
        
        # Assigning a Name to a Name (line 330):
        # Getting the type of 'tuple_assignment_606749' (line 330)
        tuple_assignment_606749_607489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_assignment_606749')
        # Assigning a type to the variable 'tot' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tot', tuple_assignment_606749_607489)
        
        # Assigning a Name to a Name (line 330):
        # Getting the type of 'tuple_assignment_606750' (line 330)
        tuple_assignment_606750_607490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_assignment_606750')
        # Assigning a type to the variable 'good' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 13), 'good', tuple_assignment_606750_607490)
        
        # Assigning a BinOp to a Name (line 331):
        
        # Assigning a BinOp to a Name (line 331):
        # Getting the type of 'tot' (line 331)
        tot_607491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'tot')
        # Getting the type of 'good' (line 331)
        good_607492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'good')
        # Applying the binary operator '-' (line 331)
        result_sub_607493 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 14), '-', tot_607491, good_607492)
        
        # Assigning a type to the variable 'bad' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'bad', result_sub_607493)
        
        # Call to betaln(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'good' (line 332)
        good_607495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 22), 'good', False)
        int_607496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 27), 'int')
        # Applying the binary operator '+' (line 332)
        result_add_607497 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 22), '+', good_607495, int_607496)
        
        int_607498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 30), 'int')
        # Processing the call keyword arguments (line 332)
        kwargs_607499 = {}
        # Getting the type of 'betaln' (line 332)
        betaln_607494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'betaln', False)
        # Calling betaln(args, kwargs) (line 332)
        betaln_call_result_607500 = invoke(stypy.reporting.localization.Localization(__file__, 332, 15), betaln_607494, *[result_add_607497, int_607498], **kwargs_607499)
        
        
        # Call to betaln(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'bad' (line 332)
        bad_607502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 42), 'bad', False)
        int_607503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 46), 'int')
        # Applying the binary operator '+' (line 332)
        result_add_607504 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 42), '+', bad_607502, int_607503)
        
        int_607505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 48), 'int')
        # Processing the call keyword arguments (line 332)
        kwargs_607506 = {}
        # Getting the type of 'betaln' (line 332)
        betaln_607501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'betaln', False)
        # Calling betaln(args, kwargs) (line 332)
        betaln_call_result_607507 = invoke(stypy.reporting.localization.Localization(__file__, 332, 35), betaln_607501, *[result_add_607504, int_607505], **kwargs_607506)
        
        # Applying the binary operator '+' (line 332)
        result_add_607508 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 15), '+', betaln_call_result_607500, betaln_call_result_607507)
        
        
        # Call to betaln(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'tot' (line 332)
        tot_607510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 60), 'tot', False)
        # Getting the type of 'N' (line 332)
        N_607511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 64), 'N', False)
        # Applying the binary operator '-' (line 332)
        result_sub_607512 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 60), '-', tot_607510, N_607511)
        
        int_607513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 66), 'int')
        # Applying the binary operator '+' (line 332)
        result_add_607514 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 65), '+', result_sub_607512, int_607513)
        
        # Getting the type of 'N' (line 332)
        N_607515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 69), 'N', False)
        int_607516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 71), 'int')
        # Applying the binary operator '+' (line 332)
        result_add_607517 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 69), '+', N_607515, int_607516)
        
        # Processing the call keyword arguments (line 332)
        kwargs_607518 = {}
        # Getting the type of 'betaln' (line 332)
        betaln_607509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 53), 'betaln', False)
        # Calling betaln(args, kwargs) (line 332)
        betaln_call_result_607519 = invoke(stypy.reporting.localization.Localization(__file__, 332, 53), betaln_607509, *[result_add_607514, result_add_607517], **kwargs_607518)
        
        # Applying the binary operator '+' (line 332)
        result_add_607520 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 51), '+', result_add_607508, betaln_call_result_607519)
        
        
        # Call to betaln(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'k' (line 333)
        k_607522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 21), 'k', False)
        int_607523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 23), 'int')
        # Applying the binary operator '+' (line 333)
        result_add_607524 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 21), '+', k_607522, int_607523)
        
        # Getting the type of 'good' (line 333)
        good_607525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 26), 'good', False)
        # Getting the type of 'k' (line 333)
        k_607526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 31), 'k', False)
        # Applying the binary operator '-' (line 333)
        result_sub_607527 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 26), '-', good_607525, k_607526)
        
        int_607528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 33), 'int')
        # Applying the binary operator '+' (line 333)
        result_add_607529 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 32), '+', result_sub_607527, int_607528)
        
        # Processing the call keyword arguments (line 333)
        kwargs_607530 = {}
        # Getting the type of 'betaln' (line 333)
        betaln_607521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 14), 'betaln', False)
        # Calling betaln(args, kwargs) (line 333)
        betaln_call_result_607531 = invoke(stypy.reporting.localization.Localization(__file__, 333, 14), betaln_607521, *[result_add_607524, result_add_607529], **kwargs_607530)
        
        # Applying the binary operator '-' (line 333)
        result_sub_607532 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 12), '-', result_add_607520, betaln_call_result_607531)
        
        
        # Call to betaln(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'N' (line 333)
        N_607534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 45), 'N', False)
        # Getting the type of 'k' (line 333)
        k_607535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 47), 'k', False)
        # Applying the binary operator '-' (line 333)
        result_sub_607536 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 45), '-', N_607534, k_607535)
        
        int_607537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 49), 'int')
        # Applying the binary operator '+' (line 333)
        result_add_607538 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 48), '+', result_sub_607536, int_607537)
        
        # Getting the type of 'bad' (line 333)
        bad_607539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 51), 'bad', False)
        # Getting the type of 'N' (line 333)
        N_607540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 55), 'N', False)
        # Applying the binary operator '-' (line 333)
        result_sub_607541 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 51), '-', bad_607539, N_607540)
        
        # Getting the type of 'k' (line 333)
        k_607542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 57), 'k', False)
        # Applying the binary operator '+' (line 333)
        result_add_607543 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 56), '+', result_sub_607541, k_607542)
        
        int_607544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 59), 'int')
        # Applying the binary operator '+' (line 333)
        result_add_607545 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 58), '+', result_add_607543, int_607544)
        
        # Processing the call keyword arguments (line 333)
        kwargs_607546 = {}
        # Getting the type of 'betaln' (line 333)
        betaln_607533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 38), 'betaln', False)
        # Calling betaln(args, kwargs) (line 333)
        betaln_call_result_607547 = invoke(stypy.reporting.localization.Localization(__file__, 333, 38), betaln_607533, *[result_add_607538, result_add_607545], **kwargs_607546)
        
        # Applying the binary operator '-' (line 333)
        result_sub_607548 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 36), '-', result_sub_607532, betaln_call_result_607547)
        
        
        # Call to betaln(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'tot' (line 334)
        tot_607550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 21), 'tot', False)
        int_607551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 25), 'int')
        # Applying the binary operator '+' (line 334)
        result_add_607552 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 21), '+', tot_607550, int_607551)
        
        int_607553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 28), 'int')
        # Processing the call keyword arguments (line 334)
        kwargs_607554 = {}
        # Getting the type of 'betaln' (line 334)
        betaln_607549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 14), 'betaln', False)
        # Calling betaln(args, kwargs) (line 334)
        betaln_call_result_607555 = invoke(stypy.reporting.localization.Localization(__file__, 334, 14), betaln_607549, *[result_add_607552, int_607553], **kwargs_607554)
        
        # Applying the binary operator '-' (line 334)
        result_sub_607556 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 12), '-', result_sub_607548, betaln_call_result_607555)
        
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', result_sub_607556)
        
        # ################# End of '_logpmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logpmf' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_607557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607557)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logpmf'
        return stypy_return_type_607557


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_function_name', 'hypergeom_gen._pmf')
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'M', 'n', 'N'])
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        hypergeom_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen._pmf', ['k', 'M', 'n', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'M', 'n', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Call to exp(...): (line 339)
        # Processing the call arguments (line 339)
        
        # Call to _logpmf(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'k' (line 339)
        k_607561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 32), 'k', False)
        # Getting the type of 'M' (line 339)
        M_607562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 35), 'M', False)
        # Getting the type of 'n' (line 339)
        n_607563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 38), 'n', False)
        # Getting the type of 'N' (line 339)
        N_607564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 41), 'N', False)
        # Processing the call keyword arguments (line 339)
        kwargs_607565 = {}
        # Getting the type of 'self' (line 339)
        self_607559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'self', False)
        # Obtaining the member '_logpmf' of a type (line 339)
        _logpmf_607560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 19), self_607559, '_logpmf')
        # Calling _logpmf(args, kwargs) (line 339)
        _logpmf_call_result_607566 = invoke(stypy.reporting.localization.Localization(__file__, 339, 19), _logpmf_607560, *[k_607561, M_607562, n_607563, N_607564], **kwargs_607565)
        
        # Processing the call keyword arguments (line 339)
        kwargs_607567 = {}
        # Getting the type of 'exp' (line 339)
        exp_607558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'exp', False)
        # Calling exp(args, kwargs) (line 339)
        exp_call_result_607568 = invoke(stypy.reporting.localization.Localization(__file__, 339, 15), exp_607558, *[_logpmf_call_result_607566], **kwargs_607567)
        
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', exp_call_result_607568)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_607569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607569)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_607569


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 341, 4, False)
        # Assigning a type to the variable 'self' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        hypergeom_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        hypergeom_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        hypergeom_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        hypergeom_gen._stats.__dict__.__setitem__('stypy_function_name', 'hypergeom_gen._stats')
        hypergeom_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['M', 'n', 'N'])
        hypergeom_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        hypergeom_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        hypergeom_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        hypergeom_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        hypergeom_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        hypergeom_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen._stats', ['M', 'n', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['M', 'n', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 344):
        
        # Assigning a BinOp to a Name (line 344):
        float_607570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 18), 'float')
        # Getting the type of 'M' (line 344)
        M_607571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 'M')
        # Applying the binary operator '*' (line 344)
        result_mul_607572 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 18), '*', float_607570, M_607571)
        
        # Assigning a type to the variable 'tuple_assignment_606751' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'tuple_assignment_606751', result_mul_607572)
        
        # Assigning a BinOp to a Name (line 344):
        float_607573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 24), 'float')
        # Getting the type of 'n' (line 344)
        n_607574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 27), 'n')
        # Applying the binary operator '*' (line 344)
        result_mul_607575 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 24), '*', float_607573, n_607574)
        
        # Assigning a type to the variable 'tuple_assignment_606752' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'tuple_assignment_606752', result_mul_607575)
        
        # Assigning a BinOp to a Name (line 344):
        float_607576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 30), 'float')
        # Getting the type of 'N' (line 344)
        N_607577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), 'N')
        # Applying the binary operator '*' (line 344)
        result_mul_607578 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 30), '*', float_607576, N_607577)
        
        # Assigning a type to the variable 'tuple_assignment_606753' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'tuple_assignment_606753', result_mul_607578)
        
        # Assigning a Name to a Name (line 344):
        # Getting the type of 'tuple_assignment_606751' (line 344)
        tuple_assignment_606751_607579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'tuple_assignment_606751')
        # Assigning a type to the variable 'M' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'M', tuple_assignment_606751_607579)
        
        # Assigning a Name to a Name (line 344):
        # Getting the type of 'tuple_assignment_606752' (line 344)
        tuple_assignment_606752_607580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'tuple_assignment_606752')
        # Assigning a type to the variable 'n' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 11), 'n', tuple_assignment_606752_607580)
        
        # Assigning a Name to a Name (line 344):
        # Getting the type of 'tuple_assignment_606753' (line 344)
        tuple_assignment_606753_607581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'tuple_assignment_606753')
        # Assigning a type to the variable 'N' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 14), 'N', tuple_assignment_606753_607581)
        
        # Assigning a BinOp to a Name (line 345):
        
        # Assigning a BinOp to a Name (line 345):
        # Getting the type of 'M' (line 345)
        M_607582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'M')
        # Getting the type of 'n' (line 345)
        n_607583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 16), 'n')
        # Applying the binary operator '-' (line 345)
        result_sub_607584 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 12), '-', M_607582, n_607583)
        
        # Assigning a type to the variable 'm' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'm', result_sub_607584)
        
        # Assigning a BinOp to a Name (line 346):
        
        # Assigning a BinOp to a Name (line 346):
        # Getting the type of 'n' (line 346)
        n_607585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'n')
        # Getting the type of 'M' (line 346)
        M_607586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 14), 'M')
        # Applying the binary operator 'div' (line 346)
        result_div_607587 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 12), 'div', n_607585, M_607586)
        
        # Assigning a type to the variable 'p' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'p', result_div_607587)
        
        # Assigning a BinOp to a Name (line 347):
        
        # Assigning a BinOp to a Name (line 347):
        # Getting the type of 'N' (line 347)
        N_607588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 13), 'N')
        # Getting the type of 'p' (line 347)
        p_607589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'p')
        # Applying the binary operator '*' (line 347)
        result_mul_607590 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 13), '*', N_607588, p_607589)
        
        # Assigning a type to the variable 'mu' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'mu', result_mul_607590)
        
        # Assigning a BinOp to a Name (line 349):
        
        # Assigning a BinOp to a Name (line 349):
        # Getting the type of 'm' (line 349)
        m_607591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 14), 'm')
        # Getting the type of 'n' (line 349)
        n_607592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'n')
        # Applying the binary operator '*' (line 349)
        result_mul_607593 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 14), '*', m_607591, n_607592)
        
        # Getting the type of 'N' (line 349)
        N_607594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 18), 'N')
        # Applying the binary operator '*' (line 349)
        result_mul_607595 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 17), '*', result_mul_607593, N_607594)
        
        # Getting the type of 'M' (line 349)
        M_607596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'M')
        # Getting the type of 'N' (line 349)
        N_607597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 25), 'N')
        # Applying the binary operator '-' (line 349)
        result_sub_607598 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 21), '-', M_607596, N_607597)
        
        # Applying the binary operator '*' (line 349)
        result_mul_607599 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 19), '*', result_mul_607595, result_sub_607598)
        
        float_607600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 28), 'float')
        # Applying the binary operator '*' (line 349)
        result_mul_607601 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 27), '*', result_mul_607599, float_607600)
        
        # Getting the type of 'M' (line 349)
        M_607602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 33), 'M')
        # Getting the type of 'M' (line 349)
        M_607603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 35), 'M')
        # Applying the binary operator '*' (line 349)
        result_mul_607604 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 33), '*', M_607602, M_607603)
        
        # Getting the type of 'M' (line 349)
        M_607605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 38), 'M')
        int_607606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 40), 'int')
        # Applying the binary operator '-' (line 349)
        result_sub_607607 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 38), '-', M_607605, int_607606)
        
        # Applying the binary operator '*' (line 349)
        result_mul_607608 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 36), '*', result_mul_607604, result_sub_607607)
        
        # Applying the binary operator 'div' (line 349)
        result_div_607609 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 31), 'div', result_mul_607601, result_mul_607608)
        
        # Assigning a type to the variable 'var' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'var', result_div_607609)
        
        # Assigning a BinOp to a Name (line 350):
        
        # Assigning a BinOp to a Name (line 350):
        # Getting the type of 'm' (line 350)
        m_607610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 14), 'm')
        # Getting the type of 'n' (line 350)
        n_607611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 18), 'n')
        # Applying the binary operator '-' (line 350)
        result_sub_607612 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 14), '-', m_607610, n_607611)
        
        # Getting the type of 'M' (line 350)
        M_607613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'M')
        int_607614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 24), 'int')
        # Getting the type of 'N' (line 350)
        N_607615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'N')
        # Applying the binary operator '*' (line 350)
        result_mul_607616 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 24), '*', int_607614, N_607615)
        
        # Applying the binary operator '-' (line 350)
        result_sub_607617 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 22), '-', M_607613, result_mul_607616)
        
        # Applying the binary operator '*' (line 350)
        result_mul_607618 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 13), '*', result_sub_607612, result_sub_607617)
        
        # Getting the type of 'M' (line 350)
        M_607619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 32), 'M')
        float_607620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 34), 'float')
        # Applying the binary operator '-' (line 350)
        result_sub_607621 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 32), '-', M_607619, float_607620)
        
        # Applying the binary operator 'div' (line 350)
        result_div_607622 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 29), 'div', result_mul_607618, result_sub_607621)
        
        
        # Call to sqrt(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'M' (line 350)
        M_607624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 47), 'M', False)
        float_607625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 49), 'float')
        # Applying the binary operator '-' (line 350)
        result_sub_607626 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 47), '-', M_607624, float_607625)
        
        # Getting the type of 'm' (line 350)
        m_607627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 57), 'm', False)
        # Getting the type of 'n' (line 350)
        n_607628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 59), 'n', False)
        # Applying the binary operator '*' (line 350)
        result_mul_607629 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 57), '*', m_607627, n_607628)
        
        # Getting the type of 'N' (line 350)
        N_607630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 61), 'N', False)
        # Applying the binary operator '*' (line 350)
        result_mul_607631 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 60), '*', result_mul_607629, N_607630)
        
        # Getting the type of 'M' (line 350)
        M_607632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 64), 'M', False)
        # Getting the type of 'N' (line 350)
        N_607633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 66), 'N', False)
        # Applying the binary operator '-' (line 350)
        result_sub_607634 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 64), '-', M_607632, N_607633)
        
        # Applying the binary operator '*' (line 350)
        result_mul_607635 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 62), '*', result_mul_607631, result_sub_607634)
        
        # Applying the binary operator 'div' (line 350)
        result_div_607636 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 46), 'div', result_sub_607626, result_mul_607635)
        
        # Processing the call keyword arguments (line 350)
        kwargs_607637 = {}
        # Getting the type of 'sqrt' (line 350)
        sqrt_607623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 350)
        sqrt_call_result_607638 = invoke(stypy.reporting.localization.Localization(__file__, 350, 41), sqrt_607623, *[result_div_607636], **kwargs_607637)
        
        # Applying the binary operator '*' (line 350)
        result_mul_607639 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 39), '*', result_div_607622, sqrt_call_result_607638)
        
        # Assigning a type to the variable 'g1' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'g1', result_mul_607639)
        
        # Assigning a BinOp to a Name (line 352):
        
        # Assigning a BinOp to a Name (line 352):
        # Getting the type of 'M' (line 352)
        M_607640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 13), 'M')
        # Getting the type of 'M' (line 352)
        M_607641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'M')
        int_607642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 18), 'int')
        # Applying the binary operator '+' (line 352)
        result_add_607643 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 16), '+', M_607641, int_607642)
        
        # Applying the binary operator '*' (line 352)
        result_mul_607644 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 13), '*', M_607640, result_add_607643)
        
        float_607645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 23), 'float')
        # Getting the type of 'N' (line 352)
        N_607646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 26), 'N')
        # Applying the binary operator '*' (line 352)
        result_mul_607647 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 23), '*', float_607645, N_607646)
        
        # Getting the type of 'M' (line 352)
        M_607648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 29), 'M')
        # Getting the type of 'N' (line 352)
        N_607649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 31), 'N')
        # Applying the binary operator '-' (line 352)
        result_sub_607650 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 29), '-', M_607648, N_607649)
        
        # Applying the binary operator '*' (line 352)
        result_mul_607651 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 27), '*', result_mul_607647, result_sub_607650)
        
        # Applying the binary operator '-' (line 352)
        result_sub_607652 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 13), '-', result_mul_607644, result_mul_607651)
        
        float_607653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 36), 'float')
        # Getting the type of 'n' (line 352)
        n_607654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 39), 'n')
        # Applying the binary operator '*' (line 352)
        result_mul_607655 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 36), '*', float_607653, n_607654)
        
        # Getting the type of 'm' (line 352)
        m_607656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 41), 'm')
        # Applying the binary operator '*' (line 352)
        result_mul_607657 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 40), '*', result_mul_607655, m_607656)
        
        # Applying the binary operator '-' (line 352)
        result_sub_607658 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 34), '-', result_sub_607652, result_mul_607657)
        
        # Assigning a type to the variable 'g2' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'g2', result_sub_607658)
        
        # Getting the type of 'g2' (line 353)
        g2_607659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'g2')
        # Getting the type of 'M' (line 353)
        M_607660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'M')
        int_607661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 17), 'int')
        # Applying the binary operator '-' (line 353)
        result_sub_607662 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 15), '-', M_607660, int_607661)
        
        # Getting the type of 'M' (line 353)
        M_607663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'M')
        # Applying the binary operator '*' (line 353)
        result_mul_607664 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 14), '*', result_sub_607662, M_607663)
        
        # Getting the type of 'M' (line 353)
        M_607665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 22), 'M')
        # Applying the binary operator '*' (line 353)
        result_mul_607666 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 21), '*', result_mul_607664, M_607665)
        
        # Applying the binary operator '*=' (line 353)
        result_imul_607667 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 8), '*=', g2_607659, result_mul_607666)
        # Assigning a type to the variable 'g2' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'g2', result_imul_607667)
        
        
        # Getting the type of 'g2' (line 354)
        g2_607668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'g2')
        float_607669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 14), 'float')
        # Getting the type of 'n' (line 354)
        n_607670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 17), 'n')
        # Applying the binary operator '*' (line 354)
        result_mul_607671 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 14), '*', float_607669, n_607670)
        
        # Getting the type of 'N' (line 354)
        N_607672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 19), 'N')
        # Applying the binary operator '*' (line 354)
        result_mul_607673 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 18), '*', result_mul_607671, N_607672)
        
        # Getting the type of 'M' (line 354)
        M_607674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 22), 'M')
        # Getting the type of 'N' (line 354)
        N_607675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'N')
        # Applying the binary operator '-' (line 354)
        result_sub_607676 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 22), '-', M_607674, N_607675)
        
        # Applying the binary operator '*' (line 354)
        result_mul_607677 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 20), '*', result_mul_607673, result_sub_607676)
        
        # Getting the type of 'm' (line 354)
        m_607678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'm')
        # Applying the binary operator '*' (line 354)
        result_mul_607679 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 26), '*', result_mul_607677, m_607678)
        
        float_607680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 30), 'float')
        # Getting the type of 'M' (line 354)
        M_607681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'M')
        # Applying the binary operator '*' (line 354)
        result_mul_607682 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 30), '*', float_607680, M_607681)
        
        int_607683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 35), 'int')
        # Applying the binary operator '-' (line 354)
        result_sub_607684 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 30), '-', result_mul_607682, int_607683)
        
        # Applying the binary operator '*' (line 354)
        result_mul_607685 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 28), '*', result_mul_607679, result_sub_607684)
        
        # Applying the binary operator '+=' (line 354)
        result_iadd_607686 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 8), '+=', g2_607668, result_mul_607685)
        # Assigning a type to the variable 'g2' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'g2', result_iadd_607686)
        
        
        # Getting the type of 'g2' (line 355)
        g2_607687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'g2')
        # Getting the type of 'n' (line 355)
        n_607688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'n')
        # Getting the type of 'N' (line 355)
        N_607689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 18), 'N')
        # Applying the binary operator '*' (line 355)
        result_mul_607690 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 14), '*', n_607688, N_607689)
        
        # Getting the type of 'M' (line 355)
        M_607691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 23), 'M')
        # Getting the type of 'N' (line 355)
        N_607692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 25), 'N')
        # Applying the binary operator '-' (line 355)
        result_sub_607693 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 23), '-', M_607691, N_607692)
        
        # Applying the binary operator '*' (line 355)
        result_mul_607694 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 20), '*', result_mul_607690, result_sub_607693)
        
        # Getting the type of 'm' (line 355)
        m_607695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'm')
        # Applying the binary operator '*' (line 355)
        result_mul_607696 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 28), '*', result_mul_607694, m_607695)
        
        # Getting the type of 'M' (line 355)
        M_607697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 35), 'M')
        float_607698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 37), 'float')
        # Applying the binary operator '-' (line 355)
        result_sub_607699 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 35), '-', M_607697, float_607698)
        
        # Applying the binary operator '*' (line 355)
        result_mul_607700 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 32), '*', result_mul_607696, result_sub_607699)
        
        # Getting the type of 'M' (line 355)
        M_607701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 44), 'M')
        float_607702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 46), 'float')
        # Applying the binary operator '-' (line 355)
        result_sub_607703 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 44), '-', M_607701, float_607702)
        
        # Applying the binary operator '*' (line 355)
        result_mul_607704 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 41), '*', result_mul_607700, result_sub_607703)
        
        # Applying the binary operator 'div=' (line 355)
        result_div_607705 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 8), 'div=', g2_607687, result_mul_607704)
        # Assigning a type to the variable 'g2' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'g2', result_div_607705)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 356)
        tuple_607706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 356)
        # Adding element type (line 356)
        # Getting the type of 'mu' (line 356)
        mu_607707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 15), tuple_607706, mu_607707)
        # Adding element type (line 356)
        # Getting the type of 'var' (line 356)
        var_607708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 15), tuple_607706, var_607708)
        # Adding element type (line 356)
        # Getting the type of 'g1' (line 356)
        g1_607709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 15), tuple_607706, g1_607709)
        # Adding element type (line 356)
        # Getting the type of 'g2' (line 356)
        g2_607710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 15), tuple_607706, g2_607710)
        
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'stypy_return_type', tuple_607706)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 341)
        stypy_return_type_607711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_607711


    @norecursion
    def _entropy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_entropy'
        module_type_store = module_type_store.open_function_context('_entropy', 358, 4, False)
        # Assigning a type to the variable 'self' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_localization', localization)
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_type_store', module_type_store)
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_function_name', 'hypergeom_gen._entropy')
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_param_names_list', ['M', 'n', 'N'])
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_varargs_param_name', None)
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_call_defaults', defaults)
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_call_varargs', varargs)
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        hypergeom_gen._entropy.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen._entropy', ['M', 'n', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_entropy', localization, ['M', 'n', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_entropy(...)' code ##################

        
        # Assigning a Subscript to a Name (line 359):
        
        # Assigning a Subscript to a Name (line 359):
        
        # Obtaining the type of the subscript
        # Getting the type of 'N' (line 359)
        N_607712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 18), 'N')
        # Getting the type of 'M' (line 359)
        M_607713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 23), 'M')
        # Getting the type of 'n' (line 359)
        n_607714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 27), 'n')
        # Applying the binary operator '-' (line 359)
        result_sub_607715 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 23), '-', M_607713, n_607714)
        
        # Applying the binary operator '-' (line 359)
        result_sub_607716 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 18), '-', N_607712, result_sub_607715)
        
        
        # Call to min(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'n' (line 359)
        n_607718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 34), 'n', False)
        # Getting the type of 'N' (line 359)
        N_607719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 37), 'N', False)
        # Processing the call keyword arguments (line 359)
        kwargs_607720 = {}
        # Getting the type of 'min' (line 359)
        min_607717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 30), 'min', False)
        # Calling min(args, kwargs) (line 359)
        min_call_result_607721 = invoke(stypy.reporting.localization.Localization(__file__, 359, 30), min_607717, *[n_607718, N_607719], **kwargs_607720)
        
        int_607722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 42), 'int')
        # Applying the binary operator '+' (line 359)
        result_add_607723 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 30), '+', min_call_result_607721, int_607722)
        
        slice_607724 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 359, 12), result_sub_607716, result_add_607723, None)
        # Getting the type of 'np' (line 359)
        np_607725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'np')
        # Obtaining the member 'r_' of a type (line 359)
        r__607726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), np_607725, 'r_')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___607727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), r__607726, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_607728 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), getitem___607727, slice_607724)
        
        # Assigning a type to the variable 'k' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'k', subscript_call_result_607728)
        
        # Assigning a Call to a Name (line 360):
        
        # Assigning a Call to a Name (line 360):
        
        # Call to pmf(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'k' (line 360)
        k_607731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 24), 'k', False)
        # Getting the type of 'M' (line 360)
        M_607732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 27), 'M', False)
        # Getting the type of 'n' (line 360)
        n_607733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 30), 'n', False)
        # Getting the type of 'N' (line 360)
        N_607734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 33), 'N', False)
        # Processing the call keyword arguments (line 360)
        kwargs_607735 = {}
        # Getting the type of 'self' (line 360)
        self_607729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'self', False)
        # Obtaining the member 'pmf' of a type (line 360)
        pmf_607730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 15), self_607729, 'pmf')
        # Calling pmf(args, kwargs) (line 360)
        pmf_call_result_607736 = invoke(stypy.reporting.localization.Localization(__file__, 360, 15), pmf_607730, *[k_607731, M_607732, n_607733, N_607734], **kwargs_607735)
        
        # Assigning a type to the variable 'vals' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'vals', pmf_call_result_607736)
        
        # Call to sum(...): (line 361)
        # Processing the call arguments (line 361)
        
        # Call to entr(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'vals' (line 361)
        vals_607740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'vals', False)
        # Processing the call keyword arguments (line 361)
        kwargs_607741 = {}
        # Getting the type of 'entr' (line 361)
        entr_607739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 22), 'entr', False)
        # Calling entr(args, kwargs) (line 361)
        entr_call_result_607742 = invoke(stypy.reporting.localization.Localization(__file__, 361, 22), entr_607739, *[vals_607740], **kwargs_607741)
        
        # Processing the call keyword arguments (line 361)
        int_607743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 39), 'int')
        keyword_607744 = int_607743
        kwargs_607745 = {'axis': keyword_607744}
        # Getting the type of 'np' (line 361)
        np_607737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 15), 'np', False)
        # Obtaining the member 'sum' of a type (line 361)
        sum_607738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 15), np_607737, 'sum')
        # Calling sum(args, kwargs) (line 361)
        sum_call_result_607746 = invoke(stypy.reporting.localization.Localization(__file__, 361, 15), sum_607738, *[entr_call_result_607742], **kwargs_607745)
        
        # Assigning a type to the variable 'stypy_return_type' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'stypy_return_type', sum_call_result_607746)
        
        # ################# End of '_entropy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_entropy' in the type store
        # Getting the type of 'stypy_return_type' (line 358)
        stypy_return_type_607747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_entropy'
        return stypy_return_type_607747


    @norecursion
    def _sf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sf'
        module_type_store = module_type_store.open_function_context('_sf', 363, 4, False)
        # Assigning a type to the variable 'self' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        hypergeom_gen._sf.__dict__.__setitem__('stypy_localization', localization)
        hypergeom_gen._sf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        hypergeom_gen._sf.__dict__.__setitem__('stypy_type_store', module_type_store)
        hypergeom_gen._sf.__dict__.__setitem__('stypy_function_name', 'hypergeom_gen._sf')
        hypergeom_gen._sf.__dict__.__setitem__('stypy_param_names_list', ['k', 'M', 'n', 'N'])
        hypergeom_gen._sf.__dict__.__setitem__('stypy_varargs_param_name', None)
        hypergeom_gen._sf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        hypergeom_gen._sf.__dict__.__setitem__('stypy_call_defaults', defaults)
        hypergeom_gen._sf.__dict__.__setitem__('stypy_call_varargs', varargs)
        hypergeom_gen._sf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        hypergeom_gen._sf.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen._sf', ['k', 'M', 'n', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sf', localization, ['k', 'M', 'n', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sf(...)' code ##################

        str_607748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 8), 'str', "More precise calculation, 1 - cdf doesn't cut it.")
        
        # Assigning a List to a Name (line 369):
        
        # Assigning a List to a Name (line 369):
        
        # Obtaining an instance of the builtin type 'list' (line 369)
        list_607749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 369)
        
        # Assigning a type to the variable 'res' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'res', list_607749)
        
        
        # Call to zip(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'k' (line 370)
        k_607751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 42), 'k', False)
        # Getting the type of 'M' (line 370)
        M_607752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 45), 'M', False)
        # Getting the type of 'n' (line 370)
        n_607753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 48), 'n', False)
        # Getting the type of 'N' (line 370)
        N_607754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 51), 'N', False)
        # Processing the call keyword arguments (line 370)
        kwargs_607755 = {}
        # Getting the type of 'zip' (line 370)
        zip_607750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 38), 'zip', False)
        # Calling zip(args, kwargs) (line 370)
        zip_call_result_607756 = invoke(stypy.reporting.localization.Localization(__file__, 370, 38), zip_607750, *[k_607751, M_607752, n_607753, N_607754], **kwargs_607755)
        
        # Testing the type of a for loop iterable (line 370)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 370, 8), zip_call_result_607756)
        # Getting the type of the for loop variable (line 370)
        for_loop_var_607757 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 370, 8), zip_call_result_607756)
        # Assigning a type to the variable 'quant' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'quant', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 8), for_loop_var_607757))
        # Assigning a type to the variable 'tot' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'tot', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 8), for_loop_var_607757))
        # Assigning a type to the variable 'good' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'good', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 8), for_loop_var_607757))
        # Assigning a type to the variable 'draw' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'draw', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 8), for_loop_var_607757))
        # SSA begins for a for statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 373):
        
        # Assigning a Call to a Name (line 373):
        
        # Call to arange(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'quant' (line 373)
        quant_607760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 27), 'quant', False)
        int_607761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 35), 'int')
        # Applying the binary operator '+' (line 373)
        result_add_607762 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 27), '+', quant_607760, int_607761)
        
        # Getting the type of 'draw' (line 373)
        draw_607763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 38), 'draw', False)
        int_607764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 45), 'int')
        # Applying the binary operator '+' (line 373)
        result_add_607765 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 38), '+', draw_607763, int_607764)
        
        # Processing the call keyword arguments (line 373)
        kwargs_607766 = {}
        # Getting the type of 'np' (line 373)
        np_607758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 373)
        arange_607759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 17), np_607758, 'arange')
        # Calling arange(args, kwargs) (line 373)
        arange_call_result_607767 = invoke(stypy.reporting.localization.Localization(__file__, 373, 17), arange_607759, *[result_add_607762, result_add_607765], **kwargs_607766)
        
        # Assigning a type to the variable 'k2' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'k2', arange_call_result_607767)
        
        # Call to append(...): (line 374)
        # Processing the call arguments (line 374)
        
        # Call to sum(...): (line 374)
        # Processing the call arguments (line 374)
        
        # Call to _pmf(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'k2' (line 374)
        k2_607774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 40), 'k2', False)
        # Getting the type of 'tot' (line 374)
        tot_607775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 44), 'tot', False)
        # Getting the type of 'good' (line 374)
        good_607776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 49), 'good', False)
        # Getting the type of 'draw' (line 374)
        draw_607777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 55), 'draw', False)
        # Processing the call keyword arguments (line 374)
        kwargs_607778 = {}
        # Getting the type of 'self' (line 374)
        self_607772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 30), 'self', False)
        # Obtaining the member '_pmf' of a type (line 374)
        _pmf_607773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 30), self_607772, '_pmf')
        # Calling _pmf(args, kwargs) (line 374)
        _pmf_call_result_607779 = invoke(stypy.reporting.localization.Localization(__file__, 374, 30), _pmf_607773, *[k2_607774, tot_607775, good_607776, draw_607777], **kwargs_607778)
        
        # Processing the call keyword arguments (line 374)
        kwargs_607780 = {}
        # Getting the type of 'np' (line 374)
        np_607770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'np', False)
        # Obtaining the member 'sum' of a type (line 374)
        sum_607771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), np_607770, 'sum')
        # Calling sum(args, kwargs) (line 374)
        sum_call_result_607781 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), sum_607771, *[_pmf_call_result_607779], **kwargs_607780)
        
        # Processing the call keyword arguments (line 374)
        kwargs_607782 = {}
        # Getting the type of 'res' (line 374)
        res_607768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'res', False)
        # Obtaining the member 'append' of a type (line 374)
        append_607769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), res_607768, 'append')
        # Calling append(args, kwargs) (line 374)
        append_call_result_607783 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), append_607769, *[sum_call_result_607781], **kwargs_607782)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to asarray(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'res' (line 375)
        res_607786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 26), 'res', False)
        # Processing the call keyword arguments (line 375)
        kwargs_607787 = {}
        # Getting the type of 'np' (line 375)
        np_607784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'np', False)
        # Obtaining the member 'asarray' of a type (line 375)
        asarray_607785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 15), np_607784, 'asarray')
        # Calling asarray(args, kwargs) (line 375)
        asarray_call_result_607788 = invoke(stypy.reporting.localization.Localization(__file__, 375, 15), asarray_607785, *[res_607786], **kwargs_607787)
        
        # Assigning a type to the variable 'stypy_return_type' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'stypy_return_type', asarray_call_result_607788)
        
        # ################# End of '_sf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sf' in the type store
        # Getting the type of 'stypy_return_type' (line 363)
        stypy_return_type_607789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sf'
        return stypy_return_type_607789


    @norecursion
    def _logsf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logsf'
        module_type_store = module_type_store.open_function_context('_logsf', 377, 4, False)
        # Assigning a type to the variable 'self' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_localization', localization)
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_type_store', module_type_store)
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_function_name', 'hypergeom_gen._logsf')
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_param_names_list', ['k', 'M', 'n', 'N'])
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_varargs_param_name', None)
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_call_defaults', defaults)
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_call_varargs', varargs)
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        hypergeom_gen._logsf.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen._logsf', ['k', 'M', 'n', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logsf', localization, ['k', 'M', 'n', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logsf(...)' code ##################

        str_607790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, (-1)), 'str', '\n        More precise calculation than log(sf)\n        ')
        
        # Assigning a List to a Name (line 381):
        
        # Assigning a List to a Name (line 381):
        
        # Obtaining an instance of the builtin type 'list' (line 381)
        list_607791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 381)
        
        # Assigning a type to the variable 'res' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'res', list_607791)
        
        
        # Call to zip(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'k' (line 382)
        k_607793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 42), 'k', False)
        # Getting the type of 'M' (line 382)
        M_607794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 45), 'M', False)
        # Getting the type of 'n' (line 382)
        n_607795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 48), 'n', False)
        # Getting the type of 'N' (line 382)
        N_607796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 51), 'N', False)
        # Processing the call keyword arguments (line 382)
        kwargs_607797 = {}
        # Getting the type of 'zip' (line 382)
        zip_607792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 38), 'zip', False)
        # Calling zip(args, kwargs) (line 382)
        zip_call_result_607798 = invoke(stypy.reporting.localization.Localization(__file__, 382, 38), zip_607792, *[k_607793, M_607794, n_607795, N_607796], **kwargs_607797)
        
        # Testing the type of a for loop iterable (line 382)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 382, 8), zip_call_result_607798)
        # Getting the type of the for loop variable (line 382)
        for_loop_var_607799 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 382, 8), zip_call_result_607798)
        # Assigning a type to the variable 'quant' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'quant', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 8), for_loop_var_607799))
        # Assigning a type to the variable 'tot' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tot', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 8), for_loop_var_607799))
        # Assigning a type to the variable 'good' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'good', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 8), for_loop_var_607799))
        # Assigning a type to the variable 'draw' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'draw', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 8), for_loop_var_607799))
        # SSA begins for a for statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 384):
        
        # Assigning a Call to a Name (line 384):
        
        # Call to arange(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'quant' (line 384)
        quant_607802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'quant', False)
        int_607803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 35), 'int')
        # Applying the binary operator '+' (line 384)
        result_add_607804 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 27), '+', quant_607802, int_607803)
        
        # Getting the type of 'draw' (line 384)
        draw_607805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 38), 'draw', False)
        int_607806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 45), 'int')
        # Applying the binary operator '+' (line 384)
        result_add_607807 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 38), '+', draw_607805, int_607806)
        
        # Processing the call keyword arguments (line 384)
        kwargs_607808 = {}
        # Getting the type of 'np' (line 384)
        np_607800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 384)
        arange_607801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 17), np_607800, 'arange')
        # Calling arange(args, kwargs) (line 384)
        arange_call_result_607809 = invoke(stypy.reporting.localization.Localization(__file__, 384, 17), arange_607801, *[result_add_607804, result_add_607807], **kwargs_607808)
        
        # Assigning a type to the variable 'k2' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'k2', arange_call_result_607809)
        
        # Call to append(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Call to logsumexp(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Call to _logpmf(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'k2' (line 385)
        k2_607815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 46), 'k2', False)
        # Getting the type of 'tot' (line 385)
        tot_607816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 50), 'tot', False)
        # Getting the type of 'good' (line 385)
        good_607817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 55), 'good', False)
        # Getting the type of 'draw' (line 385)
        draw_607818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 61), 'draw', False)
        # Processing the call keyword arguments (line 385)
        kwargs_607819 = {}
        # Getting the type of 'self' (line 385)
        self_607813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 33), 'self', False)
        # Obtaining the member '_logpmf' of a type (line 385)
        _logpmf_607814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 33), self_607813, '_logpmf')
        # Calling _logpmf(args, kwargs) (line 385)
        _logpmf_call_result_607820 = invoke(stypy.reporting.localization.Localization(__file__, 385, 33), _logpmf_607814, *[k2_607815, tot_607816, good_607817, draw_607818], **kwargs_607819)
        
        # Processing the call keyword arguments (line 385)
        kwargs_607821 = {}
        # Getting the type of 'logsumexp' (line 385)
        logsumexp_607812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'logsumexp', False)
        # Calling logsumexp(args, kwargs) (line 385)
        logsumexp_call_result_607822 = invoke(stypy.reporting.localization.Localization(__file__, 385, 23), logsumexp_607812, *[_logpmf_call_result_607820], **kwargs_607821)
        
        # Processing the call keyword arguments (line 385)
        kwargs_607823 = {}
        # Getting the type of 'res' (line 385)
        res_607810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'res', False)
        # Obtaining the member 'append' of a type (line 385)
        append_607811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 12), res_607810, 'append')
        # Calling append(args, kwargs) (line 385)
        append_call_result_607824 = invoke(stypy.reporting.localization.Localization(__file__, 385, 12), append_607811, *[logsumexp_call_result_607822], **kwargs_607823)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to asarray(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'res' (line 386)
        res_607827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 26), 'res', False)
        # Processing the call keyword arguments (line 386)
        kwargs_607828 = {}
        # Getting the type of 'np' (line 386)
        np_607825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'np', False)
        # Obtaining the member 'asarray' of a type (line 386)
        asarray_607826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 15), np_607825, 'asarray')
        # Calling asarray(args, kwargs) (line 386)
        asarray_call_result_607829 = invoke(stypy.reporting.localization.Localization(__file__, 386, 15), asarray_607826, *[res_607827], **kwargs_607828)
        
        # Assigning a type to the variable 'stypy_return_type' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'stypy_return_type', asarray_call_result_607829)
        
        # ################# End of '_logsf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logsf' in the type store
        # Getting the type of 'stypy_return_type' (line 377)
        stypy_return_type_607830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607830)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logsf'
        return stypy_return_type_607830


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 258, 0, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'hypergeom_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'hypergeom_gen' (line 258)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), 'hypergeom_gen', hypergeom_gen)

# Assigning a Call to a Name (line 387):

# Assigning a Call to a Name (line 387):

# Call to hypergeom_gen(...): (line 387)
# Processing the call keyword arguments (line 387)
str_607832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 31), 'str', 'hypergeom')
keyword_607833 = str_607832
kwargs_607834 = {'name': keyword_607833}
# Getting the type of 'hypergeom_gen' (line 387)
hypergeom_gen_607831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'hypergeom_gen', False)
# Calling hypergeom_gen(args, kwargs) (line 387)
hypergeom_gen_call_result_607835 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), hypergeom_gen_607831, *[], **kwargs_607834)

# Assigning a type to the variable 'hypergeom' (line 387)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 0), 'hypergeom', hypergeom_gen_call_result_607835)
# Declaration of the 'logser_gen' class
# Getting the type of 'rv_discrete' (line 391)
rv_discrete_607836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'rv_discrete')

class logser_gen(rv_discrete_607836, ):
    str_607837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, (-1)), 'str', 'A Logarithmic (Log-Series, Series) discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `logser` is::\n\n        logser.pmf(k) = - p**k / (k*log(1-p))\n\n    for ``k >= 1``.\n\n    `logser` takes ``p`` as shape parameter.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 411, 4, False)
        # Assigning a type to the variable 'self' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        logser_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        logser_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        logser_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        logser_gen._rvs.__dict__.__setitem__('stypy_function_name', 'logser_gen._rvs')
        logser_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['p'])
        logser_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        logser_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        logser_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        logser_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        logser_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        logser_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'logser_gen._rvs', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Call to logseries(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'p' (line 414)
        p_607841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 44), 'p', False)
        # Processing the call keyword arguments (line 414)
        # Getting the type of 'self' (line 414)
        self_607842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 52), 'self', False)
        # Obtaining the member '_size' of a type (line 414)
        _size_607843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 52), self_607842, '_size')
        keyword_607844 = _size_607843
        kwargs_607845 = {'size': keyword_607844}
        # Getting the type of 'self' (line 414)
        self_607838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'self', False)
        # Obtaining the member '_random_state' of a type (line 414)
        _random_state_607839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 15), self_607838, '_random_state')
        # Obtaining the member 'logseries' of a type (line 414)
        logseries_607840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 15), _random_state_607839, 'logseries')
        # Calling logseries(args, kwargs) (line 414)
        logseries_call_result_607846 = invoke(stypy.reporting.localization.Localization(__file__, 414, 15), logseries_607840, *[p_607841], **kwargs_607845)
        
        # Assigning a type to the variable 'stypy_return_type' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'stypy_return_type', logseries_call_result_607846)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 411)
        stypy_return_type_607847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_607847


    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 416, 4, False)
        # Assigning a type to the variable 'self' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        logser_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        logser_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        logser_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        logser_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'logser_gen._argcheck')
        logser_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['p'])
        logser_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        logser_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        logser_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        logser_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        logser_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        logser_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'logser_gen._argcheck', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Getting the type of 'p' (line 417)
        p_607848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 16), 'p')
        int_607849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 20), 'int')
        # Applying the binary operator '>' (line 417)
        result_gt_607850 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 16), '>', p_607848, int_607849)
        
        
        # Getting the type of 'p' (line 417)
        p_607851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 26), 'p')
        int_607852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 30), 'int')
        # Applying the binary operator '<' (line 417)
        result_lt_607853 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 26), '<', p_607851, int_607852)
        
        # Applying the binary operator '&' (line 417)
        result_and__607854 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 15), '&', result_gt_607850, result_lt_607853)
        
        # Assigning a type to the variable 'stypy_return_type' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'stypy_return_type', result_and__607854)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 416)
        stypy_return_type_607855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607855)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_607855


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 419, 4, False)
        # Assigning a type to the variable 'self' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        logser_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        logser_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        logser_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        logser_gen._pmf.__dict__.__setitem__('stypy_function_name', 'logser_gen._pmf')
        logser_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'p'])
        logser_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        logser_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        logser_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        logser_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        logser_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        logser_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'logser_gen._pmf', ['k', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        
        # Call to power(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'p' (line 420)
        p_607858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 25), 'p', False)
        # Getting the type of 'k' (line 420)
        k_607859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 28), 'k', False)
        # Processing the call keyword arguments (line 420)
        kwargs_607860 = {}
        # Getting the type of 'np' (line 420)
        np_607856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'np', False)
        # Obtaining the member 'power' of a type (line 420)
        power_607857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 16), np_607856, 'power')
        # Calling power(args, kwargs) (line 420)
        power_call_result_607861 = invoke(stypy.reporting.localization.Localization(__file__, 420, 16), power_607857, *[p_607858, k_607859], **kwargs_607860)
        
        # Applying the 'usub' unary operator (line 420)
        result___neg___607862 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 15), 'usub', power_call_result_607861)
        
        float_607863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 33), 'float')
        # Applying the binary operator '*' (line 420)
        result_mul_607864 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 15), '*', result___neg___607862, float_607863)
        
        # Getting the type of 'k' (line 420)
        k_607865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 39), 'k')
        # Applying the binary operator 'div' (line 420)
        result_div_607866 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 37), 'div', result_mul_607864, k_607865)
        
        
        # Call to log1p(...): (line 420)
        # Processing the call arguments (line 420)
        
        # Getting the type of 'p' (line 420)
        p_607869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 58), 'p', False)
        # Applying the 'usub' unary operator (line 420)
        result___neg___607870 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 57), 'usub', p_607869)
        
        # Processing the call keyword arguments (line 420)
        kwargs_607871 = {}
        # Getting the type of 'special' (line 420)
        special_607867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 43), 'special', False)
        # Obtaining the member 'log1p' of a type (line 420)
        log1p_607868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 43), special_607867, 'log1p')
        # Calling log1p(args, kwargs) (line 420)
        log1p_call_result_607872 = invoke(stypy.reporting.localization.Localization(__file__, 420, 43), log1p_607868, *[result___neg___607870], **kwargs_607871)
        
        # Applying the binary operator 'div' (line 420)
        result_div_607873 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 41), 'div', result_div_607866, log1p_call_result_607872)
        
        # Assigning a type to the variable 'stypy_return_type' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'stypy_return_type', result_div_607873)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 419)
        stypy_return_type_607874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_607874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_607874


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        logser_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        logser_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        logser_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        logser_gen._stats.__dict__.__setitem__('stypy_function_name', 'logser_gen._stats')
        logser_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['p'])
        logser_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        logser_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        logser_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        logser_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        logser_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        logser_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'logser_gen._stats', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a Call to a Name (line 423):
        
        # Assigning a Call to a Name (line 423):
        
        # Call to log1p(...): (line 423)
        # Processing the call arguments (line 423)
        
        # Getting the type of 'p' (line 423)
        p_607877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 27), 'p', False)
        # Applying the 'usub' unary operator (line 423)
        result___neg___607878 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 26), 'usub', p_607877)
        
        # Processing the call keyword arguments (line 423)
        kwargs_607879 = {}
        # Getting the type of 'special' (line 423)
        special_607875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'special', False)
        # Obtaining the member 'log1p' of a type (line 423)
        log1p_607876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), special_607875, 'log1p')
        # Calling log1p(args, kwargs) (line 423)
        log1p_call_result_607880 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), log1p_607876, *[result___neg___607878], **kwargs_607879)
        
        # Assigning a type to the variable 'r' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'r', log1p_call_result_607880)
        
        # Assigning a BinOp to a Name (line 424):
        
        # Assigning a BinOp to a Name (line 424):
        # Getting the type of 'p' (line 424)
        p_607881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 13), 'p')
        # Getting the type of 'p' (line 424)
        p_607882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 18), 'p')
        float_607883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 22), 'float')
        # Applying the binary operator '-' (line 424)
        result_sub_607884 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 18), '-', p_607882, float_607883)
        
        # Applying the binary operator 'div' (line 424)
        result_div_607885 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 13), 'div', p_607881, result_sub_607884)
        
        # Getting the type of 'r' (line 424)
        r_607886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 29), 'r')
        # Applying the binary operator 'div' (line 424)
        result_div_607887 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 27), 'div', result_div_607885, r_607886)
        
        # Assigning a type to the variable 'mu' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'mu', result_div_607887)
        
        # Assigning a BinOp to a Name (line 425):
        
        # Assigning a BinOp to a Name (line 425):
        
        # Getting the type of 'p' (line 425)
        p_607888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'p')
        # Applying the 'usub' unary operator (line 425)
        result___neg___607889 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 15), 'usub', p_607888)
        
        # Getting the type of 'r' (line 425)
        r_607890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'r')
        # Applying the binary operator 'div' (line 425)
        result_div_607891 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 15), 'div', result___neg___607889, r_607890)
        
        # Getting the type of 'p' (line 425)
        p_607892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 25), 'p')
        float_607893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 29), 'float')
        # Applying the binary operator '-' (line 425)
        result_sub_607894 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 25), '-', p_607892, float_607893)
        
        int_607895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 35), 'int')
        # Applying the binary operator '**' (line 425)
        result_pow_607896 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 24), '**', result_sub_607894, int_607895)
        
        # Applying the binary operator 'div' (line 425)
        result_div_607897 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 22), 'div', result_div_607891, result_pow_607896)
        
        # Assigning a type to the variable 'mu2p' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'mu2p', result_div_607897)
        
        # Assigning a BinOp to a Name (line 426):
        
        # Assigning a BinOp to a Name (line 426):
        # Getting the type of 'mu2p' (line 426)
        mu2p_607898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 14), 'mu2p')
        # Getting the type of 'mu' (line 426)
        mu_607899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 21), 'mu')
        # Getting the type of 'mu' (line 426)
        mu_607900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 24), 'mu')
        # Applying the binary operator '*' (line 426)
        result_mul_607901 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 21), '*', mu_607899, mu_607900)
        
        # Applying the binary operator '-' (line 426)
        result_sub_607902 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 14), '-', mu2p_607898, result_mul_607901)
        
        # Assigning a type to the variable 'var' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'var', result_sub_607902)
        
        # Assigning a BinOp to a Name (line 427):
        
        # Assigning a BinOp to a Name (line 427):
        
        # Getting the type of 'p' (line 427)
        p_607903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'p')
        # Applying the 'usub' unary operator (line 427)
        result___neg___607904 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 15), 'usub', p_607903)
        
        # Getting the type of 'r' (line 427)
        r_607905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 20), 'r')
        # Applying the binary operator 'div' (line 427)
        result_div_607906 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 15), 'div', result___neg___607904, r_607905)
        
        float_607907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 25), 'float')
        # Getting the type of 'p' (line 427)
        p_607908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 29), 'p')
        # Applying the binary operator '+' (line 427)
        result_add_607909 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 25), '+', float_607907, p_607908)
        
        # Applying the binary operator '*' (line 427)
        result_mul_607910 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 22), '*', result_div_607906, result_add_607909)
        
        float_607911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 35), 'float')
        # Getting the type of 'p' (line 427)
        p_607912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 41), 'p')
        # Applying the binary operator '-' (line 427)
        result_sub_607913 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 35), '-', float_607911, p_607912)
        
        int_607914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 45), 'int')
        # Applying the binary operator '**' (line 427)
        result_pow_607915 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 34), '**', result_sub_607913, int_607914)
        
        # Applying the binary operator 'div' (line 427)
        result_div_607916 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 32), 'div', result_mul_607910, result_pow_607915)
        
        # Assigning a type to the variable 'mu3p' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'mu3p', result_div_607916)
        
        # Assigning a BinOp to a Name (line 428):
        
        # Assigning a BinOp to a Name (line 428):
        # Getting the type of 'mu3p' (line 428)
        mu3p_607917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 14), 'mu3p')
        int_607918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 21), 'int')
        # Getting the type of 'mu' (line 428)
        mu_607919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 23), 'mu')
        # Applying the binary operator '*' (line 428)
        result_mul_607920 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 21), '*', int_607918, mu_607919)
        
        # Getting the type of 'mu2p' (line 428)
        mu2p_607921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'mu2p')
        # Applying the binary operator '*' (line 428)
        result_mul_607922 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 25), '*', result_mul_607920, mu2p_607921)
        
        # Applying the binary operator '-' (line 428)
        result_sub_607923 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 14), '-', mu3p_607917, result_mul_607922)
        
        int_607924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 33), 'int')
        # Getting the type of 'mu' (line 428)
        mu_607925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 35), 'mu')
        int_607926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 39), 'int')
        # Applying the binary operator '**' (line 428)
        result_pow_607927 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 35), '**', mu_607925, int_607926)
        
        # Applying the binary operator '*' (line 428)
        result_mul_607928 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 33), '*', int_607924, result_pow_607927)
        
        # Applying the binary operator '+' (line 428)
        result_add_607929 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 31), '+', result_sub_607923, result_mul_607928)
        
        # Assigning a type to the variable 'mu3' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'mu3', result_add_607929)
        
        # Assigning a BinOp to a Name (line 429):
        
        # Assigning a BinOp to a Name (line 429):
        # Getting the type of 'mu3' (line 429)
        mu3_607930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 13), 'mu3')
        
        # Call to power(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'var' (line 429)
        var_607933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 28), 'var', False)
        float_607934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 33), 'float')
        # Processing the call keyword arguments (line 429)
        kwargs_607935 = {}
        # Getting the type of 'np' (line 429)
        np_607931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 19), 'np', False)
        # Obtaining the member 'power' of a type (line 429)
        power_607932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 19), np_607931, 'power')
        # Calling power(args, kwargs) (line 429)
        power_call_result_607936 = invoke(stypy.reporting.localization.Localization(__file__, 429, 19), power_607932, *[var_607933, float_607934], **kwargs_607935)
        
        # Applying the binary operator 'div' (line 429)
        result_div_607937 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 13), 'div', mu3_607930, power_call_result_607936)
        
        # Assigning a type to the variable 'g1' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'g1', result_div_607937)
        
        # Assigning a BinOp to a Name (line 431):
        
        # Assigning a BinOp to a Name (line 431):
        
        # Getting the type of 'p' (line 431)
        p_607938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'p')
        # Applying the 'usub' unary operator (line 431)
        result___neg___607939 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 15), 'usub', p_607938)
        
        # Getting the type of 'r' (line 431)
        r_607940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 20), 'r')
        # Applying the binary operator 'div' (line 431)
        result_div_607941 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 15), 'div', result___neg___607939, r_607940)
        
        float_607942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 12), 'float')
        # Getting the type of 'p' (line 432)
        p_607943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 19), 'p')
        int_607944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 21), 'int')
        # Applying the binary operator '-' (line 432)
        result_sub_607945 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 19), '-', p_607943, int_607944)
        
        int_607946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 25), 'int')
        # Applying the binary operator '**' (line 432)
        result_pow_607947 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 18), '**', result_sub_607945, int_607946)
        
        # Applying the binary operator 'div' (line 432)
        result_div_607948 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 12), 'div', float_607942, result_pow_607947)
        
        int_607949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 29), 'int')
        # Getting the type of 'p' (line 432)
        p_607950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 31), 'p')
        # Applying the binary operator '*' (line 432)
        result_mul_607951 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 29), '*', int_607949, p_607950)
        
        # Getting the type of 'p' (line 432)
        p_607952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 36), 'p')
        int_607953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 40), 'int')
        # Applying the binary operator '-' (line 432)
        result_sub_607954 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 36), '-', p_607952, int_607953)
        
        int_607955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 44), 'int')
        # Applying the binary operator '**' (line 432)
        result_pow_607956 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 35), '**', result_sub_607954, int_607955)
        
        # Applying the binary operator 'div' (line 432)
        result_div_607957 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 33), 'div', result_mul_607951, result_pow_607956)
        
        # Applying the binary operator '-' (line 432)
        result_sub_607958 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 12), '-', result_div_607948, result_div_607957)
        
        int_607959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 48), 'int')
        # Getting the type of 'p' (line 432)
        p_607960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'p')
        # Applying the binary operator '*' (line 432)
        result_mul_607961 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 48), '*', int_607959, p_607960)
        
        # Getting the type of 'p' (line 432)
        p_607962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 52), 'p')
        # Applying the binary operator '*' (line 432)
        result_mul_607963 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 51), '*', result_mul_607961, p_607962)
        
        # Getting the type of 'p' (line 432)
        p_607964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 57), 'p')
        int_607965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 59), 'int')
        # Applying the binary operator '-' (line 432)
        result_sub_607966 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 57), '-', p_607964, int_607965)
        
        int_607967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 63), 'int')
        # Applying the binary operator '**' (line 432)
        result_pow_607968 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 56), '**', result_sub_607966, int_607967)
        
        # Applying the binary operator 'div' (line 432)
        result_div_607969 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 54), 'div', result_mul_607963, result_pow_607968)
        
        # Applying the binary operator '+' (line 432)
        result_add_607970 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 46), '+', result_sub_607958, result_div_607969)
        
        # Applying the binary operator '*' (line 431)
        result_mul_607971 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 22), '*', result_div_607941, result_add_607970)
        
        # Assigning a type to the variable 'mu4p' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'mu4p', result_mul_607971)
        
        # Assigning a BinOp to a Name (line 433):
        
        # Assigning a BinOp to a Name (line 433):
        # Getting the type of 'mu4p' (line 433)
        mu4p_607972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 14), 'mu4p')
        int_607973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 21), 'int')
        # Getting the type of 'mu3p' (line 433)
        mu3p_607974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 23), 'mu3p')
        # Applying the binary operator '*' (line 433)
        result_mul_607975 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 21), '*', int_607973, mu3p_607974)
        
        # Getting the type of 'mu' (line 433)
        mu_607976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 28), 'mu')
        # Applying the binary operator '*' (line 433)
        result_mul_607977 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 27), '*', result_mul_607975, mu_607976)
        
        # Applying the binary operator '-' (line 433)
        result_sub_607978 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 14), '-', mu4p_607972, result_mul_607977)
        
        int_607979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 33), 'int')
        # Getting the type of 'mu2p' (line 433)
        mu2p_607980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 35), 'mu2p')
        # Applying the binary operator '*' (line 433)
        result_mul_607981 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 33), '*', int_607979, mu2p_607980)
        
        # Getting the type of 'mu' (line 433)
        mu_607982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 40), 'mu')
        # Applying the binary operator '*' (line 433)
        result_mul_607983 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 39), '*', result_mul_607981, mu_607982)
        
        # Getting the type of 'mu' (line 433)
        mu_607984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 43), 'mu')
        # Applying the binary operator '*' (line 433)
        result_mul_607985 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 42), '*', result_mul_607983, mu_607984)
        
        # Applying the binary operator '+' (line 433)
        result_add_607986 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 31), '+', result_sub_607978, result_mul_607985)
        
        int_607987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 48), 'int')
        # Getting the type of 'mu' (line 433)
        mu_607988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 50), 'mu')
        int_607989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 54), 'int')
        # Applying the binary operator '**' (line 433)
        result_pow_607990 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 50), '**', mu_607988, int_607989)
        
        # Applying the binary operator '*' (line 433)
        result_mul_607991 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 48), '*', int_607987, result_pow_607990)
        
        # Applying the binary operator '-' (line 433)
        result_sub_607992 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 46), '-', result_add_607986, result_mul_607991)
        
        # Assigning a type to the variable 'mu4' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'mu4', result_sub_607992)
        
        # Assigning a BinOp to a Name (line 434):
        
        # Assigning a BinOp to a Name (line 434):
        # Getting the type of 'mu4' (line 434)
        mu4_607993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 13), 'mu4')
        # Getting the type of 'var' (line 434)
        var_607994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'var')
        int_607995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 24), 'int')
        # Applying the binary operator '**' (line 434)
        result_pow_607996 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 19), '**', var_607994, int_607995)
        
        # Applying the binary operator 'div' (line 434)
        result_div_607997 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 13), 'div', mu4_607993, result_pow_607996)
        
        float_607998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 28), 'float')
        # Applying the binary operator '-' (line 434)
        result_sub_607999 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 13), '-', result_div_607997, float_607998)
        
        # Assigning a type to the variable 'g2' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'g2', result_sub_607999)
        
        # Obtaining an instance of the builtin type 'tuple' (line 435)
        tuple_608000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 435)
        # Adding element type (line 435)
        # Getting the type of 'mu' (line 435)
        mu_608001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 15), tuple_608000, mu_608001)
        # Adding element type (line 435)
        # Getting the type of 'var' (line 435)
        var_608002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 15), tuple_608000, var_608002)
        # Adding element type (line 435)
        # Getting the type of 'g1' (line 435)
        g1_608003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 15), tuple_608000, g1_608003)
        # Adding element type (line 435)
        # Getting the type of 'g2' (line 435)
        g2_608004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 15), tuple_608000, g2_608004)
        
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'stypy_return_type', tuple_608000)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_608005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608005)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_608005


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 391, 0, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'logser_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'logser_gen' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'logser_gen', logser_gen)

# Assigning a Call to a Name (line 436):

# Assigning a Call to a Name (line 436):

# Call to logser_gen(...): (line 436)
# Processing the call keyword arguments (line 436)
int_608007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 22), 'int')
keyword_608008 = int_608007
str_608009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 30), 'str', 'logser')
keyword_608010 = str_608009
str_608011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 49), 'str', 'A logarithmic')
keyword_608012 = str_608011
kwargs_608013 = {'a': keyword_608008, 'name': keyword_608010, 'longname': keyword_608012}
# Getting the type of 'logser_gen' (line 436)
logser_gen_608006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 9), 'logser_gen', False)
# Calling logser_gen(args, kwargs) (line 436)
logser_gen_call_result_608014 = invoke(stypy.reporting.localization.Localization(__file__, 436, 9), logser_gen_608006, *[], **kwargs_608013)

# Assigning a type to the variable 'logser' (line 436)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), 'logser', logser_gen_call_result_608014)
# Declaration of the 'poisson_gen' class
# Getting the type of 'rv_discrete' (line 439)
rv_discrete_608015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 18), 'rv_discrete')

class poisson_gen(rv_discrete_608015, ):
    str_608016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, (-1)), 'str', 'A Poisson discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `poisson` is::\n\n        poisson.pmf(k) = exp(-mu) * mu**k / k!\n\n    for ``k >= 0``.\n\n    `poisson` takes ``mu`` as shape parameter.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poisson_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        poisson_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poisson_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        poisson_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'poisson_gen._argcheck')
        poisson_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['mu'])
        poisson_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        poisson_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poisson_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        poisson_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        poisson_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poisson_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen._argcheck', ['mu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['mu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Getting the type of 'mu' (line 462)
        mu_608017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'mu')
        int_608018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 21), 'int')
        # Applying the binary operator '>=' (line 462)
        result_ge_608019 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 15), '>=', mu_608017, int_608018)
        
        # Assigning a type to the variable 'stypy_return_type' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'stypy_return_type', result_ge_608019)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_608020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608020)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_608020


    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 464, 4, False)
        # Assigning a type to the variable 'self' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poisson_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        poisson_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poisson_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        poisson_gen._rvs.__dict__.__setitem__('stypy_function_name', 'poisson_gen._rvs')
        poisson_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['mu'])
        poisson_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        poisson_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poisson_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        poisson_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        poisson_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poisson_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen._rvs', ['mu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['mu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Call to poisson(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'mu' (line 465)
        mu_608024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 42), 'mu', False)
        # Getting the type of 'self' (line 465)
        self_608025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 46), 'self', False)
        # Obtaining the member '_size' of a type (line 465)
        _size_608026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 46), self_608025, '_size')
        # Processing the call keyword arguments (line 465)
        kwargs_608027 = {}
        # Getting the type of 'self' (line 465)
        self_608021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'self', False)
        # Obtaining the member '_random_state' of a type (line 465)
        _random_state_608022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 15), self_608021, '_random_state')
        # Obtaining the member 'poisson' of a type (line 465)
        poisson_608023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 15), _random_state_608022, 'poisson')
        # Calling poisson(args, kwargs) (line 465)
        poisson_call_result_608028 = invoke(stypy.reporting.localization.Localization(__file__, 465, 15), poisson_608023, *[mu_608024, _size_608026], **kwargs_608027)
        
        # Assigning a type to the variable 'stypy_return_type' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'stypy_return_type', poisson_call_result_608028)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 464)
        stypy_return_type_608029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_608029


    @norecursion
    def _logpmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logpmf'
        module_type_store = module_type_store.open_function_context('_logpmf', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poisson_gen._logpmf.__dict__.__setitem__('stypy_localization', localization)
        poisson_gen._logpmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poisson_gen._logpmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        poisson_gen._logpmf.__dict__.__setitem__('stypy_function_name', 'poisson_gen._logpmf')
        poisson_gen._logpmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'mu'])
        poisson_gen._logpmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        poisson_gen._logpmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poisson_gen._logpmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        poisson_gen._logpmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        poisson_gen._logpmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poisson_gen._logpmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen._logpmf', ['k', 'mu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logpmf', localization, ['k', 'mu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logpmf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 468):
        
        # Assigning a BinOp to a Name (line 468):
        
        # Call to xlogy(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 'k' (line 468)
        k_608032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 27), 'k', False)
        # Getting the type of 'mu' (line 468)
        mu_608033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 30), 'mu', False)
        # Processing the call keyword arguments (line 468)
        kwargs_608034 = {}
        # Getting the type of 'special' (line 468)
        special_608030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 13), 'special', False)
        # Obtaining the member 'xlogy' of a type (line 468)
        xlogy_608031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 13), special_608030, 'xlogy')
        # Calling xlogy(args, kwargs) (line 468)
        xlogy_call_result_608035 = invoke(stypy.reporting.localization.Localization(__file__, 468, 13), xlogy_608031, *[k_608032, mu_608033], **kwargs_608034)
        
        
        # Call to gamln(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 'k' (line 468)
        k_608037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 42), 'k', False)
        int_608038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 46), 'int')
        # Applying the binary operator '+' (line 468)
        result_add_608039 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 42), '+', k_608037, int_608038)
        
        # Processing the call keyword arguments (line 468)
        kwargs_608040 = {}
        # Getting the type of 'gamln' (line 468)
        gamln_608036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 36), 'gamln', False)
        # Calling gamln(args, kwargs) (line 468)
        gamln_call_result_608041 = invoke(stypy.reporting.localization.Localization(__file__, 468, 36), gamln_608036, *[result_add_608039], **kwargs_608040)
        
        # Applying the binary operator '-' (line 468)
        result_sub_608042 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 13), '-', xlogy_call_result_608035, gamln_call_result_608041)
        
        # Getting the type of 'mu' (line 468)
        mu_608043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 51), 'mu')
        # Applying the binary operator '-' (line 468)
        result_sub_608044 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 49), '-', result_sub_608042, mu_608043)
        
        # Assigning a type to the variable 'Pk' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'Pk', result_sub_608044)
        # Getting the type of 'Pk' (line 469)
        Pk_608045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 15), 'Pk')
        # Assigning a type to the variable 'stypy_return_type' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'stypy_return_type', Pk_608045)
        
        # ################# End of '_logpmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logpmf' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_608046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608046)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logpmf'
        return stypy_return_type_608046


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 471, 4, False)
        # Assigning a type to the variable 'self' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poisson_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        poisson_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poisson_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        poisson_gen._pmf.__dict__.__setitem__('stypy_function_name', 'poisson_gen._pmf')
        poisson_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'mu'])
        poisson_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        poisson_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poisson_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        poisson_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        poisson_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poisson_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen._pmf', ['k', 'mu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'mu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Call to exp(...): (line 472)
        # Processing the call arguments (line 472)
        
        # Call to _logpmf(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'k' (line 472)
        k_608050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 32), 'k', False)
        # Getting the type of 'mu' (line 472)
        mu_608051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 35), 'mu', False)
        # Processing the call keyword arguments (line 472)
        kwargs_608052 = {}
        # Getting the type of 'self' (line 472)
        self_608048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 19), 'self', False)
        # Obtaining the member '_logpmf' of a type (line 472)
        _logpmf_608049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 19), self_608048, '_logpmf')
        # Calling _logpmf(args, kwargs) (line 472)
        _logpmf_call_result_608053 = invoke(stypy.reporting.localization.Localization(__file__, 472, 19), _logpmf_608049, *[k_608050, mu_608051], **kwargs_608052)
        
        # Processing the call keyword arguments (line 472)
        kwargs_608054 = {}
        # Getting the type of 'exp' (line 472)
        exp_608047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'exp', False)
        # Calling exp(args, kwargs) (line 472)
        exp_call_result_608055 = invoke(stypy.reporting.localization.Localization(__file__, 472, 15), exp_608047, *[_logpmf_call_result_608053], **kwargs_608054)
        
        # Assigning a type to the variable 'stypy_return_type' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'stypy_return_type', exp_call_result_608055)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 471)
        stypy_return_type_608056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_608056


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 474, 4, False)
        # Assigning a type to the variable 'self' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poisson_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        poisson_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poisson_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        poisson_gen._cdf.__dict__.__setitem__('stypy_function_name', 'poisson_gen._cdf')
        poisson_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'mu'])
        poisson_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        poisson_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poisson_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        poisson_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        poisson_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poisson_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen._cdf', ['x', 'mu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'mu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 475):
        
        # Assigning a Call to a Name (line 475):
        
        # Call to floor(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'x' (line 475)
        x_608058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 18), 'x', False)
        # Processing the call keyword arguments (line 475)
        kwargs_608059 = {}
        # Getting the type of 'floor' (line 475)
        floor_608057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 475)
        floor_call_result_608060 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), floor_608057, *[x_608058], **kwargs_608059)
        
        # Assigning a type to the variable 'k' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'k', floor_call_result_608060)
        
        # Call to pdtr(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'k' (line 476)
        k_608063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 28), 'k', False)
        # Getting the type of 'mu' (line 476)
        mu_608064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 31), 'mu', False)
        # Processing the call keyword arguments (line 476)
        kwargs_608065 = {}
        # Getting the type of 'special' (line 476)
        special_608061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'special', False)
        # Obtaining the member 'pdtr' of a type (line 476)
        pdtr_608062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), special_608061, 'pdtr')
        # Calling pdtr(args, kwargs) (line 476)
        pdtr_call_result_608066 = invoke(stypy.reporting.localization.Localization(__file__, 476, 15), pdtr_608062, *[k_608063, mu_608064], **kwargs_608065)
        
        # Assigning a type to the variable 'stypy_return_type' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'stypy_return_type', pdtr_call_result_608066)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 474)
        stypy_return_type_608067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608067)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_608067


    @norecursion
    def _sf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sf'
        module_type_store = module_type_store.open_function_context('_sf', 478, 4, False)
        # Assigning a type to the variable 'self' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poisson_gen._sf.__dict__.__setitem__('stypy_localization', localization)
        poisson_gen._sf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poisson_gen._sf.__dict__.__setitem__('stypy_type_store', module_type_store)
        poisson_gen._sf.__dict__.__setitem__('stypy_function_name', 'poisson_gen._sf')
        poisson_gen._sf.__dict__.__setitem__('stypy_param_names_list', ['x', 'mu'])
        poisson_gen._sf.__dict__.__setitem__('stypy_varargs_param_name', None)
        poisson_gen._sf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poisson_gen._sf.__dict__.__setitem__('stypy_call_defaults', defaults)
        poisson_gen._sf.__dict__.__setitem__('stypy_call_varargs', varargs)
        poisson_gen._sf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poisson_gen._sf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen._sf', ['x', 'mu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sf', localization, ['x', 'mu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sf(...)' code ##################

        
        # Assigning a Call to a Name (line 479):
        
        # Assigning a Call to a Name (line 479):
        
        # Call to floor(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'x' (line 479)
        x_608069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 18), 'x', False)
        # Processing the call keyword arguments (line 479)
        kwargs_608070 = {}
        # Getting the type of 'floor' (line 479)
        floor_608068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 479)
        floor_call_result_608071 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), floor_608068, *[x_608069], **kwargs_608070)
        
        # Assigning a type to the variable 'k' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'k', floor_call_result_608071)
        
        # Call to pdtrc(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'k' (line 480)
        k_608074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 29), 'k', False)
        # Getting the type of 'mu' (line 480)
        mu_608075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 32), 'mu', False)
        # Processing the call keyword arguments (line 480)
        kwargs_608076 = {}
        # Getting the type of 'special' (line 480)
        special_608072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), 'special', False)
        # Obtaining the member 'pdtrc' of a type (line 480)
        pdtrc_608073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 15), special_608072, 'pdtrc')
        # Calling pdtrc(args, kwargs) (line 480)
        pdtrc_call_result_608077 = invoke(stypy.reporting.localization.Localization(__file__, 480, 15), pdtrc_608073, *[k_608074, mu_608075], **kwargs_608076)
        
        # Assigning a type to the variable 'stypy_return_type' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'stypy_return_type', pdtrc_call_result_608077)
        
        # ################# End of '_sf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sf' in the type store
        # Getting the type of 'stypy_return_type' (line 478)
        stypy_return_type_608078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608078)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sf'
        return stypy_return_type_608078


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 482, 4, False)
        # Assigning a type to the variable 'self' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poisson_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        poisson_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poisson_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        poisson_gen._ppf.__dict__.__setitem__('stypy_function_name', 'poisson_gen._ppf')
        poisson_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'mu'])
        poisson_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        poisson_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poisson_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        poisson_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        poisson_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poisson_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen._ppf', ['q', 'mu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'mu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Call to ceil(...): (line 483)
        # Processing the call arguments (line 483)
        
        # Call to pdtrik(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of 'q' (line 483)
        q_608082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 35), 'q', False)
        # Getting the type of 'mu' (line 483)
        mu_608083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 38), 'mu', False)
        # Processing the call keyword arguments (line 483)
        kwargs_608084 = {}
        # Getting the type of 'special' (line 483)
        special_608080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 20), 'special', False)
        # Obtaining the member 'pdtrik' of a type (line 483)
        pdtrik_608081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 20), special_608080, 'pdtrik')
        # Calling pdtrik(args, kwargs) (line 483)
        pdtrik_call_result_608085 = invoke(stypy.reporting.localization.Localization(__file__, 483, 20), pdtrik_608081, *[q_608082, mu_608083], **kwargs_608084)
        
        # Processing the call keyword arguments (line 483)
        kwargs_608086 = {}
        # Getting the type of 'ceil' (line 483)
        ceil_608079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 15), 'ceil', False)
        # Calling ceil(args, kwargs) (line 483)
        ceil_call_result_608087 = invoke(stypy.reporting.localization.Localization(__file__, 483, 15), ceil_608079, *[pdtrik_call_result_608085], **kwargs_608086)
        
        # Assigning a type to the variable 'vals' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'vals', ceil_call_result_608087)
        
        # Assigning a Call to a Name (line 484):
        
        # Assigning a Call to a Name (line 484):
        
        # Call to maximum(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'vals' (line 484)
        vals_608090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 27), 'vals', False)
        int_608091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 34), 'int')
        # Applying the binary operator '-' (line 484)
        result_sub_608092 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 27), '-', vals_608090, int_608091)
        
        int_608093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 37), 'int')
        # Processing the call keyword arguments (line 484)
        kwargs_608094 = {}
        # Getting the type of 'np' (line 484)
        np_608088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'np', False)
        # Obtaining the member 'maximum' of a type (line 484)
        maximum_608089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 16), np_608088, 'maximum')
        # Calling maximum(args, kwargs) (line 484)
        maximum_call_result_608095 = invoke(stypy.reporting.localization.Localization(__file__, 484, 16), maximum_608089, *[result_sub_608092, int_608093], **kwargs_608094)
        
        # Assigning a type to the variable 'vals1' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'vals1', maximum_call_result_608095)
        
        # Assigning a Call to a Name (line 485):
        
        # Assigning a Call to a Name (line 485):
        
        # Call to pdtr(...): (line 485)
        # Processing the call arguments (line 485)
        # Getting the type of 'vals1' (line 485)
        vals1_608098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 28), 'vals1', False)
        # Getting the type of 'mu' (line 485)
        mu_608099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 35), 'mu', False)
        # Processing the call keyword arguments (line 485)
        kwargs_608100 = {}
        # Getting the type of 'special' (line 485)
        special_608096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 15), 'special', False)
        # Obtaining the member 'pdtr' of a type (line 485)
        pdtr_608097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 15), special_608096, 'pdtr')
        # Calling pdtr(args, kwargs) (line 485)
        pdtr_call_result_608101 = invoke(stypy.reporting.localization.Localization(__file__, 485, 15), pdtr_608097, *[vals1_608098, mu_608099], **kwargs_608100)
        
        # Assigning a type to the variable 'temp' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'temp', pdtr_call_result_608101)
        
        # Call to where(...): (line 486)
        # Processing the call arguments (line 486)
        
        # Getting the type of 'temp' (line 486)
        temp_608104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 24), 'temp', False)
        # Getting the type of 'q' (line 486)
        q_608105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 32), 'q', False)
        # Applying the binary operator '>=' (line 486)
        result_ge_608106 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 24), '>=', temp_608104, q_608105)
        
        # Getting the type of 'vals1' (line 486)
        vals1_608107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 35), 'vals1', False)
        # Getting the type of 'vals' (line 486)
        vals_608108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 42), 'vals', False)
        # Processing the call keyword arguments (line 486)
        kwargs_608109 = {}
        # Getting the type of 'np' (line 486)
        np_608102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 486)
        where_608103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 15), np_608102, 'where')
        # Calling where(args, kwargs) (line 486)
        where_call_result_608110 = invoke(stypy.reporting.localization.Localization(__file__, 486, 15), where_608103, *[result_ge_608106, vals1_608107, vals_608108], **kwargs_608109)
        
        # Assigning a type to the variable 'stypy_return_type' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'stypy_return_type', where_call_result_608110)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 482)
        stypy_return_type_608111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_608111


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 488, 4, False)
        # Assigning a type to the variable 'self' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        poisson_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        poisson_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        poisson_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        poisson_gen._stats.__dict__.__setitem__('stypy_function_name', 'poisson_gen._stats')
        poisson_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['mu'])
        poisson_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        poisson_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        poisson_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        poisson_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        poisson_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        poisson_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen._stats', ['mu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['mu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a Name to a Name (line 489):
        
        # Assigning a Name to a Name (line 489):
        # Getting the type of 'mu' (line 489)
        mu_608112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 14), 'mu')
        # Assigning a type to the variable 'var' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'var', mu_608112)
        
        # Assigning a Call to a Name (line 490):
        
        # Assigning a Call to a Name (line 490):
        
        # Call to asarray(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'mu' (line 490)
        mu_608115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 25), 'mu', False)
        # Processing the call keyword arguments (line 490)
        kwargs_608116 = {}
        # Getting the type of 'np' (line 490)
        np_608113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 14), 'np', False)
        # Obtaining the member 'asarray' of a type (line 490)
        asarray_608114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 14), np_608113, 'asarray')
        # Calling asarray(args, kwargs) (line 490)
        asarray_call_result_608117 = invoke(stypy.reporting.localization.Localization(__file__, 490, 14), asarray_608114, *[mu_608115], **kwargs_608116)
        
        # Assigning a type to the variable 'tmp' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'tmp', asarray_call_result_608117)
        
        # Assigning a Compare to a Name (line 491):
        
        # Assigning a Compare to a Name (line 491):
        
        # Getting the type of 'tmp' (line 491)
        tmp_608118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 21), 'tmp')
        int_608119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 27), 'int')
        # Applying the binary operator '>' (line 491)
        result_gt_608120 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 21), '>', tmp_608118, int_608119)
        
        # Assigning a type to the variable 'mu_nonzero' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'mu_nonzero', result_gt_608120)
        
        # Assigning a Call to a Name (line 492):
        
        # Assigning a Call to a Name (line 492):
        
        # Call to _lazywhere(...): (line 492)
        # Processing the call arguments (line 492)
        # Getting the type of 'mu_nonzero' (line 492)
        mu_nonzero_608122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 24), 'mu_nonzero', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 492)
        tuple_608123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 492)
        # Adding element type (line 492)
        # Getting the type of 'tmp' (line 492)
        tmp_608124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 37), 'tmp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 37), tuple_608123, tmp_608124)
        

        @norecursion
        def _stypy_temp_lambda_524(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_524'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_524', 492, 44, True)
            # Passed parameters checking function
            _stypy_temp_lambda_524.stypy_localization = localization
            _stypy_temp_lambda_524.stypy_type_of_self = None
            _stypy_temp_lambda_524.stypy_type_store = module_type_store
            _stypy_temp_lambda_524.stypy_function_name = '_stypy_temp_lambda_524'
            _stypy_temp_lambda_524.stypy_param_names_list = ['x']
            _stypy_temp_lambda_524.stypy_varargs_param_name = None
            _stypy_temp_lambda_524.stypy_kwargs_param_name = None
            _stypy_temp_lambda_524.stypy_call_defaults = defaults
            _stypy_temp_lambda_524.stypy_call_varargs = varargs
            _stypy_temp_lambda_524.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_524', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_524', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to sqrt(...): (line 492)
            # Processing the call arguments (line 492)
            float_608126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 59), 'float')
            # Getting the type of 'x' (line 492)
            x_608127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 63), 'x', False)
            # Applying the binary operator 'div' (line 492)
            result_div_608128 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 59), 'div', float_608126, x_608127)
            
            # Processing the call keyword arguments (line 492)
            kwargs_608129 = {}
            # Getting the type of 'sqrt' (line 492)
            sqrt_608125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 54), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 492)
            sqrt_call_result_608130 = invoke(stypy.reporting.localization.Localization(__file__, 492, 54), sqrt_608125, *[result_div_608128], **kwargs_608129)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 492)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 44), 'stypy_return_type', sqrt_call_result_608130)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_524' in the type store
            # Getting the type of 'stypy_return_type' (line 492)
            stypy_return_type_608131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 44), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_608131)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_524'
            return stypy_return_type_608131

        # Assigning a type to the variable '_stypy_temp_lambda_524' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 44), '_stypy_temp_lambda_524', _stypy_temp_lambda_524)
        # Getting the type of '_stypy_temp_lambda_524' (line 492)
        _stypy_temp_lambda_524_608132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 44), '_stypy_temp_lambda_524')
        # Getting the type of 'np' (line 492)
        np_608133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 67), 'np', False)
        # Obtaining the member 'inf' of a type (line 492)
        inf_608134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 67), np_608133, 'inf')
        # Processing the call keyword arguments (line 492)
        kwargs_608135 = {}
        # Getting the type of '_lazywhere' (line 492)
        _lazywhere_608121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 13), '_lazywhere', False)
        # Calling _lazywhere(args, kwargs) (line 492)
        _lazywhere_call_result_608136 = invoke(stypy.reporting.localization.Localization(__file__, 492, 13), _lazywhere_608121, *[mu_nonzero_608122, tuple_608123, _stypy_temp_lambda_524_608132, inf_608134], **kwargs_608135)
        
        # Assigning a type to the variable 'g1' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'g1', _lazywhere_call_result_608136)
        
        # Assigning a Call to a Name (line 493):
        
        # Assigning a Call to a Name (line 493):
        
        # Call to _lazywhere(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'mu_nonzero' (line 493)
        mu_nonzero_608138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 24), 'mu_nonzero', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 493)
        tuple_608139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 493)
        # Adding element type (line 493)
        # Getting the type of 'tmp' (line 493)
        tmp_608140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 37), 'tmp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 37), tuple_608139, tmp_608140)
        

        @norecursion
        def _stypy_temp_lambda_525(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_525'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_525', 493, 44, True)
            # Passed parameters checking function
            _stypy_temp_lambda_525.stypy_localization = localization
            _stypy_temp_lambda_525.stypy_type_of_self = None
            _stypy_temp_lambda_525.stypy_type_store = module_type_store
            _stypy_temp_lambda_525.stypy_function_name = '_stypy_temp_lambda_525'
            _stypy_temp_lambda_525.stypy_param_names_list = ['x']
            _stypy_temp_lambda_525.stypy_varargs_param_name = None
            _stypy_temp_lambda_525.stypy_kwargs_param_name = None
            _stypy_temp_lambda_525.stypy_call_defaults = defaults
            _stypy_temp_lambda_525.stypy_call_varargs = varargs
            _stypy_temp_lambda_525.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_525', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_525', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            float_608141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 54), 'float')
            # Getting the type of 'x' (line 493)
            x_608142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 58), 'x', False)
            # Applying the binary operator 'div' (line 493)
            result_div_608143 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 54), 'div', float_608141, x_608142)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 44), 'stypy_return_type', result_div_608143)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_525' in the type store
            # Getting the type of 'stypy_return_type' (line 493)
            stypy_return_type_608144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 44), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_608144)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_525'
            return stypy_return_type_608144

        # Assigning a type to the variable '_stypy_temp_lambda_525' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 44), '_stypy_temp_lambda_525', _stypy_temp_lambda_525)
        # Getting the type of '_stypy_temp_lambda_525' (line 493)
        _stypy_temp_lambda_525_608145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 44), '_stypy_temp_lambda_525')
        # Getting the type of 'np' (line 493)
        np_608146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 61), 'np', False)
        # Obtaining the member 'inf' of a type (line 493)
        inf_608147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 61), np_608146, 'inf')
        # Processing the call keyword arguments (line 493)
        kwargs_608148 = {}
        # Getting the type of '_lazywhere' (line 493)
        _lazywhere_608137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 13), '_lazywhere', False)
        # Calling _lazywhere(args, kwargs) (line 493)
        _lazywhere_call_result_608149 = invoke(stypy.reporting.localization.Localization(__file__, 493, 13), _lazywhere_608137, *[mu_nonzero_608138, tuple_608139, _stypy_temp_lambda_525_608145, inf_608147], **kwargs_608148)
        
        # Assigning a type to the variable 'g2' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'g2', _lazywhere_call_result_608149)
        
        # Obtaining an instance of the builtin type 'tuple' (line 494)
        tuple_608150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 494)
        # Adding element type (line 494)
        # Getting the type of 'mu' (line 494)
        mu_608151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 15), tuple_608150, mu_608151)
        # Adding element type (line 494)
        # Getting the type of 'var' (line 494)
        var_608152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 15), tuple_608150, var_608152)
        # Adding element type (line 494)
        # Getting the type of 'g1' (line 494)
        g1_608153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 15), tuple_608150, g1_608153)
        # Adding element type (line 494)
        # Getting the type of 'g2' (line 494)
        g2_608154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 15), tuple_608150, g2_608154)
        
        # Assigning a type to the variable 'stypy_return_type' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'stypy_return_type', tuple_608150)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 488)
        stypy_return_type_608155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_608155


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 439, 0, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'poisson_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'poisson_gen' (line 439)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'poisson_gen', poisson_gen)

# Assigning a Call to a Name (line 496):

# Assigning a Call to a Name (line 496):

# Call to poisson_gen(...): (line 496)
# Processing the call keyword arguments (line 496)
str_608157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 27), 'str', 'poisson')
keyword_608158 = str_608157
str_608159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 47), 'str', 'A Poisson')
keyword_608160 = str_608159
kwargs_608161 = {'name': keyword_608158, 'longname': keyword_608160}
# Getting the type of 'poisson_gen' (line 496)
poisson_gen_608156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 10), 'poisson_gen', False)
# Calling poisson_gen(args, kwargs) (line 496)
poisson_gen_call_result_608162 = invoke(stypy.reporting.localization.Localization(__file__, 496, 10), poisson_gen_608156, *[], **kwargs_608161)

# Assigning a type to the variable 'poisson' (line 496)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), 'poisson', poisson_gen_call_result_608162)
# Declaration of the 'planck_gen' class
# Getting the type of 'rv_discrete' (line 499)
rv_discrete_608163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 17), 'rv_discrete')

class planck_gen(rv_discrete_608163, ):
    str_608164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, (-1)), 'str', 'A Planck discrete exponential random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `planck` is::\n\n        planck.pmf(k) = (1-exp(-lambda_))*exp(-lambda_*k)\n\n    for ``k*lambda_ >= 0``.\n\n    `planck` takes ``lambda_`` as shape parameter.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 519, 4, False)
        # Assigning a type to the variable 'self' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        planck_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        planck_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        planck_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        planck_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'planck_gen._argcheck')
        planck_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['lambda_'])
        planck_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        planck_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        planck_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        planck_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        planck_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        planck_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen._argcheck', ['lambda_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['lambda_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Assigning a Call to a Attribute (line 520):
        
        # Assigning a Call to a Attribute (line 520):
        
        # Call to where(...): (line 520)
        # Processing the call arguments (line 520)
        
        # Getting the type of 'lambda_' (line 520)
        lambda__608167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 26), 'lambda_', False)
        int_608168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 36), 'int')
        # Applying the binary operator '>' (line 520)
        result_gt_608169 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 26), '>', lambda__608167, int_608168)
        
        int_608170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 39), 'int')
        
        # Getting the type of 'np' (line 520)
        np_608171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 43), 'np', False)
        # Obtaining the member 'inf' of a type (line 520)
        inf_608172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 43), np_608171, 'inf')
        # Applying the 'usub' unary operator (line 520)
        result___neg___608173 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 42), 'usub', inf_608172)
        
        # Processing the call keyword arguments (line 520)
        kwargs_608174 = {}
        # Getting the type of 'np' (line 520)
        np_608165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 'np', False)
        # Obtaining the member 'where' of a type (line 520)
        where_608166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 17), np_608165, 'where')
        # Calling where(args, kwargs) (line 520)
        where_call_result_608175 = invoke(stypy.reporting.localization.Localization(__file__, 520, 17), where_608166, *[result_gt_608169, int_608170, result___neg___608173], **kwargs_608174)
        
        # Getting the type of 'self' (line 520)
        self_608176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'self')
        # Setting the type of the member 'a' of a type (line 520)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), self_608176, 'a', where_call_result_608175)
        
        # Assigning a Call to a Attribute (line 521):
        
        # Assigning a Call to a Attribute (line 521):
        
        # Call to where(...): (line 521)
        # Processing the call arguments (line 521)
        
        # Getting the type of 'lambda_' (line 521)
        lambda__608179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 26), 'lambda_', False)
        int_608180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 36), 'int')
        # Applying the binary operator '>' (line 521)
        result_gt_608181 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 26), '>', lambda__608179, int_608180)
        
        # Getting the type of 'np' (line 521)
        np_608182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 39), 'np', False)
        # Obtaining the member 'inf' of a type (line 521)
        inf_608183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 39), np_608182, 'inf')
        int_608184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 47), 'int')
        # Processing the call keyword arguments (line 521)
        kwargs_608185 = {}
        # Getting the type of 'np' (line 521)
        np_608177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 17), 'np', False)
        # Obtaining the member 'where' of a type (line 521)
        where_608178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 17), np_608177, 'where')
        # Calling where(args, kwargs) (line 521)
        where_call_result_608186 = invoke(stypy.reporting.localization.Localization(__file__, 521, 17), where_608178, *[result_gt_608181, inf_608183, int_608184], **kwargs_608185)
        
        # Getting the type of 'self' (line 521)
        self_608187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'self')
        # Setting the type of the member 'b' of a type (line 521)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 8), self_608187, 'b', where_call_result_608186)
        
        # Getting the type of 'lambda_' (line 522)
        lambda__608188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'lambda_')
        int_608189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 26), 'int')
        # Applying the binary operator '!=' (line 522)
        result_ne_608190 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 15), '!=', lambda__608188, int_608189)
        
        # Assigning a type to the variable 'stypy_return_type' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'stypy_return_type', result_ne_608190)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 519)
        stypy_return_type_608191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_608191


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 524, 4, False)
        # Assigning a type to the variable 'self' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        planck_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        planck_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        planck_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        planck_gen._pmf.__dict__.__setitem__('stypy_function_name', 'planck_gen._pmf')
        planck_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'lambda_'])
        planck_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        planck_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        planck_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        planck_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        planck_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        planck_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen._pmf', ['k', 'lambda_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'lambda_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 525):
        
        # Assigning a BinOp to a Name (line 525):
        int_608192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 16), 'int')
        
        # Call to exp(...): (line 525)
        # Processing the call arguments (line 525)
        
        # Getting the type of 'lambda_' (line 525)
        lambda__608194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'lambda_', False)
        # Applying the 'usub' unary operator (line 525)
        result___neg___608195 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 22), 'usub', lambda__608194)
        
        # Processing the call keyword arguments (line 525)
        kwargs_608196 = {}
        # Getting the type of 'exp' (line 525)
        exp_608193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 18), 'exp', False)
        # Calling exp(args, kwargs) (line 525)
        exp_call_result_608197 = invoke(stypy.reporting.localization.Localization(__file__, 525, 18), exp_608193, *[result___neg___608195], **kwargs_608196)
        
        # Applying the binary operator '-' (line 525)
        result_sub_608198 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 16), '-', int_608192, exp_call_result_608197)
        
        # Assigning a type to the variable 'fact' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'fact', result_sub_608198)
        # Getting the type of 'fact' (line 526)
        fact_608199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'fact')
        
        # Call to exp(...): (line 526)
        # Processing the call arguments (line 526)
        
        # Getting the type of 'lambda_' (line 526)
        lambda__608201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 25), 'lambda_', False)
        # Applying the 'usub' unary operator (line 526)
        result___neg___608202 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 24), 'usub', lambda__608201)
        
        # Getting the type of 'k' (line 526)
        k_608203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 33), 'k', False)
        # Applying the binary operator '*' (line 526)
        result_mul_608204 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 24), '*', result___neg___608202, k_608203)
        
        # Processing the call keyword arguments (line 526)
        kwargs_608205 = {}
        # Getting the type of 'exp' (line 526)
        exp_608200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 20), 'exp', False)
        # Calling exp(args, kwargs) (line 526)
        exp_call_result_608206 = invoke(stypy.reporting.localization.Localization(__file__, 526, 20), exp_608200, *[result_mul_608204], **kwargs_608205)
        
        # Applying the binary operator '*' (line 526)
        result_mul_608207 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 15), '*', fact_608199, exp_call_result_608206)
        
        # Assigning a type to the variable 'stypy_return_type' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'stypy_return_type', result_mul_608207)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 524)
        stypy_return_type_608208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_608208


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 528, 4, False)
        # Assigning a type to the variable 'self' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        planck_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        planck_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        planck_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        planck_gen._cdf.__dict__.__setitem__('stypy_function_name', 'planck_gen._cdf')
        planck_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'lambda_'])
        planck_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        planck_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        planck_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        planck_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        planck_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        planck_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen._cdf', ['x', 'lambda_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'lambda_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 529):
        
        # Assigning a Call to a Name (line 529):
        
        # Call to floor(...): (line 529)
        # Processing the call arguments (line 529)
        # Getting the type of 'x' (line 529)
        x_608210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 18), 'x', False)
        # Processing the call keyword arguments (line 529)
        kwargs_608211 = {}
        # Getting the type of 'floor' (line 529)
        floor_608209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 529)
        floor_call_result_608212 = invoke(stypy.reporting.localization.Localization(__file__, 529, 12), floor_608209, *[x_608210], **kwargs_608211)
        
        # Assigning a type to the variable 'k' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'k', floor_call_result_608212)
        int_608213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 15), 'int')
        
        # Call to exp(...): (line 530)
        # Processing the call arguments (line 530)
        
        # Getting the type of 'lambda_' (line 530)
        lambda__608215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 22), 'lambda_', False)
        # Applying the 'usub' unary operator (line 530)
        result___neg___608216 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 21), 'usub', lambda__608215)
        
        # Getting the type of 'k' (line 530)
        k_608217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 31), 'k', False)
        int_608218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 33), 'int')
        # Applying the binary operator '+' (line 530)
        result_add_608219 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 31), '+', k_608217, int_608218)
        
        # Applying the binary operator '*' (line 530)
        result_mul_608220 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 21), '*', result___neg___608216, result_add_608219)
        
        # Processing the call keyword arguments (line 530)
        kwargs_608221 = {}
        # Getting the type of 'exp' (line 530)
        exp_608214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 17), 'exp', False)
        # Calling exp(args, kwargs) (line 530)
        exp_call_result_608222 = invoke(stypy.reporting.localization.Localization(__file__, 530, 17), exp_608214, *[result_mul_608220], **kwargs_608221)
        
        # Applying the binary operator '-' (line 530)
        result_sub_608223 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 15), '-', int_608213, exp_call_result_608222)
        
        # Assigning a type to the variable 'stypy_return_type' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'stypy_return_type', result_sub_608223)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 528)
        stypy_return_type_608224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608224)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_608224


    @norecursion
    def _sf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sf'
        module_type_store = module_type_store.open_function_context('_sf', 532, 4, False)
        # Assigning a type to the variable 'self' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        planck_gen._sf.__dict__.__setitem__('stypy_localization', localization)
        planck_gen._sf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        planck_gen._sf.__dict__.__setitem__('stypy_type_store', module_type_store)
        planck_gen._sf.__dict__.__setitem__('stypy_function_name', 'planck_gen._sf')
        planck_gen._sf.__dict__.__setitem__('stypy_param_names_list', ['x', 'lambda_'])
        planck_gen._sf.__dict__.__setitem__('stypy_varargs_param_name', None)
        planck_gen._sf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        planck_gen._sf.__dict__.__setitem__('stypy_call_defaults', defaults)
        planck_gen._sf.__dict__.__setitem__('stypy_call_varargs', varargs)
        planck_gen._sf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        planck_gen._sf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen._sf', ['x', 'lambda_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sf', localization, ['x', 'lambda_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sf(...)' code ##################

        
        # Call to exp(...): (line 533)
        # Processing the call arguments (line 533)
        
        # Call to _logsf(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'x' (line 533)
        x_608229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 34), 'x', False)
        # Getting the type of 'lambda_' (line 533)
        lambda__608230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 37), 'lambda_', False)
        # Processing the call keyword arguments (line 533)
        kwargs_608231 = {}
        # Getting the type of 'self' (line 533)
        self_608227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'self', False)
        # Obtaining the member '_logsf' of a type (line 533)
        _logsf_608228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 22), self_608227, '_logsf')
        # Calling _logsf(args, kwargs) (line 533)
        _logsf_call_result_608232 = invoke(stypy.reporting.localization.Localization(__file__, 533, 22), _logsf_608228, *[x_608229, lambda__608230], **kwargs_608231)
        
        # Processing the call keyword arguments (line 533)
        kwargs_608233 = {}
        # Getting the type of 'np' (line 533)
        np_608225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 533)
        exp_608226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 15), np_608225, 'exp')
        # Calling exp(args, kwargs) (line 533)
        exp_call_result_608234 = invoke(stypy.reporting.localization.Localization(__file__, 533, 15), exp_608226, *[_logsf_call_result_608232], **kwargs_608233)
        
        # Assigning a type to the variable 'stypy_return_type' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'stypy_return_type', exp_call_result_608234)
        
        # ################# End of '_sf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sf' in the type store
        # Getting the type of 'stypy_return_type' (line 532)
        stypy_return_type_608235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sf'
        return stypy_return_type_608235


    @norecursion
    def _logsf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_logsf'
        module_type_store = module_type_store.open_function_context('_logsf', 535, 4, False)
        # Assigning a type to the variable 'self' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        planck_gen._logsf.__dict__.__setitem__('stypy_localization', localization)
        planck_gen._logsf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        planck_gen._logsf.__dict__.__setitem__('stypy_type_store', module_type_store)
        planck_gen._logsf.__dict__.__setitem__('stypy_function_name', 'planck_gen._logsf')
        planck_gen._logsf.__dict__.__setitem__('stypy_param_names_list', ['x', 'lambda_'])
        planck_gen._logsf.__dict__.__setitem__('stypy_varargs_param_name', None)
        planck_gen._logsf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        planck_gen._logsf.__dict__.__setitem__('stypy_call_defaults', defaults)
        planck_gen._logsf.__dict__.__setitem__('stypy_call_varargs', varargs)
        planck_gen._logsf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        planck_gen._logsf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen._logsf', ['x', 'lambda_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_logsf', localization, ['x', 'lambda_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_logsf(...)' code ##################

        
        # Assigning a Call to a Name (line 536):
        
        # Assigning a Call to a Name (line 536):
        
        # Call to floor(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'x' (line 536)
        x_608237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 18), 'x', False)
        # Processing the call keyword arguments (line 536)
        kwargs_608238 = {}
        # Getting the type of 'floor' (line 536)
        floor_608236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 536)
        floor_call_result_608239 = invoke(stypy.reporting.localization.Localization(__file__, 536, 12), floor_608236, *[x_608237], **kwargs_608238)
        
        # Assigning a type to the variable 'k' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'k', floor_call_result_608239)
        
        # Getting the type of 'lambda_' (line 537)
        lambda__608240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 16), 'lambda_')
        # Applying the 'usub' unary operator (line 537)
        result___neg___608241 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 15), 'usub', lambda__608240)
        
        # Getting the type of 'k' (line 537)
        k_608242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 25), 'k')
        int_608243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 27), 'int')
        # Applying the binary operator '+' (line 537)
        result_add_608244 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 25), '+', k_608242, int_608243)
        
        # Applying the binary operator '*' (line 537)
        result_mul_608245 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 15), '*', result___neg___608241, result_add_608244)
        
        # Assigning a type to the variable 'stypy_return_type' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'stypy_return_type', result_mul_608245)
        
        # ################# End of '_logsf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_logsf' in the type store
        # Getting the type of 'stypy_return_type' (line 535)
        stypy_return_type_608246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608246)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_logsf'
        return stypy_return_type_608246


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 539, 4, False)
        # Assigning a type to the variable 'self' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        planck_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        planck_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        planck_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        planck_gen._ppf.__dict__.__setitem__('stypy_function_name', 'planck_gen._ppf')
        planck_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'lambda_'])
        planck_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        planck_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        planck_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        planck_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        planck_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        planck_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen._ppf', ['q', 'lambda_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'lambda_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Assigning a Call to a Name (line 540):
        
        # Assigning a Call to a Name (line 540):
        
        # Call to ceil(...): (line 540)
        # Processing the call arguments (line 540)
        float_608248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 20), 'float')
        # Getting the type of 'lambda_' (line 540)
        lambda__608249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 25), 'lambda_', False)
        # Applying the binary operator 'div' (line 540)
        result_div_608250 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 20), 'div', float_608248, lambda__608249)
        
        
        # Call to log1p(...): (line 540)
        # Processing the call arguments (line 540)
        
        # Getting the type of 'q' (line 540)
        q_608252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 42), 'q', False)
        # Applying the 'usub' unary operator (line 540)
        result___neg___608253 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 41), 'usub', q_608252)
        
        # Processing the call keyword arguments (line 540)
        kwargs_608254 = {}
        # Getting the type of 'log1p' (line 540)
        log1p_608251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 35), 'log1p', False)
        # Calling log1p(args, kwargs) (line 540)
        log1p_call_result_608255 = invoke(stypy.reporting.localization.Localization(__file__, 540, 35), log1p_608251, *[result___neg___608253], **kwargs_608254)
        
        # Applying the binary operator '*' (line 540)
        result_mul_608256 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 33), '*', result_div_608250, log1p_call_result_608255)
        
        int_608257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 45), 'int')
        # Applying the binary operator '-' (line 540)
        result_sub_608258 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 20), '-', result_mul_608256, int_608257)
        
        # Processing the call keyword arguments (line 540)
        kwargs_608259 = {}
        # Getting the type of 'ceil' (line 540)
        ceil_608247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 15), 'ceil', False)
        # Calling ceil(args, kwargs) (line 540)
        ceil_call_result_608260 = invoke(stypy.reporting.localization.Localization(__file__, 540, 15), ceil_608247, *[result_sub_608258], **kwargs_608259)
        
        # Assigning a type to the variable 'vals' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'vals', ceil_call_result_608260)
        
        # Assigning a Call to a Name (line 541):
        
        # Assigning a Call to a Name (line 541):
        
        # Call to clip(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'self' (line 541)
        self_608265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 30), 'self', False)
        # Obtaining the member 'a' of a type (line 541)
        a_608266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 30), self_608265, 'a')
        # Getting the type of 'np' (line 541)
        np_608267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 38), 'np', False)
        # Obtaining the member 'inf' of a type (line 541)
        inf_608268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 38), np_608267, 'inf')
        # Processing the call keyword arguments (line 541)
        kwargs_608269 = {}
        # Getting the type of 'vals' (line 541)
        vals_608261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 17), 'vals', False)
        int_608262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 22), 'int')
        # Applying the binary operator '-' (line 541)
        result_sub_608263 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 17), '-', vals_608261, int_608262)
        
        # Obtaining the member 'clip' of a type (line 541)
        clip_608264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 17), result_sub_608263, 'clip')
        # Calling clip(args, kwargs) (line 541)
        clip_call_result_608270 = invoke(stypy.reporting.localization.Localization(__file__, 541, 17), clip_608264, *[a_608266, inf_608268], **kwargs_608269)
        
        # Assigning a type to the variable 'vals1' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'vals1', clip_call_result_608270)
        
        # Assigning a Call to a Name (line 542):
        
        # Assigning a Call to a Name (line 542):
        
        # Call to _cdf(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'vals1' (line 542)
        vals1_608273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 25), 'vals1', False)
        # Getting the type of 'lambda_' (line 542)
        lambda__608274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'lambda_', False)
        # Processing the call keyword arguments (line 542)
        kwargs_608275 = {}
        # Getting the type of 'self' (line 542)
        self_608271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 15), 'self', False)
        # Obtaining the member '_cdf' of a type (line 542)
        _cdf_608272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 15), self_608271, '_cdf')
        # Calling _cdf(args, kwargs) (line 542)
        _cdf_call_result_608276 = invoke(stypy.reporting.localization.Localization(__file__, 542, 15), _cdf_608272, *[vals1_608273, lambda__608274], **kwargs_608275)
        
        # Assigning a type to the variable 'temp' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'temp', _cdf_call_result_608276)
        
        # Call to where(...): (line 543)
        # Processing the call arguments (line 543)
        
        # Getting the type of 'temp' (line 543)
        temp_608279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 24), 'temp', False)
        # Getting the type of 'q' (line 543)
        q_608280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 32), 'q', False)
        # Applying the binary operator '>=' (line 543)
        result_ge_608281 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 24), '>=', temp_608279, q_608280)
        
        # Getting the type of 'vals1' (line 543)
        vals1_608282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 35), 'vals1', False)
        # Getting the type of 'vals' (line 543)
        vals_608283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 42), 'vals', False)
        # Processing the call keyword arguments (line 543)
        kwargs_608284 = {}
        # Getting the type of 'np' (line 543)
        np_608277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 543)
        where_608278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 15), np_608277, 'where')
        # Calling where(args, kwargs) (line 543)
        where_call_result_608285 = invoke(stypy.reporting.localization.Localization(__file__, 543, 15), where_608278, *[result_ge_608281, vals1_608282, vals_608283], **kwargs_608284)
        
        # Assigning a type to the variable 'stypy_return_type' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'stypy_return_type', where_call_result_608285)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 539)
        stypy_return_type_608286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_608286


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 545, 4, False)
        # Assigning a type to the variable 'self' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        planck_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        planck_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        planck_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        planck_gen._stats.__dict__.__setitem__('stypy_function_name', 'planck_gen._stats')
        planck_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['lambda_'])
        planck_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        planck_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        planck_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        planck_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        planck_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        planck_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen._stats', ['lambda_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['lambda_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a BinOp to a Name (line 546):
        
        # Assigning a BinOp to a Name (line 546):
        int_608287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 13), 'int')
        
        # Call to exp(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'lambda_' (line 546)
        lambda__608289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 20), 'lambda_', False)
        # Processing the call keyword arguments (line 546)
        kwargs_608290 = {}
        # Getting the type of 'exp' (line 546)
        exp_608288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'exp', False)
        # Calling exp(args, kwargs) (line 546)
        exp_call_result_608291 = invoke(stypy.reporting.localization.Localization(__file__, 546, 16), exp_608288, *[lambda__608289], **kwargs_608290)
        
        int_608292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 29), 'int')
        # Applying the binary operator '-' (line 546)
        result_sub_608293 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 16), '-', exp_call_result_608291, int_608292)
        
        # Applying the binary operator 'div' (line 546)
        result_div_608294 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 13), 'div', int_608287, result_sub_608293)
        
        # Assigning a type to the variable 'mu' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'mu', result_div_608294)
        
        # Assigning a BinOp to a Name (line 547):
        
        # Assigning a BinOp to a Name (line 547):
        
        # Call to exp(...): (line 547)
        # Processing the call arguments (line 547)
        
        # Getting the type of 'lambda_' (line 547)
        lambda__608296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 19), 'lambda_', False)
        # Applying the 'usub' unary operator (line 547)
        result___neg___608297 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 18), 'usub', lambda__608296)
        
        # Processing the call keyword arguments (line 547)
        kwargs_608298 = {}
        # Getting the type of 'exp' (line 547)
        exp_608295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 14), 'exp', False)
        # Calling exp(args, kwargs) (line 547)
        exp_call_result_608299 = invoke(stypy.reporting.localization.Localization(__file__, 547, 14), exp_608295, *[result___neg___608297], **kwargs_608298)
        
        
        # Call to expm1(...): (line 547)
        # Processing the call arguments (line 547)
        
        # Getting the type of 'lambda_' (line 547)
        lambda__608301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 36), 'lambda_', False)
        # Applying the 'usub' unary operator (line 547)
        result___neg___608302 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 35), 'usub', lambda__608301)
        
        # Processing the call keyword arguments (line 547)
        kwargs_608303 = {}
        # Getting the type of 'expm1' (line 547)
        expm1_608300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 29), 'expm1', False)
        # Calling expm1(args, kwargs) (line 547)
        expm1_call_result_608304 = invoke(stypy.reporting.localization.Localization(__file__, 547, 29), expm1_608300, *[result___neg___608302], **kwargs_608303)
        
        int_608305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 47), 'int')
        # Applying the binary operator '**' (line 547)
        result_pow_608306 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 28), '**', expm1_call_result_608304, int_608305)
        
        # Applying the binary operator 'div' (line 547)
        result_div_608307 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 14), 'div', exp_call_result_608299, result_pow_608306)
        
        # Assigning a type to the variable 'var' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'var', result_div_608307)
        
        # Assigning a BinOp to a Name (line 548):
        
        # Assigning a BinOp to a Name (line 548):
        int_608308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 13), 'int')
        
        # Call to cosh(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'lambda_' (line 548)
        lambda__608310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 20), 'lambda_', False)
        float_608311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 28), 'float')
        # Applying the binary operator 'div' (line 548)
        result_div_608312 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 20), 'div', lambda__608310, float_608311)
        
        # Processing the call keyword arguments (line 548)
        kwargs_608313 = {}
        # Getting the type of 'cosh' (line 548)
        cosh_608309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'cosh', False)
        # Calling cosh(args, kwargs) (line 548)
        cosh_call_result_608314 = invoke(stypy.reporting.localization.Localization(__file__, 548, 15), cosh_608309, *[result_div_608312], **kwargs_608313)
        
        # Applying the binary operator '*' (line 548)
        result_mul_608315 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 13), '*', int_608308, cosh_call_result_608314)
        
        # Assigning a type to the variable 'g1' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'g1', result_mul_608315)
        
        # Assigning a BinOp to a Name (line 549):
        
        # Assigning a BinOp to a Name (line 549):
        int_608316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 13), 'int')
        int_608317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 15), 'int')
        
        # Call to cosh(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'lambda_' (line 549)
        lambda__608319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 22), 'lambda_', False)
        # Processing the call keyword arguments (line 549)
        kwargs_608320 = {}
        # Getting the type of 'cosh' (line 549)
        cosh_608318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 17), 'cosh', False)
        # Calling cosh(args, kwargs) (line 549)
        cosh_call_result_608321 = invoke(stypy.reporting.localization.Localization(__file__, 549, 17), cosh_608318, *[lambda__608319], **kwargs_608320)
        
        # Applying the binary operator '*' (line 549)
        result_mul_608322 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 15), '*', int_608317, cosh_call_result_608321)
        
        # Applying the binary operator '+' (line 549)
        result_add_608323 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 13), '+', int_608316, result_mul_608322)
        
        # Assigning a type to the variable 'g2' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'g2', result_add_608323)
        
        # Obtaining an instance of the builtin type 'tuple' (line 550)
        tuple_608324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 550)
        # Adding element type (line 550)
        # Getting the type of 'mu' (line 550)
        mu_608325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 15), tuple_608324, mu_608325)
        # Adding element type (line 550)
        # Getting the type of 'var' (line 550)
        var_608326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 15), tuple_608324, var_608326)
        # Adding element type (line 550)
        # Getting the type of 'g1' (line 550)
        g1_608327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 15), tuple_608324, g1_608327)
        # Adding element type (line 550)
        # Getting the type of 'g2' (line 550)
        g2_608328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 15), tuple_608324, g2_608328)
        
        # Assigning a type to the variable 'stypy_return_type' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'stypy_return_type', tuple_608324)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 545)
        stypy_return_type_608329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608329)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_608329


    @norecursion
    def _entropy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_entropy'
        module_type_store = module_type_store.open_function_context('_entropy', 552, 4, False)
        # Assigning a type to the variable 'self' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        planck_gen._entropy.__dict__.__setitem__('stypy_localization', localization)
        planck_gen._entropy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        planck_gen._entropy.__dict__.__setitem__('stypy_type_store', module_type_store)
        planck_gen._entropy.__dict__.__setitem__('stypy_function_name', 'planck_gen._entropy')
        planck_gen._entropy.__dict__.__setitem__('stypy_param_names_list', ['lambda_'])
        planck_gen._entropy.__dict__.__setitem__('stypy_varargs_param_name', None)
        planck_gen._entropy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        planck_gen._entropy.__dict__.__setitem__('stypy_call_defaults', defaults)
        planck_gen._entropy.__dict__.__setitem__('stypy_call_varargs', varargs)
        planck_gen._entropy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        planck_gen._entropy.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen._entropy', ['lambda_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_entropy', localization, ['lambda_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_entropy(...)' code ##################

        
        # Assigning a Name to a Name (line 553):
        
        # Assigning a Name to a Name (line 553):
        # Getting the type of 'lambda_' (line 553)
        lambda__608330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'lambda_')
        # Assigning a type to the variable 'l' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'l', lambda__608330)
        
        # Assigning a BinOp to a Name (line 554):
        
        # Assigning a BinOp to a Name (line 554):
        int_608331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 13), 'int')
        
        # Call to exp(...): (line 554)
        # Processing the call arguments (line 554)
        
        # Getting the type of 'l' (line 554)
        l_608333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'l', False)
        # Applying the 'usub' unary operator (line 554)
        result___neg___608334 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 19), 'usub', l_608333)
        
        # Processing the call keyword arguments (line 554)
        kwargs_608335 = {}
        # Getting the type of 'exp' (line 554)
        exp_608332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 15), 'exp', False)
        # Calling exp(args, kwargs) (line 554)
        exp_call_result_608336 = invoke(stypy.reporting.localization.Localization(__file__, 554, 15), exp_608332, *[result___neg___608334], **kwargs_608335)
        
        # Applying the binary operator '-' (line 554)
        result_sub_608337 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 13), '-', int_608331, exp_call_result_608336)
        
        # Assigning a type to the variable 'C' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'C', result_sub_608337)
        # Getting the type of 'l' (line 555)
        l_608338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'l')
        
        # Call to exp(...): (line 555)
        # Processing the call arguments (line 555)
        
        # Getting the type of 'l' (line 555)
        l_608340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 22), 'l', False)
        # Applying the 'usub' unary operator (line 555)
        result___neg___608341 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 21), 'usub', l_608340)
        
        # Processing the call keyword arguments (line 555)
        kwargs_608342 = {}
        # Getting the type of 'exp' (line 555)
        exp_608339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 17), 'exp', False)
        # Calling exp(args, kwargs) (line 555)
        exp_call_result_608343 = invoke(stypy.reporting.localization.Localization(__file__, 555, 17), exp_608339, *[result___neg___608341], **kwargs_608342)
        
        # Applying the binary operator '*' (line 555)
        result_mul_608344 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 15), '*', l_608338, exp_call_result_608343)
        
        # Getting the type of 'C' (line 555)
        C_608345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 25), 'C')
        # Applying the binary operator 'div' (line 555)
        result_div_608346 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 24), 'div', result_mul_608344, C_608345)
        
        
        # Call to log(...): (line 555)
        # Processing the call arguments (line 555)
        # Getting the type of 'C' (line 555)
        C_608348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 33), 'C', False)
        # Processing the call keyword arguments (line 555)
        kwargs_608349 = {}
        # Getting the type of 'log' (line 555)
        log_608347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 29), 'log', False)
        # Calling log(args, kwargs) (line 555)
        log_call_result_608350 = invoke(stypy.reporting.localization.Localization(__file__, 555, 29), log_608347, *[C_608348], **kwargs_608349)
        
        # Applying the binary operator '-' (line 555)
        result_sub_608351 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 15), '-', result_div_608346, log_call_result_608350)
        
        # Assigning a type to the variable 'stypy_return_type' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'stypy_return_type', result_sub_608351)
        
        # ################# End of '_entropy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_entropy' in the type store
        # Getting the type of 'stypy_return_type' (line 552)
        stypy_return_type_608352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608352)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_entropy'
        return stypy_return_type_608352


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 499, 0, False)
        # Assigning a type to the variable 'self' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'planck_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'planck_gen' (line 499)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 0), 'planck_gen', planck_gen)

# Assigning a Call to a Name (line 556):

# Assigning a Call to a Name (line 556):

# Call to planck_gen(...): (line 556)
# Processing the call keyword arguments (line 556)
str_608354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 25), 'str', 'planck')
keyword_608355 = str_608354
str_608356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 44), 'str', 'A discrete exponential ')
keyword_608357 = str_608356
kwargs_608358 = {'name': keyword_608355, 'longname': keyword_608357}
# Getting the type of 'planck_gen' (line 556)
planck_gen_608353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 9), 'planck_gen', False)
# Calling planck_gen(args, kwargs) (line 556)
planck_gen_call_result_608359 = invoke(stypy.reporting.localization.Localization(__file__, 556, 9), planck_gen_608353, *[], **kwargs_608358)

# Assigning a type to the variable 'planck' (line 556)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 0), 'planck', planck_gen_call_result_608359)
# Declaration of the 'boltzmann_gen' class
# Getting the type of 'rv_discrete' (line 559)
rv_discrete_608360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 'rv_discrete')

class boltzmann_gen(rv_discrete_608360, ):
    str_608361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, (-1)), 'str', 'A Boltzmann (Truncated Discrete Exponential) random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `boltzmann` is::\n\n        boltzmann.pmf(k) = (1-exp(-lambda_)*exp(-lambda_*k)/(1-exp(-lambda_*N))\n\n    for ``k = 0,..., N-1``.\n\n    `boltzmann` takes ``lambda_`` and ``N`` as shape parameters.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 579, 4, False)
        # Assigning a type to the variable 'self' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_function_name', 'boltzmann_gen._pmf')
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'lambda_', 'N'])
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        boltzmann_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boltzmann_gen._pmf', ['k', 'lambda_', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'lambda_', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 580):
        
        # Assigning a BinOp to a Name (line 580):
        int_608362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 16), 'int')
        
        # Call to exp(...): (line 580)
        # Processing the call arguments (line 580)
        
        # Getting the type of 'lambda_' (line 580)
        lambda__608364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 23), 'lambda_', False)
        # Applying the 'usub' unary operator (line 580)
        result___neg___608365 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 22), 'usub', lambda__608364)
        
        # Processing the call keyword arguments (line 580)
        kwargs_608366 = {}
        # Getting the type of 'exp' (line 580)
        exp_608363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 18), 'exp', False)
        # Calling exp(args, kwargs) (line 580)
        exp_call_result_608367 = invoke(stypy.reporting.localization.Localization(__file__, 580, 18), exp_608363, *[result___neg___608365], **kwargs_608366)
        
        # Applying the binary operator '-' (line 580)
        result_sub_608368 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 16), '-', int_608362, exp_call_result_608367)
        
        int_608369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 34), 'int')
        
        # Call to exp(...): (line 580)
        # Processing the call arguments (line 580)
        
        # Getting the type of 'lambda_' (line 580)
        lambda__608371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 41), 'lambda_', False)
        # Applying the 'usub' unary operator (line 580)
        result___neg___608372 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 40), 'usub', lambda__608371)
        
        # Getting the type of 'N' (line 580)
        N_608373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 49), 'N', False)
        # Applying the binary operator '*' (line 580)
        result_mul_608374 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 40), '*', result___neg___608372, N_608373)
        
        # Processing the call keyword arguments (line 580)
        kwargs_608375 = {}
        # Getting the type of 'exp' (line 580)
        exp_608370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 36), 'exp', False)
        # Calling exp(args, kwargs) (line 580)
        exp_call_result_608376 = invoke(stypy.reporting.localization.Localization(__file__, 580, 36), exp_608370, *[result_mul_608374], **kwargs_608375)
        
        # Applying the binary operator '-' (line 580)
        result_sub_608377 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 34), '-', int_608369, exp_call_result_608376)
        
        # Applying the binary operator 'div' (line 580)
        result_div_608378 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 15), 'div', result_sub_608368, result_sub_608377)
        
        # Assigning a type to the variable 'fact' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'fact', result_div_608378)
        # Getting the type of 'fact' (line 581)
        fact_608379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'fact')
        
        # Call to exp(...): (line 581)
        # Processing the call arguments (line 581)
        
        # Getting the type of 'lambda_' (line 581)
        lambda__608381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'lambda_', False)
        # Applying the 'usub' unary operator (line 581)
        result___neg___608382 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 24), 'usub', lambda__608381)
        
        # Getting the type of 'k' (line 581)
        k_608383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 33), 'k', False)
        # Applying the binary operator '*' (line 581)
        result_mul_608384 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 24), '*', result___neg___608382, k_608383)
        
        # Processing the call keyword arguments (line 581)
        kwargs_608385 = {}
        # Getting the type of 'exp' (line 581)
        exp_608380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 20), 'exp', False)
        # Calling exp(args, kwargs) (line 581)
        exp_call_result_608386 = invoke(stypy.reporting.localization.Localization(__file__, 581, 20), exp_608380, *[result_mul_608384], **kwargs_608385)
        
        # Applying the binary operator '*' (line 581)
        result_mul_608387 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 15), '*', fact_608379, exp_call_result_608386)
        
        # Assigning a type to the variable 'stypy_return_type' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'stypy_return_type', result_mul_608387)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 579)
        stypy_return_type_608388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608388)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_608388


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 583, 4, False)
        # Assigning a type to the variable 'self' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_function_name', 'boltzmann_gen._cdf')
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'lambda_', 'N'])
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        boltzmann_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boltzmann_gen._cdf', ['x', 'lambda_', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'lambda_', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 584):
        
        # Assigning a Call to a Name (line 584):
        
        # Call to floor(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'x' (line 584)
        x_608390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 18), 'x', False)
        # Processing the call keyword arguments (line 584)
        kwargs_608391 = {}
        # Getting the type of 'floor' (line 584)
        floor_608389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 584)
        floor_call_result_608392 = invoke(stypy.reporting.localization.Localization(__file__, 584, 12), floor_608389, *[x_608390], **kwargs_608391)
        
        # Assigning a type to the variable 'k' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'k', floor_call_result_608392)
        int_608393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 16), 'int')
        
        # Call to exp(...): (line 585)
        # Processing the call arguments (line 585)
        
        # Getting the type of 'lambda_' (line 585)
        lambda__608395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 23), 'lambda_', False)
        # Applying the 'usub' unary operator (line 585)
        result___neg___608396 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 22), 'usub', lambda__608395)
        
        # Getting the type of 'k' (line 585)
        k_608397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 32), 'k', False)
        int_608398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 34), 'int')
        # Applying the binary operator '+' (line 585)
        result_add_608399 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 32), '+', k_608397, int_608398)
        
        # Applying the binary operator '*' (line 585)
        result_mul_608400 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 22), '*', result___neg___608396, result_add_608399)
        
        # Processing the call keyword arguments (line 585)
        kwargs_608401 = {}
        # Getting the type of 'exp' (line 585)
        exp_608394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 18), 'exp', False)
        # Calling exp(args, kwargs) (line 585)
        exp_call_result_608402 = invoke(stypy.reporting.localization.Localization(__file__, 585, 18), exp_608394, *[result_mul_608400], **kwargs_608401)
        
        # Applying the binary operator '-' (line 585)
        result_sub_608403 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 16), '-', int_608393, exp_call_result_608402)
        
        int_608404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 40), 'int')
        
        # Call to exp(...): (line 585)
        # Processing the call arguments (line 585)
        
        # Getting the type of 'lambda_' (line 585)
        lambda__608406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 47), 'lambda_', False)
        # Applying the 'usub' unary operator (line 585)
        result___neg___608407 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 46), 'usub', lambda__608406)
        
        # Getting the type of 'N' (line 585)
        N_608408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 55), 'N', False)
        # Applying the binary operator '*' (line 585)
        result_mul_608409 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 46), '*', result___neg___608407, N_608408)
        
        # Processing the call keyword arguments (line 585)
        kwargs_608410 = {}
        # Getting the type of 'exp' (line 585)
        exp_608405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 42), 'exp', False)
        # Calling exp(args, kwargs) (line 585)
        exp_call_result_608411 = invoke(stypy.reporting.localization.Localization(__file__, 585, 42), exp_608405, *[result_mul_608409], **kwargs_608410)
        
        # Applying the binary operator '-' (line 585)
        result_sub_608412 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 40), '-', int_608404, exp_call_result_608411)
        
        # Applying the binary operator 'div' (line 585)
        result_div_608413 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 15), 'div', result_sub_608403, result_sub_608412)
        
        # Assigning a type to the variable 'stypy_return_type' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'stypy_return_type', result_div_608413)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 583)
        stypy_return_type_608414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608414)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_608414


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 587, 4, False)
        # Assigning a type to the variable 'self' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_function_name', 'boltzmann_gen._ppf')
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'lambda_', 'N'])
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        boltzmann_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boltzmann_gen._ppf', ['q', 'lambda_', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'lambda_', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 588):
        
        # Assigning a BinOp to a Name (line 588):
        # Getting the type of 'q' (line 588)
        q_608415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 15), 'q')
        int_608416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 18), 'int')
        
        # Call to exp(...): (line 588)
        # Processing the call arguments (line 588)
        
        # Getting the type of 'lambda_' (line 588)
        lambda__608418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'lambda_', False)
        # Applying the 'usub' unary operator (line 588)
        result___neg___608419 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 24), 'usub', lambda__608418)
        
        # Getting the type of 'N' (line 588)
        N_608420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 33), 'N', False)
        # Applying the binary operator '*' (line 588)
        result_mul_608421 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 24), '*', result___neg___608419, N_608420)
        
        # Processing the call keyword arguments (line 588)
        kwargs_608422 = {}
        # Getting the type of 'exp' (line 588)
        exp_608417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'exp', False)
        # Calling exp(args, kwargs) (line 588)
        exp_call_result_608423 = invoke(stypy.reporting.localization.Localization(__file__, 588, 20), exp_608417, *[result_mul_608421], **kwargs_608422)
        
        # Applying the binary operator '-' (line 588)
        result_sub_608424 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 18), '-', int_608416, exp_call_result_608423)
        
        # Applying the binary operator '*' (line 588)
        result_mul_608425 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 15), '*', q_608415, result_sub_608424)
        
        # Assigning a type to the variable 'qnew' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'qnew', result_mul_608425)
        
        # Assigning a Call to a Name (line 589):
        
        # Assigning a Call to a Name (line 589):
        
        # Call to ceil(...): (line 589)
        # Processing the call arguments (line 589)
        float_608427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 20), 'float')
        # Getting the type of 'lambda_' (line 589)
        lambda__608428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 25), 'lambda_', False)
        # Applying the binary operator 'div' (line 589)
        result_div_608429 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 20), 'div', float_608427, lambda__608428)
        
        
        # Call to log(...): (line 589)
        # Processing the call arguments (line 589)
        int_608431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 39), 'int')
        # Getting the type of 'qnew' (line 589)
        qnew_608432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 41), 'qnew', False)
        # Applying the binary operator '-' (line 589)
        result_sub_608433 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 39), '-', int_608431, qnew_608432)
        
        # Processing the call keyword arguments (line 589)
        kwargs_608434 = {}
        # Getting the type of 'log' (line 589)
        log_608430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 35), 'log', False)
        # Calling log(args, kwargs) (line 589)
        log_call_result_608435 = invoke(stypy.reporting.localization.Localization(__file__, 589, 35), log_608430, *[result_sub_608433], **kwargs_608434)
        
        # Applying the binary operator '*' (line 589)
        result_mul_608436 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 33), '*', result_div_608429, log_call_result_608435)
        
        int_608437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 47), 'int')
        # Applying the binary operator '-' (line 589)
        result_sub_608438 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 20), '-', result_mul_608436, int_608437)
        
        # Processing the call keyword arguments (line 589)
        kwargs_608439 = {}
        # Getting the type of 'ceil' (line 589)
        ceil_608426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 15), 'ceil', False)
        # Calling ceil(args, kwargs) (line 589)
        ceil_call_result_608440 = invoke(stypy.reporting.localization.Localization(__file__, 589, 15), ceil_608426, *[result_sub_608438], **kwargs_608439)
        
        # Assigning a type to the variable 'vals' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'vals', ceil_call_result_608440)
        
        # Assigning a Call to a Name (line 590):
        
        # Assigning a Call to a Name (line 590):
        
        # Call to clip(...): (line 590)
        # Processing the call arguments (line 590)
        float_608445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 30), 'float')
        # Getting the type of 'np' (line 590)
        np_608446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 35), 'np', False)
        # Obtaining the member 'inf' of a type (line 590)
        inf_608447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 35), np_608446, 'inf')
        # Processing the call keyword arguments (line 590)
        kwargs_608448 = {}
        # Getting the type of 'vals' (line 590)
        vals_608441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 17), 'vals', False)
        int_608442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 22), 'int')
        # Applying the binary operator '-' (line 590)
        result_sub_608443 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 17), '-', vals_608441, int_608442)
        
        # Obtaining the member 'clip' of a type (line 590)
        clip_608444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 17), result_sub_608443, 'clip')
        # Calling clip(args, kwargs) (line 590)
        clip_call_result_608449 = invoke(stypy.reporting.localization.Localization(__file__, 590, 17), clip_608444, *[float_608445, inf_608447], **kwargs_608448)
        
        # Assigning a type to the variable 'vals1' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'vals1', clip_call_result_608449)
        
        # Assigning a Call to a Name (line 591):
        
        # Assigning a Call to a Name (line 591):
        
        # Call to _cdf(...): (line 591)
        # Processing the call arguments (line 591)
        # Getting the type of 'vals1' (line 591)
        vals1_608452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 25), 'vals1', False)
        # Getting the type of 'lambda_' (line 591)
        lambda__608453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 32), 'lambda_', False)
        # Getting the type of 'N' (line 591)
        N_608454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 41), 'N', False)
        # Processing the call keyword arguments (line 591)
        kwargs_608455 = {}
        # Getting the type of 'self' (line 591)
        self_608450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'self', False)
        # Obtaining the member '_cdf' of a type (line 591)
        _cdf_608451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 15), self_608450, '_cdf')
        # Calling _cdf(args, kwargs) (line 591)
        _cdf_call_result_608456 = invoke(stypy.reporting.localization.Localization(__file__, 591, 15), _cdf_608451, *[vals1_608452, lambda__608453, N_608454], **kwargs_608455)
        
        # Assigning a type to the variable 'temp' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'temp', _cdf_call_result_608456)
        
        # Call to where(...): (line 592)
        # Processing the call arguments (line 592)
        
        # Getting the type of 'temp' (line 592)
        temp_608459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 24), 'temp', False)
        # Getting the type of 'q' (line 592)
        q_608460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 32), 'q', False)
        # Applying the binary operator '>=' (line 592)
        result_ge_608461 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 24), '>=', temp_608459, q_608460)
        
        # Getting the type of 'vals1' (line 592)
        vals1_608462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 35), 'vals1', False)
        # Getting the type of 'vals' (line 592)
        vals_608463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 42), 'vals', False)
        # Processing the call keyword arguments (line 592)
        kwargs_608464 = {}
        # Getting the type of 'np' (line 592)
        np_608457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 592)
        where_608458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 15), np_608457, 'where')
        # Calling where(args, kwargs) (line 592)
        where_call_result_608465 = invoke(stypy.reporting.localization.Localization(__file__, 592, 15), where_608458, *[result_ge_608461, vals1_608462, vals_608463], **kwargs_608464)
        
        # Assigning a type to the variable 'stypy_return_type' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'stypy_return_type', where_call_result_608465)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 587)
        stypy_return_type_608466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608466)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_608466


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 594, 4, False)
        # Assigning a type to the variable 'self' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        boltzmann_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        boltzmann_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        boltzmann_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        boltzmann_gen._stats.__dict__.__setitem__('stypy_function_name', 'boltzmann_gen._stats')
        boltzmann_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['lambda_', 'N'])
        boltzmann_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        boltzmann_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        boltzmann_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        boltzmann_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        boltzmann_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        boltzmann_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boltzmann_gen._stats', ['lambda_', 'N'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['lambda_', 'N'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a Call to a Name (line 595):
        
        # Assigning a Call to a Name (line 595):
        
        # Call to exp(...): (line 595)
        # Processing the call arguments (line 595)
        
        # Getting the type of 'lambda_' (line 595)
        lambda__608468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 17), 'lambda_', False)
        # Applying the 'usub' unary operator (line 595)
        result___neg___608469 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 16), 'usub', lambda__608468)
        
        # Processing the call keyword arguments (line 595)
        kwargs_608470 = {}
        # Getting the type of 'exp' (line 595)
        exp_608467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'exp', False)
        # Calling exp(args, kwargs) (line 595)
        exp_call_result_608471 = invoke(stypy.reporting.localization.Localization(__file__, 595, 12), exp_608467, *[result___neg___608469], **kwargs_608470)
        
        # Assigning a type to the variable 'z' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'z', exp_call_result_608471)
        
        # Assigning a Call to a Name (line 596):
        
        # Assigning a Call to a Name (line 596):
        
        # Call to exp(...): (line 596)
        # Processing the call arguments (line 596)
        
        # Getting the type of 'lambda_' (line 596)
        lambda__608473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 18), 'lambda_', False)
        # Applying the 'usub' unary operator (line 596)
        result___neg___608474 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 17), 'usub', lambda__608473)
        
        # Getting the type of 'N' (line 596)
        N_608475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 26), 'N', False)
        # Applying the binary operator '*' (line 596)
        result_mul_608476 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 17), '*', result___neg___608474, N_608475)
        
        # Processing the call keyword arguments (line 596)
        kwargs_608477 = {}
        # Getting the type of 'exp' (line 596)
        exp_608472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 13), 'exp', False)
        # Calling exp(args, kwargs) (line 596)
        exp_call_result_608478 = invoke(stypy.reporting.localization.Localization(__file__, 596, 13), exp_608472, *[result_mul_608476], **kwargs_608477)
        
        # Assigning a type to the variable 'zN' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'zN', exp_call_result_608478)
        
        # Assigning a BinOp to a Name (line 597):
        
        # Assigning a BinOp to a Name (line 597):
        # Getting the type of 'z' (line 597)
        z_608479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 13), 'z')
        float_608480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 16), 'float')
        # Getting the type of 'z' (line 597)
        z_608481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 20), 'z')
        # Applying the binary operator '-' (line 597)
        result_sub_608482 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 16), '-', float_608480, z_608481)
        
        # Applying the binary operator 'div' (line 597)
        result_div_608483 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 13), 'div', z_608479, result_sub_608482)
        
        # Getting the type of 'N' (line 597)
        N_608484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 23), 'N')
        # Getting the type of 'zN' (line 597)
        zN_608485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 25), 'zN')
        # Applying the binary operator '*' (line 597)
        result_mul_608486 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 23), '*', N_608484, zN_608485)
        
        int_608487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 29), 'int')
        # Getting the type of 'zN' (line 597)
        zN_608488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 31), 'zN')
        # Applying the binary operator '-' (line 597)
        result_sub_608489 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 29), '-', int_608487, zN_608488)
        
        # Applying the binary operator 'div' (line 597)
        result_div_608490 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 27), 'div', result_mul_608486, result_sub_608489)
        
        # Applying the binary operator '-' (line 597)
        result_sub_608491 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 13), '-', result_div_608483, result_div_608490)
        
        # Assigning a type to the variable 'mu' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'mu', result_sub_608491)
        
        # Assigning a BinOp to a Name (line 598):
        
        # Assigning a BinOp to a Name (line 598):
        # Getting the type of 'z' (line 598)
        z_608492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 14), 'z')
        float_608493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 17), 'float')
        # Getting the type of 'z' (line 598)
        z_608494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 21), 'z')
        # Applying the binary operator '-' (line 598)
        result_sub_608495 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 17), '-', float_608493, z_608494)
        
        int_608496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 25), 'int')
        # Applying the binary operator '**' (line 598)
        result_pow_608497 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 16), '**', result_sub_608495, int_608496)
        
        # Applying the binary operator 'div' (line 598)
        result_div_608498 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 14), 'div', z_608492, result_pow_608497)
        
        # Getting the type of 'N' (line 598)
        N_608499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 29), 'N')
        # Getting the type of 'N' (line 598)
        N_608500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 31), 'N')
        # Applying the binary operator '*' (line 598)
        result_mul_608501 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 29), '*', N_608499, N_608500)
        
        # Getting the type of 'zN' (line 598)
        zN_608502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 33), 'zN')
        # Applying the binary operator '*' (line 598)
        result_mul_608503 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 32), '*', result_mul_608501, zN_608502)
        
        int_608504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 37), 'int')
        # Getting the type of 'zN' (line 598)
        zN_608505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 39), 'zN')
        # Applying the binary operator '-' (line 598)
        result_sub_608506 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 37), '-', int_608504, zN_608505)
        
        int_608507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 44), 'int')
        # Applying the binary operator '**' (line 598)
        result_pow_608508 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 36), '**', result_sub_608506, int_608507)
        
        # Applying the binary operator 'div' (line 598)
        result_div_608509 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 35), 'div', result_mul_608503, result_pow_608508)
        
        # Applying the binary operator '-' (line 598)
        result_sub_608510 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 14), '-', result_div_608498, result_div_608509)
        
        # Assigning a type to the variable 'var' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'var', result_sub_608510)
        
        # Assigning a BinOp to a Name (line 599):
        
        # Assigning a BinOp to a Name (line 599):
        int_608511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 15), 'int')
        # Getting the type of 'zN' (line 599)
        zN_608512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'zN')
        # Applying the binary operator '-' (line 599)
        result_sub_608513 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 15), '-', int_608511, zN_608512)
        
        int_608514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 22), 'int')
        # Getting the type of 'z' (line 599)
        z_608515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 24), 'z')
        # Applying the binary operator '-' (line 599)
        result_sub_608516 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 22), '-', int_608514, z_608515)
        
        # Applying the binary operator 'div' (line 599)
        result_div_608517 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 14), 'div', result_sub_608513, result_sub_608516)
        
        # Assigning a type to the variable 'trm' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'trm', result_div_608517)
        
        # Assigning a BinOp to a Name (line 600):
        
        # Assigning a BinOp to a Name (line 600):
        # Getting the type of 'z' (line 600)
        z_608518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'z')
        # Getting the type of 'trm' (line 600)
        trm_608519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 18), 'trm')
        int_608520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 23), 'int')
        # Applying the binary operator '**' (line 600)
        result_pow_608521 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 18), '**', trm_608519, int_608520)
        
        # Applying the binary operator '*' (line 600)
        result_mul_608522 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 16), '*', z_608518, result_pow_608521)
        
        # Getting the type of 'N' (line 600)
        N_608523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 27), 'N')
        # Getting the type of 'N' (line 600)
        N_608524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 29), 'N')
        # Applying the binary operator '*' (line 600)
        result_mul_608525 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 27), '*', N_608523, N_608524)
        
        # Getting the type of 'zN' (line 600)
        zN_608526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 31), 'zN')
        # Applying the binary operator '*' (line 600)
        result_mul_608527 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 30), '*', result_mul_608525, zN_608526)
        
        # Applying the binary operator '-' (line 600)
        result_sub_608528 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 16), '-', result_mul_608522, result_mul_608527)
        
        # Assigning a type to the variable 'trm2' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'trm2', result_sub_608528)
        
        # Assigning a BinOp to a Name (line 601):
        
        # Assigning a BinOp to a Name (line 601):
        # Getting the type of 'z' (line 601)
        z_608529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 13), 'z')
        int_608530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 16), 'int')
        # Getting the type of 'z' (line 601)
        z_608531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 18), 'z')
        # Applying the binary operator '+' (line 601)
        result_add_608532 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 16), '+', int_608530, z_608531)
        
        # Applying the binary operator '*' (line 601)
        result_mul_608533 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 13), '*', z_608529, result_add_608532)
        
        # Getting the type of 'trm' (line 601)
        trm_608534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 21), 'trm')
        int_608535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 26), 'int')
        # Applying the binary operator '**' (line 601)
        result_pow_608536 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 21), '**', trm_608534, int_608535)
        
        # Applying the binary operator '*' (line 601)
        result_mul_608537 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 20), '*', result_mul_608533, result_pow_608536)
        
        # Getting the type of 'N' (line 601)
        N_608538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 30), 'N')
        int_608539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 33), 'int')
        # Applying the binary operator '**' (line 601)
        result_pow_608540 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 30), '**', N_608538, int_608539)
        
        # Getting the type of 'zN' (line 601)
        zN_608541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 35), 'zN')
        # Applying the binary operator '*' (line 601)
        result_mul_608542 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 30), '*', result_pow_608540, zN_608541)
        
        int_608543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 39), 'int')
        # Getting the type of 'zN' (line 601)
        zN_608544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 41), 'zN')
        # Applying the binary operator '+' (line 601)
        result_add_608545 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 39), '+', int_608543, zN_608544)
        
        # Applying the binary operator '*' (line 601)
        result_mul_608546 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 37), '*', result_mul_608542, result_add_608545)
        
        # Applying the binary operator '-' (line 601)
        result_sub_608547 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 13), '-', result_mul_608537, result_mul_608546)
        
        # Assigning a type to the variable 'g1' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'g1', result_sub_608547)
        
        # Assigning a BinOp to a Name (line 602):
        
        # Assigning a BinOp to a Name (line 602):
        # Getting the type of 'g1' (line 602)
        g1_608548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 13), 'g1')
        # Getting the type of 'trm2' (line 602)
        trm2_608549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 18), 'trm2')
        float_608550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 25), 'float')
        # Applying the binary operator '**' (line 602)
        result_pow_608551 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 18), '**', trm2_608549, float_608550)
        
        # Applying the binary operator 'div' (line 602)
        result_div_608552 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 13), 'div', g1_608548, result_pow_608551)
        
        # Assigning a type to the variable 'g1' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'g1', result_div_608552)
        
        # Assigning a BinOp to a Name (line 603):
        
        # Assigning a BinOp to a Name (line 603):
        # Getting the type of 'z' (line 603)
        z_608553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 13), 'z')
        int_608554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 16), 'int')
        int_608555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 18), 'int')
        # Getting the type of 'z' (line 603)
        z_608556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 20), 'z')
        # Applying the binary operator '*' (line 603)
        result_mul_608557 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 18), '*', int_608555, z_608556)
        
        # Applying the binary operator '+' (line 603)
        result_add_608558 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 16), '+', int_608554, result_mul_608557)
        
        # Getting the type of 'z' (line 603)
        z_608559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 22), 'z')
        # Getting the type of 'z' (line 603)
        z_608560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 24), 'z')
        # Applying the binary operator '*' (line 603)
        result_mul_608561 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 22), '*', z_608559, z_608560)
        
        # Applying the binary operator '+' (line 603)
        result_add_608562 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 21), '+', result_add_608558, result_mul_608561)
        
        # Applying the binary operator '*' (line 603)
        result_mul_608563 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 13), '*', z_608553, result_add_608562)
        
        # Getting the type of 'trm' (line 603)
        trm_608564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 27), 'trm')
        int_608565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 32), 'int')
        # Applying the binary operator '**' (line 603)
        result_pow_608566 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 27), '**', trm_608564, int_608565)
        
        # Applying the binary operator '*' (line 603)
        result_mul_608567 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 26), '*', result_mul_608563, result_pow_608566)
        
        # Getting the type of 'N' (line 603)
        N_608568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 36), 'N')
        int_608569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 39), 'int')
        # Applying the binary operator '**' (line 603)
        result_pow_608570 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 36), '**', N_608568, int_608569)
        
        # Getting the type of 'zN' (line 603)
        zN_608571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 43), 'zN')
        # Applying the binary operator '*' (line 603)
        result_mul_608572 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 36), '*', result_pow_608570, zN_608571)
        
        int_608573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 47), 'int')
        int_608574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 49), 'int')
        # Getting the type of 'zN' (line 603)
        zN_608575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 51), 'zN')
        # Applying the binary operator '*' (line 603)
        result_mul_608576 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 49), '*', int_608574, zN_608575)
        
        # Applying the binary operator '+' (line 603)
        result_add_608577 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 47), '+', int_608573, result_mul_608576)
        
        # Getting the type of 'zN' (line 603)
        zN_608578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 54), 'zN')
        # Getting the type of 'zN' (line 603)
        zN_608579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 57), 'zN')
        # Applying the binary operator '*' (line 603)
        result_mul_608580 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 54), '*', zN_608578, zN_608579)
        
        # Applying the binary operator '+' (line 603)
        result_add_608581 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 53), '+', result_add_608577, result_mul_608580)
        
        # Applying the binary operator '*' (line 603)
        result_mul_608582 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 45), '*', result_mul_608572, result_add_608581)
        
        # Applying the binary operator '-' (line 603)
        result_sub_608583 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 13), '-', result_mul_608567, result_mul_608582)
        
        # Assigning a type to the variable 'g2' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'g2', result_sub_608583)
        
        # Assigning a BinOp to a Name (line 604):
        
        # Assigning a BinOp to a Name (line 604):
        # Getting the type of 'g2' (line 604)
        g2_608584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 13), 'g2')
        # Getting the type of 'trm2' (line 604)
        trm2_608585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 18), 'trm2')
        # Applying the binary operator 'div' (line 604)
        result_div_608586 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 13), 'div', g2_608584, trm2_608585)
        
        # Getting the type of 'trm2' (line 604)
        trm2_608587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 25), 'trm2')
        # Applying the binary operator 'div' (line 604)
        result_div_608588 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 23), 'div', result_div_608586, trm2_608587)
        
        # Assigning a type to the variable 'g2' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'g2', result_div_608588)
        
        # Obtaining an instance of the builtin type 'tuple' (line 605)
        tuple_608589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 605)
        # Adding element type (line 605)
        # Getting the type of 'mu' (line 605)
        mu_608590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 15), tuple_608589, mu_608590)
        # Adding element type (line 605)
        # Getting the type of 'var' (line 605)
        var_608591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 15), tuple_608589, var_608591)
        # Adding element type (line 605)
        # Getting the type of 'g1' (line 605)
        g1_608592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 15), tuple_608589, g1_608592)
        # Adding element type (line 605)
        # Getting the type of 'g2' (line 605)
        g2_608593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 15), tuple_608589, g2_608593)
        
        # Assigning a type to the variable 'stypy_return_type' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'stypy_return_type', tuple_608589)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 594)
        stypy_return_type_608594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_608594


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 559, 0, False)
        # Assigning a type to the variable 'self' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'boltzmann_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'boltzmann_gen' (line 559)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 0), 'boltzmann_gen', boltzmann_gen)

# Assigning a Call to a Name (line 606):

# Assigning a Call to a Name (line 606):

# Call to boltzmann_gen(...): (line 606)
# Processing the call keyword arguments (line 606)
str_608596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 31), 'str', 'boltzmann')
keyword_608597 = str_608596
str_608598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 17), 'str', 'A truncated discrete exponential ')
keyword_608599 = str_608598
kwargs_608600 = {'name': keyword_608597, 'longname': keyword_608599}
# Getting the type of 'boltzmann_gen' (line 606)
boltzmann_gen_608595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'boltzmann_gen', False)
# Calling boltzmann_gen(args, kwargs) (line 606)
boltzmann_gen_call_result_608601 = invoke(stypy.reporting.localization.Localization(__file__, 606, 12), boltzmann_gen_608595, *[], **kwargs_608600)

# Assigning a type to the variable 'boltzmann' (line 606)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 0), 'boltzmann', boltzmann_gen_call_result_608601)
# Declaration of the 'randint_gen' class
# Getting the type of 'rv_discrete' (line 610)
rv_discrete_608602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 18), 'rv_discrete')

class randint_gen(rv_discrete_608602, ):
    str_608603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, (-1)), 'str', 'A uniform discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `randint` is::\n\n        randint.pmf(k) = 1./(high - low)\n\n    for ``k = low, ..., high - 1``.\n\n    `randint` takes ``low`` and ``high`` as shape parameters.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 630, 4, False)
        # Assigning a type to the variable 'self' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        randint_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        randint_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        randint_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        randint_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'randint_gen._argcheck')
        randint_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['low', 'high'])
        randint_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        randint_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        randint_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        randint_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        randint_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        randint_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'randint_gen._argcheck', ['low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Assigning a Name to a Attribute (line 631):
        
        # Assigning a Name to a Attribute (line 631):
        # Getting the type of 'low' (line 631)
        low_608604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 17), 'low')
        # Getting the type of 'self' (line 631)
        self_608605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'self')
        # Setting the type of the member 'a' of a type (line 631)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 8), self_608605, 'a', low_608604)
        
        # Assigning a BinOp to a Attribute (line 632):
        
        # Assigning a BinOp to a Attribute (line 632):
        # Getting the type of 'high' (line 632)
        high_608606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 17), 'high')
        int_608607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 24), 'int')
        # Applying the binary operator '-' (line 632)
        result_sub_608608 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 17), '-', high_608606, int_608607)
        
        # Getting the type of 'self' (line 632)
        self_608609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'self')
        # Setting the type of the member 'b' of a type (line 632)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 8), self_608609, 'b', result_sub_608608)
        
        # Getting the type of 'high' (line 633)
        high_608610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'high')
        # Getting the type of 'low' (line 633)
        low_608611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 23), 'low')
        # Applying the binary operator '>' (line 633)
        result_gt_608612 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 16), '>', high_608610, low_608611)
        
        # Assigning a type to the variable 'stypy_return_type' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'stypy_return_type', result_gt_608612)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 630)
        stypy_return_type_608613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608613)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_608613


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 635, 4, False)
        # Assigning a type to the variable 'self' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        randint_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        randint_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        randint_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        randint_gen._pmf.__dict__.__setitem__('stypy_function_name', 'randint_gen._pmf')
        randint_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'low', 'high'])
        randint_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        randint_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        randint_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        randint_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        randint_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        randint_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'randint_gen._pmf', ['k', 'low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 636):
        
        # Assigning a BinOp to a Name (line 636):
        
        # Call to ones_like(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 'k' (line 636)
        k_608616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 25), 'k', False)
        # Processing the call keyword arguments (line 636)
        kwargs_608617 = {}
        # Getting the type of 'np' (line 636)
        np_608614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'np', False)
        # Obtaining the member 'ones_like' of a type (line 636)
        ones_like_608615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), np_608614, 'ones_like')
        # Calling ones_like(args, kwargs) (line 636)
        ones_like_call_result_608618 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), ones_like_608615, *[k_608616], **kwargs_608617)
        
        # Getting the type of 'high' (line 636)
        high_608619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 31), 'high')
        # Getting the type of 'low' (line 636)
        low_608620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 38), 'low')
        # Applying the binary operator '-' (line 636)
        result_sub_608621 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 31), '-', high_608619, low_608620)
        
        # Applying the binary operator 'div' (line 636)
        result_div_608622 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 12), 'div', ones_like_call_result_608618, result_sub_608621)
        
        # Assigning a type to the variable 'p' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'p', result_div_608622)
        
        # Call to where(...): (line 637)
        # Processing the call arguments (line 637)
        
        # Getting the type of 'k' (line 637)
        k_608625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 25), 'k', False)
        # Getting the type of 'low' (line 637)
        low_608626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 30), 'low', False)
        # Applying the binary operator '>=' (line 637)
        result_ge_608627 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 25), '>=', k_608625, low_608626)
        
        
        # Getting the type of 'k' (line 637)
        k_608628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 38), 'k', False)
        # Getting the type of 'high' (line 637)
        high_608629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 42), 'high', False)
        # Applying the binary operator '<' (line 637)
        result_lt_608630 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 38), '<', k_608628, high_608629)
        
        # Applying the binary operator '&' (line 637)
        result_and__608631 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 24), '&', result_ge_608627, result_lt_608630)
        
        # Getting the type of 'p' (line 637)
        p_608632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 49), 'p', False)
        float_608633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 52), 'float')
        # Processing the call keyword arguments (line 637)
        kwargs_608634 = {}
        # Getting the type of 'np' (line 637)
        np_608623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 637)
        where_608624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 15), np_608623, 'where')
        # Calling where(args, kwargs) (line 637)
        where_call_result_608635 = invoke(stypy.reporting.localization.Localization(__file__, 637, 15), where_608624, *[result_and__608631, p_608632, float_608633], **kwargs_608634)
        
        # Assigning a type to the variable 'stypy_return_type' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'stypy_return_type', where_call_result_608635)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 635)
        stypy_return_type_608636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608636)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_608636


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 639, 4, False)
        # Assigning a type to the variable 'self' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        randint_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        randint_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        randint_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        randint_gen._cdf.__dict__.__setitem__('stypy_function_name', 'randint_gen._cdf')
        randint_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'low', 'high'])
        randint_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        randint_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        randint_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        randint_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        randint_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        randint_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'randint_gen._cdf', ['x', 'low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 640):
        
        # Assigning a Call to a Name (line 640):
        
        # Call to floor(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'x' (line 640)
        x_608638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 18), 'x', False)
        # Processing the call keyword arguments (line 640)
        kwargs_608639 = {}
        # Getting the type of 'floor' (line 640)
        floor_608637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 640)
        floor_call_result_608640 = invoke(stypy.reporting.localization.Localization(__file__, 640, 12), floor_608637, *[x_608638], **kwargs_608639)
        
        # Assigning a type to the variable 'k' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'k', floor_call_result_608640)
        # Getting the type of 'k' (line 641)
        k_608641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 16), 'k')
        # Getting the type of 'low' (line 641)
        low_608642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 20), 'low')
        # Applying the binary operator '-' (line 641)
        result_sub_608643 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 16), '-', k_608641, low_608642)
        
        float_608644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 26), 'float')
        # Applying the binary operator '+' (line 641)
        result_add_608645 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 24), '+', result_sub_608643, float_608644)
        
        # Getting the type of 'high' (line 641)
        high_608646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 33), 'high')
        # Getting the type of 'low' (line 641)
        low_608647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 40), 'low')
        # Applying the binary operator '-' (line 641)
        result_sub_608648 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 33), '-', high_608646, low_608647)
        
        # Applying the binary operator 'div' (line 641)
        result_div_608649 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 15), 'div', result_add_608645, result_sub_608648)
        
        # Assigning a type to the variable 'stypy_return_type' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'stypy_return_type', result_div_608649)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 639)
        stypy_return_type_608650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608650)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_608650


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 643, 4, False)
        # Assigning a type to the variable 'self' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        randint_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        randint_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        randint_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        randint_gen._ppf.__dict__.__setitem__('stypy_function_name', 'randint_gen._ppf')
        randint_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'low', 'high'])
        randint_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        randint_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        randint_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        randint_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        randint_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        randint_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'randint_gen._ppf', ['q', 'low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 644):
        
        # Assigning a BinOp to a Name (line 644):
        
        # Call to ceil(...): (line 644)
        # Processing the call arguments (line 644)
        # Getting the type of 'q' (line 644)
        q_608652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 20), 'q', False)
        # Getting the type of 'high' (line 644)
        high_608653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 25), 'high', False)
        # Getting the type of 'low' (line 644)
        low_608654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 32), 'low', False)
        # Applying the binary operator '-' (line 644)
        result_sub_608655 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 25), '-', high_608653, low_608654)
        
        # Applying the binary operator '*' (line 644)
        result_mul_608656 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 20), '*', q_608652, result_sub_608655)
        
        # Getting the type of 'low' (line 644)
        low_608657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 39), 'low', False)
        # Applying the binary operator '+' (line 644)
        result_add_608658 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 20), '+', result_mul_608656, low_608657)
        
        # Processing the call keyword arguments (line 644)
        kwargs_608659 = {}
        # Getting the type of 'ceil' (line 644)
        ceil_608651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 15), 'ceil', False)
        # Calling ceil(args, kwargs) (line 644)
        ceil_call_result_608660 = invoke(stypy.reporting.localization.Localization(__file__, 644, 15), ceil_608651, *[result_add_608658], **kwargs_608659)
        
        int_608661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 46), 'int')
        # Applying the binary operator '-' (line 644)
        result_sub_608662 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 15), '-', ceil_call_result_608660, int_608661)
        
        # Assigning a type to the variable 'vals' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'vals', result_sub_608662)
        
        # Assigning a Call to a Name (line 645):
        
        # Assigning a Call to a Name (line 645):
        
        # Call to clip(...): (line 645)
        # Processing the call arguments (line 645)
        # Getting the type of 'low' (line 645)
        low_608667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 32), 'low', False)
        # Getting the type of 'high' (line 645)
        high_608668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 37), 'high', False)
        # Processing the call keyword arguments (line 645)
        kwargs_608669 = {}
        # Getting the type of 'vals' (line 645)
        vals_608663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 17), 'vals', False)
        int_608664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 24), 'int')
        # Applying the binary operator '-' (line 645)
        result_sub_608665 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 17), '-', vals_608663, int_608664)
        
        # Obtaining the member 'clip' of a type (line 645)
        clip_608666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 17), result_sub_608665, 'clip')
        # Calling clip(args, kwargs) (line 645)
        clip_call_result_608670 = invoke(stypy.reporting.localization.Localization(__file__, 645, 17), clip_608666, *[low_608667, high_608668], **kwargs_608669)
        
        # Assigning a type to the variable 'vals1' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'vals1', clip_call_result_608670)
        
        # Assigning a Call to a Name (line 646):
        
        # Assigning a Call to a Name (line 646):
        
        # Call to _cdf(...): (line 646)
        # Processing the call arguments (line 646)
        # Getting the type of 'vals1' (line 646)
        vals1_608673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 25), 'vals1', False)
        # Getting the type of 'low' (line 646)
        low_608674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 32), 'low', False)
        # Getting the type of 'high' (line 646)
        high_608675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 37), 'high', False)
        # Processing the call keyword arguments (line 646)
        kwargs_608676 = {}
        # Getting the type of 'self' (line 646)
        self_608671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 15), 'self', False)
        # Obtaining the member '_cdf' of a type (line 646)
        _cdf_608672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 15), self_608671, '_cdf')
        # Calling _cdf(args, kwargs) (line 646)
        _cdf_call_result_608677 = invoke(stypy.reporting.localization.Localization(__file__, 646, 15), _cdf_608672, *[vals1_608673, low_608674, high_608675], **kwargs_608676)
        
        # Assigning a type to the variable 'temp' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'temp', _cdf_call_result_608677)
        
        # Call to where(...): (line 647)
        # Processing the call arguments (line 647)
        
        # Getting the type of 'temp' (line 647)
        temp_608680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 24), 'temp', False)
        # Getting the type of 'q' (line 647)
        q_608681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 32), 'q', False)
        # Applying the binary operator '>=' (line 647)
        result_ge_608682 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 24), '>=', temp_608680, q_608681)
        
        # Getting the type of 'vals1' (line 647)
        vals1_608683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 35), 'vals1', False)
        # Getting the type of 'vals' (line 647)
        vals_608684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 42), 'vals', False)
        # Processing the call keyword arguments (line 647)
        kwargs_608685 = {}
        # Getting the type of 'np' (line 647)
        np_608678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 647)
        where_608679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 15), np_608678, 'where')
        # Calling where(args, kwargs) (line 647)
        where_call_result_608686 = invoke(stypy.reporting.localization.Localization(__file__, 647, 15), where_608679, *[result_ge_608682, vals1_608683, vals_608684], **kwargs_608685)
        
        # Assigning a type to the variable 'stypy_return_type' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'stypy_return_type', where_call_result_608686)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 643)
        stypy_return_type_608687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_608687


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 649, 4, False)
        # Assigning a type to the variable 'self' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        randint_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        randint_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        randint_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        randint_gen._stats.__dict__.__setitem__('stypy_function_name', 'randint_gen._stats')
        randint_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['low', 'high'])
        randint_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        randint_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        randint_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        randint_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        randint_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        randint_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'randint_gen._stats', ['low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 650):
        
        # Assigning a Call to a Name (line 650):
        
        # Call to asarray(...): (line 650)
        # Processing the call arguments (line 650)
        # Getting the type of 'high' (line 650)
        high_608690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 28), 'high', False)
        # Processing the call keyword arguments (line 650)
        kwargs_608691 = {}
        # Getting the type of 'np' (line 650)
        np_608688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 17), 'np', False)
        # Obtaining the member 'asarray' of a type (line 650)
        asarray_608689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 17), np_608688, 'asarray')
        # Calling asarray(args, kwargs) (line 650)
        asarray_call_result_608692 = invoke(stypy.reporting.localization.Localization(__file__, 650, 17), asarray_608689, *[high_608690], **kwargs_608691)
        
        # Assigning a type to the variable 'tuple_assignment_606754' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'tuple_assignment_606754', asarray_call_result_608692)
        
        # Assigning a Call to a Name (line 650):
        
        # Call to asarray(...): (line 650)
        # Processing the call arguments (line 650)
        # Getting the type of 'low' (line 650)
        low_608695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 46), 'low', False)
        # Processing the call keyword arguments (line 650)
        kwargs_608696 = {}
        # Getting the type of 'np' (line 650)
        np_608693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 35), 'np', False)
        # Obtaining the member 'asarray' of a type (line 650)
        asarray_608694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 35), np_608693, 'asarray')
        # Calling asarray(args, kwargs) (line 650)
        asarray_call_result_608697 = invoke(stypy.reporting.localization.Localization(__file__, 650, 35), asarray_608694, *[low_608695], **kwargs_608696)
        
        # Assigning a type to the variable 'tuple_assignment_606755' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'tuple_assignment_606755', asarray_call_result_608697)
        
        # Assigning a Name to a Name (line 650):
        # Getting the type of 'tuple_assignment_606754' (line 650)
        tuple_assignment_606754_608698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'tuple_assignment_606754')
        # Assigning a type to the variable 'm2' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'm2', tuple_assignment_606754_608698)
        
        # Assigning a Name to a Name (line 650):
        # Getting the type of 'tuple_assignment_606755' (line 650)
        tuple_assignment_606755_608699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'tuple_assignment_606755')
        # Assigning a type to the variable 'm1' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'm1', tuple_assignment_606755_608699)
        
        # Assigning a BinOp to a Name (line 651):
        
        # Assigning a BinOp to a Name (line 651):
        # Getting the type of 'm2' (line 651)
        m2_608700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 14), 'm2')
        # Getting the type of 'm1' (line 651)
        m1_608701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 19), 'm1')
        # Applying the binary operator '+' (line 651)
        result_add_608702 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 14), '+', m2_608700, m1_608701)
        
        float_608703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 24), 'float')
        # Applying the binary operator '-' (line 651)
        result_sub_608704 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 22), '-', result_add_608702, float_608703)
        
        int_608705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 31), 'int')
        # Applying the binary operator 'div' (line 651)
        result_div_608706 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 13), 'div', result_sub_608704, int_608705)
        
        # Assigning a type to the variable 'mu' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'mu', result_div_608706)
        
        # Assigning a BinOp to a Name (line 652):
        
        # Assigning a BinOp to a Name (line 652):
        # Getting the type of 'm2' (line 652)
        m2_608707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'm2')
        # Getting the type of 'm1' (line 652)
        m1_608708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 17), 'm1')
        # Applying the binary operator '-' (line 652)
        result_sub_608709 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 12), '-', m2_608707, m1_608708)
        
        # Assigning a type to the variable 'd' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'd', result_sub_608709)
        
        # Assigning a BinOp to a Name (line 653):
        
        # Assigning a BinOp to a Name (line 653):
        # Getting the type of 'd' (line 653)
        d_608710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 15), 'd')
        # Getting the type of 'd' (line 653)
        d_608711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 17), 'd')
        # Applying the binary operator '*' (line 653)
        result_mul_608712 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 15), '*', d_608710, d_608711)
        
        int_608713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 21), 'int')
        # Applying the binary operator '-' (line 653)
        result_sub_608714 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 15), '-', result_mul_608712, int_608713)
        
        float_608715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 26), 'float')
        # Applying the binary operator 'div' (line 653)
        result_div_608716 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 14), 'div', result_sub_608714, float_608715)
        
        # Assigning a type to the variable 'var' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'var', result_div_608716)
        
        # Assigning a Num to a Name (line 654):
        
        # Assigning a Num to a Name (line 654):
        float_608717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 13), 'float')
        # Assigning a type to the variable 'g1' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'g1', float_608717)
        
        # Assigning a BinOp to a Name (line 655):
        
        # Assigning a BinOp to a Name (line 655):
        float_608718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 13), 'float')
        float_608719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 18), 'float')
        # Applying the binary operator 'div' (line 655)
        result_div_608720 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 13), 'div', float_608718, float_608719)
        
        # Getting the type of 'd' (line 655)
        d_608721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 25), 'd')
        # Getting the type of 'd' (line 655)
        d_608722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 27), 'd')
        # Applying the binary operator '*' (line 655)
        result_mul_608723 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 25), '*', d_608721, d_608722)
        
        float_608724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 31), 'float')
        # Applying the binary operator '+' (line 655)
        result_add_608725 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 25), '+', result_mul_608723, float_608724)
        
        # Applying the binary operator '*' (line 655)
        result_mul_608726 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 22), '*', result_div_608720, result_add_608725)
        
        # Getting the type of 'd' (line 655)
        d_608727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 39), 'd')
        # Getting the type of 'd' (line 655)
        d_608728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 41), 'd')
        # Applying the binary operator '*' (line 655)
        result_mul_608729 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 39), '*', d_608727, d_608728)
        
        float_608730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 45), 'float')
        # Applying the binary operator '-' (line 655)
        result_sub_608731 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 39), '-', result_mul_608729, float_608730)
        
        # Applying the binary operator 'div' (line 655)
        result_div_608732 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 36), 'div', result_mul_608726, result_sub_608731)
        
        # Assigning a type to the variable 'g2' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'g2', result_div_608732)
        
        # Obtaining an instance of the builtin type 'tuple' (line 656)
        tuple_608733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 656)
        # Adding element type (line 656)
        # Getting the type of 'mu' (line 656)
        mu_608734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 15), 'mu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 15), tuple_608733, mu_608734)
        # Adding element type (line 656)
        # Getting the type of 'var' (line 656)
        var_608735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 19), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 15), tuple_608733, var_608735)
        # Adding element type (line 656)
        # Getting the type of 'g1' (line 656)
        g1_608736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 24), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 15), tuple_608733, g1_608736)
        # Adding element type (line 656)
        # Getting the type of 'g2' (line 656)
        g2_608737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 28), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 15), tuple_608733, g2_608737)
        
        # Assigning a type to the variable 'stypy_return_type' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'stypy_return_type', tuple_608733)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 649)
        stypy_return_type_608738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_608738


    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 658, 4, False)
        # Assigning a type to the variable 'self' (line 659)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        randint_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        randint_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        randint_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        randint_gen._rvs.__dict__.__setitem__('stypy_function_name', 'randint_gen._rvs')
        randint_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['low', 'high'])
        randint_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        randint_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        randint_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        randint_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        randint_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        randint_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'randint_gen._rvs', ['low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        str_608739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 8), 'str', 'An array of *size* random integers >= ``low`` and < ``high``.')
        
        
        # Getting the type of 'self' (line 660)
        self_608740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 11), 'self')
        # Obtaining the member '_size' of a type (line 660)
        _size_608741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 11), self_608740, '_size')
        # Getting the type of 'None' (line 660)
        None_608742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 29), 'None')
        # Applying the binary operator 'isnot' (line 660)
        result_is_not_608743 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 11), 'isnot', _size_608741, None_608742)
        
        # Testing the type of an if condition (line 660)
        if_condition_608744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 660, 8), result_is_not_608743)
        # Assigning a type to the variable 'if_condition_608744' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'if_condition_608744', if_condition_608744)
        # SSA begins for if statement (line 660)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 665):
        
        # Assigning a Call to a Name (line 665):
        
        # Call to broadcast_to(...): (line 665)
        # Processing the call arguments (line 665)
        # Getting the type of 'low' (line 665)
        low_608746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 31), 'low', False)
        # Getting the type of 'self' (line 665)
        self_608747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 36), 'self', False)
        # Obtaining the member '_size' of a type (line 665)
        _size_608748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 36), self_608747, '_size')
        # Processing the call keyword arguments (line 665)
        kwargs_608749 = {}
        # Getting the type of 'broadcast_to' (line 665)
        broadcast_to_608745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 18), 'broadcast_to', False)
        # Calling broadcast_to(args, kwargs) (line 665)
        broadcast_to_call_result_608750 = invoke(stypy.reporting.localization.Localization(__file__, 665, 18), broadcast_to_608745, *[low_608746, _size_608748], **kwargs_608749)
        
        # Assigning a type to the variable 'low' (line 665)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'low', broadcast_to_call_result_608750)
        
        # Assigning a Call to a Name (line 666):
        
        # Assigning a Call to a Name (line 666):
        
        # Call to broadcast_to(...): (line 666)
        # Processing the call arguments (line 666)
        # Getting the type of 'high' (line 666)
        high_608752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 32), 'high', False)
        # Getting the type of 'self' (line 666)
        self_608753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 38), 'self', False)
        # Obtaining the member '_size' of a type (line 666)
        _size_608754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 38), self_608753, '_size')
        # Processing the call keyword arguments (line 666)
        kwargs_608755 = {}
        # Getting the type of 'broadcast_to' (line 666)
        broadcast_to_608751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 19), 'broadcast_to', False)
        # Calling broadcast_to(args, kwargs) (line 666)
        broadcast_to_call_result_608756 = invoke(stypy.reporting.localization.Localization(__file__, 666, 19), broadcast_to_608751, *[high_608752, _size_608754], **kwargs_608755)
        
        # Assigning a type to the variable 'high' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'high', broadcast_to_call_result_608756)
        # SSA join for if statement (line 660)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 667):
        
        # Assigning a Call to a Name (line 667):
        
        # Call to vectorize(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'self' (line 667)
        self_608759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 31), 'self', False)
        # Obtaining the member '_random_state' of a type (line 667)
        _random_state_608760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 31), self_608759, '_random_state')
        # Obtaining the member 'randint' of a type (line 667)
        randint_608761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 31), _random_state_608760, 'randint')
        # Processing the call keyword arguments (line 667)
        
        # Obtaining an instance of the builtin type 'list' (line 667)
        list_608762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 667)
        # Adding element type (line 667)
        # Getting the type of 'np' (line 667)
        np_608763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 67), 'np', False)
        # Obtaining the member 'int_' of a type (line 667)
        int__608764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 67), np_608763, 'int_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 66), list_608762, int__608764)
        
        keyword_608765 = list_608762
        kwargs_608766 = {'otypes': keyword_608765}
        # Getting the type of 'np' (line 667)
        np_608757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 18), 'np', False)
        # Obtaining the member 'vectorize' of a type (line 667)
        vectorize_608758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 18), np_608757, 'vectorize')
        # Calling vectorize(args, kwargs) (line 667)
        vectorize_call_result_608767 = invoke(stypy.reporting.localization.Localization(__file__, 667, 18), vectorize_608758, *[randint_608761], **kwargs_608766)
        
        # Assigning a type to the variable 'randint' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'randint', vectorize_call_result_608767)
        
        # Call to randint(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'low' (line 668)
        low_608769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 23), 'low', False)
        # Getting the type of 'high' (line 668)
        high_608770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 28), 'high', False)
        # Processing the call keyword arguments (line 668)
        kwargs_608771 = {}
        # Getting the type of 'randint' (line 668)
        randint_608768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'randint', False)
        # Calling randint(args, kwargs) (line 668)
        randint_call_result_608772 = invoke(stypy.reporting.localization.Localization(__file__, 668, 15), randint_608768, *[low_608769, high_608770], **kwargs_608771)
        
        # Assigning a type to the variable 'stypy_return_type' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'stypy_return_type', randint_call_result_608772)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 658)
        stypy_return_type_608773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608773)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_608773


    @norecursion
    def _entropy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_entropy'
        module_type_store = module_type_store.open_function_context('_entropy', 670, 4, False)
        # Assigning a type to the variable 'self' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        randint_gen._entropy.__dict__.__setitem__('stypy_localization', localization)
        randint_gen._entropy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        randint_gen._entropy.__dict__.__setitem__('stypy_type_store', module_type_store)
        randint_gen._entropy.__dict__.__setitem__('stypy_function_name', 'randint_gen._entropy')
        randint_gen._entropy.__dict__.__setitem__('stypy_param_names_list', ['low', 'high'])
        randint_gen._entropy.__dict__.__setitem__('stypy_varargs_param_name', None)
        randint_gen._entropy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        randint_gen._entropy.__dict__.__setitem__('stypy_call_defaults', defaults)
        randint_gen._entropy.__dict__.__setitem__('stypy_call_varargs', varargs)
        randint_gen._entropy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        randint_gen._entropy.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'randint_gen._entropy', ['low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_entropy', localization, ['low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_entropy(...)' code ##################

        
        # Call to log(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'high' (line 671)
        high_608775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 19), 'high', False)
        # Getting the type of 'low' (line 671)
        low_608776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 26), 'low', False)
        # Applying the binary operator '-' (line 671)
        result_sub_608777 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 19), '-', high_608775, low_608776)
        
        # Processing the call keyword arguments (line 671)
        kwargs_608778 = {}
        # Getting the type of 'log' (line 671)
        log_608774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 15), 'log', False)
        # Calling log(args, kwargs) (line 671)
        log_call_result_608779 = invoke(stypy.reporting.localization.Localization(__file__, 671, 15), log_608774, *[result_sub_608777], **kwargs_608778)
        
        # Assigning a type to the variable 'stypy_return_type' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'stypy_return_type', log_call_result_608779)
        
        # ################# End of '_entropy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_entropy' in the type store
        # Getting the type of 'stypy_return_type' (line 670)
        stypy_return_type_608780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608780)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_entropy'
        return stypy_return_type_608780


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 610, 0, False)
        # Assigning a type to the variable 'self' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'randint_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'randint_gen' (line 610)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 0), 'randint_gen', randint_gen)

# Assigning a Call to a Name (line 673):

# Assigning a Call to a Name (line 673):

# Call to randint_gen(...): (line 673)
# Processing the call keyword arguments (line 673)
str_608782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 27), 'str', 'randint')
keyword_608783 = str_608782
str_608784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 47), 'str', 'A discrete uniform (random integer)')
keyword_608785 = str_608784
kwargs_608786 = {'name': keyword_608783, 'longname': keyword_608785}
# Getting the type of 'randint_gen' (line 673)
randint_gen_608781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 10), 'randint_gen', False)
# Calling randint_gen(args, kwargs) (line 673)
randint_gen_call_result_608787 = invoke(stypy.reporting.localization.Localization(__file__, 673, 10), randint_gen_608781, *[], **kwargs_608786)

# Assigning a type to the variable 'randint' (line 673)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 0), 'randint', randint_gen_call_result_608787)
# Declaration of the 'zipf_gen' class
# Getting the type of 'rv_discrete' (line 678)
rv_discrete_608788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 15), 'rv_discrete')

class zipf_gen(rv_discrete_608788, ):
    str_608789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, (-1)), 'str', 'A Zipf discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `zipf` is::\n\n        zipf.pmf(k, a) = 1/(zeta(a) * k**a)\n\n    for ``k >= 1``.\n\n    `zipf` takes ``a`` as shape parameter.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 698, 4, False)
        # Assigning a type to the variable 'self' (line 699)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        zipf_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        zipf_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        zipf_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        zipf_gen._rvs.__dict__.__setitem__('stypy_function_name', 'zipf_gen._rvs')
        zipf_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['a'])
        zipf_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        zipf_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        zipf_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        zipf_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        zipf_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        zipf_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'zipf_gen._rvs', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Call to zipf(...): (line 699)
        # Processing the call arguments (line 699)
        # Getting the type of 'a' (line 699)
        a_608793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 39), 'a', False)
        # Processing the call keyword arguments (line 699)
        # Getting the type of 'self' (line 699)
        self_608794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 47), 'self', False)
        # Obtaining the member '_size' of a type (line 699)
        _size_608795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 47), self_608794, '_size')
        keyword_608796 = _size_608795
        kwargs_608797 = {'size': keyword_608796}
        # Getting the type of 'self' (line 699)
        self_608790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 15), 'self', False)
        # Obtaining the member '_random_state' of a type (line 699)
        _random_state_608791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 15), self_608790, '_random_state')
        # Obtaining the member 'zipf' of a type (line 699)
        zipf_608792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 15), _random_state_608791, 'zipf')
        # Calling zipf(args, kwargs) (line 699)
        zipf_call_result_608798 = invoke(stypy.reporting.localization.Localization(__file__, 699, 15), zipf_608792, *[a_608793], **kwargs_608797)
        
        # Assigning a type to the variable 'stypy_return_type' (line 699)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'stypy_return_type', zipf_call_result_608798)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 698)
        stypy_return_type_608799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608799)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_608799


    @norecursion
    def _argcheck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_argcheck'
        module_type_store = module_type_store.open_function_context('_argcheck', 701, 4, False)
        # Assigning a type to the variable 'self' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        zipf_gen._argcheck.__dict__.__setitem__('stypy_localization', localization)
        zipf_gen._argcheck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        zipf_gen._argcheck.__dict__.__setitem__('stypy_type_store', module_type_store)
        zipf_gen._argcheck.__dict__.__setitem__('stypy_function_name', 'zipf_gen._argcheck')
        zipf_gen._argcheck.__dict__.__setitem__('stypy_param_names_list', ['a'])
        zipf_gen._argcheck.__dict__.__setitem__('stypy_varargs_param_name', None)
        zipf_gen._argcheck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        zipf_gen._argcheck.__dict__.__setitem__('stypy_call_defaults', defaults)
        zipf_gen._argcheck.__dict__.__setitem__('stypy_call_varargs', varargs)
        zipf_gen._argcheck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        zipf_gen._argcheck.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'zipf_gen._argcheck', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_argcheck', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_argcheck(...)' code ##################

        
        # Getting the type of 'a' (line 702)
        a_608800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 15), 'a')
        int_608801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 19), 'int')
        # Applying the binary operator '>' (line 702)
        result_gt_608802 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 15), '>', a_608800, int_608801)
        
        # Assigning a type to the variable 'stypy_return_type' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'stypy_return_type', result_gt_608802)
        
        # ################# End of '_argcheck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_argcheck' in the type store
        # Getting the type of 'stypy_return_type' (line 701)
        stypy_return_type_608803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608803)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_argcheck'
        return stypy_return_type_608803


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 704, 4, False)
        # Assigning a type to the variable 'self' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        zipf_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        zipf_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        zipf_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        zipf_gen._pmf.__dict__.__setitem__('stypy_function_name', 'zipf_gen._pmf')
        zipf_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'a'])
        zipf_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        zipf_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        zipf_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        zipf_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        zipf_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        zipf_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'zipf_gen._pmf', ['k', 'a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 705):
        
        # Assigning a BinOp to a Name (line 705):
        float_608804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 13), 'float')
        
        # Call to zeta(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'a' (line 705)
        a_608807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 32), 'a', False)
        int_608808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 35), 'int')
        # Processing the call keyword arguments (line 705)
        kwargs_608809 = {}
        # Getting the type of 'special' (line 705)
        special_608805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 19), 'special', False)
        # Obtaining the member 'zeta' of a type (line 705)
        zeta_608806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 19), special_608805, 'zeta')
        # Calling zeta(args, kwargs) (line 705)
        zeta_call_result_608810 = invoke(stypy.reporting.localization.Localization(__file__, 705, 19), zeta_608806, *[a_608807, int_608808], **kwargs_608809)
        
        # Applying the binary operator 'div' (line 705)
        result_div_608811 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 13), 'div', float_608804, zeta_call_result_608810)
        
        # Getting the type of 'k' (line 705)
        k_608812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 40), 'k')
        # Getting the type of 'a' (line 705)
        a_608813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 43), 'a')
        # Applying the binary operator '**' (line 705)
        result_pow_608814 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 40), '**', k_608812, a_608813)
        
        # Applying the binary operator 'div' (line 705)
        result_div_608815 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 38), 'div', result_div_608811, result_pow_608814)
        
        # Assigning a type to the variable 'Pk' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'Pk', result_div_608815)
        # Getting the type of 'Pk' (line 706)
        Pk_608816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'Pk')
        # Assigning a type to the variable 'stypy_return_type' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'stypy_return_type', Pk_608816)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 704)
        stypy_return_type_608817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_608817


    @norecursion
    def _munp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_munp'
        module_type_store = module_type_store.open_function_context('_munp', 708, 4, False)
        # Assigning a type to the variable 'self' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        zipf_gen._munp.__dict__.__setitem__('stypy_localization', localization)
        zipf_gen._munp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        zipf_gen._munp.__dict__.__setitem__('stypy_type_store', module_type_store)
        zipf_gen._munp.__dict__.__setitem__('stypy_function_name', 'zipf_gen._munp')
        zipf_gen._munp.__dict__.__setitem__('stypy_param_names_list', ['n', 'a'])
        zipf_gen._munp.__dict__.__setitem__('stypy_varargs_param_name', None)
        zipf_gen._munp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        zipf_gen._munp.__dict__.__setitem__('stypy_call_defaults', defaults)
        zipf_gen._munp.__dict__.__setitem__('stypy_call_varargs', varargs)
        zipf_gen._munp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        zipf_gen._munp.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'zipf_gen._munp', ['n', 'a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_munp', localization, ['n', 'a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_munp(...)' code ##################

        
        # Call to _lazywhere(...): (line 709)
        # Processing the call arguments (line 709)
        
        # Getting the type of 'a' (line 710)
        a_608819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 12), 'a', False)
        # Getting the type of 'n' (line 710)
        n_608820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 16), 'n', False)
        int_608821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 20), 'int')
        # Applying the binary operator '+' (line 710)
        result_add_608822 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 16), '+', n_608820, int_608821)
        
        # Applying the binary operator '>' (line 710)
        result_gt_608823 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 12), '>', a_608819, result_add_608822)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 710)
        tuple_608824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 710)
        # Adding element type (line 710)
        # Getting the type of 'a' (line 710)
        a_608825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 24), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 24), tuple_608824, a_608825)
        # Adding element type (line 710)
        # Getting the type of 'n' (line 710)
        n_608826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 24), tuple_608824, n_608826)
        

        @norecursion
        def _stypy_temp_lambda_526(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_526'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_526', 711, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_526.stypy_localization = localization
            _stypy_temp_lambda_526.stypy_type_of_self = None
            _stypy_temp_lambda_526.stypy_type_store = module_type_store
            _stypy_temp_lambda_526.stypy_function_name = '_stypy_temp_lambda_526'
            _stypy_temp_lambda_526.stypy_param_names_list = ['a', 'n']
            _stypy_temp_lambda_526.stypy_varargs_param_name = None
            _stypy_temp_lambda_526.stypy_kwargs_param_name = None
            _stypy_temp_lambda_526.stypy_call_defaults = defaults
            _stypy_temp_lambda_526.stypy_call_varargs = varargs
            _stypy_temp_lambda_526.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_526', ['a', 'n'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_526', ['a', 'n'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to zeta(...): (line 711)
            # Processing the call arguments (line 711)
            # Getting the type of 'a' (line 711)
            a_608829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 38), 'a', False)
            # Getting the type of 'n' (line 711)
            n_608830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 42), 'n', False)
            # Applying the binary operator '-' (line 711)
            result_sub_608831 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 38), '-', a_608829, n_608830)
            
            int_608832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 45), 'int')
            # Processing the call keyword arguments (line 711)
            kwargs_608833 = {}
            # Getting the type of 'special' (line 711)
            special_608827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 25), 'special', False)
            # Obtaining the member 'zeta' of a type (line 711)
            zeta_608828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 25), special_608827, 'zeta')
            # Calling zeta(args, kwargs) (line 711)
            zeta_call_result_608834 = invoke(stypy.reporting.localization.Localization(__file__, 711, 25), zeta_608828, *[result_sub_608831, int_608832], **kwargs_608833)
            
            
            # Call to zeta(...): (line 711)
            # Processing the call arguments (line 711)
            # Getting the type of 'a' (line 711)
            a_608837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 63), 'a', False)
            int_608838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 66), 'int')
            # Processing the call keyword arguments (line 711)
            kwargs_608839 = {}
            # Getting the type of 'special' (line 711)
            special_608835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 50), 'special', False)
            # Obtaining the member 'zeta' of a type (line 711)
            zeta_608836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 50), special_608835, 'zeta')
            # Calling zeta(args, kwargs) (line 711)
            zeta_call_result_608840 = invoke(stypy.reporting.localization.Localization(__file__, 711, 50), zeta_608836, *[a_608837, int_608838], **kwargs_608839)
            
            # Applying the binary operator 'div' (line 711)
            result_div_608841 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 25), 'div', zeta_call_result_608834, zeta_call_result_608840)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 711)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), 'stypy_return_type', result_div_608841)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_526' in the type store
            # Getting the type of 'stypy_return_type' (line 711)
            stypy_return_type_608842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_608842)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_526'
            return stypy_return_type_608842

        # Assigning a type to the variable '_stypy_temp_lambda_526' (line 711)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), '_stypy_temp_lambda_526', _stypy_temp_lambda_526)
        # Getting the type of '_stypy_temp_lambda_526' (line 711)
        _stypy_temp_lambda_526_608843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), '_stypy_temp_lambda_526')
        # Getting the type of 'np' (line 712)
        np_608844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 12), 'np', False)
        # Obtaining the member 'inf' of a type (line 712)
        inf_608845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 12), np_608844, 'inf')
        # Processing the call keyword arguments (line 709)
        kwargs_608846 = {}
        # Getting the type of '_lazywhere' (line 709)
        _lazywhere_608818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 15), '_lazywhere', False)
        # Calling _lazywhere(args, kwargs) (line 709)
        _lazywhere_call_result_608847 = invoke(stypy.reporting.localization.Localization(__file__, 709, 15), _lazywhere_608818, *[result_gt_608823, tuple_608824, _stypy_temp_lambda_526_608843, inf_608845], **kwargs_608846)
        
        # Assigning a type to the variable 'stypy_return_type' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'stypy_return_type', _lazywhere_call_result_608847)
        
        # ################# End of '_munp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_munp' in the type store
        # Getting the type of 'stypy_return_type' (line 708)
        stypy_return_type_608848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_munp'
        return stypy_return_type_608848


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 678, 0, False)
        # Assigning a type to the variable 'self' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'zipf_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'zipf_gen' (line 678)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 0), 'zipf_gen', zipf_gen)

# Assigning a Call to a Name (line 713):

# Assigning a Call to a Name (line 713):

# Call to zipf_gen(...): (line 713)
# Processing the call keyword arguments (line 713)
int_608850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 18), 'int')
keyword_608851 = int_608850
str_608852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 26), 'str', 'zipf')
keyword_608853 = str_608852
str_608854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 43), 'str', 'A Zipf')
keyword_608855 = str_608854
kwargs_608856 = {'a': keyword_608851, 'name': keyword_608853, 'longname': keyword_608855}
# Getting the type of 'zipf_gen' (line 713)
zipf_gen_608849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 7), 'zipf_gen', False)
# Calling zipf_gen(args, kwargs) (line 713)
zipf_gen_call_result_608857 = invoke(stypy.reporting.localization.Localization(__file__, 713, 7), zipf_gen_608849, *[], **kwargs_608856)

# Assigning a type to the variable 'zipf' (line 713)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 0), 'zipf', zipf_gen_call_result_608857)
# Declaration of the 'dlaplace_gen' class
# Getting the type of 'rv_discrete' (line 716)
rv_discrete_608858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 19), 'rv_discrete')

class dlaplace_gen(rv_discrete_608858, ):
    str_608859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, (-1)), 'str', 'A  Laplacian discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    The probability mass function for `dlaplace` is::\n\n        dlaplace.pmf(k) = tanh(a/2) * exp(-a*abs(k))\n\n    for ``a > 0``.\n\n    `dlaplace` takes ``a`` as shape parameter.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 736, 4, False)
        # Assigning a type to the variable 'self' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_function_name', 'dlaplace_gen._pmf')
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['k', 'a'])
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dlaplace_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dlaplace_gen._pmf', ['k', 'a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['k', 'a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Call to tanh(...): (line 737)
        # Processing the call arguments (line 737)
        # Getting the type of 'a' (line 737)
        a_608861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 20), 'a', False)
        float_608862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 22), 'float')
        # Applying the binary operator 'div' (line 737)
        result_div_608863 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 20), 'div', a_608861, float_608862)
        
        # Processing the call keyword arguments (line 737)
        kwargs_608864 = {}
        # Getting the type of 'tanh' (line 737)
        tanh_608860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 15), 'tanh', False)
        # Calling tanh(args, kwargs) (line 737)
        tanh_call_result_608865 = invoke(stypy.reporting.localization.Localization(__file__, 737, 15), tanh_608860, *[result_div_608863], **kwargs_608864)
        
        
        # Call to exp(...): (line 737)
        # Processing the call arguments (line 737)
        
        # Getting the type of 'a' (line 737)
        a_608867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 34), 'a', False)
        # Applying the 'usub' unary operator (line 737)
        result___neg___608868 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 33), 'usub', a_608867)
        
        
        # Call to abs(...): (line 737)
        # Processing the call arguments (line 737)
        # Getting the type of 'k' (line 737)
        k_608870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 42), 'k', False)
        # Processing the call keyword arguments (line 737)
        kwargs_608871 = {}
        # Getting the type of 'abs' (line 737)
        abs_608869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 38), 'abs', False)
        # Calling abs(args, kwargs) (line 737)
        abs_call_result_608872 = invoke(stypy.reporting.localization.Localization(__file__, 737, 38), abs_608869, *[k_608870], **kwargs_608871)
        
        # Applying the binary operator '*' (line 737)
        result_mul_608873 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 33), '*', result___neg___608868, abs_call_result_608872)
        
        # Processing the call keyword arguments (line 737)
        kwargs_608874 = {}
        # Getting the type of 'exp' (line 737)
        exp_608866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 29), 'exp', False)
        # Calling exp(args, kwargs) (line 737)
        exp_call_result_608875 = invoke(stypy.reporting.localization.Localization(__file__, 737, 29), exp_608866, *[result_mul_608873], **kwargs_608874)
        
        # Applying the binary operator '*' (line 737)
        result_mul_608876 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 15), '*', tanh_call_result_608865, exp_call_result_608875)
        
        # Assigning a type to the variable 'stypy_return_type' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'stypy_return_type', result_mul_608876)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 736)
        stypy_return_type_608877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_608877


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 739, 4, False)
        # Assigning a type to the variable 'self' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_function_name', 'dlaplace_gen._cdf')
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'a'])
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dlaplace_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dlaplace_gen._cdf', ['x', 'a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 740):
        
        # Assigning a Call to a Name (line 740):
        
        # Call to floor(...): (line 740)
        # Processing the call arguments (line 740)
        # Getting the type of 'x' (line 740)
        x_608879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 18), 'x', False)
        # Processing the call keyword arguments (line 740)
        kwargs_608880 = {}
        # Getting the type of 'floor' (line 740)
        floor_608878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 740)
        floor_call_result_608881 = invoke(stypy.reporting.localization.Localization(__file__, 740, 12), floor_608878, *[x_608879], **kwargs_608880)
        
        # Assigning a type to the variable 'k' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 8), 'k', floor_call_result_608881)
        
        # Assigning a Lambda to a Name (line 741):
        
        # Assigning a Lambda to a Name (line 741):

        @norecursion
        def _stypy_temp_lambda_527(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_527'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_527', 741, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_527.stypy_localization = localization
            _stypy_temp_lambda_527.stypy_type_of_self = None
            _stypy_temp_lambda_527.stypy_type_store = module_type_store
            _stypy_temp_lambda_527.stypy_function_name = '_stypy_temp_lambda_527'
            _stypy_temp_lambda_527.stypy_param_names_list = ['k', 'a']
            _stypy_temp_lambda_527.stypy_varargs_param_name = None
            _stypy_temp_lambda_527.stypy_kwargs_param_name = None
            _stypy_temp_lambda_527.stypy_call_defaults = defaults
            _stypy_temp_lambda_527.stypy_call_varargs = varargs
            _stypy_temp_lambda_527.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_527', ['k', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_527', ['k', 'a'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            float_608882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 25), 'float')
            
            # Call to exp(...): (line 741)
            # Processing the call arguments (line 741)
            
            # Getting the type of 'a' (line 741)
            a_608884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 36), 'a', False)
            # Applying the 'usub' unary operator (line 741)
            result___neg___608885 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 35), 'usub', a_608884)
            
            # Getting the type of 'k' (line 741)
            k_608886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 40), 'k', False)
            # Applying the binary operator '*' (line 741)
            result_mul_608887 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 35), '*', result___neg___608885, k_608886)
            
            # Processing the call keyword arguments (line 741)
            kwargs_608888 = {}
            # Getting the type of 'exp' (line 741)
            exp_608883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 31), 'exp', False)
            # Calling exp(args, kwargs) (line 741)
            exp_call_result_608889 = invoke(stypy.reporting.localization.Localization(__file__, 741, 31), exp_608883, *[result_mul_608887], **kwargs_608888)
            
            
            # Call to exp(...): (line 741)
            # Processing the call arguments (line 741)
            # Getting the type of 'a' (line 741)
            a_608891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 50), 'a', False)
            # Processing the call keyword arguments (line 741)
            kwargs_608892 = {}
            # Getting the type of 'exp' (line 741)
            exp_608890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 46), 'exp', False)
            # Calling exp(args, kwargs) (line 741)
            exp_call_result_608893 = invoke(stypy.reporting.localization.Localization(__file__, 741, 46), exp_608890, *[a_608891], **kwargs_608892)
            
            int_608894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 55), 'int')
            # Applying the binary operator '+' (line 741)
            result_add_608895 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 46), '+', exp_call_result_608893, int_608894)
            
            # Applying the binary operator 'div' (line 741)
            result_div_608896 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 31), 'div', exp_call_result_608889, result_add_608895)
            
            # Applying the binary operator '-' (line 741)
            result_sub_608897 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 25), '-', float_608882, result_div_608896)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 741)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'stypy_return_type', result_sub_608897)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_527' in the type store
            # Getting the type of 'stypy_return_type' (line 741)
            stypy_return_type_608898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_608898)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_527'
            return stypy_return_type_608898

        # Assigning a type to the variable '_stypy_temp_lambda_527' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), '_stypy_temp_lambda_527', _stypy_temp_lambda_527)
        # Getting the type of '_stypy_temp_lambda_527' (line 741)
        _stypy_temp_lambda_527_608899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), '_stypy_temp_lambda_527')
        # Assigning a type to the variable 'f' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'f', _stypy_temp_lambda_527_608899)
        
        # Assigning a Lambda to a Name (line 742):
        
        # Assigning a Lambda to a Name (line 742):

        @norecursion
        def _stypy_temp_lambda_528(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_528'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_528', 742, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_528.stypy_localization = localization
            _stypy_temp_lambda_528.stypy_type_of_self = None
            _stypy_temp_lambda_528.stypy_type_store = module_type_store
            _stypy_temp_lambda_528.stypy_function_name = '_stypy_temp_lambda_528'
            _stypy_temp_lambda_528.stypy_param_names_list = ['k', 'a']
            _stypy_temp_lambda_528.stypy_varargs_param_name = None
            _stypy_temp_lambda_528.stypy_kwargs_param_name = None
            _stypy_temp_lambda_528.stypy_call_defaults = defaults
            _stypy_temp_lambda_528.stypy_call_varargs = varargs
            _stypy_temp_lambda_528.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_528', ['k', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_528', ['k', 'a'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to exp(...): (line 742)
            # Processing the call arguments (line 742)
            # Getting the type of 'a' (line 742)
            a_608901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 30), 'a', False)
            # Getting the type of 'k' (line 742)
            k_608902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 35), 'k', False)
            int_608903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 37), 'int')
            # Applying the binary operator '+' (line 742)
            result_add_608904 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 35), '+', k_608902, int_608903)
            
            # Applying the binary operator '*' (line 742)
            result_mul_608905 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 30), '*', a_608901, result_add_608904)
            
            # Processing the call keyword arguments (line 742)
            kwargs_608906 = {}
            # Getting the type of 'exp' (line 742)
            exp_608900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 26), 'exp', False)
            # Calling exp(args, kwargs) (line 742)
            exp_call_result_608907 = invoke(stypy.reporting.localization.Localization(__file__, 742, 26), exp_608900, *[result_mul_608905], **kwargs_608906)
            
            
            # Call to exp(...): (line 742)
            # Processing the call arguments (line 742)
            # Getting the type of 'a' (line 742)
            a_608909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 48), 'a', False)
            # Processing the call keyword arguments (line 742)
            kwargs_608910 = {}
            # Getting the type of 'exp' (line 742)
            exp_608908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 44), 'exp', False)
            # Calling exp(args, kwargs) (line 742)
            exp_call_result_608911 = invoke(stypy.reporting.localization.Localization(__file__, 742, 44), exp_608908, *[a_608909], **kwargs_608910)
            
            int_608912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 53), 'int')
            # Applying the binary operator '+' (line 742)
            result_add_608913 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 44), '+', exp_call_result_608911, int_608912)
            
            # Applying the binary operator 'div' (line 742)
            result_div_608914 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 26), 'div', exp_call_result_608907, result_add_608913)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 742)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 13), 'stypy_return_type', result_div_608914)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_528' in the type store
            # Getting the type of 'stypy_return_type' (line 742)
            stypy_return_type_608915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_608915)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_528'
            return stypy_return_type_608915

        # Assigning a type to the variable '_stypy_temp_lambda_528' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 13), '_stypy_temp_lambda_528', _stypy_temp_lambda_528)
        # Getting the type of '_stypy_temp_lambda_528' (line 742)
        _stypy_temp_lambda_528_608916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 13), '_stypy_temp_lambda_528')
        # Assigning a type to the variable 'f2' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'f2', _stypy_temp_lambda_528_608916)
        
        # Call to _lazywhere(...): (line 743)
        # Processing the call arguments (line 743)
        
        # Getting the type of 'k' (line 743)
        k_608918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 26), 'k', False)
        int_608919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 31), 'int')
        # Applying the binary operator '>=' (line 743)
        result_ge_608920 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 26), '>=', k_608918, int_608919)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 743)
        tuple_608921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 743)
        # Adding element type (line 743)
        # Getting the type of 'k' (line 743)
        k_608922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 35), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 35), tuple_608921, k_608922)
        # Adding element type (line 743)
        # Getting the type of 'a' (line 743)
        a_608923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 38), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 35), tuple_608921, a_608923)
        
        # Processing the call keyword arguments (line 743)
        # Getting the type of 'f' (line 743)
        f_608924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'f', False)
        keyword_608925 = f_608924
        # Getting the type of 'f2' (line 743)
        f2_608926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 50), 'f2', False)
        keyword_608927 = f2_608926
        kwargs_608928 = {'f2': keyword_608927, 'f': keyword_608925}
        # Getting the type of '_lazywhere' (line 743)
        _lazywhere_608917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 15), '_lazywhere', False)
        # Calling _lazywhere(args, kwargs) (line 743)
        _lazywhere_call_result_608929 = invoke(stypy.reporting.localization.Localization(__file__, 743, 15), _lazywhere_608917, *[result_ge_608920, tuple_608921], **kwargs_608928)
        
        # Assigning a type to the variable 'stypy_return_type' (line 743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'stypy_return_type', _lazywhere_call_result_608929)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 739)
        stypy_return_type_608930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_608930


    @norecursion
    def _ppf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ppf'
        module_type_store = module_type_store.open_function_context('_ppf', 745, 4, False)
        # Assigning a type to the variable 'self' (line 746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_localization', localization)
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_type_store', module_type_store)
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_function_name', 'dlaplace_gen._ppf')
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_param_names_list', ['q', 'a'])
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_varargs_param_name', None)
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_call_defaults', defaults)
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_call_varargs', varargs)
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dlaplace_gen._ppf.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dlaplace_gen._ppf', ['q', 'a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ppf', localization, ['q', 'a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ppf(...)' code ##################

        
        # Assigning a BinOp to a Name (line 746):
        
        # Assigning a BinOp to a Name (line 746):
        int_608931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 16), 'int')
        
        # Call to exp(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'a' (line 746)
        a_608933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 24), 'a', False)
        # Processing the call keyword arguments (line 746)
        kwargs_608934 = {}
        # Getting the type of 'exp' (line 746)
        exp_608932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 20), 'exp', False)
        # Calling exp(args, kwargs) (line 746)
        exp_call_result_608935 = invoke(stypy.reporting.localization.Localization(__file__, 746, 20), exp_608932, *[a_608933], **kwargs_608934)
        
        # Applying the binary operator '+' (line 746)
        result_add_608936 = python_operator(stypy.reporting.localization.Localization(__file__, 746, 16), '+', int_608931, exp_call_result_608935)
        
        # Assigning a type to the variable 'const' (line 746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'const', result_add_608936)
        
        # Assigning a Call to a Name (line 747):
        
        # Assigning a Call to a Name (line 747):
        
        # Call to ceil(...): (line 747)
        # Processing the call arguments (line 747)
        
        # Call to where(...): (line 747)
        # Processing the call arguments (line 747)
        
        # Getting the type of 'q' (line 747)
        q_608940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 29), 'q', False)
        float_608941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 33), 'float')
        int_608942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 40), 'int')
        
        # Call to exp(...): (line 747)
        # Processing the call arguments (line 747)
        
        # Getting the type of 'a' (line 747)
        a_608944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 49), 'a', False)
        # Applying the 'usub' unary operator (line 747)
        result___neg___608945 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 48), 'usub', a_608944)
        
        # Processing the call keyword arguments (line 747)
        kwargs_608946 = {}
        # Getting the type of 'exp' (line 747)
        exp_608943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 44), 'exp', False)
        # Calling exp(args, kwargs) (line 747)
        exp_call_result_608947 = invoke(stypy.reporting.localization.Localization(__file__, 747, 44), exp_608943, *[result___neg___608945], **kwargs_608946)
        
        # Applying the binary operator '+' (line 747)
        result_add_608948 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 40), '+', int_608942, exp_call_result_608947)
        
        # Applying the binary operator 'div' (line 747)
        result_div_608949 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 33), 'div', float_608941, result_add_608948)
        
        # Applying the binary operator '<' (line 747)
        result_lt_608950 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 29), '<', q_608940, result_div_608949)
        
        
        # Call to log(...): (line 747)
        # Processing the call arguments (line 747)
        # Getting the type of 'q' (line 747)
        q_608952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 58), 'q', False)
        # Getting the type of 'const' (line 747)
        const_608953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 60), 'const', False)
        # Applying the binary operator '*' (line 747)
        result_mul_608954 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 58), '*', q_608952, const_608953)
        
        # Processing the call keyword arguments (line 747)
        kwargs_608955 = {}
        # Getting the type of 'log' (line 747)
        log_608951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 54), 'log', False)
        # Calling log(args, kwargs) (line 747)
        log_call_result_608956 = invoke(stypy.reporting.localization.Localization(__file__, 747, 54), log_608951, *[result_mul_608954], **kwargs_608955)
        
        # Getting the type of 'a' (line 747)
        a_608957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 69), 'a', False)
        # Applying the binary operator 'div' (line 747)
        result_div_608958 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 54), 'div', log_call_result_608956, a_608957)
        
        int_608959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 73), 'int')
        # Applying the binary operator '-' (line 747)
        result_sub_608960 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 54), '-', result_div_608958, int_608959)
        
        
        
        # Call to log(...): (line 748)
        # Processing the call arguments (line 748)
        int_608962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 60), 'int')
        # Getting the type of 'q' (line 748)
        q_608963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 62), 'q', False)
        # Applying the binary operator '-' (line 748)
        result_sub_608964 = python_operator(stypy.reporting.localization.Localization(__file__, 748, 60), '-', int_608962, q_608963)
        
        # Getting the type of 'const' (line 748)
        const_608965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 67), 'const', False)
        # Applying the binary operator '*' (line 748)
        result_mul_608966 = python_operator(stypy.reporting.localization.Localization(__file__, 748, 59), '*', result_sub_608964, const_608965)
        
        # Processing the call keyword arguments (line 748)
        kwargs_608967 = {}
        # Getting the type of 'log' (line 748)
        log_608961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 55), 'log', False)
        # Calling log(args, kwargs) (line 748)
        log_call_result_608968 = invoke(stypy.reporting.localization.Localization(__file__, 748, 55), log_608961, *[result_mul_608966], **kwargs_608967)
        
        # Applying the 'usub' unary operator (line 748)
        result___neg___608969 = python_operator(stypy.reporting.localization.Localization(__file__, 748, 54), 'usub', log_call_result_608968)
        
        # Getting the type of 'a' (line 748)
        a_608970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 76), 'a', False)
        # Applying the binary operator 'div' (line 748)
        result_div_608971 = python_operator(stypy.reporting.localization.Localization(__file__, 748, 54), 'div', result___neg___608969, a_608970)
        
        # Processing the call keyword arguments (line 747)
        kwargs_608972 = {}
        # Getting the type of 'np' (line 747)
        np_608938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 20), 'np', False)
        # Obtaining the member 'where' of a type (line 747)
        where_608939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 20), np_608938, 'where')
        # Calling where(args, kwargs) (line 747)
        where_call_result_608973 = invoke(stypy.reporting.localization.Localization(__file__, 747, 20), where_608939, *[result_lt_608950, result_sub_608960, result_div_608971], **kwargs_608972)
        
        # Processing the call keyword arguments (line 747)
        kwargs_608974 = {}
        # Getting the type of 'ceil' (line 747)
        ceil_608937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 15), 'ceil', False)
        # Calling ceil(args, kwargs) (line 747)
        ceil_call_result_608975 = invoke(stypy.reporting.localization.Localization(__file__, 747, 15), ceil_608937, *[where_call_result_608973], **kwargs_608974)
        
        # Assigning a type to the variable 'vals' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'vals', ceil_call_result_608975)
        
        # Assigning a BinOp to a Name (line 749):
        
        # Assigning a BinOp to a Name (line 749):
        # Getting the type of 'vals' (line 749)
        vals_608976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 16), 'vals')
        int_608977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 23), 'int')
        # Applying the binary operator '-' (line 749)
        result_sub_608978 = python_operator(stypy.reporting.localization.Localization(__file__, 749, 16), '-', vals_608976, int_608977)
        
        # Assigning a type to the variable 'vals1' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'vals1', result_sub_608978)
        
        # Call to where(...): (line 750)
        # Processing the call arguments (line 750)
        
        
        # Call to _cdf(...): (line 750)
        # Processing the call arguments (line 750)
        # Getting the type of 'vals1' (line 750)
        vals1_608983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 34), 'vals1', False)
        # Getting the type of 'a' (line 750)
        a_608984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 41), 'a', False)
        # Processing the call keyword arguments (line 750)
        kwargs_608985 = {}
        # Getting the type of 'self' (line 750)
        self_608981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 24), 'self', False)
        # Obtaining the member '_cdf' of a type (line 750)
        _cdf_608982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 24), self_608981, '_cdf')
        # Calling _cdf(args, kwargs) (line 750)
        _cdf_call_result_608986 = invoke(stypy.reporting.localization.Localization(__file__, 750, 24), _cdf_608982, *[vals1_608983, a_608984], **kwargs_608985)
        
        # Getting the type of 'q' (line 750)
        q_608987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 47), 'q', False)
        # Applying the binary operator '>=' (line 750)
        result_ge_608988 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 24), '>=', _cdf_call_result_608986, q_608987)
        
        # Getting the type of 'vals1' (line 750)
        vals1_608989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 50), 'vals1', False)
        # Getting the type of 'vals' (line 750)
        vals_608990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 57), 'vals', False)
        # Processing the call keyword arguments (line 750)
        kwargs_608991 = {}
        # Getting the type of 'np' (line 750)
        np_608979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 750)
        where_608980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 15), np_608979, 'where')
        # Calling where(args, kwargs) (line 750)
        where_call_result_608992 = invoke(stypy.reporting.localization.Localization(__file__, 750, 15), where_608980, *[result_ge_608988, vals1_608989, vals_608990], **kwargs_608991)
        
        # Assigning a type to the variable 'stypy_return_type' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'stypy_return_type', where_call_result_608992)
        
        # ################# End of '_ppf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ppf' in the type store
        # Getting the type of 'stypy_return_type' (line 745)
        stypy_return_type_608993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_608993)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ppf'
        return stypy_return_type_608993


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 752, 4, False)
        # Assigning a type to the variable 'self' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dlaplace_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        dlaplace_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dlaplace_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        dlaplace_gen._stats.__dict__.__setitem__('stypy_function_name', 'dlaplace_gen._stats')
        dlaplace_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['a'])
        dlaplace_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        dlaplace_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dlaplace_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        dlaplace_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        dlaplace_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dlaplace_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dlaplace_gen._stats', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a Call to a Name (line 753):
        
        # Assigning a Call to a Name (line 753):
        
        # Call to exp(...): (line 753)
        # Processing the call arguments (line 753)
        # Getting the type of 'a' (line 753)
        a_608995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 17), 'a', False)
        # Processing the call keyword arguments (line 753)
        kwargs_608996 = {}
        # Getting the type of 'exp' (line 753)
        exp_608994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 13), 'exp', False)
        # Calling exp(args, kwargs) (line 753)
        exp_call_result_608997 = invoke(stypy.reporting.localization.Localization(__file__, 753, 13), exp_608994, *[a_608995], **kwargs_608996)
        
        # Assigning a type to the variable 'ea' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'ea', exp_call_result_608997)
        
        # Assigning a BinOp to a Name (line 754):
        
        # Assigning a BinOp to a Name (line 754):
        float_608998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 14), 'float')
        # Getting the type of 'ea' (line 754)
        ea_608999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 17), 'ea')
        # Applying the binary operator '*' (line 754)
        result_mul_609000 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 14), '*', float_608998, ea_608999)
        
        # Getting the type of 'ea' (line 754)
        ea_609001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 21), 'ea')
        float_609002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 24), 'float')
        # Applying the binary operator '-' (line 754)
        result_sub_609003 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 21), '-', ea_609001, float_609002)
        
        int_609004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 29), 'int')
        # Applying the binary operator '**' (line 754)
        result_pow_609005 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 20), '**', result_sub_609003, int_609004)
        
        # Applying the binary operator 'div' (line 754)
        result_div_609006 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 19), 'div', result_mul_609000, result_pow_609005)
        
        # Assigning a type to the variable 'mu2' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'mu2', result_div_609006)
        
        # Assigning a BinOp to a Name (line 755):
        
        # Assigning a BinOp to a Name (line 755):
        float_609007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 14), 'float')
        # Getting the type of 'ea' (line 755)
        ea_609008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 17), 'ea')
        # Applying the binary operator '*' (line 755)
        result_mul_609009 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 14), '*', float_609007, ea_609008)
        
        # Getting the type of 'ea' (line 755)
        ea_609010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 21), 'ea')
        int_609011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 25), 'int')
        # Applying the binary operator '**' (line 755)
        result_pow_609012 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 21), '**', ea_609010, int_609011)
        
        float_609013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 27), 'float')
        # Getting the type of 'ea' (line 755)
        ea_609014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 31), 'ea')
        # Applying the binary operator '*' (line 755)
        result_mul_609015 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 27), '*', float_609013, ea_609014)
        
        # Applying the binary operator '+' (line 755)
        result_add_609016 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 21), '+', result_pow_609012, result_mul_609015)
        
        float_609017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 34), 'float')
        # Applying the binary operator '+' (line 755)
        result_add_609018 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 33), '+', result_add_609016, float_609017)
        
        # Applying the binary operator '*' (line 755)
        result_mul_609019 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 19), '*', result_mul_609009, result_add_609018)
        
        # Getting the type of 'ea' (line 755)
        ea_609020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 41), 'ea')
        float_609021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 44), 'float')
        # Applying the binary operator '-' (line 755)
        result_sub_609022 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 41), '-', ea_609020, float_609021)
        
        int_609023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 49), 'int')
        # Applying the binary operator '**' (line 755)
        result_pow_609024 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 40), '**', result_sub_609022, int_609023)
        
        # Applying the binary operator 'div' (line 755)
        result_div_609025 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 38), 'div', result_mul_609019, result_pow_609024)
        
        # Assigning a type to the variable 'mu4' (line 755)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'mu4', result_div_609025)
        
        # Obtaining an instance of the builtin type 'tuple' (line 756)
        tuple_609026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 756)
        # Adding element type (line 756)
        float_609027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 15), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 15), tuple_609026, float_609027)
        # Adding element type (line 756)
        # Getting the type of 'mu2' (line 756)
        mu2_609028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 19), 'mu2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 15), tuple_609026, mu2_609028)
        # Adding element type (line 756)
        float_609029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 15), tuple_609026, float_609029)
        # Adding element type (line 756)
        # Getting the type of 'mu4' (line 756)
        mu4_609030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 28), 'mu4')
        # Getting the type of 'mu2' (line 756)
        mu2_609031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 32), 'mu2')
        int_609032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 37), 'int')
        # Applying the binary operator '**' (line 756)
        result_pow_609033 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 32), '**', mu2_609031, int_609032)
        
        # Applying the binary operator 'div' (line 756)
        result_div_609034 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 28), 'div', mu4_609030, result_pow_609033)
        
        float_609035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 41), 'float')
        # Applying the binary operator '-' (line 756)
        result_sub_609036 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 28), '-', result_div_609034, float_609035)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 15), tuple_609026, result_sub_609036)
        
        # Assigning a type to the variable 'stypy_return_type' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'stypy_return_type', tuple_609026)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 752)
        stypy_return_type_609037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_609037)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_609037


    @norecursion
    def _entropy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_entropy'
        module_type_store = module_type_store.open_function_context('_entropy', 758, 4, False)
        # Assigning a type to the variable 'self' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_localization', localization)
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_type_store', module_type_store)
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_function_name', 'dlaplace_gen._entropy')
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_param_names_list', ['a'])
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_varargs_param_name', None)
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_call_defaults', defaults)
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_call_varargs', varargs)
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dlaplace_gen._entropy.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dlaplace_gen._entropy', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_entropy', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_entropy(...)' code ##################

        # Getting the type of 'a' (line 759)
        a_609038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 15), 'a')
        
        # Call to sinh(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'a' (line 759)
        a_609040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 24), 'a', False)
        # Processing the call keyword arguments (line 759)
        kwargs_609041 = {}
        # Getting the type of 'sinh' (line 759)
        sinh_609039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 19), 'sinh', False)
        # Calling sinh(args, kwargs) (line 759)
        sinh_call_result_609042 = invoke(stypy.reporting.localization.Localization(__file__, 759, 19), sinh_609039, *[a_609040], **kwargs_609041)
        
        # Applying the binary operator 'div' (line 759)
        result_div_609043 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 15), 'div', a_609038, sinh_call_result_609042)
        
        
        # Call to log(...): (line 759)
        # Processing the call arguments (line 759)
        
        # Call to tanh(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'a' (line 759)
        a_609046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 38), 'a', False)
        float_609047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 40), 'float')
        # Applying the binary operator 'div' (line 759)
        result_div_609048 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 38), 'div', a_609046, float_609047)
        
        # Processing the call keyword arguments (line 759)
        kwargs_609049 = {}
        # Getting the type of 'tanh' (line 759)
        tanh_609045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 33), 'tanh', False)
        # Calling tanh(args, kwargs) (line 759)
        tanh_call_result_609050 = invoke(stypy.reporting.localization.Localization(__file__, 759, 33), tanh_609045, *[result_div_609048], **kwargs_609049)
        
        # Processing the call keyword arguments (line 759)
        kwargs_609051 = {}
        # Getting the type of 'log' (line 759)
        log_609044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 29), 'log', False)
        # Calling log(args, kwargs) (line 759)
        log_call_result_609052 = invoke(stypy.reporting.localization.Localization(__file__, 759, 29), log_609044, *[tanh_call_result_609050], **kwargs_609051)
        
        # Applying the binary operator '-' (line 759)
        result_sub_609053 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 15), '-', result_div_609043, log_call_result_609052)
        
        # Assigning a type to the variable 'stypy_return_type' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'stypy_return_type', result_sub_609053)
        
        # ################# End of '_entropy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_entropy' in the type store
        # Getting the type of 'stypy_return_type' (line 758)
        stypy_return_type_609054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_609054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_entropy'
        return stypy_return_type_609054


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 716, 0, False)
        # Assigning a type to the variable 'self' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dlaplace_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'dlaplace_gen' (line 716)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 0), 'dlaplace_gen', dlaplace_gen)

# Assigning a Call to a Name (line 760):

# Assigning a Call to a Name (line 760):

# Call to dlaplace_gen(...): (line 760)
# Processing the call keyword arguments (line 760)

# Getting the type of 'np' (line 760)
np_609056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 27), 'np', False)
# Obtaining the member 'inf' of a type (line 760)
inf_609057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 27), np_609056, 'inf')
# Applying the 'usub' unary operator (line 760)
result___neg___609058 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 26), 'usub', inf_609057)

keyword_609059 = result___neg___609058
str_609060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 29), 'str', 'dlaplace')
keyword_609061 = str_609060
str_609062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 50), 'str', 'A discrete Laplacian')
keyword_609063 = str_609062
kwargs_609064 = {'a': keyword_609059, 'name': keyword_609061, 'longname': keyword_609063}
# Getting the type of 'dlaplace_gen' (line 760)
dlaplace_gen_609055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 11), 'dlaplace_gen', False)
# Calling dlaplace_gen(args, kwargs) (line 760)
dlaplace_gen_call_result_609065 = invoke(stypy.reporting.localization.Localization(__file__, 760, 11), dlaplace_gen_609055, *[], **kwargs_609064)

# Assigning a type to the variable 'dlaplace' (line 760)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 0), 'dlaplace', dlaplace_gen_call_result_609065)
# Declaration of the 'skellam_gen' class
# Getting the type of 'rv_discrete' (line 764)
rv_discrete_609066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 18), 'rv_discrete')

class skellam_gen(rv_discrete_609066, ):
    str_609067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, (-1)), 'str', 'A  Skellam discrete random variable.\n\n    %(before_notes)s\n\n    Notes\n    -----\n    Probability distribution of the difference of two correlated or\n    uncorrelated Poisson random variables.\n\n    Let k1 and k2 be two Poisson-distributed r.v. with expected values\n    lam1 and lam2. Then, ``k1 - k2`` follows a Skellam distribution with\n    parameters ``mu1 = lam1 - rho*sqrt(lam1*lam2)`` and\n    ``mu2 = lam2 - rho*sqrt(lam1*lam2)``, where rho is the correlation\n    coefficient between k1 and k2. If the two Poisson-distributed r.v.\n    are independent then ``rho = 0``.\n\n    Parameters mu1 and mu2 must be strictly positive.\n\n    For details see: http://en.wikipedia.org/wiki/Skellam_distribution\n\n    `skellam` takes ``mu1`` and ``mu2`` as shape parameters.\n\n    %(after_notes)s\n\n    %(example)s\n\n    ')

    @norecursion
    def _rvs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rvs'
        module_type_store = module_type_store.open_function_context('_rvs', 792, 4, False)
        # Assigning a type to the variable 'self' (line 793)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        skellam_gen._rvs.__dict__.__setitem__('stypy_localization', localization)
        skellam_gen._rvs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        skellam_gen._rvs.__dict__.__setitem__('stypy_type_store', module_type_store)
        skellam_gen._rvs.__dict__.__setitem__('stypy_function_name', 'skellam_gen._rvs')
        skellam_gen._rvs.__dict__.__setitem__('stypy_param_names_list', ['mu1', 'mu2'])
        skellam_gen._rvs.__dict__.__setitem__('stypy_varargs_param_name', None)
        skellam_gen._rvs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        skellam_gen._rvs.__dict__.__setitem__('stypy_call_defaults', defaults)
        skellam_gen._rvs.__dict__.__setitem__('stypy_call_varargs', varargs)
        skellam_gen._rvs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        skellam_gen._rvs.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'skellam_gen._rvs', ['mu1', 'mu2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rvs', localization, ['mu1', 'mu2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rvs(...)' code ##################

        
        # Assigning a Attribute to a Name (line 793):
        
        # Assigning a Attribute to a Name (line 793):
        # Getting the type of 'self' (line 793)
        self_609068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 12), 'self')
        # Obtaining the member '_size' of a type (line 793)
        _size_609069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 12), self_609068, '_size')
        # Assigning a type to the variable 'n' (line 793)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'n', _size_609069)
        
        # Call to poisson(...): (line 794)
        # Processing the call arguments (line 794)
        # Getting the type of 'mu1' (line 794)
        mu1_609073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 43), 'mu1', False)
        # Getting the type of 'n' (line 794)
        n_609074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 48), 'n', False)
        # Processing the call keyword arguments (line 794)
        kwargs_609075 = {}
        # Getting the type of 'self' (line 794)
        self_609070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 16), 'self', False)
        # Obtaining the member '_random_state' of a type (line 794)
        _random_state_609071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 16), self_609070, '_random_state')
        # Obtaining the member 'poisson' of a type (line 794)
        poisson_609072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 16), _random_state_609071, 'poisson')
        # Calling poisson(args, kwargs) (line 794)
        poisson_call_result_609076 = invoke(stypy.reporting.localization.Localization(__file__, 794, 16), poisson_609072, *[mu1_609073, n_609074], **kwargs_609075)
        
        
        # Call to poisson(...): (line 795)
        # Processing the call arguments (line 795)
        # Getting the type of 'mu2' (line 795)
        mu2_609080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 43), 'mu2', False)
        # Getting the type of 'n' (line 795)
        n_609081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 48), 'n', False)
        # Processing the call keyword arguments (line 795)
        kwargs_609082 = {}
        # Getting the type of 'self' (line 795)
        self_609077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 16), 'self', False)
        # Obtaining the member '_random_state' of a type (line 795)
        _random_state_609078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 16), self_609077, '_random_state')
        # Obtaining the member 'poisson' of a type (line 795)
        poisson_609079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 16), _random_state_609078, 'poisson')
        # Calling poisson(args, kwargs) (line 795)
        poisson_call_result_609083 = invoke(stypy.reporting.localization.Localization(__file__, 795, 16), poisson_609079, *[mu2_609080, n_609081], **kwargs_609082)
        
        # Applying the binary operator '-' (line 794)
        result_sub_609084 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 16), '-', poisson_call_result_609076, poisson_call_result_609083)
        
        # Assigning a type to the variable 'stypy_return_type' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'stypy_return_type', result_sub_609084)
        
        # ################# End of '_rvs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rvs' in the type store
        # Getting the type of 'stypy_return_type' (line 792)
        stypy_return_type_609085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_609085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rvs'
        return stypy_return_type_609085


    @norecursion
    def _pmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pmf'
        module_type_store = module_type_store.open_function_context('_pmf', 797, 4, False)
        # Assigning a type to the variable 'self' (line 798)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        skellam_gen._pmf.__dict__.__setitem__('stypy_localization', localization)
        skellam_gen._pmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        skellam_gen._pmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        skellam_gen._pmf.__dict__.__setitem__('stypy_function_name', 'skellam_gen._pmf')
        skellam_gen._pmf.__dict__.__setitem__('stypy_param_names_list', ['x', 'mu1', 'mu2'])
        skellam_gen._pmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        skellam_gen._pmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        skellam_gen._pmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        skellam_gen._pmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        skellam_gen._pmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        skellam_gen._pmf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'skellam_gen._pmf', ['x', 'mu1', 'mu2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pmf', localization, ['x', 'mu1', 'mu2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pmf(...)' code ##################

        
        # Assigning a Call to a Name (line 798):
        
        # Assigning a Call to a Name (line 798):
        
        # Call to where(...): (line 798)
        # Processing the call arguments (line 798)
        
        # Getting the type of 'x' (line 798)
        x_609088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 22), 'x', False)
        int_609089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 26), 'int')
        # Applying the binary operator '<' (line 798)
        result_lt_609090 = python_operator(stypy.reporting.localization.Localization(__file__, 798, 22), '<', x_609088, int_609089)
        
        
        # Call to _ncx2_pdf(...): (line 799)
        # Processing the call arguments (line 799)
        int_609092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 26), 'int')
        # Getting the type of 'mu2' (line 799)
        mu2_609093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 28), 'mu2', False)
        # Applying the binary operator '*' (line 799)
        result_mul_609094 = python_operator(stypy.reporting.localization.Localization(__file__, 799, 26), '*', int_609092, mu2_609093)
        
        int_609095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 33), 'int')
        int_609096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 36), 'int')
        # Getting the type of 'x' (line 799)
        x_609097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 38), 'x', False)
        # Applying the binary operator '-' (line 799)
        result_sub_609098 = python_operator(stypy.reporting.localization.Localization(__file__, 799, 36), '-', int_609096, x_609097)
        
        # Applying the binary operator '*' (line 799)
        result_mul_609099 = python_operator(stypy.reporting.localization.Localization(__file__, 799, 33), '*', int_609095, result_sub_609098)
        
        int_609100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 42), 'int')
        # Getting the type of 'mu1' (line 799)
        mu1_609101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 44), 'mu1', False)
        # Applying the binary operator '*' (line 799)
        result_mul_609102 = python_operator(stypy.reporting.localization.Localization(__file__, 799, 42), '*', int_609100, mu1_609101)
        
        # Processing the call keyword arguments (line 799)
        kwargs_609103 = {}
        # Getting the type of '_ncx2_pdf' (line 799)
        _ncx2_pdf_609091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 16), '_ncx2_pdf', False)
        # Calling _ncx2_pdf(args, kwargs) (line 799)
        _ncx2_pdf_call_result_609104 = invoke(stypy.reporting.localization.Localization(__file__, 799, 16), _ncx2_pdf_609091, *[result_mul_609094, result_mul_609099, result_mul_609102], **kwargs_609103)
        
        int_609105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 49), 'int')
        # Applying the binary operator '*' (line 799)
        result_mul_609106 = python_operator(stypy.reporting.localization.Localization(__file__, 799, 16), '*', _ncx2_pdf_call_result_609104, int_609105)
        
        
        # Call to _ncx2_pdf(...): (line 800)
        # Processing the call arguments (line 800)
        int_609108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 26), 'int')
        # Getting the type of 'mu1' (line 800)
        mu1_609109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 28), 'mu1', False)
        # Applying the binary operator '*' (line 800)
        result_mul_609110 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 26), '*', int_609108, mu1_609109)
        
        int_609111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 33), 'int')
        int_609112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 36), 'int')
        # Getting the type of 'x' (line 800)
        x_609113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 38), 'x', False)
        # Applying the binary operator '+' (line 800)
        result_add_609114 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 36), '+', int_609112, x_609113)
        
        # Applying the binary operator '*' (line 800)
        result_mul_609115 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 33), '*', int_609111, result_add_609114)
        
        int_609116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 42), 'int')
        # Getting the type of 'mu2' (line 800)
        mu2_609117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 44), 'mu2', False)
        # Applying the binary operator '*' (line 800)
        result_mul_609118 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 42), '*', int_609116, mu2_609117)
        
        # Processing the call keyword arguments (line 800)
        kwargs_609119 = {}
        # Getting the type of '_ncx2_pdf' (line 800)
        _ncx2_pdf_609107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 16), '_ncx2_pdf', False)
        # Calling _ncx2_pdf(args, kwargs) (line 800)
        _ncx2_pdf_call_result_609120 = invoke(stypy.reporting.localization.Localization(__file__, 800, 16), _ncx2_pdf_609107, *[result_mul_609110, result_mul_609115, result_mul_609118], **kwargs_609119)
        
        int_609121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 49), 'int')
        # Applying the binary operator '*' (line 800)
        result_mul_609122 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 16), '*', _ncx2_pdf_call_result_609120, int_609121)
        
        # Processing the call keyword arguments (line 798)
        kwargs_609123 = {}
        # Getting the type of 'np' (line 798)
        np_609086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 13), 'np', False)
        # Obtaining the member 'where' of a type (line 798)
        where_609087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 13), np_609086, 'where')
        # Calling where(args, kwargs) (line 798)
        where_call_result_609124 = invoke(stypy.reporting.localization.Localization(__file__, 798, 13), where_609087, *[result_lt_609090, result_mul_609106, result_mul_609122], **kwargs_609123)
        
        # Assigning a type to the variable 'px' (line 798)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'px', where_call_result_609124)
        # Getting the type of 'px' (line 802)
        px_609125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 15), 'px')
        # Assigning a type to the variable 'stypy_return_type' (line 802)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'stypy_return_type', px_609125)
        
        # ################# End of '_pmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pmf' in the type store
        # Getting the type of 'stypy_return_type' (line 797)
        stypy_return_type_609126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_609126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pmf'
        return stypy_return_type_609126


    @norecursion
    def _cdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cdf'
        module_type_store = module_type_store.open_function_context('_cdf', 804, 4, False)
        # Assigning a type to the variable 'self' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        skellam_gen._cdf.__dict__.__setitem__('stypy_localization', localization)
        skellam_gen._cdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        skellam_gen._cdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        skellam_gen._cdf.__dict__.__setitem__('stypy_function_name', 'skellam_gen._cdf')
        skellam_gen._cdf.__dict__.__setitem__('stypy_param_names_list', ['x', 'mu1', 'mu2'])
        skellam_gen._cdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        skellam_gen._cdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        skellam_gen._cdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        skellam_gen._cdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        skellam_gen._cdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        skellam_gen._cdf.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'skellam_gen._cdf', ['x', 'mu1', 'mu2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cdf', localization, ['x', 'mu1', 'mu2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cdf(...)' code ##################

        
        # Assigning a Call to a Name (line 805):
        
        # Assigning a Call to a Name (line 805):
        
        # Call to floor(...): (line 805)
        # Processing the call arguments (line 805)
        # Getting the type of 'x' (line 805)
        x_609128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 18), 'x', False)
        # Processing the call keyword arguments (line 805)
        kwargs_609129 = {}
        # Getting the type of 'floor' (line 805)
        floor_609127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 12), 'floor', False)
        # Calling floor(args, kwargs) (line 805)
        floor_call_result_609130 = invoke(stypy.reporting.localization.Localization(__file__, 805, 12), floor_609127, *[x_609128], **kwargs_609129)
        
        # Assigning a type to the variable 'x' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'x', floor_call_result_609130)
        
        # Assigning a Call to a Name (line 806):
        
        # Assigning a Call to a Name (line 806):
        
        # Call to where(...): (line 806)
        # Processing the call arguments (line 806)
        
        # Getting the type of 'x' (line 806)
        x_609133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 22), 'x', False)
        int_609134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 26), 'int')
        # Applying the binary operator '<' (line 806)
        result_lt_609135 = python_operator(stypy.reporting.localization.Localization(__file__, 806, 22), '<', x_609133, int_609134)
        
        
        # Call to _ncx2_cdf(...): (line 807)
        # Processing the call arguments (line 807)
        int_609137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 26), 'int')
        # Getting the type of 'mu2' (line 807)
        mu2_609138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 28), 'mu2', False)
        # Applying the binary operator '*' (line 807)
        result_mul_609139 = python_operator(stypy.reporting.localization.Localization(__file__, 807, 26), '*', int_609137, mu2_609138)
        
        int_609140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 33), 'int')
        # Getting the type of 'x' (line 807)
        x_609141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 36), 'x', False)
        # Applying the binary operator '*' (line 807)
        result_mul_609142 = python_operator(stypy.reporting.localization.Localization(__file__, 807, 33), '*', int_609140, x_609141)
        
        int_609143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 39), 'int')
        # Getting the type of 'mu1' (line 807)
        mu1_609144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 41), 'mu1', False)
        # Applying the binary operator '*' (line 807)
        result_mul_609145 = python_operator(stypy.reporting.localization.Localization(__file__, 807, 39), '*', int_609143, mu1_609144)
        
        # Processing the call keyword arguments (line 807)
        kwargs_609146 = {}
        # Getting the type of '_ncx2_cdf' (line 807)
        _ncx2_cdf_609136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 16), '_ncx2_cdf', False)
        # Calling _ncx2_cdf(args, kwargs) (line 807)
        _ncx2_cdf_call_result_609147 = invoke(stypy.reporting.localization.Localization(__file__, 807, 16), _ncx2_cdf_609136, *[result_mul_609139, result_mul_609142, result_mul_609145], **kwargs_609146)
        
        int_609148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 16), 'int')
        
        # Call to _ncx2_cdf(...): (line 808)
        # Processing the call arguments (line 808)
        int_609150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 28), 'int')
        # Getting the type of 'mu1' (line 808)
        mu1_609151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 30), 'mu1', False)
        # Applying the binary operator '*' (line 808)
        result_mul_609152 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 28), '*', int_609150, mu1_609151)
        
        int_609153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 35), 'int')
        # Getting the type of 'x' (line 808)
        x_609154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 38), 'x', False)
        int_609155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 40), 'int')
        # Applying the binary operator '+' (line 808)
        result_add_609156 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 38), '+', x_609154, int_609155)
        
        # Applying the binary operator '*' (line 808)
        result_mul_609157 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 35), '*', int_609153, result_add_609156)
        
        int_609158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 44), 'int')
        # Getting the type of 'mu2' (line 808)
        mu2_609159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 46), 'mu2', False)
        # Applying the binary operator '*' (line 808)
        result_mul_609160 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 44), '*', int_609158, mu2_609159)
        
        # Processing the call keyword arguments (line 808)
        kwargs_609161 = {}
        # Getting the type of '_ncx2_cdf' (line 808)
        _ncx2_cdf_609149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 18), '_ncx2_cdf', False)
        # Calling _ncx2_cdf(args, kwargs) (line 808)
        _ncx2_cdf_call_result_609162 = invoke(stypy.reporting.localization.Localization(__file__, 808, 18), _ncx2_cdf_609149, *[result_mul_609152, result_mul_609157, result_mul_609160], **kwargs_609161)
        
        # Applying the binary operator '-' (line 808)
        result_sub_609163 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 16), '-', int_609148, _ncx2_cdf_call_result_609162)
        
        # Processing the call keyword arguments (line 806)
        kwargs_609164 = {}
        # Getting the type of 'np' (line 806)
        np_609131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 13), 'np', False)
        # Obtaining the member 'where' of a type (line 806)
        where_609132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 13), np_609131, 'where')
        # Calling where(args, kwargs) (line 806)
        where_call_result_609165 = invoke(stypy.reporting.localization.Localization(__file__, 806, 13), where_609132, *[result_lt_609135, _ncx2_cdf_call_result_609147, result_sub_609163], **kwargs_609164)
        
        # Assigning a type to the variable 'px' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'px', where_call_result_609165)
        # Getting the type of 'px' (line 809)
        px_609166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 15), 'px')
        # Assigning a type to the variable 'stypy_return_type' (line 809)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'stypy_return_type', px_609166)
        
        # ################# End of '_cdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cdf' in the type store
        # Getting the type of 'stypy_return_type' (line 804)
        stypy_return_type_609167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_609167)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cdf'
        return stypy_return_type_609167


    @norecursion
    def _stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stats'
        module_type_store = module_type_store.open_function_context('_stats', 811, 4, False)
        # Assigning a type to the variable 'self' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        skellam_gen._stats.__dict__.__setitem__('stypy_localization', localization)
        skellam_gen._stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        skellam_gen._stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        skellam_gen._stats.__dict__.__setitem__('stypy_function_name', 'skellam_gen._stats')
        skellam_gen._stats.__dict__.__setitem__('stypy_param_names_list', ['mu1', 'mu2'])
        skellam_gen._stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        skellam_gen._stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        skellam_gen._stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        skellam_gen._stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        skellam_gen._stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        skellam_gen._stats.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'skellam_gen._stats', ['mu1', 'mu2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_stats', localization, ['mu1', 'mu2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_stats(...)' code ##################

        
        # Assigning a BinOp to a Name (line 812):
        
        # Assigning a BinOp to a Name (line 812):
        # Getting the type of 'mu1' (line 812)
        mu1_609168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 15), 'mu1')
        # Getting the type of 'mu2' (line 812)
        mu2_609169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 21), 'mu2')
        # Applying the binary operator '-' (line 812)
        result_sub_609170 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 15), '-', mu1_609168, mu2_609169)
        
        # Assigning a type to the variable 'mean' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'mean', result_sub_609170)
        
        # Assigning a BinOp to a Name (line 813):
        
        # Assigning a BinOp to a Name (line 813):
        # Getting the type of 'mu1' (line 813)
        mu1_609171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 14), 'mu1')
        # Getting the type of 'mu2' (line 813)
        mu2_609172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 20), 'mu2')
        # Applying the binary operator '+' (line 813)
        result_add_609173 = python_operator(stypy.reporting.localization.Localization(__file__, 813, 14), '+', mu1_609171, mu2_609172)
        
        # Assigning a type to the variable 'var' (line 813)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'var', result_add_609173)
        
        # Assigning a BinOp to a Name (line 814):
        
        # Assigning a BinOp to a Name (line 814):
        # Getting the type of 'mean' (line 814)
        mean_609174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 13), 'mean')
        
        # Call to sqrt(...): (line 814)
        # Processing the call arguments (line 814)
        # Getting the type of 'var' (line 814)
        var_609176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 26), 'var', False)
        int_609177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 32), 'int')
        # Applying the binary operator '**' (line 814)
        result_pow_609178 = python_operator(stypy.reporting.localization.Localization(__file__, 814, 25), '**', var_609176, int_609177)
        
        # Processing the call keyword arguments (line 814)
        kwargs_609179 = {}
        # Getting the type of 'sqrt' (line 814)
        sqrt_609175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 20), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 814)
        sqrt_call_result_609180 = invoke(stypy.reporting.localization.Localization(__file__, 814, 20), sqrt_609175, *[result_pow_609178], **kwargs_609179)
        
        # Applying the binary operator 'div' (line 814)
        result_div_609181 = python_operator(stypy.reporting.localization.Localization(__file__, 814, 13), 'div', mean_609174, sqrt_call_result_609180)
        
        # Assigning a type to the variable 'g1' (line 814)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'g1', result_div_609181)
        
        # Assigning a BinOp to a Name (line 815):
        
        # Assigning a BinOp to a Name (line 815):
        int_609182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 13), 'int')
        # Getting the type of 'var' (line 815)
        var_609183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 17), 'var')
        # Applying the binary operator 'div' (line 815)
        result_div_609184 = python_operator(stypy.reporting.localization.Localization(__file__, 815, 13), 'div', int_609182, var_609183)
        
        # Assigning a type to the variable 'g2' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'g2', result_div_609184)
        
        # Obtaining an instance of the builtin type 'tuple' (line 816)
        tuple_609185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 816)
        # Adding element type (line 816)
        # Getting the type of 'mean' (line 816)
        mean_609186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 15), 'mean')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 15), tuple_609185, mean_609186)
        # Adding element type (line 816)
        # Getting the type of 'var' (line 816)
        var_609187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 21), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 15), tuple_609185, var_609187)
        # Adding element type (line 816)
        # Getting the type of 'g1' (line 816)
        g1_609188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 26), 'g1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 15), tuple_609185, g1_609188)
        # Adding element type (line 816)
        # Getting the type of 'g2' (line 816)
        g2_609189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 30), 'g2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 15), tuple_609185, g2_609189)
        
        # Assigning a type to the variable 'stypy_return_type' (line 816)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'stypy_return_type', tuple_609185)
        
        # ################# End of '_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 811)
        stypy_return_type_609190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_609190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stats'
        return stypy_return_type_609190


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 764, 0, False)
        # Assigning a type to the variable 'self' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'skellam_gen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'skellam_gen' (line 764)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 0), 'skellam_gen', skellam_gen)

# Assigning a Call to a Name (line 817):

# Assigning a Call to a Name (line 817):

# Call to skellam_gen(...): (line 817)
# Processing the call keyword arguments (line 817)

# Getting the type of 'np' (line 817)
np_609192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 25), 'np', False)
# Obtaining the member 'inf' of a type (line 817)
inf_609193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 25), np_609192, 'inf')
# Applying the 'usub' unary operator (line 817)
result___neg___609194 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 24), 'usub', inf_609193)

keyword_609195 = result___neg___609194
str_609196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 38), 'str', 'skellam')
keyword_609197 = str_609196
str_609198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 58), 'str', 'A Skellam')
keyword_609199 = str_609198
kwargs_609200 = {'a': keyword_609195, 'name': keyword_609197, 'longname': keyword_609199}
# Getting the type of 'skellam_gen' (line 817)
skellam_gen_609191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 10), 'skellam_gen', False)
# Calling skellam_gen(args, kwargs) (line 817)
skellam_gen_call_result_609201 = invoke(stypy.reporting.localization.Localization(__file__, 817, 10), skellam_gen_609191, *[], **kwargs_609200)

# Assigning a type to the variable 'skellam' (line 817)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 0), 'skellam', skellam_gen_call_result_609201)

# Assigning a Call to a Name (line 821):

# Assigning a Call to a Name (line 821):

# Call to list(...): (line 821)
# Processing the call arguments (line 821)

# Call to items(...): (line 821)
# Processing the call keyword arguments (line 821)
kwargs_609207 = {}

# Call to globals(...): (line 821)
# Processing the call keyword arguments (line 821)
kwargs_609204 = {}
# Getting the type of 'globals' (line 821)
globals_609203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 13), 'globals', False)
# Calling globals(args, kwargs) (line 821)
globals_call_result_609205 = invoke(stypy.reporting.localization.Localization(__file__, 821, 13), globals_609203, *[], **kwargs_609204)

# Obtaining the member 'items' of a type (line 821)
items_609206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 13), globals_call_result_609205, 'items')
# Calling items(args, kwargs) (line 821)
items_call_result_609208 = invoke(stypy.reporting.localization.Localization(__file__, 821, 13), items_609206, *[], **kwargs_609207)

# Processing the call keyword arguments (line 821)
kwargs_609209 = {}
# Getting the type of 'list' (line 821)
list_609202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'list', False)
# Calling list(args, kwargs) (line 821)
list_call_result_609210 = invoke(stypy.reporting.localization.Localization(__file__, 821, 8), list_609202, *[items_call_result_609208], **kwargs_609209)

# Assigning a type to the variable 'pairs' (line 821)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 0), 'pairs', list_call_result_609210)

# Assigning a Call to a Tuple (line 822):

# Assigning a Subscript to a Name (line 822):

# Obtaining the type of the subscript
int_609211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 0), 'int')

# Call to get_distribution_names(...): (line 822)
# Processing the call arguments (line 822)
# Getting the type of 'pairs' (line 822)
pairs_609213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 56), 'pairs', False)
# Getting the type of 'rv_discrete' (line 822)
rv_discrete_609214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 63), 'rv_discrete', False)
# Processing the call keyword arguments (line 822)
kwargs_609215 = {}
# Getting the type of 'get_distribution_names' (line 822)
get_distribution_names_609212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 33), 'get_distribution_names', False)
# Calling get_distribution_names(args, kwargs) (line 822)
get_distribution_names_call_result_609216 = invoke(stypy.reporting.localization.Localization(__file__, 822, 33), get_distribution_names_609212, *[pairs_609213, rv_discrete_609214], **kwargs_609215)

# Obtaining the member '__getitem__' of a type (line 822)
getitem___609217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 0), get_distribution_names_call_result_609216, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 822)
subscript_call_result_609218 = invoke(stypy.reporting.localization.Localization(__file__, 822, 0), getitem___609217, int_609211)

# Assigning a type to the variable 'tuple_var_assignment_606756' (line 822)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), 'tuple_var_assignment_606756', subscript_call_result_609218)

# Assigning a Subscript to a Name (line 822):

# Obtaining the type of the subscript
int_609219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 0), 'int')

# Call to get_distribution_names(...): (line 822)
# Processing the call arguments (line 822)
# Getting the type of 'pairs' (line 822)
pairs_609221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 56), 'pairs', False)
# Getting the type of 'rv_discrete' (line 822)
rv_discrete_609222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 63), 'rv_discrete', False)
# Processing the call keyword arguments (line 822)
kwargs_609223 = {}
# Getting the type of 'get_distribution_names' (line 822)
get_distribution_names_609220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 33), 'get_distribution_names', False)
# Calling get_distribution_names(args, kwargs) (line 822)
get_distribution_names_call_result_609224 = invoke(stypy.reporting.localization.Localization(__file__, 822, 33), get_distribution_names_609220, *[pairs_609221, rv_discrete_609222], **kwargs_609223)

# Obtaining the member '__getitem__' of a type (line 822)
getitem___609225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 0), get_distribution_names_call_result_609224, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 822)
subscript_call_result_609226 = invoke(stypy.reporting.localization.Localization(__file__, 822, 0), getitem___609225, int_609219)

# Assigning a type to the variable 'tuple_var_assignment_606757' (line 822)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), 'tuple_var_assignment_606757', subscript_call_result_609226)

# Assigning a Name to a Name (line 822):
# Getting the type of 'tuple_var_assignment_606756' (line 822)
tuple_var_assignment_606756_609227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), 'tuple_var_assignment_606756')
# Assigning a type to the variable '_distn_names' (line 822)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), '_distn_names', tuple_var_assignment_606756_609227)

# Assigning a Name to a Name (line 822):
# Getting the type of 'tuple_var_assignment_606757' (line 822)
tuple_var_assignment_606757_609228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), 'tuple_var_assignment_606757')
# Assigning a type to the variable '_distn_gen_names' (line 822)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 14), '_distn_gen_names', tuple_var_assignment_606757_609228)

# Assigning a BinOp to a Name (line 824):

# Assigning a BinOp to a Name (line 824):
# Getting the type of '_distn_names' (line 824)
_distn_names_609229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 10), '_distn_names')
# Getting the type of '_distn_gen_names' (line 824)
_distn_gen_names_609230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 25), '_distn_gen_names')
# Applying the binary operator '+' (line 824)
result_add_609231 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 10), '+', _distn_names_609229, _distn_gen_names_609230)

# Assigning a type to the variable '__all__' (line 824)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 0), '__all__', result_add_609231)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
