
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ========================================
3: Special functions (:mod:`scipy.special`)
4: ========================================
5: 
6: .. module:: scipy.special
7: 
8: Nearly all of the functions below are universal functions and follow
9: broadcasting and automatic array-looping rules. Exceptions are
10: noted.
11: 
12: .. seealso::
13: 
14:    `scipy.special.cython_special` -- Typed Cython versions of special functions
15: 
16: 
17: Error handling
18: ==============
19: 
20: Errors are handled by returning NaNs or other appropriate values.
21: Some of the special function routines can emit warnings or raise
22: exceptions when an error occurs. By default this is disabled; to
23: query and control the current error handling state the following
24: functions are provided.
25: 
26: .. autosummary::
27:    :toctree: generated/
28: 
29:    geterr                 -- Get the current way of handling special-function errors.
30:    seterr                 -- Set how special-function errors are handled.
31:    errstate               -- Context manager for special-function error handling.
32:    SpecialFunctionWarning -- Warning that can be emitted by special functions.
33:    SpecialFunctionError   -- Exception that can be raised by special functions.
34: 
35: Available functions
36: ===================
37: 
38: Airy functions
39: --------------
40: 
41: .. autosummary::
42:    :toctree: generated/
43: 
44:    airy     -- Airy functions and their derivatives.
45:    airye    -- Exponentially scaled Airy functions and their derivatives.
46:    ai_zeros -- [+]Compute `nt` zeros and values of the Airy function Ai and its derivative.
47:    bi_zeros -- [+]Compute `nt` zeros and values of the Airy function Bi and its derivative.
48:    itairy   -- Integrals of Airy functions
49: 
50: 
51: Elliptic Functions and Integrals
52: --------------------------------
53: 
54: .. autosummary::
55:    :toctree: generated/
56: 
57:    ellipj    -- Jacobian elliptic functions
58:    ellipk    -- Complete elliptic integral of the first kind.
59:    ellipkm1  -- Complete elliptic integral of the first kind around `m` = 1
60:    ellipkinc -- Incomplete elliptic integral of the first kind
61:    ellipe    -- Complete elliptic integral of the second kind
62:    ellipeinc -- Incomplete elliptic integral of the second kind
63: 
64: Bessel Functions
65: ----------------
66: 
67: .. autosummary::
68:    :toctree: generated/
69: 
70:    jv       -- Bessel function of the first kind of real order and complex argument.
71:    jn       -- Bessel function of the first kind of real order and complex argument
72:    jve      -- Exponentially scaled Bessel function of order `v`.
73:    yn       -- Bessel function of the second kind of integer order and real argument.
74:    yv       -- Bessel function of the second kind of real order and complex argument.
75:    yve      -- Exponentially scaled Bessel function of the second kind of real order.
76:    kn       -- Modified Bessel function of the second kind of integer order `n`
77:    kv       -- Modified Bessel function of the second kind of real order `v`
78:    kve      -- Exponentially scaled modified Bessel function of the second kind.
79:    iv       -- Modified Bessel function of the first kind of real order.
80:    ive      -- Exponentially scaled modified Bessel function of the first kind
81:    hankel1  -- Hankel function of the first kind
82:    hankel1e -- Exponentially scaled Hankel function of the first kind
83:    hankel2  -- Hankel function of the second kind
84:    hankel2e -- Exponentially scaled Hankel function of the second kind
85: 
86: The following is not an universal function:
87: 
88: .. autosummary::
89:    :toctree: generated/
90: 
91:    lmbda -- [+]Jahnke-Emden Lambda function, Lambdav(x).
92: 
93: Zeros of Bessel Functions
94: ^^^^^^^^^^^^^^^^^^^^^^^^^
95: 
96: These are not universal functions:
97: 
98: .. autosummary::
99:    :toctree: generated/
100: 
101:    jnjnp_zeros -- [+]Compute zeros of integer-order Bessel functions Jn and Jn'.
102:    jnyn_zeros  -- [+]Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).
103:    jn_zeros    -- [+]Compute zeros of integer-order Bessel function Jn(x).
104:    jnp_zeros   -- [+]Compute zeros of integer-order Bessel function derivative Jn'(x).
105:    yn_zeros    -- [+]Compute zeros of integer-order Bessel function Yn(x).
106:    ynp_zeros   -- [+]Compute zeros of integer-order Bessel function derivative Yn'(x).
107:    y0_zeros    -- [+]Compute nt zeros of Bessel function Y0(z), and derivative at each zero.
108:    y1_zeros    -- [+]Compute nt zeros of Bessel function Y1(z), and derivative at each zero.
109:    y1p_zeros   -- [+]Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.
110: 
111: Faster versions of common Bessel Functions
112: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
113: 
114: .. autosummary::
115:    :toctree: generated/
116: 
117:    j0  -- Bessel function of the first kind of order 0.
118:    j1  -- Bessel function of the first kind of order 1.
119:    y0  -- Bessel function of the second kind of order 0.
120:    y1  -- Bessel function of the second kind of order 1.
121:    i0  -- Modified Bessel function of order 0.
122:    i0e -- Exponentially scaled modified Bessel function of order 0.
123:    i1  -- Modified Bessel function of order 1.
124:    i1e -- Exponentially scaled modified Bessel function of order 1.
125:    k0  -- Modified Bessel function of the second kind of order 0, :math:`K_0`.
126:    k0e -- Exponentially scaled modified Bessel function K of order 0
127:    k1  -- Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.
128:    k1e -- Exponentially scaled modified Bessel function K of order 1
129: 
130: Integrals of Bessel Functions
131: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
132: 
133: .. autosummary::
134:    :toctree: generated/
135: 
136:    itj0y0     -- Integrals of Bessel functions of order 0
137:    it2j0y0    -- Integrals related to Bessel functions of order 0
138:    iti0k0     -- Integrals of modified Bessel functions of order 0
139:    it2i0k0    -- Integrals related to modified Bessel functions of order 0
140:    besselpoly -- [+]Weighted integral of a Bessel function.
141: 
142: Derivatives of Bessel Functions
143: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
144: 
145: .. autosummary::
146:    :toctree: generated/
147: 
148:    jvp  -- Compute nth derivative of Bessel function Jv(z) with respect to `z`.
149:    yvp  -- Compute nth derivative of Bessel function Yv(z) with respect to `z`.
150:    kvp  -- Compute nth derivative of real-order modified Bessel function Kv(z)
151:    ivp  -- Compute nth derivative of modified Bessel function Iv(z) with respect to `z`.
152:    h1vp -- Compute nth derivative of Hankel function H1v(z) with respect to `z`.
153:    h2vp -- Compute nth derivative of Hankel function H2v(z) with respect to `z`.
154: 
155: Spherical Bessel Functions
156: ^^^^^^^^^^^^^^^^^^^^^^^^^^
157: 
158: .. autosummary::
159:    :toctree: generated/
160: 
161:    spherical_jn -- Spherical Bessel function of the first kind or its derivative.
162:    spherical_yn -- Spherical Bessel function of the second kind or its derivative.
163:    spherical_in -- Modified spherical Bessel function of the first kind or its derivative.
164:    spherical_kn -- Modified spherical Bessel function of the second kind or its derivative.
165: 
166: Riccati-Bessel Functions
167: ^^^^^^^^^^^^^^^^^^^^^^^^
168: 
169: These are not universal functions:
170: 
171: .. autosummary::
172:    :toctree: generated/
173: 
174:    riccati_jn -- [+]Compute Ricatti-Bessel function of the first kind and its derivative.
175:    riccati_yn -- [+]Compute Ricatti-Bessel function of the second kind and its derivative.
176: 
177: Struve Functions
178: ----------------
179: 
180: .. autosummary::
181:    :toctree: generated/
182: 
183:    struve       -- Struve function.
184:    modstruve    -- Modified Struve function.
185:    itstruve0    -- Integral of the Struve function of order 0.
186:    it2struve0   -- Integral related to the Struve function of order 0.
187:    itmodstruve0 -- Integral of the modified Struve function of order 0.
188: 
189: 
190: Raw Statistical Functions
191: -------------------------
192: 
193: .. seealso:: :mod:`scipy.stats`: Friendly versions of these functions.
194: 
195: .. autosummary::
196:    :toctree: generated/
197: 
198:    bdtr         -- Binomial distribution cumulative distribution function.
199:    bdtrc        -- Binomial distribution survival function.
200:    bdtri        -- Inverse function to `bdtr` with respect to `p`.
201:    bdtrik       -- Inverse function to `bdtr` with respect to `k`.
202:    bdtrin       -- Inverse function to `bdtr` with respect to `n`.
203:    btdtr        -- Cumulative density function of the beta distribution.
204:    btdtri       -- The `p`-th quantile of the beta distribution.
205:    btdtria      -- Inverse of `btdtr` with respect to `a`.
206:    btdtrib      -- btdtria(a, p, x)
207:    fdtr         -- F cumulative distribution function.
208:    fdtrc        -- F survival function.
209:    fdtri        -- The `p`-th quantile of the F-distribution.
210:    fdtridfd     -- Inverse to `fdtr` vs dfd
211:    gdtr         -- Gamma distribution cumulative density function.
212:    gdtrc        -- Gamma distribution survival function.
213:    gdtria       -- Inverse of `gdtr` vs a.
214:    gdtrib       -- Inverse of `gdtr` vs b.
215:    gdtrix       -- Inverse of `gdtr` vs x.
216:    nbdtr        -- Negative binomial cumulative distribution function.
217:    nbdtrc       -- Negative binomial survival function.
218:    nbdtri       -- Inverse of `nbdtr` vs `p`.
219:    nbdtrik      -- Inverse of `nbdtr` vs `k`.
220:    nbdtrin      -- Inverse of `nbdtr` vs `n`.
221:    ncfdtr       -- Cumulative distribution function of the non-central F distribution.
222:    ncfdtridfd   -- Calculate degrees of freedom (denominator) for the noncentral F-distribution.
223:    ncfdtridfn   -- Calculate degrees of freedom (numerator) for the noncentral F-distribution.
224:    ncfdtri      -- Inverse cumulative distribution function of the non-central F distribution.
225:    ncfdtrinc    -- Calculate non-centrality parameter for non-central F distribution.
226:    nctdtr       -- Cumulative distribution function of the non-central `t` distribution.
227:    nctdtridf    -- Calculate degrees of freedom for non-central t distribution.
228:    nctdtrit     -- Inverse cumulative distribution function of the non-central t distribution.
229:    nctdtrinc    -- Calculate non-centrality parameter for non-central t distribution.
230:    nrdtrimn     -- Calculate mean of normal distribution given other params.
231:    nrdtrisd     -- Calculate standard deviation of normal distribution given other params.
232:    pdtr         -- Poisson cumulative distribution function
233:    pdtrc        -- Poisson survival function
234:    pdtri        -- Inverse to `pdtr` vs m
235:    pdtrik       -- Inverse to `pdtr` vs k
236:    stdtr        -- Student t distribution cumulative density function
237:    stdtridf     -- Inverse of `stdtr` vs df
238:    stdtrit      -- Inverse of `stdtr` vs `t`
239:    chdtr        -- Chi square cumulative distribution function
240:    chdtrc       -- Chi square survival function
241:    chdtri       -- Inverse to `chdtrc`
242:    chdtriv      -- Inverse to `chdtr` vs `v`
243:    ndtr         -- Gaussian cumulative distribution function.
244:    log_ndtr     -- Logarithm of Gaussian cumulative distribution function.
245:    ndtri        -- Inverse of `ndtr` vs x
246:    chndtr       -- Non-central chi square cumulative distribution function
247:    chndtridf    -- Inverse to `chndtr` vs `df`
248:    chndtrinc    -- Inverse to `chndtr` vs `nc`
249:    chndtrix     -- Inverse to `chndtr` vs `x`
250:    smirnov      -- Kolmogorov-Smirnov complementary cumulative distribution function
251:    smirnovi     -- Inverse to `smirnov`
252:    kolmogorov   -- Complementary cumulative distribution function of Kolmogorov distribution
253:    kolmogi      -- Inverse function to kolmogorov
254:    tklmbda      -- Tukey-Lambda cumulative distribution function
255:    logit        -- Logit ufunc for ndarrays.
256:    expit        -- Expit ufunc for ndarrays.
257:    boxcox       -- Compute the Box-Cox transformation.
258:    boxcox1p     -- Compute the Box-Cox transformation of 1 + `x`.
259:    inv_boxcox   -- Compute the inverse of the Box-Cox transformation.
260:    inv_boxcox1p -- Compute the inverse of the Box-Cox transformation.
261: 
262: 
263: Information Theory Functions
264: ----------------------------
265: 
266: .. autosummary::
267:    :toctree: generated/
268: 
269:    entr         -- Elementwise function for computing entropy.
270:    rel_entr     -- Elementwise function for computing relative entropy.
271:    kl_div       -- Elementwise function for computing Kullback-Leibler divergence.
272:    huber        -- Huber loss function.
273:    pseudo_huber -- Pseudo-Huber loss function.
274: 
275: 
276: Gamma and Related Functions
277: ---------------------------
278: 
279: .. autosummary::
280:    :toctree: generated/
281: 
282:    gamma        -- Gamma function.
283:    gammaln      -- Logarithm of the absolute value of the Gamma function for real inputs.
284:    loggamma     -- Principal branch of the logarithm of the Gamma function.
285:    gammasgn     -- Sign of the gamma function.
286:    gammainc     -- Regularized lower incomplete gamma function.
287:    gammaincinv  -- Inverse to `gammainc`
288:    gammaincc    -- Regularized upper incomplete gamma function.
289:    gammainccinv -- Inverse to `gammaincc`
290:    beta         -- Beta function.
291:    betaln       -- Natural logarithm of absolute value of beta function.
292:    betainc      -- Incomplete beta integral.
293:    betaincinv   -- Inverse function to beta integral.
294:    psi          -- The digamma function.
295:    rgamma       -- Gamma function inverted
296:    polygamma    -- Polygamma function n.
297:    multigammaln -- Returns the log of multivariate gamma, also sometimes called the generalized gamma.
298:    digamma      -- psi(x[, out])
299:    poch         -- Rising factorial (z)_m
300: 
301: 
302: Error Function and Fresnel Integrals
303: ------------------------------------
304: 
305: .. autosummary::
306:    :toctree: generated/
307: 
308:    erf           -- Returns the error function of complex argument.
309:    erfc          -- Complementary error function, ``1 - erf(x)``.
310:    erfcx         -- Scaled complementary error function, ``exp(x**2) * erfc(x)``.
311:    erfi          -- Imaginary error function, ``-i erf(i z)``.
312:    erfinv        -- Inverse function for erf.
313:    erfcinv       -- Inverse function for erfc.
314:    wofz          -- Faddeeva function
315:    dawsn         -- Dawson's integral.
316:    fresnel       -- Fresnel sin and cos integrals
317:    fresnel_zeros -- Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).
318:    modfresnelp   -- Modified Fresnel positive integrals
319:    modfresnelm   -- Modified Fresnel negative integrals
320: 
321: These are not universal functions:
322: 
323: .. autosummary::
324:    :toctree: generated/
325: 
326:    erf_zeros      -- [+]Compute nt complex zeros of error function erf(z).
327:    fresnelc_zeros -- [+]Compute nt complex zeros of cosine Fresnel integral C(z).
328:    fresnels_zeros -- [+]Compute nt complex zeros of sine Fresnel integral S(z).
329: 
330: Legendre Functions
331: ------------------
332: 
333: .. autosummary::
334:    :toctree: generated/
335: 
336:    lpmv     -- Associated Legendre function of integer order and real degree.
337:    sph_harm -- Compute spherical harmonics.
338: 
339: These are not universal functions:
340: 
341: .. autosummary::
342:    :toctree: generated/
343: 
344:    clpmn -- [+]Associated Legendre function of the first kind for complex arguments.
345:    lpn   -- [+]Legendre function of the first kind.
346:    lqn   -- [+]Legendre function of the second kind.
347:    lpmn  -- [+]Sequence of associated Legendre functions of the first kind.
348:    lqmn  -- [+]Sequence of associated Legendre functions of the second kind.
349: 
350: Ellipsoidal Harmonics
351: ---------------------
352: 
353: .. autosummary::
354:    :toctree: generated/
355: 
356:    ellip_harm   -- Ellipsoidal harmonic functions E^p_n(l)
357:    ellip_harm_2 -- Ellipsoidal harmonic functions F^p_n(l)
358:    ellip_normal -- Ellipsoidal harmonic normalization constants gamma^p_n
359: 
360: Orthogonal polynomials
361: ----------------------
362: 
363: The following functions evaluate values of orthogonal polynomials:
364: 
365: .. autosummary::
366:    :toctree: generated/
367: 
368:    assoc_laguerre   -- Compute the generalized (associated) Laguerre polynomial of degree n and order k.
369:    eval_legendre    -- Evaluate Legendre polynomial at a point.
370:    eval_chebyt      -- Evaluate Chebyshev polynomial of the first kind at a point.
371:    eval_chebyu      -- Evaluate Chebyshev polynomial of the second kind at a point.
372:    eval_chebyc      -- Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a point.
373:    eval_chebys      -- Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a point.
374:    eval_jacobi      -- Evaluate Jacobi polynomial at a point.
375:    eval_laguerre    -- Evaluate Laguerre polynomial at a point.
376:    eval_genlaguerre -- Evaluate generalized Laguerre polynomial at a point.
377:    eval_hermite     -- Evaluate physicist's Hermite polynomial at a point.
378:    eval_hermitenorm -- Evaluate probabilist's (normalized) Hermite polynomial at a point.
379:    eval_gegenbauer  -- Evaluate Gegenbauer polynomial at a point.
380:    eval_sh_legendre -- Evaluate shifted Legendre polynomial at a point.
381:    eval_sh_chebyt   -- Evaluate shifted Chebyshev polynomial of the first kind at a point.
382:    eval_sh_chebyu   -- Evaluate shifted Chebyshev polynomial of the second kind at a point.
383:    eval_sh_jacobi   -- Evaluate shifted Jacobi polynomial at a point.
384: 
385: The following functions compute roots and quadrature weights for
386: orthogonal polynomials:
387: 
388: .. autosummary::
389:    :toctree: generated/
390: 
391:    roots_legendre    -- Gauss-Legendre quadrature.
392:    roots_chebyt      -- Gauss-Chebyshev (first kind) quadrature.
393:    roots_chebyu      -- Gauss-Chebyshev (second kind) quadrature.
394:    roots_chebyc      -- Gauss-Chebyshev (first kind) quadrature.
395:    roots_chebys      -- Gauss-Chebyshev (second kind) quadrature.
396:    roots_jacobi      -- Gauss-Jacobi quadrature.
397:    roots_laguerre    -- Gauss-Laguerre quadrature.
398:    roots_genlaguerre -- Gauss-generalized Laguerre quadrature.
399:    roots_hermite     -- Gauss-Hermite (physicst's) quadrature.
400:    roots_hermitenorm -- Gauss-Hermite (statistician's) quadrature.
401:    roots_gegenbauer  -- Gauss-Gegenbauer quadrature.
402:    roots_sh_legendre -- Gauss-Legendre (shifted) quadrature.
403:    roots_sh_chebyt   -- Gauss-Chebyshev (first kind, shifted) quadrature.
404:    roots_sh_chebyu   -- Gauss-Chebyshev (second kind, shifted) quadrature.
405:    roots_sh_jacobi   -- Gauss-Jacobi (shifted) quadrature.
406: 
407: The functions below, in turn, return the polynomial coefficients in
408: :class:`~.orthopoly1d` objects, which function similarly as :ref:`numpy.poly1d`.
409: The :class:`~.orthopoly1d` class also has an attribute ``weights`` which returns
410: the roots, weights, and total weights for the appropriate form of Gaussian
411: quadrature.  These are returned in an ``n x 3`` array with roots in the first
412: column, weights in the second column, and total weights in the final column.
413: Note that :class:`~.orthopoly1d` objects are converted to ``poly1d`` when doing
414: arithmetic, and lose information of the original orthogonal polynomial.
415: 
416: .. autosummary::
417:    :toctree: generated/
418: 
419:    legendre    -- [+]Legendre polynomial.
420:    chebyt      -- [+]Chebyshev polynomial of the first kind.
421:    chebyu      -- [+]Chebyshev polynomial of the second kind.
422:    chebyc      -- [+]Chebyshev polynomial of the first kind on :math:`[-2, 2]`.
423:    chebys      -- [+]Chebyshev polynomial of the second kind on :math:`[-2, 2]`.
424:    jacobi      -- [+]Jacobi polynomial.
425:    laguerre    -- [+]Laguerre polynomial.
426:    genlaguerre -- [+]Generalized (associated) Laguerre polynomial.
427:    hermite     -- [+]Physicist's Hermite polynomial.
428:    hermitenorm -- [+]Normalized (probabilist's) Hermite polynomial.
429:    gegenbauer  -- [+]Gegenbauer (ultraspherical) polynomial.
430:    sh_legendre -- [+]Shifted Legendre polynomial.
431:    sh_chebyt   -- [+]Shifted Chebyshev polynomial of the first kind.
432:    sh_chebyu   -- [+]Shifted Chebyshev polynomial of the second kind.
433:    sh_jacobi   -- [+]Shifted Jacobi polynomial.
434: 
435: .. warning::
436: 
437:    Computing values of high-order polynomials (around ``order > 20``) using
438:    polynomial coefficients is numerically unstable. To evaluate polynomial
439:    values, the ``eval_*`` functions should be used instead.
440: 
441: 
442: Hypergeometric Functions
443: ------------------------
444: 
445: .. autosummary::
446:    :toctree: generated/
447: 
448:    hyp2f1 -- Gauss hypergeometric function 2F1(a, b; c; z).
449:    hyp1f1 -- Confluent hypergeometric function 1F1(a, b; x)
450:    hyperu -- Confluent hypergeometric function U(a, b, x) of the second kind
451:    hyp0f1 -- Confluent hypergeometric limit function 0F1.
452:    hyp2f0 -- Hypergeometric function 2F0 in y and an error estimate
453:    hyp1f2 -- Hypergeometric function 1F2 and error estimate
454:    hyp3f0 -- Hypergeometric function 3F0 in y and an error estimate
455: 
456: 
457: Parabolic Cylinder Functions
458: ----------------------------
459: 
460: .. autosummary::
461:    :toctree: generated/
462: 
463:    pbdv -- Parabolic cylinder function D
464:    pbvv -- Parabolic cylinder function V
465:    pbwa -- Parabolic cylinder function W
466: 
467: These are not universal functions:
468: 
469: .. autosummary::
470:    :toctree: generated/
471: 
472:    pbdv_seq -- [+]Parabolic cylinder functions Dv(x) and derivatives.
473:    pbvv_seq -- [+]Parabolic cylinder functions Vv(x) and derivatives.
474:    pbdn_seq -- [+]Parabolic cylinder functions Dn(z) and derivatives.
475: 
476: Mathieu and Related Functions
477: -----------------------------
478: 
479: .. autosummary::
480:    :toctree: generated/
481: 
482:    mathieu_a -- Characteristic value of even Mathieu functions
483:    mathieu_b -- Characteristic value of odd Mathieu functions
484: 
485: These are not universal functions:
486: 
487: .. autosummary::
488:    :toctree: generated/
489: 
490:    mathieu_even_coef -- [+]Fourier coefficients for even Mathieu and modified Mathieu functions.
491:    mathieu_odd_coef  -- [+]Fourier coefficients for even Mathieu and modified Mathieu functions.
492: 
493: The following return both function and first derivative:
494: 
495: .. autosummary::
496:    :toctree: generated/
497: 
498:    mathieu_cem     -- Even Mathieu function and its derivative
499:    mathieu_sem     -- Odd Mathieu function and its derivative
500:    mathieu_modcem1 -- Even modified Mathieu function of the first kind and its derivative
501:    mathieu_modcem2 -- Even modified Mathieu function of the second kind and its derivative
502:    mathieu_modsem1 -- Odd modified Mathieu function of the first kind and its derivative
503:    mathieu_modsem2 -- Odd modified Mathieu function of the second kind and its derivative
504: 
505: Spheroidal Wave Functions
506: -------------------------
507: 
508: .. autosummary::
509:    :toctree: generated/
510: 
511:    pro_ang1   -- Prolate spheroidal angular function of the first kind and its derivative
512:    pro_rad1   -- Prolate spheroidal radial function of the first kind and its derivative
513:    pro_rad2   -- Prolate spheroidal radial function of the secon kind and its derivative
514:    obl_ang1   -- Oblate spheroidal angular function of the first kind and its derivative
515:    obl_rad1   -- Oblate spheroidal radial function of the first kind and its derivative
516:    obl_rad2   -- Oblate spheroidal radial function of the second kind and its derivative.
517:    pro_cv     -- Characteristic value of prolate spheroidal function
518:    obl_cv     -- Characteristic value of oblate spheroidal function
519:    pro_cv_seq -- Characteristic values for prolate spheroidal wave functions.
520:    obl_cv_seq -- Characteristic values for oblate spheroidal wave functions.
521: 
522: The following functions require pre-computed characteristic value:
523: 
524: .. autosummary::
525:    :toctree: generated/
526: 
527:    pro_ang1_cv -- Prolate spheroidal angular function pro_ang1 for precomputed characteristic value
528:    pro_rad1_cv -- Prolate spheroidal radial function pro_rad1 for precomputed characteristic value
529:    pro_rad2_cv -- Prolate spheroidal radial function pro_rad2 for precomputed characteristic value
530:    obl_ang1_cv -- Oblate spheroidal angular function obl_ang1 for precomputed characteristic value
531:    obl_rad1_cv -- Oblate spheroidal radial function obl_rad1 for precomputed characteristic value
532:    obl_rad2_cv -- Oblate spheroidal radial function obl_rad2 for precomputed characteristic value
533: 
534: Kelvin Functions
535: ----------------
536: 
537: .. autosummary::
538:    :toctree: generated/
539: 
540:    kelvin       -- Kelvin functions as complex numbers
541:    kelvin_zeros -- [+]Compute nt zeros of all Kelvin functions.
542:    ber          -- Kelvin function ber.
543:    bei          -- Kelvin function bei
544:    berp         -- Derivative of the Kelvin function `ber`
545:    beip         -- Derivative of the Kelvin function `bei`
546:    ker          -- Kelvin function ker
547:    kei          -- Kelvin function ker
548:    kerp         -- Derivative of the Kelvin function ker
549:    keip         -- Derivative of the Kelvin function kei
550: 
551: These are not universal functions:
552: 
553: .. autosummary::
554:    :toctree: generated/
555: 
556:    ber_zeros  -- [+]Compute nt zeros of the Kelvin function ber(x).
557:    bei_zeros  -- [+]Compute nt zeros of the Kelvin function bei(x).
558:    berp_zeros -- [+]Compute nt zeros of the Kelvin function ber'(x).
559:    beip_zeros -- [+]Compute nt zeros of the Kelvin function bei'(x).
560:    ker_zeros  -- [+]Compute nt zeros of the Kelvin function ker(x).
561:    kei_zeros  -- [+]Compute nt zeros of the Kelvin function kei(x).
562:    kerp_zeros -- [+]Compute nt zeros of the Kelvin function ker'(x).
563:    keip_zeros -- [+]Compute nt zeros of the Kelvin function kei'(x).
564: 
565: Combinatorics
566: -------------
567: 
568: .. autosummary::
569:    :toctree: generated/
570: 
571:    comb -- [+]The number of combinations of N things taken k at a time.
572:    perm -- [+]Permutations of N things taken k at a time, i.e., k-permutations of N.
573: 
574: Lambert W and Related Functions
575: -------------------------------
576: 
577: .. autosummary::
578:     :toctree: generated/
579: 
580:    lambertw    -- Lambert W function.
581:    wrightomega -- Wright Omega function.
582: 
583: Other Special Functions
584: -----------------------
585: 
586: .. autosummary::
587:    :toctree: generated/
588: 
589:    agm        -- Arithmetic, Geometric Mean.
590:    bernoulli  -- Bernoulli numbers B0..Bn (inclusive).
591:    binom      -- Binomial coefficient
592:    diric      -- Periodic sinc function, also called the Dirichlet function.
593:    euler      -- Euler numbers E0..En (inclusive).
594:    expn       -- Exponential integral E_n
595:    exp1       -- Exponential integral E_1 of complex argument z
596:    expi       -- Exponential integral Ei
597:    factorial  -- The factorial of a number or array of numbers.
598:    factorial2 -- Double factorial.
599:    factorialk -- [+]Multifactorial of n of order k, n(!!...!).
600:    shichi     -- Hyperbolic sine and cosine integrals.
601:    sici       -- Sine and cosine integrals.
602:    spence     -- Spence's function, also known as the dilogarithm.
603:    zeta       -- Riemann zeta function.
604:    zetac      -- Riemann zeta function minus 1.
605: 
606: Convenience Functions
607: ---------------------
608: 
609: .. autosummary::
610:    :toctree: generated/
611: 
612:    cbrt      -- Cube root of `x`
613:    exp10     -- 10**x
614:    exp2      -- 2**x
615:    radian    -- Convert from degrees to radians
616:    cosdg     -- Cosine of the angle `x` given in degrees.
617:    sindg     -- Sine of angle given in degrees
618:    tandg     -- Tangent of angle x given in degrees.
619:    cotdg     -- Cotangent of the angle `x` given in degrees.
620:    log1p     -- Calculates log(1+x) for use when `x` is near zero
621:    expm1     -- exp(x) - 1 for use when `x` is near zero.
622:    cosm1     -- cos(x) - 1 for use when `x` is near zero.
623:    round     -- Round to nearest integer
624:    xlogy     -- Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.
625:    xlog1py   -- Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.
626:    logsumexp -- Compute the log of the sum of exponentials of input elements.
627:    exprel    -- Relative error exponential, (exp(x)-1)/x, for use when `x` is near zero.
628:    sinc      -- Return the sinc function.
629: 
630: .. [+] in the description indicates a function which is not a universal
631: .. function and does not follow broadcasting and automatic
632: .. array-looping rules.
633: 
634: '''
635: 
636: from __future__ import division, print_function, absolute_import
637: 
638: from .sf_error import SpecialFunctionWarning, SpecialFunctionError
639: 
640: from ._ufuncs import *
641: 
642: from .basic import *
643: from ._logsumexp import logsumexp
644: from . import specfun
645: from . import orthogonal
646: from .orthogonal import *
647: from .spfun_stats import multigammaln
648: from ._ellip_harm import ellip_harm, ellip_harm_2, ellip_normal
649: from .lambertw import lambertw
650: from ._spherical_bessel import (spherical_jn, spherical_yn, spherical_in,
651:                                 spherical_kn)
652: 
653: __all__ = [s for s in dir() if not s.startswith('_')]
654: 
655: from numpy.dual import register_func
656: register_func('i0',i0)
657: del register_func
658: 
659: from scipy._lib._testutils import PytestTester
660: test = PytestTester(__name__)
661: del PytestTester
662: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, (-1)), 'str', "\n========================================\nSpecial functions (:mod:`scipy.special`)\n========================================\n\n.. module:: scipy.special\n\nNearly all of the functions below are universal functions and follow\nbroadcasting and automatic array-looping rules. Exceptions are\nnoted.\n\n.. seealso::\n\n   `scipy.special.cython_special` -- Typed Cython versions of special functions\n\n\nError handling\n==============\n\nErrors are handled by returning NaNs or other appropriate values.\nSome of the special function routines can emit warnings or raise\nexceptions when an error occurs. By default this is disabled; to\nquery and control the current error handling state the following\nfunctions are provided.\n\n.. autosummary::\n   :toctree: generated/\n\n   geterr                 -- Get the current way of handling special-function errors.\n   seterr                 -- Set how special-function errors are handled.\n   errstate               -- Context manager for special-function error handling.\n   SpecialFunctionWarning -- Warning that can be emitted by special functions.\n   SpecialFunctionError   -- Exception that can be raised by special functions.\n\nAvailable functions\n===================\n\nAiry functions\n--------------\n\n.. autosummary::\n   :toctree: generated/\n\n   airy     -- Airy functions and their derivatives.\n   airye    -- Exponentially scaled Airy functions and their derivatives.\n   ai_zeros -- [+]Compute `nt` zeros and values of the Airy function Ai and its derivative.\n   bi_zeros -- [+]Compute `nt` zeros and values of the Airy function Bi and its derivative.\n   itairy   -- Integrals of Airy functions\n\n\nElliptic Functions and Integrals\n--------------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   ellipj    -- Jacobian elliptic functions\n   ellipk    -- Complete elliptic integral of the first kind.\n   ellipkm1  -- Complete elliptic integral of the first kind around `m` = 1\n   ellipkinc -- Incomplete elliptic integral of the first kind\n   ellipe    -- Complete elliptic integral of the second kind\n   ellipeinc -- Incomplete elliptic integral of the second kind\n\nBessel Functions\n----------------\n\n.. autosummary::\n   :toctree: generated/\n\n   jv       -- Bessel function of the first kind of real order and complex argument.\n   jn       -- Bessel function of the first kind of real order and complex argument\n   jve      -- Exponentially scaled Bessel function of order `v`.\n   yn       -- Bessel function of the second kind of integer order and real argument.\n   yv       -- Bessel function of the second kind of real order and complex argument.\n   yve      -- Exponentially scaled Bessel function of the second kind of real order.\n   kn       -- Modified Bessel function of the second kind of integer order `n`\n   kv       -- Modified Bessel function of the second kind of real order `v`\n   kve      -- Exponentially scaled modified Bessel function of the second kind.\n   iv       -- Modified Bessel function of the first kind of real order.\n   ive      -- Exponentially scaled modified Bessel function of the first kind\n   hankel1  -- Hankel function of the first kind\n   hankel1e -- Exponentially scaled Hankel function of the first kind\n   hankel2  -- Hankel function of the second kind\n   hankel2e -- Exponentially scaled Hankel function of the second kind\n\nThe following is not an universal function:\n\n.. autosummary::\n   :toctree: generated/\n\n   lmbda -- [+]Jahnke-Emden Lambda function, Lambdav(x).\n\nZeros of Bessel Functions\n^^^^^^^^^^^^^^^^^^^^^^^^^\n\nThese are not universal functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   jnjnp_zeros -- [+]Compute zeros of integer-order Bessel functions Jn and Jn'.\n   jnyn_zeros  -- [+]Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).\n   jn_zeros    -- [+]Compute zeros of integer-order Bessel function Jn(x).\n   jnp_zeros   -- [+]Compute zeros of integer-order Bessel function derivative Jn'(x).\n   yn_zeros    -- [+]Compute zeros of integer-order Bessel function Yn(x).\n   ynp_zeros   -- [+]Compute zeros of integer-order Bessel function derivative Yn'(x).\n   y0_zeros    -- [+]Compute nt zeros of Bessel function Y0(z), and derivative at each zero.\n   y1_zeros    -- [+]Compute nt zeros of Bessel function Y1(z), and derivative at each zero.\n   y1p_zeros   -- [+]Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.\n\nFaster versions of common Bessel Functions\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n.. autosummary::\n   :toctree: generated/\n\n   j0  -- Bessel function of the first kind of order 0.\n   j1  -- Bessel function of the first kind of order 1.\n   y0  -- Bessel function of the second kind of order 0.\n   y1  -- Bessel function of the second kind of order 1.\n   i0  -- Modified Bessel function of order 0.\n   i0e -- Exponentially scaled modified Bessel function of order 0.\n   i1  -- Modified Bessel function of order 1.\n   i1e -- Exponentially scaled modified Bessel function of order 1.\n   k0  -- Modified Bessel function of the second kind of order 0, :math:`K_0`.\n   k0e -- Exponentially scaled modified Bessel function K of order 0\n   k1  -- Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.\n   k1e -- Exponentially scaled modified Bessel function K of order 1\n\nIntegrals of Bessel Functions\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n.. autosummary::\n   :toctree: generated/\n\n   itj0y0     -- Integrals of Bessel functions of order 0\n   it2j0y0    -- Integrals related to Bessel functions of order 0\n   iti0k0     -- Integrals of modified Bessel functions of order 0\n   it2i0k0    -- Integrals related to modified Bessel functions of order 0\n   besselpoly -- [+]Weighted integral of a Bessel function.\n\nDerivatives of Bessel Functions\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n.. autosummary::\n   :toctree: generated/\n\n   jvp  -- Compute nth derivative of Bessel function Jv(z) with respect to `z`.\n   yvp  -- Compute nth derivative of Bessel function Yv(z) with respect to `z`.\n   kvp  -- Compute nth derivative of real-order modified Bessel function Kv(z)\n   ivp  -- Compute nth derivative of modified Bessel function Iv(z) with respect to `z`.\n   h1vp -- Compute nth derivative of Hankel function H1v(z) with respect to `z`.\n   h2vp -- Compute nth derivative of Hankel function H2v(z) with respect to `z`.\n\nSpherical Bessel Functions\n^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n.. autosummary::\n   :toctree: generated/\n\n   spherical_jn -- Spherical Bessel function of the first kind or its derivative.\n   spherical_yn -- Spherical Bessel function of the second kind or its derivative.\n   spherical_in -- Modified spherical Bessel function of the first kind or its derivative.\n   spherical_kn -- Modified spherical Bessel function of the second kind or its derivative.\n\nRiccati-Bessel Functions\n^^^^^^^^^^^^^^^^^^^^^^^^\n\nThese are not universal functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   riccati_jn -- [+]Compute Ricatti-Bessel function of the first kind and its derivative.\n   riccati_yn -- [+]Compute Ricatti-Bessel function of the second kind and its derivative.\n\nStruve Functions\n----------------\n\n.. autosummary::\n   :toctree: generated/\n\n   struve       -- Struve function.\n   modstruve    -- Modified Struve function.\n   itstruve0    -- Integral of the Struve function of order 0.\n   it2struve0   -- Integral related to the Struve function of order 0.\n   itmodstruve0 -- Integral of the modified Struve function of order 0.\n\n\nRaw Statistical Functions\n-------------------------\n\n.. seealso:: :mod:`scipy.stats`: Friendly versions of these functions.\n\n.. autosummary::\n   :toctree: generated/\n\n   bdtr         -- Binomial distribution cumulative distribution function.\n   bdtrc        -- Binomial distribution survival function.\n   bdtri        -- Inverse function to `bdtr` with respect to `p`.\n   bdtrik       -- Inverse function to `bdtr` with respect to `k`.\n   bdtrin       -- Inverse function to `bdtr` with respect to `n`.\n   btdtr        -- Cumulative density function of the beta distribution.\n   btdtri       -- The `p`-th quantile of the beta distribution.\n   btdtria      -- Inverse of `btdtr` with respect to `a`.\n   btdtrib      -- btdtria(a, p, x)\n   fdtr         -- F cumulative distribution function.\n   fdtrc        -- F survival function.\n   fdtri        -- The `p`-th quantile of the F-distribution.\n   fdtridfd     -- Inverse to `fdtr` vs dfd\n   gdtr         -- Gamma distribution cumulative density function.\n   gdtrc        -- Gamma distribution survival function.\n   gdtria       -- Inverse of `gdtr` vs a.\n   gdtrib       -- Inverse of `gdtr` vs b.\n   gdtrix       -- Inverse of `gdtr` vs x.\n   nbdtr        -- Negative binomial cumulative distribution function.\n   nbdtrc       -- Negative binomial survival function.\n   nbdtri       -- Inverse of `nbdtr` vs `p`.\n   nbdtrik      -- Inverse of `nbdtr` vs `k`.\n   nbdtrin      -- Inverse of `nbdtr` vs `n`.\n   ncfdtr       -- Cumulative distribution function of the non-central F distribution.\n   ncfdtridfd   -- Calculate degrees of freedom (denominator) for the noncentral F-distribution.\n   ncfdtridfn   -- Calculate degrees of freedom (numerator) for the noncentral F-distribution.\n   ncfdtri      -- Inverse cumulative distribution function of the non-central F distribution.\n   ncfdtrinc    -- Calculate non-centrality parameter for non-central F distribution.\n   nctdtr       -- Cumulative distribution function of the non-central `t` distribution.\n   nctdtridf    -- Calculate degrees of freedom for non-central t distribution.\n   nctdtrit     -- Inverse cumulative distribution function of the non-central t distribution.\n   nctdtrinc    -- Calculate non-centrality parameter for non-central t distribution.\n   nrdtrimn     -- Calculate mean of normal distribution given other params.\n   nrdtrisd     -- Calculate standard deviation of normal distribution given other params.\n   pdtr         -- Poisson cumulative distribution function\n   pdtrc        -- Poisson survival function\n   pdtri        -- Inverse to `pdtr` vs m\n   pdtrik       -- Inverse to `pdtr` vs k\n   stdtr        -- Student t distribution cumulative density function\n   stdtridf     -- Inverse of `stdtr` vs df\n   stdtrit      -- Inverse of `stdtr` vs `t`\n   chdtr        -- Chi square cumulative distribution function\n   chdtrc       -- Chi square survival function\n   chdtri       -- Inverse to `chdtrc`\n   chdtriv      -- Inverse to `chdtr` vs `v`\n   ndtr         -- Gaussian cumulative distribution function.\n   log_ndtr     -- Logarithm of Gaussian cumulative distribution function.\n   ndtri        -- Inverse of `ndtr` vs x\n   chndtr       -- Non-central chi square cumulative distribution function\n   chndtridf    -- Inverse to `chndtr` vs `df`\n   chndtrinc    -- Inverse to `chndtr` vs `nc`\n   chndtrix     -- Inverse to `chndtr` vs `x`\n   smirnov      -- Kolmogorov-Smirnov complementary cumulative distribution function\n   smirnovi     -- Inverse to `smirnov`\n   kolmogorov   -- Complementary cumulative distribution function of Kolmogorov distribution\n   kolmogi      -- Inverse function to kolmogorov\n   tklmbda      -- Tukey-Lambda cumulative distribution function\n   logit        -- Logit ufunc for ndarrays.\n   expit        -- Expit ufunc for ndarrays.\n   boxcox       -- Compute the Box-Cox transformation.\n   boxcox1p     -- Compute the Box-Cox transformation of 1 + `x`.\n   inv_boxcox   -- Compute the inverse of the Box-Cox transformation.\n   inv_boxcox1p -- Compute the inverse of the Box-Cox transformation.\n\n\nInformation Theory Functions\n----------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   entr         -- Elementwise function for computing entropy.\n   rel_entr     -- Elementwise function for computing relative entropy.\n   kl_div       -- Elementwise function for computing Kullback-Leibler divergence.\n   huber        -- Huber loss function.\n   pseudo_huber -- Pseudo-Huber loss function.\n\n\nGamma and Related Functions\n---------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   gamma        -- Gamma function.\n   gammaln      -- Logarithm of the absolute value of the Gamma function for real inputs.\n   loggamma     -- Principal branch of the logarithm of the Gamma function.\n   gammasgn     -- Sign of the gamma function.\n   gammainc     -- Regularized lower incomplete gamma function.\n   gammaincinv  -- Inverse to `gammainc`\n   gammaincc    -- Regularized upper incomplete gamma function.\n   gammainccinv -- Inverse to `gammaincc`\n   beta         -- Beta function.\n   betaln       -- Natural logarithm of absolute value of beta function.\n   betainc      -- Incomplete beta integral.\n   betaincinv   -- Inverse function to beta integral.\n   psi          -- The digamma function.\n   rgamma       -- Gamma function inverted\n   polygamma    -- Polygamma function n.\n   multigammaln -- Returns the log of multivariate gamma, also sometimes called the generalized gamma.\n   digamma      -- psi(x[, out])\n   poch         -- Rising factorial (z)_m\n\n\nError Function and Fresnel Integrals\n------------------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   erf           -- Returns the error function of complex argument.\n   erfc          -- Complementary error function, ``1 - erf(x)``.\n   erfcx         -- Scaled complementary error function, ``exp(x**2) * erfc(x)``.\n   erfi          -- Imaginary error function, ``-i erf(i z)``.\n   erfinv        -- Inverse function for erf.\n   erfcinv       -- Inverse function for erfc.\n   wofz          -- Faddeeva function\n   dawsn         -- Dawson's integral.\n   fresnel       -- Fresnel sin and cos integrals\n   fresnel_zeros -- Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).\n   modfresnelp   -- Modified Fresnel positive integrals\n   modfresnelm   -- Modified Fresnel negative integrals\n\nThese are not universal functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   erf_zeros      -- [+]Compute nt complex zeros of error function erf(z).\n   fresnelc_zeros -- [+]Compute nt complex zeros of cosine Fresnel integral C(z).\n   fresnels_zeros -- [+]Compute nt complex zeros of sine Fresnel integral S(z).\n\nLegendre Functions\n------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   lpmv     -- Associated Legendre function of integer order and real degree.\n   sph_harm -- Compute spherical harmonics.\n\nThese are not universal functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   clpmn -- [+]Associated Legendre function of the first kind for complex arguments.\n   lpn   -- [+]Legendre function of the first kind.\n   lqn   -- [+]Legendre function of the second kind.\n   lpmn  -- [+]Sequence of associated Legendre functions of the first kind.\n   lqmn  -- [+]Sequence of associated Legendre functions of the second kind.\n\nEllipsoidal Harmonics\n---------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   ellip_harm   -- Ellipsoidal harmonic functions E^p_n(l)\n   ellip_harm_2 -- Ellipsoidal harmonic functions F^p_n(l)\n   ellip_normal -- Ellipsoidal harmonic normalization constants gamma^p_n\n\nOrthogonal polynomials\n----------------------\n\nThe following functions evaluate values of orthogonal polynomials:\n\n.. autosummary::\n   :toctree: generated/\n\n   assoc_laguerre   -- Compute the generalized (associated) Laguerre polynomial of degree n and order k.\n   eval_legendre    -- Evaluate Legendre polynomial at a point.\n   eval_chebyt      -- Evaluate Chebyshev polynomial of the first kind at a point.\n   eval_chebyu      -- Evaluate Chebyshev polynomial of the second kind at a point.\n   eval_chebyc      -- Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a point.\n   eval_chebys      -- Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a point.\n   eval_jacobi      -- Evaluate Jacobi polynomial at a point.\n   eval_laguerre    -- Evaluate Laguerre polynomial at a point.\n   eval_genlaguerre -- Evaluate generalized Laguerre polynomial at a point.\n   eval_hermite     -- Evaluate physicist's Hermite polynomial at a point.\n   eval_hermitenorm -- Evaluate probabilist's (normalized) Hermite polynomial at a point.\n   eval_gegenbauer  -- Evaluate Gegenbauer polynomial at a point.\n   eval_sh_legendre -- Evaluate shifted Legendre polynomial at a point.\n   eval_sh_chebyt   -- Evaluate shifted Chebyshev polynomial of the first kind at a point.\n   eval_sh_chebyu   -- Evaluate shifted Chebyshev polynomial of the second kind at a point.\n   eval_sh_jacobi   -- Evaluate shifted Jacobi polynomial at a point.\n\nThe following functions compute roots and quadrature weights for\northogonal polynomials:\n\n.. autosummary::\n   :toctree: generated/\n\n   roots_legendre    -- Gauss-Legendre quadrature.\n   roots_chebyt      -- Gauss-Chebyshev (first kind) quadrature.\n   roots_chebyu      -- Gauss-Chebyshev (second kind) quadrature.\n   roots_chebyc      -- Gauss-Chebyshev (first kind) quadrature.\n   roots_chebys      -- Gauss-Chebyshev (second kind) quadrature.\n   roots_jacobi      -- Gauss-Jacobi quadrature.\n   roots_laguerre    -- Gauss-Laguerre quadrature.\n   roots_genlaguerre -- Gauss-generalized Laguerre quadrature.\n   roots_hermite     -- Gauss-Hermite (physicst's) quadrature.\n   roots_hermitenorm -- Gauss-Hermite (statistician's) quadrature.\n   roots_gegenbauer  -- Gauss-Gegenbauer quadrature.\n   roots_sh_legendre -- Gauss-Legendre (shifted) quadrature.\n   roots_sh_chebyt   -- Gauss-Chebyshev (first kind, shifted) quadrature.\n   roots_sh_chebyu   -- Gauss-Chebyshev (second kind, shifted) quadrature.\n   roots_sh_jacobi   -- Gauss-Jacobi (shifted) quadrature.\n\nThe functions below, in turn, return the polynomial coefficients in\n:class:`~.orthopoly1d` objects, which function similarly as :ref:`numpy.poly1d`.\nThe :class:`~.orthopoly1d` class also has an attribute ``weights`` which returns\nthe roots, weights, and total weights for the appropriate form of Gaussian\nquadrature.  These are returned in an ``n x 3`` array with roots in the first\ncolumn, weights in the second column, and total weights in the final column.\nNote that :class:`~.orthopoly1d` objects are converted to ``poly1d`` when doing\narithmetic, and lose information of the original orthogonal polynomial.\n\n.. autosummary::\n   :toctree: generated/\n\n   legendre    -- [+]Legendre polynomial.\n   chebyt      -- [+]Chebyshev polynomial of the first kind.\n   chebyu      -- [+]Chebyshev polynomial of the second kind.\n   chebyc      -- [+]Chebyshev polynomial of the first kind on :math:`[-2, 2]`.\n   chebys      -- [+]Chebyshev polynomial of the second kind on :math:`[-2, 2]`.\n   jacobi      -- [+]Jacobi polynomial.\n   laguerre    -- [+]Laguerre polynomial.\n   genlaguerre -- [+]Generalized (associated) Laguerre polynomial.\n   hermite     -- [+]Physicist's Hermite polynomial.\n   hermitenorm -- [+]Normalized (probabilist's) Hermite polynomial.\n   gegenbauer  -- [+]Gegenbauer (ultraspherical) polynomial.\n   sh_legendre -- [+]Shifted Legendre polynomial.\n   sh_chebyt   -- [+]Shifted Chebyshev polynomial of the first kind.\n   sh_chebyu   -- [+]Shifted Chebyshev polynomial of the second kind.\n   sh_jacobi   -- [+]Shifted Jacobi polynomial.\n\n.. warning::\n\n   Computing values of high-order polynomials (around ``order > 20``) using\n   polynomial coefficients is numerically unstable. To evaluate polynomial\n   values, the ``eval_*`` functions should be used instead.\n\n\nHypergeometric Functions\n------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   hyp2f1 -- Gauss hypergeometric function 2F1(a, b; c; z).\n   hyp1f1 -- Confluent hypergeometric function 1F1(a, b; x)\n   hyperu -- Confluent hypergeometric function U(a, b, x) of the second kind\n   hyp0f1 -- Confluent hypergeometric limit function 0F1.\n   hyp2f0 -- Hypergeometric function 2F0 in y and an error estimate\n   hyp1f2 -- Hypergeometric function 1F2 and error estimate\n   hyp3f0 -- Hypergeometric function 3F0 in y and an error estimate\n\n\nParabolic Cylinder Functions\n----------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   pbdv -- Parabolic cylinder function D\n   pbvv -- Parabolic cylinder function V\n   pbwa -- Parabolic cylinder function W\n\nThese are not universal functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   pbdv_seq -- [+]Parabolic cylinder functions Dv(x) and derivatives.\n   pbvv_seq -- [+]Parabolic cylinder functions Vv(x) and derivatives.\n   pbdn_seq -- [+]Parabolic cylinder functions Dn(z) and derivatives.\n\nMathieu and Related Functions\n-----------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   mathieu_a -- Characteristic value of even Mathieu functions\n   mathieu_b -- Characteristic value of odd Mathieu functions\n\nThese are not universal functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   mathieu_even_coef -- [+]Fourier coefficients for even Mathieu and modified Mathieu functions.\n   mathieu_odd_coef  -- [+]Fourier coefficients for even Mathieu and modified Mathieu functions.\n\nThe following return both function and first derivative:\n\n.. autosummary::\n   :toctree: generated/\n\n   mathieu_cem     -- Even Mathieu function and its derivative\n   mathieu_sem     -- Odd Mathieu function and its derivative\n   mathieu_modcem1 -- Even modified Mathieu function of the first kind and its derivative\n   mathieu_modcem2 -- Even modified Mathieu function of the second kind and its derivative\n   mathieu_modsem1 -- Odd modified Mathieu function of the first kind and its derivative\n   mathieu_modsem2 -- Odd modified Mathieu function of the second kind and its derivative\n\nSpheroidal Wave Functions\n-------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   pro_ang1   -- Prolate spheroidal angular function of the first kind and its derivative\n   pro_rad1   -- Prolate spheroidal radial function of the first kind and its derivative\n   pro_rad2   -- Prolate spheroidal radial function of the secon kind and its derivative\n   obl_ang1   -- Oblate spheroidal angular function of the first kind and its derivative\n   obl_rad1   -- Oblate spheroidal radial function of the first kind and its derivative\n   obl_rad2   -- Oblate spheroidal radial function of the second kind and its derivative.\n   pro_cv     -- Characteristic value of prolate spheroidal function\n   obl_cv     -- Characteristic value of oblate spheroidal function\n   pro_cv_seq -- Characteristic values for prolate spheroidal wave functions.\n   obl_cv_seq -- Characteristic values for oblate spheroidal wave functions.\n\nThe following functions require pre-computed characteristic value:\n\n.. autosummary::\n   :toctree: generated/\n\n   pro_ang1_cv -- Prolate spheroidal angular function pro_ang1 for precomputed characteristic value\n   pro_rad1_cv -- Prolate spheroidal radial function pro_rad1 for precomputed characteristic value\n   pro_rad2_cv -- Prolate spheroidal radial function pro_rad2 for precomputed characteristic value\n   obl_ang1_cv -- Oblate spheroidal angular function obl_ang1 for precomputed characteristic value\n   obl_rad1_cv -- Oblate spheroidal radial function obl_rad1 for precomputed characteristic value\n   obl_rad2_cv -- Oblate spheroidal radial function obl_rad2 for precomputed characteristic value\n\nKelvin Functions\n----------------\n\n.. autosummary::\n   :toctree: generated/\n\n   kelvin       -- Kelvin functions as complex numbers\n   kelvin_zeros -- [+]Compute nt zeros of all Kelvin functions.\n   ber          -- Kelvin function ber.\n   bei          -- Kelvin function bei\n   berp         -- Derivative of the Kelvin function `ber`\n   beip         -- Derivative of the Kelvin function `bei`\n   ker          -- Kelvin function ker\n   kei          -- Kelvin function ker\n   kerp         -- Derivative of the Kelvin function ker\n   keip         -- Derivative of the Kelvin function kei\n\nThese are not universal functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   ber_zeros  -- [+]Compute nt zeros of the Kelvin function ber(x).\n   bei_zeros  -- [+]Compute nt zeros of the Kelvin function bei(x).\n   berp_zeros -- [+]Compute nt zeros of the Kelvin function ber'(x).\n   beip_zeros -- [+]Compute nt zeros of the Kelvin function bei'(x).\n   ker_zeros  -- [+]Compute nt zeros of the Kelvin function ker(x).\n   kei_zeros  -- [+]Compute nt zeros of the Kelvin function kei(x).\n   kerp_zeros -- [+]Compute nt zeros of the Kelvin function ker'(x).\n   keip_zeros -- [+]Compute nt zeros of the Kelvin function kei'(x).\n\nCombinatorics\n-------------\n\n.. autosummary::\n   :toctree: generated/\n\n   comb -- [+]The number of combinations of N things taken k at a time.\n   perm -- [+]Permutations of N things taken k at a time, i.e., k-permutations of N.\n\nLambert W and Related Functions\n-------------------------------\n\n.. autosummary::\n    :toctree: generated/\n\n   lambertw    -- Lambert W function.\n   wrightomega -- Wright Omega function.\n\nOther Special Functions\n-----------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   agm        -- Arithmetic, Geometric Mean.\n   bernoulli  -- Bernoulli numbers B0..Bn (inclusive).\n   binom      -- Binomial coefficient\n   diric      -- Periodic sinc function, also called the Dirichlet function.\n   euler      -- Euler numbers E0..En (inclusive).\n   expn       -- Exponential integral E_n\n   exp1       -- Exponential integral E_1 of complex argument z\n   expi       -- Exponential integral Ei\n   factorial  -- The factorial of a number or array of numbers.\n   factorial2 -- Double factorial.\n   factorialk -- [+]Multifactorial of n of order k, n(!!...!).\n   shichi     -- Hyperbolic sine and cosine integrals.\n   sici       -- Sine and cosine integrals.\n   spence     -- Spence's function, also known as the dilogarithm.\n   zeta       -- Riemann zeta function.\n   zetac      -- Riemann zeta function minus 1.\n\nConvenience Functions\n---------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   cbrt      -- Cube root of `x`\n   exp10     -- 10**x\n   exp2      -- 2**x\n   radian    -- Convert from degrees to radians\n   cosdg     -- Cosine of the angle `x` given in degrees.\n   sindg     -- Sine of angle given in degrees\n   tandg     -- Tangent of angle x given in degrees.\n   cotdg     -- Cotangent of the angle `x` given in degrees.\n   log1p     -- Calculates log(1+x) for use when `x` is near zero\n   expm1     -- exp(x) - 1 for use when `x` is near zero.\n   cosm1     -- cos(x) - 1 for use when `x` is near zero.\n   round     -- Round to nearest integer\n   xlogy     -- Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.\n   xlog1py   -- Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.\n   logsumexp -- Compute the log of the sum of exponentials of input elements.\n   exprel    -- Relative error exponential, (exp(x)-1)/x, for use when `x` is near zero.\n   sinc      -- Return the sinc function.\n\n.. [+] in the description indicates a function which is not a universal\n.. function and does not follow broadcasting and automatic\n.. array-looping rules.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 638, 0))

# 'from scipy.special.sf_error import SpecialFunctionWarning, SpecialFunctionError' statement (line 638)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_2 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 638, 0), 'scipy.special.sf_error')

if (type(import_2) is not StypyTypeError):

    if (import_2 != 'pyd_module'):
        __import__(import_2)
        sys_modules_3 = sys.modules[import_2]
        import_from_module(stypy.reporting.localization.Localization(__file__, 638, 0), 'scipy.special.sf_error', sys_modules_3.module_type_store, module_type_store, ['SpecialFunctionWarning', 'SpecialFunctionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 638, 0), __file__, sys_modules_3, sys_modules_3.module_type_store, module_type_store)
    else:
        from scipy.special.sf_error import SpecialFunctionWarning, SpecialFunctionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 638, 0), 'scipy.special.sf_error', None, module_type_store, ['SpecialFunctionWarning', 'SpecialFunctionError'], [SpecialFunctionWarning, SpecialFunctionError])

else:
    # Assigning a type to the variable 'scipy.special.sf_error' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 0), 'scipy.special.sf_error', import_2)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 640, 0))

# 'from scipy.special._ufuncs import ' statement (line 640)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_4 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 640, 0), 'scipy.special._ufuncs')

if (type(import_4) is not StypyTypeError):

    if (import_4 != 'pyd_module'):
        __import__(import_4)
        sys_modules_5 = sys.modules[import_4]
        import_from_module(stypy.reporting.localization.Localization(__file__, 640, 0), 'scipy.special._ufuncs', sys_modules_5.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 640, 0), __file__, sys_modules_5, sys_modules_5.module_type_store, module_type_store)
    else:
        from scipy.special._ufuncs import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 640, 0), 'scipy.special._ufuncs', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.special._ufuncs' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), 'scipy.special._ufuncs', import_4)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 642, 0))

# 'from scipy.special.basic import ' statement (line 642)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_6 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 642, 0), 'scipy.special.basic')

if (type(import_6) is not StypyTypeError):

    if (import_6 != 'pyd_module'):
        __import__(import_6)
        sys_modules_7 = sys.modules[import_6]
        import_from_module(stypy.reporting.localization.Localization(__file__, 642, 0), 'scipy.special.basic', sys_modules_7.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 642, 0), __file__, sys_modules_7, sys_modules_7.module_type_store, module_type_store)
    else:
        from scipy.special.basic import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 642, 0), 'scipy.special.basic', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.special.basic' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 0), 'scipy.special.basic', import_6)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 643, 0))

# 'from scipy.special._logsumexp import logsumexp' statement (line 643)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_8 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 643, 0), 'scipy.special._logsumexp')

if (type(import_8) is not StypyTypeError):

    if (import_8 != 'pyd_module'):
        __import__(import_8)
        sys_modules_9 = sys.modules[import_8]
        import_from_module(stypy.reporting.localization.Localization(__file__, 643, 0), 'scipy.special._logsumexp', sys_modules_9.module_type_store, module_type_store, ['logsumexp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 643, 0), __file__, sys_modules_9, sys_modules_9.module_type_store, module_type_store)
    else:
        from scipy.special._logsumexp import logsumexp

        import_from_module(stypy.reporting.localization.Localization(__file__, 643, 0), 'scipy.special._logsumexp', None, module_type_store, ['logsumexp'], [logsumexp])

else:
    # Assigning a type to the variable 'scipy.special._logsumexp' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 0), 'scipy.special._logsumexp', import_8)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 644, 0))

# 'from scipy.special import specfun' statement (line 644)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_10 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 644, 0), 'scipy.special')

if (type(import_10) is not StypyTypeError):

    if (import_10 != 'pyd_module'):
        __import__(import_10)
        sys_modules_11 = sys.modules[import_10]
        import_from_module(stypy.reporting.localization.Localization(__file__, 644, 0), 'scipy.special', sys_modules_11.module_type_store, module_type_store, ['specfun'])
        nest_module(stypy.reporting.localization.Localization(__file__, 644, 0), __file__, sys_modules_11, sys_modules_11.module_type_store, module_type_store)
    else:
        from scipy.special import specfun

        import_from_module(stypy.reporting.localization.Localization(__file__, 644, 0), 'scipy.special', None, module_type_store, ['specfun'], [specfun])

else:
    # Assigning a type to the variable 'scipy.special' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 0), 'scipy.special', import_10)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 645, 0))

# 'from scipy.special import orthogonal' statement (line 645)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_12 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 645, 0), 'scipy.special')

if (type(import_12) is not StypyTypeError):

    if (import_12 != 'pyd_module'):
        __import__(import_12)
        sys_modules_13 = sys.modules[import_12]
        import_from_module(stypy.reporting.localization.Localization(__file__, 645, 0), 'scipy.special', sys_modules_13.module_type_store, module_type_store, ['orthogonal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 645, 0), __file__, sys_modules_13, sys_modules_13.module_type_store, module_type_store)
    else:
        from scipy.special import orthogonal

        import_from_module(stypy.reporting.localization.Localization(__file__, 645, 0), 'scipy.special', None, module_type_store, ['orthogonal'], [orthogonal])

else:
    # Assigning a type to the variable 'scipy.special' (line 645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 0), 'scipy.special', import_12)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 646, 0))

# 'from scipy.special.orthogonal import ' statement (line 646)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_14 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 646, 0), 'scipy.special.orthogonal')

if (type(import_14) is not StypyTypeError):

    if (import_14 != 'pyd_module'):
        __import__(import_14)
        sys_modules_15 = sys.modules[import_14]
        import_from_module(stypy.reporting.localization.Localization(__file__, 646, 0), 'scipy.special.orthogonal', sys_modules_15.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 646, 0), __file__, sys_modules_15, sys_modules_15.module_type_store, module_type_store)
    else:
        from scipy.special.orthogonal import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 646, 0), 'scipy.special.orthogonal', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.special.orthogonal' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 0), 'scipy.special.orthogonal', import_14)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 647, 0))

# 'from scipy.special.spfun_stats import multigammaln' statement (line 647)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_16 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 647, 0), 'scipy.special.spfun_stats')

if (type(import_16) is not StypyTypeError):

    if (import_16 != 'pyd_module'):
        __import__(import_16)
        sys_modules_17 = sys.modules[import_16]
        import_from_module(stypy.reporting.localization.Localization(__file__, 647, 0), 'scipy.special.spfun_stats', sys_modules_17.module_type_store, module_type_store, ['multigammaln'])
        nest_module(stypy.reporting.localization.Localization(__file__, 647, 0), __file__, sys_modules_17, sys_modules_17.module_type_store, module_type_store)
    else:
        from scipy.special.spfun_stats import multigammaln

        import_from_module(stypy.reporting.localization.Localization(__file__, 647, 0), 'scipy.special.spfun_stats', None, module_type_store, ['multigammaln'], [multigammaln])

else:
    # Assigning a type to the variable 'scipy.special.spfun_stats' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 0), 'scipy.special.spfun_stats', import_16)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 648, 0))

# 'from scipy.special._ellip_harm import ellip_harm, ellip_harm_2, ellip_normal' statement (line 648)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_18 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 648, 0), 'scipy.special._ellip_harm')

if (type(import_18) is not StypyTypeError):

    if (import_18 != 'pyd_module'):
        __import__(import_18)
        sys_modules_19 = sys.modules[import_18]
        import_from_module(stypy.reporting.localization.Localization(__file__, 648, 0), 'scipy.special._ellip_harm', sys_modules_19.module_type_store, module_type_store, ['ellip_harm', 'ellip_harm_2', 'ellip_normal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 648, 0), __file__, sys_modules_19, sys_modules_19.module_type_store, module_type_store)
    else:
        from scipy.special._ellip_harm import ellip_harm, ellip_harm_2, ellip_normal

        import_from_module(stypy.reporting.localization.Localization(__file__, 648, 0), 'scipy.special._ellip_harm', None, module_type_store, ['ellip_harm', 'ellip_harm_2', 'ellip_normal'], [ellip_harm, ellip_harm_2, ellip_normal])

else:
    # Assigning a type to the variable 'scipy.special._ellip_harm' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 0), 'scipy.special._ellip_harm', import_18)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 649, 0))

# 'from scipy.special.lambertw import lambertw' statement (line 649)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_20 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 649, 0), 'scipy.special.lambertw')

if (type(import_20) is not StypyTypeError):

    if (import_20 != 'pyd_module'):
        __import__(import_20)
        sys_modules_21 = sys.modules[import_20]
        import_from_module(stypy.reporting.localization.Localization(__file__, 649, 0), 'scipy.special.lambertw', sys_modules_21.module_type_store, module_type_store, ['lambertw'])
        nest_module(stypy.reporting.localization.Localization(__file__, 649, 0), __file__, sys_modules_21, sys_modules_21.module_type_store, module_type_store)
    else:
        from scipy.special.lambertw import lambertw

        import_from_module(stypy.reporting.localization.Localization(__file__, 649, 0), 'scipy.special.lambertw', None, module_type_store, ['lambertw'], [lambertw])

else:
    # Assigning a type to the variable 'scipy.special.lambertw' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 0), 'scipy.special.lambertw', import_20)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 650, 0))

# 'from scipy.special._spherical_bessel import spherical_jn, spherical_yn, spherical_in, spherical_kn' statement (line 650)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_22 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 650, 0), 'scipy.special._spherical_bessel')

if (type(import_22) is not StypyTypeError):

    if (import_22 != 'pyd_module'):
        __import__(import_22)
        sys_modules_23 = sys.modules[import_22]
        import_from_module(stypy.reporting.localization.Localization(__file__, 650, 0), 'scipy.special._spherical_bessel', sys_modules_23.module_type_store, module_type_store, ['spherical_jn', 'spherical_yn', 'spherical_in', 'spherical_kn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 650, 0), __file__, sys_modules_23, sys_modules_23.module_type_store, module_type_store)
    else:
        from scipy.special._spherical_bessel import spherical_jn, spherical_yn, spherical_in, spherical_kn

        import_from_module(stypy.reporting.localization.Localization(__file__, 650, 0), 'scipy.special._spherical_bessel', None, module_type_store, ['spherical_jn', 'spherical_yn', 'spherical_in', 'spherical_kn'], [spherical_jn, spherical_yn, spherical_in, spherical_kn])

else:
    # Assigning a type to the variable 'scipy.special._spherical_bessel' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 0), 'scipy.special._spherical_bessel', import_22)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


# Assigning a ListComp to a Name (line 653):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 653)
# Processing the call keyword arguments (line 653)
kwargs_32 = {}
# Getting the type of 'dir' (line 653)
dir_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 22), 'dir', False)
# Calling dir(args, kwargs) (line 653)
dir_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 653, 22), dir_31, *[], **kwargs_32)

comprehension_34 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 11), dir_call_result_33)
# Assigning a type to the variable 's' (line 653)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 11), 's', comprehension_34)


# Call to startswith(...): (line 653)
# Processing the call arguments (line 653)
str_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 48), 'str', '_')
# Processing the call keyword arguments (line 653)
kwargs_28 = {}
# Getting the type of 's' (line 653)
s_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 653)
startswith_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 35), s_25, 'startswith')
# Calling startswith(args, kwargs) (line 653)
startswith_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 653, 35), startswith_26, *[str_27], **kwargs_28)

# Applying the 'not' unary operator (line 653)
result_not__30 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 31), 'not', startswith_call_result_29)

# Getting the type of 's' (line 653)
s_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 11), 's')
list_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 11), list_35, s_24)
# Assigning a type to the variable '__all__' (line 653)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 0), '__all__', list_35)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 655, 0))

# 'from numpy.dual import register_func' statement (line 655)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_36 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 655, 0), 'numpy.dual')

if (type(import_36) is not StypyTypeError):

    if (import_36 != 'pyd_module'):
        __import__(import_36)
        sys_modules_37 = sys.modules[import_36]
        import_from_module(stypy.reporting.localization.Localization(__file__, 655, 0), 'numpy.dual', sys_modules_37.module_type_store, module_type_store, ['register_func'])
        nest_module(stypy.reporting.localization.Localization(__file__, 655, 0), __file__, sys_modules_37, sys_modules_37.module_type_store, module_type_store)
    else:
        from numpy.dual import register_func

        import_from_module(stypy.reporting.localization.Localization(__file__, 655, 0), 'numpy.dual', None, module_type_store, ['register_func'], [register_func])

else:
    # Assigning a type to the variable 'numpy.dual' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 0), 'numpy.dual', import_36)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


# Call to register_func(...): (line 656)
# Processing the call arguments (line 656)
str_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 14), 'str', 'i0')
# Getting the type of 'i0' (line 656)
i0_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 19), 'i0', False)
# Processing the call keyword arguments (line 656)
kwargs_41 = {}
# Getting the type of 'register_func' (line 656)
register_func_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 0), 'register_func', False)
# Calling register_func(args, kwargs) (line 656)
register_func_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 656, 0), register_func_38, *[str_39, i0_40], **kwargs_41)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 657, 0), module_type_store, 'register_func')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 659, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 659)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_43 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 659, 0), 'scipy._lib._testutils')

if (type(import_43) is not StypyTypeError):

    if (import_43 != 'pyd_module'):
        __import__(import_43)
        sys_modules_44 = sys.modules[import_43]
        import_from_module(stypy.reporting.localization.Localization(__file__, 659, 0), 'scipy._lib._testutils', sys_modules_44.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 659, 0), __file__, sys_modules_44, sys_modules_44.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 659, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 0), 'scipy._lib._testutils', import_43)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


# Assigning a Call to a Name (line 660):

# Call to PytestTester(...): (line 660)
# Processing the call arguments (line 660)
# Getting the type of '__name__' (line 660)
name___46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 20), '__name__', False)
# Processing the call keyword arguments (line 660)
kwargs_47 = {}
# Getting the type of 'PytestTester' (line 660)
PytestTester_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 660)
PytestTester_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 660, 7), PytestTester_45, *[name___46], **kwargs_47)

# Assigning a type to the variable 'test' (line 660)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 0), 'test', PytestTester_call_result_48)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 661, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
