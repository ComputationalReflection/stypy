
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Some simple financial calculations
2: 
3: patterned after spreadsheet computations.
4: 
5: There is some complexity in each function
6: so that the functions behave like ufuncs with
7: broadcasting and being able to be called with scalars
8: or arrays (or other sequences).
9: 
10: '''
11: from __future__ import division, absolute_import, print_function
12: 
13: import numpy as np
14: 
15: __all__ = ['fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate',
16:            'irr', 'npv', 'mirr']
17: 
18: _when_to_num = {'end':0, 'begin':1,
19:                 'e':0, 'b':1,
20:                 0:0, 1:1,
21:                 'beginning':1,
22:                 'start':1,
23:                 'finish':0}
24: 
25: def _convert_when(when):
26:     #Test to see if when has already been converted to ndarray
27:     #This will happen if one function calls another, for example ppmt
28:     if isinstance(when, np.ndarray):
29:         return when
30:     try:
31:         return _when_to_num[when]
32:     except (KeyError, TypeError):
33:         return [_when_to_num[x] for x in when]
34: 
35: 
36: def fv(rate, nper, pmt, pv, when='end'):
37:     '''
38:     Compute the future value.
39: 
40:     Given:
41:      * a present value, `pv`
42:      * an interest `rate` compounded once per period, of which
43:        there are
44:      * `nper` total
45:      * a (fixed) payment, `pmt`, paid either
46:      * at the beginning (`when` = {'begin', 1}) or the end
47:        (`when` = {'end', 0}) of each period
48: 
49:     Return:
50:        the value at the end of the `nper` periods
51: 
52:     Parameters
53:     ----------
54:     rate : scalar or array_like of shape(M, )
55:         Rate of interest as decimal (not per cent) per period
56:     nper : scalar or array_like of shape(M, )
57:         Number of compounding periods
58:     pmt : scalar or array_like of shape(M, )
59:         Payment
60:     pv : scalar or array_like of shape(M, )
61:         Present value
62:     when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
63:         When payments are due ('begin' (1) or 'end' (0)).
64:         Defaults to {'end', 0}.
65: 
66:     Returns
67:     -------
68:     out : ndarray
69:         Future values.  If all input is scalar, returns a scalar float.  If
70:         any input is array_like, returns future values for each input element.
71:         If multiple inputs are array_like, they all must have the same shape.
72: 
73:     Notes
74:     -----
75:     The future value is computed by solving the equation::
76: 
77:      fv +
78:      pv*(1+rate)**nper +
79:      pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0
80: 
81:     or, when ``rate == 0``::
82: 
83:      fv + pv + pmt * nper == 0
84: 
85:     References
86:     ----------
87:     .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
88:        Open Document Format for Office Applications (OpenDocument)v1.2,
89:        Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
90:        Pre-Draft 12. Organization for the Advancement of Structured Information
91:        Standards (OASIS). Billerica, MA, USA. [ODT Document].
92:        Available:
93:        http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
94:        OpenDocument-formula-20090508.odt
95: 
96:     Examples
97:     --------
98:     What is the future value after 10 years of saving $100 now, with
99:     an additional monthly savings of $100.  Assume the interest rate is
100:     5% (annually) compounded monthly?
101: 
102:     >>> np.fv(0.05/12, 10*12, -100, -100)
103:     15692.928894335748
104: 
105:     By convention, the negative sign represents cash flow out (i.e. money not
106:     available today).  Thus, saving $100 a month at 5% annual interest leads
107:     to $15,692.93 available to spend in 10 years.
108: 
109:     If any input is array_like, returns an array of equal shape.  Let's
110:     compare different interest rates from the example above.
111: 
112:     >>> a = np.array((0.05, 0.06, 0.07))/12
113:     >>> np.fv(a, 10*12, -100, -100)
114:     array([ 15692.92889434,  16569.87435405,  17509.44688102])
115: 
116:     '''
117:     when = _convert_when(when)
118:     (rate, nper, pmt, pv, when) = map(np.asarray, [rate, nper, pmt, pv, when])
119:     temp = (1+rate)**nper
120:     miter = np.broadcast(rate, nper, pmt, pv, when)
121:     zer = np.zeros(miter.shape)
122:     fact = np.where(rate == zer, nper + zer,
123:                     (1 + rate*when)*(temp - 1)/rate + zer)
124:     return -(pv*temp + pmt*fact)
125: 
126: def pmt(rate, nper, pv, fv=0, when='end'):
127:     '''
128:     Compute the payment against loan principal plus interest.
129: 
130:     Given:
131:      * a present value, `pv` (e.g., an amount borrowed)
132:      * a future value, `fv` (e.g., 0)
133:      * an interest `rate` compounded once per period, of which
134:        there are
135:      * `nper` total
136:      * and (optional) specification of whether payment is made
137:        at the beginning (`when` = {'begin', 1}) or the end
138:        (`when` = {'end', 0}) of each period
139: 
140:     Return:
141:        the (fixed) periodic payment.
142: 
143:     Parameters
144:     ----------
145:     rate : array_like
146:         Rate of interest (per period)
147:     nper : array_like
148:         Number of compounding periods
149:     pv : array_like
150:         Present value
151:     fv : array_like,  optional
152:         Future value (default = 0)
153:     when : {{'begin', 1}, {'end', 0}}, {string, int}
154:         When payments are due ('begin' (1) or 'end' (0))
155: 
156:     Returns
157:     -------
158:     out : ndarray
159:         Payment against loan plus interest.  If all input is scalar, returns a
160:         scalar float.  If any input is array_like, returns payment for each
161:         input element. If multiple inputs are array_like, they all must have
162:         the same shape.
163: 
164:     Notes
165:     -----
166:     The payment is computed by solving the equation::
167: 
168:      fv +
169:      pv*(1 + rate)**nper +
170:      pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0
171: 
172:     or, when ``rate == 0``::
173: 
174:       fv + pv + pmt * nper == 0
175: 
176:     for ``pmt``.
177: 
178:     Note that computing a monthly mortgage payment is only
179:     one use for this function.  For example, pmt returns the
180:     periodic deposit one must make to achieve a specified
181:     future balance given an initial deposit, a fixed,
182:     periodically compounded interest rate, and the total
183:     number of periods.
184: 
185:     References
186:     ----------
187:     .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
188:        Open Document Format for Office Applications (OpenDocument)v1.2,
189:        Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
190:        Pre-Draft 12. Organization for the Advancement of Structured Information
191:        Standards (OASIS). Billerica, MA, USA. [ODT Document].
192:        Available:
193:        http://www.oasis-open.org/committees/documents.php
194:        ?wg_abbrev=office-formulaOpenDocument-formula-20090508.odt
195: 
196:     Examples
197:     --------
198:     What is the monthly payment needed to pay off a $200,000 loan in 15
199:     years at an annual interest rate of 7.5%?
200: 
201:     >>> np.pmt(0.075/12, 12*15, 200000)
202:     -1854.0247200054619
203: 
204:     In order to pay-off (i.e., have a future-value of 0) the $200,000 obtained
205:     today, a monthly payment of $1,854.02 would be required.  Note that this
206:     example illustrates usage of `fv` having a default value of 0.
207: 
208:     '''
209:     when = _convert_when(when)
210:     (rate, nper, pv, fv, when) = map(np.asarray, [rate, nper, pv, fv, when])
211:     temp = (1 + rate)**nper
212:     mask = (rate == 0.0)
213:     np.copyto(rate, 1.0, where=mask)
214:     z = np.zeros(np.broadcast(rate, nper, pv, fv, when).shape)
215:     fact = np.where(mask != z, nper + z, (1 + rate*when)*(temp - 1)/rate + z)
216:     return -(fv + pv*temp) / fact
217: 
218: def nper(rate, pmt, pv, fv=0, when='end'):
219:     '''
220:     Compute the number of periodic payments.
221: 
222:     Parameters
223:     ----------
224:     rate : array_like
225:         Rate of interest (per period)
226:     pmt : array_like
227:         Payment
228:     pv : array_like
229:         Present value
230:     fv : array_like, optional
231:         Future value
232:     when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
233:         When payments are due ('begin' (1) or 'end' (0))
234: 
235:     Notes
236:     -----
237:     The number of periods ``nper`` is computed by solving the equation::
238: 
239:      fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate*((1+rate)**nper-1) = 0
240: 
241:     but if ``rate = 0`` then::
242: 
243:      fv + pv + pmt*nper = 0
244: 
245:     Examples
246:     --------
247:     If you only had $150/month to pay towards the loan, how long would it take
248:     to pay-off a loan of $8,000 at 7% annual interest?
249: 
250:     >>> print(round(np.nper(0.07/12, -150, 8000), 5))
251:     64.07335
252: 
253:     So, over 64 months would be required to pay off the loan.
254: 
255:     The same analysis could be done with several different interest rates
256:     and/or payments and/or total amounts to produce an entire table.
257: 
258:     >>> np.nper(*(np.ogrid[0.07/12: 0.08/12: 0.01/12,
259:     ...                    -150   : -99     : 50    ,
260:     ...                    8000   : 9001    : 1000]))
261:     array([[[  64.07334877,   74.06368256],
262:             [ 108.07548412,  127.99022654]],
263:            [[  66.12443902,   76.87897353],
264:             [ 114.70165583,  137.90124779]]])
265: 
266:     '''
267:     when = _convert_when(when)
268:     (rate, pmt, pv, fv, when) = map(np.asarray, [rate, pmt, pv, fv, when])
269: 
270:     use_zero_rate = False
271:     with np.errstate(divide="raise"):
272:         try:
273:             z = pmt*(1.0+rate*when)/rate
274:         except FloatingPointError:
275:             use_zero_rate = True
276: 
277:     if use_zero_rate:
278:         return (-fv + pv) / (pmt + 0.0)
279:     else:
280:         A = -(fv + pv)/(pmt+0.0)
281:         B = np.log((-fv+z) / (pv+z))/np.log(1.0+rate)
282:         miter = np.broadcast(rate, pmt, pv, fv, when)
283:         zer = np.zeros(miter.shape)
284:         return np.where(rate == zer, A + zer, B + zer) + 0.0
285: 
286: def ipmt(rate, per, nper, pv, fv=0.0, when='end'):
287:     '''
288:     Compute the interest portion of a payment.
289: 
290:     Parameters
291:     ----------
292:     rate : scalar or array_like of shape(M, )
293:         Rate of interest as decimal (not per cent) per period
294:     per : scalar or array_like of shape(M, )
295:         Interest paid against the loan changes during the life or the loan.
296:         The `per` is the payment period to calculate the interest amount.
297:     nper : scalar or array_like of shape(M, )
298:         Number of compounding periods
299:     pv : scalar or array_like of shape(M, )
300:         Present value
301:     fv : scalar or array_like of shape(M, ), optional
302:         Future value
303:     when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
304:         When payments are due ('begin' (1) or 'end' (0)).
305:         Defaults to {'end', 0}.
306: 
307:     Returns
308:     -------
309:     out : ndarray
310:         Interest portion of payment.  If all input is scalar, returns a scalar
311:         float.  If any input is array_like, returns interest payment for each
312:         input element. If multiple inputs are array_like, they all must have
313:         the same shape.
314: 
315:     See Also
316:     --------
317:     ppmt, pmt, pv
318: 
319:     Notes
320:     -----
321:     The total payment is made up of payment against principal plus interest.
322: 
323:     ``pmt = ppmt + ipmt``
324: 
325:     Examples
326:     --------
327:     What is the amortization schedule for a 1 year loan of $2500 at
328:     8.24% interest per year compounded monthly?
329: 
330:     >>> principal = 2500.00
331: 
332:     The 'per' variable represents the periods of the loan.  Remember that
333:     financial equations start the period count at 1!
334: 
335:     >>> per = np.arange(1*12) + 1
336:     >>> ipmt = np.ipmt(0.0824/12, per, 1*12, principal)
337:     >>> ppmt = np.ppmt(0.0824/12, per, 1*12, principal)
338: 
339:     Each element of the sum of the 'ipmt' and 'ppmt' arrays should equal
340:     'pmt'.
341: 
342:     >>> pmt = np.pmt(0.0824/12, 1*12, principal)
343:     >>> np.allclose(ipmt + ppmt, pmt)
344:     True
345: 
346:     >>> fmt = '{0:2d} {1:8.2f} {2:8.2f} {3:8.2f}'
347:     >>> for payment in per:
348:     ...     index = payment - 1
349:     ...     principal = principal + ppmt[index]
350:     ...     print(fmt.format(payment, ppmt[index], ipmt[index], principal))
351:      1  -200.58   -17.17  2299.42
352:      2  -201.96   -15.79  2097.46
353:      3  -203.35   -14.40  1894.11
354:      4  -204.74   -13.01  1689.37
355:      5  -206.15   -11.60  1483.22
356:      6  -207.56   -10.18  1275.66
357:      7  -208.99    -8.76  1066.67
358:      8  -210.42    -7.32   856.25
359:      9  -211.87    -5.88   644.38
360:     10  -213.32    -4.42   431.05
361:     11  -214.79    -2.96   216.26
362:     12  -216.26    -1.49    -0.00
363: 
364:     >>> interestpd = np.sum(ipmt)
365:     >>> np.round(interestpd, 2)
366:     -112.98
367: 
368:     '''
369:     when = _convert_when(when)
370:     rate, per, nper, pv, fv, when = np.broadcast_arrays(rate, per, nper,
371:                                                         pv, fv, when)
372:     total_pmt = pmt(rate, nper, pv, fv, when)
373:     ipmt = _rbl(rate, per, total_pmt, pv, when)*rate
374:     try:
375:         ipmt = np.where(when == 1, ipmt/(1 + rate), ipmt)
376:         ipmt = np.where(np.logical_and(when == 1, per == 1), 0.0, ipmt)
377:     except IndexError:
378:         pass
379:     return ipmt
380: 
381: def _rbl(rate, per, pmt, pv, when):
382:     '''
383:     This function is here to simply have a different name for the 'fv'
384:     function to not interfere with the 'fv' keyword argument within the 'ipmt'
385:     function.  It is the 'remaining balance on loan' which might be useful as
386:     it's own function, but is easily calculated with the 'fv' function.
387:     '''
388:     return fv(rate, (per - 1), pmt, pv, when)
389: 
390: def ppmt(rate, per, nper, pv, fv=0.0, when='end'):
391:     '''
392:     Compute the payment against loan principal.
393: 
394:     Parameters
395:     ----------
396:     rate : array_like
397:         Rate of interest (per period)
398:     per : array_like, int
399:         Amount paid against the loan changes.  The `per` is the period of
400:         interest.
401:     nper : array_like
402:         Number of compounding periods
403:     pv : array_like
404:         Present value
405:     fv : array_like, optional
406:         Future value
407:     when : {{'begin', 1}, {'end', 0}}, {string, int}
408:         When payments are due ('begin' (1) or 'end' (0))
409: 
410:     See Also
411:     --------
412:     pmt, pv, ipmt
413: 
414:     '''
415:     total = pmt(rate, nper, pv, fv, when)
416:     return total - ipmt(rate, per, nper, pv, fv, when)
417: 
418: def pv(rate, nper, pmt, fv=0.0, when='end'):
419:     '''
420:     Compute the present value.
421: 
422:     Given:
423:      * a future value, `fv`
424:      * an interest `rate` compounded once per period, of which
425:        there are
426:      * `nper` total
427:      * a (fixed) payment, `pmt`, paid either
428:      * at the beginning (`when` = {'begin', 1}) or the end
429:        (`when` = {'end', 0}) of each period
430: 
431:     Return:
432:        the value now
433: 
434:     Parameters
435:     ----------
436:     rate : array_like
437:         Rate of interest (per period)
438:     nper : array_like
439:         Number of compounding periods
440:     pmt : array_like
441:         Payment
442:     fv : array_like, optional
443:         Future value
444:     when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
445:         When payments are due ('begin' (1) or 'end' (0))
446: 
447:     Returns
448:     -------
449:     out : ndarray, float
450:         Present value of a series of payments or investments.
451: 
452:     Notes
453:     -----
454:     The present value is computed by solving the equation::
455: 
456:      fv +
457:      pv*(1 + rate)**nper +
458:      pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) = 0
459: 
460:     or, when ``rate = 0``::
461: 
462:      fv + pv + pmt * nper = 0
463: 
464:     for `pv`, which is then returned.
465: 
466:     References
467:     ----------
468:     .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
469:        Open Document Format for Office Applications (OpenDocument)v1.2,
470:        Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
471:        Pre-Draft 12. Organization for the Advancement of Structured Information
472:        Standards (OASIS). Billerica, MA, USA. [ODT Document].
473:        Available:
474:        http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
475:        OpenDocument-formula-20090508.odt
476: 
477:     Examples
478:     --------
479:     What is the present value (e.g., the initial investment)
480:     of an investment that needs to total $15692.93
481:     after 10 years of saving $100 every month?  Assume the
482:     interest rate is 5% (annually) compounded monthly.
483: 
484:     >>> np.pv(0.05/12, 10*12, -100, 15692.93)
485:     -100.00067131625819
486: 
487:     By convention, the negative sign represents cash flow out
488:     (i.e., money not available today).  Thus, to end up with
489:     $15,692.93 in 10 years saving $100 a month at 5% annual
490:     interest, one's initial deposit should also be $100.
491: 
492:     If any input is array_like, ``pv`` returns an array of equal shape.
493:     Let's compare different interest rates in the example above:
494: 
495:     >>> a = np.array((0.05, 0.04, 0.03))/12
496:     >>> np.pv(a, 10*12, -100, 15692.93)
497:     array([ -100.00067132,  -649.26771385, -1273.78633713])
498: 
499:     So, to end up with the same $15692.93 under the same $100 per month
500:     "savings plan," for annual interest rates of 4% and 3%, one would
501:     need initial investments of $649.27 and $1273.79, respectively.
502: 
503:     '''
504:     when = _convert_when(when)
505:     (rate, nper, pmt, fv, when) = map(np.asarray, [rate, nper, pmt, fv, when])
506:     temp = (1+rate)**nper
507:     miter = np.broadcast(rate, nper, pmt, fv, when)
508:     zer = np.zeros(miter.shape)
509:     fact = np.where(rate == zer, nper+zer, (1+rate*when)*(temp-1)/rate+zer)
510:     return -(fv + pmt*fact)/temp
511: 
512: # Computed with Sage
513: #  (y + (r + 1)^n*x + p*((r + 1)^n - 1)*(r*w + 1)/r)/(n*(r + 1)^(n - 1)*x -
514: #  p*((r + 1)^n - 1)*(r*w + 1)/r^2 + n*p*(r + 1)^(n - 1)*(r*w + 1)/r +
515: #  p*((r + 1)^n - 1)*w/r)
516: 
517: def _g_div_gp(r, n, p, x, y, w):
518:     t1 = (r+1)**n
519:     t2 = (r+1)**(n-1)
520:     return ((y + t1*x + p*(t1 - 1)*(r*w + 1)/r) /
521:                 (n*t2*x - p*(t1 - 1)*(r*w + 1)/(r**2) + n*p*t2*(r*w + 1)/r +
522:                  p*(t1 - 1)*w/r))
523: 
524: # Use Newton's iteration until the change is less than 1e-6
525: #  for all values or a maximum of 100 iterations is reached.
526: #  Newton's rule is
527: #  r_{n+1} = r_{n} - g(r_n)/g'(r_n)
528: #     where
529: #  g(r) is the formula
530: #  g'(r) is the derivative with respect to r.
531: def rate(nper, pmt, pv, fv, when='end', guess=0.10, tol=1e-6, maxiter=100):
532:     '''
533:     Compute the rate of interest per period.
534: 
535:     Parameters
536:     ----------
537:     nper : array_like
538:         Number of compounding periods
539:     pmt : array_like
540:         Payment
541:     pv : array_like
542:         Present value
543:     fv : array_like
544:         Future value
545:     when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
546:         When payments are due ('begin' (1) or 'end' (0))
547:     guess : float, optional
548:         Starting guess for solving the rate of interest
549:     tol : float, optional
550:         Required tolerance for the solution
551:     maxiter : int, optional
552:         Maximum iterations in finding the solution
553: 
554:     Notes
555:     -----
556:     The rate of interest is computed by iteratively solving the
557:     (non-linear) equation::
558: 
559:      fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) = 0
560: 
561:     for ``rate``.
562: 
563:     References
564:     ----------
565:     Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May). Open Document
566:     Format for Office Applications (OpenDocument)v1.2, Part 2: Recalculated
567:     Formula (OpenFormula) Format - Annotated Version, Pre-Draft 12.
568:     Organization for the Advancement of Structured Information Standards
569:     (OASIS). Billerica, MA, USA. [ODT Document]. Available:
570:     http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
571:     OpenDocument-formula-20090508.odt
572: 
573:     '''
574:     when = _convert_when(when)
575:     (nper, pmt, pv, fv, when) = map(np.asarray, [nper, pmt, pv, fv, when])
576:     rn = guess
577:     iter = 0
578:     close = False
579:     while (iter < maxiter) and not close:
580:         rnp1 = rn - _g_div_gp(rn, nper, pmt, pv, fv, when)
581:         diff = abs(rnp1-rn)
582:         close = np.all(diff < tol)
583:         iter += 1
584:         rn = rnp1
585:     if not close:
586:         # Return nan's in array of the same shape as rn
587:         return np.nan + rn
588:     else:
589:         return rn
590: 
591: def irr(values):
592:     '''
593:     Return the Internal Rate of Return (IRR).
594: 
595:     This is the "average" periodically compounded rate of return
596:     that gives a net present value of 0.0; for a more complete explanation,
597:     see Notes below.
598: 
599:     Parameters
600:     ----------
601:     values : array_like, shape(N,)
602:         Input cash flows per time period.  By convention, net "deposits"
603:         are negative and net "withdrawals" are positive.  Thus, for
604:         example, at least the first element of `values`, which represents
605:         the initial investment, will typically be negative.
606: 
607:     Returns
608:     -------
609:     out : float
610:         Internal Rate of Return for periodic input values.
611: 
612:     Notes
613:     -----
614:     The IRR is perhaps best understood through an example (illustrated
615:     using np.irr in the Examples section below).  Suppose one invests 100
616:     units and then makes the following withdrawals at regular (fixed)
617:     intervals: 39, 59, 55, 20.  Assuming the ending value is 0, one's 100
618:     unit investment yields 173 units; however, due to the combination of
619:     compounding and the periodic withdrawals, the "average" rate of return
620:     is neither simply 0.73/4 nor (1.73)^0.25-1.  Rather, it is the solution
621:     (for :math:`r`) of the equation:
622: 
623:     .. math:: -100 + \\frac{39}{1+r} + \\frac{59}{(1+r)^2}
624:      + \\frac{55}{(1+r)^3} + \\frac{20}{(1+r)^4} = 0
625: 
626:     In general, for `values` :math:`= [v_0, v_1, ... v_M]`,
627:     irr is the solution of the equation: [G]_
628: 
629:     .. math:: \\sum_{t=0}^M{\\frac{v_t}{(1+irr)^{t}}} = 0
630: 
631:     References
632:     ----------
633:     .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,
634:        Addison-Wesley, 2003, pg. 348.
635: 
636:     Examples
637:     --------
638:     >>> round(irr([-100, 39, 59, 55, 20]), 5)
639:     0.28095
640:     >>> round(irr([-100, 0, 0, 74]), 5)
641:     -0.0955
642:     >>> round(irr([-100, 100, 0, -7]), 5)
643:     -0.0833
644:     >>> round(irr([-100, 100, 0, 7]), 5)
645:     0.06206
646:     >>> round(irr([-5, 10.5, 1, -8, 1]), 5)
647:     0.0886
648: 
649:     (Compare with the Example given for numpy.lib.financial.npv)
650: 
651:     '''
652:     res = np.roots(values[::-1])
653:     mask = (res.imag == 0) & (res.real > 0)
654:     if res.size == 0:
655:         return np.nan
656:     res = res[mask].real
657:     # NPV(rate) = 0 can have more than one solution so we return
658:     # only the solution closest to zero.
659:     rate = 1.0/res - 1
660:     rate = rate.item(np.argmin(np.abs(rate)))
661:     return rate
662: 
663: def npv(rate, values):
664:     '''
665:     Returns the NPV (Net Present Value) of a cash flow series.
666: 
667:     Parameters
668:     ----------
669:     rate : scalar
670:         The discount rate.
671:     values : array_like, shape(M, )
672:         The values of the time series of cash flows.  The (fixed) time
673:         interval between cash flow "events" must be the same as that for
674:         which `rate` is given (i.e., if `rate` is per year, then precisely
675:         a year is understood to elapse between each cash flow event).  By
676:         convention, investments or "deposits" are negative, income or
677:         "withdrawals" are positive; `values` must begin with the initial
678:         investment, thus `values[0]` will typically be negative.
679: 
680:     Returns
681:     -------
682:     out : float
683:         The NPV of the input cash flow series `values` at the discount
684:         `rate`.
685: 
686:     Notes
687:     -----
688:     Returns the result of: [G]_
689: 
690:     .. math :: \\sum_{t=0}^{M-1}{\\frac{values_t}{(1+rate)^{t}}}
691: 
692:     References
693:     ----------
694:     .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,
695:        Addison-Wesley, 2003, pg. 346.
696: 
697:     Examples
698:     --------
699:     >>> np.npv(0.281,[-100, 39, 59, 55, 20])
700:     -0.0084785916384548798
701: 
702:     (Compare with the Example given for numpy.lib.financial.irr)
703: 
704:     '''
705:     values = np.asarray(values)
706:     return (values / (1+rate)**np.arange(0, len(values))).sum(axis=0)
707: 
708: def mirr(values, finance_rate, reinvest_rate):
709:     '''
710:     Modified internal rate of return.
711: 
712:     Parameters
713:     ----------
714:     values : array_like
715:         Cash flows (must contain at least one positive and one negative
716:         value) or nan is returned.  The first value is considered a sunk
717:         cost at time zero.
718:     finance_rate : scalar
719:         Interest rate paid on the cash flows
720:     reinvest_rate : scalar
721:         Interest rate received on the cash flows upon reinvestment
722: 
723:     Returns
724:     -------
725:     out : float
726:         Modified internal rate of return
727: 
728:     '''
729:     values = np.asarray(values, dtype=np.double)
730:     n = values.size
731:     pos = values > 0
732:     neg = values < 0
733:     if not (pos.any() and neg.any()):
734:         return np.nan
735:     numer = np.abs(npv(reinvest_rate, values*pos))
736:     denom = np.abs(npv(finance_rate, values*neg))
737:     return (numer/denom)**(1.0/(n - 1))*(1 + reinvest_rate) - 1
738: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_105227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', 'Some simple financial calculations\n\npatterned after spreadsheet computations.\n\nThere is some complexity in each function\nso that the functions behave like ufuncs with\nbroadcasting and being able to be called with scalars\nor arrays (or other sequences).\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_105228 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_105228) is not StypyTypeError):

    if (import_105228 != 'pyd_module'):
        __import__(import_105228)
        sys_modules_105229 = sys.modules[import_105228]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_105229.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_105228)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):
__all__ = ['fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate', 'irr', 'npv', 'mirr']
module_type_store.set_exportable_members(['fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate', 'irr', 'npv', 'mirr'])

# Obtaining an instance of the builtin type 'list' (line 15)
list_105230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_105231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'fv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105231)
# Adding element type (line 15)
str_105232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'str', 'pmt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105232)
# Adding element type (line 15)
str_105233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'str', 'nper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105233)
# Adding element type (line 15)
str_105234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', 'ipmt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105234)
# Adding element type (line 15)
str_105235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 40), 'str', 'ppmt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105235)
# Adding element type (line 15)
str_105236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'str', 'pv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105236)
# Adding element type (line 15)
str_105237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 54), 'str', 'rate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105237)
# Adding element type (line 15)
str_105238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'irr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105238)
# Adding element type (line 15)
str_105239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'str', 'npv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105239)
# Adding element type (line 15)
str_105240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', 'mirr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_105230, str_105240)

# Assigning a type to the variable '__all__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__all__', list_105230)

# Assigning a Dict to a Name (line 18):

# Assigning a Dict to a Name (line 18):

# Obtaining an instance of the builtin type 'dict' (line 18)
dict_105241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 18)
# Adding element type (key, value) (line 18)
str_105242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'str', 'end')
int_105243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (str_105242, int_105243))
# Adding element type (key, value) (line 18)
str_105244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', 'begin')
int_105245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (str_105244, int_105245))
# Adding element type (key, value) (line 18)
str_105246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 16), 'str', 'e')
int_105247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (str_105246, int_105247))
# Adding element type (key, value) (line 18)
str_105248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', 'b')
int_105249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (str_105248, int_105249))
# Adding element type (key, value) (line 18)
int_105250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'int')
int_105251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (int_105250, int_105251))
# Adding element type (key, value) (line 18)
int_105252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'int')
int_105253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (int_105252, int_105253))
# Adding element type (key, value) (line 18)
str_105254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'str', 'beginning')
int_105255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (str_105254, int_105255))
# Adding element type (key, value) (line 18)
str_105256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'str', 'start')
int_105257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (str_105256, int_105257))
# Adding element type (key, value) (line 18)
str_105258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'str', 'finish')
int_105259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 15), dict_105241, (str_105258, int_105259))

# Assigning a type to the variable '_when_to_num' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '_when_to_num', dict_105241)

@norecursion
def _convert_when(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_convert_when'
    module_type_store = module_type_store.open_function_context('_convert_when', 25, 0, False)
    
    # Passed parameters checking function
    _convert_when.stypy_localization = localization
    _convert_when.stypy_type_of_self = None
    _convert_when.stypy_type_store = module_type_store
    _convert_when.stypy_function_name = '_convert_when'
    _convert_when.stypy_param_names_list = ['when']
    _convert_when.stypy_varargs_param_name = None
    _convert_when.stypy_kwargs_param_name = None
    _convert_when.stypy_call_defaults = defaults
    _convert_when.stypy_call_varargs = varargs
    _convert_when.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_convert_when', ['when'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_convert_when', localization, ['when'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_convert_when(...)' code ##################

    
    
    # Call to isinstance(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'when' (line 28)
    when_105261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'when', False)
    # Getting the type of 'np' (line 28)
    np_105262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 28)
    ndarray_105263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), np_105262, 'ndarray')
    # Processing the call keyword arguments (line 28)
    kwargs_105264 = {}
    # Getting the type of 'isinstance' (line 28)
    isinstance_105260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 28)
    isinstance_call_result_105265 = invoke(stypy.reporting.localization.Localization(__file__, 28, 7), isinstance_105260, *[when_105261, ndarray_105263], **kwargs_105264)
    
    # Testing the type of an if condition (line 28)
    if_condition_105266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 4), isinstance_call_result_105265)
    # Assigning a type to the variable 'if_condition_105266' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'if_condition_105266', if_condition_105266)
    # SSA begins for if statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'when' (line 29)
    when_105267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'when')
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', when_105267)
    # SSA join for if statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    # Getting the type of 'when' (line 31)
    when_105268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'when')
    # Getting the type of '_when_to_num' (line 31)
    _when_to_num_105269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), '_when_to_num')
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___105270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), _when_to_num_105269, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_105271 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), getitem___105270, when_105268)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', subscript_call_result_105271)
    # SSA branch for the except part of a try statement (line 30)
    # SSA branch for the except 'Tuple' branch of a try statement (line 30)
    module_type_store.open_ssa_branch('except')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'when' (line 33)
    when_105276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'when')
    comprehension_105277 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 16), when_105276)
    # Assigning a type to the variable 'x' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'x', comprehension_105277)
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 33)
    x_105272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'x')
    # Getting the type of '_when_to_num' (line 33)
    _when_to_num_105273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), '_when_to_num')
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___105274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), _when_to_num_105273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_105275 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), getitem___105274, x_105272)
    
    list_105278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 16), list_105278, subscript_call_result_105275)
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', list_105278)
    # SSA join for try-except statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_convert_when(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_convert_when' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_105279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105279)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_convert_when'
    return stypy_return_type_105279

# Assigning a type to the variable '_convert_when' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_convert_when', _convert_when)

@norecursion
def fv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_105280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 33), 'str', 'end')
    defaults = [str_105280]
    # Create a new context for function 'fv'
    module_type_store = module_type_store.open_function_context('fv', 36, 0, False)
    
    # Passed parameters checking function
    fv.stypy_localization = localization
    fv.stypy_type_of_self = None
    fv.stypy_type_store = module_type_store
    fv.stypy_function_name = 'fv'
    fv.stypy_param_names_list = ['rate', 'nper', 'pmt', 'pv', 'when']
    fv.stypy_varargs_param_name = None
    fv.stypy_kwargs_param_name = None
    fv.stypy_call_defaults = defaults
    fv.stypy_call_varargs = varargs
    fv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fv', ['rate', 'nper', 'pmt', 'pv', 'when'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fv', localization, ['rate', 'nper', 'pmt', 'pv', 'when'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fv(...)' code ##################

    str_105281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, (-1)), 'str', "\n    Compute the future value.\n\n    Given:\n     * a present value, `pv`\n     * an interest `rate` compounded once per period, of which\n       there are\n     * `nper` total\n     * a (fixed) payment, `pmt`, paid either\n     * at the beginning (`when` = {'begin', 1}) or the end\n       (`when` = {'end', 0}) of each period\n\n    Return:\n       the value at the end of the `nper` periods\n\n    Parameters\n    ----------\n    rate : scalar or array_like of shape(M, )\n        Rate of interest as decimal (not per cent) per period\n    nper : scalar or array_like of shape(M, )\n        Number of compounding periods\n    pmt : scalar or array_like of shape(M, )\n        Payment\n    pv : scalar or array_like of shape(M, )\n        Present value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional\n        When payments are due ('begin' (1) or 'end' (0)).\n        Defaults to {'end', 0}.\n\n    Returns\n    -------\n    out : ndarray\n        Future values.  If all input is scalar, returns a scalar float.  If\n        any input is array_like, returns future values for each input element.\n        If multiple inputs are array_like, they all must have the same shape.\n\n    Notes\n    -----\n    The future value is computed by solving the equation::\n\n     fv +\n     pv*(1+rate)**nper +\n     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0\n\n    or, when ``rate == 0``::\n\n     fv + pv + pmt * nper == 0\n\n    References\n    ----------\n    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).\n       Open Document Format for Office Applications (OpenDocument)v1.2,\n       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,\n       Pre-Draft 12. Organization for the Advancement of Structured Information\n       Standards (OASIS). Billerica, MA, USA. [ODT Document].\n       Available:\n       http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula\n       OpenDocument-formula-20090508.odt\n\n    Examples\n    --------\n    What is the future value after 10 years of saving $100 now, with\n    an additional monthly savings of $100.  Assume the interest rate is\n    5% (annually) compounded monthly?\n\n    >>> np.fv(0.05/12, 10*12, -100, -100)\n    15692.928894335748\n\n    By convention, the negative sign represents cash flow out (i.e. money not\n    available today).  Thus, saving $100 a month at 5% annual interest leads\n    to $15,692.93 available to spend in 10 years.\n\n    If any input is array_like, returns an array of equal shape.  Let's\n    compare different interest rates from the example above.\n\n    >>> a = np.array((0.05, 0.06, 0.07))/12\n    >>> np.fv(a, 10*12, -100, -100)\n    array([ 15692.92889434,  16569.87435405,  17509.44688102])\n\n    ")
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to _convert_when(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'when' (line 117)
    when_105283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'when', False)
    # Processing the call keyword arguments (line 117)
    kwargs_105284 = {}
    # Getting the type of '_convert_when' (line 117)
    _convert_when_105282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), '_convert_when', False)
    # Calling _convert_when(args, kwargs) (line 117)
    _convert_when_call_result_105285 = invoke(stypy.reporting.localization.Localization(__file__, 117, 11), _convert_when_105282, *[when_105283], **kwargs_105284)
    
    # Assigning a type to the variable 'when' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'when', _convert_when_call_result_105285)
    
    # Assigning a Call to a Tuple (line 118):
    
    # Assigning a Call to a Name:
    
    # Call to map(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'np' (line 118)
    np_105287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'np', False)
    # Obtaining the member 'asarray' of a type (line 118)
    asarray_105288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 38), np_105287, 'asarray')
    
    # Obtaining an instance of the builtin type 'list' (line 118)
    list_105289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 118)
    # Adding element type (line 118)
    # Getting the type of 'rate' (line 118)
    rate_105290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 51), 'rate', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 50), list_105289, rate_105290)
    # Adding element type (line 118)
    # Getting the type of 'nper' (line 118)
    nper_105291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 57), 'nper', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 50), list_105289, nper_105291)
    # Adding element type (line 118)
    # Getting the type of 'pmt' (line 118)
    pmt_105292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 63), 'pmt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 50), list_105289, pmt_105292)
    # Adding element type (line 118)
    # Getting the type of 'pv' (line 118)
    pv_105293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 68), 'pv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 50), list_105289, pv_105293)
    # Adding element type (line 118)
    # Getting the type of 'when' (line 118)
    when_105294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 72), 'when', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 50), list_105289, when_105294)
    
    # Processing the call keyword arguments (line 118)
    kwargs_105295 = {}
    # Getting the type of 'map' (line 118)
    map_105286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'map', False)
    # Calling map(args, kwargs) (line 118)
    map_call_result_105296 = invoke(stypy.reporting.localization.Localization(__file__, 118, 34), map_105286, *[asarray_105288, list_105289], **kwargs_105295)
    
    # Assigning a type to the variable 'call_assignment_105190' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105190', map_call_result_105296)
    
    # Assigning a Call to a Name (line 118):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105300 = {}
    # Getting the type of 'call_assignment_105190' (line 118)
    call_assignment_105190_105297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105190', False)
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___105298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), call_assignment_105190_105297, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105301 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105298, *[int_105299], **kwargs_105300)
    
    # Assigning a type to the variable 'call_assignment_105191' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105191', getitem___call_result_105301)
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'call_assignment_105191' (line 118)
    call_assignment_105191_105302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105191')
    # Assigning a type to the variable 'rate' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 5), 'rate', call_assignment_105191_105302)
    
    # Assigning a Call to a Name (line 118):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105306 = {}
    # Getting the type of 'call_assignment_105190' (line 118)
    call_assignment_105190_105303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105190', False)
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___105304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), call_assignment_105190_105303, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105307 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105304, *[int_105305], **kwargs_105306)
    
    # Assigning a type to the variable 'call_assignment_105192' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105192', getitem___call_result_105307)
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'call_assignment_105192' (line 118)
    call_assignment_105192_105308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105192')
    # Assigning a type to the variable 'nper' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'nper', call_assignment_105192_105308)
    
    # Assigning a Call to a Name (line 118):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105312 = {}
    # Getting the type of 'call_assignment_105190' (line 118)
    call_assignment_105190_105309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105190', False)
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___105310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), call_assignment_105190_105309, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105313 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105310, *[int_105311], **kwargs_105312)
    
    # Assigning a type to the variable 'call_assignment_105193' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105193', getitem___call_result_105313)
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'call_assignment_105193' (line 118)
    call_assignment_105193_105314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105193')
    # Assigning a type to the variable 'pmt' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'pmt', call_assignment_105193_105314)
    
    # Assigning a Call to a Name (line 118):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105318 = {}
    # Getting the type of 'call_assignment_105190' (line 118)
    call_assignment_105190_105315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105190', False)
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___105316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), call_assignment_105190_105315, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105319 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105316, *[int_105317], **kwargs_105318)
    
    # Assigning a type to the variable 'call_assignment_105194' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105194', getitem___call_result_105319)
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'call_assignment_105194' (line 118)
    call_assignment_105194_105320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105194')
    # Assigning a type to the variable 'pv' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'pv', call_assignment_105194_105320)
    
    # Assigning a Call to a Name (line 118):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105324 = {}
    # Getting the type of 'call_assignment_105190' (line 118)
    call_assignment_105190_105321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105190', False)
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___105322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), call_assignment_105190_105321, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105325 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105322, *[int_105323], **kwargs_105324)
    
    # Assigning a type to the variable 'call_assignment_105195' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105195', getitem___call_result_105325)
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'call_assignment_105195' (line 118)
    call_assignment_105195_105326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_105195')
    # Assigning a type to the variable 'when' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'when', call_assignment_105195_105326)
    
    # Assigning a BinOp to a Name (line 119):
    
    # Assigning a BinOp to a Name (line 119):
    int_105327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 12), 'int')
    # Getting the type of 'rate' (line 119)
    rate_105328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'rate')
    # Applying the binary operator '+' (line 119)
    result_add_105329 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 12), '+', int_105327, rate_105328)
    
    # Getting the type of 'nper' (line 119)
    nper_105330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'nper')
    # Applying the binary operator '**' (line 119)
    result_pow_105331 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '**', result_add_105329, nper_105330)
    
    # Assigning a type to the variable 'temp' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'temp', result_pow_105331)
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to broadcast(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'rate' (line 120)
    rate_105334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'rate', False)
    # Getting the type of 'nper' (line 120)
    nper_105335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'nper', False)
    # Getting the type of 'pmt' (line 120)
    pmt_105336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'pmt', False)
    # Getting the type of 'pv' (line 120)
    pv_105337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 42), 'pv', False)
    # Getting the type of 'when' (line 120)
    when_105338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 46), 'when', False)
    # Processing the call keyword arguments (line 120)
    kwargs_105339 = {}
    # Getting the type of 'np' (line 120)
    np_105332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'np', False)
    # Obtaining the member 'broadcast' of a type (line 120)
    broadcast_105333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), np_105332, 'broadcast')
    # Calling broadcast(args, kwargs) (line 120)
    broadcast_call_result_105340 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), broadcast_105333, *[rate_105334, nper_105335, pmt_105336, pv_105337, when_105338], **kwargs_105339)
    
    # Assigning a type to the variable 'miter' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'miter', broadcast_call_result_105340)
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to zeros(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'miter' (line 121)
    miter_105343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'miter', False)
    # Obtaining the member 'shape' of a type (line 121)
    shape_105344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 19), miter_105343, 'shape')
    # Processing the call keyword arguments (line 121)
    kwargs_105345 = {}
    # Getting the type of 'np' (line 121)
    np_105341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 121)
    zeros_105342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 10), np_105341, 'zeros')
    # Calling zeros(args, kwargs) (line 121)
    zeros_call_result_105346 = invoke(stypy.reporting.localization.Localization(__file__, 121, 10), zeros_105342, *[shape_105344], **kwargs_105345)
    
    # Assigning a type to the variable 'zer' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'zer', zeros_call_result_105346)
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to where(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Getting the type of 'rate' (line 122)
    rate_105349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'rate', False)
    # Getting the type of 'zer' (line 122)
    zer_105350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'zer', False)
    # Applying the binary operator '==' (line 122)
    result_eq_105351 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 20), '==', rate_105349, zer_105350)
    
    # Getting the type of 'nper' (line 122)
    nper_105352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'nper', False)
    # Getting the type of 'zer' (line 122)
    zer_105353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'zer', False)
    # Applying the binary operator '+' (line 122)
    result_add_105354 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 33), '+', nper_105352, zer_105353)
    
    int_105355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'int')
    # Getting the type of 'rate' (line 123)
    rate_105356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'rate', False)
    # Getting the type of 'when' (line 123)
    when_105357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'when', False)
    # Applying the binary operator '*' (line 123)
    result_mul_105358 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 25), '*', rate_105356, when_105357)
    
    # Applying the binary operator '+' (line 123)
    result_add_105359 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 21), '+', int_105355, result_mul_105358)
    
    # Getting the type of 'temp' (line 123)
    temp_105360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'temp', False)
    int_105361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'int')
    # Applying the binary operator '-' (line 123)
    result_sub_105362 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 37), '-', temp_105360, int_105361)
    
    # Applying the binary operator '*' (line 123)
    result_mul_105363 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 20), '*', result_add_105359, result_sub_105362)
    
    # Getting the type of 'rate' (line 123)
    rate_105364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'rate', False)
    # Applying the binary operator 'div' (line 123)
    result_div_105365 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 46), 'div', result_mul_105363, rate_105364)
    
    # Getting the type of 'zer' (line 123)
    zer_105366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 54), 'zer', False)
    # Applying the binary operator '+' (line 123)
    result_add_105367 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 20), '+', result_div_105365, zer_105366)
    
    # Processing the call keyword arguments (line 122)
    kwargs_105368 = {}
    # Getting the type of 'np' (line 122)
    np_105347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'np', False)
    # Obtaining the member 'where' of a type (line 122)
    where_105348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 11), np_105347, 'where')
    # Calling where(args, kwargs) (line 122)
    where_call_result_105369 = invoke(stypy.reporting.localization.Localization(__file__, 122, 11), where_105348, *[result_eq_105351, result_add_105354, result_add_105367], **kwargs_105368)
    
    # Assigning a type to the variable 'fact' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'fact', where_call_result_105369)
    
    # Getting the type of 'pv' (line 124)
    pv_105370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'pv')
    # Getting the type of 'temp' (line 124)
    temp_105371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'temp')
    # Applying the binary operator '*' (line 124)
    result_mul_105372 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 13), '*', pv_105370, temp_105371)
    
    # Getting the type of 'pmt' (line 124)
    pmt_105373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'pmt')
    # Getting the type of 'fact' (line 124)
    fact_105374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'fact')
    # Applying the binary operator '*' (line 124)
    result_mul_105375 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 23), '*', pmt_105373, fact_105374)
    
    # Applying the binary operator '+' (line 124)
    result_add_105376 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 13), '+', result_mul_105372, result_mul_105375)
    
    # Applying the 'usub' unary operator (line 124)
    result___neg___105377 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), 'usub', result_add_105376)
    
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type', result___neg___105377)
    
    # ################# End of 'fv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fv' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_105378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105378)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fv'
    return stypy_return_type_105378

# Assigning a type to the variable 'fv' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'fv', fv)

@norecursion
def pmt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_105379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 27), 'int')
    str_105380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 35), 'str', 'end')
    defaults = [int_105379, str_105380]
    # Create a new context for function 'pmt'
    module_type_store = module_type_store.open_function_context('pmt', 126, 0, False)
    
    # Passed parameters checking function
    pmt.stypy_localization = localization
    pmt.stypy_type_of_self = None
    pmt.stypy_type_store = module_type_store
    pmt.stypy_function_name = 'pmt'
    pmt.stypy_param_names_list = ['rate', 'nper', 'pv', 'fv', 'when']
    pmt.stypy_varargs_param_name = None
    pmt.stypy_kwargs_param_name = None
    pmt.stypy_call_defaults = defaults
    pmt.stypy_call_varargs = varargs
    pmt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pmt', ['rate', 'nper', 'pv', 'fv', 'when'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pmt', localization, ['rate', 'nper', 'pv', 'fv', 'when'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pmt(...)' code ##################

    str_105381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, (-1)), 'str', "\n    Compute the payment against loan principal plus interest.\n\n    Given:\n     * a present value, `pv` (e.g., an amount borrowed)\n     * a future value, `fv` (e.g., 0)\n     * an interest `rate` compounded once per period, of which\n       there are\n     * `nper` total\n     * and (optional) specification of whether payment is made\n       at the beginning (`when` = {'begin', 1}) or the end\n       (`when` = {'end', 0}) of each period\n\n    Return:\n       the (fixed) periodic payment.\n\n    Parameters\n    ----------\n    rate : array_like\n        Rate of interest (per period)\n    nper : array_like\n        Number of compounding periods\n    pv : array_like\n        Present value\n    fv : array_like,  optional\n        Future value (default = 0)\n    when : {{'begin', 1}, {'end', 0}}, {string, int}\n        When payments are due ('begin' (1) or 'end' (0))\n\n    Returns\n    -------\n    out : ndarray\n        Payment against loan plus interest.  If all input is scalar, returns a\n        scalar float.  If any input is array_like, returns payment for each\n        input element. If multiple inputs are array_like, they all must have\n        the same shape.\n\n    Notes\n    -----\n    The payment is computed by solving the equation::\n\n     fv +\n     pv*(1 + rate)**nper +\n     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0\n\n    or, when ``rate == 0``::\n\n      fv + pv + pmt * nper == 0\n\n    for ``pmt``.\n\n    Note that computing a monthly mortgage payment is only\n    one use for this function.  For example, pmt returns the\n    periodic deposit one must make to achieve a specified\n    future balance given an initial deposit, a fixed,\n    periodically compounded interest rate, and the total\n    number of periods.\n\n    References\n    ----------\n    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).\n       Open Document Format for Office Applications (OpenDocument)v1.2,\n       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,\n       Pre-Draft 12. Organization for the Advancement of Structured Information\n       Standards (OASIS). Billerica, MA, USA. [ODT Document].\n       Available:\n       http://www.oasis-open.org/committees/documents.php\n       ?wg_abbrev=office-formulaOpenDocument-formula-20090508.odt\n\n    Examples\n    --------\n    What is the monthly payment needed to pay off a $200,000 loan in 15\n    years at an annual interest rate of 7.5%?\n\n    >>> np.pmt(0.075/12, 12*15, 200000)\n    -1854.0247200054619\n\n    In order to pay-off (i.e., have a future-value of 0) the $200,000 obtained\n    today, a monthly payment of $1,854.02 would be required.  Note that this\n    example illustrates usage of `fv` having a default value of 0.\n\n    ")
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to _convert_when(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'when' (line 209)
    when_105383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 25), 'when', False)
    # Processing the call keyword arguments (line 209)
    kwargs_105384 = {}
    # Getting the type of '_convert_when' (line 209)
    _convert_when_105382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), '_convert_when', False)
    # Calling _convert_when(args, kwargs) (line 209)
    _convert_when_call_result_105385 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), _convert_when_105382, *[when_105383], **kwargs_105384)
    
    # Assigning a type to the variable 'when' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'when', _convert_when_call_result_105385)
    
    # Assigning a Call to a Tuple (line 210):
    
    # Assigning a Call to a Name:
    
    # Call to map(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'np' (line 210)
    np_105387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 37), 'np', False)
    # Obtaining the member 'asarray' of a type (line 210)
    asarray_105388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 37), np_105387, 'asarray')
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_105389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of 'rate' (line 210)
    rate_105390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 50), 'rate', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 49), list_105389, rate_105390)
    # Adding element type (line 210)
    # Getting the type of 'nper' (line 210)
    nper_105391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 56), 'nper', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 49), list_105389, nper_105391)
    # Adding element type (line 210)
    # Getting the type of 'pv' (line 210)
    pv_105392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 62), 'pv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 49), list_105389, pv_105392)
    # Adding element type (line 210)
    # Getting the type of 'fv' (line 210)
    fv_105393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 66), 'fv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 49), list_105389, fv_105393)
    # Adding element type (line 210)
    # Getting the type of 'when' (line 210)
    when_105394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 70), 'when', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 49), list_105389, when_105394)
    
    # Processing the call keyword arguments (line 210)
    kwargs_105395 = {}
    # Getting the type of 'map' (line 210)
    map_105386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 33), 'map', False)
    # Calling map(args, kwargs) (line 210)
    map_call_result_105396 = invoke(stypy.reporting.localization.Localization(__file__, 210, 33), map_105386, *[asarray_105388, list_105389], **kwargs_105395)
    
    # Assigning a type to the variable 'call_assignment_105196' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105196', map_call_result_105396)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105400 = {}
    # Getting the type of 'call_assignment_105196' (line 210)
    call_assignment_105196_105397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105196', False)
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___105398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 4), call_assignment_105196_105397, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105401 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105398, *[int_105399], **kwargs_105400)
    
    # Assigning a type to the variable 'call_assignment_105197' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105197', getitem___call_result_105401)
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'call_assignment_105197' (line 210)
    call_assignment_105197_105402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105197')
    # Assigning a type to the variable 'rate' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 5), 'rate', call_assignment_105197_105402)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105406 = {}
    # Getting the type of 'call_assignment_105196' (line 210)
    call_assignment_105196_105403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105196', False)
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___105404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 4), call_assignment_105196_105403, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105407 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105404, *[int_105405], **kwargs_105406)
    
    # Assigning a type to the variable 'call_assignment_105198' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105198', getitem___call_result_105407)
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'call_assignment_105198' (line 210)
    call_assignment_105198_105408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105198')
    # Assigning a type to the variable 'nper' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'nper', call_assignment_105198_105408)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105412 = {}
    # Getting the type of 'call_assignment_105196' (line 210)
    call_assignment_105196_105409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105196', False)
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___105410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 4), call_assignment_105196_105409, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105413 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105410, *[int_105411], **kwargs_105412)
    
    # Assigning a type to the variable 'call_assignment_105199' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105199', getitem___call_result_105413)
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'call_assignment_105199' (line 210)
    call_assignment_105199_105414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105199')
    # Assigning a type to the variable 'pv' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'pv', call_assignment_105199_105414)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105418 = {}
    # Getting the type of 'call_assignment_105196' (line 210)
    call_assignment_105196_105415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105196', False)
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___105416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 4), call_assignment_105196_105415, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105419 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105416, *[int_105417], **kwargs_105418)
    
    # Assigning a type to the variable 'call_assignment_105200' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105200', getitem___call_result_105419)
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'call_assignment_105200' (line 210)
    call_assignment_105200_105420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105200')
    # Assigning a type to the variable 'fv' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'fv', call_assignment_105200_105420)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105424 = {}
    # Getting the type of 'call_assignment_105196' (line 210)
    call_assignment_105196_105421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105196', False)
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___105422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 4), call_assignment_105196_105421, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105425 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105422, *[int_105423], **kwargs_105424)
    
    # Assigning a type to the variable 'call_assignment_105201' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105201', getitem___call_result_105425)
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'call_assignment_105201' (line 210)
    call_assignment_105201_105426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_105201')
    # Assigning a type to the variable 'when' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'when', call_assignment_105201_105426)
    
    # Assigning a BinOp to a Name (line 211):
    
    # Assigning a BinOp to a Name (line 211):
    int_105427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
    # Getting the type of 'rate' (line 211)
    rate_105428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'rate')
    # Applying the binary operator '+' (line 211)
    result_add_105429 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 12), '+', int_105427, rate_105428)
    
    # Getting the type of 'nper' (line 211)
    nper_105430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'nper')
    # Applying the binary operator '**' (line 211)
    result_pow_105431 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), '**', result_add_105429, nper_105430)
    
    # Assigning a type to the variable 'temp' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'temp', result_pow_105431)
    
    # Assigning a Compare to a Name (line 212):
    
    # Assigning a Compare to a Name (line 212):
    
    # Getting the type of 'rate' (line 212)
    rate_105432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'rate')
    float_105433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 20), 'float')
    # Applying the binary operator '==' (line 212)
    result_eq_105434 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 12), '==', rate_105432, float_105433)
    
    # Assigning a type to the variable 'mask' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'mask', result_eq_105434)
    
    # Call to copyto(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'rate' (line 213)
    rate_105437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 14), 'rate', False)
    float_105438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 20), 'float')
    # Processing the call keyword arguments (line 213)
    # Getting the type of 'mask' (line 213)
    mask_105439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 31), 'mask', False)
    keyword_105440 = mask_105439
    kwargs_105441 = {'where': keyword_105440}
    # Getting the type of 'np' (line 213)
    np_105435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'np', False)
    # Obtaining the member 'copyto' of a type (line 213)
    copyto_105436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 4), np_105435, 'copyto')
    # Calling copyto(args, kwargs) (line 213)
    copyto_call_result_105442 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), copyto_105436, *[rate_105437, float_105438], **kwargs_105441)
    
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to zeros(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Call to broadcast(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'rate' (line 214)
    rate_105447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'rate', False)
    # Getting the type of 'nper' (line 214)
    nper_105448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 36), 'nper', False)
    # Getting the type of 'pv' (line 214)
    pv_105449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'pv', False)
    # Getting the type of 'fv' (line 214)
    fv_105450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 46), 'fv', False)
    # Getting the type of 'when' (line 214)
    when_105451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 50), 'when', False)
    # Processing the call keyword arguments (line 214)
    kwargs_105452 = {}
    # Getting the type of 'np' (line 214)
    np_105445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 17), 'np', False)
    # Obtaining the member 'broadcast' of a type (line 214)
    broadcast_105446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 17), np_105445, 'broadcast')
    # Calling broadcast(args, kwargs) (line 214)
    broadcast_call_result_105453 = invoke(stypy.reporting.localization.Localization(__file__, 214, 17), broadcast_105446, *[rate_105447, nper_105448, pv_105449, fv_105450, when_105451], **kwargs_105452)
    
    # Obtaining the member 'shape' of a type (line 214)
    shape_105454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 17), broadcast_call_result_105453, 'shape')
    # Processing the call keyword arguments (line 214)
    kwargs_105455 = {}
    # Getting the type of 'np' (line 214)
    np_105443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 214)
    zeros_105444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), np_105443, 'zeros')
    # Calling zeros(args, kwargs) (line 214)
    zeros_call_result_105456 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), zeros_105444, *[shape_105454], **kwargs_105455)
    
    # Assigning a type to the variable 'z' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'z', zeros_call_result_105456)
    
    # Assigning a Call to a Name (line 215):
    
    # Assigning a Call to a Name (line 215):
    
    # Call to where(...): (line 215)
    # Processing the call arguments (line 215)
    
    # Getting the type of 'mask' (line 215)
    mask_105459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'mask', False)
    # Getting the type of 'z' (line 215)
    z_105460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 28), 'z', False)
    # Applying the binary operator '!=' (line 215)
    result_ne_105461 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 20), '!=', mask_105459, z_105460)
    
    # Getting the type of 'nper' (line 215)
    nper_105462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 31), 'nper', False)
    # Getting the type of 'z' (line 215)
    z_105463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 38), 'z', False)
    # Applying the binary operator '+' (line 215)
    result_add_105464 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 31), '+', nper_105462, z_105463)
    
    int_105465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 42), 'int')
    # Getting the type of 'rate' (line 215)
    rate_105466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 46), 'rate', False)
    # Getting the type of 'when' (line 215)
    when_105467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 51), 'when', False)
    # Applying the binary operator '*' (line 215)
    result_mul_105468 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 46), '*', rate_105466, when_105467)
    
    # Applying the binary operator '+' (line 215)
    result_add_105469 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 42), '+', int_105465, result_mul_105468)
    
    # Getting the type of 'temp' (line 215)
    temp_105470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 58), 'temp', False)
    int_105471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 65), 'int')
    # Applying the binary operator '-' (line 215)
    result_sub_105472 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 58), '-', temp_105470, int_105471)
    
    # Applying the binary operator '*' (line 215)
    result_mul_105473 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 41), '*', result_add_105469, result_sub_105472)
    
    # Getting the type of 'rate' (line 215)
    rate_105474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 68), 'rate', False)
    # Applying the binary operator 'div' (line 215)
    result_div_105475 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 67), 'div', result_mul_105473, rate_105474)
    
    # Getting the type of 'z' (line 215)
    z_105476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 75), 'z', False)
    # Applying the binary operator '+' (line 215)
    result_add_105477 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 41), '+', result_div_105475, z_105476)
    
    # Processing the call keyword arguments (line 215)
    kwargs_105478 = {}
    # Getting the type of 'np' (line 215)
    np_105457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'np', False)
    # Obtaining the member 'where' of a type (line 215)
    where_105458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 11), np_105457, 'where')
    # Calling where(args, kwargs) (line 215)
    where_call_result_105479 = invoke(stypy.reporting.localization.Localization(__file__, 215, 11), where_105458, *[result_ne_105461, result_add_105464, result_add_105477], **kwargs_105478)
    
    # Assigning a type to the variable 'fact' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'fact', where_call_result_105479)
    
    # Getting the type of 'fv' (line 216)
    fv_105480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'fv')
    # Getting the type of 'pv' (line 216)
    pv_105481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'pv')
    # Getting the type of 'temp' (line 216)
    temp_105482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'temp')
    # Applying the binary operator '*' (line 216)
    result_mul_105483 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 18), '*', pv_105481, temp_105482)
    
    # Applying the binary operator '+' (line 216)
    result_add_105484 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 13), '+', fv_105480, result_mul_105483)
    
    # Applying the 'usub' unary operator (line 216)
    result___neg___105485 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 11), 'usub', result_add_105484)
    
    # Getting the type of 'fact' (line 216)
    fact_105486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 29), 'fact')
    # Applying the binary operator 'div' (line 216)
    result_div_105487 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 11), 'div', result___neg___105485, fact_105486)
    
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type', result_div_105487)
    
    # ################# End of 'pmt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pmt' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_105488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105488)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pmt'
    return stypy_return_type_105488

# Assigning a type to the variable 'pmt' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'pmt', pmt)

@norecursion
def nper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_105489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 27), 'int')
    str_105490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 35), 'str', 'end')
    defaults = [int_105489, str_105490]
    # Create a new context for function 'nper'
    module_type_store = module_type_store.open_function_context('nper', 218, 0, False)
    
    # Passed parameters checking function
    nper.stypy_localization = localization
    nper.stypy_type_of_self = None
    nper.stypy_type_store = module_type_store
    nper.stypy_function_name = 'nper'
    nper.stypy_param_names_list = ['rate', 'pmt', 'pv', 'fv', 'when']
    nper.stypy_varargs_param_name = None
    nper.stypy_kwargs_param_name = None
    nper.stypy_call_defaults = defaults
    nper.stypy_call_varargs = varargs
    nper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nper', ['rate', 'pmt', 'pv', 'fv', 'when'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nper', localization, ['rate', 'pmt', 'pv', 'fv', 'when'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nper(...)' code ##################

    str_105491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, (-1)), 'str', "\n    Compute the number of periodic payments.\n\n    Parameters\n    ----------\n    rate : array_like\n        Rate of interest (per period)\n    pmt : array_like\n        Payment\n    pv : array_like\n        Present value\n    fv : array_like, optional\n        Future value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional\n        When payments are due ('begin' (1) or 'end' (0))\n\n    Notes\n    -----\n    The number of periods ``nper`` is computed by solving the equation::\n\n     fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate*((1+rate)**nper-1) = 0\n\n    but if ``rate = 0`` then::\n\n     fv + pv + pmt*nper = 0\n\n    Examples\n    --------\n    If you only had $150/month to pay towards the loan, how long would it take\n    to pay-off a loan of $8,000 at 7% annual interest?\n\n    >>> print(round(np.nper(0.07/12, -150, 8000), 5))\n    64.07335\n\n    So, over 64 months would be required to pay off the loan.\n\n    The same analysis could be done with several different interest rates\n    and/or payments and/or total amounts to produce an entire table.\n\n    >>> np.nper(*(np.ogrid[0.07/12: 0.08/12: 0.01/12,\n    ...                    -150   : -99     : 50    ,\n    ...                    8000   : 9001    : 1000]))\n    array([[[  64.07334877,   74.06368256],\n            [ 108.07548412,  127.99022654]],\n           [[  66.12443902,   76.87897353],\n            [ 114.70165583,  137.90124779]]])\n\n    ")
    
    # Assigning a Call to a Name (line 267):
    
    # Assigning a Call to a Name (line 267):
    
    # Call to _convert_when(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'when' (line 267)
    when_105493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'when', False)
    # Processing the call keyword arguments (line 267)
    kwargs_105494 = {}
    # Getting the type of '_convert_when' (line 267)
    _convert_when_105492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), '_convert_when', False)
    # Calling _convert_when(args, kwargs) (line 267)
    _convert_when_call_result_105495 = invoke(stypy.reporting.localization.Localization(__file__, 267, 11), _convert_when_105492, *[when_105493], **kwargs_105494)
    
    # Assigning a type to the variable 'when' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'when', _convert_when_call_result_105495)
    
    # Assigning a Call to a Tuple (line 268):
    
    # Assigning a Call to a Name:
    
    # Call to map(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'np' (line 268)
    np_105497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'np', False)
    # Obtaining the member 'asarray' of a type (line 268)
    asarray_105498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 36), np_105497, 'asarray')
    
    # Obtaining an instance of the builtin type 'list' (line 268)
    list_105499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 268)
    # Adding element type (line 268)
    # Getting the type of 'rate' (line 268)
    rate_105500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 49), 'rate', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 48), list_105499, rate_105500)
    # Adding element type (line 268)
    # Getting the type of 'pmt' (line 268)
    pmt_105501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 55), 'pmt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 48), list_105499, pmt_105501)
    # Adding element type (line 268)
    # Getting the type of 'pv' (line 268)
    pv_105502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 60), 'pv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 48), list_105499, pv_105502)
    # Adding element type (line 268)
    # Getting the type of 'fv' (line 268)
    fv_105503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 64), 'fv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 48), list_105499, fv_105503)
    # Adding element type (line 268)
    # Getting the type of 'when' (line 268)
    when_105504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 68), 'when', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 48), list_105499, when_105504)
    
    # Processing the call keyword arguments (line 268)
    kwargs_105505 = {}
    # Getting the type of 'map' (line 268)
    map_105496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 32), 'map', False)
    # Calling map(args, kwargs) (line 268)
    map_call_result_105506 = invoke(stypy.reporting.localization.Localization(__file__, 268, 32), map_105496, *[asarray_105498, list_105499], **kwargs_105505)
    
    # Assigning a type to the variable 'call_assignment_105202' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105202', map_call_result_105506)
    
    # Assigning a Call to a Name (line 268):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105510 = {}
    # Getting the type of 'call_assignment_105202' (line 268)
    call_assignment_105202_105507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105202', False)
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___105508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 4), call_assignment_105202_105507, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105511 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105508, *[int_105509], **kwargs_105510)
    
    # Assigning a type to the variable 'call_assignment_105203' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105203', getitem___call_result_105511)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'call_assignment_105203' (line 268)
    call_assignment_105203_105512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105203')
    # Assigning a type to the variable 'rate' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 5), 'rate', call_assignment_105203_105512)
    
    # Assigning a Call to a Name (line 268):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105516 = {}
    # Getting the type of 'call_assignment_105202' (line 268)
    call_assignment_105202_105513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105202', False)
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___105514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 4), call_assignment_105202_105513, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105517 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105514, *[int_105515], **kwargs_105516)
    
    # Assigning a type to the variable 'call_assignment_105204' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105204', getitem___call_result_105517)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'call_assignment_105204' (line 268)
    call_assignment_105204_105518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105204')
    # Assigning a type to the variable 'pmt' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'pmt', call_assignment_105204_105518)
    
    # Assigning a Call to a Name (line 268):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105522 = {}
    # Getting the type of 'call_assignment_105202' (line 268)
    call_assignment_105202_105519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105202', False)
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___105520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 4), call_assignment_105202_105519, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105523 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105520, *[int_105521], **kwargs_105522)
    
    # Assigning a type to the variable 'call_assignment_105205' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105205', getitem___call_result_105523)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'call_assignment_105205' (line 268)
    call_assignment_105205_105524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105205')
    # Assigning a type to the variable 'pv' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'pv', call_assignment_105205_105524)
    
    # Assigning a Call to a Name (line 268):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105528 = {}
    # Getting the type of 'call_assignment_105202' (line 268)
    call_assignment_105202_105525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105202', False)
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___105526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 4), call_assignment_105202_105525, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105529 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105526, *[int_105527], **kwargs_105528)
    
    # Assigning a type to the variable 'call_assignment_105206' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105206', getitem___call_result_105529)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'call_assignment_105206' (line 268)
    call_assignment_105206_105530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105206')
    # Assigning a type to the variable 'fv' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'fv', call_assignment_105206_105530)
    
    # Assigning a Call to a Name (line 268):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105534 = {}
    # Getting the type of 'call_assignment_105202' (line 268)
    call_assignment_105202_105531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105202', False)
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___105532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 4), call_assignment_105202_105531, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105535 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105532, *[int_105533], **kwargs_105534)
    
    # Assigning a type to the variable 'call_assignment_105207' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105207', getitem___call_result_105535)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'call_assignment_105207' (line 268)
    call_assignment_105207_105536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'call_assignment_105207')
    # Assigning a type to the variable 'when' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 24), 'when', call_assignment_105207_105536)
    
    # Assigning a Name to a Name (line 270):
    
    # Assigning a Name to a Name (line 270):
    # Getting the type of 'False' (line 270)
    False_105537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'False')
    # Assigning a type to the variable 'use_zero_rate' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'use_zero_rate', False_105537)
    
    # Call to errstate(...): (line 271)
    # Processing the call keyword arguments (line 271)
    str_105540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 28), 'str', 'raise')
    keyword_105541 = str_105540
    kwargs_105542 = {'divide': keyword_105541}
    # Getting the type of 'np' (line 271)
    np_105538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 271)
    errstate_105539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 9), np_105538, 'errstate')
    # Calling errstate(args, kwargs) (line 271)
    errstate_call_result_105543 = invoke(stypy.reporting.localization.Localization(__file__, 271, 9), errstate_105539, *[], **kwargs_105542)
    
    with_105544 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 271, 9), errstate_call_result_105543, 'with parameter', '__enter__', '__exit__')

    if with_105544:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 271)
        enter___105545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 9), errstate_call_result_105543, '__enter__')
        with_enter_105546 = invoke(stypy.reporting.localization.Localization(__file__, 271, 9), enter___105545)
        
        
        # SSA begins for try-except statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a BinOp to a Name (line 273):
        
        # Assigning a BinOp to a Name (line 273):
        # Getting the type of 'pmt' (line 273)
        pmt_105547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'pmt')
        float_105548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 21), 'float')
        # Getting the type of 'rate' (line 273)
        rate_105549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'rate')
        # Getting the type of 'when' (line 273)
        when_105550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'when')
        # Applying the binary operator '*' (line 273)
        result_mul_105551 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 25), '*', rate_105549, when_105550)
        
        # Applying the binary operator '+' (line 273)
        result_add_105552 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 21), '+', float_105548, result_mul_105551)
        
        # Applying the binary operator '*' (line 273)
        result_mul_105553 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 16), '*', pmt_105547, result_add_105552)
        
        # Getting the type of 'rate' (line 273)
        rate_105554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 36), 'rate')
        # Applying the binary operator 'div' (line 273)
        result_div_105555 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 35), 'div', result_mul_105553, rate_105554)
        
        # Assigning a type to the variable 'z' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'z', result_div_105555)
        # SSA branch for the except part of a try statement (line 272)
        # SSA branch for the except 'FloatingPointError' branch of a try statement (line 272)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 275):
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'True' (line 275)
        True_105556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 28), 'True')
        # Assigning a type to the variable 'use_zero_rate' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'use_zero_rate', True_105556)
        # SSA join for try-except statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 271)
        exit___105557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 9), errstate_call_result_105543, '__exit__')
        with_exit_105558 = invoke(stypy.reporting.localization.Localization(__file__, 271, 9), exit___105557, None, None, None)

    
    # Getting the type of 'use_zero_rate' (line 277)
    use_zero_rate_105559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 7), 'use_zero_rate')
    # Testing the type of an if condition (line 277)
    if_condition_105560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 4), use_zero_rate_105559)
    # Assigning a type to the variable 'if_condition_105560' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'if_condition_105560', if_condition_105560)
    # SSA begins for if statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'fv' (line 278)
    fv_105561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'fv')
    # Applying the 'usub' unary operator (line 278)
    result___neg___105562 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 16), 'usub', fv_105561)
    
    # Getting the type of 'pv' (line 278)
    pv_105563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'pv')
    # Applying the binary operator '+' (line 278)
    result_add_105564 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 16), '+', result___neg___105562, pv_105563)
    
    # Getting the type of 'pmt' (line 278)
    pmt_105565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'pmt')
    float_105566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 35), 'float')
    # Applying the binary operator '+' (line 278)
    result_add_105567 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 29), '+', pmt_105565, float_105566)
    
    # Applying the binary operator 'div' (line 278)
    result_div_105568 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 15), 'div', result_add_105564, result_add_105567)
    
    # Assigning a type to the variable 'stypy_return_type' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'stypy_return_type', result_div_105568)
    # SSA branch for the else part of an if statement (line 277)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 280):
    
    # Assigning a BinOp to a Name (line 280):
    
    # Getting the type of 'fv' (line 280)
    fv_105569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 14), 'fv')
    # Getting the type of 'pv' (line 280)
    pv_105570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'pv')
    # Applying the binary operator '+' (line 280)
    result_add_105571 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 14), '+', fv_105569, pv_105570)
    
    # Applying the 'usub' unary operator (line 280)
    result___neg___105572 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 12), 'usub', result_add_105571)
    
    # Getting the type of 'pmt' (line 280)
    pmt_105573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'pmt')
    float_105574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 28), 'float')
    # Applying the binary operator '+' (line 280)
    result_add_105575 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 24), '+', pmt_105573, float_105574)
    
    # Applying the binary operator 'div' (line 280)
    result_div_105576 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 12), 'div', result___neg___105572, result_add_105575)
    
    # Assigning a type to the variable 'A' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'A', result_div_105576)
    
    # Assigning a BinOp to a Name (line 281):
    
    # Assigning a BinOp to a Name (line 281):
    
    # Call to log(...): (line 281)
    # Processing the call arguments (line 281)
    
    # Getting the type of 'fv' (line 281)
    fv_105579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'fv', False)
    # Applying the 'usub' unary operator (line 281)
    result___neg___105580 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), 'usub', fv_105579)
    
    # Getting the type of 'z' (line 281)
    z_105581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'z', False)
    # Applying the binary operator '+' (line 281)
    result_add_105582 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), '+', result___neg___105580, z_105581)
    
    # Getting the type of 'pv' (line 281)
    pv_105583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 30), 'pv', False)
    # Getting the type of 'z' (line 281)
    z_105584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 33), 'z', False)
    # Applying the binary operator '+' (line 281)
    result_add_105585 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 30), '+', pv_105583, z_105584)
    
    # Applying the binary operator 'div' (line 281)
    result_div_105586 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 19), 'div', result_add_105582, result_add_105585)
    
    # Processing the call keyword arguments (line 281)
    kwargs_105587 = {}
    # Getting the type of 'np' (line 281)
    np_105577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'np', False)
    # Obtaining the member 'log' of a type (line 281)
    log_105578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), np_105577, 'log')
    # Calling log(args, kwargs) (line 281)
    log_call_result_105588 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), log_105578, *[result_div_105586], **kwargs_105587)
    
    
    # Call to log(...): (line 281)
    # Processing the call arguments (line 281)
    float_105591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 44), 'float')
    # Getting the type of 'rate' (line 281)
    rate_105592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 48), 'rate', False)
    # Applying the binary operator '+' (line 281)
    result_add_105593 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 44), '+', float_105591, rate_105592)
    
    # Processing the call keyword arguments (line 281)
    kwargs_105594 = {}
    # Getting the type of 'np' (line 281)
    np_105589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 37), 'np', False)
    # Obtaining the member 'log' of a type (line 281)
    log_105590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 37), np_105589, 'log')
    # Calling log(args, kwargs) (line 281)
    log_call_result_105595 = invoke(stypy.reporting.localization.Localization(__file__, 281, 37), log_105590, *[result_add_105593], **kwargs_105594)
    
    # Applying the binary operator 'div' (line 281)
    result_div_105596 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 12), 'div', log_call_result_105588, log_call_result_105595)
    
    # Assigning a type to the variable 'B' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'B', result_div_105596)
    
    # Assigning a Call to a Name (line 282):
    
    # Assigning a Call to a Name (line 282):
    
    # Call to broadcast(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'rate' (line 282)
    rate_105599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 29), 'rate', False)
    # Getting the type of 'pmt' (line 282)
    pmt_105600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 35), 'pmt', False)
    # Getting the type of 'pv' (line 282)
    pv_105601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 40), 'pv', False)
    # Getting the type of 'fv' (line 282)
    fv_105602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 44), 'fv', False)
    # Getting the type of 'when' (line 282)
    when_105603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 48), 'when', False)
    # Processing the call keyword arguments (line 282)
    kwargs_105604 = {}
    # Getting the type of 'np' (line 282)
    np_105597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'np', False)
    # Obtaining the member 'broadcast' of a type (line 282)
    broadcast_105598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), np_105597, 'broadcast')
    # Calling broadcast(args, kwargs) (line 282)
    broadcast_call_result_105605 = invoke(stypy.reporting.localization.Localization(__file__, 282, 16), broadcast_105598, *[rate_105599, pmt_105600, pv_105601, fv_105602, when_105603], **kwargs_105604)
    
    # Assigning a type to the variable 'miter' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'miter', broadcast_call_result_105605)
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to zeros(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'miter' (line 283)
    miter_105608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 23), 'miter', False)
    # Obtaining the member 'shape' of a type (line 283)
    shape_105609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 23), miter_105608, 'shape')
    # Processing the call keyword arguments (line 283)
    kwargs_105610 = {}
    # Getting the type of 'np' (line 283)
    np_105606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 14), 'np', False)
    # Obtaining the member 'zeros' of a type (line 283)
    zeros_105607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 14), np_105606, 'zeros')
    # Calling zeros(args, kwargs) (line 283)
    zeros_call_result_105611 = invoke(stypy.reporting.localization.Localization(__file__, 283, 14), zeros_105607, *[shape_105609], **kwargs_105610)
    
    # Assigning a type to the variable 'zer' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'zer', zeros_call_result_105611)
    
    # Call to where(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Getting the type of 'rate' (line 284)
    rate_105614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'rate', False)
    # Getting the type of 'zer' (line 284)
    zer_105615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 32), 'zer', False)
    # Applying the binary operator '==' (line 284)
    result_eq_105616 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 24), '==', rate_105614, zer_105615)
    
    # Getting the type of 'A' (line 284)
    A_105617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 37), 'A', False)
    # Getting the type of 'zer' (line 284)
    zer_105618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 41), 'zer', False)
    # Applying the binary operator '+' (line 284)
    result_add_105619 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 37), '+', A_105617, zer_105618)
    
    # Getting the type of 'B' (line 284)
    B_105620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 46), 'B', False)
    # Getting the type of 'zer' (line 284)
    zer_105621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 50), 'zer', False)
    # Applying the binary operator '+' (line 284)
    result_add_105622 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 46), '+', B_105620, zer_105621)
    
    # Processing the call keyword arguments (line 284)
    kwargs_105623 = {}
    # Getting the type of 'np' (line 284)
    np_105612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), 'np', False)
    # Obtaining the member 'where' of a type (line 284)
    where_105613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 15), np_105612, 'where')
    # Calling where(args, kwargs) (line 284)
    where_call_result_105624 = invoke(stypy.reporting.localization.Localization(__file__, 284, 15), where_105613, *[result_eq_105616, result_add_105619, result_add_105622], **kwargs_105623)
    
    float_105625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 57), 'float')
    # Applying the binary operator '+' (line 284)
    result_add_105626 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 15), '+', where_call_result_105624, float_105625)
    
    # Assigning a type to the variable 'stypy_return_type' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'stypy_return_type', result_add_105626)
    # SSA join for if statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'nper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nper' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_105627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105627)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nper'
    return stypy_return_type_105627

# Assigning a type to the variable 'nper' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'nper', nper)

@norecursion
def ipmt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_105628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 33), 'float')
    str_105629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 43), 'str', 'end')
    defaults = [float_105628, str_105629]
    # Create a new context for function 'ipmt'
    module_type_store = module_type_store.open_function_context('ipmt', 286, 0, False)
    
    # Passed parameters checking function
    ipmt.stypy_localization = localization
    ipmt.stypy_type_of_self = None
    ipmt.stypy_type_store = module_type_store
    ipmt.stypy_function_name = 'ipmt'
    ipmt.stypy_param_names_list = ['rate', 'per', 'nper', 'pv', 'fv', 'when']
    ipmt.stypy_varargs_param_name = None
    ipmt.stypy_kwargs_param_name = None
    ipmt.stypy_call_defaults = defaults
    ipmt.stypy_call_varargs = varargs
    ipmt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ipmt', ['rate', 'per', 'nper', 'pv', 'fv', 'when'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ipmt', localization, ['rate', 'per', 'nper', 'pv', 'fv', 'when'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ipmt(...)' code ##################

    str_105630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, (-1)), 'str', "\n    Compute the interest portion of a payment.\n\n    Parameters\n    ----------\n    rate : scalar or array_like of shape(M, )\n        Rate of interest as decimal (not per cent) per period\n    per : scalar or array_like of shape(M, )\n        Interest paid against the loan changes during the life or the loan.\n        The `per` is the payment period to calculate the interest amount.\n    nper : scalar or array_like of shape(M, )\n        Number of compounding periods\n    pv : scalar or array_like of shape(M, )\n        Present value\n    fv : scalar or array_like of shape(M, ), optional\n        Future value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional\n        When payments are due ('begin' (1) or 'end' (0)).\n        Defaults to {'end', 0}.\n\n    Returns\n    -------\n    out : ndarray\n        Interest portion of payment.  If all input is scalar, returns a scalar\n        float.  If any input is array_like, returns interest payment for each\n        input element. If multiple inputs are array_like, they all must have\n        the same shape.\n\n    See Also\n    --------\n    ppmt, pmt, pv\n\n    Notes\n    -----\n    The total payment is made up of payment against principal plus interest.\n\n    ``pmt = ppmt + ipmt``\n\n    Examples\n    --------\n    What is the amortization schedule for a 1 year loan of $2500 at\n    8.24% interest per year compounded monthly?\n\n    >>> principal = 2500.00\n\n    The 'per' variable represents the periods of the loan.  Remember that\n    financial equations start the period count at 1!\n\n    >>> per = np.arange(1*12) + 1\n    >>> ipmt = np.ipmt(0.0824/12, per, 1*12, principal)\n    >>> ppmt = np.ppmt(0.0824/12, per, 1*12, principal)\n\n    Each element of the sum of the 'ipmt' and 'ppmt' arrays should equal\n    'pmt'.\n\n    >>> pmt = np.pmt(0.0824/12, 1*12, principal)\n    >>> np.allclose(ipmt + ppmt, pmt)\n    True\n\n    >>> fmt = '{0:2d} {1:8.2f} {2:8.2f} {3:8.2f}'\n    >>> for payment in per:\n    ...     index = payment - 1\n    ...     principal = principal + ppmt[index]\n    ...     print(fmt.format(payment, ppmt[index], ipmt[index], principal))\n     1  -200.58   -17.17  2299.42\n     2  -201.96   -15.79  2097.46\n     3  -203.35   -14.40  1894.11\n     4  -204.74   -13.01  1689.37\n     5  -206.15   -11.60  1483.22\n     6  -207.56   -10.18  1275.66\n     7  -208.99    -8.76  1066.67\n     8  -210.42    -7.32   856.25\n     9  -211.87    -5.88   644.38\n    10  -213.32    -4.42   431.05\n    11  -214.79    -2.96   216.26\n    12  -216.26    -1.49    -0.00\n\n    >>> interestpd = np.sum(ipmt)\n    >>> np.round(interestpd, 2)\n    -112.98\n\n    ")
    
    # Assigning a Call to a Name (line 369):
    
    # Assigning a Call to a Name (line 369):
    
    # Call to _convert_when(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'when' (line 369)
    when_105632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 25), 'when', False)
    # Processing the call keyword arguments (line 369)
    kwargs_105633 = {}
    # Getting the type of '_convert_when' (line 369)
    _convert_when_105631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), '_convert_when', False)
    # Calling _convert_when(args, kwargs) (line 369)
    _convert_when_call_result_105634 = invoke(stypy.reporting.localization.Localization(__file__, 369, 11), _convert_when_105631, *[when_105632], **kwargs_105633)
    
    # Assigning a type to the variable 'when' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'when', _convert_when_call_result_105634)
    
    # Assigning a Call to a Tuple (line 370):
    
    # Assigning a Call to a Name:
    
    # Call to broadcast_arrays(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'rate' (line 370)
    rate_105637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 56), 'rate', False)
    # Getting the type of 'per' (line 370)
    per_105638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 62), 'per', False)
    # Getting the type of 'nper' (line 370)
    nper_105639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 67), 'nper', False)
    # Getting the type of 'pv' (line 371)
    pv_105640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 56), 'pv', False)
    # Getting the type of 'fv' (line 371)
    fv_105641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 60), 'fv', False)
    # Getting the type of 'when' (line 371)
    when_105642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 64), 'when', False)
    # Processing the call keyword arguments (line 370)
    kwargs_105643 = {}
    # Getting the type of 'np' (line 370)
    np_105635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 36), 'np', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 370)
    broadcast_arrays_105636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 36), np_105635, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 370)
    broadcast_arrays_call_result_105644 = invoke(stypy.reporting.localization.Localization(__file__, 370, 36), broadcast_arrays_105636, *[rate_105637, per_105638, nper_105639, pv_105640, fv_105641, when_105642], **kwargs_105643)
    
    # Assigning a type to the variable 'call_assignment_105208' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105208', broadcast_arrays_call_result_105644)
    
    # Assigning a Call to a Name (line 370):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105648 = {}
    # Getting the type of 'call_assignment_105208' (line 370)
    call_assignment_105208_105645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105208', False)
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___105646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 4), call_assignment_105208_105645, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105649 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105646, *[int_105647], **kwargs_105648)
    
    # Assigning a type to the variable 'call_assignment_105209' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105209', getitem___call_result_105649)
    
    # Assigning a Name to a Name (line 370):
    # Getting the type of 'call_assignment_105209' (line 370)
    call_assignment_105209_105650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105209')
    # Assigning a type to the variable 'rate' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'rate', call_assignment_105209_105650)
    
    # Assigning a Call to a Name (line 370):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105654 = {}
    # Getting the type of 'call_assignment_105208' (line 370)
    call_assignment_105208_105651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105208', False)
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___105652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 4), call_assignment_105208_105651, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105655 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105652, *[int_105653], **kwargs_105654)
    
    # Assigning a type to the variable 'call_assignment_105210' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105210', getitem___call_result_105655)
    
    # Assigning a Name to a Name (line 370):
    # Getting the type of 'call_assignment_105210' (line 370)
    call_assignment_105210_105656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105210')
    # Assigning a type to the variable 'per' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 10), 'per', call_assignment_105210_105656)
    
    # Assigning a Call to a Name (line 370):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105660 = {}
    # Getting the type of 'call_assignment_105208' (line 370)
    call_assignment_105208_105657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105208', False)
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___105658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 4), call_assignment_105208_105657, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105661 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105658, *[int_105659], **kwargs_105660)
    
    # Assigning a type to the variable 'call_assignment_105211' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105211', getitem___call_result_105661)
    
    # Assigning a Name to a Name (line 370):
    # Getting the type of 'call_assignment_105211' (line 370)
    call_assignment_105211_105662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105211')
    # Assigning a type to the variable 'nper' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'nper', call_assignment_105211_105662)
    
    # Assigning a Call to a Name (line 370):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105666 = {}
    # Getting the type of 'call_assignment_105208' (line 370)
    call_assignment_105208_105663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105208', False)
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___105664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 4), call_assignment_105208_105663, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105667 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105664, *[int_105665], **kwargs_105666)
    
    # Assigning a type to the variable 'call_assignment_105212' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105212', getitem___call_result_105667)
    
    # Assigning a Name to a Name (line 370):
    # Getting the type of 'call_assignment_105212' (line 370)
    call_assignment_105212_105668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105212')
    # Assigning a type to the variable 'pv' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 21), 'pv', call_assignment_105212_105668)
    
    # Assigning a Call to a Name (line 370):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105672 = {}
    # Getting the type of 'call_assignment_105208' (line 370)
    call_assignment_105208_105669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105208', False)
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___105670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 4), call_assignment_105208_105669, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105673 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105670, *[int_105671], **kwargs_105672)
    
    # Assigning a type to the variable 'call_assignment_105213' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105213', getitem___call_result_105673)
    
    # Assigning a Name to a Name (line 370):
    # Getting the type of 'call_assignment_105213' (line 370)
    call_assignment_105213_105674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105213')
    # Assigning a type to the variable 'fv' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 25), 'fv', call_assignment_105213_105674)
    
    # Assigning a Call to a Name (line 370):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105678 = {}
    # Getting the type of 'call_assignment_105208' (line 370)
    call_assignment_105208_105675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105208', False)
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___105676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 4), call_assignment_105208_105675, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105679 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105676, *[int_105677], **kwargs_105678)
    
    # Assigning a type to the variable 'call_assignment_105214' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105214', getitem___call_result_105679)
    
    # Assigning a Name to a Name (line 370):
    # Getting the type of 'call_assignment_105214' (line 370)
    call_assignment_105214_105680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'call_assignment_105214')
    # Assigning a type to the variable 'when' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 29), 'when', call_assignment_105214_105680)
    
    # Assigning a Call to a Name (line 372):
    
    # Assigning a Call to a Name (line 372):
    
    # Call to pmt(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'rate' (line 372)
    rate_105682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'rate', False)
    # Getting the type of 'nper' (line 372)
    nper_105683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 26), 'nper', False)
    # Getting the type of 'pv' (line 372)
    pv_105684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 32), 'pv', False)
    # Getting the type of 'fv' (line 372)
    fv_105685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 36), 'fv', False)
    # Getting the type of 'when' (line 372)
    when_105686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 40), 'when', False)
    # Processing the call keyword arguments (line 372)
    kwargs_105687 = {}
    # Getting the type of 'pmt' (line 372)
    pmt_105681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'pmt', False)
    # Calling pmt(args, kwargs) (line 372)
    pmt_call_result_105688 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), pmt_105681, *[rate_105682, nper_105683, pv_105684, fv_105685, when_105686], **kwargs_105687)
    
    # Assigning a type to the variable 'total_pmt' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'total_pmt', pmt_call_result_105688)
    
    # Assigning a BinOp to a Name (line 373):
    
    # Assigning a BinOp to a Name (line 373):
    
    # Call to _rbl(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'rate' (line 373)
    rate_105690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'rate', False)
    # Getting the type of 'per' (line 373)
    per_105691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 'per', False)
    # Getting the type of 'total_pmt' (line 373)
    total_pmt_105692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 27), 'total_pmt', False)
    # Getting the type of 'pv' (line 373)
    pv_105693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 38), 'pv', False)
    # Getting the type of 'when' (line 373)
    when_105694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 42), 'when', False)
    # Processing the call keyword arguments (line 373)
    kwargs_105695 = {}
    # Getting the type of '_rbl' (line 373)
    _rbl_105689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 11), '_rbl', False)
    # Calling _rbl(args, kwargs) (line 373)
    _rbl_call_result_105696 = invoke(stypy.reporting.localization.Localization(__file__, 373, 11), _rbl_105689, *[rate_105690, per_105691, total_pmt_105692, pv_105693, when_105694], **kwargs_105695)
    
    # Getting the type of 'rate' (line 373)
    rate_105697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 48), 'rate')
    # Applying the binary operator '*' (line 373)
    result_mul_105698 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 11), '*', _rbl_call_result_105696, rate_105697)
    
    # Assigning a type to the variable 'ipmt' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'ipmt', result_mul_105698)
    
    
    # SSA begins for try-except statement (line 374)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 375):
    
    # Assigning a Call to a Name (line 375):
    
    # Call to where(...): (line 375)
    # Processing the call arguments (line 375)
    
    # Getting the type of 'when' (line 375)
    when_105701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 24), 'when', False)
    int_105702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 32), 'int')
    # Applying the binary operator '==' (line 375)
    result_eq_105703 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 24), '==', when_105701, int_105702)
    
    # Getting the type of 'ipmt' (line 375)
    ipmt_105704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 35), 'ipmt', False)
    int_105705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 41), 'int')
    # Getting the type of 'rate' (line 375)
    rate_105706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 45), 'rate', False)
    # Applying the binary operator '+' (line 375)
    result_add_105707 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 41), '+', int_105705, rate_105706)
    
    # Applying the binary operator 'div' (line 375)
    result_div_105708 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 35), 'div', ipmt_105704, result_add_105707)
    
    # Getting the type of 'ipmt' (line 375)
    ipmt_105709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 52), 'ipmt', False)
    # Processing the call keyword arguments (line 375)
    kwargs_105710 = {}
    # Getting the type of 'np' (line 375)
    np_105699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'np', False)
    # Obtaining the member 'where' of a type (line 375)
    where_105700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 15), np_105699, 'where')
    # Calling where(args, kwargs) (line 375)
    where_call_result_105711 = invoke(stypy.reporting.localization.Localization(__file__, 375, 15), where_105700, *[result_eq_105703, result_div_105708, ipmt_105709], **kwargs_105710)
    
    # Assigning a type to the variable 'ipmt' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'ipmt', where_call_result_105711)
    
    # Assigning a Call to a Name (line 376):
    
    # Assigning a Call to a Name (line 376):
    
    # Call to where(...): (line 376)
    # Processing the call arguments (line 376)
    
    # Call to logical_and(...): (line 376)
    # Processing the call arguments (line 376)
    
    # Getting the type of 'when' (line 376)
    when_105716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 39), 'when', False)
    int_105717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 47), 'int')
    # Applying the binary operator '==' (line 376)
    result_eq_105718 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 39), '==', when_105716, int_105717)
    
    
    # Getting the type of 'per' (line 376)
    per_105719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 50), 'per', False)
    int_105720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 57), 'int')
    # Applying the binary operator '==' (line 376)
    result_eq_105721 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 50), '==', per_105719, int_105720)
    
    # Processing the call keyword arguments (line 376)
    kwargs_105722 = {}
    # Getting the type of 'np' (line 376)
    np_105714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 24), 'np', False)
    # Obtaining the member 'logical_and' of a type (line 376)
    logical_and_105715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 24), np_105714, 'logical_and')
    # Calling logical_and(args, kwargs) (line 376)
    logical_and_call_result_105723 = invoke(stypy.reporting.localization.Localization(__file__, 376, 24), logical_and_105715, *[result_eq_105718, result_eq_105721], **kwargs_105722)
    
    float_105724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 61), 'float')
    # Getting the type of 'ipmt' (line 376)
    ipmt_105725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 66), 'ipmt', False)
    # Processing the call keyword arguments (line 376)
    kwargs_105726 = {}
    # Getting the type of 'np' (line 376)
    np_105712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'np', False)
    # Obtaining the member 'where' of a type (line 376)
    where_105713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 15), np_105712, 'where')
    # Calling where(args, kwargs) (line 376)
    where_call_result_105727 = invoke(stypy.reporting.localization.Localization(__file__, 376, 15), where_105713, *[logical_and_call_result_105723, float_105724, ipmt_105725], **kwargs_105726)
    
    # Assigning a type to the variable 'ipmt' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'ipmt', where_call_result_105727)
    # SSA branch for the except part of a try statement (line 374)
    # SSA branch for the except 'IndexError' branch of a try statement (line 374)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 374)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ipmt' (line 379)
    ipmt_105728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'ipmt')
    # Assigning a type to the variable 'stypy_return_type' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type', ipmt_105728)
    
    # ################# End of 'ipmt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ipmt' in the type store
    # Getting the type of 'stypy_return_type' (line 286)
    stypy_return_type_105729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105729)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ipmt'
    return stypy_return_type_105729

# Assigning a type to the variable 'ipmt' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'ipmt', ipmt)

@norecursion
def _rbl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_rbl'
    module_type_store = module_type_store.open_function_context('_rbl', 381, 0, False)
    
    # Passed parameters checking function
    _rbl.stypy_localization = localization
    _rbl.stypy_type_of_self = None
    _rbl.stypy_type_store = module_type_store
    _rbl.stypy_function_name = '_rbl'
    _rbl.stypy_param_names_list = ['rate', 'per', 'pmt', 'pv', 'when']
    _rbl.stypy_varargs_param_name = None
    _rbl.stypy_kwargs_param_name = None
    _rbl.stypy_call_defaults = defaults
    _rbl.stypy_call_varargs = varargs
    _rbl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_rbl', ['rate', 'per', 'pmt', 'pv', 'when'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_rbl', localization, ['rate', 'per', 'pmt', 'pv', 'when'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_rbl(...)' code ##################

    str_105730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, (-1)), 'str', "\n    This function is here to simply have a different name for the 'fv'\n    function to not interfere with the 'fv' keyword argument within the 'ipmt'\n    function.  It is the 'remaining balance on loan' which might be useful as\n    it's own function, but is easily calculated with the 'fv' function.\n    ")
    
    # Call to fv(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'rate' (line 388)
    rate_105732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 14), 'rate', False)
    # Getting the type of 'per' (line 388)
    per_105733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 21), 'per', False)
    int_105734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 27), 'int')
    # Applying the binary operator '-' (line 388)
    result_sub_105735 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 21), '-', per_105733, int_105734)
    
    # Getting the type of 'pmt' (line 388)
    pmt_105736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 31), 'pmt', False)
    # Getting the type of 'pv' (line 388)
    pv_105737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 36), 'pv', False)
    # Getting the type of 'when' (line 388)
    when_105738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 40), 'when', False)
    # Processing the call keyword arguments (line 388)
    kwargs_105739 = {}
    # Getting the type of 'fv' (line 388)
    fv_105731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), 'fv', False)
    # Calling fv(args, kwargs) (line 388)
    fv_call_result_105740 = invoke(stypy.reporting.localization.Localization(__file__, 388, 11), fv_105731, *[rate_105732, result_sub_105735, pmt_105736, pv_105737, when_105738], **kwargs_105739)
    
    # Assigning a type to the variable 'stypy_return_type' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type', fv_call_result_105740)
    
    # ################# End of '_rbl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_rbl' in the type store
    # Getting the type of 'stypy_return_type' (line 381)
    stypy_return_type_105741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105741)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_rbl'
    return stypy_return_type_105741

# Assigning a type to the variable '_rbl' (line 381)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 0), '_rbl', _rbl)

@norecursion
def ppmt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_105742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 33), 'float')
    str_105743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 43), 'str', 'end')
    defaults = [float_105742, str_105743]
    # Create a new context for function 'ppmt'
    module_type_store = module_type_store.open_function_context('ppmt', 390, 0, False)
    
    # Passed parameters checking function
    ppmt.stypy_localization = localization
    ppmt.stypy_type_of_self = None
    ppmt.stypy_type_store = module_type_store
    ppmt.stypy_function_name = 'ppmt'
    ppmt.stypy_param_names_list = ['rate', 'per', 'nper', 'pv', 'fv', 'when']
    ppmt.stypy_varargs_param_name = None
    ppmt.stypy_kwargs_param_name = None
    ppmt.stypy_call_defaults = defaults
    ppmt.stypy_call_varargs = varargs
    ppmt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ppmt', ['rate', 'per', 'nper', 'pv', 'fv', 'when'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ppmt', localization, ['rate', 'per', 'nper', 'pv', 'fv', 'when'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ppmt(...)' code ##################

    str_105744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, (-1)), 'str', "\n    Compute the payment against loan principal.\n\n    Parameters\n    ----------\n    rate : array_like\n        Rate of interest (per period)\n    per : array_like, int\n        Amount paid against the loan changes.  The `per` is the period of\n        interest.\n    nper : array_like\n        Number of compounding periods\n    pv : array_like\n        Present value\n    fv : array_like, optional\n        Future value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}\n        When payments are due ('begin' (1) or 'end' (0))\n\n    See Also\n    --------\n    pmt, pv, ipmt\n\n    ")
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to pmt(...): (line 415)
    # Processing the call arguments (line 415)
    # Getting the type of 'rate' (line 415)
    rate_105746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'rate', False)
    # Getting the type of 'nper' (line 415)
    nper_105747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 22), 'nper', False)
    # Getting the type of 'pv' (line 415)
    pv_105748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 28), 'pv', False)
    # Getting the type of 'fv' (line 415)
    fv_105749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 32), 'fv', False)
    # Getting the type of 'when' (line 415)
    when_105750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 36), 'when', False)
    # Processing the call keyword arguments (line 415)
    kwargs_105751 = {}
    # Getting the type of 'pmt' (line 415)
    pmt_105745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'pmt', False)
    # Calling pmt(args, kwargs) (line 415)
    pmt_call_result_105752 = invoke(stypy.reporting.localization.Localization(__file__, 415, 12), pmt_105745, *[rate_105746, nper_105747, pv_105748, fv_105749, when_105750], **kwargs_105751)
    
    # Assigning a type to the variable 'total' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'total', pmt_call_result_105752)
    # Getting the type of 'total' (line 416)
    total_105753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 11), 'total')
    
    # Call to ipmt(...): (line 416)
    # Processing the call arguments (line 416)
    # Getting the type of 'rate' (line 416)
    rate_105755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 24), 'rate', False)
    # Getting the type of 'per' (line 416)
    per_105756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'per', False)
    # Getting the type of 'nper' (line 416)
    nper_105757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 35), 'nper', False)
    # Getting the type of 'pv' (line 416)
    pv_105758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 41), 'pv', False)
    # Getting the type of 'fv' (line 416)
    fv_105759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 45), 'fv', False)
    # Getting the type of 'when' (line 416)
    when_105760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 49), 'when', False)
    # Processing the call keyword arguments (line 416)
    kwargs_105761 = {}
    # Getting the type of 'ipmt' (line 416)
    ipmt_105754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'ipmt', False)
    # Calling ipmt(args, kwargs) (line 416)
    ipmt_call_result_105762 = invoke(stypy.reporting.localization.Localization(__file__, 416, 19), ipmt_105754, *[rate_105755, per_105756, nper_105757, pv_105758, fv_105759, when_105760], **kwargs_105761)
    
    # Applying the binary operator '-' (line 416)
    result_sub_105763 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), '-', total_105753, ipmt_call_result_105762)
    
    # Assigning a type to the variable 'stypy_return_type' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'stypy_return_type', result_sub_105763)
    
    # ################# End of 'ppmt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ppmt' in the type store
    # Getting the type of 'stypy_return_type' (line 390)
    stypy_return_type_105764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105764)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ppmt'
    return stypy_return_type_105764

# Assigning a type to the variable 'ppmt' (line 390)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 0), 'ppmt', ppmt)

@norecursion
def pv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_105765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 27), 'float')
    str_105766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 37), 'str', 'end')
    defaults = [float_105765, str_105766]
    # Create a new context for function 'pv'
    module_type_store = module_type_store.open_function_context('pv', 418, 0, False)
    
    # Passed parameters checking function
    pv.stypy_localization = localization
    pv.stypy_type_of_self = None
    pv.stypy_type_store = module_type_store
    pv.stypy_function_name = 'pv'
    pv.stypy_param_names_list = ['rate', 'nper', 'pmt', 'fv', 'when']
    pv.stypy_varargs_param_name = None
    pv.stypy_kwargs_param_name = None
    pv.stypy_call_defaults = defaults
    pv.stypy_call_varargs = varargs
    pv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pv', ['rate', 'nper', 'pmt', 'fv', 'when'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pv', localization, ['rate', 'nper', 'pmt', 'fv', 'when'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pv(...)' code ##################

    str_105767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, (-1)), 'str', '\n    Compute the present value.\n\n    Given:\n     * a future value, `fv`\n     * an interest `rate` compounded once per period, of which\n       there are\n     * `nper` total\n     * a (fixed) payment, `pmt`, paid either\n     * at the beginning (`when` = {\'begin\', 1}) or the end\n       (`when` = {\'end\', 0}) of each period\n\n    Return:\n       the value now\n\n    Parameters\n    ----------\n    rate : array_like\n        Rate of interest (per period)\n    nper : array_like\n        Number of compounding periods\n    pmt : array_like\n        Payment\n    fv : array_like, optional\n        Future value\n    when : {{\'begin\', 1}, {\'end\', 0}}, {string, int}, optional\n        When payments are due (\'begin\' (1) or \'end\' (0))\n\n    Returns\n    -------\n    out : ndarray, float\n        Present value of a series of payments or investments.\n\n    Notes\n    -----\n    The present value is computed by solving the equation::\n\n     fv +\n     pv*(1 + rate)**nper +\n     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) = 0\n\n    or, when ``rate = 0``::\n\n     fv + pv + pmt * nper = 0\n\n    for `pv`, which is then returned.\n\n    References\n    ----------\n    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).\n       Open Document Format for Office Applications (OpenDocument)v1.2,\n       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,\n       Pre-Draft 12. Organization for the Advancement of Structured Information\n       Standards (OASIS). Billerica, MA, USA. [ODT Document].\n       Available:\n       http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula\n       OpenDocument-formula-20090508.odt\n\n    Examples\n    --------\n    What is the present value (e.g., the initial investment)\n    of an investment that needs to total $15692.93\n    after 10 years of saving $100 every month?  Assume the\n    interest rate is 5% (annually) compounded monthly.\n\n    >>> np.pv(0.05/12, 10*12, -100, 15692.93)\n    -100.00067131625819\n\n    By convention, the negative sign represents cash flow out\n    (i.e., money not available today).  Thus, to end up with\n    $15,692.93 in 10 years saving $100 a month at 5% annual\n    interest, one\'s initial deposit should also be $100.\n\n    If any input is array_like, ``pv`` returns an array of equal shape.\n    Let\'s compare different interest rates in the example above:\n\n    >>> a = np.array((0.05, 0.04, 0.03))/12\n    >>> np.pv(a, 10*12, -100, 15692.93)\n    array([ -100.00067132,  -649.26771385, -1273.78633713])\n\n    So, to end up with the same $15692.93 under the same $100 per month\n    "savings plan," for annual interest rates of 4% and 3%, one would\n    need initial investments of $649.27 and $1273.79, respectively.\n\n    ')
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to _convert_when(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'when' (line 504)
    when_105769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 25), 'when', False)
    # Processing the call keyword arguments (line 504)
    kwargs_105770 = {}
    # Getting the type of '_convert_when' (line 504)
    _convert_when_105768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 11), '_convert_when', False)
    # Calling _convert_when(args, kwargs) (line 504)
    _convert_when_call_result_105771 = invoke(stypy.reporting.localization.Localization(__file__, 504, 11), _convert_when_105768, *[when_105769], **kwargs_105770)
    
    # Assigning a type to the variable 'when' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'when', _convert_when_call_result_105771)
    
    # Assigning a Call to a Tuple (line 505):
    
    # Assigning a Call to a Name:
    
    # Call to map(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'np' (line 505)
    np_105773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 38), 'np', False)
    # Obtaining the member 'asarray' of a type (line 505)
    asarray_105774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 38), np_105773, 'asarray')
    
    # Obtaining an instance of the builtin type 'list' (line 505)
    list_105775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 505)
    # Adding element type (line 505)
    # Getting the type of 'rate' (line 505)
    rate_105776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 51), 'rate', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 50), list_105775, rate_105776)
    # Adding element type (line 505)
    # Getting the type of 'nper' (line 505)
    nper_105777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 57), 'nper', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 50), list_105775, nper_105777)
    # Adding element type (line 505)
    # Getting the type of 'pmt' (line 505)
    pmt_105778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 63), 'pmt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 50), list_105775, pmt_105778)
    # Adding element type (line 505)
    # Getting the type of 'fv' (line 505)
    fv_105779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 68), 'fv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 50), list_105775, fv_105779)
    # Adding element type (line 505)
    # Getting the type of 'when' (line 505)
    when_105780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 72), 'when', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 50), list_105775, when_105780)
    
    # Processing the call keyword arguments (line 505)
    kwargs_105781 = {}
    # Getting the type of 'map' (line 505)
    map_105772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 34), 'map', False)
    # Calling map(args, kwargs) (line 505)
    map_call_result_105782 = invoke(stypy.reporting.localization.Localization(__file__, 505, 34), map_105772, *[asarray_105774, list_105775], **kwargs_105781)
    
    # Assigning a type to the variable 'call_assignment_105215' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105215', map_call_result_105782)
    
    # Assigning a Call to a Name (line 505):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105786 = {}
    # Getting the type of 'call_assignment_105215' (line 505)
    call_assignment_105215_105783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105215', False)
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___105784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 4), call_assignment_105215_105783, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105787 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105784, *[int_105785], **kwargs_105786)
    
    # Assigning a type to the variable 'call_assignment_105216' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105216', getitem___call_result_105787)
    
    # Assigning a Name to a Name (line 505):
    # Getting the type of 'call_assignment_105216' (line 505)
    call_assignment_105216_105788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105216')
    # Assigning a type to the variable 'rate' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 5), 'rate', call_assignment_105216_105788)
    
    # Assigning a Call to a Name (line 505):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105792 = {}
    # Getting the type of 'call_assignment_105215' (line 505)
    call_assignment_105215_105789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105215', False)
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___105790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 4), call_assignment_105215_105789, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105793 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105790, *[int_105791], **kwargs_105792)
    
    # Assigning a type to the variable 'call_assignment_105217' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105217', getitem___call_result_105793)
    
    # Assigning a Name to a Name (line 505):
    # Getting the type of 'call_assignment_105217' (line 505)
    call_assignment_105217_105794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105217')
    # Assigning a type to the variable 'nper' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 11), 'nper', call_assignment_105217_105794)
    
    # Assigning a Call to a Name (line 505):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105798 = {}
    # Getting the type of 'call_assignment_105215' (line 505)
    call_assignment_105215_105795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105215', False)
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___105796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 4), call_assignment_105215_105795, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105799 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105796, *[int_105797], **kwargs_105798)
    
    # Assigning a type to the variable 'call_assignment_105218' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105218', getitem___call_result_105799)
    
    # Assigning a Name to a Name (line 505):
    # Getting the type of 'call_assignment_105218' (line 505)
    call_assignment_105218_105800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105218')
    # Assigning a type to the variable 'pmt' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 17), 'pmt', call_assignment_105218_105800)
    
    # Assigning a Call to a Name (line 505):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105804 = {}
    # Getting the type of 'call_assignment_105215' (line 505)
    call_assignment_105215_105801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105215', False)
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___105802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 4), call_assignment_105215_105801, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105805 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105802, *[int_105803], **kwargs_105804)
    
    # Assigning a type to the variable 'call_assignment_105219' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105219', getitem___call_result_105805)
    
    # Assigning a Name to a Name (line 505):
    # Getting the type of 'call_assignment_105219' (line 505)
    call_assignment_105219_105806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105219')
    # Assigning a type to the variable 'fv' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 22), 'fv', call_assignment_105219_105806)
    
    # Assigning a Call to a Name (line 505):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105810 = {}
    # Getting the type of 'call_assignment_105215' (line 505)
    call_assignment_105215_105807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105215', False)
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___105808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 4), call_assignment_105215_105807, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105811 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105808, *[int_105809], **kwargs_105810)
    
    # Assigning a type to the variable 'call_assignment_105220' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105220', getitem___call_result_105811)
    
    # Assigning a Name to a Name (line 505):
    # Getting the type of 'call_assignment_105220' (line 505)
    call_assignment_105220_105812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'call_assignment_105220')
    # Assigning a type to the variable 'when' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 26), 'when', call_assignment_105220_105812)
    
    # Assigning a BinOp to a Name (line 506):
    
    # Assigning a BinOp to a Name (line 506):
    int_105813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 12), 'int')
    # Getting the type of 'rate' (line 506)
    rate_105814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 14), 'rate')
    # Applying the binary operator '+' (line 506)
    result_add_105815 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 12), '+', int_105813, rate_105814)
    
    # Getting the type of 'nper' (line 506)
    nper_105816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 21), 'nper')
    # Applying the binary operator '**' (line 506)
    result_pow_105817 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 11), '**', result_add_105815, nper_105816)
    
    # Assigning a type to the variable 'temp' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'temp', result_pow_105817)
    
    # Assigning a Call to a Name (line 507):
    
    # Assigning a Call to a Name (line 507):
    
    # Call to broadcast(...): (line 507)
    # Processing the call arguments (line 507)
    # Getting the type of 'rate' (line 507)
    rate_105820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 25), 'rate', False)
    # Getting the type of 'nper' (line 507)
    nper_105821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 31), 'nper', False)
    # Getting the type of 'pmt' (line 507)
    pmt_105822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 37), 'pmt', False)
    # Getting the type of 'fv' (line 507)
    fv_105823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 42), 'fv', False)
    # Getting the type of 'when' (line 507)
    when_105824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 46), 'when', False)
    # Processing the call keyword arguments (line 507)
    kwargs_105825 = {}
    # Getting the type of 'np' (line 507)
    np_105818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'np', False)
    # Obtaining the member 'broadcast' of a type (line 507)
    broadcast_105819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 12), np_105818, 'broadcast')
    # Calling broadcast(args, kwargs) (line 507)
    broadcast_call_result_105826 = invoke(stypy.reporting.localization.Localization(__file__, 507, 12), broadcast_105819, *[rate_105820, nper_105821, pmt_105822, fv_105823, when_105824], **kwargs_105825)
    
    # Assigning a type to the variable 'miter' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'miter', broadcast_call_result_105826)
    
    # Assigning a Call to a Name (line 508):
    
    # Assigning a Call to a Name (line 508):
    
    # Call to zeros(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'miter' (line 508)
    miter_105829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 19), 'miter', False)
    # Obtaining the member 'shape' of a type (line 508)
    shape_105830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 19), miter_105829, 'shape')
    # Processing the call keyword arguments (line 508)
    kwargs_105831 = {}
    # Getting the type of 'np' (line 508)
    np_105827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 508)
    zeros_105828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 10), np_105827, 'zeros')
    # Calling zeros(args, kwargs) (line 508)
    zeros_call_result_105832 = invoke(stypy.reporting.localization.Localization(__file__, 508, 10), zeros_105828, *[shape_105830], **kwargs_105831)
    
    # Assigning a type to the variable 'zer' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'zer', zeros_call_result_105832)
    
    # Assigning a Call to a Name (line 509):
    
    # Assigning a Call to a Name (line 509):
    
    # Call to where(...): (line 509)
    # Processing the call arguments (line 509)
    
    # Getting the type of 'rate' (line 509)
    rate_105835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 20), 'rate', False)
    # Getting the type of 'zer' (line 509)
    zer_105836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 28), 'zer', False)
    # Applying the binary operator '==' (line 509)
    result_eq_105837 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 20), '==', rate_105835, zer_105836)
    
    # Getting the type of 'nper' (line 509)
    nper_105838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 33), 'nper', False)
    # Getting the type of 'zer' (line 509)
    zer_105839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 38), 'zer', False)
    # Applying the binary operator '+' (line 509)
    result_add_105840 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 33), '+', nper_105838, zer_105839)
    
    int_105841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 44), 'int')
    # Getting the type of 'rate' (line 509)
    rate_105842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 46), 'rate', False)
    # Getting the type of 'when' (line 509)
    when_105843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 51), 'when', False)
    # Applying the binary operator '*' (line 509)
    result_mul_105844 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 46), '*', rate_105842, when_105843)
    
    # Applying the binary operator '+' (line 509)
    result_add_105845 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 44), '+', int_105841, result_mul_105844)
    
    # Getting the type of 'temp' (line 509)
    temp_105846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 58), 'temp', False)
    int_105847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 63), 'int')
    # Applying the binary operator '-' (line 509)
    result_sub_105848 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 58), '-', temp_105846, int_105847)
    
    # Applying the binary operator '*' (line 509)
    result_mul_105849 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 43), '*', result_add_105845, result_sub_105848)
    
    # Getting the type of 'rate' (line 509)
    rate_105850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 66), 'rate', False)
    # Applying the binary operator 'div' (line 509)
    result_div_105851 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 65), 'div', result_mul_105849, rate_105850)
    
    # Getting the type of 'zer' (line 509)
    zer_105852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 71), 'zer', False)
    # Applying the binary operator '+' (line 509)
    result_add_105853 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 43), '+', result_div_105851, zer_105852)
    
    # Processing the call keyword arguments (line 509)
    kwargs_105854 = {}
    # Getting the type of 'np' (line 509)
    np_105833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'np', False)
    # Obtaining the member 'where' of a type (line 509)
    where_105834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 11), np_105833, 'where')
    # Calling where(args, kwargs) (line 509)
    where_call_result_105855 = invoke(stypy.reporting.localization.Localization(__file__, 509, 11), where_105834, *[result_eq_105837, result_add_105840, result_add_105853], **kwargs_105854)
    
    # Assigning a type to the variable 'fact' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'fact', where_call_result_105855)
    
    # Getting the type of 'fv' (line 510)
    fv_105856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 13), 'fv')
    # Getting the type of 'pmt' (line 510)
    pmt_105857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 18), 'pmt')
    # Getting the type of 'fact' (line 510)
    fact_105858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 22), 'fact')
    # Applying the binary operator '*' (line 510)
    result_mul_105859 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 18), '*', pmt_105857, fact_105858)
    
    # Applying the binary operator '+' (line 510)
    result_add_105860 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 13), '+', fv_105856, result_mul_105859)
    
    # Applying the 'usub' unary operator (line 510)
    result___neg___105861 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 11), 'usub', result_add_105860)
    
    # Getting the type of 'temp' (line 510)
    temp_105862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 28), 'temp')
    # Applying the binary operator 'div' (line 510)
    result_div_105863 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 11), 'div', result___neg___105861, temp_105862)
    
    # Assigning a type to the variable 'stypy_return_type' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'stypy_return_type', result_div_105863)
    
    # ################# End of 'pv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pv' in the type store
    # Getting the type of 'stypy_return_type' (line 418)
    stypy_return_type_105864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105864)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pv'
    return stypy_return_type_105864

# Assigning a type to the variable 'pv' (line 418)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), 'pv', pv)

@norecursion
def _g_div_gp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_g_div_gp'
    module_type_store = module_type_store.open_function_context('_g_div_gp', 517, 0, False)
    
    # Passed parameters checking function
    _g_div_gp.stypy_localization = localization
    _g_div_gp.stypy_type_of_self = None
    _g_div_gp.stypy_type_store = module_type_store
    _g_div_gp.stypy_function_name = '_g_div_gp'
    _g_div_gp.stypy_param_names_list = ['r', 'n', 'p', 'x', 'y', 'w']
    _g_div_gp.stypy_varargs_param_name = None
    _g_div_gp.stypy_kwargs_param_name = None
    _g_div_gp.stypy_call_defaults = defaults
    _g_div_gp.stypy_call_varargs = varargs
    _g_div_gp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_g_div_gp', ['r', 'n', 'p', 'x', 'y', 'w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_g_div_gp', localization, ['r', 'n', 'p', 'x', 'y', 'w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_g_div_gp(...)' code ##################

    
    # Assigning a BinOp to a Name (line 518):
    
    # Assigning a BinOp to a Name (line 518):
    # Getting the type of 'r' (line 518)
    r_105865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 10), 'r')
    int_105866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 12), 'int')
    # Applying the binary operator '+' (line 518)
    result_add_105867 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 10), '+', r_105865, int_105866)
    
    # Getting the type of 'n' (line 518)
    n_105868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'n')
    # Applying the binary operator '**' (line 518)
    result_pow_105869 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 9), '**', result_add_105867, n_105868)
    
    # Assigning a type to the variable 't1' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 't1', result_pow_105869)
    
    # Assigning a BinOp to a Name (line 519):
    
    # Assigning a BinOp to a Name (line 519):
    # Getting the type of 'r' (line 519)
    r_105870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 10), 'r')
    int_105871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 12), 'int')
    # Applying the binary operator '+' (line 519)
    result_add_105872 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 10), '+', r_105870, int_105871)
    
    # Getting the type of 'n' (line 519)
    n_105873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 17), 'n')
    int_105874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 19), 'int')
    # Applying the binary operator '-' (line 519)
    result_sub_105875 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 17), '-', n_105873, int_105874)
    
    # Applying the binary operator '**' (line 519)
    result_pow_105876 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 9), '**', result_add_105872, result_sub_105875)
    
    # Assigning a type to the variable 't2' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 't2', result_pow_105876)
    # Getting the type of 'y' (line 520)
    y_105877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 13), 'y')
    # Getting the type of 't1' (line 520)
    t1_105878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 't1')
    # Getting the type of 'x' (line 520)
    x_105879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 20), 'x')
    # Applying the binary operator '*' (line 520)
    result_mul_105880 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 17), '*', t1_105878, x_105879)
    
    # Applying the binary operator '+' (line 520)
    result_add_105881 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 13), '+', y_105877, result_mul_105880)
    
    # Getting the type of 'p' (line 520)
    p_105882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 24), 'p')
    # Getting the type of 't1' (line 520)
    t1_105883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 27), 't1')
    int_105884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 32), 'int')
    # Applying the binary operator '-' (line 520)
    result_sub_105885 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 27), '-', t1_105883, int_105884)
    
    # Applying the binary operator '*' (line 520)
    result_mul_105886 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 24), '*', p_105882, result_sub_105885)
    
    # Getting the type of 'r' (line 520)
    r_105887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 36), 'r')
    # Getting the type of 'w' (line 520)
    w_105888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 38), 'w')
    # Applying the binary operator '*' (line 520)
    result_mul_105889 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 36), '*', r_105887, w_105888)
    
    int_105890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 42), 'int')
    # Applying the binary operator '+' (line 520)
    result_add_105891 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 36), '+', result_mul_105889, int_105890)
    
    # Applying the binary operator '*' (line 520)
    result_mul_105892 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 34), '*', result_mul_105886, result_add_105891)
    
    # Getting the type of 'r' (line 520)
    r_105893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 45), 'r')
    # Applying the binary operator 'div' (line 520)
    result_div_105894 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 44), 'div', result_mul_105892, r_105893)
    
    # Applying the binary operator '+' (line 520)
    result_add_105895 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 22), '+', result_add_105881, result_div_105894)
    
    # Getting the type of 'n' (line 521)
    n_105896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 17), 'n')
    # Getting the type of 't2' (line 521)
    t2_105897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 19), 't2')
    # Applying the binary operator '*' (line 521)
    result_mul_105898 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 17), '*', n_105896, t2_105897)
    
    # Getting the type of 'x' (line 521)
    x_105899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 22), 'x')
    # Applying the binary operator '*' (line 521)
    result_mul_105900 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 21), '*', result_mul_105898, x_105899)
    
    # Getting the type of 'p' (line 521)
    p_105901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 26), 'p')
    # Getting the type of 't1' (line 521)
    t1_105902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 29), 't1')
    int_105903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 34), 'int')
    # Applying the binary operator '-' (line 521)
    result_sub_105904 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 29), '-', t1_105902, int_105903)
    
    # Applying the binary operator '*' (line 521)
    result_mul_105905 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 26), '*', p_105901, result_sub_105904)
    
    # Getting the type of 'r' (line 521)
    r_105906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 38), 'r')
    # Getting the type of 'w' (line 521)
    w_105907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 40), 'w')
    # Applying the binary operator '*' (line 521)
    result_mul_105908 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 38), '*', r_105906, w_105907)
    
    int_105909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 44), 'int')
    # Applying the binary operator '+' (line 521)
    result_add_105910 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 38), '+', result_mul_105908, int_105909)
    
    # Applying the binary operator '*' (line 521)
    result_mul_105911 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 36), '*', result_mul_105905, result_add_105910)
    
    # Getting the type of 'r' (line 521)
    r_105912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 48), 'r')
    int_105913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 51), 'int')
    # Applying the binary operator '**' (line 521)
    result_pow_105914 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 48), '**', r_105912, int_105913)
    
    # Applying the binary operator 'div' (line 521)
    result_div_105915 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 46), 'div', result_mul_105911, result_pow_105914)
    
    # Applying the binary operator '-' (line 521)
    result_sub_105916 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 17), '-', result_mul_105900, result_div_105915)
    
    # Getting the type of 'n' (line 521)
    n_105917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 56), 'n')
    # Getting the type of 'p' (line 521)
    p_105918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 58), 'p')
    # Applying the binary operator '*' (line 521)
    result_mul_105919 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 56), '*', n_105917, p_105918)
    
    # Getting the type of 't2' (line 521)
    t2_105920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 60), 't2')
    # Applying the binary operator '*' (line 521)
    result_mul_105921 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 59), '*', result_mul_105919, t2_105920)
    
    # Getting the type of 'r' (line 521)
    r_105922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 64), 'r')
    # Getting the type of 'w' (line 521)
    w_105923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 66), 'w')
    # Applying the binary operator '*' (line 521)
    result_mul_105924 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 64), '*', r_105922, w_105923)
    
    int_105925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 70), 'int')
    # Applying the binary operator '+' (line 521)
    result_add_105926 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 64), '+', result_mul_105924, int_105925)
    
    # Applying the binary operator '*' (line 521)
    result_mul_105927 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 62), '*', result_mul_105921, result_add_105926)
    
    # Getting the type of 'r' (line 521)
    r_105928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 73), 'r')
    # Applying the binary operator 'div' (line 521)
    result_div_105929 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 72), 'div', result_mul_105927, r_105928)
    
    # Applying the binary operator '+' (line 521)
    result_add_105930 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 54), '+', result_sub_105916, result_div_105929)
    
    # Getting the type of 'p' (line 522)
    p_105931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 17), 'p')
    # Getting the type of 't1' (line 522)
    t1_105932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 20), 't1')
    int_105933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 25), 'int')
    # Applying the binary operator '-' (line 522)
    result_sub_105934 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 20), '-', t1_105932, int_105933)
    
    # Applying the binary operator '*' (line 522)
    result_mul_105935 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 17), '*', p_105931, result_sub_105934)
    
    # Getting the type of 'w' (line 522)
    w_105936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 28), 'w')
    # Applying the binary operator '*' (line 522)
    result_mul_105937 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 27), '*', result_mul_105935, w_105936)
    
    # Getting the type of 'r' (line 522)
    r_105938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 30), 'r')
    # Applying the binary operator 'div' (line 522)
    result_div_105939 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 29), 'div', result_mul_105937, r_105938)
    
    # Applying the binary operator '+' (line 521)
    result_add_105940 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 75), '+', result_add_105930, result_div_105939)
    
    # Applying the binary operator 'div' (line 520)
    result_div_105941 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 12), 'div', result_add_105895, result_add_105940)
    
    # Assigning a type to the variable 'stypy_return_type' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'stypy_return_type', result_div_105941)
    
    # ################# End of '_g_div_gp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_g_div_gp' in the type store
    # Getting the type of 'stypy_return_type' (line 517)
    stypy_return_type_105942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105942)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_g_div_gp'
    return stypy_return_type_105942

# Assigning a type to the variable '_g_div_gp' (line 517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), '_g_div_gp', _g_div_gp)

@norecursion
def rate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_105943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 33), 'str', 'end')
    float_105944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 46), 'float')
    float_105945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 56), 'float')
    int_105946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 70), 'int')
    defaults = [str_105943, float_105944, float_105945, int_105946]
    # Create a new context for function 'rate'
    module_type_store = module_type_store.open_function_context('rate', 531, 0, False)
    
    # Passed parameters checking function
    rate.stypy_localization = localization
    rate.stypy_type_of_self = None
    rate.stypy_type_store = module_type_store
    rate.stypy_function_name = 'rate'
    rate.stypy_param_names_list = ['nper', 'pmt', 'pv', 'fv', 'when', 'guess', 'tol', 'maxiter']
    rate.stypy_varargs_param_name = None
    rate.stypy_kwargs_param_name = None
    rate.stypy_call_defaults = defaults
    rate.stypy_call_varargs = varargs
    rate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rate', ['nper', 'pmt', 'pv', 'fv', 'when', 'guess', 'tol', 'maxiter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rate', localization, ['nper', 'pmt', 'pv', 'fv', 'when', 'guess', 'tol', 'maxiter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rate(...)' code ##################

    str_105947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, (-1)), 'str', "\n    Compute the rate of interest per period.\n\n    Parameters\n    ----------\n    nper : array_like\n        Number of compounding periods\n    pmt : array_like\n        Payment\n    pv : array_like\n        Present value\n    fv : array_like\n        Future value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional\n        When payments are due ('begin' (1) or 'end' (0))\n    guess : float, optional\n        Starting guess for solving the rate of interest\n    tol : float, optional\n        Required tolerance for the solution\n    maxiter : int, optional\n        Maximum iterations in finding the solution\n\n    Notes\n    -----\n    The rate of interest is computed by iteratively solving the\n    (non-linear) equation::\n\n     fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) = 0\n\n    for ``rate``.\n\n    References\n    ----------\n    Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May). Open Document\n    Format for Office Applications (OpenDocument)v1.2, Part 2: Recalculated\n    Formula (OpenFormula) Format - Annotated Version, Pre-Draft 12.\n    Organization for the Advancement of Structured Information Standards\n    (OASIS). Billerica, MA, USA. [ODT Document]. Available:\n    http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula\n    OpenDocument-formula-20090508.odt\n\n    ")
    
    # Assigning a Call to a Name (line 574):
    
    # Assigning a Call to a Name (line 574):
    
    # Call to _convert_when(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'when' (line 574)
    when_105949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 25), 'when', False)
    # Processing the call keyword arguments (line 574)
    kwargs_105950 = {}
    # Getting the type of '_convert_when' (line 574)
    _convert_when_105948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 11), '_convert_when', False)
    # Calling _convert_when(args, kwargs) (line 574)
    _convert_when_call_result_105951 = invoke(stypy.reporting.localization.Localization(__file__, 574, 11), _convert_when_105948, *[when_105949], **kwargs_105950)
    
    # Assigning a type to the variable 'when' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'when', _convert_when_call_result_105951)
    
    # Assigning a Call to a Tuple (line 575):
    
    # Assigning a Call to a Name:
    
    # Call to map(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'np' (line 575)
    np_105953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 36), 'np', False)
    # Obtaining the member 'asarray' of a type (line 575)
    asarray_105954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 36), np_105953, 'asarray')
    
    # Obtaining an instance of the builtin type 'list' (line 575)
    list_105955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 575)
    # Adding element type (line 575)
    # Getting the type of 'nper' (line 575)
    nper_105956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 49), 'nper', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 48), list_105955, nper_105956)
    # Adding element type (line 575)
    # Getting the type of 'pmt' (line 575)
    pmt_105957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 55), 'pmt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 48), list_105955, pmt_105957)
    # Adding element type (line 575)
    # Getting the type of 'pv' (line 575)
    pv_105958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 60), 'pv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 48), list_105955, pv_105958)
    # Adding element type (line 575)
    # Getting the type of 'fv' (line 575)
    fv_105959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 64), 'fv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 48), list_105955, fv_105959)
    # Adding element type (line 575)
    # Getting the type of 'when' (line 575)
    when_105960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 68), 'when', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 48), list_105955, when_105960)
    
    # Processing the call keyword arguments (line 575)
    kwargs_105961 = {}
    # Getting the type of 'map' (line 575)
    map_105952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 32), 'map', False)
    # Calling map(args, kwargs) (line 575)
    map_call_result_105962 = invoke(stypy.reporting.localization.Localization(__file__, 575, 32), map_105952, *[asarray_105954, list_105955], **kwargs_105961)
    
    # Assigning a type to the variable 'call_assignment_105221' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105221', map_call_result_105962)
    
    # Assigning a Call to a Name (line 575):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105966 = {}
    # Getting the type of 'call_assignment_105221' (line 575)
    call_assignment_105221_105963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105221', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___105964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 4), call_assignment_105221_105963, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105967 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105964, *[int_105965], **kwargs_105966)
    
    # Assigning a type to the variable 'call_assignment_105222' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105222', getitem___call_result_105967)
    
    # Assigning a Name to a Name (line 575):
    # Getting the type of 'call_assignment_105222' (line 575)
    call_assignment_105222_105968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105222')
    # Assigning a type to the variable 'nper' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 5), 'nper', call_assignment_105222_105968)
    
    # Assigning a Call to a Name (line 575):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105972 = {}
    # Getting the type of 'call_assignment_105221' (line 575)
    call_assignment_105221_105969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105221', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___105970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 4), call_assignment_105221_105969, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105973 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105970, *[int_105971], **kwargs_105972)
    
    # Assigning a type to the variable 'call_assignment_105223' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105223', getitem___call_result_105973)
    
    # Assigning a Name to a Name (line 575):
    # Getting the type of 'call_assignment_105223' (line 575)
    call_assignment_105223_105974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105223')
    # Assigning a type to the variable 'pmt' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), 'pmt', call_assignment_105223_105974)
    
    # Assigning a Call to a Name (line 575):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105978 = {}
    # Getting the type of 'call_assignment_105221' (line 575)
    call_assignment_105221_105975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105221', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___105976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 4), call_assignment_105221_105975, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105979 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105976, *[int_105977], **kwargs_105978)
    
    # Assigning a type to the variable 'call_assignment_105224' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105224', getitem___call_result_105979)
    
    # Assigning a Name to a Name (line 575):
    # Getting the type of 'call_assignment_105224' (line 575)
    call_assignment_105224_105980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105224')
    # Assigning a type to the variable 'pv' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'pv', call_assignment_105224_105980)
    
    # Assigning a Call to a Name (line 575):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105984 = {}
    # Getting the type of 'call_assignment_105221' (line 575)
    call_assignment_105221_105981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105221', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___105982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 4), call_assignment_105221_105981, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105985 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105982, *[int_105983], **kwargs_105984)
    
    # Assigning a type to the variable 'call_assignment_105225' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105225', getitem___call_result_105985)
    
    # Assigning a Name to a Name (line 575):
    # Getting the type of 'call_assignment_105225' (line 575)
    call_assignment_105225_105986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105225')
    # Assigning a type to the variable 'fv' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 20), 'fv', call_assignment_105225_105986)
    
    # Assigning a Call to a Name (line 575):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_105989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 4), 'int')
    # Processing the call keyword arguments
    kwargs_105990 = {}
    # Getting the type of 'call_assignment_105221' (line 575)
    call_assignment_105221_105987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105221', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___105988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 4), call_assignment_105221_105987, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_105991 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___105988, *[int_105989], **kwargs_105990)
    
    # Assigning a type to the variable 'call_assignment_105226' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105226', getitem___call_result_105991)
    
    # Assigning a Name to a Name (line 575):
    # Getting the type of 'call_assignment_105226' (line 575)
    call_assignment_105226_105992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'call_assignment_105226')
    # Assigning a type to the variable 'when' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 24), 'when', call_assignment_105226_105992)
    
    # Assigning a Name to a Name (line 576):
    
    # Assigning a Name to a Name (line 576):
    # Getting the type of 'guess' (line 576)
    guess_105993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 9), 'guess')
    # Assigning a type to the variable 'rn' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'rn', guess_105993)
    
    # Assigning a Num to a Name (line 577):
    
    # Assigning a Num to a Name (line 577):
    int_105994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 11), 'int')
    # Assigning a type to the variable 'iter' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'iter', int_105994)
    
    # Assigning a Name to a Name (line 578):
    
    # Assigning a Name to a Name (line 578):
    # Getting the type of 'False' (line 578)
    False_105995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'False')
    # Assigning a type to the variable 'close' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'close', False_105995)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'iter' (line 579)
    iter_105996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 11), 'iter')
    # Getting the type of 'maxiter' (line 579)
    maxiter_105997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 18), 'maxiter')
    # Applying the binary operator '<' (line 579)
    result_lt_105998 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 11), '<', iter_105996, maxiter_105997)
    
    
    # Getting the type of 'close' (line 579)
    close_105999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 35), 'close')
    # Applying the 'not' unary operator (line 579)
    result_not__106000 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 31), 'not', close_105999)
    
    # Applying the binary operator 'and' (line 579)
    result_and_keyword_106001 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 10), 'and', result_lt_105998, result_not__106000)
    
    # Testing the type of an if condition (line 579)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 4), result_and_keyword_106001)
    # SSA begins for while statement (line 579)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 580):
    
    # Assigning a BinOp to a Name (line 580):
    # Getting the type of 'rn' (line 580)
    rn_106002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'rn')
    
    # Call to _g_div_gp(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 'rn' (line 580)
    rn_106004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 30), 'rn', False)
    # Getting the type of 'nper' (line 580)
    nper_106005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 34), 'nper', False)
    # Getting the type of 'pmt' (line 580)
    pmt_106006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 40), 'pmt', False)
    # Getting the type of 'pv' (line 580)
    pv_106007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 45), 'pv', False)
    # Getting the type of 'fv' (line 580)
    fv_106008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 49), 'fv', False)
    # Getting the type of 'when' (line 580)
    when_106009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 53), 'when', False)
    # Processing the call keyword arguments (line 580)
    kwargs_106010 = {}
    # Getting the type of '_g_div_gp' (line 580)
    _g_div_gp_106003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 20), '_g_div_gp', False)
    # Calling _g_div_gp(args, kwargs) (line 580)
    _g_div_gp_call_result_106011 = invoke(stypy.reporting.localization.Localization(__file__, 580, 20), _g_div_gp_106003, *[rn_106004, nper_106005, pmt_106006, pv_106007, fv_106008, when_106009], **kwargs_106010)
    
    # Applying the binary operator '-' (line 580)
    result_sub_106012 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 15), '-', rn_106002, _g_div_gp_call_result_106011)
    
    # Assigning a type to the variable 'rnp1' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'rnp1', result_sub_106012)
    
    # Assigning a Call to a Name (line 581):
    
    # Assigning a Call to a Name (line 581):
    
    # Call to abs(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'rnp1' (line 581)
    rnp1_106014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 19), 'rnp1', False)
    # Getting the type of 'rn' (line 581)
    rn_106015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 24), 'rn', False)
    # Applying the binary operator '-' (line 581)
    result_sub_106016 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 19), '-', rnp1_106014, rn_106015)
    
    # Processing the call keyword arguments (line 581)
    kwargs_106017 = {}
    # Getting the type of 'abs' (line 581)
    abs_106013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'abs', False)
    # Calling abs(args, kwargs) (line 581)
    abs_call_result_106018 = invoke(stypy.reporting.localization.Localization(__file__, 581, 15), abs_106013, *[result_sub_106016], **kwargs_106017)
    
    # Assigning a type to the variable 'diff' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'diff', abs_call_result_106018)
    
    # Assigning a Call to a Name (line 582):
    
    # Assigning a Call to a Name (line 582):
    
    # Call to all(...): (line 582)
    # Processing the call arguments (line 582)
    
    # Getting the type of 'diff' (line 582)
    diff_106021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 23), 'diff', False)
    # Getting the type of 'tol' (line 582)
    tol_106022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 30), 'tol', False)
    # Applying the binary operator '<' (line 582)
    result_lt_106023 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 23), '<', diff_106021, tol_106022)
    
    # Processing the call keyword arguments (line 582)
    kwargs_106024 = {}
    # Getting the type of 'np' (line 582)
    np_106019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 582)
    all_106020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 16), np_106019, 'all')
    # Calling all(args, kwargs) (line 582)
    all_call_result_106025 = invoke(stypy.reporting.localization.Localization(__file__, 582, 16), all_106020, *[result_lt_106023], **kwargs_106024)
    
    # Assigning a type to the variable 'close' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'close', all_call_result_106025)
    
    # Getting the type of 'iter' (line 583)
    iter_106026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'iter')
    int_106027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 16), 'int')
    # Applying the binary operator '+=' (line 583)
    result_iadd_106028 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 8), '+=', iter_106026, int_106027)
    # Assigning a type to the variable 'iter' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'iter', result_iadd_106028)
    
    
    # Assigning a Name to a Name (line 584):
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'rnp1' (line 584)
    rnp1_106029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 13), 'rnp1')
    # Assigning a type to the variable 'rn' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'rn', rnp1_106029)
    # SSA join for while statement (line 579)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'close' (line 585)
    close_106030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 11), 'close')
    # Applying the 'not' unary operator (line 585)
    result_not__106031 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 7), 'not', close_106030)
    
    # Testing the type of an if condition (line 585)
    if_condition_106032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 4), result_not__106031)
    # Assigning a type to the variable 'if_condition_106032' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'if_condition_106032', if_condition_106032)
    # SSA begins for if statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 587)
    np_106033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'np')
    # Obtaining the member 'nan' of a type (line 587)
    nan_106034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 15), np_106033, 'nan')
    # Getting the type of 'rn' (line 587)
    rn_106035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 24), 'rn')
    # Applying the binary operator '+' (line 587)
    result_add_106036 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 15), '+', nan_106034, rn_106035)
    
    # Assigning a type to the variable 'stypy_return_type' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'stypy_return_type', result_add_106036)
    # SSA branch for the else part of an if statement (line 585)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'rn' (line 589)
    rn_106037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 15), 'rn')
    # Assigning a type to the variable 'stypy_return_type' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'stypy_return_type', rn_106037)
    # SSA join for if statement (line 585)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'rate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rate' in the type store
    # Getting the type of 'stypy_return_type' (line 531)
    stypy_return_type_106038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106038)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rate'
    return stypy_return_type_106038

# Assigning a type to the variable 'rate' (line 531)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'rate', rate)

@norecursion
def irr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'irr'
    module_type_store = module_type_store.open_function_context('irr', 591, 0, False)
    
    # Passed parameters checking function
    irr.stypy_localization = localization
    irr.stypy_type_of_self = None
    irr.stypy_type_store = module_type_store
    irr.stypy_function_name = 'irr'
    irr.stypy_param_names_list = ['values']
    irr.stypy_varargs_param_name = None
    irr.stypy_kwargs_param_name = None
    irr.stypy_call_defaults = defaults
    irr.stypy_call_varargs = varargs
    irr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'irr', ['values'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'irr', localization, ['values'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'irr(...)' code ##################

    str_106039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, (-1)), 'str', '\n    Return the Internal Rate of Return (IRR).\n\n    This is the "average" periodically compounded rate of return\n    that gives a net present value of 0.0; for a more complete explanation,\n    see Notes below.\n\n    Parameters\n    ----------\n    values : array_like, shape(N,)\n        Input cash flows per time period.  By convention, net "deposits"\n        are negative and net "withdrawals" are positive.  Thus, for\n        example, at least the first element of `values`, which represents\n        the initial investment, will typically be negative.\n\n    Returns\n    -------\n    out : float\n        Internal Rate of Return for periodic input values.\n\n    Notes\n    -----\n    The IRR is perhaps best understood through an example (illustrated\n    using np.irr in the Examples section below).  Suppose one invests 100\n    units and then makes the following withdrawals at regular (fixed)\n    intervals: 39, 59, 55, 20.  Assuming the ending value is 0, one\'s 100\n    unit investment yields 173 units; however, due to the combination of\n    compounding and the periodic withdrawals, the "average" rate of return\n    is neither simply 0.73/4 nor (1.73)^0.25-1.  Rather, it is the solution\n    (for :math:`r`) of the equation:\n\n    .. math:: -100 + \\frac{39}{1+r} + \\frac{59}{(1+r)^2}\n     + \\frac{55}{(1+r)^3} + \\frac{20}{(1+r)^4} = 0\n\n    In general, for `values` :math:`= [v_0, v_1, ... v_M]`,\n    irr is the solution of the equation: [G]_\n\n    .. math:: \\sum_{t=0}^M{\\frac{v_t}{(1+irr)^{t}}} = 0\n\n    References\n    ----------\n    .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,\n       Addison-Wesley, 2003, pg. 348.\n\n    Examples\n    --------\n    >>> round(irr([-100, 39, 59, 55, 20]), 5)\n    0.28095\n    >>> round(irr([-100, 0, 0, 74]), 5)\n    -0.0955\n    >>> round(irr([-100, 100, 0, -7]), 5)\n    -0.0833\n    >>> round(irr([-100, 100, 0, 7]), 5)\n    0.06206\n    >>> round(irr([-5, 10.5, 1, -8, 1]), 5)\n    0.0886\n\n    (Compare with the Example given for numpy.lib.financial.npv)\n\n    ')
    
    # Assigning a Call to a Name (line 652):
    
    # Assigning a Call to a Name (line 652):
    
    # Call to roots(...): (line 652)
    # Processing the call arguments (line 652)
    
    # Obtaining the type of the subscript
    int_106042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 28), 'int')
    slice_106043 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 652, 19), None, None, int_106042)
    # Getting the type of 'values' (line 652)
    values_106044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 19), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 652)
    getitem___106045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 19), values_106044, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 652)
    subscript_call_result_106046 = invoke(stypy.reporting.localization.Localization(__file__, 652, 19), getitem___106045, slice_106043)
    
    # Processing the call keyword arguments (line 652)
    kwargs_106047 = {}
    # Getting the type of 'np' (line 652)
    np_106040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 10), 'np', False)
    # Obtaining the member 'roots' of a type (line 652)
    roots_106041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 10), np_106040, 'roots')
    # Calling roots(args, kwargs) (line 652)
    roots_call_result_106048 = invoke(stypy.reporting.localization.Localization(__file__, 652, 10), roots_106041, *[subscript_call_result_106046], **kwargs_106047)
    
    # Assigning a type to the variable 'res' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'res', roots_call_result_106048)
    
    # Assigning a BinOp to a Name (line 653):
    
    # Assigning a BinOp to a Name (line 653):
    
    # Getting the type of 'res' (line 653)
    res_106049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 12), 'res')
    # Obtaining the member 'imag' of a type (line 653)
    imag_106050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 12), res_106049, 'imag')
    int_106051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 24), 'int')
    # Applying the binary operator '==' (line 653)
    result_eq_106052 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 12), '==', imag_106050, int_106051)
    
    
    # Getting the type of 'res' (line 653)
    res_106053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 30), 'res')
    # Obtaining the member 'real' of a type (line 653)
    real_106054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 30), res_106053, 'real')
    int_106055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 41), 'int')
    # Applying the binary operator '>' (line 653)
    result_gt_106056 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 30), '>', real_106054, int_106055)
    
    # Applying the binary operator '&' (line 653)
    result_and__106057 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 11), '&', result_eq_106052, result_gt_106056)
    
    # Assigning a type to the variable 'mask' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'mask', result_and__106057)
    
    
    # Getting the type of 'res' (line 654)
    res_106058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 7), 'res')
    # Obtaining the member 'size' of a type (line 654)
    size_106059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 7), res_106058, 'size')
    int_106060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 19), 'int')
    # Applying the binary operator '==' (line 654)
    result_eq_106061 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 7), '==', size_106059, int_106060)
    
    # Testing the type of an if condition (line 654)
    if_condition_106062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 4), result_eq_106061)
    # Assigning a type to the variable 'if_condition_106062' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'if_condition_106062', if_condition_106062)
    # SSA begins for if statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 655)
    np_106063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'np')
    # Obtaining the member 'nan' of a type (line 655)
    nan_106064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), np_106063, 'nan')
    # Assigning a type to the variable 'stypy_return_type' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'stypy_return_type', nan_106064)
    # SSA join for if statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 656):
    
    # Assigning a Attribute to a Name (line 656):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 656)
    mask_106065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 14), 'mask')
    # Getting the type of 'res' (line 656)
    res_106066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 10), 'res')
    # Obtaining the member '__getitem__' of a type (line 656)
    getitem___106067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 10), res_106066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 656)
    subscript_call_result_106068 = invoke(stypy.reporting.localization.Localization(__file__, 656, 10), getitem___106067, mask_106065)
    
    # Obtaining the member 'real' of a type (line 656)
    real_106069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 10), subscript_call_result_106068, 'real')
    # Assigning a type to the variable 'res' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'res', real_106069)
    
    # Assigning a BinOp to a Name (line 659):
    
    # Assigning a BinOp to a Name (line 659):
    float_106070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 11), 'float')
    # Getting the type of 'res' (line 659)
    res_106071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 15), 'res')
    # Applying the binary operator 'div' (line 659)
    result_div_106072 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 11), 'div', float_106070, res_106071)
    
    int_106073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 21), 'int')
    # Applying the binary operator '-' (line 659)
    result_sub_106074 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 11), '-', result_div_106072, int_106073)
    
    # Assigning a type to the variable 'rate' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'rate', result_sub_106074)
    
    # Assigning a Call to a Name (line 660):
    
    # Assigning a Call to a Name (line 660):
    
    # Call to item(...): (line 660)
    # Processing the call arguments (line 660)
    
    # Call to argmin(...): (line 660)
    # Processing the call arguments (line 660)
    
    # Call to abs(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'rate' (line 660)
    rate_106081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 38), 'rate', False)
    # Processing the call keyword arguments (line 660)
    kwargs_106082 = {}
    # Getting the type of 'np' (line 660)
    np_106079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 31), 'np', False)
    # Obtaining the member 'abs' of a type (line 660)
    abs_106080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 31), np_106079, 'abs')
    # Calling abs(args, kwargs) (line 660)
    abs_call_result_106083 = invoke(stypy.reporting.localization.Localization(__file__, 660, 31), abs_106080, *[rate_106081], **kwargs_106082)
    
    # Processing the call keyword arguments (line 660)
    kwargs_106084 = {}
    # Getting the type of 'np' (line 660)
    np_106077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 21), 'np', False)
    # Obtaining the member 'argmin' of a type (line 660)
    argmin_106078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 21), np_106077, 'argmin')
    # Calling argmin(args, kwargs) (line 660)
    argmin_call_result_106085 = invoke(stypy.reporting.localization.Localization(__file__, 660, 21), argmin_106078, *[abs_call_result_106083], **kwargs_106084)
    
    # Processing the call keyword arguments (line 660)
    kwargs_106086 = {}
    # Getting the type of 'rate' (line 660)
    rate_106075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 11), 'rate', False)
    # Obtaining the member 'item' of a type (line 660)
    item_106076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 11), rate_106075, 'item')
    # Calling item(args, kwargs) (line 660)
    item_call_result_106087 = invoke(stypy.reporting.localization.Localization(__file__, 660, 11), item_106076, *[argmin_call_result_106085], **kwargs_106086)
    
    # Assigning a type to the variable 'rate' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'rate', item_call_result_106087)
    # Getting the type of 'rate' (line 661)
    rate_106088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 11), 'rate')
    # Assigning a type to the variable 'stypy_return_type' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'stypy_return_type', rate_106088)
    
    # ################# End of 'irr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'irr' in the type store
    # Getting the type of 'stypy_return_type' (line 591)
    stypy_return_type_106089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106089)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'irr'
    return stypy_return_type_106089

# Assigning a type to the variable 'irr' (line 591)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 0), 'irr', irr)

@norecursion
def npv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'npv'
    module_type_store = module_type_store.open_function_context('npv', 663, 0, False)
    
    # Passed parameters checking function
    npv.stypy_localization = localization
    npv.stypy_type_of_self = None
    npv.stypy_type_store = module_type_store
    npv.stypy_function_name = 'npv'
    npv.stypy_param_names_list = ['rate', 'values']
    npv.stypy_varargs_param_name = None
    npv.stypy_kwargs_param_name = None
    npv.stypy_call_defaults = defaults
    npv.stypy_call_varargs = varargs
    npv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'npv', ['rate', 'values'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'npv', localization, ['rate', 'values'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'npv(...)' code ##################

    str_106090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, (-1)), 'str', '\n    Returns the NPV (Net Present Value) of a cash flow series.\n\n    Parameters\n    ----------\n    rate : scalar\n        The discount rate.\n    values : array_like, shape(M, )\n        The values of the time series of cash flows.  The (fixed) time\n        interval between cash flow "events" must be the same as that for\n        which `rate` is given (i.e., if `rate` is per year, then precisely\n        a year is understood to elapse between each cash flow event).  By\n        convention, investments or "deposits" are negative, income or\n        "withdrawals" are positive; `values` must begin with the initial\n        investment, thus `values[0]` will typically be negative.\n\n    Returns\n    -------\n    out : float\n        The NPV of the input cash flow series `values` at the discount\n        `rate`.\n\n    Notes\n    -----\n    Returns the result of: [G]_\n\n    .. math :: \\sum_{t=0}^{M-1}{\\frac{values_t}{(1+rate)^{t}}}\n\n    References\n    ----------\n    .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,\n       Addison-Wesley, 2003, pg. 346.\n\n    Examples\n    --------\n    >>> np.npv(0.281,[-100, 39, 59, 55, 20])\n    -0.0084785916384548798\n\n    (Compare with the Example given for numpy.lib.financial.irr)\n\n    ')
    
    # Assigning a Call to a Name (line 705):
    
    # Assigning a Call to a Name (line 705):
    
    # Call to asarray(...): (line 705)
    # Processing the call arguments (line 705)
    # Getting the type of 'values' (line 705)
    values_106093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 24), 'values', False)
    # Processing the call keyword arguments (line 705)
    kwargs_106094 = {}
    # Getting the type of 'np' (line 705)
    np_106091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 705)
    asarray_106092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 13), np_106091, 'asarray')
    # Calling asarray(args, kwargs) (line 705)
    asarray_call_result_106095 = invoke(stypy.reporting.localization.Localization(__file__, 705, 13), asarray_106092, *[values_106093], **kwargs_106094)
    
    # Assigning a type to the variable 'values' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'values', asarray_call_result_106095)
    
    # Call to sum(...): (line 706)
    # Processing the call keyword arguments (line 706)
    int_106112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 67), 'int')
    keyword_106113 = int_106112
    kwargs_106114 = {'axis': keyword_106113}
    # Getting the type of 'values' (line 706)
    values_106096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 12), 'values', False)
    int_106097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 22), 'int')
    # Getting the type of 'rate' (line 706)
    rate_106098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 24), 'rate', False)
    # Applying the binary operator '+' (line 706)
    result_add_106099 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 22), '+', int_106097, rate_106098)
    
    
    # Call to arange(...): (line 706)
    # Processing the call arguments (line 706)
    int_106102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 41), 'int')
    
    # Call to len(...): (line 706)
    # Processing the call arguments (line 706)
    # Getting the type of 'values' (line 706)
    values_106104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 48), 'values', False)
    # Processing the call keyword arguments (line 706)
    kwargs_106105 = {}
    # Getting the type of 'len' (line 706)
    len_106103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 44), 'len', False)
    # Calling len(args, kwargs) (line 706)
    len_call_result_106106 = invoke(stypy.reporting.localization.Localization(__file__, 706, 44), len_106103, *[values_106104], **kwargs_106105)
    
    # Processing the call keyword arguments (line 706)
    kwargs_106107 = {}
    # Getting the type of 'np' (line 706)
    np_106100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 31), 'np', False)
    # Obtaining the member 'arange' of a type (line 706)
    arange_106101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 31), np_106100, 'arange')
    # Calling arange(args, kwargs) (line 706)
    arange_call_result_106108 = invoke(stypy.reporting.localization.Localization(__file__, 706, 31), arange_106101, *[int_106102, len_call_result_106106], **kwargs_106107)
    
    # Applying the binary operator '**' (line 706)
    result_pow_106109 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 21), '**', result_add_106099, arange_call_result_106108)
    
    # Applying the binary operator 'div' (line 706)
    result_div_106110 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 12), 'div', values_106096, result_pow_106109)
    
    # Obtaining the member 'sum' of a type (line 706)
    sum_106111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 12), result_div_106110, 'sum')
    # Calling sum(args, kwargs) (line 706)
    sum_call_result_106115 = invoke(stypy.reporting.localization.Localization(__file__, 706, 12), sum_106111, *[], **kwargs_106114)
    
    # Assigning a type to the variable 'stypy_return_type' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'stypy_return_type', sum_call_result_106115)
    
    # ################# End of 'npv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'npv' in the type store
    # Getting the type of 'stypy_return_type' (line 663)
    stypy_return_type_106116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106116)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'npv'
    return stypy_return_type_106116

# Assigning a type to the variable 'npv' (line 663)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 0), 'npv', npv)

@norecursion
def mirr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mirr'
    module_type_store = module_type_store.open_function_context('mirr', 708, 0, False)
    
    # Passed parameters checking function
    mirr.stypy_localization = localization
    mirr.stypy_type_of_self = None
    mirr.stypy_type_store = module_type_store
    mirr.stypy_function_name = 'mirr'
    mirr.stypy_param_names_list = ['values', 'finance_rate', 'reinvest_rate']
    mirr.stypy_varargs_param_name = None
    mirr.stypy_kwargs_param_name = None
    mirr.stypy_call_defaults = defaults
    mirr.stypy_call_varargs = varargs
    mirr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mirr', ['values', 'finance_rate', 'reinvest_rate'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mirr', localization, ['values', 'finance_rate', 'reinvest_rate'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mirr(...)' code ##################

    str_106117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, (-1)), 'str', '\n    Modified internal rate of return.\n\n    Parameters\n    ----------\n    values : array_like\n        Cash flows (must contain at least one positive and one negative\n        value) or nan is returned.  The first value is considered a sunk\n        cost at time zero.\n    finance_rate : scalar\n        Interest rate paid on the cash flows\n    reinvest_rate : scalar\n        Interest rate received on the cash flows upon reinvestment\n\n    Returns\n    -------\n    out : float\n        Modified internal rate of return\n\n    ')
    
    # Assigning a Call to a Name (line 729):
    
    # Assigning a Call to a Name (line 729):
    
    # Call to asarray(...): (line 729)
    # Processing the call arguments (line 729)
    # Getting the type of 'values' (line 729)
    values_106120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 24), 'values', False)
    # Processing the call keyword arguments (line 729)
    # Getting the type of 'np' (line 729)
    np_106121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 38), 'np', False)
    # Obtaining the member 'double' of a type (line 729)
    double_106122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 38), np_106121, 'double')
    keyword_106123 = double_106122
    kwargs_106124 = {'dtype': keyword_106123}
    # Getting the type of 'np' (line 729)
    np_106118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 729)
    asarray_106119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 13), np_106118, 'asarray')
    # Calling asarray(args, kwargs) (line 729)
    asarray_call_result_106125 = invoke(stypy.reporting.localization.Localization(__file__, 729, 13), asarray_106119, *[values_106120], **kwargs_106124)
    
    # Assigning a type to the variable 'values' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'values', asarray_call_result_106125)
    
    # Assigning a Attribute to a Name (line 730):
    
    # Assigning a Attribute to a Name (line 730):
    # Getting the type of 'values' (line 730)
    values_106126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'values')
    # Obtaining the member 'size' of a type (line 730)
    size_106127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 8), values_106126, 'size')
    # Assigning a type to the variable 'n' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'n', size_106127)
    
    # Assigning a Compare to a Name (line 731):
    
    # Assigning a Compare to a Name (line 731):
    
    # Getting the type of 'values' (line 731)
    values_106128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 10), 'values')
    int_106129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 19), 'int')
    # Applying the binary operator '>' (line 731)
    result_gt_106130 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 10), '>', values_106128, int_106129)
    
    # Assigning a type to the variable 'pos' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'pos', result_gt_106130)
    
    # Assigning a Compare to a Name (line 732):
    
    # Assigning a Compare to a Name (line 732):
    
    # Getting the type of 'values' (line 732)
    values_106131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 10), 'values')
    int_106132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 19), 'int')
    # Applying the binary operator '<' (line 732)
    result_lt_106133 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 10), '<', values_106131, int_106132)
    
    # Assigning a type to the variable 'neg' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'neg', result_lt_106133)
    
    
    
    # Evaluating a boolean operation
    
    # Call to any(...): (line 733)
    # Processing the call keyword arguments (line 733)
    kwargs_106136 = {}
    # Getting the type of 'pos' (line 733)
    pos_106134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'pos', False)
    # Obtaining the member 'any' of a type (line 733)
    any_106135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 12), pos_106134, 'any')
    # Calling any(args, kwargs) (line 733)
    any_call_result_106137 = invoke(stypy.reporting.localization.Localization(__file__, 733, 12), any_106135, *[], **kwargs_106136)
    
    
    # Call to any(...): (line 733)
    # Processing the call keyword arguments (line 733)
    kwargs_106140 = {}
    # Getting the type of 'neg' (line 733)
    neg_106138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 26), 'neg', False)
    # Obtaining the member 'any' of a type (line 733)
    any_106139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 26), neg_106138, 'any')
    # Calling any(args, kwargs) (line 733)
    any_call_result_106141 = invoke(stypy.reporting.localization.Localization(__file__, 733, 26), any_106139, *[], **kwargs_106140)
    
    # Applying the binary operator 'and' (line 733)
    result_and_keyword_106142 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 12), 'and', any_call_result_106137, any_call_result_106141)
    
    # Applying the 'not' unary operator (line 733)
    result_not__106143 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 7), 'not', result_and_keyword_106142)
    
    # Testing the type of an if condition (line 733)
    if_condition_106144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 733, 4), result_not__106143)
    # Assigning a type to the variable 'if_condition_106144' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'if_condition_106144', if_condition_106144)
    # SSA begins for if statement (line 733)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 734)
    np_106145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 15), 'np')
    # Obtaining the member 'nan' of a type (line 734)
    nan_106146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 15), np_106145, 'nan')
    # Assigning a type to the variable 'stypy_return_type' (line 734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'stypy_return_type', nan_106146)
    # SSA join for if statement (line 733)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 735):
    
    # Assigning a Call to a Name (line 735):
    
    # Call to abs(...): (line 735)
    # Processing the call arguments (line 735)
    
    # Call to npv(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'reinvest_rate' (line 735)
    reinvest_rate_106150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 23), 'reinvest_rate', False)
    # Getting the type of 'values' (line 735)
    values_106151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 38), 'values', False)
    # Getting the type of 'pos' (line 735)
    pos_106152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 45), 'pos', False)
    # Applying the binary operator '*' (line 735)
    result_mul_106153 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 38), '*', values_106151, pos_106152)
    
    # Processing the call keyword arguments (line 735)
    kwargs_106154 = {}
    # Getting the type of 'npv' (line 735)
    npv_106149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), 'npv', False)
    # Calling npv(args, kwargs) (line 735)
    npv_call_result_106155 = invoke(stypy.reporting.localization.Localization(__file__, 735, 19), npv_106149, *[reinvest_rate_106150, result_mul_106153], **kwargs_106154)
    
    # Processing the call keyword arguments (line 735)
    kwargs_106156 = {}
    # Getting the type of 'np' (line 735)
    np_106147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'np', False)
    # Obtaining the member 'abs' of a type (line 735)
    abs_106148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 12), np_106147, 'abs')
    # Calling abs(args, kwargs) (line 735)
    abs_call_result_106157 = invoke(stypy.reporting.localization.Localization(__file__, 735, 12), abs_106148, *[npv_call_result_106155], **kwargs_106156)
    
    # Assigning a type to the variable 'numer' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'numer', abs_call_result_106157)
    
    # Assigning a Call to a Name (line 736):
    
    # Assigning a Call to a Name (line 736):
    
    # Call to abs(...): (line 736)
    # Processing the call arguments (line 736)
    
    # Call to npv(...): (line 736)
    # Processing the call arguments (line 736)
    # Getting the type of 'finance_rate' (line 736)
    finance_rate_106161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 23), 'finance_rate', False)
    # Getting the type of 'values' (line 736)
    values_106162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 37), 'values', False)
    # Getting the type of 'neg' (line 736)
    neg_106163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 44), 'neg', False)
    # Applying the binary operator '*' (line 736)
    result_mul_106164 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 37), '*', values_106162, neg_106163)
    
    # Processing the call keyword arguments (line 736)
    kwargs_106165 = {}
    # Getting the type of 'npv' (line 736)
    npv_106160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 19), 'npv', False)
    # Calling npv(args, kwargs) (line 736)
    npv_call_result_106166 = invoke(stypy.reporting.localization.Localization(__file__, 736, 19), npv_106160, *[finance_rate_106161, result_mul_106164], **kwargs_106165)
    
    # Processing the call keyword arguments (line 736)
    kwargs_106167 = {}
    # Getting the type of 'np' (line 736)
    np_106158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 12), 'np', False)
    # Obtaining the member 'abs' of a type (line 736)
    abs_106159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 12), np_106158, 'abs')
    # Calling abs(args, kwargs) (line 736)
    abs_call_result_106168 = invoke(stypy.reporting.localization.Localization(__file__, 736, 12), abs_106159, *[npv_call_result_106166], **kwargs_106167)
    
    # Assigning a type to the variable 'denom' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'denom', abs_call_result_106168)
    # Getting the type of 'numer' (line 737)
    numer_106169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'numer')
    # Getting the type of 'denom' (line 737)
    denom_106170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 18), 'denom')
    # Applying the binary operator 'div' (line 737)
    result_div_106171 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 12), 'div', numer_106169, denom_106170)
    
    float_106172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 27), 'float')
    # Getting the type of 'n' (line 737)
    n_106173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 32), 'n')
    int_106174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 36), 'int')
    # Applying the binary operator '-' (line 737)
    result_sub_106175 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 32), '-', n_106173, int_106174)
    
    # Applying the binary operator 'div' (line 737)
    result_div_106176 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 27), 'div', float_106172, result_sub_106175)
    
    # Applying the binary operator '**' (line 737)
    result_pow_106177 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 11), '**', result_div_106171, result_div_106176)
    
    int_106178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 41), 'int')
    # Getting the type of 'reinvest_rate' (line 737)
    reinvest_rate_106179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 45), 'reinvest_rate')
    # Applying the binary operator '+' (line 737)
    result_add_106180 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 41), '+', int_106178, reinvest_rate_106179)
    
    # Applying the binary operator '*' (line 737)
    result_mul_106181 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 11), '*', result_pow_106177, result_add_106180)
    
    int_106182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 62), 'int')
    # Applying the binary operator '-' (line 737)
    result_sub_106183 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 11), '-', result_mul_106181, int_106182)
    
    # Assigning a type to the variable 'stypy_return_type' (line 737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'stypy_return_type', result_sub_106183)
    
    # ################# End of 'mirr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mirr' in the type store
    # Getting the type of 'stypy_return_type' (line 708)
    stypy_return_type_106184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106184)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mirr'
    return stypy_return_type_106184

# Assigning a type to the variable 'mirr' (line 708)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 0), 'mirr', mirr)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
